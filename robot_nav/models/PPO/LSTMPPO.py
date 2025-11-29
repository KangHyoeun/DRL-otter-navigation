# LSTMPPO - Multi-Modal PPO with LSTM (Vector + 2D Grid + Recurrent)
# Based on MLPCNNPPO.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from robot_nav.utils import RunningMeanStd, prepare_multi_modal_state

def init_weights(module, gain=1.0):
    """
    Orthogonal initialization.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)

class RolloutBuffer:
    """
    Buffer to store multi-modal rollout data with Hidden States.
    """
    def __init__(self):
        self.actions = []
        self.vec_states = []      # 1D Vector states
        self.grid_states = []     # 2D Grid states
        self.hidden_states = []   # LSTM Hidden states (h, c)
        self.next_vec_states = []
        self.next_grid_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.vec_states[:]
        del self.grid_states[:]
        del self.hidden_states[:]
        del self.next_vec_states[:]
        del self.next_grid_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state, logprob, state_value, hidden_state):
        # state is tuple (vec, grid)
        self.vec_states.append(state[0])
        self.grid_states.append(state[1])
        self.hidden_states.append(hidden_state) # Store (h, c) tuple
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)
        
        # next_state is tuple (vec, grid)
        self.next_vec_states.append(next_state[0])
        self.next_grid_states.append(next_state[1])
        
        self.logprobs.append(logprob)
        self.state_values.append(state_value)

    def get_generator(self, batch_size, seq_len):
        """
        Yields (Batch, Seq_Len, ...) tensors for BPTT training.
        """
        n_samples = len(self.actions)
        
        # Convert lists to numpy arrays first
        vec_states = np.array(self.vec_states)
        grid_states = np.array(self.grid_states)
        actions = np.array(self.actions)
        logprobs = np.array(self.logprobs)
        rewards = np.array(self.rewards)
        state_values = np.array(self.state_values)
        is_terminals = np.array(self.is_terminals)
        
        # Hidden states: list of (h, c) tuples. h: (1, 1, Hidden)
        # We need to extract them.
        h_states = np.array([h[0].cpu().numpy().squeeze() for h in self.hidden_states]) # (N, Hidden)
        c_states = np.array([h[1].cpu().numpy().squeeze() for h in self.hidden_states]) # (N, Hidden)
        
        # Advantages and Returns (Computed in train)
        if hasattr(self, 'advantages'):
            advantages = self.advantages
            returns = self.returns
        else:
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
        
        # Create sequences
        # We iterate through the data and cut it into sequences of length seq_len.
        # If a terminal state is encountered, the sequence ends there (or we pad).
        # For simplicity in this implementation, we will treat the entire buffer as one continuous stream
        # but reset hidden states at terminals? No, on_policy collects episodes.
        # Better approach: Just slice fixed chunks. If a chunk crosses a terminal, 
        # the LSTM should technically be reset, but with masking we can handle it.
        # However, standard PPO implementation often just gathers trajectories.
        
        # Let's implement a simple sliding window or non-overlapping chunks.
        # Non-overlapping chunks of seq_len.
        
        n_sequences = n_samples // seq_len
        if n_sequences == 0:
            return # Not enough data
            
        # Truncate extra data
        limit = n_sequences * seq_len
        
        vec_states = vec_states[:limit].reshape(n_sequences, seq_len, -1)
        grid_states = grid_states[:limit].reshape(n_sequences, seq_len, 2, 128, 128) # Assuming 2 channels
        actions = actions[:limit].reshape(n_sequences, seq_len, -1)
        logprobs = logprobs[:limit].reshape(n_sequences, seq_len)
        rewards = rewards[:limit].reshape(n_sequences, seq_len)
        state_values = state_values[:limit].reshape(n_sequences, seq_len)
        is_terminals = is_terminals[:limit].reshape(n_sequences, seq_len)
        advantages = advantages[:limit].reshape(n_sequences, seq_len)
        returns = returns[:limit].reshape(n_sequences, seq_len)
        
        # Hidden states: We only need the hidden state at the START of each sequence.
        h_states = h_states[:limit].reshape(n_sequences, seq_len, -1)[:, 0, :] # (n_seq, Hidden)
        c_states = c_states[:limit].reshape(n_sequences, seq_len, -1)[:, 0, :] # (n_seq, Hidden)
        
        # Indices for shuffling
        indices = np.arange(n_sequences)
        np.random.shuffle(indices)
        
        # Yield batches
        n_batches = n_sequences // batch_size
        
        for i in range(n_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            
            yield (
                vec_states[batch_indices],
                grid_states[batch_indices],
                actions[batch_indices],
                logprobs[batch_indices],
                state_values[batch_indices],
                is_terminals[batch_indices],
                h_states[batch_indices],
                c_states[batch_indices],
                advantages[batch_indices],
                returns[batch_indices]
            )


class LSTMPPOActorCritic(nn.Module):
    """
    Multi-Modal Actor-Critic Network with LSTM.
    
    Architecture:
        - Split MLPs & CNN (Same as MLPCNNPPO)
        - Fusion
        - LSTM Layer
        - Heads (Actor/Critic)
    """

    def __init__(self, vec_dim, action_dim, log_std_init, max_action, device, hidden_size=512):
        super(LSTMPPOActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # ========== CNN Branch (for 128x128 Grid) ==========
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4), # 128->31 (Input channels: 2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 31->14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 14->12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512), # Reduce CNN output to 512
            nn.ReLU()
        )
        
        # ========== Split MLP Branches ==========
        self.vel_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.goal_mlp = nn.Sequential(nn.Linear(3, 32), nn.Tanh())
        self.error_mlp = nn.Sequential(nn.Linear(4, 32), nn.Tanh())
        self.rps_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.cr_mlp = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        
        # ========== Fusion & Compression ==========
        # Combined dim: 512 (CNN reduced) + 104 (MLP) = 616
        self.fusion_dim = 512 + 104
        
        # FC Layer for compression (Fusion -> Compressed Feature)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU()
        )
        
        # ========== LSTM Layer ==========
        # Input dim is now 512 (compressed feature)
        self.lstm = nn.LSTM(512, hidden_size)
        
        # ========== Heads ==========
        # Actor Head
        self.actor = nn.Linear(hidden_size, action_dim)
        
        # Critic Head
        self.critic = nn.Linear(hidden_size, 1)
        
        # ========== Initialization ==========
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.actor, gain=0.5)
        init_weights(self.critic, gain=1.0)
        init_weights(self.lstm, gain=1.0)

    def forward(self):
        raise NotImplementedError

    def _extract_features(self, vec, grid):
        """
        Extract and fuse features from both branches.
        """
        # Process Grid
        if grid.dim() == 3: grid = grid.unsqueeze(0)
            
        # Process Vector
        if vec.dim() == 1: vec = vec.unsqueeze(0)
            
        # Split Vector Features
        vel_in = vec[:, 0:2]
        goal_in = torch.cat([vec[:, 2:4], vec[:, 6:7]], dim=1)
        error_in = torch.cat([vec[:, 4:6], vec[:, 8:10]], dim=1)
        rps_in = vec[:, 10:12]
        cr_in = vec[:, 7:8]
        
        # Pass through MLPs
        vel_out = self.vel_mlp(vel_in)
        goal_out = self.goal_mlp(goal_in)
        error_out = self.error_mlp(error_in)
        rps_out = self.rps_mlp(rps_in)
        cr_out = self.cr_mlp(cr_in)
        
        # CNN Branch
        cnn_out = self.cnn(grid)
        
        # Fusion
        combined = torch.cat([cnn_out, vel_out, goal_out, error_out, rps_out, cr_out], dim=1)
        
        # Compression (FC Layer)
        compressed = self.fusion_layer(combined)
        
        return compressed

    def act(self, vec, grid, hidden, sample=True):
        """
        Get action with LSTM state.
        hidden: (h, c) tuple
        """
        features = self._extract_features(vec, grid)
        
        # LSTM Forward
        # LSTM expects (Seq, Batch, Feature)
        # Here we process single step: (1, 1, Feature)
        features = features.unsqueeze(0) 
        
        lstm_out, next_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out.squeeze(0) # (1, Hidden)
        
        action_mean = self.actor(lstm_out)
        action_std = torch.exp(self.log_std)
        
        if action_mean.dim() > 1:
            action_std = action_std.expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if sample:
            action = dist.sample()
        else:
            action = action_mean
        
        action_clipped = torch.clamp(action, -self.max_action, self.max_action)

        if action.dim() > 1:
            action_logprob = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprob = dist.log_prob(action).sum()
            
        state_val = self.critic(lstm_out)

        return action_clipped, action_logprob, state_val, next_hidden

    def evaluate(self, vec, grid, action, hidden):
        """
        Evaluate actions with LSTM state (Sequence Processing).
        vec: (Batch, Seq_Len, Vec_Dim)
        grid: (Batch, Seq_Len, C, H, W)
        action: (Batch, Seq_Len, Action_Dim)
        hidden: (h, c) where h/c is (1, Batch, Hidden) - Initial hidden state for the sequence
        """
        batch_size, seq_len, _ = vec.shape
        
        # Flatten for CNN/MLP processing
        vec_flat = vec.reshape(-1, vec.shape[-1]) # (Batch*Seq, Vec_Dim)
        grid_flat = grid.reshape(-1, grid.shape[-3], grid.shape[-2], grid.shape[-1]) # (Batch*Seq, C, H, W)
        action_flat = action.reshape(-1, action.shape[-1]) # (Batch*Seq, Action_Dim)
        
        # Extract features
        features = self._extract_features(vec_flat, grid_flat) # (Batch*Seq, Feature_Dim)
        
        # Reshape back to sequence for LSTM
        features = features.reshape(batch_size, seq_len, -1) # (Batch, Seq, Feature_Dim)
        
        # LSTM Forward
        # LSTM expects (Seq, Batch, Feature) if batch_first=False (default)
        features = features.permute(1, 0, 2) # (Seq, Batch, Feature)
        
        lstm_out, _ = self.lstm(features, hidden) # hidden is initial state
        
        # Reshape output back to (Batch*Seq, Hidden) for Heads
        lstm_out = lstm_out.permute(1, 0, 2) # (Batch, Seq, Hidden)
        lstm_out = lstm_out.reshape(-1, self.hidden_size) # (Batch*Seq, Hidden)
        
        # Heads
        action_mean = self.actor(lstm_out)
        action_std = torch.exp(self.log_std)
        
        if action_mean.dim() > 1:
            action_std = action_std.expand_as(action_mean)
            
        dist = Normal(action_mean, action_std)

        if self.action_dim == 1:
            action_flat = action_flat.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action_flat).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(lstm_out)
        
        # Reshape back to (Batch, Seq) if needed, but PPO usually works on flattened batch
        # But we return flattened to match the targets which will also be flattened
        
        return action_logprobs, state_values, dist_entropy

    def get_value(self, vec, grid, hidden):
        features = self._extract_features(vec, grid)
        features = features.unsqueeze(0)
        lstm_out, _ = self.lstm(features, hidden)
        lstm_out = lstm_out.squeeze(0)
        return self.critic(lstm_out)


class LSTMPPO:
    """
    Multi-Modal PPO Agent with LSTM.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        log_std_init=0.0,
        ent_coef_init=0.01,
        ent_coef_decay_rate=0.0,
        min_ent_coef=0.001,
        target_kl=5.0,
        device="cpu",
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name="LSTMPPO",
        load_directory=Path("robot_nav/models/PPO/checkpoint"),
        hidden_size=512,
        seq_len=8,
        lr_decay_epochs=1000,
        lr_min_factor=0.1
    ):
        self.max_action = max_action
        self.vec_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.ent_coef = ent_coef_init
        self.ent_coef_decay_rate = ent_coef_decay_rate
        self.min_ent_coef = min_ent_coef
        self.target_kl = target_kl
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0 

        self.buffer = RolloutBuffer()

        self.policy = LSTMPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device, hidden_size
        ).to(device)
        
        critic_head_params = list(self.policy.critic.parameters())
        critic_head_ids = list(map(id, critic_head_params))
        base_params = [p for p in self.policy.parameters() if id(p) not in critic_head_ids]
        
        self.optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': lr_actor},
            {'params': critic_head_params, 'lr': lr_critic}
        ], eps=1e-5)
        
        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=lr_decay_epochs, eta_min=lr_actor * lr_min_factor
        )

        self.policy_old = LSTMPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device, hidden_size
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.SmoothL1Loss()
        
        # Ensure runs directory exists for TensorBoard
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(comment=model_name)
        
        if load_model:
            self.load(model_name, load_directory)
            
        print(f"✅ LSTMPPO Initialized (Hidden Size: {hidden_size})")

    def get_action(self, state, hidden, add_noise=True, update_rms=True):
        """
        state: (vec, grid)
        hidden: (h, c)
        """
        vec_raw, grid_raw = state
        
        with torch.no_grad():
            vec_np = np.array(vec_raw, dtype=np.float32)
            if update_rms:
                self.obs_rms.update(vec_np.reshape(1, -1))
            
            vec_norm = (vec_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            vec_tensor = torch.FloatTensor(vec_norm).to(self.device)
            
            if vec_tensor.dim() == 1:
                vec_tensor = vec_tensor.unsqueeze(0)
            
            grid_tensor = torch.FloatTensor(grid_raw).to(self.device)
            if update_rms and add_noise:
                grid_tensor += torch.randn_like(grid_tensor) * 0.01
                grid_tensor = torch.clamp(grid_tensor, 0, 1)
            
            if grid_tensor.dim() == 2:
                grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0) 
            elif grid_tensor.dim() == 3:
                grid_tensor = grid_tensor.unsqueeze(0)

            action, log_prob, state_val, next_hidden = self.policy_old.act(vec_tensor, grid_tensor, hidden, sample=add_noise)
            
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), state_val.cpu().numpy().flatten(), next_hidden

    def train(self, _, training_iterations, batch_size):
        # 1. Calculate GAE/Returns for entire buffer
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(np.array(self.buffer.is_terminals), dtype=torch.bool).to(self.device)
        values = torch.tensor(np.array(self.buffer.state_values), dtype=torch.float32).to(self.device)
        
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        advantages = returns - values
        
        # Store advantages/returns in buffer temporarily to be yielded
        self.buffer.advantages = advantages.cpu().numpy()
        self.buffer.returns = returns.cpu().numpy()
        
        # Optimize policy for K epochs
        for _ in range(training_iterations):
            # Use generator to get sequence batches
            n_sequences_per_batch = max(1, batch_size // self.seq_len)
            
            data_generator = self.buffer.get_generator(n_sequences_per_batch, self.seq_len)
            
            for batch in data_generator:
                (vec_states, grid_states, actions, logprobs, state_values, is_terminals, h_states, c_states, adv_batch, ret_batch) = batch
                
                # Convert to tensors
                vec_states = torch.tensor(vec_states, dtype=torch.float32).to(self.device)
                grid_states = torch.tensor(grid_states, dtype=torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
                logprobs = torch.tensor(logprobs, dtype=torch.float32).to(self.device)
                # state_values = torch.tensor(state_values, dtype=torch.float32).to(self.device) # Not used directly if we use ret_batch
                
                adv_batch = torch.tensor(adv_batch, dtype=torch.float32).to(self.device)
                ret_batch = torch.tensor(ret_batch, dtype=torch.float32).to(self.device)
                
                # Hidden states (Initial for sequence)
                h_states = torch.tensor(h_states, dtype=torch.float32).to(self.device).unsqueeze(0) # (1, Batch, Hidden)
                c_states = torch.tensor(c_states, dtype=torch.float32).to(self.device).unsqueeze(0) # (1, Batch, Hidden)
                hidden_batch = (h_states, c_states)
                
                # Normalize Vec
                vec_states = (vec_states - torch.tensor(self.obs_rms.mean, device=self.device)) / torch.sqrt(torch.tensor(self.obs_rms.var, device=self.device) + 1e-8)
                
                # Evaluate old actions and values
                # Pass hidden_batch!
                # evaluate returns flattened outputs
                logprobs_new, state_values_new, dist_entropy = self.policy.evaluate(vec_states, grid_states, actions, hidden_batch)
                
                # Flatten targets to match evaluate output
                logprobs = logprobs.view(-1)
                adv_batch = adv_batch.view(-1)
                ret_batch = ret_batch.view(-1)
                
                # Ratios
                ratios = torch.exp(logprobs_new - logprobs)
                
                # Surrogate Loss
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batch
                
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values_new, ret_batch) - self.ent_coef * dist_entropy
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                # Logging (Log last batch)
                self.writer.add_scalar("train/loss", loss.mean().item(), self.iter_count)
                self.writer.add_scalar("train/entropy", dist_entropy.mean().item(), self.iter_count)
                self.writer.add_scalar("train/approx_kl", (logprobs - logprobs_new).mean().item(), self.iter_count)
                self.writer.add_scalar("train/ent_coef", self.ent_coef, self.iter_count)
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        # Decay entropy coef
        self.ent_coef = max(self.min_ent_coef, self.ent_coef * (1.0 - self.ent_coef_decay_rate))
        
        # Step LR Scheduler
        self.scheduler.step()
        
        self.iter_count += 1
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", current_lr, self.iter_count)
    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{directory}/{filename}_policy.pth")
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count
        }, f"{directory}/{filename}_rms.pth")

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load(f"{directory}/{filename}_policy.pth", weights_only=False))
        self.policy_old.load_state_dict(self.policy.state_dict())
        try:
            rms_data = torch.load(f"{directory}/{filename}_rms.pth", weights_only=False)
            self.obs_rms.mean = rms_data['mean']
            self.obs_rms.var = rms_data['var']
            self.obs_rms.count = rms_data['count']
            print(f"Loaded weights and RMS from {directory}")
        except FileNotFoundError:
            print(f"Loaded weights from {directory} (RMS not found)")

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
        """
        Uses common state preparation from utils.
        """
        return prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map
        )
