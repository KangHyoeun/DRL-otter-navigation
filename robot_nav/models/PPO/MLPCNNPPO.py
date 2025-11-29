# MLPCNNPPO - Multi-Modal PPO (Vector + 2D Grid)
# Optimized for performance and stability (SB3-style improvements).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from colregs_core.geometry import math_to_maritime_velocity
from robot_nav.utils import RunningMeanStd, prepare_multi_modal_state

def init_weights(module, gain=1.0):
    """
    Orthogonal initialization.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class RolloutBuffer:
    """
    Buffer to store multi-modal rollout data.
    """
    def __init__(self):
        self.actions = []
        self.vec_states = []      # 1D Vector states
        self.grid_states = []     # 2D Grid states
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
        del self.next_vec_states[:]
        del self.next_grid_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state, logprob, state_value):
        # state is tuple (vec, grid)
        self.vec_states.append(state[0])
        self.grid_states.append(state[1])
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)
        
        # next_state is tuple (vec, grid)
        self.next_vec_states.append(next_state[0])
        self.next_grid_states.append(next_state[1])
        
        self.logprobs.append(logprob)
        self.state_values.append(state_value)


class MLPCNNPPOActorCritic(nn.Module):
    """
    Multi-Modal Actor-Critic Network with Semantic Vector Splitting.
    
    Inputs:
        - Vector: (Batch, 12) -> Split into [vel(2), goal(3), error(4), rps(2), cr(1)]
        - Grid:   (Batch, 2, 128, 128)
    
    Architecture:
        - Split MLPs: Process each semantic group separately
        - CNN Branch: Nature-CNN style (128x128 input)
        - Fusion: Concat all features -> Shared FC -> Heads
    """

    def __init__(self, vec_dim, action_dim, log_std_init, max_action, device):
        super(MLPCNNPPOActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # ========== CNN Branch (for 128x128 Grid) ==========
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4), # 128->31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 31->14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 14->12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512), # Reduce CNN output to 512
            nn.ReLU()
        )
        # Output dim: 512 (after reduction)
        
        # ========== Split MLP Branches ==========
        # State Vector: [v_x, v_y, distance, y_e, psi_e, chi_e, phi_tilde, CR_max, u_e, r_e, n1, n2]
        # Groups:
        # 1. Vel (2): v_x, v_y
        # 2. Goal (3): distance, y_e, phi_tilde (indices: 2, 3, 6)
        # 3. Error (4): psi_e, chi_e, u_e, r_e (indices: 4, 5, 8, 9)
        # 4. RPS (2): n1, n2 (indices: 10, 11)
        # 5. CR (1): CR_max (index: 7)
        
        self.vel_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.goal_mlp = nn.Sequential(nn.Linear(3, 32), nn.Tanh())
        self.error_mlp = nn.Sequential(nn.Linear(4, 32), nn.Tanh())
        self.rps_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.cr_mlp = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        
        # Total MLP Output Dim: 16 + 32 + 32 + 16 + 8 = 104
        
        # ========== Fusion & Heads ==========
        # Combined dim: 512 (CNN reduced) + 104 (MLP) = 616
        self.fusion_dim = 512 + 104
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.Tanh()
        )
        
        # Actor Head
        self.actor = nn.Linear(512, action_dim)
        
        # Critic Head
        self.critic = nn.Linear(512, 1)
        
        # ========== Initialization ==========
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.actor, gain=0.5) # Increased gain from 0.01 to 0.5 to encourage initial exploration
        init_weights(self.critic, gain=1.0)

    def forward(self):
        raise NotImplementedError

    def _extract_features(self, vec, grid):
        """
        Extract and fuse features from both branches.
        Vec: [v_x, v_y, distance, y_e, psi_e, chi_e, phi_tilde, CR_max, u_e, r_e, n1, n2]
        Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        """
        # Process Grid
        if grid.dim() == 3: grid = grid.unsqueeze(0)
            
        # Process Vector
        if vec.dim() == 1: vec = vec.unsqueeze(0)
            
        # Split Vector Features
        # Vel: [0, 1]
        vel_in = vec[:, 0:2]
        # Goal: [2, 3, 6] (dist, y_e, phi_tilde)
        goal_in = torch.cat([vec[:, 2:4], vec[:, 6:7]], dim=1)
        # Error: [4, 5, 8, 9] (psi_e, chi_e, u_e, r_e)
        error_in = torch.cat([vec[:, 4:6], vec[:, 8:10]], dim=1)
        # RPS: [10, 11]
        rps_in = vec[:, 10:12]
        # CR: [7]
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
        return self.shared_fc(combined)

    def act(self, vec, grid, sample=True):
        features = self._extract_features(vec, grid)
        
        action_mean = self.actor(features)
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
            
        state_val = self.critic(features)

        return action_clipped, action_logprob, state_val

    def evaluate(self, vec, grid, action):
        features = self._extract_features(vec, grid)
        
        action_mean = self.actor(features)
        action_std = torch.exp(self.log_std)
        
        if action_mean.dim() > 1:
            action_std = action_std.expand_as(action_mean)
            
        dist = Normal(action_mean, action_std)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(features)

        return action_logprobs, state_values, dist_entropy

    def get_value(self, vec, grid):
        features = self._extract_features(vec, grid)
        return self.critic(features)


class MLPCNNPPO:
    """
    Multi-Modal PPO Agent (Vector + CNN).
    """

    def __init__(
        self,
        state_dim, # 12 (Vector dim)
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
        model_name="MLPCNNPPO",
        load_directory=Path("robot_nav/models/PPO/checkpoint"),
        lr_decay_epochs=1000, # Default decay epochs
        lr_min_factor=0.1,    # Default min factor
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
        
        # Running Mean Std (Only for Vector state and Reward)
        # Grid is usually 0-1 or 0-255, handled by simple scaling
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0 

        self.buffer = RolloutBuffer()

        self.policy = MLPCNNPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device
        ).to(device)
        
        # Separate parameters for Actor (including shared) and Critic head
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

        self.policy_old = MLPCNNPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.SmoothL1Loss()
        # Ensure runs directory exists for TensorBoard
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(comment=model_name)
        self.state_log_counter = 0
        
        print(f"✅ MLPCNNPPO (Vector+Grid) Initialized")
        print(f"   - Ent Coef: {self.ent_coef} (Decay: {self.ent_coef_decay_rate}, Min: {self.min_ent_coef})")

    def decay_ent_coef(self):
        """
        Decay entropy coefficient linearly.
        """
        self.ent_coef = self.ent_coef - self.ent_coef_decay_rate
        if self.ent_coef <= self.min_ent_coef:
            self.ent_coef = self.min_ent_coef
        
        # Log to tensorboard
        # self.writer.add_scalar("train/ent_coef", self.ent_coef, self.iter_count)

    def get_action(self, state, add_noise=True, update_rms=True):
        """
        Expects state to be a tuple: (vector_state, grid_map)
        """
        vec_raw, grid_raw = state
        
        with torch.no_grad():
            # 1. Normalize Vector State
            vec_np = np.array(vec_raw, dtype=np.float32)
            if update_rms:
                self.obs_rms.update(vec_np.reshape(1, -1))
            vec_norm = (vec_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            vec_tensor = torch.FloatTensor(vec_norm).to(self.device)
            
            # 2. Process Grid State (CR values are 0-1)
            grid_tensor = torch.FloatTensor(grid_raw).to(self.device)

            if update_rms and add_noise:
                # 탐색 시 약간의 noise 추가로 sparse signal 문제 완화
                grid_tensor += torch.randn_like(grid_tensor) * 0.01
                grid_tensor = torch.clamp(grid_tensor, 0, 1)
            
            # Ensure dimension is (1, 128, 128) for single sample input
            if grid_tensor.dim() == 2:
                grid_tensor = grid_tensor.unsqueeze(0) # Add channel
            
            action, action_logprob, state_val = self.policy_old.act(
                vec_tensor, grid_tensor, sample=add_noise
            )

        return action.detach().cpu().numpy().flatten(), action_logprob.detach().cpu(), state_val.detach().cpu()
    
    def compute_gae(self, rewards, values, next_values, dones):
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        next_values = next_values.to(self.device)
        dones = dones.to(self.device)
        
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        buffer_size = len(rewards)
        advantages = torch.zeros(buffer_size, device=self.device)
        
        gae = 0.0
        for t in range(buffer_size - 1, -1, -1):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns

    def train(self, replay_buffer, iterations, batch_size):
        # 1. Prepare Batch Data
        # Vector States
        vec_states_np = np.array(self.buffer.vec_states, dtype=np.float32)
        vec_states_norm = (vec_states_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        vec_states = torch.tensor(vec_states_norm, dtype=torch.float32).to(self.device)
        
        next_vec_np = np.array(self.buffer.next_vec_states, dtype=np.float32)
        next_vec_norm = (next_vec_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        next_vec_states = torch.tensor(next_vec_norm, dtype=torch.float32).to(self.device)
        
        # Grid States
        grid_states_np = np.array(self.buffer.grid_states, dtype=np.float32)
        grid_states = torch.tensor(grid_states_np, dtype=torch.float32).to(self.device)
        # if grid_states.max() > 1.0: grid_states /= 255.0
        
        # Ensure (N, 1, H, W)
        if grid_states.dim() == 3: 
            grid_states = grid_states.unsqueeze(1) 
            
        next_grid_np = np.array(self.buffer.next_grid_states, dtype=np.float32)
        next_grid_states = torch.tensor(next_grid_np, dtype=torch.float32).to(self.device)
        # if next_grid_states.max() > 1.0: next_grid_states /= 255.0
        
        # Ensure (N, 1, H, W)
        if next_grid_states.dim() == 3: 
            next_grid_states = next_grid_states.unsqueeze(1)

        # Other data
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()
        old_state_values = torch.stack(self.buffer.state_values).squeeze().to(self.device).detach()
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards, dtype=np.float32), dtype=torch.float32).to(self.device)
        
        # 2. Compute Advantages (GAE)
        with torch.no_grad():
            next_state_values = self.policy.get_value(next_vec_states, next_grid_states).squeeze()

        advantages, returns = self.compute_gae(
            rewards, old_state_values, next_state_values, dones
        )
        
        explained_var = 1 - torch.var(returns - old_state_values) / (torch.var(returns) + 1e-8)
        
        dataset_size = vec_states.shape[0]
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        num_updates = 0
        
        # 3. Mini-batch Training
        for epoch in range(iterations):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Slice batches
                b_vec = vec_states[batch_indices]
                b_grid = grid_states[batch_indices]
                b_actions = old_actions[batch_indices]
                b_old_logprobs = old_logprobs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                b_old_values = old_state_values[batch_indices]

                # Mini-batch Advantage Normalization
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Evaluate
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    b_vec, b_grid, b_actions
                )
                
                state_values = torch.squeeze(state_values)
                
                # Loss calculation
                log_ratio = logprobs - b_old_logprobs
                ratios = torch.exp(log_ratio)
                
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate clip fraction
                clip_frac = (torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item()
                
                values_pred = state_values
                values_clipped = b_old_values + torch.clamp(
                    values_pred - b_old_values, -self.eps_clip, self.eps_clip
                )
                v_loss1 = self.MseLoss(values_pred, b_returns)
                v_loss2 = self.MseLoss(values_clipped, b_returns)
                critic_loss = 0.5 * torch.max(v_loss1, v_loss2)
                
                # Use dynamic entropy coefficient
                entropy_loss = -self.ent_coef * dist_entropy.mean()
                
                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping 강화로 안정성 향상
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_actor_loss += actor_loss.item() # Accumulate actor loss
                total_critic_loss += critic_loss.item() # Accumulate critic loss
                total_entropy += dist_entropy.mean().item() # Accumulate entropy
                total_clip_frac += clip_frac
                num_updates += 1

                # KL Check (Moved after update)
                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    total_approx_kl += approx_kl

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    break
                
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                print(f"⚠️  Early stopping at epoch {epoch+1} due to KL: {approx_kl:.4f}")
                break
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
        # Decay entropy coefficient
        self.decay_ent_coef()
        
        # Step LR Scheduler
        self.scheduler.step()
        
        self.iter_count += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
        avg_approx_kl = total_approx_kl / num_updates if num_updates > 0 else 0
        avg_clip_frac = total_clip_frac / num_updates if num_updates > 0 else 0
        
        current_std = torch.exp(self.policy.log_std).mean().item() 
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.writer.add_scalar("train/total_loss", avg_loss, self.iter_count)
        self.writer.add_scalar("train/actor_loss", avg_actor_loss, self.iter_count)
        self.writer.add_scalar("train/critic_loss", avg_critic_loss, self.iter_count)
        self.writer.add_scalar("train/entropy", avg_entropy, self.iter_count)
        self.writer.add_scalar("train/explained_variance", explained_var.item(), self.iter_count)
        self.writer.add_scalar("train/action_std", current_std, self.iter_count)
        self.writer.add_scalar("train/approx_kl", avg_approx_kl, self.iter_count)
        self.writer.add_scalar("train/clip_fraction", avg_clip_frac, self.iter_count)
        self.writer.add_scalar("train/learning_rate", current_lr, self.iter_count)
        
        if self.iter_count % 10 == 0:
            print(f"Iter {self.iter_count} | Loss: {avg_loss:.4f} | Actor: {avg_actor_loss:.4f} | Critic: {avg_critic_loss:.4f} | KL: {avg_approx_kl:.4f} | Clip: {avg_clip_frac:.2f} | LR: {current_lr:.6f}")

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
        """
        Uses common state preparation from utils.
        """
        return prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map
        )


    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.policy_old.state_dict(),
                'obs_rms_mean': self.obs_rms.mean,
                'obs_rms_var': self.obs_rms.var,
                'obs_rms_count': self.obs_rms.count,
                'ent_coef': self.ent_coef # Save ent_coef
            }, 
            "%s/%s_policy.pth" % (directory, filename)
        )

    def load(self, filename, directory):
        checkpoint = torch.load(
            "%s/%s_policy.pth" % (directory, filename),
            map_location=lambda storage, loc: storage,
        )
        
        if 'model_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.obs_rms.mean = checkpoint['obs_rms_mean']
            self.obs_rms.var = checkpoint['obs_rms_var']
            self.obs_rms.count = checkpoint['obs_rms_count']
            if 'ent_coef' in checkpoint:
                self.ent_coef = checkpoint['ent_coef']
                print(f"Loaded ent_coef: {self.ent_coef}")
            print(f"Loaded weights and RMS stats from: {directory}")
        else:
            self.policy_old.load_state_dict(checkpoint)
            self.policy.load_state_dict(checkpoint)
            print(f"Loaded legacy weights from: {directory}")
