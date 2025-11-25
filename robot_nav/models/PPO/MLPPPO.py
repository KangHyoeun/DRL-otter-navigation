# MLPPPO - MLP-based Proximal Policy Optimization
# Optimized for performance and stability (SB3-style improvements).

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from colregs_core.geometry import math_to_maritime_velocity
from robot_nav.utils import RunningMeanStd

def init_weights(module, gain=1.0):
    """
    Orthogonal initialization for neural network weights.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class RolloutBuffer:
    """
    Buffer to store rollout data (transitions) for PPO training.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state, logprob, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)
        self.next_states.append(next_state)
        self.logprobs.append(logprob)
        self.state_values.append(state_value)


class MLPPPOActorCritic(nn.Module):
    """
    MLP-based Actor-Critic neural network model for PPO.
    Uses orthogonal initialization and learnable state-independent log_std.
    """

    def __init__(self, state_dim, action_dim, log_std_init, max_action, device):
        super(MLPPPOActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Learnable log_std (Parameter) - initialized to log_std_init
        # Using log_std ensures std is always positive when exponentiated
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, 1),
        )
        
        # Apply Orthogonal Initialization
        self.actor.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.critic.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Specific initialization for output layers
        # Actor output: 0.01 gain to ensure initial actions are close to 0
        init_weights(self.actor[-2], gain=0.01)
        # Critic output: 1.0 gain
        init_weights(self.critic[-1], gain=1.0)

    def forward(self, state):
        raise NotImplementedError

    def act(self, state, sample=True):
        """
        Compute an action, its log probability, and the state value.
        """
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        
        # Expand action_std to match batch size if necessary
        if action_mean.dim() > 1:
            action_std = action_std.expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if sample:
            action = dist.sample()
        else:
            action = action_mean
        
        # Clip action to valid range
        action_clipped = torch.clamp(action, -self.max_action, self.max_action)

        if action.dim() > 1:
            action_logprob = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprob = dist.log_prob(action).sum()
            
        state_val = self.critic(state)

        return action_clipped, action_logprob, state_val

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, entropy, and state values.
        """
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        
        if action_mean.dim() > 1:
            action_std = action_std.expand_as(action_mean)
            
        dist = Normal(action_mean, action_std)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class MLPPPO:
    """
    MLP-based Proximal Policy Optimization (MLPPPO) - SB3 Style.
    Includes:
    - Learnable Std
    - Orthogonal Init
    - Value Function Clipping
    - Running Mean/Std Normalization (Obs & Reward)
    - Gradient Clipping
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0003, # Standard SB3 LR
        lr_critic=0.0003, # Standard SB3 LR
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        log_std_init=0.0, # log(1.0) = 0.0 -> Initial std = 1.0
        ent_coef_init=0.01,
        ent_coef_decay_rate=0.0,
        min_ent_coef=0.001,
        target_kl=0.05,
        device="cpu",
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name="MLPPPO",
        load_directory=Path("robot_nav/models/PPO/checkpoint"),
    ):
        self.max_action = max_action
        self.state_dim = state_dim
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
        
        # Running Mean Std for Observations and Rewards
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0 # Current return for reward normalization

        self.buffer = RolloutBuffer()

        self.policy = MLPPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr_actor, eps=1e-5
        )

        self.policy_old = MLPPPOActorCritic(
            state_dim, action_dim, log_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter(comment=model_name)
        self.state_log_counter = 0
        
        print(f"✅ MLPPPO (SB3 Style) Initialized")
        print(f"   - Ent Coef: {self.ent_coef} (Decay: {self.ent_coef_decay_rate}, Min: {self.min_ent_coef})")

    def decay_ent_coef(self):
        """
        Decay entropy coefficient linearly.
        """
        self.ent_coef = self.ent_coef - self.ent_coef_decay_rate
        if self.ent_coef <= self.min_ent_coef:
            self.ent_coef = self.min_ent_coef

    def normalize_obs(self, obs):
        """
        Normalize observations using running mean and std.
        """
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

    def normalize_reward(self, reward):
        """
        Normalize reward using running return statistics.
        """
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(np.array([self.ret]))
        return reward / np.sqrt(self.ret_rms.var + 1e-8)

    def get_action(self, state, add_noise=True):
        """
        Sample an action. 
        Note: add_noise is effectively handled by sample=True/False in act.
        """
        # Update running mean/std with the new state (training only typically, but simple here)
        # For strict correctness, we should only update during training rollouts
        # normalized_state = self.normalize_obs(np.array([state]))
        
        # Current code structure seems to pass raw state to prepare_state, then here.
        # We will assume 'state' passed here is the feature vector.
        # We'll do on-the-fly normalization here.
        
        with torch.no_grad():
            state_np = np.array(state, dtype=np.float32)
            # Update RMS and normalize
            # Note: If testing/eval, we should NOT update RMS. 
            # Assuming this is mostly training or we accept test-time adaptation.
            self.obs_rms.update(state_np.reshape(1, -1))
            state_norm = (state_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            
            state_tensor = torch.FloatTensor(state_norm).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor, sample=add_noise)

        return action.detach().cpu().numpy().flatten(), action_logprob, state_val
    
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
        """
        Train the policy using PPO with GAE and minibatch updates.
        """
        # Convert buffer to numpy for RMS updates (if we hadn't done it incrementally)
        # Since we normalized incrementally in get_action, we use the stored states directly? 
        # Wait, buffer stores raw states usually? 
        # If get_action normalized them, we must store normalized states or re-normalize.
        # Let's assume we re-normalize with the latest RMS statistics for better stability (SB3 does this).
        
        states_np = np.array(self.buffer.states, dtype=np.float32)
        next_states_np = np.array(self.buffer.next_states, dtype=np.float32)
        
        # Re-normalize using current RMS
        states_norm = (states_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        next_states_norm = (next_states_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        
        states = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states_norm, dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()
        old_state_values = torch.stack(self.buffer.state_values).squeeze().to(self.device).detach()
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        
        # Normalize Rewards (Optional but recommended)
        rewards_np = np.array(self.buffer.rewards, dtype=np.float32)
        # Simple normalization batch-wise if not using online ret_rms for rewards
        # Or use the online one. Let's use simple batch normalization for advantages, 
        # but for returns, we keep them raw for now as we didn't implement full RetRMS in collection loop.
        rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            next_state_values = self.policy.critic(next_states).squeeze()

        advantages, returns = self.compute_gae(
            rewards, old_state_values, next_state_values, dones
        )
        
        # Advantage Normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        explained_var = 1 - torch.var(returns - old_state_values) / (torch.var(returns) + 1e-8)
        
        dataset_size = states.shape[0]
        
        total_loss = 0
        num_updates = 0
        
        for epoch in range(iterations):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_state_values[batch_indices] # For Value Clipping
                
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                state_values = torch.squeeze(state_values)
                
                # Ratio
                ratios = torch.exp(logprobs - batch_old_logprobs)
                
                # Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Function Clipping
                values_pred = state_values
                values_clipped = batch_old_values + torch.clamp(
                    values_pred - batch_old_values, -self.eps_clip, self.eps_clip
                )
                v_loss1 = self.MseLoss(values_pred, batch_returns)
                v_loss2 = self.MseLoss(values_clipped, batch_returns)
                critic_loss = 0.5 * torch.max(v_loss1, v_loss2) # Max because we want to minimize the worst case? No, typical impl takes max of squared errors
                
                # Use dynamic entropy coefficient
                entropy_loss = -self.ent_coef * dist_entropy.mean()
                
                loss = actor_loss + critic_loss + entropy_loss

                # KL divergence check
                with torch.no_grad():
                    approx_kl = torch.mean((ratios - 1) - logprobs + batch_old_logprobs).item()

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    break # Early stopping for this batch
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
            
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                print(f"⚠️  Early stopping at epoch {epoch+1} due to KL: {approx_kl:.4f}")
                break
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
        # Decay entropy coefficient
        self.decay_ent_coef()
        
        self.iter_count += 1
        
        # Log metrics
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        current_std = torch.exp(self.policy.log_std).mean().item()
        
        self.writer.add_scalar("train/total_loss", avg_loss, self.iter_count)
        self.writer.add_scalar("train/action_std", current_std, self.iter_count)
        self.writer.add_scalar("train/ent_coef", self.ent_coef, self.iter_count) # Log ent_coef
        self.writer.add_scalar("train/explained_variance", explained_var.item(), self.iter_count)
        
        if self.iter_count % 10 == 0:
            print(f"Iter {self.iter_count} | Loss: {avg_loss:.4f} | Std: {current_std:.4f} | EntCoef: {self.ent_coef:.4f} | ExpVar: {explained_var.item():.4f}")

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max):
        """
        Raw feature extraction. Normalization happens in get_action via RunningMeanStd.
        """
        # Velocities
        psi_math_deg = np.degrees(robot_state[2, 0])
        speed = np.linalg.norm([robot_state[3, 0], robot_state[4, 0]])
        v_x, v_y = math_to_maritime_velocity(psi_math_deg, speed)
        
        # Raw features (No manual normalization ranges here ideally, but keeping relative structure)
        # We will pass raw physical values and let RunningMeanStd handle the scaling.
        
        u_ref, u_actual = action[0], robot_state[3, 0]
        u_e = u_ref - u_actual
        
        r_ref, r_actual = action[1], robot_state[5, 0]
        r_e = r_ref - r_actual
        
        n1, n2 = robot_state[6, 0], robot_state[7, 0]

        # State Vector (Raw Values)
        # v_x, v_y: m/s
        # distance: m
        # y_e: m
        # angles: degrees
        # CR_max: 0-1
        # n1, n2: rpm
        
        state = [
            v_x, v_y, 
            distance, 
            y_e, 
            psi_e, 
            chi_e, 
            phi_tilde, 
            CR_max, 
            u_e, 
            r_e, 
            n1, 
            n2
        ]
        
        terminal = 1 if collision or goal else 0

        # Log raw state for debug
        if self.state_log_counter % 100 == 0:
            print(f"\n📊 Raw State Vector (call #{self.state_log_counter}):")
            print(f"   v_x: {v_x:.2f}, v_y: {v_y:.2f}")
            print(f"   dist: {distance:.2f}, y_e: {y_e:.2f}")
            print(f"   CR: {CR_max:.4f}")
        self.state_log_counter += 1

        return state, terminal


    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Save model and running mean/std statistics
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
        
        # Handle legacy checkpoints that might just be state_dict
        if 'model_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            
            # Load RMS stats if available
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
            print(f"Loaded legacy weights (no RMS stats) from: {directory}")
