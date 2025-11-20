"""
CNNPPO - CNN-based Proximal Policy Optimization (Improved)
Combines PPO algorithm with CNN feature extraction from CNNTD3

Improvements:
1. Separate learning rates for Actor and Critic
2. GAE (Generalized Advantage Estimation) for better variance reduction
3. Minibatch training for better sample efficiency
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from numpy import inf
from colregs_core.utils.utils import cross_track_error, ref_course_angle, WrapTo180, WrapToPi
from colregs_core.geometry import math_to_ned_heading, math_to_maritime_position


class RolloutBuffer:
    """
    Buffer to store rollout data (transitions) for PPO training.
    Now includes next_states for GAE calculation.
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []  # Added for GAE
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


class CNNPPOActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for PPO.
    
    Architecture from CNNTD3:
    - CNN for LiDAR feature extraction (360 → 36)
    - Embeddings for goal, action, velocity, RPS (4×10 = 40)
    - Total features: 76
    - FC layers: 76 → 400 → 300
    
    Actor: 300 → action_dim (mean of Gaussian policy)
    Critic: 300 → 1 (state value)
    """

    def __init__(self, action_dim, action_std_init, max_action, device):
        super(CNNPPOActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Action variance (diagonal covariance)
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(self.device)
        
        # ========== Shared CNN Feature Extractor ==========
        # CNN layers for LiDAR (same as CNNTD3)
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)
        
        # Embedding layers
        self.goal_embed = nn.Linear(3, 10)
        self.vel_embed = nn.Linear(2, 10)
        self.error_embed = nn.Linear(2, 10)
        self.rps_embed = nn.Linear(3, 10)
        
        # Shared FC layers
        self.fc1 = nn.Linear(76, 400)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        self.fc2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="leaky_relu")
        
        # ========== Actor head (policy) ==========
        self.actor_head = nn.Sequential(
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )
        
        # ========== Critic head (value) ==========
        self.critic_head = nn.Linear(300, 1)

    def _extract_features(self, s):
        """
        Extract features using CNN and embeddings.
        
        Args:
            s (torch.Tensor): State [batch_size, 370]
                             [LiDAR(360) + u_actual + r_actual+ u_e + r_e + distance + y_e + φ_tilde + n1 + n2 + CR_max]
        
        Returns:
            torch.Tensor: Extracted features [batch_size, 76]
        """
        if len(s.shape) == 1:
            s = s.unsqueeze(0)

        # Parse state (370차원)
        laser = s[:, :360]          # LiDAR: 360
        vel = s[:, -10:-8]         # u_actual, r_actual: 2
        error = s[:, -8:-6]        # u_e, r_e: 2
        goal = s[:, -6:-3]    # distance, y_e, φ_tilde: 3
        rps_cr = s[:, -3:]        # n1, n2, CR_max: 3
        
        # CNN processing
        laser = laser.unsqueeze(1)  # [batch, 1, 360]
        l = F.leaky_relu(self.cnn1(laser))  # [batch, 4, 88]
        l = F.leaky_relu(self.cnn2(l))      # [batch, 8, 20]
        l = F.leaky_relu(self.cnn3(l))      # [batch, 4, 9]
        l = l.flatten(start_dim=1)          # [batch, 36]
        
        # Embed other features
        v = F.leaky_relu(self.vel_embed(vel))
        e = F.leaky_relu(self.error_embed(error))
        g = F.leaky_relu(self.goal_embed(goal))
        r = F.leaky_relu(self.rps_embed(rps_cr))

        # Concatenate all features
        features = torch.concat((l, v, e, g, r), dim=-1)  # [batch, 76]
        
        return features

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, sample):
        """
        Compute an action, its log probability, and the state value.
        
        Args:
            state (Tensor): Input state tensor.
            sample (bool): Whether to sample from the action distribution or use mean.
        
        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Action, log probability, state value.
        """
        # Extract features
        features = self._extract_features(state)
        
        # Shared FC layers
        x = F.leaky_relu(self.fc1(features))
        x = F.leaky_relu(self.fc2(x))
        
        # Actor: get action mean
        action_mean = self.actor_head(x)
        
        # Create Gaussian distribution
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        if sample:
            action = torch.clip(
                dist.sample(), min=-self.max_action, max=self.max_action
            )
        else:
            action = dist.mean
            
        action_logprob = dist.log_prob(action)
        
        # Critic: get state value
        state_val = self.critic_head(x)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, entropy, and state values.
        
        Args:
            state (Tensor): Batch of states.
            action (Tensor): Batch of actions.
        
        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Log probs, state values, entropy.
        """
        # Extract features
        features = self._extract_features(state)
        
        # Shared FC layers
        x = F.leaky_relu(self.fc1(features))
        x = F.leaky_relu(self.fc2(x))
        
        # Actor: get action mean
        action_mean = self.actor_head(x)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        # Critic: get state value
        state_values = self.critic_head(x)

        return action_logprobs, state_values, dist_entropy
    
    def get_shared_params(self):
        """Get parameters of shared layers (CNN + embeddings + FC)"""
        return list(self.cnn1.parameters()) + \
               list(self.cnn2.parameters()) + \
               list(self.cnn3.parameters()) + \
               list(self.goal_embed.parameters()) + \
               list(self.vel_embed.parameters()) + \
               list(self.error_embed.parameters()) + \
               list(self.rps_embed.parameters()) + \
               list(self.fc1.parameters()) + \
               list(self.fc2.parameters())
    
    def get_actor_params(self):
        """Get parameters of actor head only"""
        return list(self.actor_head.parameters())
    
    def get_critic_params(self):
        """Get parameters of critic head only"""
        return list(self.critic_head.parameters())


class CNNPPO:
    """
    CNN-based Proximal Policy Optimization (CNNPPO) - Improved Version.
    
    Improvements over original:
    1. Separate learning rates for Actor and Critic
    2. GAE (Generalized Advantage Estimation) for variance reduction
    3. Minibatch training for sample efficiency
    
    Best for stable learning with high-dimensional LiDAR input! ✅
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0001,
        lr_critic=0.0003,
        gamma=0.99,
        gae_lambda=0.95,  # NEW: GAE lambda parameter
        eps_clip=0.2,
        action_std_init=0.6,
        action_std_decay_rate=0.015,
        min_action_std=0.1,
        target_kl=0.02,  # NEW: Target KL for early stopping
        device="cpu",
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name="CNNPPO",
        load_directory=Path("robot_nav/models/PPO/checkpoint"),
    ):
        self.max_action = max_action
        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda  # NEW: Store GAE lambda
        self.eps_clip = eps_clip
        self.target_kl = target_kl  # NEW
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        self.buffer = RolloutBuffer()

        self.policy = CNNPPOActorCritic(
            action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        
        # IMPROVED: Separate learning rates for Actor and Critic
        self.optimizer = torch.optim.Adam([
            # Shared layers use actor LR
            {"params": self.policy.get_shared_params(), "lr": lr_actor},
            # Actor head uses actor LR
            {"params": self.policy.get_actor_params(), "lr": lr_actor},
            # Critic head uses critic LR (typically higher)
            {"params": self.policy.get_critic_params(), "lr": lr_critic},
        ])

        self.policy_old = CNNPPOActorCritic(
            action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter(comment=model_name)
        
        print(f"✅ CNNPPO Initialized with:")
        print(f"   - Actor LR: {lr_actor}")
        print(f"   - Critic LR: {lr_critic}")
        print(f"   - GAE Lambda: {gae_lambda}")
        print(f"   - Epsilon Clip: {eps_clip}")

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("---" * 30)
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print(f"setting actor output action_std to min_action_std: {self.action_std}")
        else:
            print(f"setting actor output action_std to: {self.action_std}")
        self.set_action_std(self.action_std)
        print("---" * 30)

    def get_action(self, state, add_noise):
        """
        Sample an action using the current policy.
        Returns the action, its log probability, and the state value.
        """
        with torch.no_grad():
            state = np.array(state, dtype=np.float32)  # 간단!
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, add_noise)

        return action.detach().cpu().numpy().flatten(), action_logprob, state_val
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        NEW: Compute Generalized Advantage Estimation (GAE).
        
        GAE formula:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        
        Args:
            rewards (Tensor): Rewards [T]
            values (Tensor): State values V(s_t) [T]
            next_values (Tensor): Next state values V(s_{t+1}) [T]
            dones (Tensor): Terminal flags [T]
        
        Returns:
            advantages (Tensor): GAE advantages [T]
            returns (Tensor): TD(λ) returns [T]
        """
        advantages = []
        gae = 0
        
        # Compute GAE in reverse order
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Terminal state: no next value
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # Non-terminal: TD error
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        # Convert to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Returns = Advantages + Values
        returns = advantages + values
        
        return advantages, returns

    def train(self, replay_buffer, iterations, batch_size):
        """
        IMPROVED: Train the policy using PPO with GAE and minibatch updates.
        
        Args:
            replay_buffer: Not used (kept for compatibility)
            iterations: Number of optimization epochs
            batch_size: Minibatch size for training
        """
        # Convert buffer to tensors
        assert len(self.buffer.actions) == len(self.buffer.states)

        
        states = torch.FloatTensor(np.array(self.buffer.states, dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.buffer.next_states, dtype=np.float32)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer.actions, dtype=np.float32)).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_state_values = torch.stack(self.buffer.state_values).squeeze().to(self.device)
        
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        
        # Compute next state values for GAE
        with torch.no_grad():
            next_features = self.policy._extract_features(next_states)
            next_x = F.leaky_relu(self.policy.fc1(next_features))
            next_x = F.leaky_relu(self.policy.fc2(next_x))
            next_state_values = self.policy.critic_head(next_x).squeeze()
        
        # NEW: Compute GAE advantages
        advantages, returns = self.compute_gae(
            rewards, old_state_values, next_state_values, dones
        )
        
        # NEW: Calculate explained variance
        explained_var = 1 - torch.var(returns - old_state_values) / (torch.var(returns) + 1e-8)
        
        # Squeeze tensors to proper shapes
        if len(states.shape) > 2:
            states = torch.squeeze(states)
        if len(old_actions.shape) > 2:
            old_actions = torch.squeeze(old_actions)
        if len(old_logprobs.shape) > 1:
            old_logprobs = torch.squeeze(old_logprobs)
        
        # Calculate dataset size
        dataset_size = states.shape[0]
        
        # Training metrics
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_approx_kl = 0  # NEW
        total_clip_frac = 0  # NEW
        num_updates = 0
        
        # NEW: Multiple epochs with minibatch updates
        for epoch in range(iterations):
            # Shuffle indices for minibatch sampling
            indices = torch.randperm(dataset_size)
            
            # Process minibatches
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample minibatch
                batch_states = states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy on minibatch
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                state_values = torch.squeeze(state_values)
                
                # Calculate ratio (π_θ / π_θ_old)
                ratios = torch.exp(logprobs - batch_old_logprobs)
                
                # Surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                # Actor loss (PPO clipped objective)
                actor_loss = -torch.min(surr1, surr2)
                
                # Critic loss (value function)
                critic_loss = 0.5 * self.MseLoss(state_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -0.01 * dist_entropy
                
                # Total loss
                loss = actor_loss + critic_loss + entropy_loss

                # NEW: Calculate approximate KL divergence and clip fraction for logging
                with torch.no_grad():
                    approx_kl = torch.mean((ratios - 1) - logprobs + batch_old_logprobs).item()
                    clip_frac = torch.mean((torch.abs(ratios - 1) > self.eps_clip).float()).item()

                # NEW: Early stopping based on KL divergence
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    print(f"⚠️  Early stopping PPO update at epoch {epoch+1} due to high KL divergence: {approx_kl:.4f}")
                    break
                
                # Gradient update
                self.optimizer.zero_grad()
                loss.mean().backward()
                # Gradient clipping for stability (Sawada et al. 2020)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.mean().item()
                total_actor_loss += actor_loss.mean().item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                num_updates += 1
            
            # NEW: Break outer loop if KL early stopping was triggered
            else:  # This 'else' belongs to the inner 'for' loop
                continue
            break
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        # Decay action std
        self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
        
        self.iter_count += 1
        
        # Write to tensorboard
        avg_loss = total_loss / num_updates
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_approx_kl = total_approx_kl / num_updates
        avg_clip_frac = total_clip_frac / num_updates
        
        self.writer.add_scalar("train/total_loss", avg_loss, self.iter_count)
        self.writer.add_scalar("train/actor_loss", avg_actor_loss, self.iter_count)
        self.writer.add_scalar("train/critic_loss", avg_critic_loss, self.iter_count)
        self.writer.add_scalar("train/entropy", avg_entropy, self.iter_count)
        self.writer.add_scalar("train/action_std", self.action_std, self.iter_count)
        
        # Log advantages statistics
        self.writer.add_scalar("train/advantages_mean", advantages.mean().item(), self.iter_count)
        self.writer.add_scalar("train/advantages_std", advantages.std().item(), self.iter_count)

        # NEW: Log advanced PPO metrics
        self.writer.add_scalar("train/explained_variance", explained_var.item(), self.iter_count)
        self.writer.add_scalar("train/approx_kl", avg_approx_kl, self.iter_count)
        self.writer.add_scalar("train/clip_fraction", avg_clip_frac, self.iter_count)
        
        # Print training info
        if self.iter_count % 10 == 0:
            print(f"Iter {self.iter_count} | Loss: {avg_loss:.4f} | "
                  f"Actor: {avg_actor_loss:.4f} | Critic: {avg_critic_loss:.4f} | "
                  f"Entropy: {avg_entropy:.4f}")

    def prepare_state(
        self, 
        latest_scan,      # LiDAR (360)
        distance,         # 목적지 거리
        y_e,              # Cross-track error
        collision,        # 충돌 여부
        goal,             # 목표 도달 여부 (boolean) - not used, kept for compatibility
        action,           # 현재 명령 [u_ref, r_ref]
        robot_state,      # 로봇 상태
        start_position,   # 출발 위치 (for y_e)
        goal_position,    # 목표 위치 [x, y] (for y_e calculation)
        CR_max
    ):
        """
        State: [LiDAR(360) + u + r + u_e + r_e + distance + y_e + φ_tilde + n1 + n2 + CR_max]
        """

        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 100.0
        latest_scan_norm = np.clip(latest_scan / 100.0, 0.0, 1.0)
        
        # Normalize distance
        distance_min, distance_max = 0, 111.8
        distance_norm = normalize_state(distance, distance_min, distance_max)
        
        # Normalize propellers
        n_min, n_max = -101.7, 103.9
        n1, n2 = robot_state[6, 0], robot_state[7, 0]
        n1_norm = normalize_state(n1, n_min, n_max)
        n2_norm = normalize_state(n2, n_min, n_max)
        
        # Normalize velocities
        u_min, u_max = 0, 3.0
        u_e_min, u_e_max = -3.0, 3.0
        u_ref, u_actual = action[0], robot_state[3, 0]
        u_e = u_ref - u_actual
        u_actual_norm = normalize_state(u_actual, u_min, u_max)
        u_e_norm = normalize_state(u_e, u_e_min, u_e_max)
        
        r_min, r_max = -0.2862, 0.2862
        r_e_min, r_e_max = -0.4607, 0.4607
        r_ref, r_actual = action[1], robot_state[5, 0]
        r_e = r_ref - r_actual
        r_actual_norm = normalize_state(r_actual, r_min, r_max)
        r_e_norm = normalize_state(r_e, r_e_min, r_e_max)

        # Calculate y_e (cross-track error)
        y_e_min, y_e_max = -50.0, 50.0
        y_e_norm = normalize_state(y_e, y_e_min, y_e_max)

        # Calculate φ_tilde (heading difference) in RADIANS
        φ_tilde_min, φ_tilde_max = -np.pi, np.pi  # radians
        os_heading_math = np.degrees(robot_state[2, 0])  # math heading in degrees
        os_heading_deg = math_to_ned_heading(os_heading_math) # NED heading in degrees
        os_heading = np.radians(os_heading_deg) # NED heading in radians

        # Get reference angle (returns degrees, convert to radians)
        ref_angle_deg = ref_course_angle(start_position, goal_position)  # Returns degrees
        ref_angle = np.radians(ref_angle_deg)  # Convert to radians
        
        # Calculate difference in radians, wrap to [-π, π]
        φ_tilde = WrapToPi(os_heading - ref_angle)  # radians
        φ_tilde_norm = normalize_state(φ_tilde, φ_tilde_min, φ_tilde_max)
        
        # Concatenate: [LiDAR(360) + u + r + u_e + r_e + distance + y_e + φ_tilde + n1 + n2 + CR_max]
        # 8. State 조합
        state = np.concatenate([
            latest_scan_norm.flatten(),
            np.array([u_actual_norm, r_actual_norm, u_e_norm, r_e_norm]),
            np.array([distance_norm, y_e_norm, φ_tilde_norm]),
            np.array([n1_norm, n2_norm, CR_max])
        ])

        assert len(state) == self.state_dim, f"State dim mismatch: {len(state)} vs {self.state_dim}"
        terminal = 1 if collision or goal else 0

        return state.tolist(), terminal  # numpy → list for buffer storage

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(
            self.policy_old.state_dict(), "%s/%s_policy.pth" % (directory, filename)
        )

    def load(self, filename, directory):
        self.policy_old.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        self.policy.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        print(f"Loaded weights from: {directory}")


def normalize_state(x, min_val, max_val):
    """Normalize state value to [0, 1] range"""
    x = np.asarray(x, dtype=np.float32)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return np.clip((x - min_val) / denom, 0.0, 1.0)
