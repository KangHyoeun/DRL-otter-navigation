"""
CNNPPO - CNN-based Proximal Policy Optimization
Combines PPO algorithm with CNN feature extraction from CNNTD3
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from numpy import inf


class RolloutBuffer:
    """
    Buffer to store rollout data (transitions) for PPO training.
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state):
        self.states.append(state)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)


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
        self.action_embed = nn.Linear(2, 10)
        self.vel_embed = nn.Linear(2, 10)
        self.rps_embed = nn.Linear(2, 10)
        
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
            s (torch.Tensor): State [batch_size, 369]
                             [LiDAR(360) + distance + cos + sin + u_ref + r_ref + u + r + n1 + n2]
        
        Returns:
            torch.Tensor: Extracted features [batch_size, 76]
        """
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        
        # Parse state
        laser = s[:, :-9]      # LiDAR: 360
        goal = s[:, -9:-6]     # distance, cos, sin: 3
        act = s[:, -6:-4]      # u_ref, r_ref: 2
        vel = s[:, -4:-2]      # u_actual, r_actual: 2
        rps = s[:, -2:]        # n1, n2: 2
        
        # CNN processing
        laser = laser.unsqueeze(1)  # [batch, 1, 360]
        l = F.leaky_relu(self.cnn1(laser))  # [batch, 4, 88]
        l = F.leaky_relu(self.cnn2(l))      # [batch, 8, 20]
        l = F.leaky_relu(self.cnn3(l))      # [batch, 4, 9]
        l = l.flatten(start_dim=1)          # [batch, 36]
        
        # Embed other features
        g = F.leaky_relu(self.goal_embed(goal))
        a = F.leaky_relu(self.action_embed(act))
        v = F.leaky_relu(self.vel_embed(vel))
        r = F.leaky_relu(self.rps_embed(rps))
        
        # Concatenate all features
        features = torch.concat((l, g, a, v, r), dim=-1)  # [batch, 76]
        
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


class CNNPPO:
    """
    CNN-based Proximal Policy Optimization (CNNPPO).
    
    Combines:
    - PPO algorithm (clipped objective, on-policy)
    - CNNTD3 architecture (CNN for LiDAR feature extraction)
    
    Best for stable learning with high-dimensional LiDAR input! ✅
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        eps_clip=0.2,
        action_std_init=0.6,
        action_std_decay_rate=0.015,
        min_action_std=0.1,
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
        self.eps_clip = eps_clip
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        self.buffer = RolloutBuffer()

        self.policy = CNNPPOActorCritic(
            action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
            ]
        )

        self.policy_old = CNNPPOActorCritic(
            action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter(comment=model_name)

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
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, add_noise)

        if add_noise:
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size):
        """
        Train the policy and value function using PPO loss.
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        assert len(self.buffer.actions) == len(self.buffer.states)

        states = [torch.tensor(st, dtype=torch.float32) for st in self.buffer.states]
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(self.device)
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        av_state_values = 0
        max_state_value = -inf
        av_loss = 0
        
        # Optimize policy for K epochs
        for _ in range(iterations):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            av_state_values += torch.mean(state_values)
            max_state_value = max(max_state_value, max(state_values))
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            av_loss += loss.mean()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        # Decay action std
        self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
        
        self.iter_count += 1
        
        # Write to tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/avg_value", av_state_values / iterations, self.iter_count
        )
        self.writer.add_scalar("train/max_value", max_state_value, self.iter_count)
        self.writer.add_scalar("train/action_std", self.action_std, self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action, robot_state):
        """
        Convert raw sensor data into state vector for CNN processing.
        
        State format: [LiDAR(360) + distance + cos + sin + u_ref + r_ref + u + r + n1 + n2] = 369
        """
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 100.0
        
        # Normalize LiDAR
        latest_scan_norm = latest_scan / 100.0
        
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
        u_ref, u_actual = action[0], robot_state[3, 0]
        u_ref_norm = normalize_state(u_ref, u_min, u_max)
        u_actual_norm = normalize_state(u_actual, u_min, u_max)
        
        r_ref_min, r_ref_max = -0.1745, 0.1745
        r_actual_min, r_actual_max = -0.2862, 0.2862
        r_ref, r_actual = action[1], robot_state[5, 0]
        r_ref_norm = normalize_state(r_ref, r_ref_min, r_ref_max)
        r_actual_norm = normalize_state(r_actual, r_actual_min, r_actual_max)
        
        # Concatenate: [LiDAR(360) + distance + cos + sin + u_ref + r_ref + u + r + n1 + n2]
        state = list(latest_scan_norm) + [distance_norm, cos, sin] + \
                [u_ref_norm, r_ref_norm] + [u_actual_norm, r_actual_norm, n1_norm, n2_norm]
        
        assert len(state) == self.state_dim, f"State dim mismatch: {len(state)} vs {self.state_dim}"
        
        terminal = 1 if collision or goal else 0
        
        return state, terminal

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
