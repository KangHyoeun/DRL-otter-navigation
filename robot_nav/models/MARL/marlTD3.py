from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.hsAttention import Attention


class Actor(nn.Module):
    def __init__(self, action_dim, embedding_dim):
        super().__init__()
        self.attention = Attention(embedding_dim)  # ➊ edge classifier

        # ➋ policy head (everything _after_ attention)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs, detach_attn=False):
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(obs)
        )
        if detach_attn:  # used in the policy phase
            attn_out = attn_out.detach()
        action = self.policy_head(attn_out)
        return action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights


class Critic(nn.Module):
    def __init__(self, action_dim, embedding_dim):
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention = Attention(self.embedding_dim)

        self.layer_1 = nn.Linear(self.embedding_dim * 2, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")

        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")

        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")

        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(self.embedding_dim * 2, 400)
        torch.nn.init.kaiming_uniform_(
            self.layer_4.weight, nonlinearity="leaky_relu"
        )  # ✅ Fixed init bug

        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")

        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")

        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, embedding, action):

        (
            embedding_with_attention,
            hard_logits,
            unnorm_rel_dist,
            mean_entropy,
            hard_weights,
            _,
        ) = self.attention(embedding)

        # Q1
        s1 = F.leaky_relu(self.layer_1(embedding_with_attention))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(action))  # ✅ No .data
        q1 = self.layer_3(s1)

        # Q2
        s2 = F.leaky_relu(self.layer_4(embedding_with_attention))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(action))  # ✅ No .data
        q2 = self.layer_6(s2)

        return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights


class TD3(object):
    """
    CNNTD3 (Twin Delayed Deep Deterministic Policy Gradient with CNN-based inputs) agent for
    continuous control tasks.

    This class encapsulates the full implementation of the TD3 algorithm using neural network
    architectures for the actor and critic, with optional bounding for critic outputs to
    regularize learning. The agent is designed to train in environments where sensor
    observations (e.g., LiDAR) are used for navigation tasks.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        max_action (float): Maximum magnitude of the action.
        device (torch.device): Torch device to use (CPU or GPU).
        lr (float): Learning rate for both actor and critic optimizers.
        save_every (int): Save model every N training iterations (0 to disable).
        load_model (bool): Whether to load a pre-trained model at initialization.
        save_directory (Path): Path to the directory for saving model checkpoints.
        model_name (str): Base name for the saved model files.
        load_directory (Path): Path to load model checkpoints from (if `load_model=True`).
        use_max_bound (bool): Whether to apply maximum Q-value bounding during training.
        bound_weight (float): Weight for the bounding loss term in total loss.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        num_robots,
        lr_actor=1e-4,
        lr_critic=3e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/MARL/checkpoint"),
        model_name="marlTD3",
        load_model_name=None,
        load_directory=Path("robot_nav/models/MARL/checkpoint"),
    ):
        # Initialize the Actor network
        self.num_robots = num_robots
        self.device = device
        self.actor = Actor(action_dim, embedding_dim=256).to(
            self.device
        )  # Using the updated Actor
        self.actor_target = Actor(action_dim, embedding_dim=256).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.attn_params = list(self.actor.attention.parameters())
        self.policy_params = list(self.actor.policy_head.parameters())

        self.actor_optimizer = torch.optim.Adam(
            self.policy_params + self.attn_params, lr=lr_actor
        )  # TD3 policy

        self.critic = Critic(action_dim, embedding_dim=256).to(
            self.device
        )  # Using the updated Critic
        self.critic_target = Critic(action_dim, embedding_dim=256).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(), lr=lr_critic
        )
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        if load_model_name is None:
            load_model_name = model_name
        if load_model:
            self.load(filename=load_model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def get_action(self, obs, add_noise):
        """
        Selects an action for a given observation.

        Args:
            obs (np.ndarray): The current observation/state.
            add_noise (bool): Whether to add exploration noise to the action.

        Returns:
            (np.ndarray): The selected action.
        """
        action, connection, combined_weights = self.act(obs)
        if add_noise:
            noise = np.random.normal(0, 0.5, size=action.shape)
            noise = [n / 4 if i % 2 else n for i, n in enumerate(noise)]
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action.reshape(-1, 2), connection, combined_weights

    def act(self, state):
        """
        Computes the deterministic action from the actor network for a given state.

        Args:
            state (np.ndarray): Input state.

        Returns:
            (np.ndarray): Action predicted by the actor network.
        """
        # Function to get the action from the actor
        state = torch.Tensor(state).to(self.device)
        # res = self.attention(state)
        action, connection, _, _, _, combined_weights = self.actor(state)
        return action.cpu().data.numpy().flatten(), connection, combined_weights

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_critic_loss = 0
        av_critic_entropy = []
        av_actor_entropy = []
        av_actor_loss = 0
        av_critic_bce_loss = []
        av_actor_bce_loss = []

        for it in range(iterations):
            # sample a batch
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state = (
                torch.Tensor(batch_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            next_state = (
                torch.Tensor(batch_next_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            action = (
                torch.Tensor(batch_actions)
                .to(self.device)
                .view(batch_size * self.num_robots, self.action_dim)
            )
            reward = (
                torch.Tensor(batch_rewards)
                .to(self.device)
                .view(batch_size * self.num_robots, 1)
            )
            done = (
                torch.Tensor(batch_dones)
                .to(self.device)
                .view(batch_size * self.num_robots, 1)
            )

            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(
                    next_state, detach_attn=True
                )

            # --- Target smoothing ---
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            ).reshape(-1, 2)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # --- Target Q values ---
            target_Q1, target_Q2, _, _, _, _ = self.critic_target(
                next_state, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += target_Q.mean()
            max_Q = max(max_Q, target_Q.max().item())
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # --- Critic update ---
            (
                current_Q1,
                current_Q2,
                mean_entropy,
                hard_logits,
                unnorm_rel_dist,
                hard_weights,
            ) = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            proximity_threshold = 4  # You may need to adjust this
            targets = (unnorm_rel_dist.flatten() < proximity_threshold).float()
            flat_logits = hard_logits.flatten()
            bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)

            bce_weight = 0.1
            av_critic_bce_loss.append(bce_loss)

            critic_entropy_weight = 1  # or tuneable
            total_loss = (
                critic_loss
                - critic_entropy_weight * mean_entropy
                + bce_weight * bce_loss
            )
            av_critic_entropy.append(mean_entropy)

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            av_loss += total_loss.item()
            av_critic_loss += critic_loss.item()

            # --- Actor update ---
            if it % policy_freq == 0:

                action, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = (
                    self.actor(state, detach_attn=False)
                )
                targets = (unnorm_rel_dist.flatten() < proximity_threshold).float()
                flat_logits = hard_logits.flatten()
                bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)

                bce_weight = 0.1
                av_actor_bce_loss.append(bce_loss)

                actor_Q, _, _, _, _, _ = self.critic(state, action)
                actor_loss = -actor_Q.mean()
                actor_entropy_weight = 0.05
                total_loss = (
                    actor_loss
                    - actor_entropy_weight * mean_entropy
                    + bce_weight * bce_loss
                )
                av_actor_entropy.append(mean_entropy)

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_params, 10.0)
                self.actor_optimizer.step()

                av_actor_loss += total_loss.item()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        self.iter_count += 1
        self.writer.add_scalar(
            "train/loss_total", av_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/critic_loss", av_critic_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/av_critic_entropy",
            sum(av_critic_entropy) / len(av_critic_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_entropy",
            sum(av_actor_entropy) / len(av_actor_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_critic_bce_loss",
            sum(av_critic_bce_loss) / len(av_critic_bce_loss),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_bce_loss",
            sum(av_actor_bce_loss) / len(av_actor_bce_loss),
            self.iter_count,
        )
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        self.writer.add_scalar(
            "train/actor_loss",
            av_actor_loss / (iterations // policy_freq),
            self.iter_count,
        )

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        """
        Saves the current model parameters to the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to save the model files.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, filename),
        )
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        """
        Loads model parameters from the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to load the model files from.
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def prepare_state(
        self, poses, distance, cos, sin, collision, action, goal_positions
    ):
        """
        Prepares the environment's raw agent state for learning.

        Args:
            poses (list): Each agent's global pose [x, y, theta].
            distance, cos, sin: Unused, can be removed or ignored.
            collision (list): Collision flags per agent.
            goal (list): Goal reached flags per agent.
            action (list): Last action taken [lin_vel, ang_vel].
            positions (list): Extra features (e.g., neighbors).
            goal_positions (list): Each agent's goal [x, y].

        Returns:
            states (list): List of processed state vectors.
            terminal (list): Flags (1 if collision or goal reached, else 0).
        """
        states = []
        terminal = []

        for i in range(self.num_robots):
            pose = poses[i]  # [x, y, theta]
            goal_pos = goal_positions[i]  # [goal_x, goal_y]
            act = action[i]  # [lin_vel, ang_vel]

            px, py, theta = pose
            gx, gy = goal_pos

            # Heading as cos/sin
            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)

            # Last velocity
            lin_vel = act[0] * 2  # Assuming original range [-0.5, 0.5]
            ang_vel = (act[1] + 1) / 2  # Assuming original range [-1, 1]

            # Final state vector
            state = [
                px,
                py,
                heading_cos,
                heading_sin,
                distance[i] / 17,
                cos[i],
                sin[i],
                lin_vel,
                ang_vel,
                gx,
                gy,
            ]

            assert (
                len(state) == self.state_dim
            ), f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            states.append(state)
            terminal.append(collision[i])

        return states, terminal
