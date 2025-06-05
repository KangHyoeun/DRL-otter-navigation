from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.utils import get_max_bound


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim

        # CNN for laser scan
        self.embedding1= nn.Linear(5, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")


        # Hard attention MLP with distance
        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Soft attention projections
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        # Soft attention score network (with distance)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        self.v_proj = nn.Linear(7, embedding_dim)
        # Decoder
        self.decode_1 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_1.weight, nonlinearity="leaky_relu")
        self.decode_2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_2.weight, nonlinearity="leaky_relu")

    def encode_agent_features(self, embed):
        embed = F.leaky_relu(self.embedding1(embed))
        embed = F.leaky_relu(self.embedding2(embed))
        return embed

    def forward(self, embedding):
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        batch_size, n_agents, _ = embedding.shape

        embed = embedding[:, :, 4:].reshape(batch_size * n_agents, -1)
        position = embedding[:, :, :2].reshape(batch_size, n_agents, 2)
        heading = embedding[:, :, 2:4].reshape(batch_size, n_agents, 2)  # assume (cos(θ), sin(θ))
        action = embedding[:, :, -2:].reshape(batch_size, n_agents, 2)

        agent_embed = self.encode_agent_features(embed)
        agent_embed = agent_embed.view(batch_size, n_agents, self.embedding_dim)

        # For hard attention
        h_i = agent_embed.unsqueeze(2)  # (B, N, 1, D)
        pos_i = position.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = position.unsqueeze(1)  # (B, 1, N, 2)
        heading_i = heading.unsqueeze(2)  # (B, N, 1, 2)
        heading_j = heading.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (B, 1, N, 2)

        # Compute relative vectors and distance
        rel_vec = pos_j - pos_i  # (B, N, N, 2)
        dx, dy = rel_vec[..., 0], rel_vec[..., 1]
        rel_dist = torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True)  # (B, N, N, 1)

        # Relative angle in agent i's frame
        angle = torch.atan2(dy, dx) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_sin = torch.sin(angle)
        rel_angle_cos = torch.cos(angle)

        # Other agent's heading
        heading_j_cos = heading_j[..., 0]  # (B, 1, N)
        heading_j_sin = heading_j[..., 1]  # (B, 1, N)

        # Stack edge features
        edge_features = torch.cat([
            rel_dist,  # (B, N, N, 1)
            rel_angle_cos.unsqueeze(-1),  # (B, N, N, 1)
            rel_angle_sin.unsqueeze(-1),  # (B, N, N, 1)
            heading_j_cos.unsqueeze(-1),  # (B, N, N, 1)
            heading_j_sin.unsqueeze(-1),  # (B, N, N, 1)
            action.unsqueeze(1).expand(-1, n_agents, -1, -1) # (B, N, N, 2)
        ], dim=-1)  # (B, N, N, 7)

        # Broadcast h_i along N (for each pair)
        h_i_expanded = h_i.expand(-1, -1, n_agents, -1)  # (B, N, N, D)

        # Remove self-pairs using mask
        mask = ~torch.eye(n_agents, dtype=torch.bool, device=embedding.device)
        h_i_flat = h_i_expanded[:, mask].reshape(batch_size * n_agents, n_agents - 1, self.embedding_dim)
        edge_flat = edge_features[:, mask].reshape(batch_size * n_agents, n_agents - 1, -1)

        # Concatenate agent embedding and edge features
        hard_input = torch.cat([h_i_flat, edge_flat], dim=-1)  # (B*N, N-1, D+7)

        # Hard attention forward
        h_hard = self.hard_mlp(hard_input)
        hard_logits = self.hard_encoding(h_hard)
        hard_weights = F.gumbel_softmax(hard_logits, hard=False, tau=0.5, dim=-1)[..., 1].unsqueeze(2)
        hard_weights = hard_weights.view(batch_size, n_agents, n_agents - 1)

        unnorm_rel_vec = rel_vec * 12
        unnorm_rel_dist = torch.linalg.vector_norm(unnorm_rel_vec, dim=-1, keepdim=True)
        unnorm_rel_dist = unnorm_rel_dist[:, mask].reshape(batch_size * n_agents, n_agents - 1, 1)

        # Soft attention
        q = self.q(agent_embed)

        attention_outputs = []
        entropy_list = []
        combined_w = []
        for i in range(n_agents):
            q_i = q[:, i:i + 1, :]  # (B, 1, D)
            mask = torch.ones(n_agents, dtype=torch.bool, device=edge_features.device)
            mask[i] = False
            edge_i_wo_self = edge_features[:, i, mask, :]
            edge_i_wo_self = edge_i_wo_self.squeeze(1)  # (B, N-1, 7)

            q_i_expanded = q_i.expand(-1, n_agents - 1, -1)  # (B, N-1, D)
            attention_input = torch.cat([q_i_expanded, edge_i_wo_self], dim=-1)  # (B, N-1, D+7)

            # Score computation
            scores = self.attn_score_layer(attention_input).transpose(1, 2)  # (B, 1, N-1)

            # Mask using hard weights
            h_weights = hard_weights[:, i].unsqueeze(1)  # (B, 1, N-1)
            mask = (h_weights > 0.5).float()

            # All-zero mask handling
            epsilon = 1e-6
            mask_sum = mask.sum(dim=-1, keepdim=True)
            all_zero_mask = (mask_sum < epsilon)

            # Apply mask to scores
            masked_scores = scores.masked_fill(mask == 0, float('-inf'))

            # Compute softmax, safely handle all-zero cases
            soft_weights = F.softmax(masked_scores, dim=-1)
            soft_weights = torch.where(all_zero_mask, torch.zeros_like(soft_weights), soft_weights)

            combined_weights = soft_weights * mask  # (B, 1, N-1)
            combined_w.append(combined_weights)

            # Normalize combined_weights for entropy calculation
            combined_weights_norm = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + epsilon)

            # Calculate entropy from combined_weights
            entropy = -(combined_weights_norm * (combined_weights_norm + epsilon).log()).sum(dim=-1).mean()
            entropy_list.append(entropy)

            # Project each other agent's features to embedding dim *before* the attention-weighted sum
            v_j = self.v_proj(edge_i_wo_self)  # (B, N-1, embedding_dim)
            attn_output = torch.bmm(combined_weights, v_j).squeeze(1)  # (B, embedding_dim)
            attention_outputs.append(attn_output)

        comb_w = torch.stack(combined_w, dim=1).reshape(n_agents, -1)
        attn_stack = torch.stack(attention_outputs, dim=1).reshape(-1, self.embedding_dim)
        self_embed = agent_embed.reshape(-1, self.embedding_dim)
        concat_embed = torch.cat([self_embed, attn_stack], dim=-1)

        x = F.leaky_relu(self.decode_1(concat_embed))
        att_embedding = F.leaky_relu(self.decode_2(x))

        mean_entropy = torch.stack(entropy_list).mean()

        return att_embedding, hard_logits[..., 1], unnorm_rel_dist, mean_entropy, hard_weights, comb_w


class Actor(nn.Module):
    def __init__(self, action_dim, embedding_dim):
        super().__init__()
        self.attention = Attention(embedding_dim)          # ➊ edge classifier

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
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = self.attention(obs)
        if detach_attn:            # used in the policy phase
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
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")  # ✅ Fixed init bug

        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")

        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")

        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, embedding, action):

        embedding_with_attention, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = self.attention(embedding)

        # Q1
        s1 = F.leaky_relu(self.layer_1(embedding_with_attention))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(action))  # ✅ No .data
        q1 = self.layer_3(s1)

        # Q2
        s2 = F.leaky_relu(self.layer_4(embedding_with_attention))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(action))  # ✅ No .data
        q2 = self.layer_6(s2)

        return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights



# CNNTD3 network
class CNNTD3(object):
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
        save_directory=Path("robot_nav/models/CNNTD3/checkpoint"),
        model_name="CNNTD3",
        load_model_name = None,
        load_directory=Path("robot_nav/models/CNNTD3/checkpoint"),
        use_max_bound=False,
        bound_weight=0.25,
    ):
        # Initialize the Actor network
        self.num_robots = num_robots
        self.device = device
        self.actor = Actor(action_dim, embedding_dim=256).to(self.device)  # Using the updated Actor
        self.actor_target = Actor(action_dim, embedding_dim=256).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)

        self.attn_params = list(self.actor.attention.parameters())
        self.policy_params = list(self.actor.policy_head.parameters())

        # self.attn_opt = torch.optim.Adam(self.attn_params, lr=1e-3)  # edge classifier
        self.actor_optimizer = torch.optim.Adam(self.policy_params + self.attn_params, lr=lr_actor)  # TD3 policy

        self.critic = Critic(action_dim, embedding_dim=256).to(self.device)  # Using the updated Critic
        self.critic_target = Critic(action_dim, embedding_dim=256).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr_critic)
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
        self.use_max_bound = use_max_bound
        self.bound_weight = bound_weight

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
            action = (action + np.random.normal(0, 0.1, size=action.shape)
            ).clip(-self.max_action, self.max_action)

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
            max_lin_vel=0.5,
            max_ang_vel=1,
            goal_reward=100,
            distance_norm=10,
            time_step=0.3,
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

            state = torch.Tensor(batch_states).to(self.device).view(batch_size, self.num_robots, self.state_dim)
            next_state = torch.Tensor(batch_next_states).to(self.device).view(batch_size, self.num_robots, self.state_dim)
            action = torch.Tensor(batch_actions).to(self.device).view(batch_size * self.num_robots, self.action_dim)
            reward = torch.Tensor(batch_rewards).to(self.device).view(batch_size * self.num_robots, 1)
            done = torch.Tensor(batch_dones).to(self.device).view(batch_size * self.num_robots, 1)

            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(next_state, detach_attn=True)

            # --- Target smoothing ---
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            ).reshape(-1, 2)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # --- Target Q values ---
            target_Q1, target_Q2, _, _, _, _ = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += target_Q.mean()
            max_Q = max(max_Q, target_Q.max().item())
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # --- Critic update ---
            current_Q1, current_Q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            proximity_threshold = 4  # You may need to adjust this
            targets = (unnorm_rel_dist.flatten() < proximity_threshold).float()
            flat_logits = hard_logits.flatten()
            bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)
            # masked_weights = hard_weights.flatten()[mask]
            # target = torch.ones_like(masked_weights)
            # num_pos = masked_weights.numel()
            # if num_pos > 0:
            #     bce_loss = F.binary_cross_entropy(masked_weights, target, reduction='sum') / num_pos
            # else:
            #     bce_loss = torch.tensor(0.0, device=masked_weights.device)

            bce_weight = 0.1
            av_critic_bce_loss.append(bce_loss)

            critic_entropy_weight = 1  # or tuneable
            total_loss = critic_loss - critic_entropy_weight * mean_entropy + bce_weight * bce_loss
            av_critic_entropy.append(mean_entropy)

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            av_loss += total_loss.item()
            av_critic_loss += critic_loss.item()

            # --- Actor update ---
            if it % policy_freq == 0:

                action, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = self.actor(state, detach_attn=False)
                targets = (unnorm_rel_dist.flatten() < proximity_threshold).float()
                flat_logits = hard_logits.flatten()
                bce_loss = F.binary_cross_entropy_with_logits(flat_logits, targets)
                # masked_weights = hard_weights.flatten()[mask]
                # target = torch.ones_like(masked_weights)
                # num_pos = masked_weights.numel()
                # if num_pos > 0:
                #     bce_loss = F.binary_cross_entropy(masked_weights, target, reduction='sum') / num_pos
                # else:
                #     bce_loss = torch.tensor(0.0, device=masked_weights.device)

                bce_weight = 0.1
                av_actor_bce_loss.append(bce_loss)

                actor_Q, _, _, _, _, _ = self.critic(state, action)
                actor_loss = -actor_Q.mean()
                actor_entropy_weight = 0.05
                total_loss = actor_loss - actor_entropy_weight * mean_entropy + bce_weight * bce_loss
                av_actor_entropy.append(mean_entropy)

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_params, 10.0)
                self.actor_optimizer.step()

                av_actor_loss += total_loss.item()

                # Soft update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1
        self.writer.add_scalar("train/loss_total", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/critic_loss", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/av_critic_entropy", sum(av_critic_entropy) / len(av_critic_entropy), self.iter_count)
        self.writer.add_scalar("train/av_actor_entropy", sum(av_actor_entropy) / len(av_actor_entropy),
                               self.iter_count)
        self.writer.add_scalar("train/av_critic_bce_loss", sum(av_critic_bce_loss) / len(av_critic_bce_loss),
                               self.iter_count)
        self.writer.add_scalar("train/av_actor_bce_loss", sum(av_actor_bce_loss) / len(av_actor_bce_loss),
                               self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        self.writer.add_scalar("train/actor_loss", av_actor_loss / (iterations // policy_freq), self.iter_count)

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

    def prepare_state(self, poses, distance, cos, sin, collision, goal, action, positions, goal_positions):
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

            # Global position (keep for boundary awareness)
            x = px / 12
            y = py / 12

            # Relative goal position in local frame
            dx = gx - px
            dy = gy - py
            rel_gx = dx * np.cos(theta) + dy * np.sin(theta)
            rel_gy = -dx * np.sin(theta) + dy * np.cos(theta)
            rel_gx /= 12
            rel_gy /= 12

            # Heading as cos/sin
            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)

            # Last velocity
            lin_vel = act[0] * 2  # Assuming original range [-0.5, 0.5]
            ang_vel = (act[1] + 1) / 2  # Assuming original range [-1, 1]

            # Final state vector
            state = [x, y, heading_cos, heading_sin, distance[i]/17, cos[i], sin[i], lin_vel, ang_vel]

            assert len(state) == self.state_dim, f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            states.append(state)
            terminal.append(collision[i])

        return states, terminal
