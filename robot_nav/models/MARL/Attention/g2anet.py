import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class G2ANet(nn.Module):
    def __init__(self, embedding_dim):
        super(G2ANet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Soft
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(10, embedding_dim, bias=False)
        self.v = nn.Linear(10, embedding_dim)

        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.decoding = nn.Linear(embedding_dim*2, embedding_dim*2)

        self.embedding1 = nn.Linear(5, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")

    def encode_agent_features(self, embed):
        """
        Encode agent features using a small MLP.

        Args:
            embed (Tensor): Input features (B*N, 5).

        Returns:
            Tensor: Encoded embedding (B*N, embedding_dim).
        """
        embed = F.leaky_relu(self.embedding1(embed))
        embed = F.leaky_relu(self.embedding2(embed))
        return embed

    def forward(self, embedding):
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        batch_size, n_agents, _ = embedding.shape

        # Extract sub-features
        embed = embedding[:, :, 4:9].reshape(batch_size * n_agents, -1)
        position = embedding[:, :, :2].reshape(batch_size, n_agents, 2)
        heading = embedding[:, :, 2:4].reshape(
            batch_size, n_agents, 2
        )  # assume (cos(θ), sin(θ))
        action = embedding[:, :, 7:9].reshape(batch_size, n_agents, 2)
        goal = embedding[:, :, -2:].reshape(batch_size, n_agents, 2)

        # Compute pairwise relative goal vectors (for each i,j)
        goal_j = goal.unsqueeze(1).expand(-1, n_agents, -1, -1)
        pos_i = position.unsqueeze(2)
        goal_rel_vec = goal_j - pos_i

        # Encode agent features
        agent_embed = self.encode_agent_features(embed)
        agent_embed = agent_embed.view(batch_size, n_agents, self.embedding_dim)

        # Prep for hard attention: compute all relative geometry for each agent pair
        h_i = agent_embed.unsqueeze(2)  # (B, N, 1, D)
        pos_i = position.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = position.unsqueeze(1)  # (B, 1, N, 2)
        heading_i = heading.unsqueeze(2)  # (B, N, 1, 2)
        heading_j = heading.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (B, 1, N, 2)

        # Compute relative vectors and distance
        rel_vec = pos_j - pos_i  # (B, N, N, 2)
        dx, dy = rel_vec[..., 0], rel_vec[..., 1]
        rel_dist = (
                torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True) / 12
        )  # (B, N, N, 1)

        # Relative angle in agent i's frame
        angle = torch.atan2(dy, dx) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_sin = torch.sin(angle)
        rel_angle_cos = torch.cos(angle)

        # Other agent's heading
        heading_j_cos = heading_j[..., 0]  # (B, 1, N)
        heading_j_sin = heading_j[..., 1]  # (B, 1, N)

        # Edge features for hard attention
        edge_features = torch.cat(
            [
                rel_dist,  # (B, N, N, 1)
                rel_angle_cos.unsqueeze(-1),  # (B, N, N, 1)
                rel_angle_sin.unsqueeze(-1),  # (B, N, N, 1)
                heading_j_cos.unsqueeze(-1),  # (B, N, N, 1)
                heading_j_sin.unsqueeze(-1),  # (B, N, N, 1)
                action.unsqueeze(1).expand(-1, n_agents, -1, -1),  # (B, N, N, 2)
            ],
            dim=-1,
        )

        # Broadcast agent embedding for all pairs (except self-pairs)
        h_i_expanded = h_i.expand(-1, -1, n_agents, -1)

        # Remove self-pairs using mask
        mask = ~torch.eye(n_agents, dtype=torch.bool, device=embedding.device)
        h_i_flat = h_i_expanded[:, mask].reshape(
            batch_size * n_agents, n_agents - 1, self.embedding_dim
        )
        edge_flat = edge_features[:, mask].reshape(
            batch_size * n_agents, n_agents - 1, -1
        )

        # Concatenate agent embedding and edge features
        hard_input = torch.cat([h_i_flat, edge_flat], dim=-1)

        # Hard attention forward
        h_hard = self.hard_mlp(hard_input)
        hard_logits = self.hard_encoding(h_hard)
        hard_weights = F.gumbel_softmax(hard_logits, hard=False, tau=0.5, dim=-1)[
            ..., 1
        ].unsqueeze(2)
        hard_weights = hard_weights.view(batch_size, n_agents, n_agents - 1)
        # hard_weights = hard_weights.permute(1, 0, 2)

        unnorm_rel_dist = torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True)
        unnorm_rel_dist = unnorm_rel_dist[:, mask].reshape(
            batch_size * n_agents, n_agents - 1, 1
        )

        # ---- Soft attention computation ----
        attention_outputs = []
        entropy_list = []
        combined_w = []

        # Goal-relative polar features for soft attention
        goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
        goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
        heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
        goal_rel_angle = goal_angle_global - heading_angle
        goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
        goal_rel_angle_cos = torch.cos(goal_rel_angle).unsqueeze(-1)
        goal_rel_angle_sin = torch.sin(goal_rel_angle).unsqueeze(-1)
        goal_polar = torch.cat(
            [goal_rel_dist, goal_rel_angle_cos, goal_rel_angle_sin], dim=-1
        )

        # Soft attention edge features (include goal polar)
        soft_edge_features = torch.cat([edge_features, goal_polar], dim=-1)

        q = self.q(agent_embed)
        epsilon = 1e-6



        # q = self.q(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(soft_edge_features)
        v = F.relu(self.v(soft_edge_features))
        x = []
        for i in range(n_agents):
            q_i = q[:, i].view(-1, 1, self.embedding_dim)
            k_b = k[:,i,:,:]
            k_i = [k_b[:, j] for j in range(n_agents) if j != i]  # 对于agent i来说，其他agent的k
            v_b = v[:,i,:,:]
            v_i = [v_b[:, j] for j in range(n_agents) if j != i]  # 对于agent i来说，其他agent的v

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, args.attention_dim， n_agents - 1)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim，n_agents - 1) = (batch_size, 1，n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # 归一化
            scaled_score = score / np.sqrt(self.embedding_dim)

            # softmax得到权重
            soft_weight = F.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)

            # 加权求和，注意三个矩阵的最后一维是n_agents - 1维度，得到(batch_size, args.attention_dim)
            x_i = (v_i * soft_weight * hard_weights[:, i, :].unsqueeze(1)).sum(dim=-1)
            x.append(x_i)

            combined_weights = soft_weight * hard_weights[:, i, :].unsqueeze(1)  # (B, 1, N-1)
            combined_w.append(combined_weights)

            # Normalize for entropy calculation
            combined_weights_norm = combined_weights / (
                    combined_weights.sum(dim=-1, keepdim=True) + epsilon
            )

            # Entropy for analysis/logging
            entropy = (
                -(combined_weights_norm * (combined_weights_norm + epsilon).log())
                .sum(dim=-1)
                .mean()
            )
            entropy_list.append(entropy)

        # 合并每个agent的h与x
        x = torch.stack(x, dim=1).reshape(-1, self.embedding_dim)  # (batch_size * n_agents, args.attention_dim)
        self_embed = agent_embed.reshape(-1, self.embedding_dim)
        final_input = torch.cat([self_embed, x], dim=-1)
        output = self.decoding(final_input)
        mean_entropy = torch.stack(entropy_list).mean()
        comb_w = torch.stack(combined_w, dim=1).reshape(n_agents, -1)

        return output, hard_logits[..., 1], unnorm_rel_dist, mean_entropy, hard_weights, comb_w