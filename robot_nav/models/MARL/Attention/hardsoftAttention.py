import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Multi-robot attention mechanism for learning hard and soft attentions.

    This module provides both hard (binary) and soft (weighted) attention,
    combining feature encoding, relative pose and goal geometry, and
    message passing between agents.

    Args:
        embedding_dim (int): Dimension of the agent embedding vector.

    Attributes:
        embedding1 (nn.Linear): First layer for agent feature encoding.
        embedding2 (nn.Linear): Second layer for agent feature encoding.
        hard_mlp (nn.Sequential): MLP to process concatenated agent and edge features.
        hard_encoding (nn.Linear): Outputs logits for hard (binary) attention.
        q, k, v (nn.Linear): Layers for query, key, value projections for soft attention.
        attn_score_layer (nn.Sequential): Computes unnormalized attention scores for each pair.
        decode_1, decode_2 (nn.Linear): Decoding layers to produce the final attended embedding.
    """

    def __init__(self, embedding_dim):
        """
        Initialize attention mechanism for multi-agent communication.

        Args:
            embedding_dim (int): Output embedding dimension per agent.
        """
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding1 = nn.Linear(5, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")

        # Hard attention MLP with distance
        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Soft attention projections
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(10, embedding_dim, bias=False)
        self.v = nn.Linear(10, embedding_dim)

        # Soft attention score network (with polar other robot goal position)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # Decoder
        self.decode_1 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_1.weight, nonlinearity="leaky_relu")
        self.decode_2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_2.weight, nonlinearity="leaky_relu")

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
        """
        Forward pass: computes both hard and soft attentions among agents,
        produces the attended embedding for each agent, as well as diagnostic info.

        Args:
            embedding (Tensor): Input tensor of shape (B, N, D), where D is at least 11.

        Returns:
            tuple:
                att_embedding (Tensor): Final attended embedding, shape (B*N, 2*embedding_dim).
                hard_logits (Tensor): Logits for hard attention, (B*N, N-1).
                unnorm_rel_dist (Tensor): Pairwise distances between agents (not normalized), (B*N, N-1, 1).
                mean_entropy (Tensor): Mean entropy of soft attention distributions.
                hard_weights (Tensor): Binary hard attention mask, (B, N, N-1).
                comb_w (Tensor): Final combined attention weights, (N, N*(N-1)).
        """
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

        unnorm_rel_dist = torch.linalg.vector_norm(rel_vec, dim=-1, keepdim=True)
        unnorm_rel_dist = unnorm_rel_dist[:, mask].reshape(
            batch_size * n_agents, n_agents - 1, 1
        )

        # ---- Soft attention computation ----
        q = self.q(agent_embed)

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
        for i in range(n_agents):
            q_i = q[:, i : i + 1, :]
            mask = torch.ones(n_agents, dtype=torch.bool, device=edge_features.device)
            mask[i] = False
            edge_i_wo_self = soft_edge_features[:, i, mask, :]
            edge_i_wo_self = edge_i_wo_self.squeeze(1)
            k = F.leaky_relu(self.k(edge_i_wo_self))

            q_i_expanded = q_i.expand(-1, n_agents - 1, -1)
            attention_input = torch.cat([q_i_expanded, k], dim=-1)

            # Score computation (per pair)
            scores = self.attn_score_layer(attention_input).transpose(1, 2)

            # Mask using hard attention
            h_weights = hard_weights[:, i].unsqueeze(1)
            mask = (h_weights > 0.5).float()

            # All-zero mask handling
            epsilon = 1e-6
            mask_sum = mask.sum(dim=-1, keepdim=True)
            all_zero_mask = mask_sum < epsilon

            # Apply mask to scores
            masked_scores = scores.masked_fill(mask == 0, float("-inf"))

            # Compute softmax, safely handle all-zero cases
            soft_weights = F.softmax(masked_scores, dim=-1)
            soft_weights = torch.where(
                all_zero_mask, torch.zeros_like(soft_weights), soft_weights
            )

            combined_weights = soft_weights * mask  # (B, 1, N-1)
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

            v_j = F.leaky_relu(self.v(edge_i_wo_self))
            attn_output = torch.bmm(combined_weights, v_j).squeeze(1)
            attention_outputs.append(attn_output)

        comb_w = torch.stack(combined_w, dim=1).reshape(n_agents, -1)
        attn_stack = torch.stack(attention_outputs, dim=1).reshape(
            -1, self.embedding_dim
        )
        self_embed = agent_embed.reshape(-1, self.embedding_dim)
        concat_embed = torch.cat([self_embed, attn_stack], dim=-1)

        x = F.leaky_relu(self.decode_1(concat_embed))
        att_embedding = F.leaky_relu(self.decode_2(x))

        mean_entropy = torch.stack(entropy_list).mean()

        return (
            att_embedding,
            hard_logits[..., 1],
            unnorm_rel_dist,
            mean_entropy,
            hard_weights,
            comb_w,
        )
