import torch
from torch import nn

import robot_nav.models.SAC.SAC_utils as utils


class QCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)

        self.outputs["q1"] = q1

        return q1

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_critic/{k}_hist", v, step)
