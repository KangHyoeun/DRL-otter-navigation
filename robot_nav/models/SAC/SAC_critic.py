import torch
from torch import nn

import robot_nav.models.SAC.SAC_utils as utils


class DoubleQCritic(nn.Module):
    """
    Double Q-learning critic network.

    Implements two independent Q-functions (Q1 and Q2) to mitigate overestimation bias in value estimates,
    as introduced in the Twin Delayed Deep Deterministic Policy Gradient (TD3) and Soft Actor-Critic (SAC) algorithms.

    Args:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Number of units in each hidden layer.
        hidden_depth (int): Number of hidden layers.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        """
        Initialize the Double Q-critic network with two MLPs.

        Q1 and Q2 share the same architecture but have separate weights.
        """
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        """
        Compute Q-values for the given observation-action pairs.

        Args:
            obs (Tensor): Observations of shape (batch_size, obs_dim).
            action (Tensor): Actions of shape (batch_size, action_dim).

        Returns:
            Tuple[Tensor, Tensor]: Q1 and Q2 values, each of shape (batch_size, 1).
        """
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, writer, step):
        """
        Log histograms of Q-value distributions to TensorBoard.

        Args:
            writer (SummaryWriter): TensorBoard writer instance.
            step (int): Current training step (global).
        """
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_critic/{k}_hist", v, step)
