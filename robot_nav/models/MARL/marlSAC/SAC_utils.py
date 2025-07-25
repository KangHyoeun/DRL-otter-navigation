import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import os
from collections import deque
import random
import math


def soft_update_params(net, target_net, tau):
    """
    Perform a soft update of the parameters of the target network.

    Args:
        net (nn.Module): Source network whose parameters are used for updating.
        target_net (nn.Module): Target network to be updated.
        tau (float): Interpolation parameter (0 < tau < 1) for soft updates.
                     A value closer to 1 means faster updates.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    """
    Set random seed for reproducibility across NumPy, random, and PyTorch.

    Args:
        seed (int): Random seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    """
    Create a directory if it does not exist.

    Args:
        *path_parts (str): Components of the path to be joined into the directory.

    Returns:
        str: The full path of the created or existing directory.
    """
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """
    Custom weight initialization for layers.

    Applies orthogonal initialization to Linear layers and zero initialization to biases.

    Args:
        m (nn.Module): Layer to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) with configurable depth and optional output activation.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in each hidden layer.
        output_dim (int): Number of output features.
        hidden_depth (int): Number of hidden layers.
        output_mod (nn.Module, optional): Optional output activation module (e.g., Tanh, Sigmoid).
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """
    Create an MLP as a `nn.Sequential` module.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer size.
        output_dim (int): Output feature dimension.
        hidden_depth (int): Number of hidden layers.
        output_mod (nn.Module, optional): Output activation module.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
