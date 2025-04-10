import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import robot_nav.models.SAC.SAC_utils as utils


class TanhTransform(pyd.transforms.Transform):
    """
    A bijective transformation that applies the hyperbolic tangent function.

    This is used to squash the output of a normal distribution to be within [-1, 1],
    making it suitable for bounded continuous action spaces.

    Attributes:
        domain: The input domain (real numbers).
        codomain: The output codomain (interval between -1 and 1).
        bijective: Whether the transform is bijective (True).
        sign: The sign of the Jacobian determinant (positive).
    """

    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        """
        Initialize the TanhTransform.

        Args:
            cache_size (int): Size of the cache for storing intermediate values.
        """
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        """
        Inverse hyperbolic tangent function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: atanh(x)
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        """
        Equality check for the transform.

        Returns:
            bool: True if the other object is also a TanhTransform.
        """
        return isinstance(other, TanhTransform)

    def _call(self, x):
        """
        Forward transformation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: tanh(x)
        """
        return x.tanh()

    def _inverse(self, y):
        """
        Inverse transformation.

        Args:
            y (Tensor): Input tensor in [-1, 1].

        Returns:
            Tensor: atanh(y)
        """
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Log absolute determinant of the Jacobian of the transformation.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Output tensor.

        Returns:
            Tensor: log|det(Jacobian)|
        """
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    A squashed (tanh-transformed) diagonal Gaussian distribution.

    This is used for stochastic policies where actions must be within bounded intervals.
    """

    def __init__(self, loc, scale):
        """
        Initialize the squashed normal distribution.

        Args:
            loc (Tensor): Mean of the Gaussian.
            scale (Tensor): Standard deviation of the Gaussian.
        """
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        """
        Compute the mean of the transformed distribution.

        Returns:
            Tensor: Mean of the squashed distribution.
        """
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """
    Diagonal Gaussian policy network with tanh squashing.

    This network outputs a squashed Gaussian distribution given an observation,
    suitable for continuous control tasks.

    Args:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Number of units in hidden layers.
        hidden_depth (int): Number of hidden layers.
        log_std_bounds (list): Min and max bounds for log standard deviation.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        """
        Initialize the actor network.
        """
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        """
        Forward pass through the network.

        Args:
            obs (Tensor): Observation input.

        Returns:
            SquashedNormal: Action distribution with mean and std tracked in `self.outputs`.
        """
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, writer, step):
        """
        Log network outputs (mu and std histograms) to TensorBoard.

        Args:
            writer (SummaryWriter): TensorBoard writer instance.
            step (int): Current global training step.
        """
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_actor/{k}_hist", v, step)
