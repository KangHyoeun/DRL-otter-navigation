"""
CNN-based SAC Actor for LiDAR navigation
Based on CNNTD3 architecture but with stochastic policy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class CNNSACActor(nn.Module):
    """
    CNN-based Actor network for SAC.
    
    Architecture identical to CNNTD3 Actor but outputs mean and log_std
    for stochastic policy instead of deterministic action.
    
    Input: [LiDAR(360) + goal(3) + action(2) + vel(2) + rps(2)] = 369
    Output: [action_mean, action_log_std] for Gaussian policy
    """
    
    def __init__(self, action_dim, log_std_bounds=[-5, 2]):
        super(CNNSACActor, self).__init__()
        
        # CNN layers for LiDAR (same as CNNTD3)
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)
        
        # Embedding layers
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)
        self.vel_embed = nn.Linear(2, 10)
        self.rps_embed = nn.Linear(2, 10)
        
        # Fully connected layers
        self.fc1 = nn.Linear(76, 400)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        self.fc2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="leaky_relu")
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(300, action_dim)
        
        self.log_std_bounds = log_std_bounds
        self.action_dim = action_dim
        
    def forward(self, s):
        """
        Forward pass through the Actor network.
        
        Args:
            s (torch.Tensor): State [batch_size, 369]
                             [LiDAR(360) + distance + cos + sin + u_ref + r_ref + u + r + n1 + n2]
        
        Returns:
            torch.distributions.Normal: Gaussian distribution over actions
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
        s = torch.concat((l, g, a, v, r), dim=-1)  # [batch, 76]
        
        # Fully connected layers
        s = F.leaky_relu(self.fc1(s))
        s = F.leaky_relu(self.fc2(s))
        
        # Output mean and log_std
        mean = self.mean_layer(s)
        log_std = self.log_std_layer(s)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        
        # Create Gaussian distribution
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        return dist
