"""
CNN-based SAC Critic for LiDAR navigation
Based on CNNTD3 architecture with double Q-learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSACCritic(nn.Module):
    """
    CNN-based Critic network for SAC with double Q-learning.
    
    Architecture identical to CNNTD3 Critic.
    Two separate Q-networks (Q1, Q2) for double Q-learning.
    
    Input: [State(369) + Action(2)]
    Output: [Q1, Q2] - two Q-value estimates
    """
    
    def __init__(self, action_dim):
        super(CNNSACCritic, self).__init__()
        
        # Shared CNN layers for LiDAR
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)
        
        # Shared embedding layers
        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)
        self.vel_embed = nn.Linear(2, 10)
        self.rps_embed = nn.Linear(2, 10)
        
        # Q1 network
        self.q1_fc1 = nn.Linear(76, 400)
        torch.nn.init.kaiming_uniform_(self.q1_fc1.weight, nonlinearity="leaky_relu")
        self.q1_fc2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q1_fc2_s.weight, nonlinearity="leaky_relu")
        self.q1_fc2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q1_fc2_a.weight, nonlinearity="leaky_relu")
        self.q1_fc3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q1_fc3.weight, nonlinearity="leaky_relu")
        
        # Q2 network
        self.q2_fc1 = nn.Linear(76, 400)
        torch.nn.init.kaiming_uniform_(self.q2_fc1.weight, nonlinearity="leaky_relu")
        self.q2_fc2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.q2_fc2_s.weight, nonlinearity="leaky_relu")
        self.q2_fc2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.q2_fc2_a.weight, nonlinearity="leaky_relu")
        self.q2_fc3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.q2_fc3.weight, nonlinearity="leaky_relu")
    
    def forward(self, s, action):
        """
        Forward pass through both Q-networks.
        
        Args:
            s (torch.Tensor): State [batch_size, 369]
            action (torch.Tensor): Action [batch_size, action_dim]
        
        Returns:
            (tuple): (q1, q2) - two Q-value estimates
        """
        # Parse state
        laser = s[:, :-9]      # LiDAR: 360
        goal = s[:, -9:-6]     # distance, cos, sin: 3
        act = s[:, -6:-4]      # u_ref, r_ref: 2
        vel = s[:, -4:-2]      # u_actual, r_actual: 2
        rps = s[:, -2:]        # n1, n2: 2
        
        # CNN processing (shared)
        laser = laser.unsqueeze(1)  # [batch, 1, 360]
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)
        
        # Embed other features (shared)
        g = F.leaky_relu(self.goal_embed(goal))
        a = F.leaky_relu(self.action_embed(act))
        v = F.leaky_relu(self.vel_embed(vel))
        r = F.leaky_relu(self.rps_embed(rps))
        
        # Concatenate all features
        s = torch.concat((l, g, a, v, r), dim=-1)  # [batch, 76]
        
        # Q1 network
        s1 = F.leaky_relu(self.q1_fc1(s))
        s1 = F.leaky_relu(self.q1_fc2_s(s1) + self.q1_fc2_a(action))
        q1 = self.q1_fc3(s1)
        
        # Q2 network
        s2 = F.leaky_relu(self.q2_fc1(s))
        s2 = F.leaky_relu(self.q2_fc2_s(s2) + self.q2_fc2_a(action))
        q2 = self.q2_fc3(s2)
        
        return q1, q2
