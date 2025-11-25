import torch
import torch.nn as nn
import robot_nav.models.SAC.SAC_utils as utils

class MLPCNNDoubleQCritic(nn.Module):
    """
    Multi-Modal Double Q-Critic Network (Vector + CNN + Action) for SAC.
    """
    def __init__(self, vec_dim, action_dim, hidden_dim):
        super(MLPCNNDoubleQCritic, self).__init__()
        
        # ========== CNN Branch (128x128 Grid) ==========
        # Shared CNN structure or separate? Usually separate for Critic in SAC to be independent.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.ReLU()
        )
        
        # ========== Split MLP Branches ==========
        self.vel_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.goal_mlp = nn.Sequential(nn.Linear(3, 32), nn.Tanh())
        self.error_mlp = nn.Sequential(nn.Linear(4, 32), nn.Tanh())
        self.rps_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.cr_mlp = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        
        self.fusion_dim = 512 + 104
        
        # Q1 Architecture
        self.Q1 = nn.Sequential(
            nn.Linear(self.fusion_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 Architecture
        self.Q2 = nn.Sequential(
            nn.Linear(self.fusion_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(utils.weight_init)
        
    def _extract_features(self, vec, grid):
        if grid.dim() == 3: grid = grid.unsqueeze(1)
        if grid.dim() == 4 and grid.shape[1] != 1: grid = grid.permute(0, 3, 1, 2)

        vel_in = vec[:, 0:2]
        goal_in = torch.cat([vec[:, 2:4], vec[:, 6:7]], dim=1)
        error_in = torch.cat([vec[:, 4:6], vec[:, 8:10]], dim=1)
        rps_in = vec[:, 10:12]
        cr_in = vec[:, 7:8]
        
        vel_out = self.vel_mlp(vel_in)
        goal_out = self.goal_mlp(goal_in)
        error_out = self.error_mlp(error_in)
        rps_out = self.rps_mlp(rps_in)
        cr_out = self.cr_mlp(cr_in)
        
        cnn_out = self.cnn(grid)
        
        combined = torch.cat([cnn_out, vel_out, goal_out, error_out, rps_out, cr_out], dim=1)
        return combined

    def forward(self, vec, grid, action):
        features = self._extract_features(vec, grid)
        
        xu = torch.cat([features, action], dim=1)
        
        q1 = self.Q1(xu)
        q2 = self.Q2(xu)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_critic/{k}_hist", v, step)
