import torch
import torch.nn as nn
from torch.distributions import Normal
import robot_nav.models.SAC.SAC_utils as utils
from robot_nav.models.SAC.SAC_actor import SquashedNormal

class MLPCNNDiagGaussianActor(nn.Module):
    """
    Multi-Modal Actor Network (Vector + CNN) for SAC.
    
    Architecture:
        - Vector Branch: Split MLPs (Vel, Goal, Error, RPS, CR)
        - CNN Branch: Nature-CNN style (128x128 input)
        - Fusion: Concat -> Shared FC -> Mu, LogStd
    """
    def __init__(self, vec_dim, action_dim, hidden_dim, log_std_bounds):
        super(MLPCNNDiagGaussianActor, self).__init__()
        
        self.log_std_bounds = log_std_bounds
        
        # ========== CNN Branch (128x128 Grid) ==========
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), # 128->31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 31->14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 14->12
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
        
        # MLP Output Dim: 16 + 32 + 32 + 16 + 8 = 104
        self.fusion_dim = 512 + 104
        
        # ========== Fusion & Heads ==========
        self.trunk = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim) # Outputs Mu and LogStd
        )
        
        self.outputs = dict()
        self.apply(utils.weight_init)
        
    def _extract_features(self, vec, grid):
        # Ensure correct dimensions
        if grid.dim() == 3: grid = grid.unsqueeze(1) # (B, H, W) -> (B, 1, H, W)
        if grid.dim() == 4 and grid.shape[1] != 1: grid = grid.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W) assuming C=1

        # Split Vector Features
        vel_in = vec[:, 0:2]
        goal_in = torch.cat([vec[:, 2:4], vec[:, 6:7]], dim=1) # dist, y_e, phi_tilde
        error_in = torch.cat([vec[:, 4:6], vec[:, 8:10]], dim=1) # psi_e, chi_e, u_e, r_e
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

    def forward(self, vec, grid):
        features = self._extract_features(vec, grid)
        mu, log_std = self.trunk(features).chunk(2, dim=-1)

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
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_actor/{k}_hist", v, step)
