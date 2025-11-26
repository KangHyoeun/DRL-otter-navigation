import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from colregs_core.geometry import math_to_maritime_velocity
from robot_nav.models.SAC.SAC_utils import MultiModalReplayBuffer, RunningMeanStd

# Initialize weights (Orthogonal)
def init_weights(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class MLPCNNDDPGActor(nn.Module):
    """
    Multi-Modal Actor for DDPG.
    """
    def __init__(self, vec_dim, action_dim, max_action):
        super(MLPCNNDDPGActor, self).__init__()
        self.max_action = max_action
        
        # ========== CNN Branch (128x128 Grid) ==========
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
        
        # Fusion
        self.fusion_dim = 512 + 104
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.Tanh()
        )
        
        # Actor Head
        self.head = nn.Linear(512, action_dim)
        
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.head, gain=0.1)

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
        return self.shared_fc(combined)

    def forward(self, vec, grid):
        features = self._extract_features(vec, grid)
        return self.max_action * torch.tanh(self.head(features))


class MLPCNNDDPGCritic(nn.Module):
    """
    Multi-Modal Single Critic for DDPG.
    """
    def __init__(self, vec_dim, action_dim):
        super(MLPCNNDDPGCritic, self).__init__()
        
        # ========== CNN Branch ==========
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
        
        # Q Architecture (Single Head)
        self.Q = nn.Sequential(
            nn.Linear(self.fusion_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

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
        return self.Q(xu)


class MLPCNNDDPG:
    """
    Multi-Modal DDPG Agent.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        lr=1e-4,
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/DDPG/checkpoint"),
        model_name="MLPCNNDDPG",
        load_directory=Path("robot_nav/models/DDPG/checkpoint"),
    ):
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        # Actor
        self.actor = MLPCNNDDPGActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = MLPCNNDDPGActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = MLPCNNDDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = MLPCNNDDPGCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Buffer
        self.buffer = MultiModalReplayBuffer(capacity=int(1e5))
        
        # Normalization
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        
        # Logging
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(comment=model_name)
        
        if load_model:
            self.load(model_name, load_directory)
            
        print(f"✅ MLPCNNDDPG Initialized")

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
        # Velocities
        psi_math_deg = np.degrees(robot_state[2, 0])
        speed = np.linalg.norm([robot_state[3, 0], robot_state[4, 0]])
        v_x, v_y = math_to_maritime_velocity(psi_math_deg, speed)
        
        u_ref, u_actual = action[0], robot_state[3, 0]
        u_e = u_ref - u_actual
        r_ref, r_actual = action[1], robot_state[5, 0]
        r_e = r_ref - r_actual
        n1, n2 = robot_state[6, 0], robot_state[7, 0]

        vector_state = [v_x, v_y, distance, y_e, psi_e, chi_e, phi_tilde, CR_max, u_e, r_e, n1, n2]
        
        if grid_map is None:
            grid_map = np.zeros((128, 128), dtype=np.float32)
            
        terminal = 1 if collision or goal else 0
        return (vector_state, grid_map), terminal

    def get_action(self, state, add_noise=True, update_rms=True):
        vec_raw, grid_raw = state
        
        with torch.no_grad():
            vec_np = np.array(vec_raw, dtype=np.float32)
            if update_rms:
                self.obs_rms.update(vec_np.reshape(1, -1))
            vec_norm = (vec_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            vec_tensor = torch.FloatTensor(vec_norm).to(self.device).unsqueeze(0)
            
            grid_tensor = torch.FloatTensor(grid_raw).to(self.device)
            if grid_tensor.dim() == 2: grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)
            elif grid_tensor.dim() == 3: grid_tensor = grid_tensor.unsqueeze(0)
            
            # Grid is already normalized (Continuous Collision Risk Grid)
            # if grid_tensor.max() > 1.0: grid_tensor /= 255.0
            
            action = self.actor(vec_tensor, grid_tensor).cpu().numpy().flatten()
            
            if add_noise:
                noise = np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
                action = (action + noise).clip(-self.max_action, self.max_action)
                
        return action, None, None

    def train(self, replay_buffer, iterations, batch_size=256):
        total_critic_loss = 0
        total_actor_loss = 0
        
        for it in range(iterations):
            # Sample
            vec, grid, action, reward, next_vec, next_grid, done = replay_buffer.sample_batch(batch_size)
            
            # Normalize Vector
            vec = (vec - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            next_vec = (next_vec - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            
            # Normalize Grid
            # Grid is already normalized (Continuous Collision Risk Grid)
            # if grid.max() > 1.0: grid /= 255.0
            # if next_grid.max() > 1.0: next_grid /= 255.0
            
            # To Tensor
            state_vec = torch.FloatTensor(vec).to(self.device)
            state_grid = torch.FloatTensor(grid).to(self.device).unsqueeze(1)
            next_state_vec = torch.FloatTensor(next_vec).to(self.device)
            next_state_grid = torch.FloatTensor(next_grid).to(self.device).unsqueeze(1)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)
            
            # Critic Update
            with torch.no_grad():
                next_action = self.actor_target(next_state_vec, next_state_grid)
                target_Q = self.critic_target(next_state_vec, next_state_grid, next_action)
                target_Q = reward + (1 - done) * self.discount * target_Q

            current_Q = self.critic(state_vec, state_grid, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()
            
            # Actor Update
            actor_loss = -self.critic(state_vec, state_grid, self.actor(state_vec, state_grid)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            
            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.iter_count += 1
        self.writer.add_scalar("train/critic_loss", total_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/actor_loss", total_actor_loss / iterations, self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(self.model_name, self.save_directory)

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count
        }, f"{directory}/{filename}_rms.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        rms_data = torch.load(f"{directory}/{filename}_rms.pth")
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
        self.obs_rms.count = rms_data['count']
        print(f"Loaded weights and RMS from {directory}")
