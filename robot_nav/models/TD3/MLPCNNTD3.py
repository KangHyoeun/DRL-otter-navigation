import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from colregs_core.geometry import math_to_maritime_velocity
from robot_nav.models.SAC.SAC_utils import MultiModalReplayBuffer, RunningMeanStd
from robot_nav.utils import prepare_multi_modal_state

# Initialize weights (Orthogonal)
def init_weights(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class MLPCNNTD3Actor(nn.Module):
    """
    Multi-Modal Actor for TD3.
    """
    def __init__(self, vec_dim, action_dim, max_action):
        super(MLPCNNTD3Actor, self).__init__()
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


class MLPCNNTD3Critic(nn.Module):
    """
    Multi-Modal Double Critic for TD3.
    """
    def __init__(self, vec_dim, action_dim):
        super(MLPCNNTD3Critic, self).__init__()
        
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
        
        # Q1 Architecture
        self.Q1 = nn.Sequential(
            nn.Linear(self.fusion_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        
        # Q2 Architecture
        self.Q2 = nn.Sequential(
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
        return self.Q1(xu), self.Q2(xu)

    def Q1_only(self, vec, grid, action):
        features = self._extract_features(vec, grid)
        xu = torch.cat([features, action], dim=1)
        return self.Q1(xu)


class MLPCNNTD3:
    """
    Multi-Modal TD3 Agent.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        exploration_noise_init=0.1,
        exploration_noise_min=0.01,
        exploration_noise_decay=0.9995,
        use_lr_scheduler=False,
        lr_scheduler_type="cosine",
        lr_decay_epochs=1000,
        lr_min_factor=0.1,
        lr_decay_rate=0.99,
        lr_step_size=100,
        lr_gamma=0.5,
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/TD3/checkpoint"),
        model_name="MLPCNNTD3",
        load_directory=Path("robot_nav/models/TD3/checkpoint"),
        replay_buffer_capacity=100000,
    ):
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0
        self.epoch_count = 0
        
        # Exploration Noise (Decaying)
        self.exploration_noise_init = exploration_noise_init
        self.exploration_noise_min = exploration_noise_min
        self.exploration_noise_decay = exploration_noise_decay
        self.current_exploration_noise = exploration_noise_init
        
        # LR Scheduler Config
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_min_factor = lr_min_factor
        self.actor_lr_initial = actor_lr
        self.critic_lr_initial = critic_lr

        # Actor
        self.actor = MLPCNNTD3Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = MLPCNNTD3Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic
        self.critic = MLPCNNTD3Critic(state_dim, action_dim).to(device)
        self.critic_target = MLPCNNTD3Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # LR Schedulers
        if self.use_lr_scheduler:
            if lr_scheduler_type == "cosine":
                self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.actor_optimizer, T_max=lr_decay_epochs, eta_min=actor_lr * lr_min_factor
                )
                self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.critic_optimizer, T_max=lr_decay_epochs, eta_min=critic_lr * lr_min_factor
                )
            elif lr_scheduler_type == "linear":
                lambda_fn = lambda epoch: max(lr_min_factor, 1.0 - epoch / lr_decay_epochs)
                self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda_fn)
                self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda_fn)
            elif lr_scheduler_type == "exponential":
                self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=lr_decay_rate)
                self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=lr_decay_rate)
            elif lr_scheduler_type == "step":
                self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=lr_step_size, gamma=lr_gamma)
                self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=lr_step_size, gamma=lr_gamma)
            else:
                raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")
            print(f"✅ LR Scheduler Enabled: {lr_scheduler_type}")
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
        
        # Buffer
        self.buffer = MultiModalReplayBuffer(capacity=replay_buffer_capacity) # Use external capacity
        
        # Normalization
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        
        # Logging
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(comment=model_name)
        self.train_metrics_dict = {
            "critic_loss_av": [],
            "actor_loss_av": [],
            "q_value_mean_av": [],
            "q1_q2_diff_av": [],
            "action_std_av": [],
        }
        
        if load_model:
            self.load(model_name, load_directory)
            
        print(f"✅ MLPCNNTD3 Initialized")

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
        """
        Uses common state preparation from utils.
        """
        return prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map
        )

    def get_action(self, state, add_noise=True, update_rms=True):
        """
        - Vector Normalization (RMS)
        - Grid Noise Injection (Robustness 강화)
        - Action Noise (TD3 exploration strategy)
        """
        vec_raw, grid_raw = state
        
        with torch.no_grad():
            # 1. Normalize Vector State
            vec_np = np.array(vec_raw, dtype=np.float32)
            if update_rms:
                self.obs_rms.update(vec_np.reshape(1, -1))
            
            vec_norm = (vec_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            vec_tensor = torch.FloatTensor(vec_norm).to(self.device)
            
            if vec_tensor.dim() == 1:
                vec_tensor = vec_tensor.unsqueeze(0) # (1, state_dim)
            
            # 2. Process Grid State
            grid_tensor = torch.FloatTensor(grid_raw).to(self.device)

            if update_rms and add_noise:
                grid_tensor += torch.randn_like(grid_tensor) * 0.01
                grid_tensor = torch.clamp(grid_tensor, 0, 1)
            
            # Dimension Check (Ensure 1, 1, H, W)
            if grid_tensor.dim() == 2:
                grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0) 
            elif grid_tensor.dim() == 3:
                grid_tensor = grid_tensor.unsqueeze(0)

            # 3. Model Forward (Actor)
            # TD3 Actor returns action in [-max_action, max_action] directly (via tanh * max_action)
            action = self.actor(vec_tensor, grid_tensor).cpu().numpy().flatten()
            
            # 4. Add Exploration Noise (Decaying Gaussian)
            if add_noise:
                # Calculate current noise std with decay
                self.current_exploration_noise = max(
                    self.exploration_noise_min,
                    self.exploration_noise_init * (self.exploration_noise_decay ** self.iter_count)
                )
                noise = np.random.normal(0, self.max_action * self.current_exploration_noise, size=self.action_dim)
                action = (action + noise).clip(-self.max_action, self.max_action)
                
        return action, None, None

    def train(self, replay_buffer, iterations, batch_size=256):
        total_critic_loss = 0
        total_actor_loss = 0
        total_q_value_mean = 0
        total_q1_q2_diff = 0
        
        for it in range(iterations):
            # Sample
            vec, grid, action, reward, next_vec, next_grid, done = replay_buffer.sample_batch(batch_size)
            
            # Normalize Vector
            vec = (vec - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            next_vec = (next_vec - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            
            # To Tensor
            state_vec = torch.FloatTensor(vec).to(self.device)
            state_grid = torch.FloatTensor(grid).to(self.device).unsqueeze(1)
            next_state_vec = torch.FloatTensor(next_vec).to(self.device)
            next_state_grid = torch.FloatTensor(next_grid).to(self.device).unsqueeze(1)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)
            
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state_vec, next_state_grid) + noise).clamp(-self.max_action, self.max_action)
                
                target_Q1, target_Q2 = self.critic_target(next_state_vec, next_state_grid, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state_vec, state_grid, action)
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()
            total_q_value_mean += torch.mean(0.5 * (current_Q1 + current_Q2)).item() # Accumulate Q_mean
            total_q1_q2_diff += torch.mean(torch.abs(current_Q1 - current_Q2)).item() # Accumulate Q_diff
            
            if it % self.policy_freq == 0:
                actor_loss = -self.critic.Q1_only(state_vec, state_grid, self.actor(state_vec, state_grid)).mean()
                
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
        avg_critic_loss = total_critic_loss / iterations
        avg_actor_loss = total_actor_loss / (iterations / self.policy_freq) if iterations / self.policy_freq > 0 else 0
        avg_q_value_mean = total_q_value_mean / iterations
        avg_q1_q2_diff = total_q1_q2_diff / iterations

        self.writer.add_scalar("train/critic_loss", avg_critic_loss, self.iter_count)
        self.writer.add_scalar("train/actor_loss", avg_actor_loss, self.iter_count)
        self.writer.add_scalar("train/q_value_mean", avg_q_value_mean, self.iter_count)
        self.writer.add_scalar("train/q1_q2_diff", avg_q1_q2_diff, self.iter_count)
        self.writer.add_scalar("train/exploration_noise", self.current_exploration_noise, self.iter_count)
        
        # LR Scheduler Step (called per training iteration, not per epoch)
        # For epoch-based schedulers, this should be called in the outer training loop
        # Here we track it for logging purposes
        if self.use_lr_scheduler:
            current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            current_critic_lr = self.critic_optimizer.param_groups[0]['lr']
            self.writer.add_scalar("train/actor_lr", current_actor_lr, self.iter_count)
            self.writer.add_scalar("train/critic_lr", current_critic_lr, self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(self.model_name, self.save_directory)

        if self.iter_count % 10 == 0:
            print(f"Iter {self.iter_count} | CriticL: {avg_critic_loss:.4f} | ActorL: {avg_actor_loss:.4f} | Q_Mean: {avg_q_value_mean:.2f} | Q_Diff: {avg_q1_q2_diff:.2f} | Noise: {self.current_exploration_noise:.4f}")

    def step_epoch(self):
        """
        Call this at the end of each epoch to step the LR schedulers.
        Should be called from the training loop (e.g., in trainers/off_policy.py).
        """
        if self.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            self.epoch_count += 1
            print(f"📉 LR Updated (Epoch {self.epoch_count}): Actor={self.actor_optimizer.param_groups[0]['lr']:.6f}, Critic={self.critic_optimizer.param_groups[0]['lr']:.6f}")

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
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", weights_only=False))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", weights_only=False))
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        try:
            rms_data = torch.load(f"{directory}/{filename}_rms.pth", weights_only=False)
            self.obs_rms.mean = rms_data['mean']
            self.obs_rms.var = rms_data['var']
            self.obs_rms.count = rms_data['count']
            print(f"Loaded weights and RMS from {directory}")
        except FileNotFoundError:
            print(f"Loaded weights from {directory} (RMS not found)")
