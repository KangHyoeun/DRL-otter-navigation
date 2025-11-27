from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
import robot_nav.models.SAC.SAC_utils as utils
from robot_nav.models.SAC.MLPCNNSAC_critic import MLPCNNDoubleQCritic
from robot_nav.models.SAC.MLPCNNSAC_actor import MLPCNNDiagGaussianActor
from torch.utils.tensorboard import SummaryWriter
from colregs_core.geometry import math_to_maritime_velocity
from robot_nav.utils import prepare_multi_modal_state

class MLPCNNSAC(object):
    """
    Multi-Modal SAC (Vector + CNN) implementation.
    Follows MLPCNNPPO's state preparation and structure.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        learnable_temperature=True,
        save_every=0,
        load_model=False,
        log_dist_and_hist=False,
        save_directory=Path("robot_nav/models/SAC/checkpoint"),
        model_name="MLPCNNSAC",
        load_directory=Path("robot_nav/models/SAC/checkpoint"),
        replay_buffer_capacity=100000, # Added
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.log_dist_and_hist = log_dist_and_hist

        # Running Mean Std for Vector State
        self.obs_rms = utils.RunningMeanStd(shape=(state_dim,))
        
        # Replay Buffer
        self.buffer = utils.MultiModalReplayBuffer(capacity=replay_buffer_capacity) # Use external capacity

        self.train_metrics_dict = {
            "critic_loss_av": [],
            "actor_loss_av": [],
            "q_value_mean_av": [],
            "q1_q2_diff_av": [],
            "alpha_av": [],
            "entropy_av": [],
            "alpha_loss_av": [],
            "action_std_av": [],
        }

        self.critic = MLPCNNDoubleQCritic(
            vec_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=256, # Slightly reduced from 400 for split MLP
        ).to(self.device)
        
        self.critic_target = MLPCNNDoubleQCritic(
            vec_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=256,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = MLPCNNDiagGaussianActor(
            vec_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            log_std_bounds=[-5, 2],
        ).to(self.device)

        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.critic_target.train()
        self.actor.train(True)
        self.critic.train(True)
        self.step = 0
        self.writer = SummaryWriter(comment=model_name)
        
        print(f"✅ MLPCNNSAC (Vector+Grid) Initialized")

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )
        # Save RMS stats
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count
        }, "%s/%s_rms.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename), weights_only=False)
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename), weights_only=False)
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename), weights_only=False)
        )
        # Load RMS stats if available
        try:
            rms_checkpoint = torch.load("%s/%s_rms.pth" % (directory, filename), weights_only=False)
            self.obs_rms.mean = rms_checkpoint['mean']
            self.obs_rms.var = rms_checkpoint['var']
            self.obs_rms.count = rms_checkpoint['count']
            print(f"Loaded weights and RMS stats from: {directory}")
        except FileNotFoundError:
            print(f"Loaded weights from: {directory} (RMS stats not found)")

    def train(self, replay_buffer, iterations, batch_size):
        for _ in range(iterations):
            self.update(
                replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
            )

        # ==========================================================================
        # Modified: Log requested metrics instead of iterating through train_metrics_dict
        # ==========================================================================
        self.step += 1 # Increment step once per train() call after all updates

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
            
        # Calculate averages for logging
        avg_critic_loss = mean(self.train_metrics_dict["critic_loss_av"]) if len(self.train_metrics_dict["critic_loss_av"]) > 0 else 0
        avg_actor_loss = mean(self.train_metrics_dict["actor_loss_av"]) if len(self.train_metrics_dict["actor_loss_av"]) > 0 else 0
        avg_q_value_mean = mean(self.train_metrics_dict["q_value_mean_av"]) if len(self.train_metrics_dict["q_value_mean_av"]) > 0 else 0
        avg_q1_q2_diff = mean(self.train_metrics_dict["q1_q2_diff_av"]) if len(self.train_metrics_dict["q1_q2_diff_av"]) > 0 else 0
        avg_alpha = mean(self.train_metrics_dict["alpha_av"]) if len(self.train_metrics_dict["alpha_av"]) > 0 else 0
        avg_entropy = mean(self.train_metrics_dict["entropy_av"]) if len(self.train_metrics_dict["entropy_av"]) > 0 else 0
        avg_alpha_loss = mean(self.train_metrics_dict["alpha_loss_av"]) if len(self.train_metrics_dict["alpha_loss_av"]) > 0 else 0
        avg_action_std = mean(self.train_metrics_dict["action_std_av"]) if len(self.train_metrics_dict["action_std_av"]) > 0 else 0

        # Log to TensorBoard
        self.writer.add_scalar("train/critic_loss", avg_critic_loss, self.step)
        self.writer.add_scalar("train/actor_loss", avg_actor_loss, self.step)
        self.writer.add_scalar("train/q_value_mean", avg_q_value_mean, self.step)
        self.writer.add_scalar("train/q1_q2_diff", avg_q1_q2_diff, self.step)
        self.writer.add_scalar("train/alpha", avg_alpha, self.step)
        self.writer.add_scalar("train/entropy", avg_entropy, self.step)
        self.writer.add_scalar("train/alpha_loss", avg_alpha_loss, self.step)
        self.writer.add_scalar("train/action_std", avg_action_std, self.step)
        
        # Clear metrics for next training step
        for key in self.train_metrics_dict.keys():
            self.train_metrics_dict[key] = []

        if self.step % 10 == 0:
            print(f"Iter {self.step} | CriticL: {avg_critic_loss:.4f} | ActorL: {avg_actor_loss:.4f} | Q_Mean: {avg_q_value_mean:.2f} | Alpha: {avg_alpha:.4f} | Entropy: {avg_entropy:.4f} | Std: {avg_action_std:.4f}")

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state, add_noise=True, update_rms=True):
        """
        Expects state to be a tuple: (vector_state, grid_map)
        Returns: action (np.array), log_prob (None for SAC inference), val (None)
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

            grid_tensor = torch.FloatTensor(grid_raw).to(self.device)

            if update_rms and add_noise:
                grid_tensor += torch.randn_like(grid_tensor) * 0.01
                grid_tensor = torch.clamp(grid_tensor, 0, 1)

            if grid_tensor.dim() == 2:
                grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0) 
            elif grid_tensor.dim() == 3:
                grid_tensor = grid_tensor.unsqueeze(0)

            dist = self.actor(vec_tensor, grid_tensor)
            
            if add_noise:
                action = dist.sample()
            else:
                action = dist.mean
                
            action = action.clamp(*self.action_range)
             
        return action.cpu().numpy().flatten(), None, None


    def update_critic(self, vec, grid, action, reward, next_vec, next_grid, done, step):
        with torch.no_grad():
            dist = self.actor(next_vec, next_grid)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_vec, next_grid, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + ((1 - done) * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(vec, grid, action)
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(
            current_Q2, target_Q
        )
        self.train_metrics_dict["critic_loss_av"].append(critic_loss.item())

        # Calculate Q-value mean and diff
        q_value_mean = torch.mean(0.5 * (current_Q1 + current_Q2)).item()
        q1_q2_diff = torch.mean(torch.abs(current_Q1 - current_Q2)).item()
        self.train_metrics_dict["q_value_mean_av"].append(q_value_mean)
        self.train_metrics_dict["q1_q2_diff_av"].append(q1_q2_diff)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        if self.log_dist_and_hist:
            self.critic.log(self.writer, step)

    def update_actor_and_alpha(self, vec, grid, step):
        dist = self.actor(vec, grid)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(vec, grid, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        
        self.train_metrics_dict["actor_loss_av"].append(actor_loss.item())
        self.train_metrics_dict["entropy_av"].append(
            -log_prob.mean().item()
        )
        self.train_metrics_dict["action_std_av"].append( # Accumulate action_std
            dist.scale.mean().item()
        )

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        if self.log_dist_and_hist:
            self.actor.log(self.writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            self.train_metrics_dict["alpha_loss_av"].append(alpha_loss.item())
            self.train_metrics_dict["alpha_av"].append(self.alpha.item())
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size):
        (
            vec_states, grid_states,
            actions,
            rewards,
            next_vec_states, next_grid_states,
            dones
        ) = replay_buffer.sample_batch(batch_size)
        
        # Normalize Vector States using stored RMS
        # Note: We use the CURRENT RMS statistics to normalize the batch.
        # This is standard practice in SB3 (normalize during sampling or env step).
        # Since we stored raw states in buffer, we normalize here.
        vec_mean = torch.FloatTensor(self.obs_rms.mean).to(self.device)
        vec_var = torch.FloatTensor(self.obs_rms.var).to(self.device)
        
        vec_states = torch.FloatTensor(vec_states).to(self.device)
        vec_states = (vec_states - vec_mean) / torch.sqrt(vec_var + 1e-8)
        
        next_vec_states = torch.FloatTensor(next_vec_states).to(self.device)
        next_vec_states = (next_vec_states - vec_mean) / torch.sqrt(vec_var + 1e-8)
        
        # Process Grid
        grid_states = torch.FloatTensor(grid_states).to(self.device)
        if grid_states.max() > 1.0: grid_states /= 255.0
        if grid_states.dim() == 3: grid_states = grid_states.unsqueeze(1)
            
        next_grid_states = torch.FloatTensor(next_grid_states).to(self.device)
        if next_grid_states.max() > 1.0: next_grid_states /= 255.0
        if next_grid_states.dim() == 3: next_grid_states = next_grid_states.unsqueeze(1)
        
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        self.update_critic(vec_states, grid_states, actions, rewards, next_vec_states, next_grid_states, dones, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(vec_states, grid_states, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
        """
        Uses common state preparation from utils.
        """
        return prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map
        )
