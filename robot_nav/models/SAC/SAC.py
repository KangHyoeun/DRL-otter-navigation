from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
import robot_nav.models.SAC.SAC_utils as utils
from robot_nav.models.SAC.SAC_critic import DoubleQCritic as critic_model
from robot_nav.models.SAC.SAC_actor import DiagGaussianActor as actor_model
from torch.utils.tensorboard import SummaryWriter


class SAC(object):
    """
    Soft Actor-Critic (SAC) implementation.

    This class implements the SAC algorithm using a Gaussian policy actor and double Q-learning critic.
    It supports automatic entropy tuning, model saving/loading, and logging via TensorBoard.

    Args:
        state_dim (int): Dimension of the observation/state space.
        action_dim (int): Dimension of the action space.
        device (torch.device): PyTorch device (e.g., 'cpu' or 'cuda').
        max_action (float): Maximum magnitude of actions.
        discount (float): Discount factor for rewards.
        init_temperature (float): Initial entropy temperature.
        alpha_lr (float): Learning rate for entropy temperature alpha.
        alpha_betas (tuple): Adam optimizer betas for alpha.
        actor_lr (float): Learning rate for actor network.
        actor_betas (tuple): Adam optimizer betas for actor.
        actor_update_frequency (int): Frequency of actor updates.
        critic_lr (float): Learning rate for critic network.
        critic_betas (tuple): Adam optimizer betas for critic.
        critic_tau (float): Soft update parameter for critic target.
        critic_target_update_frequency (int): Frequency of critic target updates.
        learnable_temperature (bool): Whether alpha is learnable.
        save_every (int): Save model every N training steps. Set 0 to disable.
        load_model (bool): Whether to load model from disk at init.
        log_dist_and_hist (bool): Log distribution and histogram if True.
        save_directory (Path): Directory to save models.
        model_name (str): Name for model checkpoints.
        load_directory (Path): Directory to load model checkpoints from.
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
        model_name="SAC",
        load_directory=Path("robot_nav/models/SAC/checkpoint"),
    ):
        super().__init__()

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

        self.train_metrics_dict = {
            "train_critic/loss_av": [],
            "train_actor/loss_av": [],
            "train_actor/target_entropy_av": [],
            "train_actor/entropy_av": [],
            "train_alpha/loss_av": [],
            "train_alpha/value_av": [],
            "train/batch_reward_av": [],
        }

        self.critic = critic_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=400,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target = critic_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=400,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=400,
            hidden_depth=2,
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

    def save(self, filename, directory):
        """
        Save the actor, critic, and target critic models to the specified directory.

        Args:
            filename (str): Base name of the saved files.
            directory (Path): Directory where models are saved.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        """
        Load the actor, critic, and target critic models from the specified directory.

        Args:
            filename (str): Base name of the saved files.
            directory (Path): Directory where models are loaded from.
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def train(self, replay_buffer, iterations, batch_size):
        """
        Run multiple training updates using data from the replay buffer.

        Args:
            replay_buffer: Buffer from which to sample training data.
            iterations (int): Number of training iterations to run.
            batch_size (int): Batch size for each update.
        """
        for _ in range(iterations):
            self.update(
                replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
            )

        for key, value in self.train_metrics_dict.items():
            if len(value):
                self.writer.add_scalar(key, mean(value), self.step)
            self.train_metrics_dict[key] = []
        self.step += 1

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    @property
    def alpha(self):
        """
        Returns:
            torch.Tensor: Current value of the entropy temperature alpha.
        """
        return self.log_alpha.exp()

    def get_action(self, obs, add_noise):
        """
        Select an action given an observation.

        Args:
            obs (np.ndarray): Input observation.
            add_noise (bool): Whether to add exploration noise.

        Returns:
            np.ndarray: Action vector.
        """
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(self.action_range[0], self.action_range[1])
        else:
            return self.act(obs)

    def act(self, obs, sample=False):
        """
        Generate an action from the actor network.

        Args:
            obs (np.ndarray): Input observation.
            sample (bool): If True, sample from the policy; otherwise use the mean.

        Returns:
            np.ndarray: Action vector.
        """
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done, step):
        """
        Update the critic network based on a batch of transitions.

        Args:
            obs (torch.Tensor): Batch of current observations.
            action (torch.Tensor): Batch of actions taken.
            reward (torch.Tensor): Batch of received rewards.
            next_obs (torch.Tensor): Batch of next observations.
            done (torch.Tensor): Batch of done flags.
            step (int): Current training step (for logging).
        """
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ((1 - done) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.train_metrics_dict["train_critic/loss_av"].append(critic_loss.item())
        self.writer.add_scalar("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.log_dist_and_hist:
            self.critic.log(self.writer, step)

    def update_actor_and_alpha(self, obs, step):
        """
        Update the actor and optionally the entropy temperature.

        Args:
            obs (torch.Tensor): Batch of observations.
            step (int): Current training step (for logging).
        """
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        self.train_metrics_dict["train_actor/loss_av"].append(actor_loss.item())
        self.train_metrics_dict["train_actor/target_entropy_av"].append(
            self.target_entropy
        )
        self.train_metrics_dict["train_actor/entropy_av"].append(
            -log_prob.mean().item()
        )
        self.writer.add_scalar("train_actor/loss", actor_loss, step)
        self.writer.add_scalar("train_actor/target_entropy", self.target_entropy, step)
        self.writer.add_scalar("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.log_dist_and_hist:
            self.actor.log(self.writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            self.train_metrics_dict["train_alpha/loss_av"].append(alpha_loss.item())
            self.train_metrics_dict["train_alpha/value_av"].append(self.alpha.item())
            self.writer.add_scalar("train_alpha/loss", alpha_loss, step)
            self.writer.add_scalar("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size):
        """
        Perform a full update step (critic, actor, alpha, target critic).

        Args:
            replay_buffer: Buffer to sample from.
            step (int): Current training step.
            batch_size (int): Size of sample batch.
        """
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
        ) = replay_buffer.sample_batch(batch_size)

        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
        done = torch.Tensor(batch_dones).to(self.device).reshape(-1, 1)
        self.train_metrics_dict["train/batch_reward_av"].append(
            batch_rewards.mean().item()
        )
        self.writer.add_scalar("train/batch_reward", batch_rewards.mean(), step)

        self.update_critic(state, action, reward, next_state, done, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action, robot_state):
        """
        Convert raw sensor input into a normalized state vector.

        Args:
            latest_scan (list or np.ndarray): Laser scan distances.
            distance (float): Distance to goal.
            cos (float): Cosine of heading angle to goal.
            sin (float): Sine of heading angle to goal.
            collision (bool): Whether the robot has collided.
            goal (bool): Whether the goal has been reached.
            action (list): Last action taken [linear_vel, angular_vel].
            robot_state (list): Robot state [x, y, psi, u, v, r, n1, n2].
        Returns:
            tuple: (state vector as list, terminal flag as int)
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 100.0

        max_bins = self.state_dim - 9
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 100.0)

        distance_min, distance_max = 0, 20
        distance_norm = normalize_state(distance, distance_min, distance_max)

        n_min, n_max = -101.7, 103.9
        n1, n2 = robot_state[6, 0], robot_state[7, 0]
        n1_norm = normalize_state(n1, n_min, n_max)
        n2_norm = normalize_state(n2, n_min, n_max)

        u_min, u_max = 0, 3.0
        u_ref, u_actual = action[0], robot_state[3, 0]
        u_ref_norm = normalize_state(u_ref, u_min, u_max)
        u_actual_norm = normalize_state(u_actual, u_min, u_max)

        r_ref_min, r_ref_max = -0.1745, 0.1745
        r_actual_min, r_actual_max = -0.2862, 0.2862
        r_ref, r_actual = action[1], robot_state[5, 0]
        r_ref_norm = normalize_state(r_ref, r_ref_min, r_ref_max)
        r_actual_norm = normalize_state(r_actual, r_actual_min, r_actual_max)

        state = min_values + [distance_norm, cos, sin] + [u_ref_norm, r_ref_norm] + [u_actual_norm, r_actual_norm, n1_norm, n2_norm]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal

def normalize_state(x, min_val, max_val):
    x = np.asarray(x, dtype=np.float32)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return np.clip((x - min_val) / denom, 0.0, 1.0)
