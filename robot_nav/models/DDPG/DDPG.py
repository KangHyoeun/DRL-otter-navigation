from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.utils import get_max_bound


class Actor(nn.Module):
    """
    Actor network for the DDPG algorithm.

    This network maps input states to actions using a fully connected feedforward architecture.
    It uses Leaky ReLU activations in the hidden layers and a tanh activation at the output
    to ensure the output actions are in the range [-1, 1].

    Architecture:
        - Linear(state_dim → 400) + LeakyReLU
        - Linear(400 → 300) + LeakyReLU
        - Linear(300 → action_dim) + Tanh

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action space.
    """

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Forward pass of the actor network.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            (torch.Tensor): Output action tensor of shape (batch_size, action_dim), scaled to [-1, 1].
        """
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """
    Critic network for the DDPG algorithm.

    This network evaluates the Q-value of a given state-action pair. It separately processes
    state and action inputs through linear layers, combines them, and passes the result through
    another linear layer to predict a scalar Q-value.

    Architecture:
        - Linear(state_dim → 400) + LeakyReLU
        - Linear(400 → 300) [state branch]
        - Linear(action_dim → 300) [action branch]
        - Combine both branches, apply LeakyReLU
        - Linear(300 → 1) for Q-value output

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the input action.
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

    def forward(self, s, a):
        """
        Forward pass of the critic network.

        Args:
            s (torch.Tensor): State tensor of shape (batch_size, state_dim).
            a (torch.Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            (torch.Tensor): Q-value tensor of shape (batch_size, 1).
        """
        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(a))
        q = self.layer_3(s1)

        # s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        # s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        # s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        # q = self.layer_3(s1)

        return q


# DDPG network
class DDPG(object):
    """
    Deep Deterministic Policy Gradient (DDPG) agent implementation.

    This class encapsulates the actor-critic learning framework using DDPG, which is suitable
    for continuous action spaces. It supports training, action selection, model saving/loading,
    and state preparation for a reinforcement learning agent, specifically designed for robot navigation.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the action space.
        max_action (float): Maximum action value allowed.
        device (torch.device): Computation device (CPU or GPU).
        lr (float): Learning rate for the optimizers. Default is 1e-4.
        save_every (int): Frequency of saving the model in training iterations. 0 means no saving. Default is 0.
        load_model (bool): Flag indicating whether to load a model from disk. Default is False.
        save_directory (Path): Directory to save the model checkpoints. Default is "robot_nav/models/DDPG/checkpoint".
        model_name (str): Name used for saving and TensorBoard logging. Default is "DDPG".
        load_directory (Path): Directory to load model checkpoints from. Default is "robot_nav/models/DDPG/checkpoint".
        use_max_bound (bool): Whether to enforce a learned upper bound on the Q-value. Default is False.
        bound_weight (float): Weight of the upper bound loss penalty. Default is 0.25.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/DDPG/checkpoint"),
        model_name="DDPG",
        load_directory=Path("robot_nav/models/DDPG/checkpoint"),
        use_max_bound=False,
        bound_weight=0.25,
    ):
        # Initialize the Actor network
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.use_max_bound = use_max_bound
        self.bound_weight = bound_weight

    def get_action(self, obs, add_noise):
        """
        Selects an action based on the observation.

        Args:
            obs (np.array): The current state observation.
            add_noise (bool): Whether to add exploration noise to the action.

        Returns:
            (np.array): Action selected by the actor network.
        """
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        """
        Computes the action for a given state using the actor network.

        Args:
            state (np.array): Environment state.

        Returns:
            (np.array): Action values as output by the actor network.
        """
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        max_lin_vel=3.0,
        max_ang_vel=0.1745,
        goal_reward=3000,
        distance_norm=20,
        time_step=0.1,
    ):
        """
        Trains the actor and critic networks using a replay buffer and soft target updates.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer object with a sample_batch method.
            iterations (int): Number of training iterations.
            batch_size (int): Size of each training batch.
            discount (float): Discount factor for future rewards.
            tau (float): Soft update factor for target networks.
            policy_noise (float): Standard deviation of noise added to target policy.
            noise_clip (float): Maximum value to clip target policy noise.
            policy_freq (int): Frequency of actor and target updates.
            max_lin_vel (float): Maximum linear velocity, used in Q-bound calculation.
            max_ang_vel (float): Maximum angular velocity, used in Q-bound calculation.
            goal_reward (float): Reward given upon reaching goal.
            distance_norm (float): Distance normalization factor.
            time_step (float): Time step used in max bound calculation.
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
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

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q = self.critic_target(next_state, next_action)

            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q, target_Q)

            if self.use_max_bound:
                max_bound = get_max_bound(
                    next_state,
                    discount,
                    max_ang_vel,
                    max_lin_vel,
                    time_step,
                    distance_norm,
                    goal_reward,
                    reward,
                    done,
                    self.device,
                )
                max_excess_Q = F.relu(current_Q - max_bound)
                max_bound_loss = (max_excess_Q**2).mean()
                # Add loss for Q values exceeding maximum possible upper bound
                loss += self.bound_weight * max_bound_loss

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        """
        Saves the model parameters to disk.

        Args:
            filename (str): Base filename for saving the model components.
            directory (str or Path): Directory where the model files will be saved.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, filename),
        )
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        """
        Loads model parameters from disk.

        Args:
            filename (str): Base filename used for loading model components.
            directory (str or Path): Directory to load the model files from.
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action, robot_state):
        """
        Processes raw sensor input and additional information into a normalized state representation.

        Args:
            latest_scan (list or np.array): Raw LIDAR or laser scan data.
            distance (float): Distance to the goal.
            cos (float): Cosine of the angle to the goal.
            sin (float): Sine of the angle to the goal.
            collision (bool): Whether a collision has occurred.
            goal (bool): Whether the goal has been reached.
            action (list or np.array): The action taken in the previous step.
            robot_state (list or np.array): The state of the robot.
        Returns:
            (tuple): (state vector, terminal flag)
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
