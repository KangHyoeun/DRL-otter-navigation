from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    """
    Actor network that outputs continuous actions for a given state input.

    Architecture:
        - Processes 1D laser scan inputs through 3 convolutional layers.
        - Embeds goal and previous action inputs using fully connected layers.
        - Combines all features and passes them through an RNN (GRU, LSTM, or RNN).
        - Outputs action values via a fully connected feedforward head with Tanh activation.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    rnn : str, optional
        Type of RNN layer to use ("lstm", "gru", or "rnn").
    """

    def __init__(self, action_dim, rnn="gru"):
        super(Actor, self).__init__()
        assert rnn in ["lstm", "gru", "rnn"], "Unsupported rnn type"

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        if rnn == "lstm":
            self.rnn = nn.LSTM(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )
        elif rnn == "gru":
            self.rnn = nn.GRU(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        if len(s.shape) == 2:
            s = s.unsqueeze(0)

        batch_n, hist_n, state_n = s.shape
        s = s.reshape(batch_n * hist_n, state_n)

        laser = s[:, :-5]
        goal = s[:, -5:-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, g, a), dim=-1)

        s = s.reshape(batch_n, hist_n, -1)
        output, _ = self.rnn(s)
        last_output = output[:, -1, :]
        s = F.leaky_relu(self.layer_1(last_output))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """
    Critic network that estimates Q-values for state-action pairs.

    Architecture:
        - Processes the same input as the Actor (laser scan, goal, and previous action).
        - Uses two separate Q-networks (double Q-learning) for stability.
        - Each Q-network receives both the RNN-processed state and current action.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    rnn : str, optional
        Type of RNN layer to use ("lstm", "gru", or "rnn").
    """

    def __init__(self, action_dim, rnn="gru"):
        super(Critic, self).__init__()
        assert rnn in ["lstm", "gru", "rnn"], "Unsupported rnn type"

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        if rnn == "lstm":
            self.rnn = nn.LSTM(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )
        elif rnn == "gru":
            self.rnn = nn.GRU(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=36, hidden_size=36, num_layers=1, batch_first=True
            )

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, action):
        batch_n, hist_n, state_n = s.shape
        s = s.reshape(batch_n * hist_n, state_n)

        laser = s[:, :-5]
        goal = s[:, -5:-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, g, a), dim=-1)

        s = s.reshape(batch_n, hist_n, -1)
        output, _ = self.rnn(s)
        last_output = output[:, -1, :]

        s1 = F.leaky_relu(self.layer_1(last_output))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(last_output))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# RCPG network
class RCPG(object):
    """
    Recurrent Convolutional Policy Gradient (RCPG) agent for continuous control tasks.

    This class implements a recurrent actor-critic architecture using twin Q-networks and soft target updates.
    It includes model initialization, training, inference, saving/loading, and ROS-based state preparation.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state.
    action_dim : int
        Dimensionality of the action space.
    max_action : float
        Maximum allowable action value.
    device : torch.device
        Device to run the model on (e.g., 'cuda' or 'cpu').
    lr : float, optional
        Learning rate for actor and critic optimizers. Default is 1e-4.
    save_every : int, optional
        Frequency (in iterations) to save model checkpoints. Default is 0 (disabled).
    load_model : bool, optional
        Whether to load pretrained model weights. Default is False.
    save_directory : Path, optional
        Directory where models are saved. Default is "robot_nav/models/RCPG/checkpoint".
    model_name : str, optional
        Name prefix for model checkpoint files. Default is "RCPG".
    load_directory : Path, optional
        Directory to load pretrained models from. Default is "robot_nav/models/RCPG/checkpoint".
    rnn : str, optional
        Type of RNN to use in networks ("lstm", "gru", or "rnn"). Default is "gru".
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
        save_directory=Path("robot_nav/models/RCPG/checkpoint"),
        model_name="RCPG",
        load_directory=Path("robot_nav/models/RCPG/checkpoint"),
        rnn="gru",
    ):
        # Initialize the Actor network
        self.device = device
        self.actor = Actor(action_dim, rnn).to(self.device)
        self.actor_target = Actor(action_dim, rnn).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(action_dim, rnn).to(self.device)
        self.critic_target = Critic(action_dim, rnn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.model_name = model_name + rnn
        if load_model:
            self.load(filename=self.model_name, directory=load_directory)
        self.save_every = save_every
        self.save_directory = save_directory

    def get_action(self, obs, add_noise):
        """
        Computes an action for the given observation, with optional exploration noise.

        Parameters
        ----------
        obs : array_like
            Input observation (state).
        add_noise : bool
            If True, adds Gaussian noise for exploration.

        Returns
        -------
        np.ndarray
            Action vector clipped to [-max_action, max_action].
        """
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        """
        Returns the actor network's raw output for a given input state.

        Parameters
        ----------
        state : array_like
            State input.

        Returns
        -------
        np.ndarray
            Deterministic action vector from actor network.
        """
        # Function to get the action from the actor
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
    ):
        """
        Performs training over a number of iterations using batches from a replay buffer.

        Parameters
        ----------
        replay_buffer : object
            Experience replay buffer with a sample_batch method.
        iterations : int
            Number of training iterations.
        batch_size : int
            Size of each training batch.
        discount : float, optional
            Discount factor for future rewards (γ). Default is 0.99.
        tau : float, optional
            Soft update parameter for target networks. Default is 0.005.
        policy_noise : float, optional
            Standard deviation of noise added to target actions. Default is 0.2.
        noise_clip : float, optional
            Range to clip the noise. Default is 0.5.
        policy_freq : int, optional
            Frequency of policy updates relative to critic updates. Default is 2.
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
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

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
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
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
        Saves actor and critic model weights to disk.

        Parameters
        ----------
        filename : str
            Base name for saved model files.
        directory : str or Path
            Target directory to save the models.
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
        Loads model weights for actor and critic networks from disk.

        Parameters
        ----------
        filename : str
            Base name of saved model files.
        directory : str or Path
            Directory from which to load model files.
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

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        Converts raw sensor and environment data into a normalized input state vector.

        Parameters
        ----------
        latest_scan : list or np.ndarray
            Laser scan data.
        distance : float
            Distance to the goal.
        cos : float
            Cosine of the heading angle.
        sin : float
            Sine of the heading angle.
        collision : bool
            Whether a collision has occurred.
        goal : bool
            Whether the goal has been reached.
        action : list or np.ndarray
            Previous action taken [linear, angular].

        Returns
        -------
        state : list
            Normalized input state vector.
        terminal : int
            Terminal flag: 1 if goal reached or collision, otherwise 0.
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        latest_scan /= 7

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        state = latest_scan.tolist() + [distance, cos, sin] + [lin_vel, ang_vel]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
