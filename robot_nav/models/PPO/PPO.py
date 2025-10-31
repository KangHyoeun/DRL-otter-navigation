import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from numpy import inf


class RolloutBuffer:
    """
    Buffer to store rollout data (transitions) for PPO training.

    Attributes:
        actions (list): Actions taken by the agent.
        states (list): States observed by the agent.
        logprobs (list): Log probabilities of the actions.
        rewards (list): Rewards received from the environment.
        state_values (list): Value estimates for the states.
        is_terminals (list): Flags indicating episode termination.
    """

    def __init__(self):
        """
        Initialize empty lists to store buffer elements.
        """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        """
        Clear all stored data from the buffer.
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state):
        """
        Add a transition to the buffer. (Partial implementation.)

        Args:
            state (list or np.array): The current observed state.
            action (list or np.array): The action taken.
            reward (float): The reward received after taking the action.
            terminal (bool): Whether the episode terminated.
            next_state (list or np.array): The resulting state after taking the action.
        """
        self.states.append(state)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network model for PPO.

    Attributes:
        actor (nn.Sequential): Policy network (actor) to output action mean.
        critic (nn.Sequential): Value network (critic) to predict state values.
        action_var (Tensor): Diagonal covariance matrix for action distribution.
        device (str): Device used for computation ('cpu' or 'cuda').
        max_action (float): Clipping range for action values.
    """

    def __init__(self, state_dim, action_dim, action_std_init, max_action, device):
        """
        Initialize the Actor and Critic networks.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            action_std_init (float): Initial standard deviation of the action distribution.
            max_action (float): Maximum value allowed for an action (clipping range).
            device (str): Device to run the model on.
        """
        super(ActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action

        self.action_dim = action_dim
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(self.device)
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, 1),
        )

    def set_action_std(self, new_action_std):
        """
        Set a new standard deviation for the action distribution.

        Args:
            new_action_std (float): New standard deviation.
        """
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(self.device)

    def forward(self):
        """
        Forward method is not implemented, as it's unused directly.

        Raises:
            NotImplementedError: Always raised when called.
        """
        raise NotImplementedError

    def act(self, state, sample):
        """
        Compute an action, its log probability, and the state value.

        Args:
            state (Tensor): Input state tensor.
            sample (bool): Whether to sample from the action distribution or use mean.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Sampled (or mean) action, log probability, and state value.
        """
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        if sample:
            action = torch.clip(
                dist.sample(), min=-self.max_action, max=self.max_action
            )
        else:
            action = dist.mean
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, entropy, and state values for given states and actions.

        Args:
            state (Tensor): Batch of states.
            action (Tensor): Batch of actions.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Action log probabilities, state values, and distribution entropy.
        """
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """
    Proximal Policy Optimization (PPO) implementation for continuous control tasks.

    Attributes:
        max_action (float): Maximum action value.
        action_std (float): Standard deviation of the action distribution.
        action_std_decay_rate (float): Rate at which to decay action standard deviation.
        min_action_std (float): Minimum allowed action standard deviation.
        state_dim (int): Dimension of the state space.
        gamma (float): Discount factor for future rewards.
        eps_clip (float): Clipping range for policy updates.
        device (str): Device for model computation ('cpu' or 'cuda').
        save_every (int): Interval (in iterations) for saving model checkpoints.
        model_name (str): Name used when saving/loading model.
        save_directory (Path): Directory to save model checkpoints.
        iter_count (int): Number of training iterations completed.
        buffer (RolloutBuffer): Buffer to store trajectories.
        policy (ActorCritic): Current actor-critic network.
        optimizer (torch.optim.Optimizer): Optimizer for actor and critic.
        policy_old (ActorCritic): Old actor-critic network for computing PPO updates.
        MseLoss (nn.Module): Mean squared error loss function.
        writer (SummaryWriter): TensorBoard summary writer.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        eps_clip=0.2,
        action_std_init=0.6,
        action_std_decay_rate=0.015,
        min_action_std=0.1,
        device="cpu",
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name="PPO",
        load_directory=Path("robot_nav/models/PPO/checkpoint"),
    ):
        self.max_action = max_action
        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter(comment=model_name)

    def set_action_std(self, new_action_std):
        """
        Set a new standard deviation for the action distribution.

        Args:
            new_action_std (float): New standard deviation value.
        """
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """
        Decay the action standard deviation by a fixed rate, down to a minimum threshold.

        Args:
            action_std_decay_rate (float): Amount to reduce standard deviation by.
            min_action_std (float): Minimum value for standard deviation.
        """
        print(
            "--------------------------------------------------------------------------------------------"
        )
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print(
                "setting actor output action_std to min_action_std : ", self.action_std
            )
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def get_action(self, state, add_noise):
        """
        Sample an action using the current policy (optionally with noise), and store in buffer if noise is added.

        Args:
            state (array_like): Input state for the policy.
            add_noise (bool): Whether to sample from the distribution (True) or use the deterministic mean (False).

        Returns:
            (np.ndarray): Sampled action.
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, add_noise)

        if add_noise:
            # self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size):
        """
        Train the policy and value function using PPO loss based on the stored rollout buffer.

        Args:
            replay_buffer (object): Placeholder for compatibility (not used).
            iterations (int): Number of epochs to optimize the policy per update.
            batch_size (int): Batch size (not used; training uses the whole buffer).
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        assert len(self.buffer.actions) == len(self.buffer.states)

        states = [torch.tensor(st, dtype=torch.float32) for st in self.buffer.states]
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(self.device)
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        av_state_values = 0
        max_state_value = -inf
        av_loss = 0
        # Optimize policy for K epochs
        for _ in range(iterations):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            av_state_values += torch.mean(state_values)
            max_state_value = max(max_state_value, max(state_values))
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            av_loss += loss.mean()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/avg_value", av_state_values / iterations, self.iter_count
        )
        self.writer.add_scalar("train/max_value", max_state_value, self.iter_count)
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action, robot_state):
        """
        Convert raw sensor and navigation data into a normalized state vector for the policy.

        Args:
            latest_scan (list[float]): LIDAR scan data.
            distance (float): Distance to the goal.
            cos (float): Cosine of angle to the goal.
            sin (float): Sine of angle to the goal.
            collision (bool): Whether the robot has collided.
            goal (bool): Whether the robot has reached the goal.
            action (tuple[float, float]): Last action taken (linear and angular velocities).
            robot_state (list or np.array): The state of the robot.
        Returns:
            (tuple[list[float], int]): Processed state vector and terminal flag (1 if terminal, else 0).
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

    def save(self, filename, directory):
        """
        Save the current policy model to the specified directory.

        Args:
            filename (str): Base name of the model file.
            directory (Path): Directory to save the model to.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(
            self.policy_old.state_dict(), "%s/%s_policy.pth" % (directory, filename)
        )

    def load(self, filename, directory):
        """
        Load the policy model from a saved checkpoint.

        Args:
            filename (str): Base name of the model file.
            directory (Path): Directory to load the model from.
        """
        self.policy_old.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        self.policy.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        print(f"Loaded weights from: {directory}")

def normalize_state(x, min_val, max_val):
    x = np.asarray(x, dtype=np.float32)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return np.clip((x - min_val) / denom, 0.0, 1.0)