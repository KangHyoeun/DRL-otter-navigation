from math import atan2

import numpy as np
from numpy import clip
from torch.utils.tensorboard import SummaryWriter
import yaml


class HCM(object):
    """
    A class representing a Hard-Coded model (HCM) for a robot's navigation system.

    This class contains methods for generating actions based on the robot's state, preparing state
    representations, training (placeholder method), saving/loading models, and logging experiences.
    The method is suboptimal in order to collect collisions for pre-training of DRL models.

    Attributes:
        max_action (float): The maximum possible action value.
        state_dim (int): The dimension of the state representation.
        writer (SummaryWriter): The writer for logging purposes.
        iterator (int): A counter for tracking sample addition.
        save_samples (bool): Whether to save the samples to a file.
        max_added_samples (int): Maximum number of samples to be added to the saved file.
        file_location (str): The file location for saving samples.
    """

    def __init__(
        self,
        state_dim,
        max_action,
        save_samples,
        max_added_samples=10_000,
        file_location="robot_nav/assets/data.yml",
    ):
        """
        Initialize the HCM class with the provided configuration.

        Args:
            state_dim (int): The dimension of the state space.
            max_action (float): The maximum possible action value.
            save_samples (bool): Whether to save samples to a file.
            max_added_samples (int): The maximum number of samples to save.
            file_location (str): The file path for saving samples.
        """
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter()
        self.iterator = 0
        self.save_samples = save_samples
        self.max_added_samples = max_added_samples
        self.file_location = file_location

    def get_action(self, state, add_noise):
        """
        Compute the action to be taken based on the current state of the robot.

        Args:
            state (list): The current state of the robot, including LIDAR scan, distance,
                          and other relevant features.
            add_noise (bool): Whether to add noise to the action for exploration.

        Returns:
            (list): The computed action [linear velocity, angular velocity].
        """
        sin = state[-3]
        cos = state[-4]
        angle = atan2(sin, cos)
        laser_nr = self.state_dim - 5
        limit = 1.5

        if min(state[4 : self.state_dim - 9]) < limit:
            state = state.tolist()
            idx = state[:laser_nr].index(min(state[:laser_nr]))
            if idx > laser_nr / 2:
                sign = -1
            else:
                sign = 1

            idx = clip(idx + sign * 5 * (limit / min(state[:laser_nr])), 0, laser_nr)

            angle = ((3.14 / (laser_nr)) * idx) - 1.57

        rot_vel = clip(angle, -1.0, 1.0)
        lin_vel = -abs(rot_vel / 2)
        return [lin_vel, rot_vel]

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99999,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """
        Placeholder method for training the hybrid control model.

        Args:
            replay_buffer (object): The replay buffer containing experiences.
            iterations (int): The number of training iterations.
            batch_size (int): The batch size for training.
            discount (float): The discount factor for future rewards.
            tau (float): The soft update parameter for target networks.
            policy_noise (float): The noise added to actions during training.
            noise_clip (float): The clipping value for action noise.
            policy_freq (int): The frequency at which to update the policy.

        Note:
            This method is a placeholder and currently does nothing.
        """
        pass

    def save(self, filename, directory):
        """
        Placeholder method to save the current model state to a file.

        Args:
            filename (str): The name of the file where the model will be saved.
            directory (str): The directory where the file will be stored.

        Note:
            This method is a placeholder and currently does nothing.
        """
        pass

    def load(self, filename, directory):
        """
        Placeholder method to load a model state from a file.

        Args:
            filename (str): The name of the file to load the model from.
            directory (str): The directory where the model file is stored.

        Note:
            This method is a placeholder and currently does nothing.
        """
        pass

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        Prepare the state representation for the model based on the current environment.

        Args:
            latest_scan (list): The LIDAR scan data.
            distance (float): The distance to the goal.
            cos (float): The cosine of the robot's orientation angle.
            sin (float): The sine of the robot's orientation angle.
            collision (bool): Whether a collision occurred.
            goal (bool): Whether the goal has been reached.
            action (list): The action taken by the robot, [linear velocity, angular velocity].

        Returns:
            (tuple): A tuple containing the prepared state and a terminal flag (1 if terminal state, 0 otherwise).
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
        state = min_values + [distance, cos, sin] + [action[0], action[1]]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        self.iterator += 1
        if self.save_samples and self.iterator < self.max_added_samples:
            action = action if type(action) is list else action
            action = [float(a) for a in action]
            sample = {
                self.iterator: {
                    "latest_scan": latest_scan.tolist(),
                    "distance": distance.tolist(),
                    "cos": cos.tolist(),
                    "sin": sin.tolist(),
                    "collision": collision,
                    "goal": goal,
                    "action": action,
                }
            }
            with open(self.file_location, "a") as outfile:
                yaml.dump(sample, outfile, default_flow_style=False)

        return state, terminal
