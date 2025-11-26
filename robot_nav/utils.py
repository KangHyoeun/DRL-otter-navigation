from typing import List
from tqdm import tqdm
import yaml
import torch

from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.replay_buffer import ReplayBuffer, RolloutReplayBuffer
from robot_nav.models.PPO.PPO import PPO


class Pretraining:
    """
    Handles loading of offline experience data and pretraining of a reinforcement learning model.

    Attributes:
        file_names (List[str]): List of YAML files containing pre-recorded environment samples.
        model (object): The model with `prepare_state` and `train` methods.
        replay_buffer (object): The buffer used to store experiences for training.
        reward_function (callable): Function to compute the reward from the environment state.
    """

    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function

    def load_buffer(self):
        """
        Load samples from the specified files and populate the replay buffer.

        Returns:
            (object): The populated replay buffer.
        """
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    latest_scan = sample["latest_scan"]
                    distance = sample["distance"]
                    cos = sample["cos"]
                    sin = sample["sin"]
                    collision = sample["collision"]
                    goal = sample["goal"]
                    action = sample["action"]

                    state, terminal = self.model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, action
                    )

                    if terminal:
                        continue

                    next_sample = samples[i + 1]
                    next_latest_scan = next_sample["latest_scan"]
                    next_distance = next_sample["distance"]
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_collision = next_sample["collision"]
                    next_goal = next_sample["goal"]
                    next_action = next_sample["action"]
                    next_state, next_terminal = self.model.prepare_state(
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                        next_collision,
                        next_goal,
                        next_action,
                    )
                    reward = self.reward_function(
                        next_goal, next_collision, action, next_latest_scan
                    )
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        """
        Run pretraining on the model using the replay buffer.

        Args:
            pretraining_iterations (int): Number of outer loop iterations for pretraining.
            replay_buffer (object): Buffer to sample training batches from.
            iterations (int): Number of training steps per pretraining iteration.
            batch_size (int): Batch size used during training.
        """
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")


def get_buffer(
    model,
    sim,
    load_saved_buffer,
    pretrain,
    pretraining_iterations,
    training_iterations,
    batch_size,
    buffer_size=50000,
    random_seed=666,
    file_names=["robot_nav/assets/data.yml"],
    history_len=10,
):
    """
    Get or construct the replay buffer depending on model type and training configuration.

    Args:
        model (object): The RL model, can be PPO, RCPG, or other.
        sim (object): Simulation environment with a `get_reward` function.
        load_saved_buffer (bool): Whether to load experiences from file.
        pretrain (bool): Whether to run pretraining using the buffer.
        pretraining_iterations (int): Number of outer loop iterations for pretraining.
        training_iterations (int): Number of iterations in each training loop.
        batch_size (int): Size of the training batch.
        buffer_size (int, optional): Maximum size of the buffer. Defaults to 50000.
        random_seed (int, optional): Seed for reproducibility. Defaults to 666.
        file_names (List[str], optional): List of YAML data file paths. Defaults to ["robot_nav/assets/data.yml"].
        history_len (int, optional): Used for RCPG buffer configuration. Defaults to 10.

    Returns:
        (object): The initialized and optionally pre-populated replay buffer.
    """
    if isinstance(model, PPO):
        return model.buffer

    if isinstance(model, RCPG):
        replay_buffer = RolloutReplayBuffer(
            buffer_size=buffer_size, random_seed=random_seed, history_len=history_len
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    if pretrain:
        assert (
            load_saved_buffer
        ), "To pre-train model, load_saved_buffer must be set to True"

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=file_names,
            model=model,
            replay_buffer=replay_buffer,
            reward_function=sim.get_reward,
        )  # instantiate pre-trainind
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training

    return replay_buffer


def get_max_bound(
    next_state,
    discount,
    max_ang_vel,
    max_lin_vel,
    time_step,
    distance_norm,
    goal_reward,
    reward,
    done,
    device,
):
    """
    Estimate the maximum possible return (upper bound) from the next state onward.

    This is used in constrained RL or safe policy optimization where a conservative
    estimate of return is useful for policy updates.

    Args:
        next_state (torch.Tensor): Tensor of next state observations.
        discount (float): Discount factor for future rewards.
        max_ang_vel (float): Maximum angular velocity of the agent.
        max_lin_vel (float): Maximum linear velocity of the agent.
        time_step (float): Duration of one time step.
        distance_norm (float): Normalization factor for distance.
        goal_reward (float): Reward received upon reaching the goal.
        reward (torch.Tensor): Immediate reward from the environment.
        done (torch.Tensor): Binary tensor indicating episode termination.
        device (torch.device): PyTorch device for computation.

    Returns:
        (torch.Tensor): Maximum return bound for each sample in the batch.
    """
    next_state = next_state.clone()  # Prevents in-place modifications
    reward = reward.clone()  # Ensures original reward is unchanged
    done = done.clone()
    
    # FIXED: Correct indexing for extended state [360 LiDAR + 3 goal + 2 action + 2 vel + 2 rps]
    cos = next_state[:, -8]  # cos is at index -8
    sin = next_state[:, -7]  # sin is at index -7
    theta = torch.atan2(sin, cos)

    # Compute turning steps
    turn_steps = theta.abs() / (max_ang_vel * time_step)
    full_turn_steps = torch.floor(turn_steps)
    turn_rew = -max_ang_vel * discount**full_turn_steps
    turn_rew[full_turn_steps == 0] = 0  # Handle zero case
    final_turn_rew = -(discount ** (full_turn_steps + 1)) * (
        turn_steps - full_turn_steps
    )
    full_turn_rew = turn_rew + final_turn_rew

    # Compute distance-based steps
    full_turn_steps += 1  # Account for the final turn step
    distances = (next_state[:, -9] * distance_norm) / (max_lin_vel * time_step)  # FIXED: distance is at index -9
    final_steps = torch.ceil(distances) + full_turn_steps
    inter_steps = torch.trunc(distances) + full_turn_steps

    final_rew = goal_reward * discount**final_steps

    # Compute intermediate rewards using a sum of discounted steps
    max_inter_steps = inter_steps.max().int().item()
    discount_exponents = discount ** torch.arange(1, max_inter_steps + 1, device=device)
    inter_rew = torch.stack(
        [
            (max_lin_vel * discount_exponents[int(start) + 1 : int(steps)]).sum()
            for start, steps in zip(full_turn_steps, inter_steps)
        ]
    )
    # Compute final max bound
    max_bound = reward + (1 - done) * (full_turn_rew + final_rew + inter_rew).view(
        -1, 1
    )
    return max_bound


class RunningMeanStd:
    """
    Calibrates the mean and standard deviation of data over time.
    Used for normalizing observations and rewards.
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

import numpy as np
from colregs_core.geometry import math_to_maritime_velocity

def prepare_multi_modal_state(distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action, robot_state, CR_max, grid_map=None):
    """
    Prepares the 12-dim vector state AND the 2D grid map.
    Common state preparation logic for MLPCNNPPO, MLPCNNSAC, MLPCNNTD3.
    
    Args:
        distance (float): Distance to goal
        y_e (float): Cross track error
        psi_e (float): Heading error
        chi_e (float): Course error
        phi_tilde (float): Path angle error
        collision (bool): Collision flag
        goal (bool): Goal reached flag
        action (list/array): Current action [u, r]
        robot_state (np.array): Robot state vector from IR-SIM
        CR_max (float): Maximum collision risk
        grid_map (np.array): 128x128 Occupancy/Risk Grid (passed externally).
                             If None, a dummy grid is created.
    
    Returns:
        tuple: ((vector_state, grid_map), terminal)
    """
    # Velocities
    psi_math_deg = np.degrees(robot_state[2, 0])
    speed = np.linalg.norm([robot_state[3, 0], robot_state[4, 0]])
    v_x, v_y = math_to_maritime_velocity(psi_math_deg, speed)
    
    u_ref, u_actual = action[0], robot_state[3, 0]
    u_e = u_ref - u_actual
    
    r_ref, r_actual = action[1], robot_state[5, 0]
    r_e = r_ref - r_actual
    
    n1, n2 = robot_state[6, 0], robot_state[7, 0]

    # 12-dim Vector
    vector_state = [
        v_x, v_y, 
        distance, 
        y_e, 
        psi_e, 
        chi_e, 
        phi_tilde, 
        CR_max, 
        u_e, 
        r_e, 
        n1, 
        n2
    ]
    
    # 2D Grid
    if grid_map is None:
        # WARNING: This is just a fallback to prevent crash if environment isn't ready
        grid_map = np.zeros((128, 128), dtype=np.float32)
    
    terminal = 1 if collision or goal else 0

    return (vector_state, grid_map), terminal

