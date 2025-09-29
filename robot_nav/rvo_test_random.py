import statistics

from tqdm import tqdm

import torch
import numpy as np


import irsim
import numpy as np
import random
import torch

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class RVO_RANDOM_SIM(SIM_ENV):
    """
    Simulation environment for randomized multi-agent robot navigation using IRSim.

    This class extends `SIM_ENV` and provides a wrapper for running a multi-robot
    simulation where robots navigate toward randomly assigned goals. It includes
    environment reset, step execution, and collision/goal state tracking.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal positions for all robots.
        num_robots (int): Number of robots in the environment.
        x_range (tuple): Range of the x-axis for the simulation world.
        y_range (tuple): Range of the y-axis for the simulation world.
    """

    def __init__(self, world_file="multi_robot_world.yaml", disable_plotting=False):
        """
        Initialize the randomized IRSim-based multi-robot environment.

        Args:
            world_file (str, optional): Path to the YAML configuration file defining
                the world layout. Defaults to "multi_robot_world.yaml".
            disable_plotting (bool, optional): If True, disables visual rendering
                and live plotting in IRSim. Defaults to False.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.num_robots = len(self.env.robot_list)
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range

    def step(self):
        """
        Perform one simulation step for all robots and update goals dynamically.

        Each robot moves based on IRSim’s physics step, with collisions, goals,
        and positions collected. If a robot reaches its goal, a new random goal
        is automatically assigned.

        Returns:
            tuple:
                collisions (list of bool): Collision flags for each robot.
                goals (list of bool): Goal completion flags for each robot.
                positions (list of [float, float]): Current [x, y] positions for each robot.
        """
        self.env.step()
        self.env.render()

        collisions = []
        goals = []
        positions = []
        for i in range(self.num_robots):
            robot_state = self.env.robot_list[i].state
            position = [robot_state[0].item(), robot_state[1].item()]
            positions.append(position)

            goal = self.env.robot_list[i].arrive
            collision = self.env.robot_list[i].collision
            collisions.append(collision)
            goals.append(goal)

            if goal:
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )

        return collisions, goals, positions

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally randomizing robots and obstacles.

        Randomly reinitializes robot positions while avoiding collisions and can
        optionally re-randomize obstacle positions.

        Args:
            robot_state (list or None, optional): Custom initial state for robots as
                [x, y, theta, speed]. If None, random states are assigned.
            robot_goal (list or None, optional): Custom goal positions. If None, random
                goals are assigned within bounds.
            random_obstacles (bool, optional): Whether to randomize obstacle locations.
                Defaults to False.
            random_obstacle_ids (list or None, optional): List of obstacle IDs to randomize.
                If None, defaults to the first 7 obstacles after robots.

        Returns:
            tuple:
                collisions (list of bool): All False immediately after reset.
                goals (list of bool): All False immediately after reset.
                positions (list of [float, float]): Initial [x, y] positions for all robots.
        """
        if robot_state is None:
            robot_state = [[random.uniform(3, 9)], [random.uniform(3, 9)], [0]]

        init_states = []
        for robot in self.env.robot_list:
            conflict = True
            while conflict:
                conflict = False
                robot_state = [
                    [random.uniform(3, 9)],
                    [random.uniform(3, 9)],
                    [random.uniform(-3.14, 3.14)],
                ]
                pos = [robot_state[0][0], robot_state[1][0]]
                for loc in init_states:
                    vector = [
                        pos[0] - loc[0],
                        pos[1] - loc[1],
                    ]
                    if np.linalg.norm(vector) < 0.6:
                        conflict = True
            init_states.append(pos)

            robot.set_state(
                state=np.array(robot_state),
                init=True,
            )

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + self.num_robots for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[self.x_range[0], self.y_range[0], -3.14],
                range_high=[self.x_range[1], self.y_range[1], 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        for robot in self.env.robot_list:
            if robot_goal is None:
                robot.set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )
            else:
                self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        _, _, positions = self.step()
        return [False] * self.num_robots, [False] * self.num_robots, positions

    def get_reward(self):
        """
        Placeholder for computing reward functions for the simulation.

        Intended for integration with learning-based algorithms (e.g. MARL),
        calculating per-agent or global rewards based on goal progress,
        collisions, or path efficiency.

        Returns:
            None
        """
        pass


def outside_of_bounds(poses):
    """
    Determine whether any robot is outside the simulation’s defined world boundary.

    The environment is assumed to be centered at (6, 6) with a 21×21 area.

    Args:
        poses (list of [float, float, float]): Robot poses in [x, y, theta] format.

    Returns:
        bool: True if any robot is outside the world boundary; False otherwise.
    """
    outside = False
    for pose in poses:
        norm_x = pose[0] - 6
        norm_y = pose[1] - 6
        if abs(norm_x) > 10.5 or abs(norm_y) > 10.5:
            outside = True
            break
    return outside


def main(args=None):
    """
    Run multiple randomized test scenarios in the RVO_RANDOM_SIM environment.

    Executes a series of randomized navigation tests, stepping through the environment
    while tracking the number of goals reached, collisions, and time steps per episode.
    Prints out aggregated performance metrics across all episodes.

    Args:
        args (Namespace or None, optional): Optional runtime arguments (unused).

    Prints:
        Summary statistics including:
            - Average collisions per episode (and standard deviation)
            - Average goals per episode (and standard deviation)
    """
    episode = 0
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    test_scenarios = 1000

    # ---- Instantiate simulation environment and model ----
    sim = RVO_RANDOM_SIM(
        world_file="worlds/multi_robot_world.yaml", disable_plotting=True
    )  # instantiate environment

    running_goals = 0
    running_collisions = 0
    running_timesteps = 0

    goals_per_ep = []
    col_per_ep = []
    pbar = tqdm(total=test_scenarios)
    # ---- Main training loop ----
    while episode < test_scenarios:

        collision, goal, poses = sim.step()  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision)

        running_timesteps += 1
        outside = outside_of_bounds(poses)

        if (
            sum(collision) > 0.5 or steps == max_steps or outside
        ):  # reset environment of terminal state reached, or max_steps were taken
            sim.reset()
            goals_per_ep.append(running_goals)
            running_goals = 0
            col_per_ep.append(running_collisions)
            running_collisions = 0

            steps = 0
            episode += 1
            pbar.update(1)
        else:
            steps += 1

    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep)
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep)

    print(f"avg_ep_col: {avg_ep_col}")
    print(f"avg_ep_col_std: {avg_ep_col_std}")
    print(f"avg_ep_goals: {avg_ep_goals}")
    print(f"avg_ep_goals_std: {avg_ep_goals_std}")
    print("..............................................")


if __name__ == "__main__":
    main()
