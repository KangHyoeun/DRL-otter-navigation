import statistics

from tqdm import tqdm

import torch
import numpy as np


import irsim
import numpy as np
import random
import torch

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class RVO_SIM(SIM_ENV):
    """
    Simulation environment for multi-agent robot navigation using IRSim.

    This class extends the SIM_ENV and provides a wrapper for multi-robot
    simulation and interaction, supporting reward computation and custom reset logic.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for the robots.
        num_robots (int): Number of robots in the environment.
        x_range (tuple): World x-range.
        y_range (tuple): World y-range.
    """

    def __init__(self, world_file="multi_robot_world.yaml", disable_plotting=False):
        """
        Initialize the MARL_SIM environment.

        Args:
            world_file (str, optional): Path to the world configuration YAML file.
            disable_plotting (bool, optional): If True, disables IRSim rendering and plotting.
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
        Perform a simulation step for all robots using the provided actions and connections.

        Args:
            action (list): List of actions for each robot [[lin_vel, ang_vel], ...].
            connection (Tensor): Tensor of shape (num_robots, num_robots-1) containing logits indicating connections between robots.
            combined_weights (Tensor or None, optional): Optional weights for each connection, shape (num_robots, num_robots-1).

        Returns:
            tuple: (
                poses (list): List of [x, y, theta] for each robot,
                distances (list): Distance to goal for each robot,
                coss (list): Cosine of angle to goal for each robot,
                sins (list): Sine of angle to goal for each robot,
                collisions (list): Collision status for each robot,
                goals (list): Goal reached status for each robot,
                action (list): Actions applied,
                rewards (list): Rewards computed,
                positions (list): Current [x, y] for each robot,
                goal_positions (list): Goal [x, y] for each robot,
            )
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
        Reset the simulation environment and optionally set robot and obstacle positions.

        Args:
            robot_state (list or None, optional): Initial state for robots as [x, y, theta, speed].
            robot_goal (list or None, optional): Goal position(s) for the robots.
            random_obstacles (bool, optional): If True, randomly position obstacles.
            random_obstacle_ids (list or None, optional): IDs of obstacles to randomize.

        Returns:
            tuple: (
                poses (list): List of [x, y, theta] for each robot,
                distances (list): Distance to goal for each robot,
                coss (list): Cosine of angle to goal for each robot,
                sins (list): Sine of angle to goal for each robot,
                collisions (list): All False after reset,
                goals (list): All False after reset,
                action (list): Initial action ([[0.0, 0.0], ...]),
                rewards (list): Rewards for initial state,
                positions (list): Initial [x, y] for each robot,
                goal_positions (list): Initial goal [x, y] for each robot,
            )
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

        _, _, positions= self.step()
        return [False] * self.num_robots, [False] * self.num_robots, positions

    def get_reward(self):
        pass


def outside_of_bounds(poses):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.

    Returns:
        bool: True if any robot is outside the 21x21 area centered at (6, 6), else False.
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
    episode = 0
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    test_scenarios = 1000

    # ---- Instantiate simulation environment and model ----
    sim = RVO_SIM(
        world_file="multi_robot_world.yaml", disable_plotting=True
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
            sum(collision)>0.5 or steps == max_steps or outside
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
