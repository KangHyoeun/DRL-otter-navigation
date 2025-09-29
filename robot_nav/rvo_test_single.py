import statistics

from tqdm import tqdm


import irsim
import numpy as np
import random
import torch

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class RVO_SINGLE_SIM(SIM_ENV):
    """
    Simulation environment for single-scenario multi-robot navigation using IRSim.

    This class extends `SIM_ENV` and wraps the IRSim-based environment to simulate
    multiple robots navigating toward goals while avoiding collisions. It provides
    reset and step functionality with boundary and collision checks.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for all robots.
        num_robots (int): Total number of robots in the environment.
        x_range (tuple): World horizontal range (min_x, max_x).
        y_range (tuple): World vertical range (min_y, max_y).
    """

    def __init__(self, world_file="multi_robot_world.yaml", disable_plotting=False):
        """
        Initialize the IRSim-based multi-robot simulation environment.

        Args:
            world_file (str, optional): Path to the YAML configuration for the world.
                Defaults to "multi_robot_world.yaml".
            disable_plotting (bool, optional): If True, disables rendering and live plotting.
                Defaults to False.
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
        Perform one simulation step for all robots and retrieve updated environment state.

        Executes the IRSim environment's step and render functions, and gathers
        each robot’s position, goal status, and collision flag.

        Returns:
            tuple:
                collisions (list of bool): Collision flags for each robot.
                goals (list of bool): Goal reached flags for each robot.
                positions (list of [float, float]): Current [x, y] positions of robots.
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

        return collisions, goals, positions

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment and optionally randomize robot or obstacle positions.

        Args:
            robot_state (list or None, optional): Custom initial states for robots as
                [x, y, theta, speed]. If None, resets to defaults.
            robot_goal (list or None, optional): Goal positions for robots. If None,
                goals are retained from the previous configuration.
            random_obstacles (bool, optional): Whether to randomize obstacle positions.
                Defaults to False.
            random_obstacle_ids (list or None, optional): List of obstacle IDs to randomize.
                Defaults to None.

        Returns:
            tuple:
                collisions (list of bool): All False after reset.
                goals (list of bool): All False after reset.
                positions (list of [float, float]): Initial [x, y] positions for each robot.
        """
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        _, _, positions = self.step()
        return [False] * self.num_robots, [False] * self.num_robots, positions

    def get_reward(self):
        """
        Compute reward signals for each robot.

        This method is a placeholder for reward computation logic based on
        task progress, collisions, or goal attainment.

        Returns:
            None
        """
        pass


def outside_of_bounds(poses):
    """
    Check whether any robot is outside the defined simulation world boundaries.

    The world is assumed to be centered around (6, 6) with a width and height of 21 units.

    Args:
        poses (list of [float, float, float]): Robot poses [x, y, theta] for all robots.

    Returns:
        bool: True if any robot is outside the bounds (|x-6|>10.5 or |y-6|>10.5), else False.
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
    Run the evaluation loop for multiple test scenarios in the RVO_SINGLE_SIM environment.

    Executes a number of test episodes, stepping through the simulation,
    tracking goals, collisions, and timeouts, and summarizing performance statistics.

    Args:
        args (Namespace or None, optional): Optional runtime arguments (unused).

    Prints:
        - Average collisions per episode and standard deviation.
        - Average goals per episode and standard deviation.
        - Average timesteps per episode and standard deviation.
        - Success rate and timeout count.
    """
    episode = 0
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    test_scenarios = 100

    # ---- Instantiate simulation environment and model ----
    sim = RVO_SINGLE_SIM(
        world_file="worlds/circle_world.yaml", disable_plotting=True
    )  # instantiate environment

    running_goals = 0
    running_collisions = 0
    running_timesteps = 0
    ran_out_of_time = 0

    goals_per_ep = []
    col_per_ep = []
    timesteps_per_ep = []
    pbar = tqdm(total=test_scenarios)
    # ---- Main training loop ----
    while episode < test_scenarios:

        collision, goal, poses = sim.step()  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision) / 2

        running_timesteps += 1
        outside = outside_of_bounds(poses)

        if (
            sum(collision) > 0.5
            or steps == max_steps
            or outside
            or int(sum(goal)) == len(goal)
        ):  # reset environment of terminal state reached, or max_steps were taken
            sim.reset()
            goals_per_ep.append(running_goals)
            running_goals = 0
            col_per_ep.append(running_collisions)
            running_collisions = 0
            timesteps_per_ep.append(running_timesteps)
            running_timesteps = 0
            if steps == max_steps:
                ran_out_of_time += 1

            steps = 0
            episode += 1
            pbar.update(1)
        else:
            steps += 1

    cols = sum(col_per_ep)
    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    t_per_ep = np.array(timesteps_per_ep, dtype=np.float32)
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep)
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep)
    avg_ep_t = statistics.mean(t_per_ep)
    avg_ep_t_std = statistics.stdev(t_per_ep)

    print(f"avg_ep_col: {avg_ep_col}")
    print(f"avg_ep_col_std: {avg_ep_col_std}")
    print(f"success rate: {test_scenarios - cols}")
    print(f"avg_ep_goals: {avg_ep_goals}")
    print(f"avg_ep_goals_std: {avg_ep_goals_std}")
    print(f"avg_ep_t: {avg_ep_t}")
    print(f"avg_ep_t_std: {avg_ep_t_std}")
    print(f"ran out of time: {ran_out_of_time}")
    print("..............................................")


if __name__ == "__main__":
    main()
