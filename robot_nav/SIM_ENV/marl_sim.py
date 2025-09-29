import irsim
import numpy as np
import random
import torch

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class MARL_SIM(SIM_ENV):
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

    def __init__(
        self,
        world_file="multi_robot_world.yaml",
        disable_plotting=False,
        reward_phase=1,
    ):
        """
        Initialize the MARL_SIM environment.

        Args:
            world_file (str, optional):
                Path to an IRSim YAML world configuration. Defaults to
                "multi_robot_world.yaml".
            disable_plotting (bool, optional):
                If True, disables all IRSim plotting and display windows.
                Useful for headless training. Defaults to False.
            reward_phase (int, optional):
                Selects the reward function variant used by `get_reward`.
                Supported values: {1, 2}. Defaults to 1.
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
        self.reward_phase = reward_phase

    def step(self, action, connection, combined_weights=None):
        """
        Perform a simulation step for all robots using the provided actions and connections.

        Args:
            action (list):
                A list of length `num_robots`. Each element is
                `[linear_velocity, angular_velocity]` to be applied to the
                corresponding robot for this step. Units follow IRSim's
                kinematics setup (typically m/s and rad/s).
            connection (Tensor):
                A torch.Tensor of shape `(num_robots, num_robots-1)` containing
                *logits* that indicate pairwise connections per robot *without*
                the self-connection column. This is not used for logic here,
                but preserved for compatibility (see commented code); you may
                use it upstream to form `combined_weights`.
            combined_weights (Tensor or None, optional):
                A torch.Tensor of shape `(num_robots, num_robots)` **or**
                `(num_robots, num_robots-1)` that encodes visualization weights
                for robot-to-robot edges. When provided, edges from robot *i* to
                *j* are drawn with line width `weight * 2`. If you pass the
                `(num_robots, num_robots-1)` form, ensure indexing aligns with
                how you constructed it (self-column typically omitted).

        Returns:
            tuple:
                (
                    poses (list[list[float]]):
                        `[x, y, theta]` for each robot after the step.
                    distances (list[float]):
                        Euclidean distance from each robot to its goal.
                    coss (list[float]):
                        Cosine of the angle between robot heading and goal vector.
                    sins (list[float]):
                        Sine of the angle between robot heading and goal vector.
                    collisions (list[bool]):
                        Collision flags for each robot, as reported by IRSim.
                    goals (list[bool]):
                        Arrival flags for each robot. If True this step, a new
                        random goal was scheduled for that robot.
                    action (list):
                        Echo of the input `action`.
                    rewards (list[float]):
                        Per-robot rewards computed by `get_reward(...)`.
                    positions (list[list[float]]):
                        `[x, y]` positions for each robot after the step.
                    goal_positions (list[list[float]]):
                        `[x, y]` goal coordinates for each robot after any
                        arrival updates this step.
                )
        """
        self.env.step(action_id=[i for i in range(self.num_robots)], action=action)
        self.env.render()

        poses = []
        distances = []
        coss = []
        sins = []
        collisions = []
        goals = []
        rewards = []
        positions = []
        goal_positions = []
        robot_states = [
            [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
            for i in range(self.num_robots)
        ]
        for i in range(self.num_robots):

            robot_state = self.env.robot_list[i].state
            closest_robots = [
                np.linalg.norm(
                    [
                        robot_states[j][0] - robot_state[0],
                        robot_states[j][1] - robot_state[1],
                    ]
                )
                for j in range(self.num_robots)
                if j != i
            ]
            robot_goal = self.env.robot_list[i].goal
            goal_vector = [
                robot_goal[0].item() - robot_state[0].item(),
                robot_goal[1].item() - robot_state[1].item(),
            ]
            distance = np.linalg.norm(goal_vector)
            goal = self.env.robot_list[i].arrive
            pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
            cos, sin = self.cossin(pose_vector, goal_vector)
            collision = self.env.robot_list[i].collision
            action_i = action[i]
            reward = self.get_reward(
                goal, collision, action_i, closest_robots, distance, self.reward_phase
            )

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            distances.append(distance)
            coss.append(cos)
            sins.append(sin)
            collisions.append(collision)
            goals.append(goal)
            rewards.append(reward)
            positions.append(position)
            poses.append(
                [robot_state[0].item(), robot_state[1].item(), robot_state[2].item()]
            )
            goal_positions.append(goal_position)

            if combined_weights is not None:
                i_weights = combined_weights[i].tolist()

            for j in range(self.num_robots):
                if combined_weights is not None:
                    weight = i_weights[j]
                    # else:
                    #     weight = 1
                    other_robot_state = self.env.robot_list[j].state
                    other_pos = [
                        other_robot_state[0].item(),
                        other_robot_state[1].item(),
                    ]
                    rx = [position[0], other_pos[0]]
                    ry = [position[1], other_pos[1]]
                    self.env.draw_trajectory(
                        np.array([rx, ry]), refresh=True, linewidth=weight * 2
                    )

            if goal:
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[
                        [self.x_range[0] + 1, self.y_range[0] + 1, -3.141592653589793],
                        [self.x_range[1] - 1, self.y_range[1] - 1, 3.141592653589793],
                    ],
                )

        return (
            poses,
            distances,
            coss,
            sins,
            collisions,
            goals,
            action,
            rewards,
            positions,
            goal_positions,
        )

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
            robot_state (list or None, optional): Initial robot state(s) as [[x], [y], [theta]].
                If None, random states are assigned ensuring minimum spacing between robots.
            robot_goal (list or None, optional): Fixed goal position(s) for robots. If None, random
                goals are generated within the environment boundaries.
            random_obstacles (bool, optional): If True, reposition obstacles randomly within bounds.
            random_obstacle_ids (list or None, optional): IDs of obstacles to randomize. Defaults to
                seven obstacles starting from index equal to the number of robots.

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

        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        con = torch.tensor(
            [[0.0 for _ in range(self.num_robots - 1)] for _ in range(self.num_robots)]
        )
        poses, distance, cos, sin, _, _, action, reward, positions, goal_positions = (
            self.step(action, con)
        )
        return (
            poses,
            distance,
            cos,
            sin,
            [False] * self.num_robots,
            [False] * self.num_robots,
            action,
            reward,
            positions,
            goal_positions,
        )

    @staticmethod
    def get_reward(goal, collision, action, closest_robots, distance, phase=1):
        """
        Compute the reward for a single robot based on goal progress, collisions,
        control effort, and proximity to other robots.

        Args:
            goal (bool): Whether the robot reached its goal.
            collision (bool): Whether the robot collided with an obstacle or another robot.
            action (list): [linear_velocity, angular_velocity] applied by the robot.
            closest_robots (list): Distances to other robots.
            distance (float): Current distance to the goal.
            phase (int, optional): Reward configuration (1 or 2). Default is 1.

        Returns:
            float: Computed scalar reward.

        Raises:
            ValueError: If an unknown reward phase is specified.
        """

        match phase:
            case 1:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 1.5 / distance
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    return action[0] - 0.5 * abs(action[1]) - cl_pen + r_dist

            case 2:
                if goal:
                    return 70.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (3 - rob) ** 2 if rob < 3 else 0
                        cl_pen += add

                    return -0.5 * abs(action[1]) - cl_pen

            case _:
                raise ValueError("Unknown reward phase")
