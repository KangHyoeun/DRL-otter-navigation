import irsim
import numpy as np
import random

import torch


class SIM_ENV:
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.num_robots = len(self.env.robot_list)

    def step(self, action, connection, combined_weights = None):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        # action = [[lin_velocity, ang_velocity], [lin_velocity, ang_velocity], [lin_velocity, ang_velocity], [lin_velocity, ang_velocity], [lin_velocity, ang_velocity]]
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
        robot_states = [[self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]] for i in range(self.num_robots)]
        for i in range(self.num_robots):

            robot_state = self.env.robot_list[i].state
            closest_robots = [np.linalg.norm([robot_states[j][0] - robot_state[0], robot_states[j][1] - robot_state[1]]) for j in
                     range(self.num_robots) if j != i]
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
            reward = self.get_reward(goal, collision, action_i, closest_robots, distance)

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            distances.append(distance)
            coss.append(cos)
            sins.append(sin)
            collisions.append(collision)
            goals.append(goal)
            rewards.append(reward)
            positions.append(position)
            poses.append([robot_state[0].item(), robot_state[1].item(), robot_state[2].item()])
            goal_positions.append(goal_position)

            # gumbel_sample = torch.nn.functional.gumbel_softmax(connection[i], tau=0.5, dim=-1)
            # i_connections = gumbel_sample.tolist()
            # i_connections.insert(i, 0)

            i_probs = torch.sigmoid(connection[i])  # connection[i] is logits for "connect" per pair

            # Now we need to insert the self-connection (optional, typically skipped)
            i_connections = i_probs.tolist()
            i_connections.insert(i, 0)
            if combined_weights is not None:
                i_weights = combined_weights[i].tolist()
                i_weights.insert(i, 0)



            for j in range(self.num_robots):
                if i_connections[j] > 0.5:
                    if combined_weights is not None:
                        weight = i_weights[j]
                    else:
                        weight = 1
                    other_robot_state = self.env.robot_list[j].state
                    other_pos = [other_robot_state[0].item(), other_robot_state[1].item()]
                    rx = [position[0], other_pos[0]]
                    ry = [position[1], other_pos[1]]
                    self.env.draw_trajectory(np.array([rx, ry]), refresh=True, linewidth=weight)

            if goal:
                self.env.robot_list[i].set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    # range_limits=[[self.env.robot_list[i].position[0].item()-3, self.env.robot_list[i].position[1].item()-3, -3.141592653589793], [self.env.robot_list[i].position[0].item()+3, self.env.robot_list[i].position[1].item()+3, 3.141592653589793]],
                    range_limits=[[1, 1, -3.141592653589793],
                                  [11, 11, 3.141592653589793]],
                )

        return poses, distances, coss, sins, collisions, goals, action, rewards, positions, goal_positions

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally setting robot and obstacle states.

        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles.
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize.

        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        if robot_state is None:
            robot_state = [[random.uniform(3, 9)], [random.uniform(3, 9)], [0]]

        init_states = []
        for robot in self.env.robot_list:
            conflict = True
            while conflict:
                conflict = False
                robot_state = [[random.uniform(3, 9)], [random.uniform(3, 9)], [random.uniform(-3.14, 3.14)]]
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
                range_low=[0, 0, -3.14],
                range_high=[12, 12, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        for robot in self.env.robot_list:
            if robot_goal is None:
                robot.set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    # range_limits=[[robot.position[0].item()-3, robot.position[1].item()-3, -3.141592653589793], [robot.position[0].item()+3, robot.position[1].item()+3, 3.141592653589793]],
                    range_limits=[[1, 1, -3.141592653589793],
                                  [11, 11, 3.141592653589793]],
                )
            else:
                self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        con = torch.tensor([[0. for _ in range(self.num_robots-1)] for _ in range(self.num_robots)])
        poses, distance, cos, sin, _, _, action, reward, positions, goal_positions = self.step(action, con)
        return poses, distance, cos, sin, [False]*self.num_robots, [False]*self.num_robots, action, reward, positions, goal_positions

    @staticmethod
    def cossin(vec1, vec2):
        """
        Compute the cosine and sine of the angle between two 2D vectors.

        Args:
            vec1 (list): First 2D vector.
            vec2 (list): Second 2D vector.

        Returns:
            (tuple): (cosine, sine) of the angle between the vectors.
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    @staticmethod
    def get_reward(goal, collision, action, closest_robots, distance):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.

        Returns:
            (float): Computed reward for the current state.
        """
        # if goal:
        #     return 60.0
        # elif collision:
        #     return -100.0
        # else:
        #     cl_pen = 0
        #     for rob in closest_robots:
        #         add = 1.5 - rob if rob < 1.5 else 0
        #         cl_pen += add
        #     return -cl_pen
            # r_dist = 1.25/distance
            # cl_robot = min(closest_robots)
            # cl_pen = 0 - cl_robot if cl_robot < 0 else 0
            # return 2*action[0] - abs(action[1]) - cl_pen + r_dist

        # phase1
        # if goal:
        #     return 100.0
        # elif collision:
        #     return -100.0 * 3 * action[0]
        # else:
        #     r_dist = 1.5/distance
        #     cl_pen = 0
        #     for rob in closest_robots:
        #         add = 1.5 - rob if rob < 1.5 else 0
        #         cl_pen += add
        #
        #     return action[0] - 0.5 * abs(action[1])-cl_pen + r_dist


        # phase2
        # if goal:
        #     return 100.0
        # elif collision:
        #     return -100.0
        # else:
        #     r_dist = 1.5/distance
        #     cl_pen = 0
        #     for rob in closest_robots:
        #         add = 1.5 - rob if rob < 1.5 else 0
        #         cl_pen += add
        #
        #     return -0.5*abs(action[1])-cl_pen

        # phase3
        if goal:
            return 70.0
        elif collision:
            return -100.0 * 3 * action[0]
        else:
            r_dist = 1.5 / distance
            cl_pen = 0
            for rob in closest_robots:
                add = 2.5 - rob if rob < 2.5 else 0
                cl_pen += add

            return -0.5 * abs(action[1]) - cl_pen

