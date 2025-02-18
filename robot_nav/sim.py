import irsim
import numpy as np
import random


class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml"):
        self.env = irsim.make(world_file)
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(self):
        self.env.robot.set_state(
            state=np.array([[random.uniform(1, 9)], [random.uniform(1, 9)], [0], [0]]),
            init=True,
        )
        self.env.reset()
        self.env.robot.set_goal(
            np.array([[random.uniform(1, 9)], [random.uniform(1, 9)], [0]])
        )
        self.env.random_obstacle_position(
            range_low=[0, 0, -3.14],
            range_high=[10, 10, 3.14],
            ids=[i + 1 for i in range(5)],
        )


        self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    @staticmethod
    def get_reward(goal, collision, action, laser_scan):
        if goal:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
