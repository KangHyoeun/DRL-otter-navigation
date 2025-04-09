from models.TD3.TD3 import TD3
from models.DDPG.DDPG import DDPG
from models.SAC.SAC import SAC
from models.HCM.hardcoded_model import HCM
from models.PPO.PPO import PPO
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

import torch
import numpy as np
from sim import SIM_ENV
import yaml


def main(args=None):
    """Main testing function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 0  # epoch number
    max_steps = 300  # maximum number of steps in single episode

    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name="TD3",
    )  # instantiate a model

    sim = SIM_ENV(world_file="eval_world.yaml")  # instantiate environment
    with open("robot_nav/eval_points.yaml") as file:
        points = yaml.safe_load(file)
    robot_poses = points["robot"]["poses"]
    robot_goals = points["robot"]["goals"]

    assert len(robot_poses) == len(
        robot_goals
    ), "Number of robot poses do not equal the robot goals"

    print("..............................................")
    print(f"Testing {len(robot_poses)} scenarios")
    total_reward = 0.0
    total_steps = 0
    col = 0
    goals = 0
    for idx in range(len(robot_poses)):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=robot_poses[idx],
            robot_goal=robot_goals[idx],
            random_obstacles=False,
        )
        done = False
        while not done and count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            total_reward += reward
            total_steps += 1
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    avg_step_reward = total_reward / total_steps
    avg_reward = total_reward / len(robot_poses)
    avg_col = col / len(robot_poses)
    avg_goal = goals / len(robot_poses)
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward: {avg_reward}")
    print(f"Average Step Reward: {avg_step_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("test/total_reward", total_reward, epoch)
    model.writer.add_scalar("test/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("test/avg_step_reward", avg_step_reward, epoch)
    model.writer.add_scalar("test/avg_col", avg_col, epoch)
    model.writer.add_scalar("test/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
