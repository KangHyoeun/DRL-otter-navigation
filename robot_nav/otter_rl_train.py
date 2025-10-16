#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otter USV DRL Navigation Training Script

This script trains a DRL agent to navigate the Otter USV using the velocity control interface.
It integrates the realistic marine vehicle dynamics with the existing DRL framework.

Usage:
    poetry run python robot_nav/otter_rl_train.py
"""

import numpy as np
import torch

from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.SIM_ENV.otter_sim import OtterUSVSim
from utils import get_buffer


def main():
    """Main training function for Otter USV DRL navigation."""
    
    # Training parameters
    action_dim = 2  # [linear_velocity, angular_velocity]
    max_action = 1.0  # Maximum action value
    state_dim = 185  # LIDAR (180) + distance + cos + sin (3) + action (2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = 60
    epoch = 0
    episodes_per_epoch = 70
    episode = 0
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 300
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5
    
    print(f"Using device: {device}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="OtterUSV_CNNTD3",
    )  # instantiate a model
    
    sim = OtterUSVSim(
        world_file="worlds/otter_world.yaml", disable_plotting=False
    )  # instantiate environment
    replay_buffer = get_buffer(
        model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )
    
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0
    ) # get the initial step state
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        
        # Get action from DRL model
        action = model.get_action(np.array(state), True)
        a_in = [
            (action[0] + 1) * 1.5,  # Linear velocity: [0, 3.0] m/s
            action[1] * 0.3,  # Angular velocity: [-0.3, 0.3] rad/s
        ]

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )  # get data from the environment
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer
        
        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1
        
        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [
                (action[0] + 1) * 1.5,  # Linear velocity: [0, 3.0] m/s
                action[1] * 0.3,  # Angular velocity: [-0.3, 0.3] rad/s
            ]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
