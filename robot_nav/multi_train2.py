from robot_nav.models.TD3.TD3 import TD3
from robot_nav.models.DDPG.DDPG import DDPG
from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.HCM.hardcoded_model import HCM
from robot_nav.models.PPO.PPO import PPO
from robot_nav.models.CNNTD3.att import CNNTD3

import torch
import numpy as np
from sim2 import SIM_ENV
from utils import get_buffer

def outside_of_bounds(poses):
    outside = False
    for pose in poses:
        norm_x = pose[0] - 6
        norm_y = pose[1] - 6
        if abs(norm_x) > 10.5 or abs(norm_y) > 10.5:
            outside = True
            break
    return outside

def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 11  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 10  # how many episodes to use to run evaluation
    max_epochs = 100  # max number of epochs
    epoch = 0  # starting epoch number
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 10  # train and update network parameters every n episodes
    training_iterations = 80  # how many batches to use for single training cycle
    batch_size = 16  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        10  # number of training iterations to run during pre-training
    )
    save_every = 5  # save the model every n training cycles



    sim = SIM_ENV(world_file="multi_robot_world2.yaml",disable_plotting=False)  # instantiate environment

    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
    save_every=save_every,
    load_model=True,
    model_name="phase2",
    load_model_name="phase1"
    )  # instantiate a model

    replay_buffer = get_buffer(
        model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )
    con = torch.tensor([[0. for _ in range(sim.num_robots-1)] for _ in range(sim.num_robots) ])

    poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = sim.step([[0, 0] for _ in range(sim.num_robots)], con)  # get the initial step state
    running_goals = 0
    running_collisions = 0
    running_timesteps = 0
    iter = 1
    while epoch < max_epochs:  # train until max_epochs is reached
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, goal, a, positions, goal_positions
        )  # get state a state representation from returned data from the environment

        action, connection, combined_weights = model.get_action(np.array(state), True) # get an action from the model

        a_in = [[(a[0] + 1) / 4, a[1]] for a in action]  # clip linear velocity to [0, 0.5] m/s range

        poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = sim.step(a_in, connection, combined_weights)  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1
        next_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, goal, a, positions, goal_positions
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer
        outside = outside_of_bounds(poses)
        if (
            any(terminal) or steps == max_steps or outside
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = sim.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.writer.add_scalar("run/avg_goal", running_goals/running_timesteps, iter)
                model.writer.add_scalar("run/avg_collision", running_collisions / running_timesteps, iter)
                running_goals = 0
                running_collisions = 0
                running_timesteps = 0
                iter += 1
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

        # if (
        #     episode + 1
        # ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
        #     episode = 0
        #     epoch += 1
        #     # evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0
        poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                poses, distance, cos, sin, collision, goal, a, positions, goal_positions
            )
            action, connection, combined_weights = model.get_action(np.array(state), False)
            a_in = [[(a[0] + 1) / 4, a[1]] for a in action]
            poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = sim.step(a_in, connection, combined_weights)
            avg_reward += sum(reward)/len(reward)
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
