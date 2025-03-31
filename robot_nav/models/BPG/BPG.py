from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

    def forward(self, s, a):
        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q = self.layer_3(s1)

        return q


# TD3 network
class BPG(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/BPG/checkpoint"),
        model_name="BPG",
        load_directory=Path("robot_nav/models/BPG/checkpoint"),
        bound_weight=8,
    ):
        # Initialize the Actor network
        self.bound_weight = bound_weight
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def get_action(self, obs, add_noise):
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        max_lin_vel=0.5,
        max_ang_vel=1,
        goal_reward=100,
        distance_norm=10,
        time_step=0.3,
    ):
        av_Q = 0
        av_bound = 0
        max_b = 0
        max_Q = -inf
        av_loss = 0
        av_target_loss = 0
        av_max_bound_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q = self.critic_target(next_state, next_action)

            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q = self.critic(state, action)

            max_bound = self.get_max_bound(
                next_state,
                discount,
                max_ang_vel,
                max_lin_vel,
                time_step,
                distance_norm,
                goal_reward,
                reward,
                done,
            )
            max_b = max(max_b, torch.max(max_bound))
            av_bound += torch.mean(max_bound)

            max_bound_Q = torch.min(current_Q, max_bound)
            max_bound_loss = F.mse_loss(current_Q, max_bound_Q)
            # Calculate the loss between the current Q value and the target Q value
            loss_target_Q = F.mse_loss(current_Q, target_Q)

            max_bound_loss = self.bound_weight * max_bound_loss
            loss = loss_target_Q + max_bound_loss
            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad = self.critic(state, self.actor(state))
                actor_grad = torch.min(actor_grad, max_bound)
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
            av_target_loss += loss_target_Q
            av_max_bound_loss += max_bound_loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/av_target_loss", av_target_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/av_max_bound_loss", av_max_bound_loss / iterations, self.iter_count
        )
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/avg_bound", av_bound / iterations, self.iter_count
        )
        self.writer.add_scalar("train/max_b", max_b, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def get_max_bound(
        self,
        next_state,
        discount,
        max_ang_vel,
        max_lin_vel,
        time_step,
        distance_norm,
        goal_reward,
        reward,
        done,
    ):
        cos = next_state[:, -4]
        sin = next_state[:, -3]
        theta = torch.atan2(sin, cos)

        turn_steps = theta / (max_ang_vel * time_step)
        full_turn_steps = torch.floor(turn_steps.abs())
        turn_rew = [
            (
                -1 * discount**step * max_ang_vel
                if step
                else torch.zeros(1, device=self.device)
            )
            for step in full_turn_steps
        ]
        final_turn = turn_steps.abs() - full_turn_steps
        final_turn_rew = [
            -1 * discount ** (full_turn_steps[i] + 1) * final_turn[i]
            for i in range(len(final_turn))
        ]
        full_turn_rew = torch.tensor(turn_rew, device=self.device) + torch.tensor(
            final_turn_rew, device=self.device
        )

        full_turn_steps += 1
        distances = next_state[:, -5]
        distances *= distance_norm
        distances /= max_lin_vel * time_step
        final_steps = torch.ceil(distances) + full_turn_steps
        inter_steps = torch.trunc(distances) + full_turn_steps
        final_discount = torch.tensor(
            [discount**pw for pw in final_steps], device=self.device
        )
        final_rew = (
            torch.ones_like(distances, device=self.device)
            * goal_reward
            * final_discount
        )

        max_inter_steps = inter_steps.max()
        exponents = torch.arange(
            1, max_inter_steps + 1, dtype=torch.float32, device=self.device
        )
        discount_exponents = torch.tensor(
            [discount**e for e in exponents], device=self.device
        )
        inter_rew = torch.tensor(
            [
                sum(
                    [
                        max_lin_vel * discount_exponents[i]
                        for i in range(int(start) + 1, int(steps))
                    ]
                )
                for start, steps in zip(full_turn_steps, inter_steps)
            ],
            device=self.device,
        )
        max_future_rew = full_turn_rew + final_rew + inter_rew
        max_bound = reward + (1 - done) * max_future_rew.view(-1, 1)
        return max_bound

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, filename),
        )
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        # update the returned data from ROS into a form used for learning in the current model
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 7)

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
