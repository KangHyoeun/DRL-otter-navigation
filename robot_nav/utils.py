from typing import List
from tqdm import tqdm
import yaml

from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.replay_buffer import ReplayBuffer, RolloutReplayBuffer
from robot_nav.models.PPO.PPO import PPO


class Pretraining:
    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function

    def load_buffer(self):
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    latest_scan = sample["latest_scan"]
                    distance = sample["distance"]
                    cos = sample["cos"]
                    sin = sample["sin"]
                    collision = sample["collision"]
                    goal = sample["goal"]
                    action = sample["action"]

                    state, terminal = self.model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, action
                    )

                    if terminal:
                        continue

                    next_sample = samples[i + 1]
                    next_latest_scan = next_sample["latest_scan"]
                    next_distance = next_sample["distance"]
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_collision = next_sample["collision"]
                    next_goal = next_sample["goal"]
                    next_action = next_sample["action"]
                    next_state, next_terminal = self.model.prepare_state(
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                        next_collision,
                        next_goal,
                        next_action,
                    )
                    reward = self.reward_function(
                        next_goal, next_collision, action, next_latest_scan
                    )
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")


def get_buffer(
    model,
    sim,
    load_saved_buffer,
    pretrain,
    pretraining_iterations,
    training_iterations,
    batch_size,
    buffer_size=50000,
    random_seed=666,
    file_names=["robot_nav/assets/data.yml"],
    history_len=10,
):
    if isinstance(model, PPO):
        return model.buffer

    if isinstance(model, RCPG):
        replay_buffer = RolloutReplayBuffer(
            buffer_size=buffer_size, random_seed=random_seed, history_len=history_len
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    if pretrain:
        assert (
            load_saved_buffer
        ), "To pre-train model, load_saved_buffer must be set to True"

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=file_names,
            model=model,
            replay_buffer=replay_buffer,
            reward_function=sim.get_reward,
        )  # instantiate pre-trainind
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training

    return replay_buffer
