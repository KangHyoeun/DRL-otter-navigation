from pathlib import Path

from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.PPO.PPO import PPO, RolloutBuffer
from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.utils import get_buffer, RolloutReplayBuffer, ReplayBuffer
from robot_nav.SIM_ENV.sim import SIM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_buffer():
    model = SAC(
        state_dim=10,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM
    buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=False,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
    )
    assert buffer.count == 0

    assert isinstance(buffer, ReplayBuffer)

    prefilled_buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=[PROJECT_ROOT.joinpath("tests/test_data.yml")],
    )
    assert prefilled_buffer.count == 100


def test_rollout_buffer():
    model = RCPG(
        state_dim=185,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM
    buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=False,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
    )
    assert buffer.count == 0

    assert isinstance(buffer, RolloutReplayBuffer)

    prefilled_buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=[PROJECT_ROOT.joinpath("tests/test_data.yml")],
    )
    assert prefilled_buffer.count == 6


def test_ppo_buffer():
    model = PPO(
        state_dim=10,
        action_dim=10,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )
    sim = SIM
    buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=False,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
    )

    assert isinstance(buffer, RolloutBuffer)
