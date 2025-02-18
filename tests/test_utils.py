from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.PPO.PPO import PPO, RolloutBuffer
from robot_nav.utils import get_buffer
from robot_nav.sim import SIM_ENV


def test_buffer():
    model = SAC(
        state_dim=10,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM_ENV
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

    prefilled_buffer = get_buffer(
        model=model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=["test_data.yml"]
    )
    assert prefilled_buffer.count == 100


def test_ppo_buffer():
    model = PPO(
        state_dim=10,
        action_dim=10,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )
    sim = SIM_ENV
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
