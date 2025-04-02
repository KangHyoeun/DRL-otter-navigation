from robot_nav.models.BPG.BTD3 import BTD3
from robot_nav.models.BPG.BPG import BPG
from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.models.TD3.TD3 import TD3
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.SAC.BSAC import BSAC
from robot_nav.models.DDPG.DDPG import DDPG
from robot_nav.utils import get_buffer
from robot_nav.sim import SIM_ENV
import pytest


@pytest.mark.parametrize(
    "model, state_dim",
    [
        (BSAC, 25),
        (BPG, 10),
        (RCPG, 185),
        (CNNTD3, 185),
        (TD3, 10),
        (SAC, 10),
        (DDPG, 10),
    ],
)
def test_models(model, state_dim):
    test_model = model(
        state_dim=state_dim,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM_ENV

    prefilled_buffer = get_buffer(
        model=test_model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=["test_data.yml"],
    )

    test_model.train(
        replay_buffer=prefilled_buffer,
        iterations=2,
        batch_size=8,
    )
