import os

import pytest

from robot_nav.SIM_ENV.sim import SIM
import numpy as np

skip_on_ci = pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipped on CI (GitHub Actions)"
)


@skip_on_ci
def test_sim():
    sim = SIM("/tests/test_world.yaml")
    robot_state = sim.env.get_robot_state()
    state = sim.step(1, 0)
    next_robot_state = sim.env.get_robot_state()
    assert np.isclose(robot_state[0], next_robot_state[0] - 1)
    assert np.isclose(robot_state[1], robot_state[1])

    assert len(state[0]) == 180
    assert len(sim.env.obstacle_list) == 7

    sim.reset(random_obstacle_ids=[i + 1 for i in range(6)])
    new_robot_state = sim.env.get_robot_state()
    assert np.not_equal(robot_state[0], new_robot_state[0])
    assert np.not_equal(robot_state[1], new_robot_state[1])


@skip_on_ci
def test_sincos():
    sim = SIM("/tests/test_world.yaml")
    cos, sin = sim.cossin([1, 0], [0, 1])
    assert np.isclose(cos, 0)
    assert np.isclose(sin, 1)
