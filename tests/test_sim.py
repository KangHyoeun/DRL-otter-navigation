from robot_nav.path_planners.probabilistic_road_map import PRMPlanner
from robot_nav.path_planners.rrt import RRT
from robot_nav.sim import SIM_ENV
import numpy as np
from robot_nav.path_planners.a_star import AStarPlanner
import matplotlib.pyplot as plt


def test_sim():
    sim = SIM_ENV("/tests/test_world.yaml")
    robot_state = sim.env.get_robot_state()
    state = sim.step(1, 0)
    next_robot_state = sim.env.get_robot_state()
    assert np.isclose(robot_state[0], next_robot_state[0] - 1)
    assert np.isclose(robot_state[1], robot_state[1])

    assert len(state[0]) == 180
    assert len(sim.env.obstacle_list) == 7

    sim.reset()
    new_robot_state = sim.env.get_robot_state()
    assert np.not_equal(robot_state[0], new_robot_state[0])
    assert np.not_equal(robot_state[1], new_robot_state[1])


def test_sincos():
    sim = SIM_ENV("/tests/test_world.yaml")
    cos, sin = sim.cossin([1, 0], [0, 1])
    assert np.isclose(cos, 0)
    assert np.isclose(sin, 1)


def test_astar_planner():
    sim = SIM_ENV("/tests/test_world.yaml")
    planner = AStarPlanner(env=sim, resolution=0.3)
    robot_info = sim.env.get_robot_info()
    robot_state = sim.env.get_robot_state()
    sim.env.get_robot_info()
    rx, ry = planner.planning(
        robot_state[0].item(),
        robot_state[1].item(),
        robot_info.goal[0].item(),
        robot_info.goal[1].item(),
    )
    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()


def test_prm_planner():
    sim = SIM_ENV("/tests/test_world.yaml")
    planner = PRMPlanner(env=sim, robot_radius=0.3)
    robot_info = sim.env.get_robot_info()
    robot_state = sim.env.get_robot_state()
    sim.env.get_robot_info()
    rx, ry = planner.planning(
        robot_state[0].item(),
        robot_state[1].item(),
        robot_info.goal[0].item(),
        robot_info.goal[1].item(),
    )

    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()


def test_rrt_planner():
    sim = SIM_ENV("/tests/test_world.yaml")
    planner = RRT(env=sim, robot_radius=0.3)
    robot_info = sim.env.get_robot_info()
    robot_state = sim.env.get_robot_state()
    sim.env.get_robot_info()
    rx, ry = planner.planning(
        robot_state[0].item(),
        robot_state[1].item(),
        robot_info.goal[0].item(),
        robot_info.goal[1].item(),
    )

    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()
