from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO
import statistics
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random

import torch
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from pathlib import Path
from colregs_core.utils.utils import WrapTo180


def main(args=None):
    """Main testing function for Phase 4"""
    
    # Phase 4 Worlds
    phase4_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_12.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_13.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_14.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_15.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_16.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_17.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_18.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_19.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_20.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_21.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_22.yaml"
    ]

    action_dim = 2
    max_action = 1
    state_dim = 12
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epoch = 0
    max_steps = 512 # Frame Skip adjusted
    test_scenarios = 1000

    # Model Name
    model_name = "otter_MLPCNNPPO_imazu_12_phase4_BEST"

    model = MLPCNNPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name=model_name,
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )

    # Initialize Simulation
    # Note: We initialize with one world, but will reset with random worlds
    sim = OtterSIM(
        world_file=phase4_worlds[0],
        disable_plotting=True, # Enable visualization
        enable_phase1=True, 
        max_steps=max_steps,
        cr_method='jeon', 
        w_efficiency=1.0, 
        w_safety=3.0,
        os_speed_for_cr=3.0, 
        ts_speed_for_cr=3.0,
        chi_inf=1.0, 
        k=1.0
    )

    print("..............................................")
    print(f"Testing {test_scenarios} scenarios")
    print(f"Model: {model_name}")
    
    total_reward = []
    reward_per_ep = []
    lin_actions = []
    ang_actions = []
    total_steps = 0
    col = 0
    goals = 0
    inter_rew = []
    steps_to_goal = []
    
    for _ in tqdm.tqdm(range(test_scenarios)):
        selected_world = random.choice(phase4_worlds)
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        count = 0
        ep_reward = 0
        done = False
        
        while not done and count < max_steps:
            state, terminal = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            # Get deterministic action
            action, _, _ = model.get_action(state, add_noise=False, update_rms=False)
            
            a_in = [
                (action[0] + 1) * 1.5, 
                action[1] * 0.1745
            ]
            
            lin_actions.append(a_in[0])
            ang_actions.append(a_in[1])
            
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            ep_reward += reward
            total_reward.append(reward)
            total_steps += 1
            count += 1
            
            if collision:
                col += 1
            if goal:
                goals += 1
                steps_to_goal.append(count)
            done = collision or goal
            
            if done:
                reward_per_ep.append(ep_reward)
        
        if not done and count >= max_steps:
            reward_per_ep.append(ep_reward)

    total_reward = np.array(total_reward)
    reward_per_ep = np.array(reward_per_ep)
    inter_rew = np.array(inter_rew)
    steps_to_goal = np.array(steps_to_goal)
    lin_actions = np.array(lin_actions)
    ang_actions = np.array(ang_actions)
    
    avg_step_reward = statistics.mean(total_reward) if len(total_reward) > 0 else 0.0
    avg_step_reward_std = statistics.stdev(total_reward) if len(total_reward) > 1 else 0.0
    avg_ep_reward = statistics.mean(reward_per_ep) if len(reward_per_ep) > 0 else 0.0
    avg_ep_reward_std = statistics.stdev(reward_per_ep) if len(reward_per_ep) > 1 else 0.0
    avg_col = col / test_scenarios
    avg_goal = goals / test_scenarios
    avg_inter_step_rew = statistics.mean(inter_rew) if len(inter_rew) > 0 else 0.0
    avg_inter_step_rew_std = statistics.stdev(inter_rew) if len(inter_rew) > 1 else 0.0
    avg_steps_to_goal = statistics.mean(steps_to_goal) if len(steps_to_goal) > 0 else 0.0
    avg_steps_to_goal_std = statistics.stdev(steps_to_goal) if len(steps_to_goal) > 1 else 0.0
    mean_lin_action = statistics.mean(lin_actions) if len(lin_actions) > 0 else 0.0
    lin_actions_std = statistics.stdev(lin_actions) if len(lin_actions) > 1 else 0.0
    mean_ang_action = statistics.mean(ang_actions) if len(ang_actions) > 0 else 0.0
    ang_actions_std = statistics.stdev(ang_actions) if len(ang_actions) > 1 else 0.0
    
    print(f"avg_step_reward {avg_step_reward}")
    print(f"avg_step_reward_std: {avg_step_reward_std}")
    print(f"avg_ep_reward: {avg_ep_reward}")
    print(f"avg_ep_reward_std: {avg_ep_reward_std}")
    print(f"avg_col: {avg_col}")
    print(f"avg_goal: {avg_goal}")
    print(f"avg_inter_step_rew: {avg_inter_step_rew}")
    print(f"avg_inter_step_rew_std: {avg_inter_step_rew_std}")
    print(f"avg_steps_to_goal: {avg_steps_to_goal}")
    print(f"avg_steps_to_goal_std: {avg_steps_to_goal_std}")
    print(f"mean_lin_action: {mean_lin_action}")
    print(f"lin_actions_std: {lin_actions_std}")
    print(f"mean_ang_action: {mean_ang_action}")
    print(f"ang_actions_std: {ang_actions_std}")
    print("..............................................")
    
    model.writer.add_scalar("test/avg_step_reward", avg_step_reward, epoch)
    model.writer.add_scalar("test/avg_step_reward_std", avg_step_reward_std, epoch)
    model.writer.add_scalar("test/avg_ep_reward", avg_ep_reward, epoch)
    model.writer.add_scalar("test/avg_ep_reward_std", avg_ep_reward_std, epoch)
    model.writer.add_scalar("test/avg_col", avg_col, epoch)
    model.writer.add_scalar("test/avg_goal", avg_goal, epoch)
    model.writer.add_scalar("test/avg_inter_step_rew", avg_inter_step_rew, epoch)
    model.writer.add_scalar(
        "test/avg_inter_step_rew_std", avg_inter_step_rew_std, epoch
    )
    model.writer.add_scalar("test/avg_steps_to_goal", avg_steps_to_goal, epoch)
    model.writer.add_scalar("test/avg_steps_to_goal_std", avg_steps_to_goal_std, epoch)
    model.writer.add_scalar("test/mean_lin_action", mean_lin_action, epoch)
    model.writer.add_scalar("test/lin_actions_std", lin_actions_std, epoch)
    model.writer.add_scalar("test/mean_ang_action", mean_ang_action, epoch)
    model.writer.add_scalar("test/ang_actions_std", ang_actions_std, epoch)
    bins = 100
    model.writer.add_histogram("test/lin_actions", lin_actions, epoch, max_bins=bins)
    model.writer.add_histogram("test/ang_actions", ang_actions, epoch, max_bins=bins)

    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/lin_actions_hist", fig)

    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/ang_actions_hist", fig)


if __name__ == "__main__":
    main()
