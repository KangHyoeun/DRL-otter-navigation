#!/usr/bin/env python3
"""
CNNSAC 90m Scratch Testing
==========================

Test the trained CNNSAC model in 90m environment.
"""

from robot_nav.models.SAC.CNNSAC import CNNSAC
import statistics
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from pathlib import Path


def main(args=None):
    """Main testing function"""
    
    # Model configuration
    action_dim = 2
    max_action = 1
    state_dim = 369  # 360 LiDAR + 9 other features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Testing parameters
    max_steps = 1000
    test_scenarios = 100
    
    print("=" * 80)
    print("🧪 CNNSAC 90m Scratch Testing")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Test scenarios: {test_scenarios}")
    print(f"Max steps: {max_steps}")
    print("=" * 80)

    # Load model
    model = CNNSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name="otter_CNNSAC_90m_scratch_BEST",
        load_directory=Path("robot_nav/models/SAC/best_checkpoint"),
    )

    # Initialize simulation
    sim = OtterSIM(
        world_file="robot_nav/worlds/imazu_scenario/s1_90m.yaml",
        disable_plotting=True,
        enable_phase1=True,  # ⭐ Phase 1 enabled (same as training)
        max_steps=max_steps
    )

    print(f"\nTesting {test_scenarios} scenarios...")
    
    # Metrics
    total_reward = []
    reward_per_ep = []
    lin_actions = []
    ang_actions = []
    total_steps = 0
    col = 0
    goals = 0
    inter_rew = []
    steps_to_goal = []
    
    for scenario_idx in tqdm.tqdm(range(test_scenarios)):
        count = 0
        ep_reward = 0
        
        # Reset environment
        latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset(
            robot_state=None,
            robot_goal=None,
            random_obstacles=False,
        )
        
        done = False
        
        while not done and count < max_steps:
            # Get state
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a, robot_state
            )
            
            # Get action (deterministic - no noise during testing)
            action = model.get_action(np.array(state), add_noise=False)
            
            # Convert to environment action
            a_in = [
                (action[0] + 1) * 1.5,   # [0, 3.0] m/s
                action[1] * 0.1745       # [-10, 10] deg/s
            ]
            
            lin_actions.append(a_in[0])
            ang_actions.append(a_in[1])
            
            # Step environment
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            ep_reward += reward
            total_reward.append(reward)
            total_steps += 1
            count += 1
            
            # Check terminal conditions
            if collision:
                col += 1
            if goal:
                goals += 1
                steps_to_goal.append(count)
            
            done = collision or goal
            
            if done:
                reward_per_ep.append(ep_reward)
            if not done:
                inter_rew.append(reward)
        
        # If timeout
        if not done and count >= max_steps:
            reward_per_ep.append(ep_reward)

    # Convert to numpy arrays
    total_reward = np.array(total_reward)
    reward_per_ep = np.array(reward_per_ep)
    inter_rew = np.array(inter_rew)
    steps_to_goal = np.array(steps_to_goal)
    lin_actions = np.array(lin_actions)
    ang_actions = np.array(ang_actions)
    
    # Compute statistics
    avg_step_reward = statistics.mean(total_reward) if len(total_reward) > 0 else 0.0
    avg_step_reward_std = statistics.stdev(total_reward) if len(total_reward) > 1 else 0.0
    avg_ep_reward = statistics.mean(reward_per_ep) if len(reward_per_ep) > 0 else 0.0
    avg_ep_reward_std = statistics.stdev(reward_per_ep) if len(reward_per_ep) > 1 else 0.0
    avg_col = col / test_scenarios
    avg_goal = goals / test_scenarios
    avg_inter_step_rew = statistics.mean(inter_rew) if len(inter_rew) > 0 else 0.0
    avg_inter_step_rew_std = statistics.stdev(inter_rew) if len(inter_rew) > 1 else 0.0
    avg_steps_to_goal = int(statistics.mean(steps_to_goal)) if len(steps_to_goal) > 0 else 0
    avg_steps_to_goal_std = statistics.stdev(steps_to_goal) if len(steps_to_goal) > 1 else 0.0
    mean_lin_action = statistics.mean(lin_actions) if len(lin_actions) > 0 else 0.0
    lin_actions_std = statistics.stdev(lin_actions) if len(lin_actions) > 1 else 0.0
    mean_ang_action = statistics.mean(ang_actions) if len(ang_actions) > 0 else 0.0
    ang_actions_std = statistics.stdev(ang_actions) if len(ang_actions) > 1 else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("📊 TESTING RESULTS (90m environment)")
    print("=" * 80)
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
    print("=" * 80)
    
    # Summary
    print("\n📈 SUMMARY:")
    print(f"   Success Rate:        {avg_goal * 100:.1f}%")
    print(f"   Collision Rate:      {avg_col * 100:.1f}%")
    print(f"   Avg Episode Reward:  {avg_ep_reward:.1f} ± {avg_ep_reward_std:.1f}")
    print(f"   Avg Steps to Goal:   {avg_steps_to_goal} ± {avg_steps_to_goal_std:.1f}")
    print(f"   Avg Linear Velocity: {mean_lin_action:.3f} ± {lin_actions_std:.3f} m/s")
    print(f"   Avg Angular Velocity: {mean_ang_action:.4f} ± {ang_actions_std:.4f} rad/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
