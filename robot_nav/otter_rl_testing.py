import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import statistics
import random
from pathlib import Path

from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.utils.utils import WrapTo180
from colregs_core.geometry import math_to_ned_heading

def main():
    """Test function for Otter USV Imazu Case Collision Avoidance - PHASE 2"""
    
    # Phase 2 Worlds
    phase2_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_01.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_02.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_03.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_04.yaml"
    ]

    action_dim = 2
    max_action = 1
    state_dim = 12
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Test parameters
    max_steps = 512  # Frame Skip adjusted
    test_episodes = 20 # Number of episodes to test
    
    # Model to load
    model_name = "otter_MLPCNNPPO_imazu_01_phase2_BEST"
    
    print("\n" + "=" * 60)
    print("🧪 TESTING PHASE 2: COLLISION AVOIDANCE (1 Target Ship)")
    print("=" * 60)
    print(f"   Model: {model_name}")
    print(f"   Worlds: {len(phase2_worlds)} scenarios")
    print(f"   Max Steps: {max_steps}")
    print("=" * 60)

    # Initialize Model
    model = MLPCNNPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name=model_name,
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )

    # Initialize Simulation (Plotting Enabled)
    # Note: We initialize with one world, but will reset with random worlds
    sim = OtterSIM(
        world_file=phase2_worlds[0],
        disable_plotting=False, # Enable visualization for testing
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

    # Metrics storage
    total_rewards = []
    steps_per_episode = []
    goal_counts = 0
    collision_counts = 0
    
    for ep in range(test_episodes):
        selected_world = random.choice(phase2_worlds)
        print(f"\n▶️  Episode {ep+1}/{test_episodes} | World: {Path(selected_world).name}")
        
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        ep_reward = 0
        step_count = 0
        done = False
        
        # Progress bar for steps within an episode
        pbar = tqdm.tqdm(total=max_steps, desc=f"   Running", leave=False)
        
        while not done and step_count < max_steps:
            # Prepare state
            state, terminal = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            # Get deterministic action (no noise)
            action, _, _ = model.get_action(state, add_noise=False, update_rms=False)
            
            # Scale action
            a_in = [
                (action[0] + 1) * 1.5,   # [0, 3.0] m/s
                action[1] * 0.1745,      # ±10 deg/s
            ]
            
            # Step simulation
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            ep_reward += reward
            step_count += 1
            pbar.update(1)
            
            if collision:
                print(f"   💥 COLLISION! at step {step_count}")
                collision_counts += 1
                done = True
            elif goal:
                print(f"   🏁 GOAL REACHED! at step {step_count}")
                goal_counts += 1
                done = True
                
        pbar.close()
        
        total_rewards.append(ep_reward)
        steps_per_episode.append(step_count)
        print(f"   Reward: {ep_reward:.2f} | Steps: {step_count}")

    # Final Statistics
    avg_reward = statistics.mean(total_rewards) if total_rewards else 0
    avg_steps = statistics.mean(steps_per_episode) if steps_per_episode else 0
    success_rate = (goal_counts / test_episodes) * 100
    collision_rate = (collision_counts / test_episodes) * 100

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Total Episodes: {test_episodes}")
    print(f"   Success Rate:   {success_rate:.1f}%")
    print(f"   Collision Rate: {collision_rate:.1f}%")
    print(f"   Avg Reward:     {avg_reward:.2f}")
    print(f"   Avg Steps:      {avg_steps:.1f}")
    print("=" * 60)
    
    # Keep plots open if needed
    # plt.show()

if __name__ == "__main__":
    main()
