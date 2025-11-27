import argparse
import yaml
import torch
import numpy as np
import random
import os
import sys
import tqdm
import statistics
import matplotlib.pyplot as plt
from pathlib import Path
from colregs_core.geometry import math_to_ned_heading, heading_speed_to_velocity, math_to_maritime_position

# Import Models
from robot_nav.models.SAC.MLPCNNSAC import MLPCNNSAC
from robot_nav.baselines.orca_agent import ORCAAgent

# Import Environment
from robot_nav.SIM_ENV.otter_sim import OtterSIM

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(default, override):
    config = default.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in config:
            config[k] = merge_configs(config[k], v)
        else:
            config[k] = v
    return config

def run_test(agent, sim, config, agent_type="DRL", test_scenarios=50):
    print(f"\n🔹 Running {agent_type} Agent ({test_scenarios} episodes)...")
    
    total_rewards = []
    collisions = 0
    goals = 0
    steps_list = []
    path_lengths = []
    
    # Fix seed for reproducibility across agents
    # We will use a list of seeds
    seeds = list(range(1000, 1000 + test_scenarios))

    for i in tqdm.tqdm(range(test_scenarios)):
        seed = seeds[i]
        random.seed(seed)
        np.random.seed(seed)
        
        selected_world = config['worlds'][i % len(config['worlds'])]
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        ep_reward = 0
        path_len = 0.0
        prev_pos = sim.start_position
        
        count = 0
        done = False
        
        while not done and count < sim.max_steps:
            # 1. Get Action
            if agent_type == "DRL":
                state, _ = agent.prepare_state(
                    distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
                )
                action, _, _ = agent.get_action(state, add_noise=False, update_rms=False)
                
            elif agent_type == "ORCA":
                # Construct Observation Dict for ORCA
                # Need to extract dynamic obstacles from env
                obstacles_data = []
                for obs in sim.env.obstacle_list:
                    obs_state = obs.state # [x, y, theta, v, w]
                    # Convert to Maritime [N, E], [vn, ve]
                    om_pos_y, om_pos_x = math_to_maritime_position(obs_state[0,0], obs_state[1,0])
                    om_head = math_to_ned_heading(np.degrees(obs_state[2,0]))
                    om_spd = np.linalg.norm([obs_state[3,0], obs_state[4,0]])
                    om_vel = heading_speed_to_velocity(om_head, om_spd)
                    
                    obstacles_data.append({
                        'pos': [om_pos_y, om_pos_x],
                        'vel': om_vel,
                        'radius': 10.0 # Assumption for safety radius
                    })
                
                # OS State
                # robot_state is [x, y, theta, u, v, r]
                os_pos_math = [robot_state[0,0], robot_state[1,0]]
                os_pos = list(math_to_maritime_position(os_pos_math[0], os_pos_math[1]))
                os_head = math_to_ned_heading(np.degrees(robot_state[2,0]))
                os_spd = np.linalg.norm([robot_state[3,0], robot_state[4,0]])
                os_vel = heading_speed_to_velocity(os_head, os_spd)
                
                obs_dict = {
                    'os_pos': os_pos,
                    'os_vel': os_vel,
                    'os_heading': np.radians(os_head),
                    'goal_pos': sim.goal_position,
                    'obstacles': obstacles_data
                }
                
                action, _, _ = agent.get_action(obs_dict)

            # 2. Scale Action (Same for both)
            # Normalized [-1, 1] -> Physical
            a_in = [
                (action[0] + 1) * 1.5, 
                action[1] * 0.1745
            ]
            
            # 3. Step
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            # Track Metrics
            ep_reward += reward
            
            # Calculate path length
            curr_pos_math = [robot_state[0,0], robot_state[1,0]]
            curr_pos = math_to_maritime_position(curr_pos_math[0], curr_pos_math[1])
            dist_step = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            path_len += dist_step
            prev_pos = curr_pos
            
            count += 1
            
            if collision:
                collisions += 1
            if goal:
                goals += 1
                steps_list.append(count)
                path_lengths.append(path_len)
                
            done = collision or goal
            
        if not done: # Timeout
            pass
            
        total_rewards.append(ep_reward)

    # Stats
    avg_reward = statistics.mean(total_rewards)
    success_rate = goals / test_scenarios
    collision_rate = collisions / test_scenarios
    avg_steps = statistics.mean(steps_list) if steps_list else sim.max_steps
    avg_path = statistics.mean(path_lengths) if path_lengths else 0.0
    
    return {
        "Avg Reward": avg_reward,
        "Success Rate": success_rate,
        "Collision Rate": collision_rate,
        "Avg Steps": avg_steps,
        "Avg Path Len": avg_path
    }

def main():
    parser = argparse.ArgumentParser(description="DRL vs ORCA Comparison")
    parser.add_argument("--phase", type=int, required=True, choices=[3, 4], help="Phase to compare")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per agent")
    parser.add_argument("--model_tag", type=str, default="BEST", help="DRL Model tag")
    args = parser.parse_args()

    # Load Config
    root_dir = Path(__file__).parent
    default_config = load_config(root_dir / "configs/default.yaml")
    sac_config = load_config(root_dir / "configs/sac.yaml")
    config = merge_configs(default_config, sac_config)
    
    config['phase'] = args.phase
    if args.phase == 3:
        config['worlds'] = config['phase3_worlds']
        case_id = "05"
    elif args.phase == 4:
        config['worlds'] = config['phase4_worlds']
        case_id = "12"
        
    # 1. Load DRL Agent (SAC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"otter_MLPCNNSAC_imazu_{case_id}_phase{args.phase}_{args.model_tag}"
    load_dir = Path(config['best_checkpoint_dir'].format(algo="SAC"))
    
    drl_agent = MLPCNNSAC(
        state_dim=config['state_dim'], action_dim=config['action_dim'], max_action=config['max_action'],
        device=device, load_model=True, model_name=model_name, load_directory=load_dir,
        replay_buffer_capacity=1000
    )
    
    # 2. Load ORCA Agent
    orca_agent = ORCAAgent(
        time_step=1.0, # Match action_dt of OtterSIM (1.0s)
        max_speed=3.0,
        neighbor_dist=300.0, # Look further
        time_horizon=10.0,
        radius=15.0 # Conservative radius
    )
    
    # 3. Setup Sim
    sim = OtterSIM(
        world_file=config['worlds'][0],
        disable_plotting=True, enable_phase1=True, max_steps=512,
        cr_method='jeon', w_efficiency=1.0, w_safety=1.0,
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=1.0, k=1.0
    )
    
    # 4. Run Tests
    print(f"⚔️  COMPARISON BATTLE: Phase {args.phase} ⚔️")
    print(f"   DRL Model: {model_name}")
    
    drl_stats = run_test(drl_agent, sim, config, "DRL", args.episodes)
    orca_stats = run_test(orca_agent, sim, config, "ORCA", args.episodes)
    
    # 5. Print Results
    print("\n" + "="*60)
    print(f"{ 'Metric':<20} | {'DRL (SAC)':<15} | {'ORCA (RVO2)':<15}")
    print("-" * 60)
    
    metrics = ["Success Rate", "Collision Rate", "Avg Reward", "Avg Steps", "Avg Path Len"]
    
    for m in metrics:
        drl_val = drl_stats[m]
        orca_val = orca_stats[m]
        
        # Format
        if "Rate" in m:
            d_str = f"{drl_val*100:.1f}%"
            o_str = f"{orca_val*100:.1f}%"
        else:
            d_str = f"{drl_val:.2f}"
            o_str = f"{orca_val:.2f}"
            
        print(f"{m:<20} | {d_str:<15} | {o_str:<15}")
    print("="*60)

if __name__ == "__main__":
    main()
