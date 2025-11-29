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
from torch.utils.tensorboard import SummaryWriter

# Import Models
from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO
from robot_nav.models.PPO.LSTMPPO import LSTMPPO
from robot_nav.models.SAC.MLPCNNSAC import MLPCNNSAC
from robot_nav.models.TD3.MLPCNNTD3 import MLPCNNTD3

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

def main():
    parser = argparse.ArgumentParser(description="Otter USV Test Manager")
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'sac', 'td3'], help="Algorithm to use")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4], help="Curriculum phase")
    parser.add_argument("--test_scenarios", type=int, default=100, help="Number of test scenarios")
    parser.add_argument("--render", action='store_true', help="Enable rendering")
    parser.add_argument("--model_path", type=str, default=None, help="Specific model path to load (optional)")
    parser.add_argument("--model_tag", type=str, default="BEST", help="Tag for the model to load (e.g., 'BEST', 'LAST', '100')") # Added
    parser.add_argument("--use_lstm", action='store_true', help="Use LSTM-PPO model")
    args = parser.parse_args()

    # 1. Load Configurations
    root_dir = Path(__file__).parent
    default_config = load_config(root_dir / "configs/default.yaml")
    algo_config = load_config(root_dir / f"configs/{args.algo}.yaml")
    config = merge_configs(default_config, algo_config)
    config['algo'] = args.algo.upper()
    
    # 2. Phase Specific Setup
    config['phase'] = args.phase
    
    if args.phase == 1:
        config['worlds'] = config['phase1_worlds']
        case_id = "00"
        # Phase 1 Hyperparameters Override (same as train_manager)
        config['w_efficiency'] = 3.0
        config['w_safety'] = 0.1
        config['ts_speed_for_cr'] = 0.0
    elif args.phase == 2:
        config['worlds'] = config['phase2_worlds']
        case_id = "01"
    elif args.phase == 3:
        config['worlds'] = config['phase3_worlds']
        case_id = "05"
    elif args.phase == 4:
        config['worlds'] = config['phase4_worlds']
        case_id = "12"

    # Determine Model Name and Path
    if args.model_path:
        model_name = Path(args.model_path).stem
        # Remove extension if present in stem (stem usually removes extension, but safe check)
        # Assuming filename format: "name.pth" -> stem is "name"
        # The model classes expect filename WITHOUT extension
        if model_name.endswith("_actor"): model_name = model_name.replace("_actor", "")
        if model_name.endswith("_critic"): model_name = model_name.replace("_critic", "")
        
        load_directory = Path(args.model_path).parent
    else:
        # Use model_tag to construct filename
        model_name = f"otter_MLPCNN{config['algo']}_imazu_{case_id}_phase{args.phase}_{args.model_tag}"
        # We assume models are in the best_checkpoint_dir (which might be generic)
        # If you want to load from a specific run folder, use --model_path
        load_directory = Path(config['best_checkpoint_dir'].format(algo=config['algo']))

    print("..............................................")
    print(f"🚀 Starting Test: {config['algo']} Phase {args.phase}")
    print(f"   Model: {model_name}")
    print(f"   Loading from: {load_directory}")
    print(f"   Scenarios: {args.test_scenarios}")
    print("..............................................")

    # 3. Model Initialization
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model = None

    if args.algo == 'ppo':
        if args.use_lstm:
            model = LSTMPPO(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                max_action=config['max_action'],
                lr_actor=config['lr_actor'],
                lr_critic=config['lr_critic'],
                gamma=config['gamma'],
                eps_clip=config['eps_clip'],
                log_std_init=config['log_std_init'],
                ent_coef_init=config['ent_coef_init'],
                ent_coef_decay_rate=config['ent_coef_decay_rate'],
                min_ent_coef=config['min_ent_coef'],
                target_kl=config['target_kl'],
                device=device,
                save_every=config['save_every'],
                load_model=True, # Always load for test
                save_directory=Path(config['save_directory']),
                model_name=model_name, # Use constructed name
                load_directory=load_directory,
                hidden_size=config.get('hidden_size', 512)
            )
        else:
            model = MLPCNNPPO(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                max_action=config['max_action'],
                device=device,
                load_model=True,
                model_name=model_name,
                load_directory=load_directory,
            )
    elif args.algo == 'sac':
        model = MLPCNNSAC(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            load_model=True,
            model_name=model_name,
            load_directory=load_directory,
            replay_buffer_capacity=1000, # Minimal buffer for testing
        )
    elif args.algo == 'td3':
        model = MLPCNNTD3(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            load_model=True,
            model_name=model_name,
            load_directory=load_directory,
            replay_buffer_capacity=1000, # Minimal buffer for testing
        )
    
    # 4. Simulation Initialization
    # Use the first world for initialization, but reset with random worlds later
    init_world = config['worlds'][0]
    
    sim = OtterSIM(
        world_file=init_world,
        disable_plotting=not args.render,
        enable_phase1=True, # Always enable phase 1 physics for consistency
        max_steps=512, # Standard max steps
        cr_method='jeon',
        w_efficiency=config.get('w_efficiency', 1.0),
        w_safety=config.get('w_safety', 1.0),
        os_speed_for_cr=config.get('os_speed_for_cr', 3.0),
        ts_speed_for_cr=config.get('ts_speed_for_cr', 3.0),
        chi_inf=config.get('chi_inf', 0.5), 
        k=config.get('k', 1.0),
        use_enhanced_grid=args.use_lstm
    )

    # 5. Test Loop
    total_reward = []
    reward_per_ep = []
    lin_actions = []
    ang_actions = []
    col = 0
    goals = 0
    steps_to_goal = []
    
    # TensorBoard for Test Results
    # Use a separate test run directory
    test_run_dir = Path(f"runs/test_{config['algo']}_phase{args.phase}")
    writer = SummaryWriter(log_dir=test_run_dir)

    for i in tqdm.tqdm(range(args.test_scenarios)):
        selected_world = random.choice(config['worlds'])
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        count = 0
        ep_reward = 0
        done = False
        # Initialize Hidden State for LSTM
        hidden = None
        
        while not done and count < sim.max_steps:
            steps = count # Renamed from `count` to `steps` for clarity, matching snippet's `steps`
            current_time = steps * sim.dt
            
            # DRL Action Generation
            state, terminal = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            # Get deterministic action
            if args.algo == 'ppo':
                if args.use_lstm:
                    action, _, _, hidden = model.get_action(state, hidden, add_noise=False)
                else:
                    action, _, _, _ = model.get_action(state, add_noise=False)
            elif args.algo == 'sac':
                action, _, _ = model.get_action(state, add_noise=False)
            elif args.algo == 'td3':
                action, _, _ = model.get_action(state, add_noise=False)
            
            # Action Scaling (Same as training)
            # Raw action is [-1, 1]
            # Surge: [-1, 1] -> [0, 3] => (a+1)*1.5
            # Yaw: [-1, 1] -> [-10deg, 10deg] => a * 0.1745
            a_in = [
                (action[0] + 1) * 1.5, 
                action[1] * 0.1745
            ]
            
            lin_actions.append(a_in[0])
            ang_actions.append(a_in[1])
            
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            if args.render:
                sim.render(
                    u_ref=a_in[0], 
                    r_ref=a_in[1], 
                    reward=reward, 
                    grid_map=cr_grid[0] if isinstance(cr_grid, list) or (isinstance(cr_grid, np.ndarray) and cr_grid.ndim == 3) else cr_grid
                )
            
            ep_reward += reward
            total_reward.append(reward)
            count += 1
            
            if collision:
                col += 1
            if goal:
                goals += 1
                steps_to_goal.append(count)
            done = collision or goal
            
            if done:
                reward_per_ep.append(ep_reward)
        
        if not done and count >= sim.max_steps:
            reward_per_ep.append(ep_reward)

    # 6. Calculate Statistics
    total_reward = np.array(total_reward)
    reward_per_ep = np.array(reward_per_ep)
    steps_to_goal = np.array(steps_to_goal)
    lin_actions = np.array(lin_actions)
    ang_actions = np.array(ang_actions)
    
    avg_step_reward = statistics.mean(total_reward) if len(total_reward) > 0 else 0.0
    avg_ep_reward = statistics.mean(reward_per_ep) if len(reward_per_ep) > 0 else 0.0
    avg_col = col / args.test_scenarios
    avg_goal = goals / args.test_scenarios
    avg_steps_to_goal = statistics.mean(steps_to_goal) if len(steps_to_goal) > 0 else 0.0
    mean_lin_action = statistics.mean(lin_actions) if len(lin_actions) > 0 else 0.0
    mean_ang_action = statistics.mean(ang_actions) if len(ang_actions) > 0 else 0.0
    
    print("\n" + "="*30)
    print(f"📊 Test Results ({args.test_scenarios} episodes)")
    print("="*30)
    print(f"Avg Step Reward: {avg_step_reward:.4f}")
    print(f"Avg Ep Reward:   {avg_ep_reward:.4f}")
    print(f"Collision Rate:  {avg_col:.2%}")
    print(f"Goal Rate:       {avg_goal:.2%}")
    print(f"Avg Steps to Goal: {avg_steps_to_goal:.1f}")
    print(f"Mean Surge Cmd:  {mean_lin_action:.2f} m/s")
    print(f"Mean Yaw Cmd:    {mean_ang_action:.4f} rad/s")
    print("="*30)
    
    # Log to TensorBoard
    writer.add_scalar("test/avg_step_reward", avg_step_reward, 0)
    writer.add_scalar("test/avg_ep_reward", avg_ep_reward, 0)
    writer.add_scalar("test/collision_rate", avg_col, 0)
    writer.add_scalar("test/goal_rate", avg_goal, 0)
    writer.add_scalar("test/avg_steps_to_goal", avg_steps_to_goal, 0)
    
    # Histograms
    bins = 100
    writer.add_histogram("test/lin_actions", lin_actions, 0, max_bins=bins)
    writer.add_histogram("test/ang_actions", ang_actions, 0, max_bins=bins)

    # Figure Histograms
    fig, ax = plt.subplots()
    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True)
    ax.set_title("Surge Action Distribution (Log Scale)")
    writer.add_figure("test/lin_actions_hist", fig, 0)
    
    fig, ax = plt.subplots()
    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True)
    ax.set_title("Yaw Action Distribution (Log Scale)")
    writer.add_figure("test/ang_actions_hist", fig, 0)
    
    writer.close()
    print(f"📝 Logs saved to {test_run_dir}")

if __name__ == "__main__":
    main()
