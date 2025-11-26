import argparse
import yaml
import torch
import numpy as np
import random
import os
import sys
from pathlib import Path

# Import Models
from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO
from robot_nav.models.SAC.MLPCNNSAC import MLPCNNSAC
from robot_nav.models.TD3.MLPCNNTD3 import MLPCNNTD3

# Import Trainers
from trainers.on_policy import train_on_policy
from trainers.off_policy import train_off_policy

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
    parser = argparse.ArgumentParser(description="Otter USV Training Manager")
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'sac', 'td3'], help="Algorithm to use")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4], help="Curriculum phase")
    parser.add_argument("--load_model", action='store_true', help="Load model from checkpoint")
    args = parser.parse_args()

    # 1. Load Configurations
    root_dir = Path(__file__).parent
    default_config = load_config(root_dir / "configs/default.yaml")
    algo_config = load_config(root_dir / f"configs/{args.algo}.yaml")
    
    config = merge_configs(default_config, algo_config)
    
    # 2. Phase Specific Overrides
    config['phase'] = args.phase
    config['algo'] = args.algo.upper()
    
    if args.phase == 1:
        config['warmup_steps'] = 10000
        config['worlds'] = config['phase1_worlds']
        case_id = "00"
        prev_phase_model = None # No previous model for phase 0
        # Phase 1 Hyperparameters Override
        config['w_efficiency'] = 3.0  # 효율성 강조
        config['w_safety'] = 0.1      # 안전성 낮게 (장애물 없음)
        config['ts_speed_for_cr'] = 0.0 # 장애물 없음

    elif args.phase == 2:
        config['warmup_steps'] = 10000
        config['worlds'] = config['phase2_worlds']
        case_id = "01"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_00_scratch_BEST" # Phase 1 model

    elif args.phase == 3:
        config['warmup_steps'] = 20000
        config['worlds'] = config['phase3_worlds']
        case_id = "05"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_01_phase2_BEST"

    elif args.phase == 4:
        config['warmup_steps'] = 30000
        config['worlds'] = config['phase4_worlds']
        case_id = "12"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_05_phase3_BEST"

    config['model_name'] = f"otter_MLPCNN{config['algo']}_imazu_{case_id}_phase{args.phase}"
    
    # Resolve Paths
    config['save_directory'] = config['save_directory'].format(algo=config['algo'])
    config['load_directory'] = config['load_directory'].format(algo=config['algo'])
    config['best_checkpoint_dir'] = config['best_checkpoint_dir'].format(algo=config['algo'])
    
    # 3. Model Initialization
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing {config['algo']} on {device} for Phase {args.phase}")
    
    if args.algo == 'ppo':
        model = MLPCNNPPO(
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
            load_model=args.load_model, # Use arg to force load
            save_directory=Path(config['save_directory']),
            model_name=config['model_name'],
            load_directory=Path(config['load_directory']),
        )
    elif args.algo == 'sac':
        model = MLPCNNSAC(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            discount=config['discount'],
            init_temperature=config['init_temperature'],
            alpha_lr=config['alpha_lr'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            critic_tau=config['critic_tau'],
            actor_update_frequency=config['actor_update_frequency'],
            critic_target_update_frequency=config['critic_target_update_frequency'],
            save_every=config['save_every'],
            load_model=args.load_model,
            save_directory=Path(config['save_directory']),
            model_name=config['model_name'],
            load_directory=Path(config['load_directory']),
            replay_buffer_capacity=config['replay_buffer_capacity'], # Added
        )
    elif args.algo == 'td3':
        model = MLPCNNTD3(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            discount=config['discount'],
            tau=config['tau'],
            policy_noise=config['policy_noise'],
            noise_clip=config['noise_clip'],
            policy_freq=config['policy_freq'],
            lr=config['lr'],
            save_every=config['save_every'],
            load_model=args.load_model,
            save_directory=Path(config['save_directory']),
            model_name=config['model_name'],
            load_directory=Path(config['load_directory']),
            replay_buffer_capacity=config['replay_buffer_capacity'], # Added
        )

    
    if not args.load_model and prev_phase_model is not None:
        print(f"🔄 Attempting to load previous phase model: {prev_phase_model}")
        try:
            model.load(filename=prev_phase_model, directory=Path(config['best_checkpoint_dir']))
            print("   ✅ Loaded previous phase model for transfer learning.")
            config['load_model'] = True
        except FileNotFoundError:
            print(f"   ⚠️ Previous model {prev_phase_model} not found. Starting from scratch (or random init).")
    
    # 4. Start Training
    if args.algo == 'ppo':
        train_on_policy(config, model, config['worlds'])
    else:
        train_off_policy(config, model, config['worlds'])

if __name__ == "__main__":
    main()
