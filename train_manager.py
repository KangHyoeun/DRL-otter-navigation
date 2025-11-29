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
from robot_nav.models.PPO.LSTMPPO import LSTMPPO
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
    parser.add_argument("--load_model", action='store_true', help="Load saved model")
    parser.add_argument("--scratch", action='store_true', help="Train from scratch (ignore previous phase model)")
    parser.add_argument("--use_lstm", action='store_true', help="Use LSTM-PPO model")
    args = parser.parse_args()

    # 1. Load Configurations
    root_dir = Path(__file__).parent
    default_config = load_config(root_dir / "configs/default.yaml")
    algo_config = load_config(root_dir / f"configs/{args.algo}.yaml")
    
    config = merge_configs(default_config, algo_config)
    
    # 2. Phase Specific Overrides
    config['phase'] = args.phase
    config['algo'] = args.algo.upper()
    
    prev_phase_model = None # Initialize

    if args.phase == 1:
        config['warmup_steps'] = 10000
        config['worlds'] = config['phase1_worlds']
        config['max_epochs'] = 30 # Phase 1 always scratch/resume current
        # Phase 1 Hyperparameters Override
        config['w_efficiency'] = 3.0  # 효율성 강조
        config['w_safety'] = 0.1      # 안전성 낮게 (장애물 없음)
        config['ts_speed_for_cr'] = 0.0 # 장애물 없음 (CR 계산에 사용)
        
        # Composite Score Weights for Phase 1
        config['W_REWARD'] = 0.1
        config['W_GOAL'] = 100.0
        config['W_COLLISION'] = -200.0
        
        case_id = "00"

    elif args.phase == 2:
        config['worlds'] = config['phase2_worlds']
        case_id = "01"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_00_phase1_BEST"

        if args.scratch:
            config['max_epochs'] = 200
            config['warmup_steps'] = 10000
        elif args.load_model:
            config['max_epochs'] = 100
            config['warmup_steps'] = 2000 # Reduced warmup for resume
        else: # Transfer learning
            config['max_epochs'] = 100
            config['warmup_steps'] = 2000 # Reduced warmup for transfer
            
        # Composite Score Weights for Phase 2 (increased reward influence)
        config['W_REWARD'] = 1.0  # Increased for efficiency/smoothness
        config['W_GOAL'] = 100.0
        config['W_COLLISION'] = -200.0

    elif args.phase == 3:
        config['worlds'] = config['phase3_worlds']
        case_id = "05"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_01_phase2_BEST"
        
        if args.scratch:
            config['max_epochs'] = 300
            config['warmup_steps'] = 20000
        elif args.load_model:
            config['max_epochs'] = 200
            config['warmup_steps'] = 2000
        else: # Transfer learning
            config['max_epochs'] = 200
            config['warmup_steps'] = 2000
            
        # Composite Score Weights for Phase 3
        config['W_REWARD'] = 3.0
        config['W_GOAL'] = 100.0
        config['W_COLLISION'] = -200.0

    elif args.phase == 4:
        config['worlds'] = config['phase4_worlds']
        case_id = "12"
        prev_phase_model = f"otter_MLPCNN{config['algo']}_imazu_05_phase3_BEST"
        
        if args.scratch:
            config['max_epochs'] = 400
            config['warmup_steps'] = 30000
        elif args.load_model:
            config['max_epochs'] = 300
            config['warmup_steps'] = 2000
        else: # Transfer learning
            config['max_epochs'] = 300
            config['warmup_steps'] = 2000
            
        # Composite Score Weights for Phase 4
        config['W_REWARD'] = 10.0
        config['W_GOAL'] = 100.0
        config['W_COLLISION'] = -200.0

    config['lr_decay_epochs'] = config['max_epochs']
    
    # Enhanced Grid Configuration
    # Use enhanced grid (2 channels)
    if args.algo == 'ppo':
        config['use_enhanced_grid'] = True
    else:
        config['use_enhanced_grid'] = False

    # Define Current Model Name
    # Scratch 모드일 경우 이름에 'scratch' 포함 (선택 사항이지만 구분하기 좋음)
    if args.scratch:
        config['model_name'] = f"otter_MLPCNN{config['algo']}_imazu_{case_id}_scratch_phase{args.phase}"
    else:
        config['model_name'] = f"otter_MLPCNN{config['algo']}_imazu_{case_id}_phase{args.phase}"
    
    # Resolve Paths
    config['save_directory'] = config['save_directory'].format(algo=config['algo'])
    config['load_directory'] = config['load_directory'].format(algo=config['algo'])
    config['best_checkpoint_dir'] = config['best_checkpoint_dir'].format(algo=config['algo'])
    
    # 3. Model Initialization
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing {config['algo']} on {device} for Phase {args.phase}")
    
    if args.scratch:
        print("   ✨ Training from SCRATCH (ignoring previous phase)")
    if args.load_model:
        print("   🔄 Resuming training from CURRENT phase checkpoint")

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
                load_model=args.load_model,
                save_directory=Path(config['save_directory']),
                model_name=config['model_name'] + "_LSTM" if args.use_lstm else config['model_name'],
                load_directory=Path(config['load_directory']),
                hidden_size=config.get('hidden_size', 512)
            )
        else:
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
                load_model=args.load_model, # Resume training if True
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
            replay_buffer_capacity=config['replay_buffer_capacity'],
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
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            exploration_noise_init=config.get('exploration_noise_init', 0.1),
            exploration_noise_min=config.get('exploration_noise_min', 0.01),
            exploration_noise_decay=config.get('exploration_noise_decay', 0.9995),
            use_lr_scheduler=config.get('use_lr_scheduler', False),
            lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
            lr_decay_epochs=config.get('lr_decay_epochs', 300),
            lr_min_factor=config.get('lr_min_factor', 0.1),
            lr_decay_rate=config.get('lr_decay_rate', 0.99),
            lr_step_size=config.get('lr_step_size', 100),
            lr_gamma=config.get('lr_gamma', 0.5),
            save_every=config['save_every'],
            load_model=args.load_model,
            save_directory=Path(config['save_directory']),
            model_name=config['model_name'],
            load_directory=Path(config['load_directory']),
            replay_buffer_capacity=config['replay_buffer_capacity'],
        )

    # Handle Transfer Learning (Explicit Load)
    # Only if NOT resuming current training AND NOT scratch AND previous model exists
    if not args.load_model and not args.scratch and prev_phase_model is not None:
        print(f"🔄 Attempting to load previous phase model: {prev_phase_model}")
        prev_model_path = Path(config['best_checkpoint_dir'])
        
        try:
            # Explicitly load the previous phase weights
            model.load(filename=prev_phase_model, directory=prev_model_path)
            print("   ✅ Loaded previous phase model for transfer learning.")
            
            # Note: We do NOT set args.load_model = True.
            # The model is initialized, weights loaded, but we are starting a NEW training session (Phase X).
            # config['load_model'] remains False, so warmup will execute if configured.
            
        except FileNotFoundError:
            print(f"   ⚠️ Previous model '{prev_phase_model}' not found in {prev_model_path}.")
            print("   ⚠️ Starting from scratch instead.")
    
    # 4. Start Training
    if args.algo == 'ppo':
        train_on_policy(config, model, config['worlds'])
    else:
        train_off_policy(config, model, config['worlds'])

if __name__ == "__main__":
    main()