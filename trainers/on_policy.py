import torch
import numpy as np
import random
import time
import json
from datetime import datetime # Added datetime
from pathlib import Path
from tqdm import tqdm
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from robot_nav.utils import prepare_multi_modal_state

def train_on_policy(config, model, worlds):
    """
    Generic On-Policy Training Loop (PPO)
    """
    print(f"🚀 Starting On-Policy Training (PPO) - Phase {config.get('phase', '?')}")
    
    # Unpack Config
    max_epochs = config['max_epochs']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    training_iterations = config['training_iterations']
    train_every_n_episodes = config['train_every_n_episodes']
    episodes_per_epoch = config['episodes_per_epoch']
    nr_eval_episodes = config['nr_eval_episodes']
    save_every = config['save_every']
    warmup_steps = config['warmup_steps']
    
    chi_inf = config['chi_inf']
    k = config['k']
    
    model_name = config['model_name']
    save_dir = Path(config['save_directory'])
    best_checkpoint_dir = Path(config['best_checkpoint_dir'])
    
    # Initialize Sim
    sim = OtterSIM(
        world_file=random.choice(worlds),
        disable_plotting=True, 
        enable_phase1=True, 
        max_steps=max_steps,
        cr_method='jeon', 
        w_efficiency=config['w_efficiency'], 
        w_safety=config['w_safety'],
        os_speed_for_cr=config['os_speed_for_cr'], 
        ts_speed_for_cr=config['ts_speed_for_cr'],
        chi_inf=chi_inf, 
        k=k,
        use_enhanced_grid=config.get('use_enhanced_grid', False)
    )

    # RMS Warmup
    if not config['load_model']:
        print(f"\n🔥 Starting RMS Warmup ({warmup_steps} steps)...")
        w_sim = OtterSIM(
            world_file=random.choice(worlds),
            disable_plotting=True, enable_phase1=True, max_steps=max_steps,
            cr_method='jeon', w_efficiency=config['w_efficiency'], w_safety=config['w_safety'],
            os_speed_for_cr=config['os_speed_for_cr'], ts_speed_for_cr=config['ts_speed_for_cr'],
            chi_inf=chi_inf, k=k,
            use_enhanced_grid=config.get('use_enhanced_grid', False)
        )
        w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset()
        
        for _ in tqdm(range(warmup_steps), desc="Warmup"):
            s_w, _ = prepare_multi_modal_state(
                w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_state, w_cr, grid_map=w_grid
            )

            model.get_action(s_w, add_noise=True, update_rms=True)
            
            # Random action for exploration (Raw [-1, 1])
            w_action_raw = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
            
            # Scale to physical
            w_u = (w_action_raw[0] + 1.0) * 1.5
            w_r = w_action_raw[1] * 0.1745
            
            w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.step(u_ref=w_u, r_ref=w_r)
            
            if w_col or w_goal:
                w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset(world_file=random.choice(worlds))
        
        print("✅ RMS Warmup Complete.")
        print(f"   Vector RMS Mean: {model.obs_rms.mean[:3]}...")

    # Main Loop
    episode_start_time = time.time()
    patience = 100000
    patience_counter = 0
    best_reward = -np.inf
    best_goal_rate = 0.0
    
    epoch = 0
    episode_count = 0
    episode_count_since_last_train = 0
    
    distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=random.choice(worlds))
    steps = 0
    
    # Initialize Hidden State for LSTM (if applicable)
    hidden = None
    if hasattr(model, 'hidden_size'):
        h = torch.zeros(1, 1, model.hidden_size).to(model.device)
        c = torch.zeros(1, 1, model.hidden_size).to(model.device)
        hidden = (h, c)
    
    while epoch < max_epochs:
        state, terminal = prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        
        # Get Action (Handle LSTM)
        if hidden is not None:
            current_hidden = hidden # Store for buffer
            action, log_prob, state_val, next_hidden = model.get_action(state, hidden, add_noise=True)
            hidden = next_hidden
        else:
            action, log_prob, state_val = model.get_action(state, add_noise=True)

        a_in = [
            (action[0] + 1) * 1.5,
            action[1] * 0.1745, 
        ]

        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )

        next_state, terminal = prepare_multi_modal_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        
        if hidden is not None:
            model.buffer.add(
                state, action, reward, terminal, next_state, log_prob, state_val, current_hidden
            )
        else:
            model.buffer.add(
                state, action, reward, terminal, next_state, log_prob, state_val
            )

        steps += 1

        # Log reward components
        if hasattr(sim, 'latest_rewards') and sim.latest_rewards and steps % 100 == 0:
            model.writer.add_scalar("train/r_risk", sim.latest_rewards.get('r_risk', 0), episode_count * max_steps + steps)
            model.writer.add_scalar("train/r_colregs", sim.latest_rewards.get('r_colregs', 0), episode_count * max_steps + steps)
            model.writer.add_scalar("train/r_total", sim.latest_rewards.get('r_total', 0), episode_count * max_steps + steps)

        if terminal or steps >= max_steps:
            episode_time = time.time() - episode_start_time
            print(f"📊 Episode {episode_count + 1} completed in {episode_time:.2f}s ({steps} steps)")
            episode_count += 1
            episode_count_since_last_train += 1
            episode_start_time = time.time()
            
            if episode_count_since_last_train >= train_every_n_episodes:
                print(f"\n🔄 Training on {episode_count_since_last_train} episodes of experience...")
                model.train(None, training_iterations, batch_size)
                episode_count_since_last_train = 0
            
            if episode_count % episodes_per_epoch == 0:
                epoch += 1
                avg_reward, avg_goal, avg_col = evaluate(model, epoch, worlds, nr_eval_episodes, max_steps, config)
                
                current_std = torch.exp(model.policy.log_std).mean().item()
                
                # Composite Score
                W_REWARD = 0.1
                W_GOAL = 100.0
                W_COLLISION = -200.0
                
                normalized_avg_reward = np.clip((avg_reward + 50.0) / 100.0, 0.0, 1.0)
                composite_score = (normalized_avg_reward * W_REWARD + avg_goal * W_GOAL + avg_col * W_COLLISION)
                
                model.writer.add_scalar("eval/composite_score", composite_score, epoch)

                if composite_score > best_reward:
                    print("=" * 60)
                    print(f"🎉 NEW BEST MODEL! (Composite Score: {composite_score:.1f})")
                    print(f"   Goal: {avg_goal*100:.1f}%, Col: {avg_col*100:.1f}%")
                    print("=" * 60)

                    best_reward = composite_score
                    best_goal_rate = avg_goal
                    patience_counter = 0
                    model.save(filename=f"{model_name}_BEST", directory=best_checkpoint_dir)

                    metrics = {
                        "epoch": epoch,
                        "avg_reward": float(avg_reward),
                        "goal_rate": float(avg_goal),
                        "collision_rate": float(avg_col),
                        "action_std": float(current_std),
                        "composite_score": float(composite_score),
                        "phase": config.get('phase', 'unknown'),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Added timestamp
                        "config": config # Added full config
                    }
                    with open(best_checkpoint_dir / f"best_metrics_{model_name}.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement. Best: {best_reward:.1f}, Current: {composite_score:.1f}")

                if patience_counter >= patience:
                    print("🛑 EARLY STOPPING!")
                    break
            
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=random.choice(worlds))
            steps = 0
            
            # Reset Hidden State
            if hasattr(model, 'hidden_size'):
                h = torch.zeros(1, 1, model.hidden_size).to(model.device)
                c = torch.zeros(1, 1, model.hidden_size).to(model.device)
                hidden = (h, c)

    print("✅ TRAINING COMPLETED!")


def evaluate(model, epoch, world_files, eval_episodes, max_steps, config):
    print(f"\n📈 EVALUATION | Epoch {epoch}")
    
    total_rewards = []
    total_steps = []
    goal_reached_count = 0
    collision_count = 0

    sim = OtterSIM(
        world_file=world_files[0],
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=config['w_efficiency'], w_safety=config['w_safety'],
        os_speed_for_cr=config['os_speed_for_cr'], ts_speed_for_cr=config['ts_speed_for_cr'],
        chi_inf=config['chi_inf'], k=config['k'],
        use_enhanced_grid=config.get('use_enhanced_grid', False)
    )

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        selected_world = random.choice(world_files)
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        # Init Hidden
        hidden = None
        if hasattr(model, 'hidden_size'):
            h = torch.zeros(1, 1, model.device).to(model.device) # Wait, model.device is string or device? Usually device.
            # Correction: model.device might be 'cuda' string.
            # Safe way:
            h = torch.zeros(1, 1, model.hidden_size).to(model.policy.device)
            c = torch.zeros(1, 1, model.hidden_size).to(model.policy.device)
            hidden = (h, c)
        
        ep_reward = 0
        for s in range(max_steps):
            state, terminal = prepare_multi_modal_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            if hidden is not None:
                action_np, _, _, next_hidden = model.get_action(state, hidden, add_noise=False, update_rms=False)
                hidden = next_hidden
            else:
                action_np, _, _ = model.get_action(state, add_noise=False, update_rms=False)
            a_in = [
                (action_np[0] + 1) * 1.5,
                action_np[1] * 0.1745, 
            ]
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(u_ref=a_in[0], r_ref=a_in[1])
            ep_reward += reward
            if collision or goal:
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(s + 1)
        if goal: goal_reached_count += 1
        if collision: collision_count += 1

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    goal_rate = goal_reached_count / eval_episodes
    collision_rate = collision_count / eval_episodes

    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    model.writer.add_scalar("eval/goal_rate", goal_rate, epoch)
    model.writer.add_scalar("eval/collision_rate", collision_rate, epoch)

    print(f"📊 Eval Results: Reward: {avg_reward:.2f} | Goal: {goal_rate:.2f} | Col: {collision_rate:.2f}")
    return avg_reward, goal_rate, collision_rate
