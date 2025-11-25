import torch
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend for visualization
import matplotlib.pyplot as plt # Import after backend setting
plt.ion() # Enable interactive mode for animation

# CHANGED: Import Multi-Modal PPO
from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.utils.utils import WrapTo180
from colregs_core.geometry import math_to_ned_heading

def main():
    """Main training function for Otter USV Imazu Case Collision Avoidance - PHASE 2"""

    # Fix world file to head-on for specific debug
    phase2_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_01.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_02.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_03.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_04.yaml"
    ]
    
    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 12
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"🚀 CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   Using CPU (slower training)")
    
    # Training parameters
    nr_eval_episodes = 10
    max_epochs = 4000
    episodes_per_epoch = 40
    train_every_n_episodes = 10
    training_iterations = 40
    
    # OPTIMIZED HYPERPARAMETERS
    batch_size = 2048 # Increased for stability with CNN
    max_steps = 512   # Simulation max steps (Frame Skip 10x -> 512s duration)

    save_every = 10
    load_model = False
    
    # PPO specific parameters
    lr_actor = 0.0001 # Reduced for stability
    lr_critic = 0.0003
    gamma = 0.99      # Adjusted
    eps_clip = 0.2
    log_std_init = -0.5
    ent_coef_init = 0.01
    ent_coef_decay_rate = 0.0
    min_ent_coef = 0.001
    target_kl = None  # CHANGED: Disable KL early stopping to force updates
    
    chi_inf = 1.0
    k = 1.0
    
    # Model names
    phase1_model_name = "otter_MLPCNNPPO_imazu_00_scratch_BEST"
    phase2_model_name = "otter_MLPCNNPPO_imazu_01_phase2"
    
    print("\n" + "=" * 60)
    print("🎯 CURRICULUM PHASE 2: COLLISION AVOIDANCE (1 Target Ship)")
    print("=" * 60)
    print(f"   Environments: {len(phase2_worlds)} random scenarios (cases 01-04)")
    print(f"   Model Type: Multi-Modal (Vector + CNN Grid)")
    print("   Optimized Hyperparameters Applied")
    print("=" * 60)
    
    model = MLPCNNPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        eps_clip=eps_clip,
        log_std_init=log_std_init,
        ent_coef_init=ent_coef_init,
        ent_coef_decay_rate=ent_coef_decay_rate,
        min_ent_coef=min_ent_coef,
        target_kl=target_kl,
        device=device,
        save_every=save_every,
        load_model=load_model,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name=phase2_model_name,
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )

    if load_model:
        print(f"\n🔄 Loading pre-trained model from Phase 1: {phase1_model_name}")
        try:
            model.load(filename=phase1_model_name, directory=Path("robot_nav/models/PPO/best_checkpoint"))
            print("   ✅ Model loaded successfully.")
        except FileNotFoundError:
            print(f"   ❌ ERROR: Model file not found. Starting from scratch.")

    # Performance monitoring
    import time
    episode_start_time = time.time()
            
    # Early stopping parameters
    patience = 100000
    patience_counter = 0
    best_reward = -np.inf
    best_goal_rate = 0.0 # Initialize best_goal_rate
    
    epoch = 0
    episode_count = 0
    episode_count_since_last_train = 0
    
    sim = OtterSIM(
        world_file=random.choice(phase2_worlds),
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=1.0, w_safety=3.0,  # 안전 우선!
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=chi_inf, k=k
    )
    
    if not load_model:
        print("\n🔥 Starting RMS Warmup (10000 steps)...")
        print("   Collecting random experiences to stabilize RunningMeanStd.")
        
        warmup_steps = 10000  # 더 많은 warmup
        w_steps = 0
        w_sim = OtterSIM(
            world_file=random.choice(phase2_worlds),
            disable_plotting=True, enable_phase1=True, max_steps=max_steps,
            cr_method='jeon', w_efficiency=1.0, w_safety=3.0,  # 안전 우선!
            os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
            chi_inf=chi_inf, k=k
        )
        w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset()

        for _ in tqdm(range(warmup_steps), desc="Warmup"):
            # Prepare state (updates RMS)
            s_w, _ = model.prepare_state(
                w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_state, w_cr, grid_map=w_grid
            )
            # Update RMS without using the action
            model.get_action(s_w, add_noise=True, update_rms=True)
            
            # Random action for exploration
            w_u = random.uniform(0.0, 3.0)
            w_r = random.uniform(-10.0, 10.0) * 0.01745
            
            w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.step(u_ref=w_u, r_ref=w_r)
            
            w_steps += 1
            if w_col or w_goal:
                w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset(world_file=random.choice(phase2_worlds))
        
        print("✅ RMS Warmup Complete. Mean/Var stabilized.")
        print(f"   Vector RMS Mean: {model.obs_rms.mean[:3]}...")
        print("=" * 60)

    # Re-initialize sim after warmup just in case
    distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=random.choice(phase2_worlds))
    steps = 0
    
    # Main training loop
    while epoch < max_epochs:
        # ... (Prepare state, get action, step logic remains same) ...
        state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        
        action, log_prob, state_val = model.get_action(state, add_noise=True)

        a_in = [
            (action[0] + 1) * 1.5,
            action[1] * 0.1745,  # ±10 deg/s
        ]

        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )

        next_state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        
        model.buffer.add(
            state, action, reward, terminal, next_state, log_prob, state_val
        )

        steps += 1 # Increment steps immediately

        if terminal or steps >= max_steps:
            episode_time = time.time() - episode_start_time
            # print(f"📊 Episode {episode_count + 1} completed in {episode_time:.2f}s ({steps} steps)")
            episode_count += 1
            episode_count_since_last_train += 1
            episode_start_time = time.time()
            
            if episode_count_since_last_train >= train_every_n_episodes:
                print(f"\n🔄 Training on {episode_count_since_last_train} episodes of experience...")
                model.train(None, training_iterations, batch_size)
                episode_count_since_last_train = 0
            
            if episode_count % episodes_per_epoch == 0:
                epoch += 1
                avg_reward, avg_goal, avg_col = evaluate(model, epoch, phase2_worlds, nr_eval_episodes, max_steps, chi_inf=chi_inf, k=k)
                
                # Calculate current action std properly
                current_std = torch.exp(model.policy.log_std).mean().item()
                
                if avg_reward > best_reward:
                    print("=" * 60)
                    print(f"🎉 NEW BEST MODEL!")
                    print(f"   Previous best reward: {best_reward:.1f}")
                    print(f"   New best reward:      {avg_reward:.1f}")
                    print(f"   Goal rate:            {avg_goal * 100:.1f}%")
                    print(f"   Collision rate:       {avg_col * 100:.1f}%")
                    print(f"   Current action std:   {current_std:.4f}")
                    print("=" * 60)

                    best_reward = avg_reward
                    best_goal_rate = avg_goal
                    patience_counter = 0
                    model.save(filename=f"{phase2_model_name}_BEST", directory=Path("robot_nav/models/PPO/best_checkpoint"))

                    metrics = {
                        "epoch": epoch,
                        "avg_reward": float(avg_reward),
                        "goal_rate": float(avg_goal),
                        "collision_rate": float(avg_col),
                        "action_std": float(current_std),
                        "training_mode": "random_phase2",
                        "imazu_scenario": "01-04"
                    }
                    with open("robot_nav/models/PPO/best_checkpoint/best_metrics_imazu_01_MLPCNNPPO_phase2.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement for {patience_counter} epochs")
                    print(f"   Current reward: {avg_reward:.1f}")
                    print(f"   Best reward:    {best_reward:.1f}")
                    print(f"   Current action std: {current_std:.4f}")

                if patience_counter >= patience:
                    print("=" * 60)
                    print("🛑 EARLY STOPPING!")
                    print(f"   No improvement for {patience_counter} consecutive epochs")
                    print(f"   Best reward: {best_reward:.1f}")
                    print(f"   Best goal rate: {best_goal_rate * 100:.1f}%")
                    break
            
            # Reset for next episode
            selected_world = random.choice(phase2_worlds)
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
            steps = 0

    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED - CURRICULUM PHASE 2!")
    print(f"   Best reward achieved:  {best_reward:.1f}")
    print(f"   Best goal rate:        {best_goal_rate * 100:.1f}%")
    print(f"   Best model saved to:   robot_nav/models/PPO/best_checkpoint/")
    print(f"   Model name:            otter_MLPCNNPPO_imazu_01_phase2_BEST")
    print("=" * 60)

def evaluate(model, epoch, world_files, eval_episodes, max_steps, chi_inf, k):
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print(f"📈 EVALUATION | Epoch {epoch}")
    print("=" * 60)
    
    total_rewards = []
    total_steps = []
    goal_reached_count = 0
    collision_count = 0

    # Create sim instance once for evaluation
    sim = OtterSIM(
        world_file=world_files[0], # Initialize with first world
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=1.0, w_safety=3.0,  # 안전 우선!
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=chi_inf, k=k
    )

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        
        selected_world = random.choice(world_files)
        # Reuse sim instance, reload world if needed
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        ep_reward = 0
        for s in range(max_steps):
            state, terminal = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            action, _, _ = model.get_action(state, add_noise=False, update_rms=False)
            # print(f"Eval Action: {action}")
            a_in = [
                (action[0] + 1) * 1.5,
                action[1] * 0.1745,  # ±10 deg/s
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

    print(f"\n📊 Eval Results: Avg Reward: {avg_reward:.2f} | Goal Rate: {goal_rate:.2f} | Collision Rate: {collision_rate:.2f} | Avg Steps: {avg_steps:.1f}")
    import matplotlib.pyplot as plt
    plt.close('all')
    return avg_reward, goal_rate, collision_rate

if __name__ == "__main__":
    main()
