import matplotlib
# Force TkAgg backend before importing any other modules that might import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
plt.ion() # Enable interactive mode for animation

import torch
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

# CHANGED: Import Multi-Modal SAC
from robot_nav.models.SAC.MLPCNNSAC import MLPCNNSAC
from robot_nav.models.SAC.SAC_utils import MultiModalReplayBuffer # Import the new buffer
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.utils.utils import WrapTo180
from colregs_core.geometry import math_to_ned_heading

def main():
    """Main training function for Otter USV Imazu Case Collision Avoidance - PHASE 2 using MLPCNNSAC"""

    phase2_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_01.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_02.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_03.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_04.yaml"
    ]
    
    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 12 # Vector state dim
    
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
    episodes_per_epoch = 40 # Still used to trigger evaluation
    train_every_n_steps = 50 # SAC trains every N steps (Frequency increased)
    training_iterations = 50 # Update N times per training cycle (1:1 ratio maintained)
    
    batch_size = 256 # SAC can use smaller batch sizes
    max_steps = 512   # Simulation max steps
    replay_buffer_capacity = int(1e5) # 100k capacity, can be adjusted

    save_every = 10
    load_model = False
    
    # SAC specific parameters (Optimized)
    discount = 0.99
    init_temperature = 0.1
    alpha_lr = 1e-4
    actor_lr = 3e-4 # Often higher for SAC actor
    critic_lr = 3e-4 # Often higher for SAC critic
    critic_tau = 0.005 # Soft update for target critic
    actor_update_frequency = 1 # Update actor every step
    critic_target_update_frequency = 2 # Update target critic every 2 steps
    
    chi_inf = 1.0
    k = 1.0
    
    # Model names
    phase1_model_name = "otter_MLPCNNSAC_imazu_00_scratch_BEST"
    phase2_model_name = "otter_MLPCNNSAC_imazu_01_phase2"
    
    print("\n" + "=" * 60)
    print("🎯 CURRICULUM PHASE 2: COLLISION AVOIDANCE (1 Target Ship) - SAC")
    print("=" * 60)
    print(f"   Environments: {len(phase2_worlds)} random scenarios (cases 01-04)")
    print(f"   Model Type: Multi-Modal (Vector + CNN Grid)")
    print("   Optimized Hyperparameters Applied")
    print("   Replay Buffer Capacity: {:,}".format(replay_buffer_capacity))
    print("=" * 60)
    
    model = MLPCNNSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        max_action=max_action,
        discount=discount,
        init_temperature=init_temperature,
        alpha_lr=alpha_lr,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        critic_tau=critic_tau,
        actor_update_frequency=actor_update_frequency,
        critic_target_update_frequency=critic_target_update_frequency,
        save_every=save_every,
        load_model=load_model,
        save_directory=Path("robot_nav/models/SAC/checkpoint"),
        model_name=phase2_model_name,
        load_directory=Path("robot_nav/models/SAC/best_checkpoint"),
    )
    
    # Override internal buffer if needed, or use model.buffer directly
    replay_buffer = model.buffer 
    # Initial fill for buffer for SAC
    print("\nFilling replay buffer for SAC...")
    prefill_steps = batch_size * 5 # Minimum steps to ensure initial batch can be sampled
    current_prefill_steps = 0

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
    total_steps = 0 # Track total steps for SAC training trigger
    
    sim = OtterSIM(
        world_file=random.choice(phase2_worlds),
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=1.0, w_safety=3.0,
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=chi_inf, k=k
    )
    
    # =================================================================================
    # 🔥 RMS WARMUP PHASE
    # =================================================================================
    print("\n🔥 Starting RMS Warmup (10000 steps)...")
    print("   Collecting random experiences to stabilize RunningMeanStd.")
    
    warmup_steps = 10000 # Same as PPO for consistency
    w_sim = OtterSIM(
        world_file=random.choice(phase2_worlds),
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=1.0, w_safety=3.0,
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=chi_inf, k=k
    )
    w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset()
    
    for _ in tqdm(range(warmup_steps), desc="Warmup"):
        s_w, _ = model.prepare_state(
            w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_state, w_cr, grid_map=w_grid
        )
        model.get_action(s_w, add_noise=True, update_rms=True)
        
        w_u = random.uniform(0.0, 3.0)
        w_r = random.uniform(-10.0, 10.0) * 0.01745 # Radian conversion
        
        w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.step(u_ref=w_u, r_ref=w_r)
        
        if w_col or w_goal:
            w_dist, w_ye, w_psi, w_chi, w_phi, w_col, w_goal, w_a, w_rew, w_state, w_cr, w_grid = w_sim.reset(world_file=random.choice(phase2_worlds))
    
    print("✅ RMS Warmup Complete. Mean/Var stabilized.")
    print(f"   Vector RMS Mean: {model.obs_rms.mean[:3]}...")
    print("=" * 60)
    
    distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=random.choice(phase2_worlds))
    steps_in_episode = 0
    
    # Main training loop
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        
        # For SAC, get_action returns (action, None, None)
        action_np, _, _ = model.get_action(state, add_noise=True) 

        a_in = [
            (action_np[0] + 1) * 1.5, # Scale surge from [-1,1] to [0,3]
            action_np[1] * 0.1745,  # Scale yaw rate from [-1,1] to [-10deg/s, 10deg/s]
        ]

        # Save current state for buffer
        current_vec_state = state[0]
        current_grid_state = state[1]

        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )

        next_state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
        )
        next_vec_state = next_state[0]
        next_grid_state = next_state[1]
        
        # Add to MultiModalReplayBuffer (state, action, reward, next_state, done)
        replay_buffer.add(
            (current_vec_state, current_grid_state), 
            action_np, 
            reward, 
            (next_vec_state, next_grid_state), 
            terminal
        )

        steps_in_episode += 1
        total_steps += 1

        # Train if enough samples in buffer and every N total_steps
        if len(replay_buffer) >= batch_size and total_steps % train_every_n_steps == 0:
            print(f"\n🔄 Training SAC (Total Steps: {total_steps})...")
            model.train(replay_buffer, training_iterations, batch_size)

        if terminal or steps_in_episode >= max_steps:
            episode_time = time.time() - episode_start_time
            print(f"📊 Episode {episode_count + 1} completed in {episode_time:.2f}s ({steps_in_episode} steps)")
            episode_count += 1
            episode_start_time = time.time()
            steps_in_episode = 0 # Reset episode steps
            
            if episode_count % episodes_per_epoch == 0:
                epoch += 1
                avg_reward, avg_goal, avg_col = evaluate(model, epoch, phase2_worlds, nr_eval_episodes, max_steps, chi_inf=chi_inf, k=k)
                
                # SAC does not have a direct action_std attribute like PPO's learnable log_std
                # We will log alpha (temperature) instead as an indicator of exploration/exploitation
                current_alpha = model.alpha.item()
                
                if avg_reward > best_reward:
                    print("=" * 60)
                    print(f"🎉 NEW BEST MODEL!")
                    print(f"   Previous best reward: {best_reward:.1f}")
                    print(f"   New best reward:      {avg_reward:.1f}")
                    print(f"   Goal rate:            {avg_goal * 100:.1f}%")
                    print(f"   Collision rate:       {avg_col * 100:.1f}%")
                    print(f"   Current Alpha (Temp): {current_alpha:.4f}")
                    print("=" * 60)

                    best_reward = avg_reward
                    best_goal_rate = avg_goal
                    patience_counter = 0
                    model.save(filename=f"{phase2_model_name}_BEST", directory=Path("robot_nav/models/SAC/best_checkpoint"))

                    metrics = {
                        "epoch": epoch,
                        "avg_reward": float(avg_reward),
                        "goal_rate": float(avg_goal),
                        "collision_rate": float(avg_col),
                        "alpha_temp": float(current_alpha),
                        "training_mode": "random_phase2_SAC",
                        "imazu_scenario": "01-04"
                    }
                    with open("robot_nav/models/SAC/best_checkpoint/best_metrics_imazu_01_MLPCNNSAC_phase2.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement for {patience_counter} epochs")
                    print(f"   Current reward: {avg_reward:.1f}")
                    print(f"   Best reward:    {best_reward:.1f}")
                    print(f"   Current Alpha (Temp): {current_alpha:.4f}")

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

    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED - CURRICULUM PHASE 2 (SAC)!")
    print(f"   Best reward achieved:  {best_reward:.1f}")
    print(f"   Best goal rate:        {best_goal_rate * 100:.1f}%")
    print(f"   Best model saved to:   robot_nav/models/SAC/best_checkpoint/")
    print(f"   Model name:            otter_MLPCNNSAC_imazu_01_phase2_BEST")
    print("=" * 60)

def evaluate(model, epoch, world_files, eval_episodes, max_steps, chi_inf, k):
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print(f"📈 EVALUATION | Epoch {epoch} (SAC)")
    print("=" * 60)
    
    total_rewards = []
    total_steps = []
    goal_reached_count = 0
    collision_count = 0

    sim = OtterSIM(
        world_file=world_files[0], 
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='jeon', w_efficiency=1.0, w_safety=3.0,
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=chi_inf, k=k
    )

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        selected_world = random.choice(world_files)
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, reward, robot_state, CR_max, cr_grid = sim.reset(world_file=selected_world)
        
        ep_reward = 0
        for s in range(max_steps):
            state, terminal = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=cr_grid
            )
            
            # SAC get_action returns (action, None, None)
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

    model.writer.add_scalar("eval/avg_reward", np.mean(total_rewards), epoch)
    model.writer.add_scalar("eval/avg_steps", np.mean(total_steps), epoch)
    model.writer.add_scalar("eval/goal_rate", goal_reached_count / eval_episodes, epoch)
    model.writer.add_scalar("eval/collision_rate", collision_count / eval_episodes, epoch)

    print(f"\n📊 Eval Results: Avg Reward: {np.mean(total_rewards):.2f} | Goal Rate: {goal_reached_count / eval_episodes:.2f} | Collision Rate: {collision_count / eval_episodes:.2f} | Avg Steps: {np.mean(total_steps):.1f}")
    plt.close('all')
    return np.mean(total_rewards), goal_reached_count / eval_episodes, collision_count / eval_episodes

if __name__ == "__main__":
    main()
