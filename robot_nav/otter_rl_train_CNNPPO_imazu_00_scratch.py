from robot_nav.models.PPO.CNNPPO import CNNPPO
import torch
import numpy as np
import json
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from pathlib import Path


def main():
    """Main training function for Otter USV Imazu Case Collision Avoidance - SCRATCH"""

    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 370  # 360 LiDAR + 2 velocity + 2 error + 3 goal (dist, y_e, φ_tilde) + 2 propeller + 1 CR_max
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"🚀 CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   Using CPU (slower training)")
    
    # Training parameters
    nr_eval_episodes = 10
    max_epochs = 100
    epoch = 0
    episodes_per_epoch = 10
    episode = 0
    train_every_n_episodes = 5  # PPO는 여러 episode 모은 후 학습
    training_iterations = 10  # PPO epoch per update
    batch_size = 256           # Minibatch size for PPO training
    max_steps = 2000 
    steps = 0
    save_every = 5
    load_model = False  # ⭐ SCRATCH 학습!
    
    # PPO specific parameters
    lr_actor = 0.0001
    lr_critic = 0.0003
    gamma = 0.995
    eps_clip = 0.2
    action_std_init = 0.6  
    action_std_decay_rate = 0.015  
    min_action_std = 0.1  # Minimum exploration
    
    print("\n" + "=" * 60)
    print("🎯 CURRICULUM PHASE 1: GOAL REACHING (IMAZU CASE 00)")
    print("=" * 60)
    print("   Environment: Imazu Case 00 (No Obstacles)")
    print("   Start: (0, -90)")
    print("   Goal: (0, 90)")
    print("   Load model: NO (learning from scratch)")
    print("   Max steps: 2000")
    print("=" * 60)
    
    model = CNNPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        eps_clip=eps_clip,
        action_std_init=action_std_init,
        action_std_decay_rate=action_std_decay_rate,
        min_action_std=min_action_std,
        device=device,
        save_every=save_every,
        load_model=load_model,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name="otter_CNNPPO_imazu_00_scratch",  # ⭐ 새로운 모델 이름
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )
    
    # Initialize simulation
    print("\n🔧 Performance Settings:")
    print("   - Plotting: DISABLED (faster simulation)")
    print("   - Phase 1: ENABLED (action frequency control)")
    print("   - Max steps: 1000")
    print("\n🎯 CNNPPO Configuration:")
    print(f"   - CNN Feature Extraction: ENABLED ✅")
    print(f"   - LiDAR: 360 → 36 features (via CNN)")
    print(f"   - Total features: 76 (36 CNN + 40 embeddings)")
    print(f"   - Action std init: {action_std_init}")
    print(f"   - Action std decay: {action_std_decay_rate}")
    print(f"   - Min action std: {min_action_std}")
    print(f"   - Clip epsilon: {eps_clip}")
    print(f"   - Train every {train_every_n_episodes} episodes")
    
    sim = OtterSIM(
        world_file="robot_nav/worlds/imazu_scenario/imazu_case_00.yaml",
        disable_plotting=True,
        enable_phase1=True,
        max_steps=max_steps,
        cr_method='chun',
        w_efficiency=1.0,
        w_safety=0.0,
        os_speed_for_cr=2.0, # Reference speed
        ts_speed_for_cr=0.0  # No target ships in phase 1
    )

    latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(
        u_ref=0.0, r_ref=0.0
    )

    # Performance monitoring
    import time
    episode_start_time = time.time()
    
    # Best model tracking
    best_reward = -float('inf')
    best_goal_rate = 0.0
    patience = 10  # 더 긴 patience (90m은 더 어려울 수 있음)
    epochs_without_improvement = 0

    print("\n🎯 BEST MODEL CHECKPOINT ENABLED!")
    print(f"   - Patience: {patience} epochs")
    print(f"   - Checkpoints saved to: robot_nav/models/PPO/checkpoint/")
    print("=" * 60)
    
    # Episode accumulator for training
    episode_count_since_last_train = 0
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, y_e, collision, goal, a, robot_state,
            sim.start_position, sim.goal_position, CR_max
        )
        # PPO: Sample from stochastic policy during training
        action, log_prob, state_val = model.get_action(state, add_noise=True)
        
        # Clip action to environment limits
        a_in = [
            (action[0] + 1) * 1.5,   # [0, 3.0] m/s
            action[1] * 0.1745,      # [-10, 10] deg/s
        ]

        latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )
        
        next_state, terminal = model.prepare_state(
            latest_scan, distance, y_e, collision, goal, a, robot_state,
            sim.start_position, sim.goal_position, CR_max
        )
        
        # Add to rollout buffer
        model.buffer.add(
            state, action, reward, terminal, next_state, log_prob, state_val
        )

        if terminal or steps == max_steps:
            episode_time = time.time() - episode_start_time
            print(f"📊 Episode {episode + 1} completed in {episode_time:.2f}s ({steps} steps)")
            
            latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
            episode += 1
            episode_count_since_last_train += 1
            episode_start_time = time.time()
            
            # Train every N episodes (PPO accumulates experiences)
            if episode_count_since_last_train >= train_every_n_episodes:
                print(f"🔄 Training on {episode_count_since_last_train} episodes of experience...")
                model.train(
                    replay_buffer=None,  # Not used in PPO
                    iterations=training_iterations,
                    batch_size=batch_size,  # Not used in PPO
                )
                episode_count_since_last_train = 0

            steps = 0
        else:
            steps += 1

        if episode % episodes_per_epoch == 0 and episode > 0:
            epoch += 1
            avg_reward, avg_goal, avg_col = evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes, max_steps=max_steps)
            
            # Save best model
            if avg_reward > best_reward:
                print("=" * 60)
                print(f"🎉 NEW BEST MODEL!")
                print(f"   Previous best reward: {best_reward:.1f}")
                print(f"   New best reward:      {avg_reward:.1f}")
                print(f"   Goal rate:            {avg_goal * 100:.1f}%")
                print(f"   Collision rate:       {avg_col * 100:.1f}%")
                print(f"   Current action std:   {model.action_std:.4f}")
                print("=" * 60)
                
                best_reward = avg_reward
                best_goal_rate = avg_goal
                epochs_without_improvement = 0
                
                best_dir = Path("robot_nav/models/PPO/best_checkpoint")
                model.save(filename="otter_CNNPPO_imazu_00_scratch_BEST", directory=best_dir)
                
                metrics = {
                    "epoch": epoch,
                    "avg_reward": float(avg_reward),
                    "goal_rate": float(avg_goal),
                    "collision_rate": float(avg_col),
                    "action_std": float(model.action_std),
                    "training_mode": "scratch_phase1",
                    "distance": "90m"
                }
                with open(best_dir / "best_metrics_imazu_00_scratch.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
            else:
                epochs_without_improvement += 1
                print(f"⚠️  No improvement for {epochs_without_improvement} epochs")
                print(f"   Current reward: {avg_reward:.1f}")
                print(f"   Best reward:    {best_reward:.1f}")
                print(f"   Current action std: {model.action_std:.4f}")
                
                if epochs_without_improvement >= patience:
                    print("=" * 60)
                    print("🛑 EARLY STOPPING!")
                    print(f"   No improvement for {patience} consecutive epochs")
                    print(f"   Best reward: {best_reward:.1f}")
                    print(f"   Best goal rate: {best_goal_rate * 100:.1f}%")
                    break
            
            episode = 0
            steps = 0
            episode_count_since_last_train = 0
            episode_start_time = time.time()
            latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED - CURRICULUM PHASE 1!")
    print(f"   Best reward achieved:  {best_reward:.1f}")
    print(f"   Best goal rate:        {best_goal_rate * 100:.1f}%")
    print(f"   Best model saved to:   robot_nav/models/PPO/best_checkpoint/")
    print(f"   Model name:            otter_CNNPPO_imazu_00_scratch_BEST")
    print("=" * 60)


def evaluate(model, epoch, sim, eval_episodes=10, max_steps=2000):
    """Evaluate model performance"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
        done = False
        
        while not done and count < max_steps + 1:
            state, terminal = model.prepare_state(
                latest_scan, distance, y_e, collision, goal, a, robot_state,
                sim.start_position, sim.goal_position, CR_max
            )
            
            # NO noise during evaluation (use mean action)
            action, _, _ = model.get_action(state, add_noise=False)
            a_in = [(action[0] + 1) * 1.5, action[1] * 0.1745]
            
            latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            avg_reward += reward
            count += 1
            
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
            
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    
    return avg_reward, avg_goal, avg_col


if __name__ == "__main__":
    main()
