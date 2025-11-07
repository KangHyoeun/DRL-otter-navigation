#!/usr/bin/env python3
"""
CNNSAC 90m Scratch Training
===========================

Otter USV navigation in 90m straight-line environment using CNNSAC.

Configuration:
- Environment: s1_90m.yaml (90m straight navigation)
- Algorithm: CNNSAC (CNN-based Soft Actor-Critic)
- Phase 1: ENABLED (action frequency control, 0.5s interval)
- Max steps: 1000
- Init temperature: 0.1
- Epochs: 60
- Episodes per epoch: 70
"""

from robot_nav.models.SAC.CNNSAC import CNNSAC
import torch
import numpy as np
import json
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from robot_nav.utils import get_buffer
from pathlib import Path


def main():
    """Main training function for Otter USV with CNNSAC - 90m scratch training"""

    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 369  # 360 LiDAR + 3 goal + 2 action + 2 velocity + 2 propeller
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print("=" * 80)
    print("🚀 CNNSAC 90m Scratch Training")
    print("=" * 80)
    print(f"🖥️  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   Using CPU (slower training)")
    
    # Training parameters
    nr_eval_episodes = 10
    max_epochs = 60              # ⭐ 60 epochs
    epoch = 0
    episodes_per_epoch = 70      # ⭐ 70 episodes per epoch
    episode = 0
    train_every_n = 2            # Train every 2 episodes
    training_iterations = 80
    batch_size = 128
    max_steps = 1000             # ⭐ 1000 max steps
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5
    load_model = False           # ⭐ Scratch training (no pre-trained model)
    
    initial_random_episodes = 35  # Initial random exploration
    
    model = CNNSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        init_temperature=0.1,    # ⭐ 0.1 initial temperature
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        save_every=save_every,
        load_model=load_model,
        model_name="otter_CNNSAC_90m_scratch",
        load_directory=Path("robot_nav/models/SAC/checkpoint"),
    )
    
    # Initialize simulation
    print("\n🔧 Environment Configuration:")
    print("   - World file: s1_90m.yaml")
    print("   - Distance: 90m straight navigation")
    print("   - Plotting: DISABLED (faster simulation)")
    print("   - Phase 1: ENABLED (action frequency control)")
    print(f"   - Max steps: {max_steps}")
    print(f"   - Action interval: 0.5s (5 physics steps)")
    
    sim = OtterSIM(
        world_file="robot_nav/worlds/imazu_scenario/s1_90m.yaml",
        disable_plotting=True,
        enable_phase1=True,      # ⭐ Phase 1 ENABLED
        max_steps=max_steps
    )
    
    # Initialize replay buffer
    replay_buffer = get_buffer(
        model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )

    latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
        u_ref=0.0, r_ref=0.0
    )

    # Performance monitoring
    import time
    episode_start_time = time.time()
    
    # Best model tracking
    best_reward = -float('inf')
    best_goal_rate = 0.0
    patience = 10  # 90m는 더 긴 학습 필요
    epochs_without_improvement = 0

    print("\n" + "=" * 80)
    print("🎯 TRAINING HYPERPARAMETERS:")
    print(f"   - Epochs:                  {max_epochs}")
    print(f"   - Episodes per epoch:      {episodes_per_epoch}")
    print(f"   - Total episodes:          {max_epochs * episodes_per_epoch} = {max_epochs * episodes_per_epoch}")
    print(f"   - Max steps per episode:   {max_steps}")
    print(f"   - Initial temperature (α): {0.1}")
    print(f"   - Initial random episodes: {initial_random_episodes}")
    print(f"   - Train every n episodes:  {train_every_n}")
    print(f"   - Training iterations:     {training_iterations}")
    print(f"   - Batch size:              {batch_size}")
    print(f"   - Patience:                {patience} epochs")
    print("=" * 80)
    
    # Episode counter for initial exploration
    total_episodes = 0
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, robot_state
        )
        
        if total_episodes < initial_random_episodes:
            # Random action for initial exploration
            action = np.random.uniform(-1, 1, size=2)
            if steps == 0:  # Episode start
                print(f"🎲 Random exploration episode {total_episodes + 1}/{initial_random_episodes}")
        else:
            # SAC policy with entropy-based exploration
            action = model.get_action(np.array(state), add_noise=True)
        
        # Clip action to environment limits
        a_in = [
            (action[0] + 1) * 1.5,   # [0, 3.0] m/s
            action[1] * 0.1745,      # [-10, 10] deg/s
        ]

        latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )
        
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, robot_state
        )
        
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )

        if terminal or steps >= max_steps:
            episode_time = time.time() - episode_start_time
            
            # Episode summary
            status = "✅ Goal" if goal else ("💥 Collision" if collision else "⏱️  Timeout")
            print(f"{status} | Episode {episode + 1}/{episodes_per_epoch} | "
                  f"Epoch {epoch + 1}/{max_epochs} | "
                  f"Steps: {steps} | Time: {episode_time:.1f}s")
            
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
            episode += 1
            total_episodes += 1
            episode_start_time = time.time()
            
            # Training
            if total_episodes > initial_random_episodes and episode % train_every_n == 0:
                print(f"🔧 Training... (iteration {episode // train_every_n})")
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )

            steps = 0
        else:
            steps += 1

        # Epoch evaluation
        if episode >= episodes_per_epoch:
            epoch += 1
            
            if total_episodes > initial_random_episodes:
                print("\n" + "=" * 80)
                print(f"📊 EPOCH {epoch} EVALUATION")
                print("=" * 80)
                
                avg_reward, avg_goal, avg_col = evaluate(
                    model, epoch, sim, eval_episodes=nr_eval_episodes, max_steps=max_steps
                )
                
                # Save best model
                if avg_reward > best_reward or (avg_reward == best_reward and avg_goal > best_goal_rate):
                    print("\n" + "🎉" * 40)
                    print(f"NEW BEST MODEL!")
                    print(f"   Previous best reward: {best_reward:.1f}")
                    print(f"   New best reward:      {avg_reward:.1f}")
                    print(f"   Goal rate:            {avg_goal * 100:.1f}%")
                    print(f"   Collision rate:       {avg_col * 100:.1f}%")
                    print("🎉" * 40 + "\n")
                    
                    best_reward = avg_reward
                    best_goal_rate = avg_goal
                    epochs_without_improvement = 0
                    
                    # Save best model
                    best_dir = Path("robot_nav/models/SAC/best_checkpoint")
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save(filename="otter_CNNSAC_90m_scratch_BEST", directory=best_dir)
                    
                    # Save metrics
                    metrics = {
                        "epoch": epoch,
                        "avg_reward": float(avg_reward),
                        "goal_rate": float(avg_goal),
                        "collision_rate": float(avg_col),
                        "total_episodes": total_episodes
                    }
                    with open(best_dir / "best_metrics_90m_scratch.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
                        
                else:
                    epochs_without_improvement += 1
                    print(f"\n⚠️  No improvement for {epochs_without_improvement}/{patience} epochs")
                    print(f"   Current reward: {avg_reward:.1f}")
                    print(f"   Best reward:    {best_reward:.1f}")
                    
                    if epochs_without_improvement >= patience:
                        print("\n" + "🛑" * 40)
                        print("EARLY STOPPING!")
                        print(f"   No improvement for {patience} consecutive epochs")
                        print(f"   Best reward: {best_reward:.1f} at epoch {epoch - patience}")
                        print(f"   Best goal rate: {best_goal_rate * 100:.1f}%")
                        print("🛑" * 40 + "\n")
                        break
            else:
                print(f"\n⏭️  Epoch {epoch}: Skipping evaluation (still in initial exploration)")
            
            episode = 0
            steps = 0
            episode_start_time = time.time()
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(f"   Environment:           90m straight navigation")
    print(f"   Total episodes:        {total_episodes}")
    print(f"   Initial random:        {initial_random_episodes}")
    print(f"   Training episodes:     {total_episodes - initial_random_episodes}")
    print(f"   Final epoch:           {epoch}/{max_epochs}")
    print(f"   Best reward achieved:  {best_reward:.1f}")
    print(f"   Best goal rate:        {best_goal_rate * 100:.1f}%")
    print(f"   Best model saved to:   robot_nav/models/SAC/best_checkpoint/")
    print(f"   Model name:            otter_CNNSAC_90m_scratch_BEST")
    print("=" * 80)


def evaluate(model, epoch, sim, eval_episodes=10, max_steps=1000):
    """Evaluate model performance"""
    print(f"\nEvaluating {eval_episodes} episodes...")
    
    avg_reward = 0.0
    col = 0
    goals = 0
    total_steps = 0
    
    for ep in range(eval_episodes):
        count = 0
        ep_reward = 0.0
        latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
        done = False
        
        while not done and count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a, robot_state
            )
            
            # NO noise during evaluation (deterministic policy)
            action = model.get_action(np.array(state), add_noise=False)
            a_in = [(action[0] + 1) * 1.5, action[1] * 0.1745]
            
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
                u_ref=a_in[0], r_ref=a_in[1]
            )
            
            ep_reward += reward
            count += 1
            
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
        
        avg_reward += ep_reward
        total_steps += count
        
        # Episode result
        status = "✅" if goal else ("💥" if collision else "⏱️")
        print(f"  {status} Eval {ep+1}/{eval_episodes}: {count} steps, reward={ep_reward:.1f}")
    
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    avg_steps = total_steps / eval_episodes
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Average Reward:     {avg_reward:.1f}")
    print(f"   Goal Rate:          {avg_goal * 100:.1f}%")
    print(f"   Collision Rate:     {avg_col * 100:.1f}%")
    print(f"   Average Steps:      {avg_steps:.1f}")
    
    # TensorBoard logging
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    
    return avg_reward, avg_goal, avg_col


if __name__ == "__main__":
    main()
