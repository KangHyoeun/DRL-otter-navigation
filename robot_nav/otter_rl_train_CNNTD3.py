from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
import json
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from utils import get_buffer
from pathlib import Path


def main():
    """Main training function for Otter USV with CNNTD3 - IMPROVED with initial exploration"""

    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 369  # 360 LiDAR + 3 goal + 2 action + 2 velocity + 2 propeller             
    
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
    max_epochs = 30
    epoch = 0
    episodes_per_epoch = 35
    episode = 0
    train_every_n = 2           # Train every n episodes
    training_iterations = 80    # Batches per training cycle
    batch_size = 128
    max_steps = 300             # Max steps per episode
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5
    load_model = False
    
    initial_random_episodes = 10  # 처음 10 episodes는 완전 random
    
    # Initialize model
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=load_model,
        model_name="otter_CNNTD3_imazu_scenario",
        load_directory=Path("robot_nav/models/CNNTD3/checkpoint"),
        use_max_bound=True,
    )
    
    # Initialize simulation with performance optimizations
    print("\n🔧 Performance Settings:")
    print("   - Plotting: DISABLED (faster simulation)")
    print("   - Phase 1: ENABLED (action frequency control)")
    print("   - Max steps: 300")
    sim = OtterSIM(
        world_file="/worlds/imazu_scenario/s1.yaml", 
        disable_plotting=True, 
        enable_phase1=True
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
    patience = 5
    epochs_without_improvement = 0
    
    if load_model:
        best_dir = Path("robot_nav/models/CNNTD3/best_checkpoint")
        metrics_file = best_dir / "best_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if "avg_reward" in metrics:
                    best_reward = metrics.get("avg_reward", -float('inf'))
                    best_goal_rate = metrics.get("goal_rate", 0.0)
                    print(f"✅ Successfully loaded previous best metrics from {metrics_file}")
                    print(f"   - Best reward: {best_reward:.1f}")
                    print(f"   - Best goal rate: {best_goal_rate * 100:.1f}%")
                    print(f"   - Epoch: {metrics.get('epoch', 'N/A')}")
                    print(f"   - Collision rate: {metrics.get('collision_rate', 0.0) * 100:.1f}%")
    
    print("=" * 60)
    print("🎯 IMPROVED TD3 SETTINGS:")
    print(f"   - Gaussian noise std: 0.3 (increased from 0.1)")
    print(f"   - Initial random exploration: {initial_random_episodes} episodes")
    print(f"   - Best model checkpoint enabled")
    print(f"   - Patience: {patience} epochs")
    print("=" * 60)
    
    # Episode counter for initial exploration
    total_episodes = 0
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, robot_state
        )
        
        if total_episodes < initial_random_episodes:
            # 완전 random action
            action = np.random.uniform(-1, 1, size=2)
            if steps == 0:  # Episode 시작
                print(f"🎲 Random exploration episode {total_episodes + 1}/{initial_random_episodes}")
        else:
            # TD3 policy 사용 (with Gaussian noise)
            action = model.get_action(np.array(state), True)
        
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

        if terminal or steps == max_steps:
            episode_time = time.time() - episode_start_time
            print(f"📊 Episode {episode + 1} completed in {episode_time:.2f}s ({steps} steps)")
            
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
            episode += 1
            total_episodes += 1  # ⭐ Total episode counter
            episode_start_time = time.time()
            
            if total_episodes > initial_random_episodes and episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                    max_lin_vel=3.0,
                    max_ang_vel=0.1745,
                    goal_reward=3000,
                    distance_norm=20,
                    time_step=0.1,
                )

            steps = 0
        else:
            steps += 1

        if episode % episodes_per_epoch == 0 and episode > 0:
            epoch += 1
            
            # ⭐ Evaluation도 initial explration 끝난 후부터
            if total_episodes > initial_random_episodes:
                avg_reward, avg_goal, avg_col = evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)
                
                if avg_reward > best_reward:
                    print("=" * 60)
                    print(f"🎉 NEW BEST MODEL!")
                    print(f"   Previous best reward: {best_reward:.1f}")
                    print(f"   New best reward:      {avg_reward:.1f}")
                    print(f"   Goal rate:            {avg_goal * 100:.1f}%")
                    print(f"   Collision rate:       {avg_col * 100:.1f}%")
                    print("=" * 60)
                    
                    best_reward = avg_reward
                    best_goal_rate = avg_goal
                    epochs_without_improvement = 0
                    
                    # Save best model
                    best_dir = Path("robot_nav/models/CNNTD3/best_checkpoint")
                    model.save(filename="otter_CNNTD3_BEST", directory=best_dir)
                    
                    # Also save metrics
                    metrics = {
                        "epoch": epoch,
                        "avg_reward": float(avg_reward),
                        "goal_rate": float(avg_goal),
                        "collision_rate": float(avg_col)
                    }
                    with open(best_dir / "best_metrics.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
                        
                else:
                    epochs_without_improvement += 1
                    print(f"⚠️  No improvement for {epochs_without_improvement} epochs")
                    print(f"   Current reward: {avg_reward:.1f}")
                    print(f"   Best reward:    {best_reward:.1f}")
                    
                    # Early stopping
                    if epochs_without_improvement >= patience:
                        print("=" * 60)
                        print("🛑 EARLY STOPPING!")
                        print(f"   No improvement for {patience} consecutive epochs")
                        print(f"   Best reward: {best_reward:.1f} at epoch {epoch - patience}")
                        break
            else:
                print(f"⏭️  Skipping evaluation (still in initial exploration phase)")
            
            episode = 0
            steps = 0
            episode_start_time = time.time()
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print(f"   Total episodes:        {total_episodes}")
    print(f"   Initial random:        {initial_random_episodes}")
    print(f"   Best reward achieved:  {best_reward:.1f}")
    print(f"   Best goal rate:        {best_goal_rate * 100:.1f}%")
    print(f"   Best model saved to:   robot_nav/models/CNNTD3/best_checkpoint/")
    print("=" * 60)


def evaluate(model, epoch, sim, eval_episodes=10):
    """
    Evaluate model performance
    
    Returns:
        avg_reward, avg_goal, avg_col
    """
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
        done = False
        
        while not done and count < 301:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a, robot_state
            )
            # NO noise during evaluation
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) * 1.5, action[1] * 0.1745]
            
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
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
