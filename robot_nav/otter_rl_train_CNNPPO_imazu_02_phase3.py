import torch
import numpy as np
from pathlib import Path
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robot_nav.models.PPO.CNNPPO import CNNPPO
from robot_nav.SIM_ENV.otter_sim import OtterSIM

def main():
    """Main training function for Otter USV Imazu Case Collision Avoidance - PHASE 3"""

    # --- World Files for Phase 3 ---
    phase3_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_05.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_06.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_07.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_08.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_09.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_10.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_11.yaml",
    ]
    
    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 370
    
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
    max_epochs = 100
    episodes_per_epoch = 10
    train_every_n_episodes = 5
    training_iterations = 10
    batch_size = 256
    max_steps = 2000 
    save_every = 5
    load_model = True
    
    # PPO specific parameters
    lr_actor = 0.0001
    lr_critic = 0.0003
    gamma = 0.995
    eps_clip = 0.2
    action_std_init = 0.4  # Continue with a smaller exploration for fine-tuning
    action_std_decay_rate = 0.015  
    min_action_std = 0.1
    
    # Model names
    phase2_model_name = "otter_CNNPPO_imazu_01_phase2_BEST"
    phase3_model_name = "otter_CNNPPO_imazu_02_phase3"
    
    print("\n" + "=" * 60)
    print("🎯 CURRICULUM PHASE 3: COLLISION AVOIDANCE (2 Target Ships)")
    print("=" * 60)
    print(f"   Environments: {len(phase3_worlds)} random scenarios (cases 05-11)")
    print(f"   Load model: YES (loading '{phase2_model_name}')")
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
        load_model=False,
        save_directory=Path("robot_nav/models/PPO/checkpoint"),
        model_name=phase3_model_name,
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )

    if load_model:
        print(f"\n🔄 Loading pre-trained model from Phase 2: {phase2_model_name}")
        try:
            model.load(filename=phase2_model_name, directory=Path("robot_nav/models/PPO/best_checkpoint"))
            print("   ✅ Model loaded successfully.")
        except FileNotFoundError:
            print(f"   ❌ ERROR: Model file not found at 'robot_nav/models/PPO/best_checkpoint/{phase2_model_name}_policy.pth'.")
            print("   Please ensure the Phase 2 model was trained and saved correctly.")
            return
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{phase3_model_name}")
    
    # Early stopping parameters
    patience = 10
    patience_counter = 0
    best_avg_reward = -np.inf
    
    episode_count = 0
    
    # Main training loop
    for epoch in range(max_epochs):
        
        # --- Evaluation Loop ---
        print("\n" + "=" * 60)
        print(f"📈 EVALUATION | Epoch {epoch+1}/{max_epochs}")
        print("=" * 60)
        
        total_rewards = []
        total_steps = []
        goal_reached_count = 0
        collision_count = 0

        for _ in tqdm(range(nr_eval_episodes), desc="Evaluating"):
            
            selected_world = random.choice(phase3_worlds)
            sim = OtterSIM(
                world_file=selected_world,
                disable_plotting=True, enable_phase1=True, max_steps=max_steps,
                cr_method='chun', w_efficiency=1.0, w_safety=1.0,
                os_speed_for_cr=2.0, ts_speed_for_cr=2.0
            )
            
            latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
            current_state, _ = model.prepare_state(
                latest_scan, distance, y_e, collision, goal, a, robot_state,
                sim.start_position, sim.goal_position, CR_max
            )
            
            ep_reward = 0
            for s in range(max_steps):
                action = model.select_action(current_state)
                a_in = action * max_action
                latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(u_ref=a_in[0], r_ref=a_in[1])
                next_state, terminal = model.prepare_state(
                    latest_scan, distance, y_e, collision, goal, a, robot_state,
                    sim.start_position, sim.goal_position, CR_max
                )
                current_state = next_state
                ep_reward += reward
                if terminal:
                    break
            
            total_rewards.append(ep_reward)
            total_steps.append(s + 1)
            if goal: goal_reached_count += 1
            if collision: collision_count += 1

        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        goal_rate = goal_reached_count / nr_eval_episodes
        collision_rate = collision_count / nr_eval_episodes

        writer.add_scalar("eval/avg_reward", avg_reward, epoch)
        writer.add_scalar("eval/avg_steps", avg_steps, epoch)
        writer.add_scalar("eval/goal_rate", goal_rate, epoch)
        writer.add_scalar("eval/collision_rate", collision_rate, epoch)

        print(f"\n📊 Eval Results: Avg Reward: {avg_reward:.2f} | Goal Rate: {goal_rate:.2f} | Collision Rate: {collision_rate:.2f} | Avg Steps: {avg_steps:.1f}")

        # Early stopping check
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            patience_counter = 0
            model.save(Path("robot_nav/models/PPO/best_checkpoint"), f"{phase3_model_name}_BEST")
            print(f"🏆 New best model saved with avg reward: {best_avg_reward:.2f}")
        else:
            patience_counter += 1
            print(f"📉 No improvement for {patience_counter}/{patience} epochs. Best reward: {best_avg_reward:.2f}")

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
            break
            
        # --- Training Loop ---
        print("\n" + "=" * 60)
        print(f"💪 TRAINING | Epoch {epoch+1}/{max_epochs}")
        print("=" * 60)
        
        for _ in tqdm(range(episodes_per_epoch), desc="Training"):
            
            selected_world = random.choice(phase3_worlds)
            sim = OtterSIM(
                world_file=selected_world,
                disable_plotting=True, enable_phase1=True, max_steps=max_steps,
                cr_method='chun', w_efficiency=1.0, w_safety=1.0,
                os_speed_for_cr=2.0, ts_speed_for_cr=2.0
            )
            
            latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
            current_state, _ = model.prepare_state(
                latest_scan, distance, y_e, collision, goal, a, robot_state,
                sim.start_position, sim.goal_position, CR_max
            )
            
            for _ in range(max_steps):
                action = model.select_action(current_state)
                a_in = action * max_action
                
                latest_scan, distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(u_ref=a_in[0], r_ref=a_in[1])
                
                next_state, terminal = model.prepare_state(
                    latest_scan, distance, y_e, collision, goal, a, robot_state,
                    sim.start_position, sim.goal_position, CR_max
                )
                
                model.buffer.rewards.append(reward)
                model.buffer.terminals.append(terminal)
                current_state = next_state
                
                if terminal:
                    break
            
            episode_count += 1
            
            if episode_count % train_every_n_episodes == 0:
                print(f"\n🔄 Training on {train_every_n_episodes} episodes of experience...")
                model.train(training_iterations, batch_size, writer)
                model.buffer.clear()
        
    writer.close()
    print("\n" + "=" * 60)
    print("✅ Training complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
