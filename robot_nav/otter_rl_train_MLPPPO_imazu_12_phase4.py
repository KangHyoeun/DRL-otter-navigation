import torch
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

from robot_nav.models.PPO.MLPPPO import MLPPPO
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.utils.utils import calculate_ref_path, calculate_desired_course_angle, WrapTo180
from colregs_core.geometry import math_to_ned_heading, math_to_maritime_position

def main():
    """Main training function for Otter USV Imazu Case Collision Avoidance - PHASE 4"""

    # --- World Files for Phase 4 ---
    phase4_worlds = [
        "robot_nav/worlds/imazu_scenario/imazu_case_12.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_13.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_14.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_15.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_16.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_17.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_18.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_19.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_20.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_21.yaml",
        "robot_nav/worlds/imazu_scenario/imazu_case_22.yaml",
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
    action_std_init = 0.8  # Rely more on learned policy in complex scenarios
    action_std_decay_rate = 0.015  
    min_action_std = 0.1
    
    # Model names
    phase3_model_name = "otter_MLPPPO_imazu_02_phase3_BEST"
    phase4_model_name = "otter_MLPPPO_imazu_03_phase4"
    
    print("\n" + "=" * 60)
    print("🎯 CURRICULUM PHASE 4: COLLISION AVOIDANCE (3 Target Ships)")
    print("=" * 60)
    print(f"   Environments: {len(phase4_worlds)} random scenarios (cases 12-22)")
    print(f"   Load model: YES (loading '{phase3_model_name}')")
    print("   Max steps: 2000")
    print("=" * 60)
    
    model = MLPPPO(
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
        model_name=phase4_model_name,
        load_directory=Path("robot_nav/models/PPO/best_checkpoint"),
    )

    if load_model:
        print(f"\n🔄 Loading pre-trained model from Phase 3: {phase3_model_name}")
        try:
            model.load(filename=phase3_model_name, directory=Path("robot_nav/models/PPO/best_checkpoint"))
            print("   ✅ Model loaded successfully.")
        except FileNotFoundError:
            print(f"   ❌ ERROR: Model file not found at 'robot_nav/models/PPO/best_checkpoint/{phase3_model_name}_policy.pth'.")
            print("   Please ensure the Phase 3 model was trained and saved correctly.")
            return
    
    # Early stopping parameters
    patience = 10
    patience_counter = 0
    best_avg_reward = -np.inf
    
    epoch = 0
    episode_count = 0
    episode_count_since_last_train = 0
    
    sim = OtterSIM(
        world_file=random.choice(phase4_worlds),
        disable_plotting=True, enable_phase1=True, max_steps=max_steps,
        cr_method='chun', w_efficiency=1.0, w_safety=1.0,
        os_speed_for_cr=2.0, ts_speed_for_cr=2.0
    )
    distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
    steps = 0
    
    # Main training loop
    while epoch < max_epochs:
        
        # Calculate angle errors in degrees
        os_heading_deg = math_to_ned_heading(np.degrees(robot_state[2, 0])) # Current heading in NED degrees
        ref_angle_deg = ref_course_angle(sim.start_position, sim.goal_position) # Reference course angle in degrees

        phi_tilde = WrapTo180(os_heading_deg - ref_angle_deg) # Heading error in degrees
        psi_e = phi_tilde  # Using phi_tilde as placeholder for psi_e
        chi_e = phi_tilde  # Using phi_tilde as placeholder for chi_e

        state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max
        )
        action, log_prob, state_val = model.get_action(state, add_noise=True)

        a_in = [
            (action[0] + 1) * 1.5,
            action[1] * 0.1745,
        ]

        distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )

        # Calculate angle errors in degrees
        os_heading_deg = math_to_ned_heading(np.degrees(robot_state[2, 0])) # Current heading in NED degrees
        ref_angle_deg = ref_course_angle(sim.start_position, sim.goal_position) # Reference course angle in degrees

        phi_tilde = WrapTo180(os_heading_deg - ref_angle_deg) # Heading error in degrees
        psi_e = phi_tilde  # Using phi_tilde as placeholder for psi_e
        chi_e = phi_tilde  # Using phi_tilde as placeholder for chi_e

        next_state, terminal = model.prepare_state(
            distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max
        )

        model.buffer.add(
            state, action, reward, terminal, next_state, log_prob, state_val
        )
        steps += 1

        if terminal or steps >= max_steps:
            episode_count += 1
            episode_count_since_last_train += 1
            
            if episode_count_since_last_train >= train_every_n_episodes:
                print(f"\n🔄 Training on {episode_count_since_last_train} episodes of experience...")
                model.train(None, training_iterations, batch_size)
                episode_count_since_last_train = 0
            
            if episode_count % episodes_per_epoch == 0:
                epoch += 1
                avg_reward, avg_goal, avg_col = evaluate(model, epoch, phase4_worlds, nr_eval_episodes, max_steps)
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                    model.save(filename=f"{phase4_model_name}_BEST", directory=Path("robot_nav/models/PPO/best_checkpoint"))
                    print(f"🏆 New best model saved with avg reward: {best_avg_reward:.2f}")
                else:
                    patience_counter += 1
                    print(f"📉 No improvement for {patience_counter}/{patience} epochs. Best reward: {best_avg_reward:.2f}")

                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                    break
            
            sim.world_file = random.choice(phase4_worlds)
            distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
            steps = 0


def evaluate(model, epoch, world_files, eval_episodes, max_steps):
    print("\n" + "=" * 60)
    print(f"📈 EVALUATION | Epoch {epoch}")
    print("=" * 60)
    
    total_rewards = []
    total_steps = []
    goal_reached_count = 0
    collision_count = 0

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        
        selected_world = random.choice(world_files)
        sim = OtterSIM(
            world_file=selected_world,
            disable_plotting=True, enable_phase1=True, max_steps=max_steps,
            cr_method='chun', w_efficiency=1.0, w_safety=1.0,
            os_speed_for_cr=2.0, ts_speed_for_cr=2.0
        )
        
        distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.reset()
        
        ep_reward = 0
        for s in range(max_steps):
            # Calculate angle errors in degrees
            os_heading_deg = math_to_ned_heading(np.degrees(robot_state[2, 0])) # Current heading in NED degrees
            ref_angle_deg = ref_course_angle(sim.start_position, sim.goal_position) # Reference course angle in degrees

            phi_tilde = WrapTo180(os_heading_deg - ref_angle_deg) # Heading error in degrees
            psi_e = phi_tilde  # Using phi_tilde as placeholder for psi_e
            chi_e = phi_tilde  # Using phi_tilde as placeholder for chi_e

            state, _ = model.prepare_state(
                distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max
            )
            action, _, _ = model.get_action(state, add_noise=False)
            a_in = [
                (action[0] + 1) * 1.5,
                action[1] * 0.1745,
            ]
            distance, y_e, collision, goal, a, reward, robot_state, CR_max = sim.step(u_ref=a_in[0], r_ref=a_in[1])
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
    collision_rate = collision_count / nr_eval_episodes

    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    model.writer.add_scalar("eval/goal_rate", goal_rate, epoch)
    model.writer.add_scalar("eval/collision_rate", collision_rate, epoch)

    print(f"\n📊 Eval Results: Avg Reward: {avg_reward:.2f} | Goal Rate: {goal_rate:.2f} | Collision Rate: {collision_rate:.2f} | Avg Steps: {avg_steps:.1f}")
    return avg_reward, goal_rate, collision_rate

if __name__ == "__main__":
    main()