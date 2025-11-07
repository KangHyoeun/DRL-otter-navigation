from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from utils import get_buffer


def main():
    """Test function for Otter USV with Episode 1 special behavior"""

    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 369             
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"🚀 CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   Using CPU (slower training)")
    max_epochs = 1  # Only run 1 epoch
    epoch = 0
    episodes_per_epoch = 10  # Only run 10 episodes
    episode = 0
    train_every_n = 2           # Train every n episodes
    training_iterations = 80   # Batches per training cycle
    batch_size = 128
    max_steps = 500             # Max steps per episode
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5
    
    # Initialize model
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="otter_CNNTD3_imazu_scenario",
    )
    
    # Initialize simulation with performance optimizations
    print("🔧 Performance Settings:")
    print("   - Plotting: ENABLED (for testing)")
    print("   - Phase 1: ENABLED (action frequency control)")
    print("   - Max steps: 1000 (longer episodes)")
    print("   - Episode 1: SPECIAL BEHAVIOR (straight to goal)")
    sim = OtterSIM(world_file="/worlds/imazu_scenario/s1.yaml", disable_plotting=True, enable_phase1=True)
    
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
    )  # get the initial step state
    
    # Performance monitoring
    import time
    episode_start_time = time.time()
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, robot_state
        )  # get state a state representation from returned data from the environment
        
        # Episodes 2+: Normal DRL action
        action = model.get_action(np.array(state), True)  # get an action from the model
        a_in = [
            (action[0] + 1) * 1.5,
            action[1] * 0.1745,
        ]  
        latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )  # get data from the environment
        
        # Log robot state and actions for each timestep
        robot_state = sim.env.robot.state
        print(f"Step {steps}: State=[{robot_state[0,0]:.2f}, {robot_state[1,0]:.2f}, {robot_state[2,0]:.2f}, "
              f"{robot_state[3,0]:.2f}, {robot_state[4,0]:.2f}, {robot_state[5,0]:.2f}, "
              f"{robot_state[6,0]:.2f}, {robot_state[7,0]:.2f}] "
              f"Actions=[{a_in[0]:.2f}, {a_in[1]:.2f}]")
        
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, robot_state
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            # Performance monitoring
            episode_time = time.time() - episode_start_time
            print(f"📊 Episode {episode + 1} completed in {episode_time:.2f}s ({steps} steps)")
            
            latest_scan, distance, cos, sin, collision, goal, a, reward, robot_state = sim.reset()
            episode += 1
            episode_start_time = time.time()  # Reset timer for next episode
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                    max_lin_vel=3.0,
                    max_ang_vel=0.1745,
                    goal_reward=3000,
                    distance_norm=20,
                    time_step=0.1,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded
            episode = 0
            epoch += 1


if __name__ == "__main__":
    main()