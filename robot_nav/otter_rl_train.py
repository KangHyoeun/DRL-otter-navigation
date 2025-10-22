from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from utils import get_buffer


def main():
    """Main training function for Otter USV"""

    # Hyperparameters
    action_dim = 2           
    max_action = 1
    state_dim = 185             
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"🚀 CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   Using CPU (slower training)")
    nr_eval_episodes = 10
    max_epochs = 60
    epoch = 0
    episodes_per_epoch = 70
    episode = 0
    train_every_n = 2           # Train every n episodes
    training_iterations = 80   # Batches per training cycle
    batch_size = 128
    max_steps = 2000             # Max steps per episode (Phase 1: 2x for longer episodes)
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
        model_name="otter_CNNTD3",
    )
    
    # Initialize simulation with performance optimizations
    print("🔧 Performance Settings:")
    print("   - Plotting: DISABLED (faster simulation)")
    print("   - Phase 1: ENABLED (action frequency control)")
    print("   - Max steps: 2000 (longer episodes)")
    sim = OtterSIM(world_file="/worlds/otter_world.yaml", disable_plotting=True, enable_phase1=True)
    
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

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        u_ref=0.0, r_ref=0.0
    )  # get the initial step state
    
    # Performance monitoring
    import time
    episode_start_time = time.time()
    
    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment
        
        action = model.get_action(np.array(state), True)  # get an action from the model
        a_in = [
            (action[0] + 1) * 1.5,
            action[1] * 0.3,
        ]  # clip linear velocity to [0, 3.0] m/s range

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            u_ref=a_in[0], r_ref=a_in[1]
        )  # get data from the environment
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
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
            
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1
            episode_start_time = time.time()  # Reset timer for next episode
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)

def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) * 1.5, action[1] * 0.3]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
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


if __name__ == "__main__":
    main()
