import irsim
import matplotlib.pyplot as plt # Import matplotlib for saving figure

env = irsim.make('verify_case_00.yaml') # Removed disable_all_plot=True to ensure save_figure works
for i in range(1000):

    env.step()
    env.render(0.05)
    
    # Check if current time is around 20s (assuming env.step_time is 0.1s, so 200 steps)
    if i == 199: # Adjusted to 199 for closer to 20.0s
        save_path = 'empty_world_20s.png'
        print(f"Saving figure at step {i} (t={i*env.step_time:.1f}s) to {save_path} with dpi=300")
        env.save_figure(save_path, dpi=300)

    if env.done():
        break

env.end(3)
