import sys
import os

# Ensure the local irsim package is used
sys.path.insert(0, '/home/hyo/ir-sim')

import irsim
import matplotlib.pyplot as plt # Import matplotlib for saving figure

env = irsim.make('verify_case_01.yaml') # Removed disable_all_plot=True to ensure save_figure works
for i in range(1000):

    env.step()
    
    # Update Info Box manually for demonstration
    if hasattr(env.plot, 'update_info_box'):
        robot_state = env.robot.state
        info_str = f"u: {robot_state[3, 0]:.2f} m/s\nr: {robot_state[5, 0]:.3f} rad/s"
        env.plot.update_info_box(info_str)

    env.render(0.05)
    
    # Check if current time is around 20s (assuming env.step_time is 0.1s, so 200 steps)
    if i == 199: # Adjusted to 199 for closer to 20.0s
        save_path = 'phase2_world_20s.png'
        print(f"Saving figure at step {i} (t={i*env.step_time:.1f}s) to {save_path} with dpi=300")
        env.save_figure(save_path, dpi=300)

    if env.done():
        break

env.end(3)
