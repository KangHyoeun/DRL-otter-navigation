import numpy as np
import matplotlib.pyplot as plt
import time
import os
from math import degrees, radians
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.geometry import math_to_maritime_position, ned_to_math_heading

# Set matplotlib backend to Agg for non-interactive plotting
plt.switch_backend('Agg')

def visualize_grid_map(world_file="./robot_nav/worlds/verify_scenario/verify_case_18.yaml", num_steps=500):
    print(f"\n" + "=" * 60)
    print("📊 Grid Map Visualization Test")
    print(f"   World File: {world_file}")
    print("=" * 60)

    # Initialize OtterSIM with plotting disabled for IR-SIM itself
    sim = OtterSIM(
        world_file=world_file,
        disable_plotting=True,  # Disable IR-SIM's default plotting
        max_steps=num_steps,
        cr_method='jeon',  # or 'chun'
        w_efficiency=1.0, w_safety=1.0,
        os_speed_for_cr=3.0, ts_speed_for_cr=3.0,
        chi_inf=1.0, k=1.0
    )

    # Create a directory to save images
    output_dir = "grid_map_frames_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving grid map frames to: {output_dir}/")

    # 2 Subplots for 2 Channels
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Initial reset
    sim.reset()
    print("Starting grid map visualization...")

    for step in range(num_steps):
        current_time = step * sim.dt
        
        # Example action (straight ahead at 3 m/s)
        u_ref = 3.0
        r_ref = 0.0

        # Step simulation
        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action_return, reward, robot_state, CR_max, _ = sim.step(u_ref=u_ref, r_ref=r_ref)

        # Calculate current speed from robot_state
        # robot_state: [x, y, theta, u, v, r] (math coordinates usually)
        # u is surge velocity, v is sway velocity
        current_u = robot_state[3, 0]
        current_v = robot_state[4, 0]
        current_speed = np.sqrt(current_u**2 + current_v**2)

        # Manually create enhanced grid
        # Note: sim.prev_position, etc. are updated in step()
        enhanced_grid = sim._create_enhanced_cr_grid(sim.prev_position, sim.prev_heading, current_speed)
        
        # Clear axes
        axes[0].clear()
        axes[1].clear()
        
        # Channel 0: OS Path
        im0 = axes[0].imshow(enhanced_grid[0], origin='lower', cmap='hot', vmin=0, vmax=1.0,
                        extent=[0, sim.grid_lateral, 0, sim.grid_forward])
        axes[0].set_title(f'Channel 0: OS Path (Step {step})')
        axes[0].set_xlabel('Lateral')
        axes[0].set_ylabel('Forward')
        
        # Channel 1: TS Velocity
        im1 = axes[1].imshow(enhanced_grid[1], origin='lower', cmap='hot', vmin=0, vmax=1.0,
                        extent=[0, sim.grid_lateral, 0, sim.grid_forward])
        axes[1].set_title(f'Channel 1: TS Velocity (CR_max: {CR_max:.2f})')
        axes[1].set_xlabel('Lateral')
        
        # Overlay OS position on both
        os_grid_x = sim.grid_lateral / 2.0
        os_grid_y = sim.grid_forward / 2.0
        axes[0].scatter(os_grid_x, os_grid_y, color='blue', marker='o', s=100, label='Own Ship')
        axes[1].scatter(os_grid_x, os_grid_y, color='blue', marker='o', s=100, label='Own Ship')
        
        if step == 0:
            fig.colorbar(im0, ax=axes[0], orientation='vertical', label='Value')
            fig.colorbar(im1, ax=axes[1], orientation='vertical', label='Value')

        plt.tight_layout()
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"grid_map_frame_{step:04d}.png")
        fig.savefig(frame_filename, dpi=300)

        if goal or collision:
            print(f"Simulation terminated at step {step}. Goal: {goal}, Collision: {collision}")
            break

    print("Grid map visualization finished. Frames saved.")
    plt.close(fig) # Close the figure to free up memory

    # Guidance for creating GIF
    print("\nTo create an animation (GIF) from the saved frames, run the following command in your shell:")
    print(f"cd {os.path.join(os.getcwd(), output_dir)} && convert -delay 10 -loop 0 grid_map_frame_*.png grid_map_animation.gif")
    print(" (Note: 'convert' is part of ImageMagick. Install it if you don't have it: sudo apt-get install imagemagick)")

if __name__ == "__main__":
    visualize_grid_map()