import numpy as np
import matplotlib.pyplot as plt
import time
import os
from math import degrees, radians
from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core.geometry import math_to_maritime_position, ned_to_math_heading

# Set matplotlib backend to Agg for non-interactive plotting
plt.switch_backend('Agg')

def visualize_grid_map(world_file="robot_nav/worlds/imazu_scenario/imazu_case_18.yaml", num_steps=500):
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
    output_dir = "grid_map_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving grid map frames to: {output_dir}/")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Initial reset to get the first state and grid map
    distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action_return, reward, robot_state, CR_max, cr_grid = sim.reset()

    print("Starting grid map visualization...")

    for step in range(num_steps):
        current_time = step * sim.dt
        
        # Example action (straight ahead at 3 m/s)
        u_ref = 3.0
        r_ref = 0.0

        distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action_return, reward, robot_state, CR_max, cr_grid = sim.step(u_ref=u_ref, r_ref=r_ref)

        # Clear previous plot for animation
        ax.clear() 
        
        # Plot the grid map
        # cr_grid is [Forward, Lateral]. imshow expects [Rows(Y), Cols(X)].
        # So we plot cr_grid directly with origin='lower' (Forward 0 at bottom).
        im = ax.imshow(cr_grid, origin='lower', cmap='hot', vmin=0, vmax=1.0,
                        extent=[0, sim.grid_lateral, 0, sim.grid_forward])
        
        # Add colorbar only once (for the first frame)
        if step == 0:
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Collision Risk (0-1)')

        # Overlay OS position (center of the grid in body frame)
        os_grid_x = sim.grid_lateral / 2.0
        os_grid_y = sim.grid_forward / 2.0
        ax.scatter(os_grid_x, os_grid_y, color='blue', marker='o', s=100, label='Own Ship')
        
        ax.set_title(f'Step: {step}, Time: {current_time:.1f}s, CR_max: {CR_max:.2f}')
        ax.set_xlabel('Lateral (Right) [cells]')
        ax.set_ylabel('Forward [cells]')
        ax.set_xlim(0, sim.grid_lateral)
        ax.set_ylim(0, sim.grid_forward)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        # Save the current frame to a file
        frame_filename = os.path.join(output_dir, f"grid_map_frame_{step:04d}.png")
        fig.savefig(frame_filename)

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