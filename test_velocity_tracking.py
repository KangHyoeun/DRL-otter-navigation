#!/usr/bin/env python3
"""
Advanced velocity tracking plot for Otter USV.

This creates a real-time plot showing:
1. Robot trajectory in 2D space (top subplot)
2. Surge velocity (u) tracking over time (middle subplot)
3. Yaw rate (r) tracking over time (bottom subplot)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import irsim

class VelocityTracker:
    """
    Track and visualize Otter USV velocity commands vs actual velocities.
    """
    
    def __init__(self, max_history=200):
        """
        Initialize velocity tracker.
        
        Args:
            max_history: Maximum number of time steps to keep in history
        """
        self.max_history = max_history
        
        # History buffers
        self.time_history = deque(maxlen=max_history)
        self.u_ref_history = deque(maxlen=max_history)
        self.u_actual_history = deque(maxlen=max_history)
        self.r_ref_history = deque(maxlen=max_history)
        self.r_actual_history = deque(maxlen=max_history)
        self.x_history = deque(maxlen=max_history)
        self.y_history = deque(maxlen=max_history)
        
        self.current_time = 0.0
        
        # Initialize figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle('Otter USV Velocity Tracking', fontsize=14, fontweight='bold')
        
        # Setup axes
        self._setup_axes()
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def _setup_axes(self):
        """Setup the three subplots."""
        # Trajectory plot (top)
        self.axes[0].set_xlabel('X Position (m)')
        self.axes[0].set_ylabel('Y Position (m)')
        self.axes[0].set_title('Robot Trajectory')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_aspect('equal')
        
        # Surge velocity plot (middle)
        self.axes[1].set_xlabel('Time (s)')
        self.axes[1].set_ylabel('Surge Velocity u (m/s)')
        self.axes[1].set_title('Surge Velocity Tracking')
        self.axes[1].grid(True, alpha=0.3)
        
        # Yaw rate plot (bottom)
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Yaw Rate r (°/s)')
        self.axes[2].set_title('Yaw Rate Tracking')
        self.axes[2].grid(True, alpha=0.3)
    
    def update(self, state, u_ref, r_ref, dt=0.05):
        """
        Update history and plots.
        
        Args:
            state: Robot state [x, y, psi, u, v, r, n1, n2]
            u_ref: Commanded surge velocity (m/s)
            r_ref: Commanded yaw rate (rad/s)
            dt: Time step (s)
        """
        # Extract data
        x = state[0, 0]
        y = state[1, 0]
        u_actual = state[3, 0]
        r_actual = state[5, 0]
        
        # Update time
        self.current_time += dt
        
        # Append to history
        self.time_history.append(self.current_time)
        self.x_history.append(x)
        self.y_history.append(y)
        self.u_ref_history.append(u_ref)
        self.u_actual_history.append(u_actual)
        self.r_ref_history.append(r_ref)
        self.r_actual_history.append(r_actual)
    
    def plot(self):
        """Update the plots."""
        if len(self.time_history) < 2:
            return
        
        # Convert to arrays
        time = np.array(self.time_history)
        x = np.array(self.x_history)
        y = np.array(self.y_history)
        u_ref = np.array(self.u_ref_history)
        u_actual = np.array(self.u_actual_history)
        r_ref = np.rad2deg(np.array(self.r_ref_history))
        r_actual = np.rad2deg(np.array(self.r_actual_history))
        
        # Clear axes
        for ax in self.axes:
            ax.cla()
        
        # Re-setup after clearing
        self._setup_axes()
        
        # Plot trajectory
        self.axes[0].plot(x, y, 'b-', linewidth=2, label='Trajectory')
        self.axes[0].plot(x[0], y[0], 'go', markersize=10, label='Start')
        self.axes[0].plot(x[-1], y[-1], 'ro', markersize=10, label='Current')
        self.axes[0].legend(loc='best')
        
        # Plot surge velocity
        self.axes[1].plot(time, u_ref, 'r--', linewidth=2, label='u_ref (command)')
        self.axes[1].plot(time, u_actual, 'b-', linewidth=2, label='u (actual)')
        self.axes[1].legend(loc='best')
        
        # Plot yaw rate
        self.axes[2].plot(time, r_ref, 'r--', linewidth=2, label='r_ref (command)')
        self.axes[2].plot(time, r_actual, 'b-', linewidth=2, label='r (actual)')
        self.axes[2].legend(loc='best')
        
        # Add error statistics
        u_error = np.mean(np.abs(u_actual - u_ref))
        r_error = np.mean(np.abs(r_actual - r_ref))
        
        self.axes[1].text(0.02, 0.98, f'MAE: {u_error:.3f} m/s',
                         transform=self.axes[1].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.axes[2].text(0.02, 0.98, f'MAE: {r_error:.2f}°/s',
                         transform=self.axes[2].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.pause(0.001)
    
    def close(self):
        """Close the plot."""
        plt.close(self.fig)


def main():
    print("=" * 70)
    print("Otter USV Advanced Velocity Tracking")
    print("=" * 70)
    
    # Initialize environment
    world_file = "robot_nav/worlds/otter_world_native.yaml"
    print(f"\nWorld file: {world_file}")
    print("Initializing environment...")
    
    env = irsim.make(world_file)
    print(f"✓ Environment initialized")
    
    # Initialize tracker
    tracker = VelocityTracker(max_history=300)
    print("✓ Velocity tracker initialized")
    
    # Test commands
    print("\n" + "=" * 70)
    print("Running simulation with varying commands")
    print("=" * 70)
    
    test_commands = [
        (1.5, 0.2, 50, "Forward with right turn"),
        (1.0, -0.2, 50, "Forward with left turn"),
        (0.5, 0.0, 50, "Slow forward"),
        (2.0, 0.3, 50, "Fast forward with turn"),
        (1.0, 0.0, 50, "Steady forward"),
    ]
    
    dt = 0.05
    
    for cmd_idx, (u_ref, r_ref, steps, description) in enumerate(test_commands):
        print(f"\n--- Command {cmd_idx + 1}/{len(test_commands)}: {description} ---")
        print(f"    u_ref = {u_ref:.2f} m/s, r_ref = {np.rad2deg(r_ref):.1f}°/s")
        
        for step in range(steps):
            # Create action
            action = np.array([[u_ref], [r_ref]])
            
            # Step environment
            env.step(action_id=0, action=action)
            
            # Update tracker
            tracker.update(env.robot.state, u_ref, r_ref, dt)
            
            # Update plot every 5 steps
            if step % 5 == 0:
                tracker.plot()
            
            # Print progress every 20 steps
            if step % 20 == 0:
                state = env.robot.state
                u = state[3, 0]
                r = state[5, 0]
                print(f"    Step {step:2d}: u={u:5.2f}, r={np.rad2deg(r):5.1f}°/s")
    
    # Final update
    tracker.plot()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("\nVelocity tracking plots show:")
    print("  - Top: Robot trajectory (X-Y plane)")
    print("  - Middle: Surge velocity (u) command vs actual")
    print("  - Bottom: Yaw rate (r) command vs actual")
    print("  - MAE (Mean Absolute Error) displayed on each plot")
    
    # Keep plot open
    print("\nClosing in 10 seconds...")
    plt.pause(10)
    
    tracker.close()
    env.end()

if __name__ == "__main__":
    main()
