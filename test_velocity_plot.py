#!/usr/bin/env python3
"""
Test velocity tracking visualization for Otter USV.

This script tests the plot_velocity_text() functionality that displays:
- u_ref, r_ref: Desired velocity commands
- u, r: Actual velocities from the Otter dynamics
"""

import numpy as np
import irsim
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("Testing Otter USV Velocity Tracking Visualization")
    print("=" * 70)
    
    # Initialize environment
    world_file = "robot_nav/worlds/otter_world_native.yaml"
    print(f"\nWorld file: {world_file}")
    print("Initializing environment...")
    
    env = irsim.make(world_file)
    
    print(f"✓ Environment initialized")
    print(f"✓ Robot type: {type(env.robot).__name__}")
    print(f"✓ State dimension: {env.robot.state.shape[0]}")
    
    # Enable velocity text plotting
    env.robot.plot_kwargs['show_velocity_text'] = True
    print("\n✓ Velocity text plotting enabled")
    
    # Test with varying commands
    print("\n" + "=" * 70)
    print("Running simulation with varying velocity commands")
    print("=" * 70)
    
    # Define test trajectory
    test_commands = [
        (1.5, 0.2, "Forward with right turn"),
        (1.0, -0.2, "Forward with left turn"),
        (0.5, 0.0, "Slow forward"),
        (2.0, 0.3, "Fast forward with right turn"),
        (0.0, 0.0, "Stop"),
    ]
    
    steps_per_command = 40  # Run each command for 40 steps
    
    for cmd_idx, (u_ref, r_ref, description) in enumerate(test_commands):
        print(f"\n--- Command {cmd_idx + 1}/{len(test_commands)}: {description} ---")
        print(f"    u_ref = {u_ref:.2f} m/s, r_ref = {np.rad2deg(r_ref):.1f}°/s")
        
        for step in range(steps_per_command):
            # Create action
            action = np.array([[u_ref], [r_ref]])
            
            # Step environment
            env.step(action_id=0, action=action)
            env.render()
            
            # Print state every 10 steps
            if step % 10 == 0:
                robot = env.robot
                state = robot.state
                x, y, psi = state[0, 0], state[1, 0], state[2, 0]
                u, r = state[3, 0], state[5, 0]
                
                print(f"    Step {step:2d}: "
                      f"Pos=[{x:5.1f}, {y:5.1f}, {np.rad2deg(psi):5.1f}°], "
                      f"u={u:5.2f}, r={np.rad2deg(r):5.1f}°/s")
            
            # Check for termination
            if env.done():
                print(f"    Terminated at step {step}")
                break
    
    print("\n" + "=" * 70)
    print("Test completed! Check the visualization window.")
    print("=" * 70)
    print("\nVelocity text should be displayed below the robot showing:")
    print("  Cmd: u=X.XX, r=XX.X°/s  (commanded velocities)")
    print("  Act: u=X.XX, r=XX.X°/s  (actual velocities)")
    
    env.end()

if __name__ == "__main__":
    main()
