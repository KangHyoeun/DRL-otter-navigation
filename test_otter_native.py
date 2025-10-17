#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for IR-SIM native Otter USV integration (Phase 2).

This tests the complete integration of Otter USV into IR-SIM with:
- Custom otter_usv_kinematics
- RobotOtter class
- Full 6-DOF dynamics support
"""

import sys
sys.path.append('/home/hyo/DRL-otter-navigation')

import irsim
import numpy as np
import time

def main():
    """Test IR-SIM native Otter USV integration."""
    
    print("=" * 70)
    print("IR-SIM Native Otter USV Integration Test (Phase 2)")
    print("=" * 70)
    
    # World file with native otter_usv kinematics
    world_file = "/robot_nav/worlds/otter_world_native.yaml"
    
    print(f"\nWorld file: {world_file}")
    print("Initializing environment with native Otter USV kinematics...")
    
    try:
        # Initialize IR-SIM with Otter USV
        env = irsim.make(world_file, disable_all_plot=False, display=True)
        print("✓ Environment initialized successfully!")
        
        # Get robot info
        robot = env.robot
        print(f"\n✓ Robot type: {type(robot).__name__}")
        print(f"✓ Kinematics: {robot.kinematics}")
        print(f"✓ State dimension: {robot.state.shape[0]}")
        print(f"✓ Initial state: {robot.state.T}")
        
        # Check if full dynamics is available
        if hasattr(robot, 'use_full_dynamics'):
            print(f"✓ Full 6-DOF dynamics: {robot.use_full_dynamics}")
        
        # Test simulation
        print("\n" + "=" * 70)
        print("Running simulation test...")
        print("=" * 70)
        
        # Control commands
        u_ref = 1.5  # surge velocity (m/s)
        r_ref = 0.2  # yaw rate (rad/s)
        
        print(f"\nControl commands: u_ref={u_ref} m/s, r_ref={r_ref} rad/s")
        
        for step in range(100):
            # Set velocity commands
            action = np.array([[u_ref], [r_ref]])
            
            # Step environment
            env.step(action_id=0, action=action)
            env.render()
            
            # Print status every 20 steps
            if step % 20 == 0:
                state = env.robot.state
                print(f"\nStep {step}:")
                print(f"  Position: [{state[0,0]:.2f}, {state[1,0]:.2f}, {np.rad2deg(state[2,0]):.1f}°]")
                
                if state.shape[0] >= 6:
                    print(f"  Velocities: u={state[3,0]:.2f} m/s, v={state[4,0]:.2f} m/s, r={np.rad2deg(state[5,0]):.1f}°/s")
                
                if state.shape[0] >= 8:
                    print(f"  Propellers: n1={state[6,0]:.1f} RPM, n2={state[7,0]:.1f} RPM")
            
            # Check collision or goal
            if env.robot.collision:
                print("\n💥 Collision detected!")
                break
            
            if env.robot.arrive:
                print("\n🎯 Goal reached!")
                break
        
        print("\n" + "=" * 70)
        print("Test completed successfully! ✓")
        print("=" * 70)
        
        # Summary
        final_state = env.robot.state
        print("\nFinal state:")
        print(f"  Position: [{final_state[0,0]:.2f}, {final_state[1,0]:.2f}]")
        print(f"  Heading: {np.rad2deg(final_state[2,0]):.1f}°")
        
        if final_state.shape[0] >= 6:
            print(f"  Velocities:")
            print(f"    Surge: {final_state[3,0]:.2f} m/s")
            print(f"    Sway: {final_state[4,0]:.2f} m/s")
            print(f"    Yaw rate: {np.rad2deg(final_state[5,0]):.1f}°/s")
        
        # Wait for user
        print("\nPress Enter to close...")
        input()
        
        # Close environment
        env.end()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
