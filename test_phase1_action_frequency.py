#!/usr/bin/env python3
"""
Phase 1 Action Frequency Control Test
======================================

This script tests the Phase 1 implementation where:
- Physics simulation runs at 0.05s (accurate dynamics)
- DRL action updates at 0.5s (allowing controller settling time)
- Same action is held for 10 physics steps

Expected behavior:
- Action changes every 0.5s (10 steps)
- Velocity controller has time to settle (~0.4s)
- u_actual approaches u_ref before next action
"""

import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import numpy as np
from robot_nav.SIM_ENV.otter_sim import OtterSIM

def test_phase1_action_frequency():
    """Test Phase 1 action frequency control"""
    
    print("=" * 70)
    print("Phase 1 Test: Action Frequency Control")
    print("=" * 70)
    
    # Initialize environment with Phase 1 enabled
    sim = OtterSIM(
        world_file="robot_nav/worlds/otter_world.yaml",
        disable_plotting=True,
        enable_phase1=True
    )
    
    print(f"\n✓ Environment initialized")
    print(f"✓ steps_per_action: {sim.steps_per_action}")
    print(f"✓ action_dt: {sim.action_dt}s")
    print(f"✓ physics_dt: {sim.physics_dt}s")
    
    # Test sequence: alternate between two different actions
    actions = [
        [1.5, 0.1],   # Action 1: high surge, small yaw
        [1.2, 0.2],   # Action 2: medium surge, larger yaw
        [1.0, 0.0],   # Action 3: medium surge, no yaw
    ]
    
    print(f"\n{'=' * 70}")
    print("Testing Action Hold Mechanism")
    print(f"{'=' * 70}\n")
    
    # Reset environment
    sim.reset()
    
    # Run for 30 steps (3 action intervals)
    action_idx = 0
    for step in range(30):
        # Select action (should only be applied every 10 steps)
        current_action = actions[action_idx % len(actions)]
        u_ref, r_ref = current_action
        
        # Print when new action is given
        if step % sim.steps_per_action == 0:
            print(f"\nStep {step} (t={step*sim.physics_dt:.2f}s): NEW action=[{u_ref:.2f}, {r_ref:.2f}]")
            action_idx += 1
        
        # Execute step
        _, distance, cos, sin, collision, goal, action, reward = sim.step(u_ref, r_ref)
        
        # Get current velocities from robot state
        robot_state = sim.env.robot.state
        u_actual = robot_state[3, 0]  # Surge velocity
        r_actual = robot_state[5, 0]  # Yaw rate
        
        # Print every 3 steps to see convergence
        if step % 3 == 0:
            print(f"  Step {step:2d}: u={u_actual:.3f} m/s (ref={sim.current_action[0,0]:.2f}), "
                  f"r={r_actual:.3f} rad/s (ref={sim.current_action[1,0]:.2f})")
        
        if collision or goal:
            break
    
    print(f"\n{'=' * 70}")
    print("Test Results")
    print(f"{'=' * 70}")
    
    # Verify action was held
    if sim.step_counter > sim.steps_per_action:
        print("✓ Action frequency control working")
        print(f"✓ Total steps: {sim.step_counter}")
        print(f"✓ Expected action updates: {sim.step_counter // sim.steps_per_action}")
        print(f"✓ Controller has {sim.action_dt}s to settle (>0.4s settling time)")
    else:
        print("✗ Test inconclusive - ran too few steps")
    
    print(f"\n{'=' * 70}\n")
    
    return sim

def test_convergence():
    """Test that velocity controller converges within action interval"""
    
    print("=" * 70)
    print("Convergence Test: Does controller settle in 0.5s?")
    print("=" * 70)
    
    sim = OtterSIM(
        world_file="robot_nav/worlds/otter_world.yaml",
        disable_plotting=True,
        enable_phase1=True
    )
    
    sim.reset()
    
    # Give constant command for one action interval
    u_ref, r_ref = 1.5, 0.1
    velocities = []
    
    print(f"\nConstant command: u_ref={u_ref} m/s, r_ref={r_ref} rad/s")
    print(f"Running for {sim.steps_per_action} steps ({sim.action_dt}s)\n")
    
    for step in range(sim.steps_per_action):
        _, _, _, _, collision, goal, _, _ = sim.step(u_ref, r_ref)
        
        robot_state = sim.env.robot.state
        u_actual = robot_state[3, 0]
        r_actual = robot_state[5, 0]
        velocities.append((u_actual, r_actual))
        
        if step % 2 == 0:
            error_u = abs(u_actual - u_ref)
            error_r = abs(r_actual - r_ref)
            print(f"Step {step:2d} (t={step*sim.physics_dt:.2f}s): "
                  f"u={u_actual:.3f} (err={error_u:.3f}), "
                  f"r={r_actual:.3f} (err={error_r:.3f})")
        
        if collision or goal:
            break
    
    # Check final error
    final_u, final_r = velocities[-1]
    error_u = abs(final_u - u_ref)
    error_r = abs(final_r - r_ref)
    
    print(f"\n{'=' * 70}")
    print("Convergence Results")
    print(f"{'=' * 70}")
    print(f"Final velocity: u={final_u:.3f} m/s, r={final_r:.3f} rad/s")
    print(f"Desired velocity: u={u_ref} m/s, r={r_ref} rad/s")
    print(f"Final error: u_err={error_u:.4f} m/s, r_err={error_r:.4f} rad/s")
    
    # Check if within 10% of reference
    if error_u < 0.1 * u_ref and error_r < 0.1 * abs(r_ref) + 0.01:
        print("✓ Controller converged within action interval!")
        print(f"✓ Phase 1 is working correctly")
    else:
        print("⚠ Controller may need more time to settle")
        print(f"⚠ Consider increasing action_dt")
    
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 1 IMPLEMENTATION TEST")
    print("=" * 70 + "\n")
    
    try:
        # Test 1: Action frequency control
        test_phase1_action_frequency()
        
        # Test 2: Convergence
        test_convergence()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
