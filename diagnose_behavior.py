"""
Diagnostic script to analyze Otter USV behavior and verify the problem.

This script will:
1. Record trajectory (x, y positions)
2. Record velocities (u_actual, r_actual)
3. Record actions (u_ref, r_ref)
4. Analyze patterns
5. Generate plots
"""

import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import numpy as np
import matplotlib.pyplot as plt
from robot_nav.SIM_ENV.otter_sim import OtterSIM
import time

def diagnose_behavior(num_steps=500, save_plots=True):
    """
    Run simulation and record all data for analysis.
    """
    print("=" * 60)
    print("🔬 DIAGNOSTIC: Analyzing Otter USV Behavior")
    print("=" * 60)
    
    # Initialize environment
    sim = OtterSIM(
        world_file="/home/hyo/DRL-otter-navigation/robot_nav/worlds/imazu_scenario/s1.yaml",
        disable_plotting=False,  # Enable visualization!
        enable_phase1=True
    )
    
    # Reset
    sim.reset()
    
    # Data storage
    data = {
        'x': [],
        'y': [],
        'psi': [],
        'u_actual': [],
        'r_actual': [],
        'u_ref': [],
        'r_ref': [],
        'reward': [],
        'distance': [],
        'time': []
    }
    
    # Fixed action for testing
    # Test 1: Turning test (yaw command only)
    u_ref = 1.5           # no forward surge
    r_ref = 0.1745        # ~10 deg/s turn
    
    print(f"\nTest: Turning action (yaw-only)")
    print(f"  u_ref = {u_ref:.2f} m/s")
    print(f"  r_ref = {r_ref:.3f} rad/s")
    print(f"  Steps = {num_steps}")
    print(f"  Duration = {num_steps * sim.dt:.1f} seconds")
    print()
    
    # Run simulation
    for step in range(num_steps):
        # Step
        scan, distance, cos, sin, collision, goal, action, reward, robot_state = sim.step(
            u_ref=u_ref, r_ref=r_ref
        )
        
        # Get robot state
        robot_state = sim.env.robot.state
        x = robot_state[0, 0]
        y = robot_state[1, 0]
        psi = robot_state[2, 0]
        
        # Record data
        data['x'].append(x)
        data['y'].append(y)
        data['psi'].append(psi)
        data['u_actual'].append(robot_state[3, 0])
        data['r_actual'].append(robot_state[5, 0])
        data['u_ref'].append(action[0])
        data['r_ref'].append(action[1])
        data['reward'].append(reward)
        data['distance'].append(distance)
        data['time'].append(step * sim.dt)
        
        # Print progress
        if step % 50 == 0:
            print(f"Step {step:3d}: pos=({x:6.2f}, {y:6.2f}), "
                  f"u={robot_state[3, 0]:5.2f}, r={robot_state[5, 0]:6.3f}, "
                  f"reward={reward:7.2f}")
        
        # Check termination
        if collision or goal:
            print(f"\nTerminated at step {step}")
            print(f"  Collision: {collision}")
            print(f"  Goal: {goal}")
            break
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    # 1. Trajectory analysis
    total_distance = np.sqrt(
        (data['x'][-1] - data['x'][0])**2 + 
        (data['y'][-1] - data['y'][0])**2
    )
    path_length = np.sum(np.sqrt(
        np.diff(data['x'])**2 + 
        np.diff(data['y'])**2
    ))
    
    print(f"\n1. Trajectory:")
    print(f"   Start: ({data['x'][0]:.2f}, {data['y'][0]:.2f})")
    print(f"   End:   ({data['x'][-1]:.2f}, {data['y'][-1]:.2f})")
    print(f"   Straight-line distance: {total_distance:.2f} m")
    print(f"   Actual path length: {path_length:.2f} m")
    print(f"   Efficiency: {total_distance/path_length*100:.1f}%")
    
    # Check for circular motion
    x_std = np.std(data['x'])
    y_std = np.std(data['y'])
    if x_std < 5 and y_std < 5 and path_length > 20:
        print(f"   ⚠️ WARNING: Circular motion detected!")
        print(f"      X std: {x_std:.2f}, Y std: {y_std:.2f}")
    
    # 2. Velocity analysis
    print(f"\n2. Velocities:")
    print(f"   u_ref (command): {u_ref:.2f} m/s")
    print(f"   r_ref (command): {r_ref:.3f} rad/s (turning)")
    print(f"   u_actual (mean): {np.mean(data['u_actual']):.2f} m/s")
    print(f"   u_actual (final): {data['u_actual'][-1]:.2f} m/s")
    print(f"   r_actual (mean): {np.mean(data['r_actual']):.3f} rad/s")
    print(f"   r_actual (std): {np.std(data['r_actual']):.3f} rad/s")
    
    # Check if yaw rate converged (to r_ref)
    settling_time = None
    if abs(r_ref) > 1e-6:
        for i, r in enumerate(data['r_actual']):
            if abs(r - r_ref) < 0.1 * abs(r_ref):  # Within 10%
                settling_time = data['time'][i]
                break
    
    if settling_time is not None:
        print(f"   Yaw-rate settling time (90%): {settling_time:.2f} s")
    else:
        print(f"   ⚠️ WARNING: Yaw rate never converged to r_ref!")
    
    # 3. Reward analysis
    print(f"\n3. Rewards:")
    print(f"   Mean: {np.mean(data['reward']):.2f}")
    print(f"   Std: {np.std(data['reward']):.2f}")
    print(f"   Total: {np.sum(data['reward']):.2f}")
    print(f"   Min: {np.min(data['reward']):.2f}")
    print(f"   Max: {np.max(data['reward']):.2f}")
    
    # Check if reward is constant
    if np.std(data['reward']) < 1:
        print(f"   ⚠️ WARNING: Reward is nearly constant!")
    
    # 4. Goal distance
    print(f"\n4. Progress:")
    print(f"   Initial distance: {data['distance'][0]:.2f} m")
    print(f"   Final distance: {data['distance'][-1]:.2f} m")
    print(f"   Progress: {data['distance'][0] - data['distance'][-1]:.2f} m")
    
    if data['distance'][-1] > data['distance'][0] * 0.9:
        print(f"   ⚠️ WARNING: Little to no progress toward goal!")
    
    # Generate plots
    if save_plots:
        print(f"\n5. Generating plots...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Trajectory
        ax = axes[0, 0]
        ax.plot(data['x'], data['y'], 'b-', linewidth=1)
        ax.plot(data['x'][0], data['y'][0], 'go', markersize=10, label='Start')
        ax.plot(data['x'][-1], data['y'][-1], 'ro', markersize=10, label='End')
        ax.plot(0, 0, 'r*', markersize=15, label='Goal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Plot 2: Surge velocity
        ax = axes[0, 1]
        ax.plot(data['time'], data['u_ref'], 'r--', label='u_ref (command)')
        ax.plot(data['time'], data['u_actual'], 'b-', label='u_actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surge velocity (m/s)')
        ax.set_title('Surge Velocity Tracking')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Yaw rate
        ax = axes[0, 2]
        ax.plot(data['time'], data['r_ref'], 'r--', label='r_ref (command)')
        ax.plot(data['time'], data['r_actual'], 'b-', label='r_actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw rate (rad/s)')
        ax.set_title('Yaw Rate Tracking')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Heading
        ax = axes[1, 0]
        ax.plot(data['time'], np.rad2deg(data['psi']), 'b-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Heading Angle')
        ax.grid(True)
        
        # Plot 5: Reward
        ax = axes[1, 1]
        ax.plot(data['time'], data['reward'], 'b-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Step')
        ax.grid(True)
        
        # Plot 6: Distance to goal
        ax = axes[1, 2]
        ax.plot(data['time'], data['distance'], 'b-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to goal (m)')
        ax.set_title('Distance to Goal')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/hyo/DRL-otter-navigation/diagnostic_plots.png', dpi=150)
        print(f"   ✓ Plots saved to diagnostic_plots.png")
    
    print("\n" + "=" * 60)
    return data

def test_different_scenarios():
    """
    Test multiple scenarios to understand behavior
    """
    print("\n" + "=" * 60)
    print("🧪 TESTING MULTIPLE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        ("Forward only", 2.0, 0.0),
        ("Turn right", 1.5, -0.1),
        ("Turn left", 1.5, 0.1),
        ("Sharp turn", 1.0, -0.15),
    ]
    
    sim = OtterSIM(
        world_file="/home/hyo/DRL-otter-navigation/robot_nav/worlds/imazu_scenario/s1.yaml",
        disable_plotting=False,
        enable_phase1=True
    )
    sim.reset()
    
    for name, u_ref, r_ref in scenarios:
        print(f"\nScenario: {name}")
        print(f"  Commands: u={u_ref:.2f}, r={r_ref:.2f}")
    
        
        # Run for 100 steps
        total_reward = 0
        final_pos = None
        
        for step in range(200):
            _, distance, _, _, collision, goal, _, reward, robot_state = sim.step(u_ref, r_ref)
            total_reward += reward
            
            if step == 199:
                robot_state = sim.env.robot.state
                final_pos = (robot_state[0, 0], robot_state[1, 0])
                final_vel = robot_state[3, 0], robot_state[5, 0]
        
        print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        print(f"  Final velocity: u={final_vel[0]:.2f}, r={final_vel[1]:.3f}")
        print(f"  Final yaw rate: {robot_state[5, 0]:.3f} rad/s")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final distance to goal: {distance:.2f} m")

def main():
    """Main diagnostic function"""
    print("\n🔬 Starting comprehensive diagnostics...\n")
    
    # # Test 1: Detailed single run
    print("=" * 60)
    print("TEST 1: Detailed Single Run Analysis")
    print("=" * 60)
    data = diagnose_behavior(num_steps=200, save_plots=True)
    
    # Test 2: Multiple scenarios
    # test_different_scenarios()
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nCheck the plots: diagnostic_plots.png")
    print("\nBased on the results:")
    print("1. Does the robot move in circles? (Check trajectory)")
    print("2. Does velocity converge? (Check settling time)")
    print("3. Is reward meaningful? (Check reward variation)")
    print("4. Is there progress toward goal? (Check distance)")

if __name__ == "__main__":
    main()
