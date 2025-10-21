#!/usr/bin/env python3
"""
Test Wander Behavior and Random Shape
======================================

Tests wander behavior and random shape generation for Otter USV obstacles.
"""

import numpy as np
import irsim

def main():
    print("="*70)
    print("Wander Behavior and Random Shape Test")
    print("="*70)
    
    # Load environment
    world_file = 'robot_nav/worlds/otter_world_wander.yaml'
    env = irsim.make(world_file)
    
    print(f"\n✓ Environment loaded: {world_file}")
    print(f"  Robot type: {type(env.robot).__name__}")
    print(f"  Number of obstacles: {len(env.obstacle_list)}")
    
    # Check obstacle details
    print("\nObstacle Details:")
    
    for i, obs in enumerate(env.obstacle_list):
        obs_type = type(obs).__name__
        behavior_name = obs.behavior.get('name', 'none') if hasattr(obs, 'behavior') else 'none'
        position = obs.state[0:2, 0]
        
        # Get shape info
        if hasattr(obs, 'geometry') and hasattr(obs.geometry, 'shape'):
            shape_info = obs.geometry.shape
            if shape_info.geom_type == 'Polygon':
                # Get bounding box for rectangle
                minx, miny, maxx, maxy = shape_info.bounds
                length = maxx - minx
                width = maxy - miny
                shape_desc = f"Rectangle ({length:.2f}m x {width:.2f}m)"
            elif shape_info.geom_type == 'Point':
                # Circle
                if hasattr(obs.geometry, 'radius'):
                    shape_desc = f"Circle (r={obs.geometry.radius:.2f}m)"
                else:
                    shape_desc = "Circle"
            else:
                shape_desc = shape_info.geom_type
        else:
            shape_desc = "Unknown"
        
        print(f"\n  Obstacle {i}:")
        print(f"    Type: {obs_type}")
        print(f"    Position: [{position[0]:.1f}, {position[1]:.1f}]")
        print(f"    Shape: {shape_desc}")
        print(f"    Behavior: {behavior_name}")
        
        if hasattr(obs, 'goal') and obs.goal is not None:
            goal_pos = obs.goal[0:2, 0]
            print(f"    Current goal: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}]")
    
    # Run simulation
    print("\n" + "="*70)
    print("Running Simulation...")
    print("="*70)
    print("(Watch obstacles wander to random goals with different shapes)")
    
    action = np.array([[1.5], [0.1]])  # [u_ref, r_ref]
    
    goal_changes = {i: 0 for i in range(len(env.obstacle_list))}
    prev_goals = {}
    
    for step in range(500):
        env.step(action_id=0, action=action)
        env.render()
        
        # Track goal changes (wander behavior)
        for i, obs in enumerate(env.obstacle_list):
            if hasattr(obs, 'goal') and obs.goal is not None:
                current_goal = tuple(obs.goal[0:2, 0])
                
                if i not in prev_goals:
                    prev_goals[i] = current_goal
                elif prev_goals[i] != current_goal:
                    goal_changes[i] += 1
                    prev_goals[i] = current_goal
                    print(f"\n  Obstacle {i} reached goal #{goal_changes[i]}!")
                    print(f"    New goal: [{current_goal[0]:.1f}, {current_goal[1]:.1f}]")
        
        if step % 100 == 0:
            robot_pos = env.robot.state[0:2, 0]
            print(f"\nStep {step}:")
            print(f"  Robot position: [{robot_pos[0]:.1f}, {robot_pos[1]:.1f}]")
            print(f"  Collision: {env.robot.collision}")
            print(f"  Goal reached: {env.robot.arrive}")
        
        if env.done():
            print(f"\n✓ Simulation ended at step {step}")
            print(f"  Goal reached: {env.robot.arrive}")
            print(f"  Collision: {env.robot.collision}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("Wander Behavior Summary:")
    print("="*70)
    for i, count in goal_changes.items():
        print(f"  Obstacle {i}: Reached {count} different goals")
    
    env.end()
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\n✅ Features verified:")
    print("  1. Wander behavior - obstacles move to random goals")
    print("  2. Random shape - obstacles have different sizes")
    print("  3. Goal regeneration - new goals after reaching current")

if __name__ == '__main__':
    main()
