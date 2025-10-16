#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otter USV Simulation Environment for DRL Navigation Training

This module integrates the velocity-controlled Otter USV with the DRL framework,
providing a marine vehicle simulation environment for navigation training.
"""

import sys
import os
import numpy as np
import math
import random
from typing import Tuple, List, Optional

# Add PythonVehicleSimulator to path
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

from python_vehicle_simulator.vehicles import otter
from python_vehicle_simulator.lib.gnc import Rzyx, ssa


class OtterUSVSim:
    """
    Otter USV Simulation Environment for DRL Navigation Training.
    
    This class provides a marine vehicle simulation environment that:
    - Uses realistic Otter USV dynamics (Fossen's model)
    - Accepts mobile robot-style velocity commands [linear, angular]
    - Provides LIDAR-like sensor simulation
    - Integrates with DRL training framework
    """
    
    def __init__(self, 
                 world_size: Tuple[float, float] = (10.0, 10.0),
                 max_velocity: float = 2.0,
                 max_angular_velocity: float = 1.0,
                 sensor_range: float = 7.0,
                 sensor_angles: int = 180,
                 goal_threshold: float = 0.5,
                 collision_threshold: float = 0.3):
        """
        Initialize the Otter USV simulation environment.
        
        Args:
            world_size: (width, height) of the simulation world
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            sensor_range: Maximum sensor range (m)
            sensor_angles: Number of sensor rays
            goal_threshold: Distance threshold for goal achievement
            collision_threshold: Distance threshold for collision detection
        """
        self.world_width, self.world_height = world_size
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.sensor_range = sensor_range
        self.sensor_angles = sensor_angles
        self.goal_threshold = goal_threshold
        self.collision_threshold = collision_threshold
        
        # Initialize Otter USV with velocity control
        self.vehicle = otter('velocityControl', r=1.5)
        
        # Simulation parameters
        self.dt = 0.02  # 50 Hz simulation
        self.max_steps = 500
        
        # State variables
        self.eta = np.zeros(6)  # [x, y, z, phi, theta, psi] - position and orientation
        self.nu = np.zeros(6)   # [u, v, w, p, q, r] - velocities
        self.u_actual = np.zeros(2)  # [n1, n2] - propeller states
        
        # Environment state
        self.obstacles = []
        self.goal = np.array([9.0, 9.0])
        self.start = np.array([1.0, 1.0])
        
        # Tracking variables
        self.step_count = 0
        self.previous_distance = None
        
        # Initialize environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup the simulation environment with obstacles and initial state."""
        # Create static obstacles (simplified for marine environment)
        self.obstacles = [
            {'type': 'circle', 'center': [5.0, 2.0], 'radius': 0.8},
            {'type': 'rectangle', 'center': [8.0, 5.0], 'size': [1.0, 1.2]},
            {'type': 'rectangle', 'center': [1.0, 8.0], 'size': [0.5, 2.1]},
            {'type': 'circle', 'center': [3.0, 6.0], 'radius': 0.6},
            {'type': 'circle', 'center': [7.0, 3.0], 'radius': 0.4},
        ]
        
        # Set initial position
        self.eta[0] = self.start[0]  # x
        self.eta[1] = self.start[1]   # y
        self.eta[5] = 0.0  # psi (heading)
        
        # Reset vehicle states
        self.nu = np.zeros(6)
        self.u_actual = np.zeros(2)
        self.step_count = 0
        self.previous_distance = np.linalg.norm(self.goal - self.start)
    
    def step(self, lin_velocity: float = 0.0, ang_velocity: float = 0.0) -> Tuple:
        """
        Perform one simulation step.
        
        Args:
            lin_velocity: Linear velocity command (m/s)
            ang_velocity: Angular velocity command (rad/s)
            
        Returns:
            Tuple containing: (laser_scan, distance, cos, sin, collision, goal, action, reward)
        """
        # Clip velocities to limits
        lin_velocity = np.clip(lin_velocity, 0.0, self.max_velocity)
        ang_velocity = np.clip(ang_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Convert to Otter USV velocity control
        u_ref = lin_velocity
        r_ref = ang_velocity
        
        # Get control commands from Otter USV
        u_control = self.vehicle.velocityControl(self.nu, u_ref, r_ref, self.dt)
        
        # Update vehicle dynamics
        self.nu, self.u_actual = self.vehicle.dynamics(self.eta, self.nu, self.u_actual, u_control, self.dt)
        
        # Update position (simple integration for 2D navigation)
        self.eta[0] += self.nu[0] * np.cos(self.eta[5]) * self.dt  # x
        self.eta[1] += self.nu[0] * np.sin(self.eta[5]) * self.dt  # y
        self.eta[5] += self.nu[5] * self.dt  # psi
        
        # Keep within world bounds
        self.eta[0] = np.clip(self.eta[0], 0.1, self.world_width - 0.1)
        self.eta[1] = np.clip(self.eta[1], 0.1, self.world_height - 0.1)
        
        # Get sensor data
        laser_scan = self._get_laser_scan()
        
        # Calculate goal-related metrics
        goal_vector = self.goal - np.array([self.eta[0], self.eta[1]])
        distance = np.linalg.norm(goal_vector)
        
        # Calculate orientation to goal
        pose_vector = [np.cos(self.eta[5]), np.sin(self.eta[5])]
        cos, sin = self._cossin(pose_vector, goal_vector.tolist())
        
        # Check for collision
        collision = self._check_collision()
        
        # Check for goal achievement
        goal_reached = distance < self.goal_threshold
        
        # Calculate reward
        action = [lin_velocity, ang_velocity]
        reward = self._calculate_reward(goal_reached, collision, action, laser_scan, distance)
        
        # Update tracking
        self.previous_distance = distance
        self.step_count += 1
        
        return laser_scan, distance, cos, sin, collision, goal_reached, action, reward
    
    def _get_laser_scan(self) -> List[float]:
        """
        Simulate LIDAR sensor by casting rays and checking for obstacles.
        
        Returns:
            List of distances to obstacles for each sensor angle
        """
        scan = []
        current_pos = np.array([self.eta[0], self.eta[1]])
        
        for i in range(self.sensor_angles):
            # Calculate ray direction
            angle = (i / self.sensor_angles) * 2 * np.pi - np.pi
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Cast ray and find intersection with obstacles
            min_distance = self.sensor_range
            
            for obstacle in self.obstacles:
                if obstacle['type'] == 'circle':
                    distance = self._ray_circle_intersection(
                        current_pos, ray_direction, 
                        obstacle['center'], obstacle['radius']
                    )
                elif obstacle['type'] == 'rectangle':
                    distance = self._ray_rectangle_intersection(
                        current_pos, ray_direction,
                        obstacle['center'], obstacle['size']
                    )
                else:
                    continue
                
                if distance is not None and distance < min_distance:
                    min_distance = distance
            
            # Add noise (optional)
            noise = np.random.normal(0, 0.05)
            min_distance = max(0.0, min_distance + noise)
            
            scan.append(min_distance)
        
        return scan
    
    def _ray_circle_intersection(self, origin: np.ndarray, direction: np.ndarray, 
                               center: List[float], radius: float) -> Optional[float]:
        """Calculate intersection of ray with circle."""
        oc = origin - np.array(center)
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None
    
    def _ray_rectangle_intersection(self, origin: np.ndarray, direction: np.ndarray,
                                   center: List[float], size: List[float]) -> Optional[float]:
        """Calculate intersection of ray with rectangle."""
        # Simplified rectangle intersection
        # For now, treat as circle with average radius
        avg_radius = (size[0] + size[1]) / 4
        return self._ray_circle_intersection(origin, direction, center, avg_radius)
    
    def _check_collision(self) -> bool:
        """Check if the vehicle has collided with any obstacle."""
        current_pos = np.array([self.eta[0], self.eta[1]])
        
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                distance = np.linalg.norm(current_pos - np.array(obstacle['center']))
                if distance < obstacle['radius'] + self.collision_threshold:
                    return True
            elif obstacle['type'] == 'rectangle':
                # Simplified rectangle collision check
                center = np.array(obstacle['center'])
                size = np.array(obstacle['size'])
                if (abs(current_pos[0] - center[0]) < size[0]/2 + self.collision_threshold and
                    abs(current_pos[1] - center[1]) < size[1]/2 + self.collision_threshold):
                    return True
        
        return False
    
    def _calculate_reward(self, goal: bool, collision: bool, action: List[float], 
                        laser_scan: List[float], distance: float) -> float:
        """
        Calculate reward for the current state.
        
        Args:
            goal: Whether goal is reached
            collision: Whether collision occurred
            action: Action taken [linear_velocity, angular_velocity]
            laser_scan: LIDAR scan data
            distance: Distance to goal
            
        Returns:
            Reward value
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        
        # Progress reward
        progress_reward = 0.0
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - distance
            progress_reward = distance_improvement * 10.0
        
        # Safety reward
        min_scan = min(laser_scan)
        safety_reward = 0.0
        if min_scan < 0.3:
            safety_reward = -20.0
        elif min_scan < 0.5:
            safety_reward = -5.0
        elif min_scan > 1.0:
            safety_reward = 2.0
        
        # Efficiency reward
        efficiency_reward = 0.0
        if action[0] > 0.1:  # Moving forward
            efficiency_reward += 0.5
        if abs(action[1]) < 0.5:  # Smooth turning
            efficiency_reward += 0.2
        
        # Time penalty
        time_penalty = -0.01
        
        # Distance reward
        distance_reward = 0.0
        if distance < 1.0:
            distance_reward = 5.0
        elif distance < 2.0:
            distance_reward = 2.0
        
        total_reward = (progress_reward + safety_reward + efficiency_reward + 
                       time_penalty + distance_reward)
        
        return total_reward
    
    def _cossin(self, pose_vector: List[float], goal_vector: List[float]) -> Tuple[float, float]:
        """Calculate cosine and sine of angle between pose and goal vectors."""
        dot_product = pose_vector[0] * goal_vector[0] + pose_vector[1] * goal_vector[1]
        cross_product = pose_vector[0] * goal_vector[1] - pose_vector[1] * goal_vector[0]
        
        magnitude = np.linalg.norm(goal_vector)
        if magnitude == 0:
            return 0.0, 0.0
        
        cos = dot_product / magnitude
        sin = cross_product / magnitude
        
        return cos, sin
    
    def reset(self, robot_state: Optional[List[float]] = None, 
              robot_goal: Optional[List[float]] = None) -> Tuple:
        """
        Reset the simulation environment.
        
        Args:
            robot_state: Initial robot state [x, y, theta]
            robot_goal: Goal position [x, y]
            
        Returns:
            Initial observation tuple
        """
        # Reset vehicle state
        if robot_state is None:
            self.start = np.array([random.uniform(1, 3), random.uniform(1, 3)])
        else:
            self.start = np.array(robot_state[:2])
        
        if robot_goal is None:
            self.goal = np.array([random.uniform(7, 9), random.uniform(7, 9)])
        else:
            self.goal = np.array(robot_goal[:2])
        
        # Reset vehicle position and orientation
        self.eta[0] = self.start[0]
        self.eta[1] = self.start[1]
        self.eta[5] = robot_state[2] if robot_state and len(robot_state) > 2 else 0.0
        
        # Reset velocities
        self.nu = np.zeros(6)
        self.u_actual = np.zeros(2)
        
        # Reset tracking
        self.step_count = 0
        self.previous_distance = np.linalg.norm(self.goal - self.start)
        
        # Get initial observation
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])
    
    def get_robot_state(self) -> List[float]:
        """Get current robot state [x, y, theta]."""
        return [self.eta[0], self.eta[1], self.eta[5]]
    
    def get_goal(self) -> List[float]:
        """Get current goal position [x, y]."""
        return self.goal.tolist()
    
    def render(self):
        """Render the simulation (placeholder for visualization)."""
        # This could be implemented with matplotlib or other visualization
        pass
