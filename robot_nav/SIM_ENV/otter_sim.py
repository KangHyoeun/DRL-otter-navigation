import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import irsim
import numpy as np
import random
from python_vehicle_simulator.vehicles import otter
from robot_nav.SIM_ENV.sim_env import SIM_ENV

class OtterSIM(SIM_ENV):
    """
    Otter USV simulation environment wrapper.
    
    This class integrates Otter USV dynamics from Python Vehicle Simulator
    with IR-SIM for visualization and sensor simulation.
    
    Attributes:
        env (object): IR-SIM environment instance
        otter (object): Otter USV dynamics model
        eta (np.ndarray): 6-DOF position/orientation [x, y, z, phi, theta, psi]
        nu (np.ndarray): 6-DOF velocities [u, v, w, p, q, r]
        u_actual (np.ndarray): Actual propeller states [n1, n2]
        dt (float): Simulation time step
        robot_goal (np.ndarray): Goal position [x, y, psi]
    """
    
    def __init__(self, world_file="otter_world.yaml", disable_plotting=False):
        """
        Initialize the Otter USV simulation environment.
        
        Args:
            world_file (str): Path to the world configuration YAML file
            disable_plotting (bool): If True, disables rendering and plotting
        """
        # Initialize IR-SIM environment for visualization
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        # Initialize Otter USV with velocity controller
        self.otter = otter('velocityControl', r=1.5)
        
        # Get simulation parameters
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.dt = self.env.step_time
        
        # Initialize Otter state (6-DOF)
        robot_state = self.env.get_robot_state()
        self.eta = np.zeros(6)  # [x, y, z, phi, theta, psi]
        self.eta[0] = robot_state[0].item()  # x
        self.eta[1] = robot_state[1].item()  # y
        self.eta[5] = robot_state[2].item()  # psi (yaw)
        
        self.nu = np.zeros(6)  # [u, v, w, p, q, r]
        self.u_actual = np.zeros(2)  # [n1, n2] propeller states
        
        print("=" * 60)
        print("Otter USV Environment Initialized")
        print("=" * 60)
        print(f"Robot position: [{self.eta[0]:.2f}, {self.eta[1]:.2f}, {self.eta[5]:.2f}]")
        print(f"Goal position: {self.robot_goal}")
        print(f"Time step: {self.dt} s")
        print("=" * 60)
    
    def step(self, u_ref=0.0, r_ref=0.0):
        """
        Perform one step in the simulation using velocity commands.
        
        Args:
            u_ref (float): Desired surge velocity (m/s)
            r_ref (float): Desired yaw rate (rad/s)
            
        Returns:
            tuple: (scan, distance, cos, sin, collision, goal, action, reward)
                - scan: LIDAR scan data
                - distance: Distance to goal
                - cos, sin: Orientation relative to goal
                - collision: Collision flag
                - goal: Goal reached flag
                - action: Applied action [u_ref, r_ref]
                - reward: Computed reward
        """
        # Otter velocity controller
        u_control = self.otter.velocityControl(self.nu, u_ref, r_ref, self.dt)
        
        # Otter dynamics (6-DOF)
        [self.nu, self.u_actual] = self.otter.dynamics(
            self.eta, self.nu, self.u_actual, u_control, self.dt
        )
        
        # Update position using kinematic equations
        # For surface vessel: z=0, phi=0, theta=0, w=0, p=0, q=0
        self.eta[0] += self.dt * (self.nu[0] * np.cos(self.eta[5]) - self.nu[1] * np.sin(self.eta[5]))
        self.eta[1] += self.dt * (self.nu[0] * np.sin(self.eta[5]) + self.nu[1] * np.cos(self.eta[5]))
        self.eta[5] += self.dt * self.nu[5]
        
        # Wrap yaw angle to [-pi, pi]
        self.eta[5] = np.arctan2(np.sin(self.eta[5]), np.cos(self.eta[5]))
        
        # Update IR-SIM for visualization and collision detection
        irsim_state = np.array([[self.eta[0]], [self.eta[1]], [self.eta[5]]])
        self.env.robot.set_state(irsim_state)
        
        # Step IR-SIM environment
        self.env.step()
        self.env.render()
        
        # Get sensor data
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        
        # Get robot state from IR-SIM (for collision detection)
        robot_state = self.env.get_robot_state()
        
        # Calculate distance and angle to goal
        goal_vector = [
            self.robot_goal[0].item() - self.eta[0],
            self.robot_goal[1].item() - self.eta[1],
        ]
        distance = np.linalg.norm(goal_vector)
        
        # Check if goal is reached
        goal = self.env.robot.arrive
        
        # Calculate orientation relative to goal
        pose_vector = [np.cos(self.eta[5]), np.sin(self.eta[5])]
        cos, sin = self.cossin(pose_vector, goal_vector)
        
        # Check collision
        collision = self.env.robot.collision
        
        # Action for logging
        action = [u_ref, r_ref]
        
        # Compute reward
        reward = self.get_reward(goal, collision, action, latest_scan)
        
        return latest_scan, distance, cos, sin, collision, goal, action, reward
    
    def reset(
        self, 
        robot_state=None, 
        robot_goal=None, 
        random_obstacles=True,
        random_obstacle_ids=None
    ):
        """
        Reset the simulation environment.
        
        Args:
            robot_state (list or None): Initial state [x, y, theta]
            robot_goal (list or None): Goal state [x, y, theta]
            random_obstacles (bool): Whether to randomly reposition obstacles
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize
            
        Returns:
            tuple: Initial observation after reset
        """
        # Reset robot state
        if robot_state is None:
            robot_state = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]
        
        # Reset Otter state
        self.eta = np.zeros(6)
        self.eta[0] = robot_state[0][0]
        self.eta[1] = robot_state[1][0]
        self.eta[5] = robot_state[2][0]
        self.nu = np.zeros(6)
        self.u_actual = np.zeros(2)
        
        # Reset IR-SIM robot
        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )
        
        # Randomize obstacles
        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[100, 100, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )
        
        # Set goal
        if robot_goal is None:
            self.env.robot.set_random_goal(
                obstacle_list=self.env.obstacle_list,
                init=True,
                range_limits=[[10, 10, -3.141592653589793], [90, 90, 3.141592653589793]],
            )
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        # Initial step with zero action
        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            u_ref=action[0], r_ref=action[1]
        )
        
        return latest_scan, distance, cos, sin, False, False, action, reward
    
    @staticmethod
    def get_reward(goal, collision, action, laser_scan):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.

        Returns:
            (float): Computed reward for the current state.
        """
        if goal:
            return 300.0
        elif collision:
            return -300.0
        else:
            r3 = lambda x: 5.0 - x if x < 5.0 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2