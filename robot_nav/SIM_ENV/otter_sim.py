import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import irsim
import numpy as np
import random
from robot_nav.SIM_ENV.sim_env import SIM_ENV

class OtterSIM(SIM_ENV):
    """
    Otter USV simulation environment wrapper for DRL training.
    
    This class provides a simplified interface to the IR-SIM native Otter USV.
    The Otter dynamics are handled entirely by IR-SIM's otter_usv_kinematics,
    which integrates the full 6-DOF model from Python Vehicle Simulator.
    
    Attributes:
        env (object): IR-SIM environment instance with native Otter USV
        dt (float): Simulation time step
        robot_goal (np.ndarray): Goal position [x, y, psi]
    """
    
    def __init__(self, world_file="otter_world.yaml", disable_plotting=False, enable_phase1=True):
        """
        Initialize the Otter USV simulation environment.
        
        Args:
            world_file (str): Path to the world configuration YAML file
            disable_plotting (bool): If True, disables rendering and plotting
            enable_phase1 (bool): If True, enables Phase 1 action frequency control
        """
        # Initialize IR-SIM environment with native Otter USV
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        # Get simulation parameters
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.dt = self.env.step_time
        
        # Phase 1: Action Frequency Control
        self.enable_phase1 = enable_phase1
        if self.enable_phase1:
            # Physics simulation: 0.05s (accurate dynamics)
            self.physics_dt = self.dt
            # DRL action update: 1.0s (allow controller settling)
            self.action_dt = 1.0
            # Number of physics steps per action
            self.steps_per_action = int(self.action_dt / self.physics_dt)
            # Step counter for action frequency control
            self.step_counter = 0
            # Current action held for multiple steps
            self.current_action = np.array([[0.0], [0.0]])
            
            print("=" * 60)
            print("Otter USV Environment Initialized (Native IR-SIM)")
            print("⭐ PHASE 1 ENABLED: Action Frequency Control")
            print("=" * 60)
            print(f"Physics time step: {self.physics_dt:.3f} s")
            print(f"DRL action interval: {self.action_dt:.3f} s")
            print(f"Steps per action: {self.steps_per_action}")
            print(f"Controller settling time: ~1.0s (within action interval)")
        else:
            print("=" * 60)
            print("Otter USV Environment Initialized (Native IR-SIM)")
            print("=" * 60)
        
        robot_state = self.env.robot.state
        print(f"Robot position: [{robot_state[0,0]:.2f}, {robot_state[1,0]:.2f}, {robot_state[2,0]:.2f}]")
        print(f"Goal position: {self.robot_goal.T}")
        print(f"Time step: {self.dt} s")
        print(f"State dimension: {robot_state.shape[0]}")
        print("=" * 60)
    
    def step(self, u_ref=0.0, r_ref=0.0):
        """
        Perform one step in the simulation using velocity commands.
        
        Phase 1: Action frequency control
        - Physics updates every 0.05s (accurate)
        - DRL action updates every 0.5s (controller settling time)
        - Same action held for multiple physics steps
        
        IR-SIM's native otter_usv_kinematics handles:
        - Velocity controller
        - 6-DOF dynamics 
        - Control allocation
        - Propeller saturation
        
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
        if self.enable_phase1:
            # Phase 1: Action frequency control
            # Update action only at specified intervals (0.5s)
            if self.step_counter % self.steps_per_action == 0:
                self.current_action = np.array([[u_ref], [r_ref]])
            
            # Use the current (held) action for physics step
            action = self.current_action
            self.step_counter += 1
        else:
            # Normal mode: action updated every step
            action = np.array([[u_ref], [r_ref]])
        
        # Pass velocity commands to IR-SIM
        # IR-SIM will use otter_usv_kinematics to compute full dynamics
        self.env.step(action_id=0, action=action)
        self.env.render()
        
        # Get updated state from IR-SIM
        robot_state = self.env.robot.state
        
        # Extract position and orientation
        x = robot_state[0, 0]
        y = robot_state[1, 0]
        psi = robot_state[2, 0]
        
        # Get sensor data
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        
        # Calculate distance and angle to goal
        goal_vector = [
            self.robot_goal[0].item() - x,
            self.robot_goal[1].item() - y,
        ]
        distance = np.linalg.norm(goal_vector)
        
        # Check if goal is reached
        goal = self.env.robot.arrive
        
        # Calculate orientation relative to goal
        pose_vector = [np.cos(psi), np.sin(psi)]
        cos, sin = self.cossin(pose_vector, goal_vector)
        
        # Check collision
        collision = self.env.robot.collision
        
        # Action for logging
        action_list = [u_ref, r_ref]
        
        # Compute reward
        reward = self.get_reward(goal, collision, action_list, latest_scan)
        
        return latest_scan, distance, cos, sin, collision, goal, action_list, reward
    
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
        # Generate random initial state if not provided
        if robot_state is None:
            robot_state = [[random.uniform(10, 90)], [random.uniform(10, 90)], [0]]
        
        # Reset IR-SIM robot state
        # For RobotOtter with 8-DOF state, we only set position/orientation
        # Velocities and propeller states will be initialized to zero
        self.env.robot.state[0, 0] = robot_state[0][0]  # x
        self.env.robot.state[1, 0] = robot_state[1][0]  # y
        self.env.robot.state[2, 0] = robot_state[2][0]  # psi
        
        # Reset velocities to zero (if state_dim >= 6)
        if self.env.robot.state.shape[0] >= 6:
            self.env.robot.state[3, 0] = 0.0  # u
            self.env.robot.state[4, 0] = 0.0  # v
            self.env.robot.state[5, 0] = 0.0  # r
        
        # Reset propeller states to zero (if state_dim >= 8)
        if self.env.robot.state.shape[0] >= 8:
            self.env.robot.state[6, 0] = 0.0  # n1
            self.env.robot.state[7, 0] = 0.0  # n2
        
        # Update geometry
        self.env.robot._geometry = self.env.robot.gf.step(self.env.robot.state)
        
        # Update init_state for reset functionality
        self.env.robot._init_state = self.env.robot.state.copy()
        
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
        
        # Reset IR-SIM environment
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        # Phase 1: Reset step counter
        if self.enable_phase1:
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
        
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
            return 2000.0
        elif collision:
            return -2000.0
        else:
            r3 = lambda x: 5.0 - x if x < 5.0 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
