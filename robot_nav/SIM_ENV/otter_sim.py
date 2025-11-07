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
    
    def __init__(self, world_file="robot_nav/worlds/imazu_scenario/s1.yaml", disable_plotting=False, enable_phase1=True, max_steps=300):
        """
        Initialize the Otter USV simulation environment.
        
        Args:
            world_file (str): Path to the world configuration YAML file
            disable_plotting (bool): If True, disables rendering and plotting
            enable_phase1 (bool): If True, enables Phase 1 action frequency control
            max_steps (int): Maximum steps per episode (used to compute terminal rewards)
        """
        # Initialize IR-SIM environment with native Otter USV
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        # Check if robots were loaded
        if len(self.env.robot_list) == 0:
            raise ValueError(
                f"No robots found in the environment! "
                f"World file: {world_file}\n"
                f"Please check that the YAML file contains a 'robot' section."
            )
        
        # Get simulation parameters
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.dt = self.env.step_time

        # Previous distance for progress reward
        self.prev_distance = None
        
        # Store max_steps for reward calculation
        self.max_steps = max_steps
        
        # Phase 1: Action Frequency Control
        self.enable_phase1 = enable_phase1
        if self.enable_phase1:
            # Physics simulation: 0.1s (accurate dynamics)
            self.physics_dt = self.dt
            # DRL action update: 1.0s (allow controller settling)
            self.action_dt = 0.5 
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
            print(f"Controller settling time: ~9.0s (within action interval)")
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
    
    def step(self, u_ref=0.0, r_ref=0.087):
        """
        Perform one step in the simulation using velocity commands.
        
        Phase 1: Action frequency control
        - Physics updates every 0.1s (accurate)
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
            # Update action only at specified intervals (1.0s)
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
        u = robot_state[3, 0]
        r = robot_state[5, 0]
        n1 = robot_state[6, 0]
        n2 = robot_state[7, 0]

        # Get sensor data
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        
        # Calculate distance and angle to goal
        goal_vector = [
            self.robot_goal[0].item() - x,
            self.robot_goal[1].item() - y,
        ]
        distance = np.linalg.norm(goal_vector)

        # ⭐ Progress reward 계산
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
        else:
            progress = 0.0

        goal = self.env.robot.arrive
        pose_vector = [np.cos(psi), np.sin(psi)]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [u_ref, r_ref]
        reward = self.get_reward(goal, collision, action, latest_scan, progress)
        self.prev_distance = distance

        return latest_scan, distance, cos, sin, collision, goal, action, reward, robot_state
    
    def reset(
        self, 
        robot_state=None, 
        robot_goal=None, 
        random_obstacles=False,
        random_obstacle_ids=None
    ):
        """
        Reset the simulation environment.
        
        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles
            random_obstacle_ids (list or None): Specific obstacle IDs to randomi     episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)
ze
            
        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        # If robot_state is provided, set it explicitly
        if robot_state is not None:
            # Convert to numpy array if needed
            if isinstance(robot_state, list):
                robot_state = np.array(robot_state)
            # Set state manually (overrides YAML)
            self.env.robot.set_state(robot_state, init=True)
        
        # if robot_state is None:
        #     self.env.robot.state[0, 0] = 0.0
        #     self.env.robot.state[1, 0] = -90 # initial position = (0, -40)m
        #     self.env.robot.state[2, 0] = np.pi / 2 # heading angle = 90 degrees
        #     self.env.robot.state[3, 0] = 0.0  # u
        #     self.env.robot.state[4, 0] = 0.0  # v
        #     self.env.robot.state[5, 0] = 0.0  # r
        #     self.env.robot.state[6, 0] = 0.0  # n1
        #     self.env.robot.state[7, 0] = 0.0  # n2
        
        # Randomize obstacles (only if Otter USV obstacles exist)
        if random_obstacles and len(self.env.obstacle_list) > 0:
            # Check if first obstacle is an Otter USV (8-DOF state)
            first_obs = self.env.obstacle_list[0]
            if hasattr(first_obs, 'state') and first_obs.state.shape[0] >= 8:
                if random_obstacle_ids is None:
                    random_obstacle_ids = [i + 1 for i in range(min(7, len(self.env.obstacle_list)))]
                self.env.random_obstacle_position(
                    range_low=[0, 0, -3.14],
                    range_high=[100, 100, 3.14],
                    ids=random_obstacle_ids,
                    non_overlapping=True,
                )

        if robot_goal is None:
            robot_goal = [0, 0, np.pi / 2]
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        self.prev_distance = None
        
        # Phase 1: Reset step counter
        if self.enable_phase1:
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
        
        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward, robot_state = self.step(
            u_ref=action[0], r_ref=action[1]
        )
        
        return latest_scan, distance, cos, sin, False, False, action, reward, robot_state

    def get_reward(self, goal, collision, action, latest_scan, progress):
        """
        Calculate reward based on goal, collision, progress, and obstacles.
        
        Terminal rewards are computed as:
        - Goal reward: max_steps * 10
        - Collision penalty: max_steps * -10
        
        Args:
            goal (bool): Whether goal was reached
            collision (bool): Whether collision occurred
            action (list): Action taken [u_ref, r_ref]
            latest_scan (list): LIDAR scan data
            progress (float): Distance progress since last step
            
        Returns:
            float: Reward value
        """
        # Terminal rewards (computed from max_steps)
        goal_reward = self.max_steps * 10.0
        collision_penalty = self.max_steps * -10.0
        
        if goal:
            return goal_reward
        elif collision:
            return collision_penalty
        
        if self.max_steps == 300:
            progress_reward = progress * 100.0  
        elif self.max_steps == 1000:
            progress_reward = progress * 300.0  # Was 100.0!
        else:
            progress_reward = progress * 50.0
        
        # 3. OBSTACLE PENALTY
        min_distance = min(latest_scan)
        if min_distance < 5.0:
            if self.max_steps == 300:
                obstacle_penalty = -(5.0 - min_distance) * 10.0
            elif self.max_steps == 1000:
                obstacle_penalty = -(5.0 - min_distance) * 30.0
            else:
                obstacle_penalty = -(5.0 - min_distance) * 5.0
        else:
            obstacle_penalty = 0.0
        
        # 5. STEP PENALTY 
        if self.max_steps == 300:
            step_penalty = -1.0  # Encourage faster completion
        elif self.max_steps == 1000:
            step_penalty = -3.0  # Encourage faster completion
        else:
            step_penalty = -1.0  # Encourage faster completion
                
        total_reward = (
            progress_reward +       # +0.5~1.0 bonus
            obstacle_penalty +      # -15 near obstacles
            step_penalty            # -2 per step
        )
        
        return total_reward