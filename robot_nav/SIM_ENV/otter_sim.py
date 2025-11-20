import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import irsim
import numpy as np
import random
from robot_nav.SIM_ENV.sim_env import SIM_ENV
from irsim.util.util import WrapToPi
from colregs_core.utils import distance, cross_track_error
from colregs_core.geometry import heading_speed_to_velocity, math_to_ned_heading, math_to_maritime_position
from colregs_core.reward import JeonRewardCalculator
from colregs_core.risk import ShipDomainParams, JeonCollisionRisk, ChunCollisionRisk

class OtterSIM(SIM_ENV):
    """
    Otter USV simulation environment wrapper for DRL training.
    Integrates IR-SIM's native Otter USV with 6-DOF dynamics.
    """
    
    def __init__(self, world_file="robot_nav/worlds/imazu_scenario/imazu_case_01.yaml", 
                 disable_plotting=True, enable_phase1=True, max_steps=1000,
                 cr_method='jeon', w_efficiency=1.0, w_safety=1.0,
                 os_speed_for_cr: float = 3.0, ts_speed_for_cr: float = 3.0):
        """
        Initialize Otter USV simulation environment.
        
        Args:
            world_file: Path to world configuration YAML
            disable_plotting: Disable rendering if True
            enable_phase1: Enable action frequency control if True
            max_steps: Maximum steps per episode for terminal reward calculation
            cr_method: Collision risk calculation method ('jeon' or 'chun')
            w_efficiency: Weight for efficiency rewards
            w_safety: Weight for safety rewards
            os_speed_for_cr: OS speed for CR calculation (Jeon)
            ts_speed_for_cr: TS speed for CR calculation (Jeon)
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        if len(self.env.robot_list) == 0:
            raise ValueError(
                f"No robots found! World file: {world_file}\n"
                f"Check YAML file contains 'robot' section."
            )
        
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.dt = self.env.step_time

        self.prev_distance = None
        self.prev_heading = None
        # Initialize start_position from robot's initial state
        robot_state = self.env.robot.state
        start_pos_math = [robot_state[0, 0], robot_state[1, 0]]
        self.start_position = list(math_to_maritime_position(start_pos_math[0], start_pos_math[1]))
        self.goal_position = list(math_to_maritime_position(self.robot_goal[0, 0], self.robot_goal[1, 0]))
        self.max_steps = max_steps
        self.cr_method = cr_method.lower()
        self.w_efficiency = w_efficiency
        self.w_safety = w_safety
        self.os_speed_for_cr = os_speed_for_cr
        self.ts_speed_for_cr = ts_speed_for_cr
        
        # Initialize Ship Domain for CR calculation
        self.ship_domain = ShipDomainParams(
            r_bow=6.0,
            r_stern=2.0,
            r_starboard=6.0,
            r_port=2.0
        )
        
        # Initialize CR calculator
        if self.cr_method == 'jeon':
            self.cr_calculator = JeonCollisionRisk(
                ship_domain=self.ship_domain,
                d_obs=200.0,
                cr_obs=0.3,
                os_speed=2.0,
                ts_speed=2.0
            )
        elif self.cr_method == 'chun':
            self.cr_calculator = ChunCollisionRisk(
                ship_domain=self.ship_domain
            )
        else:
            raise ValueError(f"Unknown CR method: {cr_method}. Use 'jeon' or 'chun'.")
        
        # Initialize Reward Calculator
        self.reward_calculator = JeonRewardCalculator(
            d_max=10.0,
            v_ref=2.0,
            cr_allowable=0.3,
            dt=self.dt,
            ship_domain=self.ship_domain,
            d_obs=200.0,
            phi_max=4.0,
            cr_method=self.cr_method,
            os_speed_for_cr=self.os_speed_for_cr,
            ts_speed_for_cr=self.ts_speed_for_cr
        )

        self.enable_phase1 = enable_phase1
        if self.enable_phase1:
            self.physics_dt = self.dt
            self.action_dt = 0.5
            self.steps_per_action = int(self.action_dt / self.physics_dt)
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
            
            print("=" * 60)
            print("Otter USV Environment - PHASE 1 ENABLED")
            print("=" * 60)
            print(f"Physics time step: {self.physics_dt:.3f} s")
            print(f"DRL action interval: {self.action_dt:.3f} s")
            print(f"Steps per action: {self.steps_per_action}")
        else:
            print("=" * 60)
            print("Otter USV Environment Initialized")
            print("=" * 60)
        
        robot_state = self.env.robot.state
        print(f"Robot position: [{robot_state[0,0]:.2f}, {robot_state[1,0]:.2f}, {robot_state[2,0]:.2f}]")
        print(f"Goal position: {self.robot_goal.T}")
        print(f"Time step: {self.dt} s")
        print(f"State dimension: {robot_state.shape[0]}")
        print("=" * 60)
    
    def step(self, u_ref=0.0, r_ref=0.087):
        """
        Execute one simulation step with velocity commands.
        
        Args:
            u_ref: Desired surge velocity (m/s)
            r_ref: Desired yaw rate (rad/s)
            
        Returns:
            tuple: (scan, distance_to_goal, y_e, collision, goal_reached, action, reward, robot_state, CR_max)
        """
        if self.enable_phase1:
            if self.step_counter % self.steps_per_action == 0:
                self.current_action = np.array([[u_ref], [r_ref]])
            action = self.current_action
            self.step_counter += 1
        else:
            action = np.array([[u_ref], [r_ref]])
        
        self.env.step(action_id=0, action=action)
        self.env.render()
        
        robot_state = self.env.robot.state
        
        # Extract OS (Own Ship) information
        os_position_math = [robot_state[0, 0], robot_state[1, 0]]
        os_position = list(math_to_maritime_position(os_position_math[0], os_position_math[1]))
        os_heading_math = np.degrees(robot_state[2, 0])
        os_heading = math_to_ned_heading(os_heading_math)
        os_speed = np.linalg.norm([robot_state[3, 0], robot_state[4, 0]])
        os_velocity = heading_speed_to_velocity(os_heading, os_speed)

        # Calculate navigation metrics
        y_e = cross_track_error(self.start_position, self.goal_position, os_position)
        dist_to_goal = distance(os_position, self.goal_position)
        
        # Extract TS (Target Ship) information - select TS with highest CR
        CR_max = 0.0
        selected_ts_idx = None
        ts_position = [999.0, 999.0]
        ts_velocity = [0.0, 0.0]
        ts_speed = 0.0
        ts_heading = 0.0
        encounter_type = None
        
        if len(self.env.obstacle_list) > 0:
            # Calculate CR for all obstacles and find the most dangerous one
            for idx, obstacle in enumerate(self.env.obstacle_list):
                if hasattr(obstacle, 'static') and obstacle.static:
                    continue
                
                ts_state = obstacle.state
                if ts_state.shape[0] < 5:
                    continue
                
                temp_ts_position_math = [ts_state[0, 0], ts_state[1, 0]]
                temp_ts_position = list(math_to_maritime_position(temp_ts_position_math[0], temp_ts_position_math[1]))
                temp_ts_heading_math = np.degrees(ts_state[2, 0])
                temp_ts_heading = math_to_ned_heading(temp_ts_heading_math)
                temp_ts_speed = np.linalg.norm([ts_state[3, 0], ts_state[4, 0]])
                temp_ts_velocity = heading_speed_to_velocity(temp_ts_heading, temp_ts_speed)
                
                # Calculate collision risk for this obstacle
                cr_result = self.cr_calculator.calculate_collision_risk(
                    os_position=os_position,
                    os_velocity=os_velocity,
                    os_heading=os_heading,
                    ts_position=temp_ts_position,
                    ts_velocity=temp_ts_velocity,
                    ts_heading=temp_ts_heading
                )
                
                if cr_result['cr'] > CR_max:
                    CR_max = cr_result['cr']
                    selected_ts_idx = idx
                    ts_position = temp_ts_position
                    ts_velocity = temp_ts_velocity
                    ts_speed = temp_ts_speed
                    ts_heading = temp_ts_heading
        
        # Determine encounter type for the most dangerous ship
        if selected_ts_idx is not None:
            situation = self.reward_calculator.encounter_classifier.classify(
                os_position, os_heading, os_speed,
                ts_position, ts_heading, ts_speed
            )
            encounter_type = situation.encounter_type

        # Get sensor data
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        
        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        action_return = [u_ref, r_ref]
        
        # The selected TS is dynamic by definition of the loop above
        is_static_obstacle = False

        reward = self.get_reward(
            goal=goal, 
            collision=collision, 
            dist_to_goal=dist_to_goal, 
            y_e=y_e, 
            os_speed=os_speed, 
            os_position=os_position, 
            os_velocity=os_velocity, 
            os_heading=os_heading, 
            ts_speed=ts_speed, 
            ts_position=ts_position, 
            ts_velocity=ts_velocity,
            ts_heading=ts_heading,
            CR_max=CR_max, 
            encounter_type=encounter_type, 
            is_static_obstacle=is_static_obstacle
        )
        
        self.prev_heading = os_heading
        self.prev_distance = dist_to_goal
        
        return latest_scan, dist_to_goal, y_e, collision, goal, action_return, reward, robot_state, CR_max
    
    def reset(self, robot_state=None, robot_goal=None, random_obstacles=False, random_obstacle_ids=None):
        """
        Reset simulation environment.
        
        Args:
            robot_state: Initial robot state [x, y, theta, ...]
            robot_goal: Goal position [x, y, theta]
            random_obstacles: Randomize obstacle positions if True
            random_obstacle_ids: Specific obstacle IDs to randomize
            
        Returns:
            tuple: Initial observation (scan, distance, collision, goal, action, reward, robot_state)
        """
        if robot_state is not None:
            if isinstance(robot_state, list):
                robot_state = np.array(robot_state)
            self.env.robot.set_state(robot_state, init=True)
        
        # Store start position after state is set
        current_state = self.env.robot.state
        start_pos_math = [current_state[0, 0], current_state[1, 0]]
        self.start_position = list(math_to_maritime_position(start_pos_math[0], start_pos_math[1]))

        if random_obstacles and len(self.env.obstacle_list) > 0:
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
            robot_goal = [90, 0, np.pi / 2]
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        goal_pos_math = [self.robot_goal[0, 0], self.robot_goal[1, 0]]
        self.goal_position = list(math_to_maritime_position(goal_pos_math[0], goal_pos_math[1]))

        self.prev_distance = None
        self.prev_heading = None
        
        if self.enable_phase1:
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
        
        action = [0.0, 0.0]
        latest_scan, dist_to_goal, y_e, _, _, action, reward, robot_state, CR_max = self.step(
            u_ref=action[0], r_ref=action[1]
        )
        
        return latest_scan, dist_to_goal, y_e, False, False, action, reward, robot_state, CR_max

    def get_reward(self, goal, collision, dist_to_goal, y_e, 
                   os_speed, os_position, os_velocity, os_heading, 
                   ts_speed, ts_position, ts_velocity, ts_heading,
                   CR_max, encounter_type, is_static_obstacle):
        """
        Calculate reward using Jeon's reward function.
        
        Terminal rewards:
        - Goal: max_steps
        - Collision: max_steps 
        
        Step rewards: Jeon's efficiency + safety components
        """
        if goal:
            return self.max_steps/2
        elif collision:
            return -self.max_steps/2 
        
        reward_dict = self.reward_calculator.calculate_total_reward(
            current_distance=dist_to_goal,
            previous_distance=self.prev_distance,
            cross_track_error=y_e,
            os_speed=os_speed,
            os_position=os_position,
            os_velocity=os_velocity,
            os_heading=os_heading,
            previous_heading=self.prev_heading,
            ts_speed=ts_speed,
            ts_position=ts_position,
            ts_velocity=ts_velocity,
            ts_heading=ts_heading,
            start_position=self.start_position,
            CR_max=CR_max,
            encounter_type=encounter_type,
            goal_position=self.goal_position,
            is_static_obstacle=is_static_obstacle,
            w_efficiency=self.w_efficiency,
            w_safety=self.w_safety
        )
        
        # Extract total reward from dictionary
        return reward_dict['r_total']
