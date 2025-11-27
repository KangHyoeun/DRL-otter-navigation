import sys
sys.path.append('/home/hyo/PythonVehicleSimulator/src')

import irsim
import numpy as np
import random
import matplotlib.pyplot as plt
from robot_nav.SIM_ENV.sim_env import SIM_ENV
from irsim.util.util import WrapToPi
from colregs_core.utils import calculate_distance, calculate_cte, calculate_ref_path, calculate_desired_course_angle, WrapTo180
from colregs_core.geometry import heading_speed_to_velocity, math_to_ned_heading, math_to_maritime_position
from colregs_core.reward import JeonRewardCalculator
from colregs_core.risk import ShipDomainParams, JeonCollisionRisk, ChunCollisionRisk

class OtterSIM(SIM_ENV):
    """
    Otter USV simulation environment wrapper for DRL training.
    Integrates IR-SIM's native Otter USV with 6-DOF dynamics.
    """
    
    def __init__(self, world_file="robot_nav/worlds/imazu_scenario/imazu_case_01.yaml", 
                 disable_plotting=True, enable_phase1=True, max_steps=512,
                 cr_method='jeon', w_efficiency=1.0, w_safety=1.0,
                 os_speed_for_cr: float = 3.0, ts_speed_for_cr: float = 3.0,
                 r_ref_deadzone: float = 0.01,
                 grid_forward: int = 128, grid_lateral: int = 128,
                 obs_distance_forward: float = 200.0, obs_distance_lateral: float = 200.0,
                 chi_inf=0.5, k=1.0):
        """
        Initialize Otter USV simulation environment.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        self.current_world_file = world_file # Track current world file
        
        if len(self.env.robot_list) == 0:
            raise ValueError(
                f"No robots found! World file: {world_file}\n"
                f"Check YAML file contains 'robot' section."
            )
        
        self.robot_info = self.env.get_robot_info(0)
        self.dt = self.env.step_time

        self.prev_position = None
        self.prev_distance = None
        self.prev_heading = None
        
        robot_state = self.env.robot.state
        robot_goal = self.robot_info.goal
        start_pos_math = [robot_state[0, 0], robot_state[1, 0]]
        goal_pos_math = [robot_goal[0, 0], robot_goal[1, 0]]
        self.start_position = list(math_to_maritime_position(start_pos_math[0], start_pos_math[1]))
        self.goal_position = list(math_to_maritime_position(goal_pos_math[0], goal_pos_math[1]))
        self.max_steps = max_steps
        self.cr_method = cr_method.lower()
        self.w_efficiency = w_efficiency
        self.w_safety = w_safety
        self.os_speed_for_cr = os_speed_for_cr
        self.ts_speed_for_cr = ts_speed_for_cr
        self.r_ref_deadzone = r_ref_deadzone
        self.chi_inf = chi_inf
        self.k = k
        
        # Grid configuration
        self.grid_forward = 128
        self.grid_lateral = 128
        self.obs_distance_forward = 200.0
        self.obs_distance_lateral = 200.0
        self.cell_size_forward = self.obs_distance_forward / self.grid_forward
        self.cell_size_lateral = self.obs_distance_lateral / self.grid_lateral
        
        # Pre-compute grid indices for fast ellipse generation
        # Coordinate grid: Y is forward, X is lateral
        y_idx, x_idx = np.meshgrid(np.arange(self.grid_forward), np.arange(self.grid_lateral), indexing='ij')
        self.grid_y_idx = y_idx
        self.grid_x_idx = x_idx
        
        self.initial_robot_state_from_yaml = self.env.robot.state.copy()
        self.initial_goal_from_yaml = self.robot_info.goal.copy()
        
        self.ship_domain = ShipDomainParams(
            r_bow=10.0,
            r_stern=2.0,
            r_starboard=10.0,
            r_port=2.0
        )
        
        if self.cr_method == 'jeon':
            self.cr_calculator = JeonCollisionRisk(
                ship_domain=self.ship_domain,
                d_obs=200.0,
                cr_obs=0.3,
                os_speed=3.0,
                ts_speed=3.0
            )
        elif self.cr_method == 'chun':
            self.cr_calculator = ChunCollisionRisk(
                ship_domain=self.ship_domain
            )
        else:
            raise ValueError(f"Unknown CR method: {cr_method}. Use 'jeon' or 'chun'.")
        
        self.reward_calculator = JeonRewardCalculator(
            d_max=25.0,
            v_ref=2.9,
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
            self.action_dt = 1.0
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
        
        self.reward_log_counter = 0 # Initialize reward log counter
        self.episode_step_count = 0 # Initialize episode step counter
            
    def step(self, u_ref=3.0, r_ref=0.0):
        if abs(r_ref) < self.r_ref_deadzone:
            r_ref = 0.0
        
        if self.enable_phase1:
            # Frame Skip Implementation: Run physics loop multiple times for one action
            # This replaces the old "wait for counter" logic with a proper internal loop
            accumulated_reward = 0.0
            
            for _ in range(self.steps_per_action):
                self.env.step(action_id=0, action=np.array([[u_ref], [r_ref]]))
                self.episode_step_count += 1
                
                # Check for terminal conditions inside the loop
                if self.env.robot.arrive or self.env.robot.collision:
                    break
        else:
            # No frame skip (1:1 mapping)
            self.env.step(action_id=0, action=np.array([[u_ref], [r_ref]]))
            self.episode_step_count += 1
        
        robot_state = self.env.robot.state
        
        os_position_math = [robot_state[0, 0], robot_state[1, 0]]
        os_position = list(math_to_maritime_position(os_position_math[0], os_position_math[1]))
        os_heading_math = np.degrees(robot_state[2, 0])                    
        os_heading = math_to_ned_heading(os_heading_math)                  
        os_speed = np.linalg.norm([robot_state[3, 0], robot_state[4, 0]])  
        os_velocity = heading_speed_to_velocity(os_heading, os_speed)
        
        os_beta = np.degrees(np.arcsin(robot_state[4, 0] / (os_speed + 1e-8))) if os_speed > 1e-8 else 0.0
        os_course = WrapTo180(os_heading + os_beta)
        ref_path = calculate_ref_path(os_position, self.goal_position)
        desired_course_angle = calculate_desired_course_angle(os_position, self.start_position, self.goal_position, self.chi_inf, self.k)
        chi_e = WrapTo180(desired_course_angle - os_course)
        desired_heading_angle = WrapTo180(desired_course_angle - os_beta)
        psi_e = WrapTo180(desired_heading_angle - os_heading)
        phi_tilde = WrapTo180(ref_path - os_course)

        distance = calculate_distance(os_position, self.goal_position)
        y_e = calculate_cte(self.start_position, self.goal_position, os_position)
        
        CR_max = 0.0
        selected_ts_idx = None
        ts_position = [999.0, 999.0]
        ts_velocity = [0.0, 0.0]
        ts_speed = 0.0
        ts_heading = 0.0
        encounter_type = None
        is_static_obstacle = False 
        
        if len(self.env.obstacle_list) > 0:
            for idx, obstacle in enumerate(self.env.obstacle_list):
                is_static = hasattr(obstacle, 'static') and obstacle.static
                
                ts_state = obstacle.state
                if ts_state.shape[0] < 5: continue
                
                temp_ts_position_math = [ts_state[0, 0], ts_state[1, 0]]
                temp_ts_position = list(math_to_maritime_position(temp_ts_position_math[0], temp_ts_position_math[1]))
                temp_ts_heading_math = np.degrees(ts_state[2, 0])
                temp_ts_heading = math_to_ned_heading(temp_ts_heading_math)
                temp_ts_speed = np.linalg.norm([ts_state[3, 0], ts_state[4, 0]])
                temp_ts_velocity = heading_speed_to_velocity(temp_ts_heading, temp_ts_speed)
                
                if temp_ts_speed < 0.1: is_static = True
                
                cr_result = self.cr_calculator.calculate_collision_risk(
                    os_position=os_position, os_velocity=os_velocity, os_heading=os_heading, os_speed=os_speed,
                    ts_position=temp_ts_position, ts_velocity=temp_ts_velocity, ts_heading=temp_ts_heading, ts_speed=temp_ts_speed
                )
                
                if cr_result['cr'] > CR_max:
                    CR_max = cr_result['cr']
                    selected_ts_idx = idx
                    ts_position = temp_ts_position
                    ts_velocity = temp_ts_velocity
                    ts_speed = temp_ts_speed
                    ts_heading = temp_ts_heading
                    is_static_obstacle = is_static
        
        if selected_ts_idx is not None:
            situation = self.reward_calculator.encounter_classifier.classify(
                os_position, os_heading, os_speed,
                ts_position, ts_heading, ts_speed
            )
            encounter_type = situation.encounter_type

        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        action_return = [u_ref, r_ref]
        
        cr_grid = self._create_cr_grid(os_position, os_heading, os_speed)

        # Pass -phi_tilde as relative_course (positive for starboard turn relative to ref path)
        relative_course = -phi_tilde

        reward = self.get_reward(
            goal, collision, distance, y_e, os_speed, os_position, os_velocity, os_heading,
            ts_speed, ts_position, ts_velocity, ts_heading, CR_max, encounter_type, is_static_obstacle, cr_grid, relative_course
        )
        
        # Update previous state AFTER reward calculation
        self.prev_position = os_position
        self.prev_heading = os_heading
        self.prev_distance = distance
        
        # Render if enabled
        if not self.env.disable_all_plot:
            self.env.render(interval=0.001)

        return distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, action_return, reward, robot_state, CR_max, cr_grid
    
    def reset(self, robot_state=None, robot_goal=None, random_obstacles=False, random_obstacle_ids=None, world_file=None):
        # Reload environment if world_file is provided and different
        if world_file is not None and world_file != self.current_world_file:
            plt.close('all') # Close previous plots to prevent accumulation
            display = self.env.display
            disable_plotting = self.env.disable_all_plot
            self.env = irsim.make(world_file, disable_all_plot=disable_plotting, display=display)
            self.current_world_file = world_file
            self.robot_info = self.env.get_robot_info(0)
            # Update initial states from new yaml
            self.initial_robot_state_from_yaml = self.env.robot.state.copy()
            self.initial_goal_from_yaml = self.robot_info.goal.copy()
            print(f"🔄 Environment reloaded: {world_file}")

        # Reset environment first
        self.env.reset()

        # Force reset robot state to initial state
        if robot_state is None:
            # Debug: Check if initial state is corrupted
            self.env.robot.set_state(self.initial_robot_state_from_yaml, init=True)
        else:
            if isinstance(robot_state, list): robot_state = np.array(robot_state)
            self.env.robot.set_state(robot_state, init=True)
        
        current_state = self.env.robot.state
        start_pos_math = [current_state[0, 0], current_state[1, 0]]
        self.start_position = list(math_to_maritime_position(start_pos_math[0], start_pos_math[1]))

        if random_obstacles and len(self.env.obstacle_list) > 0:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(min(7, len(self.env.obstacle_list)))]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14], range_high=[100, 100, 3.14],
                ids=random_obstacle_ids, non_overlapping=True,
            )

        if robot_goal is None:
            self.env.robot.set_goal(self.initial_goal_from_yaml, init=True)
        else:
            if isinstance(robot_goal, list): robot_goal = np.array(robot_goal)
            self.env.robot.set_goal(robot_goal, init=True)

        # self.env.reset() # Moved to beginning
        self.robot_goal = self.env.robot.goal
        goal_pos_math = [self.robot_goal[0, 0], self.robot_goal[1, 0]]
        self.goal_position = list(math_to_maritime_position(goal_pos_math[0], goal_pos_math[1]))

        self.prev_position = None
        self.prev_distance = None
        self.prev_heading = None
        
        if self.enable_phase1:
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])

        self.episode_step_count = 0 # Reset episode step counter
        action = [3.0, 0.0]
        return self.step(u_ref=action[0], r_ref=action[1])

    def get_reward(self, goal, collision, distance, y_e, 
                   os_speed, os_position, os_velocity, os_heading, 
                   ts_speed, ts_position, ts_velocity, ts_heading,
                   CR_max, encounter_type, is_static_obstacle, cr_grid, relative_course):
        
        reward_dict = self.reward_calculator.calculate_total_reward(
            current_distance=distance, previous_distance=self.prev_distance,
            cross_track_error=y_e, os_speed=os_speed,
            os_position=os_position, os_velocity=os_velocity, os_heading=os_heading,
            previous_heading=self.prev_heading, ts_speed=ts_speed,
            ts_position=ts_position, ts_velocity=ts_velocity, ts_heading=ts_heading,
            CR_max=CR_max, encounter_type=encounter_type, relative_course=relative_course,
            is_static_obstacle=is_static_obstacle,
            w_efficiency=self.w_efficiency, w_safety=self.w_safety
        )

        
        # Terminal rewards
        if goal:
            # Goal 도달 보상: +20.0 (스케일링 후)
            # 빨리 도착할수록 Step 페널티가 적게 쌓이므로 자연스럽게 보상이 커짐
            total_reward = 2000.0 
            print(f"\n🏆 Reward Log (Global Step #{self.reward_log_counter}): GOAL! Steps: {self.episode_step_count}, Reward: {total_reward*0.01:.2f}")
            return total_reward * 0.01

        elif collision:
            # 충돌 페널티: -20.0 (스케일링 후)
            total_reward = -2500.0
            print(f"\n💥 Reward Log (Global Step #{self.reward_log_counter}): COLLISION! Reward: {total_reward*0.01:.2f}")
            return total_reward * 0.01
        
        # 일반 스텝 보상:
        # r_total 범위 -2 ~ +2 가정
        total_reward = reward_dict['r_total'] 
        total_reward -= 0.01  # Living penalty

        # if self.reward_log_counter % 100 == 0:
        #     print(f"\n💰 Reward Log (step #{self.reward_log_counter}):")
        #     print(f"   Total Reward: {total_reward*0.01:.4f}")
        #     print(f"   CR_max: {CR_max:.4f}, Distance: {distance:.2f}, CTE (y_e): {y_e:.2f}, OS_Speed: {os_speed:.2f}")
        #     for key, val in reward_dict.items():
        #         if key != 'r_total':
        #             print(f"   - {key}: {val:.4f}")
        self.reward_log_counter += 1
        
        # 보상 스케일링 적용

        self.latest_rewards = reward_dict
        return total_reward * 0.01
    
    def _create_cr_grid(self, os_position, os_heading, os_speed):
        """
        Optimized CR grid creation with enhanced obstacle representation.
        """
        grid = np.zeros((self.grid_forward, self.grid_lateral), dtype=np.float32)
        # 기본 배경값 설정 (약간의 noise로 CNN 학습 돕기)
        grid += 0.01  # 모든 셀에 미세한 배경값
        os_velocity = heading_speed_to_velocity(os_heading, os_speed)
        
        # Pre-calculate common values
        heading_rad = np.radians(os_heading)
        cos_head = np.cos(heading_rad)
        sin_head = np.sin(heading_rad)
        center_y = self.grid_forward / 2.0
        center_x = self.grid_lateral / 2.0
        
        # Domain semi-axes in grid cells
        cells_forward = (self.ship_domain.r_bow + self.ship_domain.r_stern) / 2.0 / self.cell_size_forward
        cells_lateral = (self.ship_domain.r_starboard + self.ship_domain.r_port) / 2.0 / self.cell_size_lateral
        
        # Ensure minimum 1 cell radius
        cells_forward = max(1.0, cells_forward)
        cells_lateral = max(1.0, cells_lateral)
        
        for obstacle in self.env.obstacle_list:
            ts_state = obstacle.state
            if ts_state.shape[0] < 5: continue
            
            # Extract TS state
            ts_pos_math_x, ts_pos_math_y = ts_state[0, 0], ts_state[1, 0]
            ts_pos_y, ts_pos_x = math_to_maritime_position(ts_pos_math_x, ts_pos_math_y) # (N, E)
            ts_position = [ts_pos_y, ts_pos_x]
            
            ts_heading_math = np.degrees(ts_state[2, 0])
            ts_heading = math_to_ned_heading(ts_heading_math)
            ts_speed = np.linalg.norm([ts_state[3, 0], ts_state[4, 0]])
            ts_velocity = heading_speed_to_velocity(ts_heading, ts_speed)
            
            # CR Calculation
            cr_result = self.cr_calculator.calculate_collision_risk(
                os_position=os_position, os_velocity=os_velocity, os_heading=os_heading, os_speed=os_speed,
                ts_position=ts_position, ts_velocity=ts_velocity, ts_heading=ts_heading, ts_speed=ts_speed
            )
            cr_value = cr_result['cr']
            
            # Coordinate Transformation (World to Grid)
            rel_n = ts_position[0] - os_position[0]
            rel_e = ts_position[1] - os_position[1]
            
            body_forward =  rel_n * cos_head + rel_e * sin_head
            body_right   = -rel_n * sin_head + rel_e * cos_head
            
            grid_y_center = center_y + (body_forward / self.cell_size_forward)
            grid_x_center = center_x + (body_right / self.cell_size_lateral)
            
            # Fast Vectorized Assignment
            enhanced_cr = max(cr_value, 0.1)  # 최소 0.1 보장
            self._assign_cr_to_grid_vectorized(grid, grid_x_center, grid_y_center, enhanced_cr, cells_lateral, cells_forward)  
        
        return grid
    
    def _assign_cr_to_grid_vectorized(self, grid, center_x, center_y, cr_value, semi_axis_x, semi_axis_y):
        """
        Optimized vectorized assignment of CR values to grid using elliptical masking.
        """
        # Define bounding box indices (clipped to grid limits)
        # Adding margin to ensure coverage
        margin_x = int(np.ceil(semi_axis_x))
        margin_y = int(np.ceil(semi_axis_y))
        
        y_min = max(0, int(center_y - margin_y))
        y_max = min(self.grid_forward, int(center_y + margin_y + 1))
        x_min = max(0, int(center_x - margin_x))
        x_max = min(self.grid_lateral, int(center_x + margin_x + 1))
        
        if x_min >= x_max or y_min >= y_max:
            return

        # Extract grid slice coordinates
        # Use pre-computed grid indices but slice them locally
        # Local grid indices
        Y, X = np.ogrid[y_min:y_max, x_min:x_max]
        
        # Ellipse Equation: ((x - cx) / a)^2 + ((y - cy) / b)^2 <= 1
        dist_sq = ((X - center_x) / semi_axis_x)**2 + ((Y - center_y) / semi_axis_y)**2
        
        # Boolean mask for ellipse
        mask = dist_sq <= 1.0
        
        # Apply CR value (max pooling)
        # We work on the slice of the grid
        grid_slice = grid[y_min:y_max, x_min:x_max]
        
        # Update using maximum: grid[mask] = max(grid[mask], cr_value)
        # Numpy's maximum function is efficient
        np.maximum(grid_slice, cr_value, out=grid_slice, where=mask)

    def render(self, mode='human', **kwargs):
        """
        Render the simulation with enhanced visualization.
        """
        if not hasattr(self.env, 'plot'):
            return

        # 1. Basic Render
        # Pass show_vectors and show_domain to object plot
        self.env.render(show_trajectory=True, show_vectors=True, show_domain=True, **kwargs)
        
        # 2. Update Info Box
        u_ref = kwargs.get('u_ref', 0.0)
        r_ref = kwargs.get('r_ref', 0.0)
        reward = kwargs.get('reward', 0.0)
        
        robot_state = self.env.robot.state
        u_actual = robot_state[3, 0]
        r_actual = robot_state[5, 0]
        
        info_str = (f"u_cmd: {u_ref:.2f} m/s\n"
                    f"u_act: {u_actual:.2f} m/s\n"
                    f"r_cmd: {r_ref:.3f} rad/s\n"
                    f"r_act: {r_actual:.3f} rad/s\n"
                    f"Rew: {reward:.2f}")
        
        if hasattr(self.env.plot, 'update_info_box'):
            self.env.plot.update_info_box(info_str)
        
        # 3. Update Action Bar
        if hasattr(self.env.plot, 'update_action_bar'):
            self.env.plot.update_action_bar(u_ref, r_ref)
        
        # 4. Overlay Risk Grid Map (if provided)
        grid_map = kwargs.get('grid_map', None)
        if grid_map is not None:
            if not hasattr(self, 'grid_ax'):
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                # Create inset axes at top right
                self.grid_ax = inset_axes(self.env.plot.ax, width="30%", height="30%", loc='upper right', borderpad=1)
                self.grid_ax.set_title("Risk Grid", fontsize=8, color='white')
                self.grid_ax.axis('off')
                self.grid_img = self.grid_ax.imshow(grid_map, cmap='jet', vmin=0, vmax=1, origin='lower')
            else:
                self.grid_img.set_data(grid_map)
```