import rvo2
import numpy as np
from colregs_core.utils import WrapToPi

class ORCAAgent:
    """
    ORCA (Optimal Reciprocal Collision Avoidance) Agent Wrapper using Python-RVO2.
    
    Since ORCA assumes holonomic kinematics (can move in any direction),
    this agent includes a low-level controller to convert the optimal velocity vector
    into non-holonomic control inputs (Surge, Yaw Rate) for the Otter USV.
    """
    def __init__(self, time_step=0.1, neighbor_dist=200.0, max_neighbors=10, 
                 time_horizon=10.0, time_horizon_obst=5.0, radius=10.0, max_speed=3.0):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        
        # PID Gains for Heading Control
        self.kp_heading = 1.0
        self.kd_heading = 0.1
        self.prev_heading_error = 0.0

    def get_action(self, obs_dict, add_noise=False, update_rms=False):
        """
        Calculate control action [surge, yaw_rate] using ORCA.
        
        Args:
            obs_dict: Dictionary containing raw observation data from OtterSIM.
                      Must provide 'os_pos', 'os_vel', 'os_heading', 'goal_pos', 'obstacles'
        """
        # 1. Initialize RVO2 Simulator for this step
        # We recreate it every step to ensure perfect sync with the environment's dynamic state
        sim = rvo2.PyRVOSimulator(
            self.time_step, 
            self.neighbor_dist, 
            self.max_neighbors, 
            self.time_horizon, 
            self.time_horizon_obst, 
            self.radius, 
            self.max_speed
        )

        # Extract State
        os_pos = obs_dict['os_pos']       # [N, E]
        os_vel = obs_dict['os_vel']       # [vn, ve]
        os_heading = obs_dict['os_heading'] # rad
        goal_pos = obs_dict['goal_pos']   # [N, E]
        obstacles = obs_dict['obstacles'] # List of dicts {'pos': [N, E], 'vel': [vn, ve], 'radius': r}

        # 2. Add Own Ship (Agent 0)
        # RVO2 uses (x, y). We map (E, N) -> (x, y) or (N, E) -> (y, x). 
        # Let's stick to: x=East, y=North for RVO2 to match math conventions
        # os_pos is usually [N, E] in maritime code.
        
        # Mapping: x = East, y = North
        os_rvo_pos = (os_pos[1], os_pos[0]) 
        os_rvo_vel = (os_vel[1], os_vel[0])
        
        agent_id = sim.addAgent(os_rvo_pos, 
                                self.neighbor_dist, self.max_neighbors, 
                                self.time_horizon, self.time_horizon_obst, 
                                self.radius, self.max_speed, os_rvo_vel)

        # 3. Add Obstacles (Target Ships) as Agents
        # We treat dynamic obstacles as agents with a preferred velocity equal to their current velocity.
        # This tells ORCA "There is an agent here moving like this, avoid it assuming it keeps moving".
        for obs in obstacles:
            obs_rvo_pos = (obs['pos'][1], obs['pos'][0])
            obs_rvo_vel = (obs['vel'][1], obs['vel'][0])
            
            obs_id = sim.addAgent(obs_rvo_pos, 
                                  self.neighbor_dist, self.max_neighbors, 
                                  self.time_horizon, self.time_horizon_obst, 
                                  self.radius, self.max_speed, obs_rvo_vel)
            
            # Set TS preferred velocity to its current velocity (Constant Velocity Model)
            sim.setAgentPrefVelocity(obs_id, obs_rvo_vel)

        # 4. Set OS Preferred Velocity (Towards Goal)
        # Vector to Goal
        goal_vec = np.array([goal_pos[1] - os_pos[1], goal_pos[0] - os_pos[0]]) # [dE, dN]
        dist_to_goal = np.linalg.norm(goal_vec)
        
        if dist_to_goal > 1.0:
            pref_vel = (goal_vec / dist_to_goal) * self.max_speed
        else:
            pref_vel = (0.0, 0.0)
            
        sim.setAgentPrefVelocity(agent_id, tuple(pref_vel))

        # 5. Run ORCA Step
        sim.doStep()

        # 6. Get New Optimal Velocity
        new_vel = sim.getAgentVelocity(agent_id) # (vx, vy) -> (ve, vn)
        
        # 7. Low-Level Control: Convert (ve, vn) to (surge, yaw_rate)
        desired_speed = np.linalg.norm(new_vel)
        desired_heading = np.arctan2(new_vel[0], new_vel[1]) # atan2(E, N) -> NED Heading (0 is North)
        
        # Heading Error
        heading_error = WrapToPi(desired_heading - os_heading)
        
        # Surge Control (Simple Proportional, bounded)
        # If heading error is large, slow down to turn
        surge_cmd = desired_speed * np.cos(heading_error)
        surge_cmd = np.clip(surge_cmd, 0.0, self.max_speed)
        
        # Yaw Rate Control (PD)
        yaw_rate_cmd = self.kp_heading * heading_error + self.kd_heading * (heading_error - self.prev_heading_error)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0) # Limit turn rate
        
        self.prev_heading_error = heading_error
        
        # Normalize to [-1, 1] for compatibility with environment interface if needed,
        # BUT OtterSIM step() takes physical values. 
        # DRL models output [-1, 1] and then are scaled. 
        # Here we return physical values directly? 
        # Wait, OtterSIM.step expects scaled values if we follow the DRL pattern, 
        # but the `test_manager.py` scales the DRL output: `a_in = [(a[0]+1)*1.5, a[1]*0.1745]`.
        # So this class should probably return **normalized** values to be swappable,
        # OR we modify the test loop to handle 'raw' actions.
        
        # Let's return NORMALIZED actions to match DRL agent interface [-1, 1]
        # Surge: [0, 3] -> [-1, 1].  norm = (phy / 1.5) - 1
        norm_surge = (surge_cmd / 1.5) - 1.0
        
        # Yaw: [-0.1745, 0.1745] -> [-1, 1]. norm = phy / 0.1745
        norm_yaw = yaw_rate_cmd / 0.1745
        
        return np.array([norm_surge, norm_yaw]), None, None

    def prepare_state(self, distance, y_e, psi_e, chi_e, phi_tilde, collision, goal, a, robot_state, CR_max, grid_map=None):
        """
        Dummy method to match DRL agent interface. 
        We don't need neural network state preparation, but we need to construct the `obs_dict` 
        from the environment's raw data available in `test_manager`.
        
        However, `test_manager` passes individual variables. We need to access the `sim` object 
        to get full obstacle list.
        
        Wait, standard `prepare_state` returns a tensor. 
        Here we will return a dictionary that `get_action` expects.
        """
        # This method in DRL agent transforms raw inputs to Neural Net inputs.
        # For ORCA, we need the Raw Environment Object to extract obstacle positions.
        # Since this method signature doesn't include `sim` or `obstacle_list`, 
        # we have to rely on `test_manager` extracting it.
        
        # Strategy: We will modify `compare_manager` to pass the necessary info.
        # For now, return None.
        return None, False
