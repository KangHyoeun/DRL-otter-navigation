"""
COLREGs-compliant Reward Function for Otter USV Navigation
============================================================

이 reward function은 다음을 목표로 합니다:
1. COLREGs 규칙 준수 (Rule 13, 14, 15, 17)
2. 효율적인 충돌 회피 (DCPA/TCPA 기반)
3. 연료 효율성 (불필요한 선회 최소화)
4. ETA 준수 (제시간 도착)
"""

import numpy as np
from typing import Tuple, List, Optional
import irsim
import random

from robot_nav.SIM_ENV.sim_env import SIM_ENV
from colregs_core.encounter.classifier import EncounterClassifier
from colregs_core.encounter.types import EncounterType, RiskLevel
from colregs_core.risk.risk_matrix import RiskAssessment
from colregs_core.risk.cpa_tcpa import calculate_cpa_tcpa
from colregs_core.geometry.bearings import calculate_relative_bearing, normalize_angle


class COLREGsRewardCalculator:
    """
    COLREGs 준수 reward 계산기
    
    Reward Components:
    1. Terminal Rewards (Goal/Collision)
    2. COLREGs Compliance Rewards
    3. Risk-based Rewards (DCPA/TCPA)
    4. Progress Rewards
    5. Efficiency Penalties
    6. ETA Compliance
    """
    
    def __init__(
        self,
        goal_distance: float = 180.0,  # meters
        expected_speed: float = 3.0,   # m/s
        safe_distance: float = 100.0,  # meters (COLREGs safe distance)
        dcpa_threshold: float = 30.0,  # meters (minimum safe DCPA)
        tcpa_threshold: float = 60.0,  # seconds (minimum safe TCPA)
    ):
        """
        Args:
            goal_distance: 목표까지의 거리 (ETA 계산용)
            expected_speed: 예상 항해 속도 (ETA 계산용)
            safe_distance: 안전 거리 (조우 상황 판단 임계값)
            dcpa_threshold: 최소 안전 DCPA
            tcpa_threshold: 최소 안전 TCPA
        """
        self.goal_distance = goal_distance
        self.expected_speed = expected_speed
        self.expected_eta = goal_distance / expected_speed  # seconds
        
        # COLREGs 분류기
        self.encounter_classifier = EncounterClassifier(
            safe_distance=safe_distance
        )
        
        # 위험도 평가기
        self.risk_assessor = RiskAssessment(
            dcpa_critical=dcpa_threshold,
            dcpa_high=dcpa_threshold * 2,
            dcpa_medium=dcpa_threshold * 4,
            dcpa_low=dcpa_threshold * 8
        )
        
        # Tracking variables
        self.prev_distance_to_goal = None
        self.prev_dcpa = {}  # {ts_id: dcpa}
        self.cumulative_course_change = 0.0  # Total course change for fuel efficiency
        self.prev_heading = None
        self.episode_start_time = 0.0
        self.current_step = 0
        
    def reset(self):
        """에피소드 시작 시 호출"""
        self.prev_distance_to_goal = None
        self.prev_dcpa = {}
        self.cumulative_course_change = 0.0
        self.prev_heading = None
        self.episode_start_time = 0.0
        self.current_step = 0
    
    def calculate_reward(
        self,
        # 상태 정보
        os_position: Tuple[float, float],
        os_heading: float,  # radians
        os_velocity: Tuple[float, float],
        goal_position: Tuple[float, float],
        goal_reached: bool,
        collision: bool,
        # 타겟 선박 정보 (리스트)
        target_ships: List[dict],  # [{id, position, heading, velocity}, ...]
        # 액션 정보
        action: List[float],  # [u_ref, r_ref]
        # 센서 정보
        laser_scan: np.ndarray,
        dt: float = 0.1  # time step
    ) -> Tuple[float, dict]:
        """
        COLREGs 준수 reward 계산
        
        Args:
            os_position: Own Ship 위치 (x, y)
            os_heading: Own Ship heading (radians)
            os_velocity: Own Ship 속도 (vx, vy)
            goal_position: 목표 위치
            goal_reached: 목표 도달 여부
            collision: 충돌 여부
            target_ships: 타겟 선박 리스트
            action: 취한 액션 [surge_velocity, yaw_rate]
            laser_scan: LiDAR 스캔 데이터
            dt: 시뮬레이션 time step
        
        Returns:
            (total_reward, reward_breakdown)
        """
        self.current_step += 1
        
        reward_breakdown = {
            'terminal': 0.0,
            'colregs_compliance': 0.0,
            'risk_management': 0.0,
            'progress': 0.0,
            'efficiency': 0.0,
            'eta_compliance': 0.0,
            'safety_margin': 0.0
        }
        
        # ==========================================
        # 1. Terminal Rewards
        # ==========================================
        if goal_reached:
            reward_breakdown['terminal'] = 10000.0
            # ETA bonus/penalty
            elapsed_time = self.current_step * dt
            eta_error = abs(elapsed_time - self.expected_eta)
            if eta_error < self.expected_eta * 0.1:  # ±10% 이내
                reward_breakdown['eta_compliance'] = 2000.0
            elif eta_error < self.expected_eta * 0.2:  # ±20% 이내
                reward_breakdown['eta_compliance'] = 1000.0
            else:
                reward_breakdown['eta_compliance'] = -500.0
            
            return sum(reward_breakdown.values()), reward_breakdown
        
        if collision:
            reward_breakdown['terminal'] = -10000.0
            return sum(reward_breakdown.values()), reward_breakdown
        
        # ==========================================
        # 2. COLREGs Compliance Rewards
        # ==========================================
        colregs_reward = self._calculate_colregs_compliance(
            os_position, os_heading, os_velocity,
            target_ships, action
        )
        reward_breakdown['colregs_compliance'] = colregs_reward
        
        # ==========================================
        # 3. Risk Management Rewards (DCPA/TCPA)
        # ==========================================
        risk_reward = self._calculate_risk_reward(
            os_position, os_velocity, target_ships
        )
        reward_breakdown['risk_management'] = risk_reward
        
        # ==========================================
        # 4. Progress Rewards
        # ==========================================
        progress_reward = self._calculate_progress_reward(
            os_position, goal_position
        )
        reward_breakdown['progress'] = progress_reward
        
        # ==========================================
        # 5. Efficiency Penalties (Fuel)
        # ==========================================
        efficiency_penalty = self._calculate_efficiency_penalty(
            os_heading, action, dt
        )
        reward_breakdown['efficiency'] = efficiency_penalty
        
        # ==========================================
        # 6. Safety Margin Penalties
        # ==========================================
        safety_penalty = self._calculate_safety_margin_penalty(laser_scan)
        reward_breakdown['safety_margin'] = safety_penalty
        
        # ==========================================
        # Total Reward
        # ==========================================
        total_reward = sum(reward_breakdown.values())
        
        return total_reward, reward_breakdown
    
    def _calculate_colregs_compliance(
        self,
        os_position: Tuple[float, float],
        os_heading: float,
        os_velocity: Tuple[float, float],
        target_ships: List[dict],
        action: List[float]
    ) -> float:
        """
        COLREGs 규칙 준수 보상
        
        각 target ship에 대해:
        - Encounter type 분류
        - Give-way / Stand-on 판단
        - 올바른 행동 → +reward
        - 잘못된 행동 → -penalty
        """
        if not target_ships:
            return 0.0
        
        total_colregs_reward = 0.0
        os_speed = np.linalg.norm(os_velocity)
        os_heading_deg = np.rad2deg(os_heading) % 360
        
        for ts in target_ships:
            ts_id = ts['id']
            ts_pos = ts['position']
            ts_heading = ts['heading']  # radians
            ts_vel = ts['velocity']
            ts_speed = np.linalg.norm(ts_vel)
            ts_heading_deg = np.rad2deg(ts_heading) % 360
            
            # Encounter situation 분류
            encounter = self.encounter_classifier.classify(
                os_position=os_position,
                os_heading=os_heading_deg,
                os_speed=os_speed,
                ts_position=ts_pos,
                ts_heading=ts_heading_deg,
                ts_speed=ts_speed
            )
            
            # Encounter type별 reward
            if encounter.encounter_type == EncounterType.SAFE:
                continue  # 안전 거리, reward 없음
            
            elif encounter.encounter_type == EncounterType.HEAD_ON:
                # Rule 14: 양 선박 모두 우현 변침
                # 우현으로 회피하고 있는지 체크
                heading_change = action[1]  # yaw_rate
                if heading_change < -0.01:  # 우현 회피 (음의 yaw rate)
                    total_colregs_reward += 20.0  # 올바른 행동
                elif heading_change > 0.01:  # 좌현 회피 (위반!)
                    total_colregs_reward -= 50.0  # 강한 패널티
            
            elif encounter.encounter_type == EncounterType.OVERTAKING:
                # Rule 13: OS가 추월선 → give-way vessel
                # 피추월선의 진로 방해하지 않도록 회피
                # TS의 좌현 또는 우현으로 충분히 피항
                dcpa, tcpa = calculate_cpa_tcpa(
                    os_position, os_velocity, ts_pos, ts_vel
                )
                if dcpa > 30.0:  # 충분한 거리 확보
                    total_colregs_reward += 15.0
                else:
                    total_colregs_reward -= 30.0
            
            elif encounter.encounter_type == EncounterType.CROSSING_GIVE_WAY:
                # Rule 15: OS가 give-way vessel (TS가 우현)
                # 일반적으로 우현 변침 또는 감속
                heading_change = action[1]
                speed_change = action[0] - 3.0  # nominal speed에서의 변화
                
                # 올바른 회피 행동 체크
                if heading_change < -0.01:  # 우현 변침
                    total_colregs_reward += 25.0
                elif speed_change < -0.5:  # 감속
                    total_colregs_reward += 15.0
                else:
                    # 회피 행동 없음 → 패널티
                    total_colregs_reward -= 40.0
            
            elif encounter.encounter_type == EncounterType.CROSSING_STAND_ON:
                # Rule 15/17: OS가 stand-on vessel (TS가 좌현)
                # 침로와 속력 유지해야 함
                heading_change = action[1]
                speed_change = action[0] - 3.0
                
                # 침로/속력 유지 체크
                if abs(heading_change) < 0.01 and abs(speed_change) < 0.2:
                    total_colregs_reward += 15.0  # 올바르게 유지
                else:
                    # Stand-on vessel이 불필요하게 변침 → 패널티
                    # (Rule 17(a)(ii) 예외 상황 제외)
                    dcpa, tcpa = calculate_cpa_tcpa(
                        os_position, os_velocity, ts_pos, ts_vel
                    )
                    if dcpa < 20.0 and tcpa < 30.0:
                        # 긴급 상황 → 회피 허용 (Rule 17(b))
                        total_colregs_reward += 10.0
                    else:
                        total_colregs_reward -= 20.0
        
        return total_colregs_reward
    
    def _calculate_risk_reward(
        self,
        os_position: Tuple[float, float],
        os_velocity: Tuple[float, float],
        target_ships: List[dict]
    ) -> float:
        """
        DCPA/TCPA 기반 위험 관리 보상
        
        - DCPA 개선 → +reward
        - DCPA 악화 → -penalty
        - 위험 수준별 차등 보상
        """
        if not target_ships:
            return 0.0
        
        total_risk_reward = 0.0
        
        for ts in target_ships:
            ts_id = ts['id']
            ts_pos = ts['position']
            ts_vel = ts['velocity']
            
            # 현재 위험도 평가
            risk = self.risk_assessor.assess(
                os_position, os_velocity, ts_pos, ts_vel
            )
            
            dcpa = risk.dcpa
            tcpa = risk.tcpa
            risk_level = risk.risk_level
            
            # DCPA 개선 보상
            if ts_id in self.prev_dcpa:
                prev_dcpa = self.prev_dcpa[ts_id]
                dcpa_improvement = dcpa - prev_dcpa
                
                if dcpa_improvement > 0:
                    # DCPA 증가 (위험 감소) → 보상
                    total_risk_reward += dcpa_improvement * 2.0
                else:
                    # DCPA 감소 (위험 증가) → 패널티
                    total_risk_reward += dcpa_improvement * 5.0  # 더 큰 패널티
            
            self.prev_dcpa[ts_id] = dcpa
            
            # 위험 수준별 패널티
            if risk_level == RiskLevel.CRITICAL:
                total_risk_reward -= 50.0
            elif risk_level == RiskLevel.HIGH:
                total_risk_reward -= 20.0
            elif risk_level == RiskLevel.MEDIUM:
                total_risk_reward -= 5.0
            elif risk_level == RiskLevel.LOW:
                total_risk_reward -= 1.0
            # SAFE는 패널티 없음
        
        return total_risk_reward
    
    def _calculate_progress_reward(
        self,
        os_position: Tuple[float, float],
        goal_position: Tuple[float, float]
    ) -> float:
        """
        목표 향한 진행 보상
        """
        distance = np.linalg.norm(
            [goal_position[0] - os_position[0],
             goal_position[1] - os_position[1]]
        )
        
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - distance
            progress_reward = progress * 50.0  # 1m 진행 → +50
        else:
            progress_reward = 0.0
        
        self.prev_distance_to_goal = distance
        
        return progress_reward
    
    def _calculate_efficiency_penalty(
        self,
        os_heading: float,
        action: List[float],
        dt: float
    ) -> float:
        """
        연료 효율성 패널티 (불필요한 선회)
        
        - 큰 침로 변경 → 패널티
        - 과도한 yaw rate → 패널티
        """
        # Heading change 누적
        if self.prev_heading is not None:
            heading_change = abs(normalize_angle(os_heading - self.prev_heading))
            self.cumulative_course_change += heading_change
        
        self.prev_heading = os_heading
        
        # Yaw rate 패널티
        yaw_rate = action[1]
        yaw_penalty = -abs(yaw_rate) * 2.0  # 큰 yaw rate → 패널티
        
        # 과도한 누적 침로 변경 패널티
        if self.cumulative_course_change > np.deg2rad(90):  # 90도 이상 선회
            cumulative_penalty = -10.0
        else:
            cumulative_penalty = 0.0
        
        return yaw_penalty + cumulative_penalty
    
    def _calculate_safety_margin_penalty(
        self,
        laser_scan: np.ndarray
    ) -> float:
        """
        안전 여유 거리 패널티
        
        너무 가까운 물체 → 패널티
        """
        min_distance = np.min(laser_scan)
        
        if min_distance < 10.0:
            penalty = -(10.0 - min_distance) * 5.0
        elif min_distance < 20.0:
            penalty = -(20.0 - min_distance) * 2.0
        else:
            penalty = 0.0
        
        return penalty


class COLREGsOtterSIM(SIM_ENV):
    """
    COLREGs 준수 Otter USV 시뮬레이션 환경
    """
    
    def __init__(
        self,
        world_file="otter_world.yaml",
        disable_plotting=False,
        enable_phase1=True,
        goal_distance=180.0,
        expected_speed=3.0
    ):
        """
        Args:
            world_file: 월드 설정 파일
            disable_plotting: 렌더링 비활성화
            enable_phase1: Action frequency control 활성화
            goal_distance: 목표까지 거리 (ETA 계산용)
            expected_speed: 예상 속도 (ETA 계산용)
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.dt = self.env.step_time
        
        # COLREGs reward calculator
        self.reward_calculator = COLREGsRewardCalculator(
            goal_distance=goal_distance,
            expected_speed=expected_speed
        )
        
        # Phase 1 설정
        self.enable_phase1 = enable_phase1
        if self.enable_phase1:
            self.physics_dt = self.dt
            self.action_dt = 0.5
            self.steps_per_action = int(self.action_dt / self.physics_dt)
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
        
        print("=" * 60)
        print("COLREGs-compliant Otter USV Environment")
        print("=" * 60)
        print(f"Goal distance: {goal_distance}m")
        print(f"Expected ETA: {goal_distance/expected_speed:.1f}s")
        print("=" * 60)
    
    def step(self, u_ref=3.0, r_ref=0.0):
        """
        시뮬레이션 step 실행 및 COLREGs reward 계산
        """
        # Phase 1 action control
        if self.enable_phase1:
            if self.step_counter % self.steps_per_action == 0:
                self.current_action = np.array([[u_ref], [r_ref]])
            action = self.current_action
            self.step_counter += 1
        else:
            action = np.array([[u_ref], [r_ref]])
        
        # IR-SIM step
        self.env.step(action_id=0, action=action)
        self.env.render()
        
        # 상태 수집
        robot_state = self.env.robot.state
        os_position = (robot_state[0, 0], robot_state[1, 0])
        os_heading = robot_state[2, 0]
        
        # 속도 계산
        u = robot_state[3, 0]  # surge velocity
        v = robot_state[4, 0]  # sway velocity
        r = robot_state[5, 0]  # yaw rate
        
        # 전역 좌표계 속도
        os_velocity = (
            u * np.cos(os_heading) - v * np.sin(os_heading),
            u * np.sin(os_heading) + v * np.cos(os_heading)
        )
        
        # Goal 정보
        goal_position = (self.robot_goal[0].item(), self.robot_goal[1].item())
        goal_reached = self.env.robot.arrive
        collision = self.env.robot.collision
        
        # 센서 데이터
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        
        # Target ships 수집
        target_ships = []
        for i, obs in enumerate(self.env.obstacle_list):
            if hasattr(obs, 'state') and obs.kinematics != 'static':
                ts_state = obs.state
                ts_pos = (ts_state[0, 0], ts_state[1, 0])
                ts_heading = ts_state[2, 0]
                
                # TS 속도
                if ts_state.shape[0] >= 6:
                    ts_u = ts_state[3, 0]
                    ts_v = ts_state[4, 0]
                    ts_velocity = (
                        ts_u * np.cos(ts_heading) - ts_v * np.sin(ts_heading),
                        ts_u * np.sin(ts_heading) + ts_v * np.cos(ts_heading)
                    )
                else:
                    ts_velocity = (0.0, 0.0)
                
                target_ships.append({
                    'id': i,
                    'position': ts_pos,
                    'heading': ts_heading,
                    'velocity': ts_velocity
                })
        
        # COLREGs reward 계산
        reward, reward_breakdown = self.reward_calculator.calculate_reward(
            os_position=os_position,
            os_heading=os_heading,
            os_velocity=os_velocity,
            goal_position=goal_position,
            goal_reached=goal_reached,
            collision=collision,
            target_ships=target_ships,
            action=[u_ref, r_ref],
            laser_scan=latest_scan,
            dt=self.dt
        )
        
        # 거리 및 방향 계산
        goal_vector = [
            goal_position[0] - os_position[0],
            goal_position[1] - os_position[1]
        ]
        distance = np.linalg.norm(goal_vector)
        pose_vector = [np.cos(os_heading), np.sin(os_heading)]
        cos, sin = self.cossin(pose_vector, goal_vector)
        
        action_list = [u_ref, r_ref]
        
        return (latest_scan, distance, cos, sin, collision, goal_reached,
                action_list, reward, reward_breakdown)
    
    def reset(self, robot_state=None, robot_goal=None):
        """환경 리셋"""
        # Reward calculator 리셋
        self.reward_calculator.reset()
        
        if robot_state is None:
            robot_state = [0], [-90], [random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)]
        
        # 로봇 상태 리셋
        self.env.robot.state[0, 0] = robot_state[0][0]
        self.env.robot.state[1, 0] = robot_state[1][0]
        self.env.robot.state[2, 0] = robot_state[2][0]
        
        if self.env.robot.state.shape[0] >= 6:
            self.env.robot.state[3, 0] = 0.0
            self.env.robot.state[4, 0] = 0.0
            self.env.robot.state[5, 0] = 0.0
        
        if self.env.robot.state.shape[0] >= 8:
            self.env.robot.state[6, 0] = 0.0
            self.env.robot.state[7, 0] = 0.0
        
        self.env.robot._geometry = self.env.robot.gf.step(self.env.robot.state)
        self.env.robot._init_state = self.env.robot.state.copy()
        
        if robot_goal is None:
            robot_goal = [0, 90, 1.5708]
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        if self.enable_phase1:
            self.step_counter = 0
            self.current_action = np.array([[0.0], [0.0]])
        
        action = [1.5, 0.0]
        result = self.step(u_ref=action[0], r_ref=action[1])
        
        return result[:8]  # reward_breakdown 제외
