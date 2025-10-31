"""
COLREGs-Enhanced Otter USV Environment
========================================

이 환경은 기존 OtterSIM에 COLREGs 분석 기능을 추가합니다:
1. Observation에 encounter type 추가
2. Observation에 risk level 추가  
3. Observation에 DCPA/TCPA 추가
4. COLREGs 준수 여부 체크

Author: Maritime Robotics Lab
Date: 2025-10-29
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import irsim
import random

from robot_nav.SIM_ENV.otter_sim import OtterSIM
from colregs_core import (
    EncounterClassifier,
    RiskAssessment,
    EncounterType,
    RiskLevel,
    calculate_cpa_tcpa,
    irsim_to_nav_heading,
    irsim_velocity_to_nav
)


class COLREGsEnhancedOtterEnv(OtterSIM):
    """
    COLREGs 정보가 포함된 Enhanced Otter USV Environment
    
    Observation Space 확장:
    - 기존: (laser_scan, distance, cos, sin)
    - 추가: (encounter_types, risk_levels, dcpa_list, tcpa_list, colregs_violations)
    """
    
    def __init__(
        self,
        world_file: str = "otter_world.yaml",
        disable_plotting: bool = False,
        enable_phase1: bool = True,
        safe_distance: float = 100.0,  # COLREGs 판단 거리
        include_colregs_in_obs: bool = True  # Observation에 COLREGs 포함 여부
    ):
        """
        Args:
            world_file: 월드 설정 파일
            disable_plotting: 렌더링 비활성화
            enable_phase1: Action frequency control 활성화
            safe_distance: COLREGs 판단 안전 거리
            include_colregs_in_obs: True면 observation에 COLREGs 정보 포함
        """
        super().__init__(
            world_file=world_file,
            disable_plotting=disable_plotting,
            enable_phase1=enable_phase1
        )
        
        self.safe_distance = safe_distance
        self.include_colregs_in_obs = include_colregs_in_obs
        
        # COLREGs 분석 모듈
        self.encounter_classifier = EncounterClassifier(
            safe_distance=safe_distance
        )
        self.risk_assessor = RiskAssessment()
        
        # 통계 추적
        self.episode_encounters = {
            'HEAD_ON': 0,
            'CROSSING_GIVE_WAY': 0,
            'CROSSING_STAND_ON': 0,
            'OVERTAKING': 0,
            'SAFE': 0
        }
        self.colregs_violations = 0
        self.total_steps = 0
        
        print("\n" + "=" * 70)
        print("COLREGs-Enhanced Otter USV Environment")
        print("=" * 70)
        print(f"World file: {world_file}")
        print(f"Safe distance: {safe_distance}m")
        print(f"COLREGs in observation: {include_colregs_in_obs}")
        print("=" * 70 + "\n")
    
    def step(self, u_ref: float = 3.0, r_ref: float = 0.0):
        """
        환경 step 실행 및 COLREGs 분석
        
        Returns:
            if include_colregs_in_obs:
                (laser_scan, distance, cos, sin, collision, arrive, 
                 action, reward, colregs_info)
            else:
                (laser_scan, distance, cos, sin, collision, arrive, 
                 action, reward)
        """
        # 기존 OtterSIM step 실행
        result = super().step(u_ref=u_ref, r_ref=r_ref)
        
        if not self.include_colregs_in_obs:
            return result
        
        # COLREGs 분석 추가
        colregs_info = self._analyze_colregs_situation()
        
        # 통계 업데이트
        self.total_steps += 1
        for enc_type, count in colregs_info['encounter_counts'].items():
            self.episode_encounters[enc_type] += count
        
        # 결과에 colregs_info 추가
        return (*result, colregs_info)
    
    def _analyze_colregs_situation(self) -> Dict:
        """
        현재 상황의 COLREGs 분석
        
        Returns:
            {
                'encounters': [EncounterType, ...],
                'risk_levels': [RiskLevel, ...],
                'dcpa_list': [float, ...],
                'tcpa_list': [float, ...],
                'encounter_counts': {'HEAD_ON': 0, ...},
                'most_critical_target': int or None,
                'colregs_compliant': bool
            }
        """
        robot_state = self.env.robot.state
        os_position = (robot_state[0, 0], robot_state[1, 0])
        os_heading_irsim = robot_state[2, 0]  # radians, ir-sim 좌표계
        
        # ir-sim → navigation 좌표계 변환
        os_heading_nav = irsim_to_nav_heading(np.rad2deg(os_heading_irsim))
        
        # OS 속도
        u = robot_state[3, 0]  # surge
        v = robot_state[4, 0]  # sway
        
        # ir-sim 좌표계 전역 속도
        os_velocity_irsim = (
            u * np.cos(os_heading_irsim) - v * np.sin(os_heading_irsim),
            u * np.sin(os_heading_irsim) + v * np.cos(os_heading_irsim)
        )
        os_speed = np.sqrt(os_velocity_irsim[0]**2 + os_velocity_irsim[1]**2)
        
        encounters = []
        risk_levels = []
        dcpa_list = []
        tcpa_list = []
        encounter_counts = {
            'HEAD_ON': 0,
            'CROSSING_GIVE_WAY': 0,
            'CROSSING_STAND_ON': 0,
            'OVERTAKING': 0,
            'SAFE': 0
        }
        
        most_critical_idx = None
        highest_risk_value = -1
        
        # 각 장애물(선박)에 대해 분석
        for i, obs in enumerate(self.env.obstacle_list):
            if hasattr(obs, 'state') and obs.kinematics != 'static':
                ts_state = obs.state
                ts_position = (ts_state[0, 0], ts_state[1, 0])
                ts_heading_irsim = ts_state[2, 0]  # radians
                
                # ir-sim → navigation 좌표계 변환
                ts_heading_nav = irsim_to_nav_heading(np.rad2deg(ts_heading_irsim))
                
                # TS 속도
                if ts_state.shape[0] >= 6:
                    ts_u = ts_state[3, 0]
                    ts_v = ts_state[4, 0]
                    ts_velocity_irsim = (
                        ts_u * np.cos(ts_heading_irsim) - ts_v * np.sin(ts_heading_irsim),
                        ts_u * np.sin(ts_heading_irsim) + ts_v * np.cos(ts_heading_irsim)
                    )
                    ts_speed = np.sqrt(ts_velocity_irsim[0]**2 + ts_velocity_irsim[1]**2)
                else:
                    ts_velocity_irsim = (0.0, 0.0)
                    ts_speed = 0.0
                
                # 1. Encounter 분류 (Navigation 좌표계)
                encounter_situation = self.encounter_classifier.classify(
                    os_position=os_position,
                    os_heading=os_heading_nav,
                    os_speed=os_speed,
                    ts_position=ts_position,
                    ts_heading=ts_heading_nav,
                    ts_speed=ts_speed
                )
                
                encounters.append(encounter_situation.encounter_type)
                encounter_counts[encounter_situation.encounter_type.name] += 1
                
                # 2. 충돌 위험 평가 (좌표계 무관, 위치-속도만 사용)
                risk = self.risk_assessor.assess(
                    os_position=os_position,
                    os_velocity=os_velocity_irsim,
                    ts_position=ts_position,
                    ts_velocity=ts_velocity_irsim
                )
                
                risk_levels.append(risk.risk_level)
                dcpa_list.append(risk.dcpa)
                tcpa_list.append(risk.tcpa)
                
                # 가장 위험한 target 추적
                if risk.risk_level.value > highest_risk_value:
                    highest_risk_value = risk.risk_level.value
                    most_critical_idx = i
        
        # COLREGs 준수 여부 (간단한 체크)
        colregs_compliant = self._check_colregs_compliance(
            encounters, risk_levels
        )
        
        return {
            'encounters': encounters,
            'risk_levels': risk_levels,
            'dcpa_list': dcpa_list,
            'tcpa_list': tcpa_list,
            'encounter_counts': encounter_counts,
            'most_critical_target': most_critical_idx,
            'colregs_compliant': colregs_compliant
        }
    
    def _check_colregs_compliance(
        self,
        encounters: List[EncounterType],
        risk_levels: List[RiskLevel]
    ) -> bool:
        """
        COLREGs 준수 여부 간단 체크
        
        위험한 상황(HIGH/CRITICAL)에서 적절한 encounter type이 아니면 위반
        """
        for encounter, risk in zip(encounters, risk_levels):
            if risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                if encounter == EncounterType.UNDEFINED:
                    return False  # 위험한데 상황 분류 안됨
        
        return True
    
    def reset(self, robot_state=None, robot_goal=None):
        """환경 리셋 및 통계 초기화"""
        # 통계 초기화
        self.episode_encounters = {
            'HEAD_ON': 0,
            'CROSSING_GIVE_WAY': 0,
            'CROSSING_STAND_ON': 0,
            'OVERTAKING': 0,
            'SAFE': 0
        }
        self.colregs_violations = 0
        self.total_steps = 0
        
        return super().reset(robot_state=robot_state, robot_goal=robot_goal)
    
    def get_episode_statistics(self) -> Dict:
        """
        에피소드 통계 반환
        
        Returns:
            {
                'total_steps': int,
                'encounters': {'HEAD_ON': count, ...},
                'encounter_rates': {'HEAD_ON': ratio, ...},
                'colregs_violations': int,
                'compliance_rate': float
            }
        """
        encounter_rates = {}
        for enc_type, count in self.episode_encounters.items():
            if self.total_steps > 0:
                encounter_rates[enc_type] = count / self.total_steps
            else:
                encounter_rates[enc_type] = 0.0
        
        compliance_rate = 1.0 - (self.colregs_violations / max(self.total_steps, 1))
        
        return {
            'total_steps': self.total_steps,
            'encounters': self.episode_encounters.copy(),
            'encounter_rates': encounter_rates,
            'colregs_violations': self.colregs_violations,
            'compliance_rate': compliance_rate
        }
    
    def print_episode_summary(self):
        """에피소드 요약 출력"""
        stats = self.get_episode_statistics()
        
        print("\n" + "=" * 70)
        print("COLREGs Episode Summary")
        print("=" * 70)
        print(f"Total steps: {stats['total_steps']}")
        print("\nEncounter Statistics:")
        for enc_type, count in stats['encounters'].items():
            rate = stats['encounter_rates'][enc_type] * 100
            print(f"  {enc_type:20s}: {count:4d} ({rate:5.1f}%)")
        print(f"\nCOLREGs Compliance Rate: {stats['compliance_rate']*100:.1f}%")
        print("=" * 70 + "\n")


def get_colregs_observation_vector(colregs_info: Dict, max_targets: int = 5) -> np.ndarray:
    """
    COLREGs 정보를 고정 크기 벡터로 변환 (RL observation 용)
    
    Args:
        colregs_info: _analyze_colregs_situation()의 반환값
        max_targets: 최대 target 수
    
    Returns:
        numpy array of shape (max_targets * 8,)
        각 target당 [encounter_type_onehot(5), risk_level(1), dcpa_normalized(1), tcpa_normalized(1)]
        = 8 values per target
    """
    encounters = colregs_info['encounters']
    risk_levels = colregs_info['risk_levels']
    dcpa_list = colregs_info['dcpa_list']
    tcpa_list = colregs_info['tcpa_list']
    
    n_targets = len(encounters)
    
    # 고정 크기 벡터 초기화
    obs_vector = np.zeros(max_targets * 8)
    
    for i in range(min(n_targets, max_targets)):
        base_idx = i * 8
        
        # Encounter type one-hot encoding (5 types)
        encounter_idx = encounters[i].value
        # EncounterType enum value를 index로 변환
        encounter_map = {
            'head_on': 0,
            'overtaking': 1,
            'crossing_give_way': 2,
            'crossing_stand_on': 3,
            'safe': 4,
            'undefined': 4  # SAFE와 동일하게 처리
        }
        enc_idx = encounter_map.get(encounter_idx, 4)
        obs_vector[base_idx + enc_idx] = 1.0
        
        # Risk level (normalized 0-1)
        obs_vector[base_idx + 5] = risk_levels[i].value / 4.0  # 0-4 → 0-1
        
        # DCPA normalized (0-200m → 0-1)
        obs_vector[base_idx + 6] = np.clip(dcpa_list[i] / 200.0, 0.0, 1.0)
        
        # TCPA normalized (0-120s → 0-1)  
        if tcpa_list[i] > 0:
            obs_vector[base_idx + 7] = np.clip(tcpa_list[i] / 120.0, 0.0, 1.0)
        else:
            obs_vector[base_idx + 7] = 0.0  # 음수 TCPA는 0으로
    
    return obs_vector


# ============================================================================
# 사용 예제
# ============================================================================

if __name__ == "__main__":
    """
    Enhanced Environment 테스트
    """
    print("\n" + "="*70)
    print("Testing COLREGs-Enhanced Otter Environment")
    print("="*70 + "\n")
    
    # 환경 생성
    env = COLREGsEnhancedOtterEnv(
        world_file="worlds/imazu_scenario/s1.yaml",
        disable_plotting=False,
        include_colregs_in_obs=True
    )
    
    # 리셋
    obs = env.reset()
    print(f"Initial observation length: {len(obs)}")
    
    # 몇 스텝 실행
    for step in range(50):
        # 간단한 액션
        action = [3.0, 0.0]  # 직진
        
        result = env.step(u_ref=action[0], r_ref=action[1])
        
        if len(result) == 9:  # COLREGs info 포함
            (laser_scan, distance, cos, sin, collision, 
             arrive, action_list, reward, colregs_info) = result
            
            if step % 10 == 0:
                print(f"\nStep {step}:")
                print(f"  Distance to goal: {distance:.1f}m")
                print(f"  Encounters: {colregs_info['encounter_counts']}")
                print(f"  COLREGs compliant: {colregs_info['colregs_compliant']}")
                
                if colregs_info['most_critical_target'] is not None:
                    idx = colregs_info['most_critical_target']
                    print(f"  Most critical target: #{idx}")
                    print(f"    - Risk: {colregs_info['risk_levels'][idx].name}")
                    print(f"    - DCPA: {colregs_info['dcpa_list'][idx]:.1f}m")
                    print(f"    - TCPA: {colregs_info['tcpa_list'][idx]:.1f}s")
        
        if collision or arrive:
            break
    
    # 에피소드 요약 출력
    env.print_episode_summary()
    
    print("\n✓ Test completed successfully!\n")
