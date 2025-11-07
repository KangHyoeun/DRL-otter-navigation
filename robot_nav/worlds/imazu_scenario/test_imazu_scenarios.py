"""
Imazu Scenarios Test - COLREGs Analysis

각 Imazu 시나리오에 대해:
1. Encounter 분류 (Head-on, Crossing, Overtaking)
2. 위험도 평가 (DCPA, TCPA, Risk Level)
3. 권장 조치 제공
"""

import yaml
import numpy as np
import math
from pathlib import Path
from colregs_core import (
    EncounterClassifier,
    RiskAssessment,
    heading_speed_to_velocity,
    EncounterType
)


def parse_imazu_yaml(yaml_path):
    """
    Parse Imazu YAML file and extract ship information
    
    Returns:
        dict: {
            'own_ship': {pos, heading, speed},
            'target_ships': [{pos, heading, speed}, ...]
        }
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Own Ship 정보 추출
    robot = data['robot']
    os_state = robot['state']
    
    # state: [x, y, psi, u, v, r, 0, 0]
    # x, y: position (m)
    # psi: heading (radians, ALREADY in MARITIME coordinates)
    # coordinate_system: maritime 이므로 좌표계 변환 불필요!
    os_x, os_y, os_psi_rad = os_state[0], os_state[1], os_state[2]
    
    # Radians → Degrees (Maritime heading, no coordinate conversion needed)
    os_heading_maritime = math.degrees(os_psi_rad)
    
    # Default speed for Own Ship (from behavior or assume 3.0 m/s)
    os_speed = 3.0
    
    own_ship = {
        'position': np.array([[os_x], [os_y]]),
        'heading': os_heading_maritime,
        'speed': os_speed
    }
    
    # Target Ships 정보 추출
    target_ships = []
    obstacles = data.get('obstacle', [])
    
    for obs in obstacles:
        # Skip world boundary
        if 'number' not in obs:
            continue
            
        ts_state = obs['state']
        ts_x, ts_y, ts_psi_rad = ts_state[0], ts_state[1], ts_state[2]
        
        # Radians → Degrees (Maritime heading, no coordinate conversion needed)
        ts_heading_maritime = math.degrees(ts_psi_rad)
        
        # Extract speed from behavior
        behavior = obs.get('behavior', {})
        ts_speed = behavior.get('reference_velocity', 3.0)
        
        target_ships.append({
            'number': obs['number'],
            'position': np.array([[ts_x], [ts_y]]),
            'heading': ts_heading_maritime,
            'speed': ts_speed
        })
    
    return {
        'own_ship': own_ship,
        'target_ships': target_ships
    }


def get_colregs_action(encounter_type):
    """Get COLREGs action requirement for encounter type"""
    actions = {
        EncounterType.HEAD_ON: "Rule 14: 양 선박 모두 우현으로 변침",
        EncounterType.CROSSING_GIVE_WAY: "Rule 15: 우현에 타선이 있으면 피항선 - 속도 감속 또는 좌현 변침",
        EncounterType.CROSSING_STAND_ON: "Rule 15: 좌현에 타선이 있으면 유지선 - 침로와 속도 유지",
        EncounterType.OVERTAKING: "Rule 13: 추월 상황 - 추월하는 선박이 피항",
        EncounterType.SAFE: "No action required - 안전 거리",
        EncounterType.UNDEFINED: "상황 불명 - 주의 항해"
    }
    return actions.get(encounter_type, "규칙 적용 없음")


def analyze_single_scenario(case_num, yaml_path):
    """
    Analyze single Imazu scenario
    """
    print("=" * 60)
    print(f"Case {case_num:02d}: {yaml_path.name}")
    print("=" * 60)
    
    # Parse YAML
    scenario = parse_imazu_yaml(yaml_path)
    own_ship = scenario['own_ship']
    target_ships = scenario['target_ships']
    
    print(f"\n📍 Own Ship:")
    print(f"   Position: ({own_ship['position'][0,0]:.1f}, {own_ship['position'][1,0]:.1f}) m")
    print(f"   Heading: {own_ship['heading']:.1f}° (Maritime)")
    print(f"   Speed: {own_ship['speed']:.1f} m/s")
    print(f"\n🚢 Target Ships: {len(target_ships)} ship(s)")
    
    # Initialize classifiers
    classifier = EncounterClassifier()
    risk_assessor = RiskAssessment()
    
    # Analyze each target ship
    for i, ts in enumerate(target_ships, 1):
        print(f"\n{'─' * 60}")
        print(f"Target Ship #{ts['number']} Analysis")
        print(f"{'─' * 60}")
        
        print(f"📍 Position: ({ts['position'][0,0]:.1f}, {ts['position'][1,0]:.1f}) m")
        print(f"📐 Heading: {ts['heading']:.1f}° (Maritime)")
        print(f"⚡ Speed: {ts['speed']:.1f} m/s")
        
        # Distance
        distance = np.linalg.norm(own_ship['position'] - ts['position'])
        print(f"📏 Distance: {distance:.1f} m")
        
        # Encounter Classification
        print(f"\n🔍 Encounter Classification:")
        situation = classifier.classify(
            os_position=own_ship['position'],
            os_heading=own_ship['heading'],
            os_speed=own_ship['speed'],
            ts_position=ts['position'],
            ts_heading=ts['heading'],
            ts_speed=ts['speed']
        )
        
        print(f"   Type: {situation.encounter_type.value.upper()}")
        print(f"   Relative Bearing: {situation.relative_bearing:.1f}° (OS → TS 방향)")
        print(f"   Relative Course: {situation.relative_course:.1f}° (TS - OS heading)")
        print(f"   Aspect Angle: {situation.aspect_angle:.1f}° (TS → OS 방향)")
        
        # Risk Assessment
        print(f"\n⚠️  Risk Assessment:")
        os_velocity = heading_speed_to_velocity(own_ship['heading'], own_ship['speed'])
        ts_velocity = heading_speed_to_velocity(ts['heading'], ts['speed'])
        
        risk = risk_assessor.assess(
            os_position=own_ship['position'],
            os_velocity=os_velocity,
            ts_position=ts['position'],
            ts_velocity=ts_velocity
        )
        
        print(f"   Risk Level: {risk.risk_level.name}")
        print(f"   DCPA: {risk.dcpa:.1f} m ({risk.dcpa/1852:.3f} NM)")
        print(f"   TCPA: {risk.tcpa:.1f} s ({risk.tcpa/60:.1f} min)")
        
        if abs(risk.bearing_rate) < 0.1:
            print(f"   ⚠️  CONSTANT BEARING - 충돌 코스!")
        
        # Recommended Actions
        print(f"\n📋 Recommended Actions:")
        
        if risk.requires_action:
            print(f"   🚨 회피 조치 필요!")
            
            # COLREGs action
            colregs_action = get_colregs_action(situation.encounter_type)
            print(f"\n   COLREGs: {colregs_action}")
            
            # Tactical recommendation
            tactical_action = risk_assessor.get_recommended_action(risk)
            print(f"   Tactical: {tactical_action}")
        else:
            print(f"   ✅ 정상 항해 유지 (계속 감시)")
    
    print(f"\n{'=' * 60}\n")


def analyze_all_scenarios(start=1, end=22):
    """
    Analyze all Imazu scenarios from start to end
    """
    base_dir = Path(__file__).parent
    
    print("🌊" * 40)
    print(" " * 30 + "Imazu Scenarios Analysis")
    print("🌊" * 40)
    print(f"\nAnalyzing Cases {start:02d} to {end:02d}")
    print(f"Using COLREGs Core Package\n")
    
    for case_num in range(start, end + 1):
        yaml_file = base_dir / f"imazu_case_{case_num:02d}.yaml"
        
        if not yaml_file.exists():
            print(f"⚠️  Case {case_num:02d} not found: {yaml_file}")
            continue
        
        try:
            analyze_single_scenario(case_num, yaml_file)
        except Exception as e:
            print(f"❌ Error analyzing Case {case_num:02d}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """
    Main entry point
    
    Usage:
        # Analyze all scenarios
        python test_imazu_scenarios.py
        
        # Analyze specific range
        python test_imazu_scenarios.py 1 5
    """
    import sys
    
    if len(sys.argv) == 1:
        # Analyze all scenarios
        analyze_all_scenarios(1, 22)
    elif len(sys.argv) == 2:
        # Analyze single scenario
        case_num = int(sys.argv[1])
        base_dir = Path(__file__).parent
        yaml_file = base_dir / f"imazu_case_{case_num:02d}.yaml"
        analyze_single_scenario(case_num, yaml_file)
    elif len(sys.argv) == 3:
        # Analyze range
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        analyze_all_scenarios(start, end)
    else:
        print("Usage:")
        print("  python test_imazu_scenarios.py           # All scenarios")
        print("  python test_imazu_scenarios.py 5         # Single scenario")
        print("  python test_imazu_scenarios.py 1 5       # Range of scenarios")


if __name__ == "__main__":
    main()
