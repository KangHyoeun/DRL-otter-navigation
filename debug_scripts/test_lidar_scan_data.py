#!/usr/bin/env python3
"""
Lidar Scan Data 검증 스크립트
- 10 step 동안 lidar scan data 360개 전부 출력
- 각 step에서 robot 위치, heading, lidar scan statistics 확인
"""

import numpy as np
import sys
import os

# DRL-otter-navigation 패키지 import를 위한 경로 설정
sys.path.insert(0, '/home/hyo/DRL-otter-navigation')
sys.path.insert(0, '/home/hyo/DRL-otter-navigation/robot_nav')

from robot_nav.SIM_ENV.otter_sim import OtterSIM


def print_separator(title=""):
    """구분선 출력"""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def analyze_lidar_scan(scan_data, step_num):
    """Lidar scan 데이터 분석 및 출력"""
    print_separator(f"STEP {step_num}: LIDAR SCAN DATA (360 beams)")
    
    # 기본 통계
    print(f"\n📊 기본 통계:")
    print(f"  - 전체 beam 개수: {len(scan_data)}")
    print(f"  - 최소 거리: {np.min(scan_data):.3f} m")
    print(f"  - 최대 거리: {np.max(scan_data):.3f} m")
    print(f"  - 평균 거리: {np.mean(scan_data):.3f} m")
    print(f"  - 중간값 거리: {np.median(scan_data):.3f} m")
    
    # 장애물 감지 통계 (100m보다 가까운 것)
    detected = scan_data < 100.0
    num_detected = np.sum(detected)
    print(f"\n🎯 장애물 감지:")
    print(f"  - 감지된 beam: {num_detected} / 360")
    print(f"  - 감지율: {num_detected/360*100:.1f}%")
    
    if num_detected > 0:
        detected_ranges = scan_data[detected]
        print(f"  - 감지된 거리 최소: {np.min(detected_ranges):.3f} m")
        print(f"  - 감지된 거리 평균: {np.mean(detected_ranges):.3f} m")
    
    # 360개 scan data 전부 출력 (10개씩 줄바꿈)
    print(f"\n📡 전체 360개 Scan Data:")
    print("  (각도 0° = North, 시계방향, 10개씩 출력)")
    print("-" * 80)
    
    for i in range(0, 360, 10):
        # 각 beam의 각도 (degree)
        angles = [f"{j:3d}°" for j in range(i, min(i+10, 360))]
        ranges = [f"{scan_data[j]:6.2f}" for j in range(i, min(i+10, 360))]
        
        print(f"  Angle: {' '.join(angles)}")
        print(f"  Range: {' '.join(ranges)} m")
        print()


def main():
    """메인 실행 함수"""
    
    # Environment 생성
    print_separator("LIDAR SCAN DATA VERIFICATION")
    print("\n🚀 Environment 초기화 중...")
    
    # OtterSIM 사용 (irsim.make 기반)
    # Use absolute path to ensure the world file is found
    world_file_path = os.path.join(
        '/home/hyo/DRL-otter-navigation',
        'robot_nav/worlds/imazu_scenario/imazu_case_01.yaml'
    )
    
    if not os.path.exists(world_file_path):
        raise FileNotFoundError(
            f"World file not found: {world_file_path}\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    sim = OtterSIM(
        world_file=world_file_path,
        disable_plotting=False,  # visualization 활성화
        enable_phase1=False,  # 테스트용이니 action frequency control 비활성화
        max_steps=1000
    )
    
    print("✅ Environment 초기화 완료!")
    
    # 초기 상태
    latest_scan, distance, cos, sin, collision, goal, action, reward, robot_state = sim.reset()
    
    # Robot 초기 상태 출력
    print(f"\n🤖 Robot 초기 상태:")
    print(f"  - Position (North, East): ({robot_state[0, 0]:.2f}, {robot_state[1, 0]:.2f}) m")
    print(f"  - Heading (rad): {robot_state[2, 0]:.4f}")
    print(f"  - Heading (deg): {np.degrees(robot_state[2, 0]):.2f}°")
    print(f"  - Goal position: {sim.robot_goal.T}")
    
    # 초기 lidar scan 분석
    analyze_lidar_scan(latest_scan, 0)
    
    # 10 step 실행
    print_separator("10 STEP 시뮬레이션 시작")
    
    for step in range(10):
        # 간단한 action (앞으로 직진)
        u_ref = 0.5  # 0.5 m/s 전진
        r_ref = 0.0  # 0 rad/s 회전 (직진)
        
        # Step 실행
        latest_scan, distance, cos, sin, collision, goal, action, reward, robot_state = sim.step(
            u_ref=u_ref, r_ref=r_ref
        )
        
        print(f"\n📍 Robot 현재 상태 (Step {step+1}):")
        print(f"  - Position (North, East): ({robot_state[0, 0]:.2f}, {robot_state[1, 0]:.2f}) m")
        print(f"  - Heading (rad): {robot_state[2, 0]:.4f}")
        print(f"  - Heading (deg): {np.degrees(robot_state[2, 0]):.2f}°")
        print(f"  - Velocity (u, r): ({robot_state[3, 0]:.3f} m/s, {robot_state[5, 0]:.3f} rad/s)")
        print(f"  - Distance to goal: {distance:.2f} m")
        print(f"  - Action: u_ref={action[0]:.3f}, r_ref={action[1]:.3f}")
        print(f"  - Reward: {reward:.3f}")
        print(f"  - Collision: {collision}, Goal: {goal}")
        
        # Lidar scan 분석 및 출력
        analyze_lidar_scan(latest_scan, step+1)
        
        # Episode 종료 체크
        if collision or goal:
            print("\n⚠️ Episode terminated!")
            if collision:
                print("  Reason: Collision detected")
            if goal:
                print("  Reason: Goal reached")
            break
    
    print_separator("VERIFICATION COMPLETE")
    print("✅ Lidar scan data 검증 완료!\n")


if __name__ == "__main__":
    main()
