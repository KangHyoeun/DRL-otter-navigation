# COLREGs Test Scenarios
========================

이 폴더는 COLREGs 규칙을 테스트하기 위한 4가지 표준 시나리오를 포함합니다.

## 📋 시나리오 목록

### 1. **head_on.yaml** (Rule 14)
**정면 조우 상황**
- Own Ship: 남쪽 → 북쪽 (90°)
- Target Ship: 북쪽 → 남쪽 (270°)
- **COLREGs Rule 14**: 양 선박 모두 우현으로 변침
- **학습 목표**: 정면 조우 시 우현 회피 학습

```bash
# 테스트 실행
cd /home/hyo/DRL-otter-navigation
conda activate DRL-otter-nav
poetry run python3 robot_nav/colregs_otter_test_v2.py --world worlds/colregs_scenarios/head_on.yaml
```

---

### 2. **crossing_giveway.yaml** (Rule 15 - Give-Way)
**횡단 상황 (피항선)**
- Own Ship: 서쪽 → 동쪽 (0°)
- Target Ship: OS 우현에서 남향 접근 (270°)
- **COLREGs Rule 15**: OS가 give-way vessel, TS의 진로 피해야 함
- **학습 목표**: 우현 변침 또는 감속으로 회피

```bash
poetry run python3 robot_nav/colregs_otter_test_v2.py --world worlds/colregs_scenarios/crossing_giveway.yaml
```

---

### 3. **crossing_standon.yaml** (Rule 15 & 17 - Stand-On)
**횡단 상황 (유지선)**
- Own Ship: 서쪽 → 동쪽 (0°)
- Target Ship: OS 좌현에서 북향 접근 (90°)
- **COLREGs Rule 15 & 17**: OS가 stand-on vessel, 침로/속력 유지
- **학습 목표**: 침로 유지, 필요시 Rule 17(a)(ii)에 따라 자체 회피

```bash
poetry run python3 robot_nav/colregs_otter_test_v2.py --world worlds/colregs_scenarios/crossing_standon.yaml
```

---

### 4. **overtaking.yaml** (Rule 13)
**추월 상황**
- Own Ship: 남쪽 → 북쪽 (90°, 고속 4m/s)
- Target Ship: OS 전방, 동일 방향 (90°, 저속 2m/s)
- **COLREGs Rule 13**: 추월선(OS)이 피항선
- **학습 목표**: TS 진로 방해 없이 좌/우현으로 추월

```bash
poetry run python3 robot_nav/colregs_otter_test_v2.py --world worlds/colregs_scenarios/overtaking.yaml
```

---

## 🎯 사용 방법

### 1. **학습 (Training)**
```bash
# 순차 학습: 간단한 시나리오부터
poetry run python3 robot_nav/colregs_otter_train_v2.py \
    --world worlds/colregs_scenarios/crossing_giveway.yaml \
    --episodes 1000 \
    --save_dir checkpoints/colregs_crossing_giveway

# 다음 단계
poetry run python3 robot_nav/colregs_otter_train_v2.py \
    --world worlds/colregs_scenarios/head_on.yaml \
    --episodes 1000 \
    --load_checkpoint checkpoints/colregs_crossing_giveway/best_model.pth
```

### 2. **테스트 (Testing)**
```bash
# 단일 시나리오 테스트
poetry run python3 robot_nav/colregs_otter_test_v2.py \
    --world worlds/colregs_scenarios/head_on.yaml \
    --checkpoint checkpoints/colregs_model/best_model.pth \
    --num_episodes 10

# 모든 시나리오 일괄 테스트
poetry run python3 robot_nav/colregs_otter_test_v2.py \
    --test_all_scenarios \
    --checkpoint checkpoints/colregs_model/best_model.pth
```

### 3. **Enhanced Environment 직접 사용**
```python
from robot_nav.SIM_ENV.colregs_enhanced_env import COLREGsEnhancedOtterEnv

# 환경 생성
env = COLREGsEnhancedOtterEnv(
    world_file="worlds/colregs_scenarios/head_on.yaml",
    disable_plotting=False,
    include_colregs_in_obs=True
)

# 리셋
obs = env.reset()

# Step
for i in range(100):
    action = [3.0, 0.0]  # [u_ref, r_ref]
    result = env.step(u_ref=action[0], r_ref=action[1])
    
    if len(result) == 9:  # COLREGs info 포함
        (laser_scan, distance, cos, sin, collision, 
         arrive, action_list, reward, colregs_info) = result
        
        # COLREGs 정보 활용
        print(f"Encounters: {colregs_info['encounter_counts']}")
        print(f"Risk levels: {colregs_info['risk_levels']}")
    
    if collision or arrive:
        break

# 에피소드 통계
env.print_episode_summary()
```

---

## 📊 평가 지표

각 시나리오에서 다음 지표를 측정합니다:

1. **COLREGs Compliance Rate**: 규칙 준수율
2. **Success Rate**: 목표 도달 성공률
3. **Collision Rate**: 충돌 발생률
4. **Average DCPA**: 평균 최소 접근 거리
5. **Average TCPA**: 평균 최접근 시간
6. **Encounter Type Distribution**: 조우 상황 분포

---

## 🔧 시나리오 커스터마이징

각 YAML 파일을 수정하여 다양한 상황을 테스트할 수 있습니다:

```yaml
# 선박 속도 변경
reactive_rule:
  type: velocity
  vel: [0, 3.0, 0]  # [vx, vy, w] - 속도 조정

# 초기 위치 변경
state: [x, y, heading]  # 위치와 방향 조정

# 여러 선박 추가
obstacle:
  - kinematics: omni
    id: 2
    state: [x2, y2, heading2]
    ...
```

---

## 📚 참고 자료

- **COLREGs Rules**: `/home/hyo/colregs-core/docs/colregs_rules.md`
- **한국 해상교통안전법**: `/mnt/project/대한민국_법률_-_해상교통안전법`
- **Integration Guide**: `docs/COLREGS_INTEGRATION_GUIDE.md`

---

## 🚢 시나리오 요약

| Scenario | Rule | OS Role | Action Required |
|----------|------|---------|-----------------|
| head_on | 14 | Give-Way | 우현 변침 |
| crossing_giveway | 15 | Give-Way | 우현 변침/감속 |
| crossing_standon | 15,17 | Stand-On | 침로 유지 |
| overtaking | 13 | Give-Way | 좌/우현 추월 |

---

**Author**: Maritime Robotics Lab  
**Date**: 2025-10-29  
**Version**: 1.0
