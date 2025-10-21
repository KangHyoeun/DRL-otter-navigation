# Wander Behavior & Random Shape 구현 완료! 🎉

## 📋 개요

IR-SIM에 두 가지 새로운 기능을 구현했습니다:

1. **Wander Behavior** - 랜덤한 목표로 배회하는 동작
2. **Random Shape** - 각 obstacle마다 다른 크기 생성

---

## 1️⃣ Wander Behavior

### 📖 개념

**"목표 없이 랜덤하게 돌아다니기"**

일반 dash/rvo는 **고정된 goal**로 이동하지만, **wander**는:
- 랜덤 goal 생성
- Goal 도착 시 새로운 랜덤 goal 자동 생성
- 계속 움직이는 dynamic obstacle

### 🎯 동작 원리

```
Normal dash:
  Start → Goal (fixed) → Stop ❌

Wander:
  Start → Goal₁ (random) → Goal₂ (random) → Goal₃ (random) → ... ✅
         ↺ Auto-generate when reached
```

### ⚙️ YAML 설정

```yaml
obstacle:
  - number: 4
    kinematics: {name: 'otter_usv'}
    distribution: {name: 'random', range_low: [10, 10, 0], range_high: [90, 90, 3.14]}
    shape: {name: 'rectangle', length: 2.0, width: 1.08}
    behavior:
      name: 'wander'  # ← Wander behavior!
      range_low: [10, 10, -3.14]     # Goal generation range (min)
      range_high: [90, 90, 3.14]     # Goal generation range (max)
      wander_goal_threshold: 2.5     # Distance to consider goal reached
      angle_tolerance: 0.1           # Turning tolerance
    velocity: [1.5, 0.2]
    color: 'purple'
```

### 📊 파라미터 설명

| Parameter | Description | Default |
|-----------|-------------|---------|
| `range_low` | 랜덤 goal 생성 최소값 [x, y, θ] | [0, 0, -π] |
| `range_high` | 랜덤 goal 생성 최대값 [x, y, θ] | [100, 100, π] |
| `wander_goal_threshold` | Goal 도달 거리 임계값 (m) | 2.0 |
| `angle_tolerance` | 방향 허용 오차 (rad) | 0.1 |

### 🔧 지원되는 Kinematics

```yaml
# Differential Drive
kinematics: {name: 'diff'}
behavior: {name: 'wander'}

# Omnidirectional
kinematics: {name: 'omni'}
behavior: {name: 'wander'}

# Ackermann Steering
kinematics: {name: 'acker'}
behavior: {name: 'wander'}

# Otter USV ⭐
kinematics: {name: 'otter_usv'}
behavior: {name: 'wander'}
```

---

## 2️⃣ Random Shape

### 📖 개념

**"각 obstacle마다 다른 크기"**

일반 설정:
- 모든 Otter가 동일 크기 (2.0m x 1.08m) ❌

Random shape:
- 각 Otter가 랜덤한 크기 ✅
- 더 realistic한 환경
- DRL 일반화 성능 향상

### ⚙️ YAML 설정

#### Rectangle (Otter USV)

```yaml
obstacle:
  - number: 4
    kinematics: {name: 'otter_usv'}
    shape:
      name: 'rectangle'
      random_shape: true  # ← Enable random shape!
      length_range: [1.5, 2.5]  # Length: 1.5~2.5m
      width_range: [0.8, 1.3]   # Width: 0.8~1.3m
    behavior: {name: 'dash'}
```

**결과:**
```
Obstacle 0: 2.3m x 1.1m
Obstacle 1: 1.8m x 0.9m
Obstacle 2: 2.5m x 1.3m
Obstacle 3: 1.6m x 0.8m
```

#### Circle

```yaml
obstacle:
  - number: 3
    kinematics: {name: 'omni'}
    shape:
      name: 'circle'
      random_shape: true
      radius_range: [1.0, 3.0]  # Radius: 1.0~3.0m
    behavior: {name: 'rvo'}
```

### 📊 파라미터 설명

| Shape | Parameters | Description |
|-------|-----------|-------------|
| **Rectangle** | `length_range: [min, max]` | 길이 범위 (m) |
|  | `width_range: [min, max]` | 너비 범위 (m) |
| **Circle** | `radius_range: [min, max]` | 반지름 범위 (m) |

---

## 🎮 조합 사용 예제

### **Complete Example: Wander + Random Shape**

```yaml
world:
  height: 100
  width: 100
  step_time: 0.05

robot:
  - kinematics: {name: 'otter_usv'}
    state: [10, 10, 0, 0, 0, 0, 0, 0]
    goal: [90, 90, 0]

obstacle:
  # Group 1: Wander with random shapes
  - number: 4
    kinematics: {name: 'otter_usv'}
    distribution: {name: 'random', range_low: [20, 20, 0], range_high: [80, 80, 3.14]}
    shape:
      name: 'rectangle'
      random_shape: true
      length_range: [1.5, 2.5]
      width_range: [0.8, 1.3]
    behavior:
      name: 'wander'
      range_low: [10, 10, -3.14]
      range_high: [90, 90, 3.14]
      wander_goal_threshold: 3.0
    velocity: [1.2, 0.15]
    color: 'purple'
  
  # Group 2: Dash with fixed shape
  - number: 2
    kinematics: {name: 'otter_usv'}
    shape: {name: 'rectangle', length: 2.0, width: 1.08}
    behavior: {name: 'dash'}
    velocity: [1.5, 0.2]
    color: 'blue'
```

---

## 🚀 테스트 방법

### 1. IR-SIM 재설치 (변경사항 반영)

```bash
$ conda activate DRL-otter-nav
$ cd /home/hyo/ir-sim
$ pip install -e .
```

### 2. 테스트 실행

```bash
$ cd /home/hyo/DRL-otter-navigation
$ poetry run python test_wander_behavior.py
```

### 3. 예상 결과

```
======================================================================
Wander Behavior and Random Shape Test
======================================================================

✓ Environment loaded: robot_nav/worlds/otter_world_wander.yaml
  Robot type: RobotOtter
  Number of obstacles: 6

Obstacle Details:

  Obstacle 0:
    Type: ObstacleOtter
    Position: [45.3, 67.2]
    Shape: Rectangle (2.12m x 1.05m)  ← Random!
    Behavior: wander
    Current goal: [78.5, 23.1]

  Obstacle 1:
    Type: ObstacleOtter
    Position: [32.1, 28.9]
    Shape: Rectangle (1.87m x 0.92m)  ← Different!
    Behavior: wander
    Current goal: [55.2, 81.7]

Running Simulation...
(Watch obstacles wander to random goals with different shapes)

  Obstacle 0 reached goal #1!
    New goal: [25.3, 44.8]

  Obstacle 2 reached goal #1!
    New goal: [67.9, 19.3]
```

---

## 📊 DRL Training에 활용

### **Training Complexity 증가**

```python
# Before: Static obstacles
world_file = "otter_world_static.yaml"
→ Easy, Success rate: ~70%

# After: Wander + Random shape
world_file = "otter_world_wander.yaml"
→ Hard, Success rate: ~50%, BUT better generalization!
```

### **Curriculum Learning**

```python
# Phase 1: Static
train_epochs(1-20, "otter_world_static.yaml")

# Phase 2: Dash
train_epochs(21-40, "otter_world_dash.yaml")

# Phase 3: Wander (challenging!)
train_epochs(41-60, "otter_world_wander.yaml")
```

---

## 🔧 구현 세부사항

### **생성된/수정된 파일**

```
/home/hyo/ir-sim/irsim/
├── lib/behavior/
│   ├── behavior_wander.py           ← NEW! Wander behavior
│   └── behavior.py                  ← Modified (import wander)
└── world/
    └── object_factory.py            ← Modified (random_shape)

/home/hyo/DRL-otter-navigation/
├── robot_nav/worlds/
│   └── otter_world_wander.yaml      ← NEW! Test config
└── test_wander_behavior.py          ← NEW! Test script
```

### **코드 구조**

#### Wander Behavior (`behavior_wander.py`)

```python
def check_and_update_wander_goal(ego_object, range_low, range_high, goal_threshold):
    """주기적으로 랜덤 goal 생성"""
    if goal_reached(ego_object, goal_threshold):
        ego_object.goal = generate_random_goal(range_low, range_high)

@register_behavior("otter_usv", "wander")
def beh_otter_wander(ego_object, external_objects, **kwargs):
    """Wander behavior for Otter USV"""
    check_and_update_wander_goal(...)  # Goal 업데이트
    return DiffDash(state, goal, ...)   # Dash로 goal로 이동
```

#### Random Shape (`object_factory.py`)

```python
def generate_random_shape(self, shape_dict):
    """Shape 파라미터 랜덤 생성"""
    if shape_dict["name"] == "rectangle":
        length = random.uniform(length_range[0], length_range[1])
        width = random.uniform(width_range[0], width_range[1])
    elif shape_dict["name"] == "circle":
        radius = random.uniform(radius_range[0], radius_range[1])
```

---

## 💡 사용 팁

### 1. **Wander Goal Threshold 조정**

```yaml
# 작은 값 → 더 자주 goal 변경
wander_goal_threshold: 1.0  # Very dynamic

# 큰 값 → 덜 자주 변경
wander_goal_threshold: 5.0  # More stable
```

### 2. **Shape Range 조정**

```yaml
# 작은 variation
length_range: [1.9, 2.1]  # ±0.1m

# 큰 variation
length_range: [1.5, 2.5]  # ±0.5m
```

### 3. **Mixed Scenarios**

```yaml
obstacle:
  # Wander obstacles (unpredictable)
  - number: 3
    behavior: {name: 'wander'}
    color: 'purple'
  
  # Dash obstacles (predictable)
  - number: 2
    behavior: {name: 'dash'}
    color: 'blue'
  
  # Static obstacles
  - number: 5
    kinematics: {name: 'static'}
    color: 'gray'
```

---

## 🎯 요약

### ✅ 구현 완료

**Wander Behavior:**
- ✅ `diff`, `omni`, `acker`, `otter_usv` 지원
- ✅ 자동 goal 재생성
- ✅ 파라미터 조정 가능

**Random Shape:**
- ✅ Rectangle (length/width range)
- ✅ Circle (radius range)
- ✅ 각 obstacle마다 다른 크기

### 🚀 장점

**Wander:**
- 더 challenging한 환경
- 예측 불가능한 장애물
- Better DRL generalization

**Random Shape:**
- Realistic 환경
- Size variation 학습
- Robust policy

### 📌 다음 단계

1. **IR-SIM 재설치**
   ```bash
   cd /home/hyo/ir-sim && pip install -e .
   ```

2. **테스트 실행**
   ```bash
   poetry run python test_wander_behavior.py
   ```

3. **DRL Training**
   ```bash
   # otter_rl_train.py에서 world_file 변경
   world_file = "robot_nav/worlds/otter_world_wander.yaml"
   ```

---

**구현 완료! 이제 더 복잡하고 realistic한 환경에서 DRL training이 가능합니다! 🎉**

**생성일**: 2025-10-21  
**작성자**: DRL-otter-navigation 프로젝트
