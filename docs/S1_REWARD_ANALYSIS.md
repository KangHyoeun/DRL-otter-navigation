## 📊 S1 Scenario Reward Function Analysis

### 🎯 **Current Reward Function**

```python
if goal:
    return 10000.0          # Goal bonus
elif collision:
    return -10000.0         # Collision penalty

# Step rewards:
progress_reward = (prev_distance - distance) × 100
velocity_reward = action[0] × 5
obstacle_penalty = -(5.0 - min_dist) × 10 if min_dist < 5m else 0
step_penalty = -1

total = progress + velocity + obstacle + step
```

---

### 📈 **Expected Reward for S1 (Straight Line)**

**Optimal trajectory (60초, 직진):**
```
Progress reward:  180m × 100 = 18,000
Goal bonus:                    10,000
Velocity bonus:   3.0 × 5 × 600 steps = 9,000
Step penalty:     -1 × 600 = -600
Obstacle penalty: 0 (no obstacles in s1)
─────────────────────────────────────
Total:            ~36,400 ✓
```

**Per-step breakdown:**
- Good step: +30 to +50 (progress + velocity - step)
- Bad step: -1 to -10 (no progress)

---

### ⚠️ **문제점 & 개선사항**

#### **1. Phase 1 Action Interval (현재: 1.0s)**
```python
action_dt = 1.0  # DRL action update every 1.0s
```

**문제:**
- 직진 시나리오에서 1.0초마다 action 업데이트는 너무 느림
- 미세한 heading 조정이 어려움

**권장:**
```python
action_dt = 0.5  # 0.5초 (5 physics steps)
# 또는
action_dt = 0.3  # 0.3초 (3 physics steps)
```

**이유:**
- Otter USV controller settling time: ~3초
- 하지만 직진은 complex maneuvering이 아니므로 더 빠른 업데이트 가능
- 0.5초면 충분한 settling + 빠른 학습

---

#### **2. Obstacle Penalty Threshold (현재: 5m)**
```python
if min_distance < 5.0:
    obstacle_penalty = -(5.0 - min_distance) × 10
```

**문제:**
- S1 환경의 벽까지 거리: **50m (좌우), 90-100m (전후)**
- 5m threshold는 **너무 작아서 벽 회피 학습 불가**

**권장 (Phase별 조정):**

**Phase 1 (직진 학습):**
```python
# 벽 회피는 나중에 학습
if min_distance < 3.0:  # 매우 가까울 때만
    obstacle_penalty = -(3.0 - min_distance) × 20
```

**Phase 2 (장애물 회피 학습 시):**
```python
# 더 먼 거리에서 회피 시작
if min_distance < 15.0:
    obstacle_penalty = -(15.0 - min_distance) × 5
elif min_distance < 5.0:
    obstacle_penalty = -(5.0 - min_distance) × 20  # 매우 가까우면 강한 penalty
```

---

#### **3. Progress Reward Scale**
```python
progress_reward = (prev_distance - distance) × 100
```

**분석:**
- 1m 진행 → +100 reward
- 0.1m 진행 → +10 reward (typical per step at 3m/s)

**판단:** ✅ **적절함!**
- velocity reward (15) + progress (10) - step (-1) = +24 per step
- 누적하면 goal bonus 포함 ~36,000

**단, 고려사항:**
- 너무 빠르게 학습하면 exploration 부족 가능
- 초반에는 reward 스케일 낮추는 것도 방법

---

#### **4. Step Penalty**
```python
step_penalty = -1.0
```

**분석:**
- 매 step마다 -1
- 빠르게 goal 도달하도록 유도

**판단:** ✅ **적절함!**
- Progress reward (10-30)가 충분히 크므로 -1은 미미
- 시간 효율성 강조

---

### 🔧 **권장 수정사항**

#### **수정 1: action_dt 줄이기**
```python
# otter_sim.py line 42
self.action_dt = 0.5  # 1.0 → 0.5초로 변경
```

#### **수정 2: obstacle threshold 조정**
```python
# otter_sim.py get_reward 함수
# 3. Obstacle penalty (WALL AVOIDANCE)
min_distance = min(laser_scan)
if min_distance < 3.0:  # 5.0 → 3.0으로 변경
    obstacle_penalty = -(3.0 - min_distance) × 20  # 10 → 20으로 강화
else:
    obstacle_penalty = 0.0
```

#### **수정 3 (선택): Progress reward smoothing**
```python
# Phase 1: 직진 학습용 (좀 더 보수적)
progress_reward = (prev_distance - distance) × 50  # 100 → 50

# Phase 2: 회피 학습 후 원복
progress_reward = (prev_distance - distance) × 100
```

---

### 📋 **최종 권장 Reward Function (S1용)**

```python
@staticmethod
def get_reward(goal, collision, action, laser_scan, distance, cos, prev_distance):
    # Terminal rewards
    if goal:
        return 10000.0
    elif collision:
        return -10000.0
    
    # Step rewards
    # 1. Progress (MAIN)
    progress_reward = 0.0
    if prev_distance is not None:
        progress = prev_distance - distance
        progress_reward = progress * 100.0  # Keep original scale
    
    # 2. Velocity bonus
    velocity_reward = action[0] * 5.0
    
    # 3. Obstacle penalty (벽 회피 - threshold 낮춤)
    min_distance = min(laser_scan)
    if min_distance < 3.0:  # 5.0 → 3.0
        obstacle_penalty = -(3.0 - min_distance) * 20.0  # 10 → 20
    else:
        obstacle_penalty = 0.0
    
    # 4. Heading alignment bonus (직진 강화)
    heading_bonus = cos * 2.0  # 정렬되면 +2, 반대면 -2
    
    # 5. Step penalty
    step_penalty = -1.0
    
    # Total
    total_reward = (progress_reward + velocity_reward + 
                   obstacle_penalty + heading_bonus + step_penalty)
    
    return total_reward
```

**변경사항:**
1. ✅ Obstacle threshold: 5m → 3m (벽이 멀어서)
2. ✅ Obstacle penalty weight: 10 → 20 (더 강하게)
3. ✅ **Heading bonus 추가**: 직진 유도 (+2 when aligned)

---

### 🎯 **Expected Performance (Revised)**

**Optimal trajectory (60초):**
```
Progress:  18,000
Goal:      10,000
Velocity:   9,000
Heading:    1,200  (cos ≈ 1.0 most of time)
Step:        -600
Obstacle:      0   (no close walls)
─────────────────
Total:    ~37,600 ✓
```

**Per-step average:** +62 (good learning signal!)

---

### 🚀 **Action Plan**

1. **Phase 1 (지금):** 직진 학습
   - action_dt = 0.5s
   - Obstacle threshold = 3m
   - Heading bonus 추가

2. **Phase 2 (나중):** 회피 학습
   - action_dt = 1.0s (복잡한 maneuvering)
   - Obstacle threshold = 10-15m
   - Multiple obstacles 추가

3. **Phase 3 (최종):** COLREGs 학습
   - 동적 장애물
   - 조우 상황별 reward shaping

---

수정할까? 아니면 일단 현재 설정으로 학습 시작해볼까?
