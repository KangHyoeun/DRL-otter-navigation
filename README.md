# Otter USV + IR-SIM + DRL Navigation

## 🎯 프로젝트 개요

Otter USV (Python Vehicle Simulator)와 IR-SIM을 통합하여 Deep Reinforcement Learning 기반 자율 내비게이션을 구현하는 프로젝트입니다.

**현재 상태**: ✅ **Phase 1 완료** - Wrapper 방식으로 Otter USV와 IR-SIM 통합 완료

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# 가상환경 활성화
conda activate DRL-otter-usv

# 프로젝트 디렉토리로 이동
cd /home/hyo/DRL-otter-navigation
```

### 2. 환경 테스트

```bash
# Simple test (자동 궤적)
python3 test_otter_simple.py
```

### 3. DRL Training

```bash
# TD3 알고리즘으로 훈련 시작
python3 robot_nav/otter_rl_train.py

# TensorBoard 모니터링 (새 터미널)
tensorboard --logdir runs
```

### 4. 훈련된 모델 테스트

```bash
poetry run python3 robot_nav/otter_rl_test.py
```

---

## 📁 파일 구조

```
DRL-otter-navigation/
├── robot_nav/
│   ├── SIM_ENV/
│   │   ├── __init__.py
│   │   └── otter_sim.py              # Otter USV Environment Wrapper ⭐
│   ├── worlds/
│   │   └── otter_world.yaml          # Environment Configuration
│   ├── otter_rl_train.py             # DRL Training Script (TD3) ⭐
│   ├── otter_rl_test.py              # Model Test Script ⭐
│   └── otter_utils.py                # Utilities
├── test_otter_simple.py              # Simple Test
├── test_otter_keyboard.py            # Keyboard Control (optional)
└── README_OTTER.md                   # Detailed Documentation
```

---

## 🎨 아키텍처

### Phase 1: Wrapper 방식 (현재)

```
DRL Agent (TD3)
    ↓ [u_ref, r_ref]
OtterSIM Wrapper
    ↓
Otter USV Dynamics (6-DOF)
    ↓ [x, y, ψ]
IR-SIM (Visualization & Collision Detection)
```

### Phase 2: 완전 통합 (향후)

```
DRL Agent
    ↓
IR-SIM (with Otter Kinematics)
    ↓
Native Otter USV Dynamics
```

---

## 🔧 기술 스택

### Otter USV Dynamics
- **Source**: Python Vehicle Simulator (Fossen 2021)
- **DOF**: 6-DOF (surface vessel simplified to 3-DOF for navigation)
- **Control**: Velocity controller with feedforward + PI feedback
- **Input**: `[u_ref, r_ref]` (surge velocity, yaw rate)

### IR-SIM
- **Purpose**: Visualization, collision detection, LiDAR simulation
- **Sensors**: 2D LiDAR (180 points, 7m range)
- **Features**: Dynamic obstacles, RVO behavior

### DRL Algorithm
- **Algorithm**: TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- **Framework**: PyTorch
- **State**: 184-dim (180 LiDAR + 4 goal info)
- **Action**: 2-dim [u_ref ∈ [0, 2.0], r_ref ∈ [-0.5, 0.5]]

---

## 📊 성능 지표

### Otter Velocity Controller
```python
# Reference Model: 2nd-order
wn_d = 0.6 rad/s    # Natural frequency
zeta_d = 1.0        # Damping ratio

# Controller: PI with pole placement
wn = 2.5 rad/s      # Bandwidth
```

**Performance:**
- Rise time: ~2-3초
- Settling time: ~5초
- Steady-state error: < 2%

### Expected Training Results
| Epoch | Avg Reward | Success Rate | Notes |
|-------|-----------|--------------|-------|
| 1-10  | -50 ~ 0   | 0-10%        | Exploration |
| 11-30 | 0 ~ 30    | 10-30%       | Basic navigation |
| 31-60 | 30 ~ 60   | 30-60%       | Obstacle avoidance |
| 61-100| 60 ~ 90   | 60-80%       | Fine-tuning |

---

## 🛠️ 설치

### Prerequisites

```bash
# Python Vehicle Simulator
pip install -e /home/hyo/PythonVehicleSimulator

# IR-SIM
pip install ir-sim
# 또는
cd /home/hyo/ir-sim
pip install -e .

# PyTorch (CUDA 지원)
pip install torch torchvision torchaudio

# TensorBoard
pip install tensorboard
```

---

## 📖 사용법

### OtterSIM API

```python
from robot_nav.SIM_ENV.otter_sim import OtterSIM
import numpy as np

# 환경 초기화
env = OtterSIM(world_file="otter_world.yaml", disable_plotting=False)

# 환경 리셋
scan, distance, cos, sin, collision, goal, action, reward = env.reset()

# 스텝 실행
scan, distance, cos, sin, collision, goal, action, reward = env.step(
    u_ref=1.5,   # surge velocity (m/s)
    r_ref=0.2    # yaw rate (rad/s)
)

# 환경 종료
env.close()
```

### Training Configuration

```yaml
# otter_world.yaml
world:
  step_time: 0.05      # 20Hz simulation
  collision_mode: 'reactive'

robot:
  shape: {name: 'circle', radius: 0.5}
  vel_min: [-1.0, -0.5]
  vel_max: [2.0, 0.5]
  sensors:
    - type: 'lidar2d'
      range_max: 7
      number: 180
```

---

## 🐛 문제 해결

### 1. ImportError: python_vehicle_simulator
```bash
pip install -e /home/hyo/PythonVehicleSimulator
```

### 2. CUDA out of memory
```python
# otter_rl_train.py에서
batch_size = 128  # 256 → 128로 감소
```

### 3. Training이 수렴하지 않음
```python
# Hyperparameter 조정
learning_rate = 1e-4  # 더 작게
exploration_noise = 0.1  # 더 작게
```

---

## 📈 Roadmap

### ✅ Phase 1: Wrapper 구현 (완료)
- [x] OtterSIM wrapper 클래스
- [x] IR-SIM 통합
- [x] DRL training script
- [x] Test scripts

### ⏳ Phase 2: IR-SIM 완전 통합 (진행 예정)
- [ ] Custom kinematics 추가 (`otter_usv_kinematics`)
- [ ] Robot class 생성 (`RobotOtter`)
- [ ] YAML configuration
- [ ] IR-SIM에 Pull Request

### 🔮 Phase 3: 고도화 (미래)
- [ ] Real USV deployment
- [ ] Multi-agent navigation
- [ ] Dynamic obstacles
- [ ] Hardware-in-the-loop simulation

---

## 🎓 참고 자료

### Papers
- Fossen, T. I. (2021). *Handbook of Marine Craft Hydrodynamics and Motion Control*. 2nd Edition, Wiley.
- Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML 2018.

### Repositories
- **Python Vehicle Simulator**: https://github.com/cybergalactic/PythonVehicleSimulator
- **IR-SIM**: https://github.com/hanruihua/ir-sim
- **DRL-robot-navigation-IR-SIM**: https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM

---

## 📝 License

This project follows the licenses of its dependencies:
- Python Vehicle Simulator: MIT License
- IR-SIM: MIT License

---

## 👥 Contact

For questions or collaboration:
- 프로젝트 관련 문의: GitHub Issues
- IR-SIM 관련: https://github.com/hanruihua/ir-sim
- Python Vehicle Simulator 관련: https://github.com/cybergalactic/PythonVehicleSimulator

---

## 🙏 Acknowledgments

- Prof. Thor I. Fossen for Python Vehicle Simulator
- Dr. Ruihua Han for IR-SIM
- Reinis Cimurs for DRL-robot-navigation framework

---

**Last Updated**: 2025-10-17
**Status**: Phase 1 Complete ✅
