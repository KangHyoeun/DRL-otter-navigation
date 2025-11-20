# DRL-Otter-Navigation: Autonomous COLREGs-Compliant Navigation

## 🎯 Project Overview

This project implements a Deep Reinforcement Learning (DRL) framework to train an Otter Unmanned Surface Vehicle (USV) for autonomous, COLREGs-compliant navigation. The agent is trained using a curriculum learning approach, progressively mastering more complex multi-vessel encounter scenarios.

The framework integrates the `ir-sim` 2D simulator for LiDAR sensing and visualization with the `PythonVehicleSimulator` for realistic 6-DOF USV dynamics.

## ✨ Key Features

-   **Curriculum Learning:** A multi-phase training structure that starts with simple goal-reaching and progressively adds 1, 2, and 3 dynamic target ships.
-   **PPO Agent:** Utilizes a Proximal Policy Optimization (PPO) agent, a state-of-the-art DRL algorithm.
-   **CNN-based Perception:** A Convolutional Neural Network (CNN) processes 360-degree LiDAR data to extract key environmental features.
-   **Domain Randomization:** At each training phase, the agent is exposed to a variety of randomized encounter scenarios (head-on, crossing, overtaking) to promote a robust and generalizable policy.
-   **Modular Reward System:** Leverages the `colregs-core` library for sophisticated, customizable reward calculations based on maritime best practices.

---

## 🚀 Getting Started

### 1. Environment Setup

It is recommended to use `conda` for managing the environment and `poetry` for installing Python dependencies.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd DRL-otter-navigation

# 2. Activate the conda environment
# Ensure you have the correct environment activated
conda activate DRL-otter-nav

# 3. Install dependencies using Poetry
# This will install all packages listed in pyproject.toml
poetry install
```

*Note: This project also depends on the `colregs-core` library, which should be located at `/home/hyo/colregs-core/` and installed in your environment.*

### 2. Project Structure

```
DRL-otter-navigation/
├── robot_nav/
│   ├── models/PPO/
│   │   └── CNNPPO.py             # Core PPO agent with CNN
│   ├── SIM_ENV/
│   │   └── otter_sim.py          # Otter USV simulation environment
│   ├── worlds/imazu_scenario/
│   │   └── imazu_case_*.yaml     # World files for each curriculum phase
│   ├── otter_rl_train_CNNPPO_imazu_00_scratch.py   # Phase 1 Training
│   ├── otter_rl_train_CNNPPO_imazu_01_phase2.py    # Phase 2 Training
│   ├── otter_rl_train_CNNPPO_imazu_02_phase3.py    # Phase 3 Training
│   └── otter_rl_train_CNNPPO_imazu_03_phase4.py    # Phase 4 Training
└── README.md
```

---

## 🧠 Training the Agent

The agent is trained using a 4-phase curriculum. You must run the training scripts sequentially, as each phase loads the best model from the previous one.

### Training Commands

Run these commands from the project root (`/home/hyo/DRL-otter-navigation`).

**Phase 1: Goal Reaching (0 Target Ships)**
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_00_scratch.py
```

**Phase 2: 1 Target Ship**
*Loads the best model from Phase 1.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_01_phase2.py
```

**Phase 3: 2 Target Ships**
*Loads the best model from Phase 2.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_02_phase3.py
```

**Phase 4: 3 Target Ships**
*Loads the best model from Phase 3.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_03_phase4.py
```

### Monitoring Training

You can monitor the training progress, including rewards, loss functions, and other metrics, using TensorBoard.

```bash
# In a new terminal, from the project root
tensorboard --logdir runs
```
Navigate to `http://localhost:6006` in your web browser.

---

## 🛠️ Technical Details

-   **DRL Algorithm**: Proximal Policy Optimization (PPO)
-   **State Space (`370-dim`):**
    -   LiDAR Scans: 360 points
    -   Velocity Info: 2 values
    -   Path Error Info: 2 values
    -   Goal Info: 3 values
    -   Propeller Info: 2 values
    -   Max Collision Risk: 1 value
-   **Action Space (`2-dim`):**
    -   `u_ref`: Surge velocity reference
    -   `r_ref`: Yaw rate reference
-   **Dependencies:** `ir-sim`, `PythonVehicleSimulator`, `pytorch`, `colregs-core`