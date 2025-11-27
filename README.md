# DRL Otter Navigation 🦦🚢

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep Reinforcement Learning for Autonomous Collision Avoidance of Unmanned Surface Vehicles (USVs).**

This project implements a robust DRL-based navigation system for the **Otter USV**, capable of avoiding dynamic obstacles while adhering to **COLREGs (International Regulations for Preventing Collisions at Sea)**. It supports Multi-Modal inputs (Lidar/Grid Map + State Vector) and state-of-the-art DRL algorithms like **PPO**, **SAC**, and **TD3**.

---

## 🌟 Key Features

*   **Multi-Modal DRL Agents:** Combines **Vector inputs** (velocity, heading, goal info) with **2D Grid Maps** (local perception) using a hybrid MLP-CNN architecture (`MLPCNN`).
*   **Supported Algorithms:**
    *   **PPO** (Proximal Policy Optimization) - On-Policy
    *   **SAC** (Soft Actor-Critic) - Off-Policy, Maximum Entropy
    *   **TD3** (Twin Delayed DDPG) - Off-Policy, Deterministic
*   **Curriculum Learning:** Structured training phases (Phase 1 $\rightarrow$ 4) to gradually increase scenario complexity.
*   **COLREGs Compliance:** Reward functions designed to encourage compliance with maritime traffic rules (Head-on, Crossing, Overtaking).
*   **Integrated Training Manager:** A single script (`train_manager.py`) to manage training, transfer learning, and hyperparameters.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DRL-otter-navigation.git
cd DRL-otter-navigation
```

### 2. Set up Environment

We recommend using **Conda** for environment management and **Poetry** for dependency management.

```bash
# Create and activate conda environment
conda create -n venv-name python=3.10
conda activate venv-name

# Install dependencies using Poetry
pip install poetry
poetry install
```

**Dependencies:**
*   `torch`, `numpy`, `matplotlib`
*   `colregs-core` (Local or submodule)
*   `ir-sim` (Simulation environment)

---

## 🚀 Usage Guide

### 1. Training (train_manager.py)

Use the `train_manager.py` script to start training. It handles algorithm selection, curriculum phases, and model saving/loading.

#### **Arguments:**
*   `--algo`: Algorithm to use (`ppo`, `sac`, `td3`).
*   `--phase`: Curriculum phase (1: Navigation, 2: One Obstacle, 3: Multiple Obstacles, 4: Complex).
*   `--scratch`: Train from scratch (ignore previous phase models).
*   `--load_model`: Resume training from the *current* phase's checkpoint.

#### **Examples:**

**Start Phase 1 (Basic Navigation) with PPO:**
```bash
poetry run python3 train_manager.py --algo ppo --phase 1
```

**Start Phase 2 (Collision Avoidance) using Phase 1 weights (Transfer Learning):**
```bash
poetry run python3 train_manager.py --algo sac --phase 2
```

**Start Phase 2 from Scratch (No Transfer):**
```bash
poetry run python3 train_manager.py --algo td3 --phase 2 --scratch
```

**Resume interrupted training (Phase 2):**
```bash
poetry run python3 train_manager.py --algo sac --phase 2 --load_model
```

### 2. Monitoring (TensorBoard)

Monitor training progress, reward curves, and losses in real-time.

```bash
tensorboard --logdir runs
```
Open your browser and go to `http://localhost:6006`.

### 3. Configuration

Hyperparameters for each algorithm and environment settings are managed in `configs/`.

*   `configs/default.yaml`: Common settings (steps, epochs, rewards).
*   `configs/ppo.yaml`: PPO-specific hyperparameters.
*   `configs/sac.yaml`: SAC-specific hyperparameters.
*   `configs/td3.yaml`: TD3-specific hyperparameters.

Example (`configs/sac.yaml`):
```yaml
batch_size: 256
learning_rate: 0.0003
gamma: 0.99
replay_buffer_capacity: 100000
```

---

## 📂 Project Structure

```
DRL-otter-navigation/
├── configs/                 # Hyperparameter configurations (YAML)
├── robot_nav/
│   ├── models/
│   │   ├── PPO/             # MLPCNNPPO
│   │   ├── SAC/             # MLPCNNSAC
│   │   └── TD3/             # MLPCNNTD3
│   ├── SIM_ENV/             # OtterSIM Environment Wrapper
│   └── worlds/              # Simulation Scenarios (YAML)
├── trainers/                # Training Loops
│   ├── on_policy.py         # Trainer for PPO
│   └── off_policy.py        # Trainer for SAC/TD3
├── train_manager.py         # Main Entry Point
└── README.md                # This file
```

---

## 🔗 Related Projects

*   [**colregs-core**](https://github.com/your-username/colregs-core): The core library for collision risk assessment and reward calculation used in this project.

---

## 📄 License

This project is licensed under the **MIT License**.

```