# Project Structure

This document provides an overview of the key directories and files in the `DRL-otter-navigation` project.

```
DRL-otter-navigation/
│
├── docs/                     # MkDocs documentation files
│
├── robot_nav/
│   │
│   ├── models/PPO/
│   │   └── CNNPPO.py         # Core PPO agent with CNN architecture
│   │
│   ├── SIM_ENV/
│   │   └── otter_sim.py      # Otter USV simulation environment wrapper
│   │
│   ├── worlds/imazu_scenario/
│   │   └── imazu_case_*.yaml # World files for each curriculum phase
│   │
│   ├── otter_rl_train_CNNPPO_imazu_00_scratch.py   # Phase 1 Training Script
│   ├── otter_rl_train_CNNPPO_imazu_01_phase2.py    # Phase 2 Training Script
│   ├── otter_rl_train_CNNPPO_imazu_02_phase3.py    # Phase 3 Training Script
│   └── otter_rl_train_CNNPPO_imazu_03_phase4.py    # Phase 4 Training Script
│
├── runs/                       # Directory for TensorBoard logs
│
├── mkdocs.yml                  # MkDocs configuration file
├── pyproject.toml              # Project metadata and dependencies for Poetry
└── README.md                   # Project overview and quick start guide
```

## Key Components

-   **`robot_nav/models/PPO/CNNPPO.py`**: This is the heart of the agent. It contains the implementation of the Proximal Policy Optimization (PPO) algorithm, the actor and critic neural networks, and the logic for processing LiDAR data with a Convolutional Neural Network (CNN).

-   **`robot_nav/SIM_ENV/otter_sim.py`**: This file acts as a wrapper, creating a unified environment that combines the `ir-sim` simulator (for visualization and sensing) with the `PythonVehicleSimulator` (for realistic USV dynamics). It exposes a standard Gym-like API (`step`, `reset`).

-   **`robot_nav/worlds/imazu_scenario/`**: This directory contains the YAML configuration files for all training scenarios. Each file defines the environment, the starting positions of the own ship and target ships, and their goals, forming the basis of the curriculum.

-   **`robot_nav/otter_rl_train_*.py`**: These are the executable scripts for running each phase of the training curriculum. Each script is configured to load the correct scenarios and, for phases 2-4, the best model from the preceding phase.
