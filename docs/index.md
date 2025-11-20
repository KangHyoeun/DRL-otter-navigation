# Welcome to DRL-Otter-Navigation

This project implements a Deep Reinforcement Learning (DRL) framework to train an Otter Unmanned Surface Vehicle (USV) for autonomous, COLREGs-compliant navigation. The agent is trained using a curriculum learning approach, progressively mastering more complex multi-vessel encounter scenarios.

The framework integrates the `ir-sim` 2D simulator for LiDAR sensing and visualization with the `PythonVehicleSimulator` for realistic 6-DOF USV dynamics.

## Key Features

-   **Curriculum Learning:** A multi-phase training structure that starts with simple goal-reaching and progressively adds 1, 2, and 3 dynamic target ships.
-   **PPO Agent:** Utilizes a Proximal Policy Optimization (PPO) agent, a state-of-the-art DRL algorithm.
-   **CNN-based Perception:** A Convolutional Neural Network (CNN) processes 360-degree LiDAR data to extract key environmental features.
-   **Domain Randomization:** At each training phase, the agent is exposed to a variety of randomized encounter scenarios (head-on, crossing, overtaking) to promote a robust and generalizable policy.
-   **Modular Reward System:** Leverages the `colregs-core` library for sophisticated, customizable reward calculations based on maritime best practices.

## Getting Started

To get started, head over to the **[Installation](getting-started/installation.md)** guide.
