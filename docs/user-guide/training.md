# Training the Agent

The agent is trained using a 4-phase curriculum. This approach, known as **Curriculum Learning**, starts with a simple task and gradually increases the difficulty. This allows the agent to build foundational skills before tackling more complex scenarios.

You must run the training scripts sequentially, as each phase loads the best model from the previous one to continue learning (fine-tuning).

## The Curriculum

-   **Phase 1: Goal Reaching**
    -   **Objective:** Navigate to a goal in an empty environment.
    -   **Scenarios:** `imazu_case_00.yaml`
    -   **Target Ships:** 0

-   **Phase 2: Simple Collision Avoidance**
    -   **Objective:** Navigate to a goal while avoiding a single target ship.
    -   **Scenarios:** Randomly selected from `imazu_case_01.yaml` to `imazu_case_04.yaml`.
    -   **Target Ships:** 1

-   **Phase 3: Multi-Vessel Encounters**
    -   **Objective:** Navigate to a goal while avoiding two simultaneous target ships.
    -   **Scenarios:** Randomly selected from `imazu_case_05.yaml` to `imazu_case_11.yaml`.
    -   **Target Ships:** 2

-   **Phase 4: Complex Multi-Vessel Encounters**
    -   **Objective:** Navigate to a goal in a complex environment with three simultaneous target ships.
    -   **Scenarios:** Randomly selected from `imazu_case_12.yaml` to `imazu_case_22.yaml`.
    -   **Target Ships:** 3

## How to Run Training

Run these commands from the project root directory (`/home/hyo/DRL-otter-navigation`). Ensure your `conda` environment is activated.

### Phase 1: Goal Reaching (from Scratch)
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_00_scratch.py
```

### Phase 2: 1 Target Ship
*This script will automatically load the best model saved from Phase 1.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_01_phase2.py
```

### Phase 3: 2 Target Ships
*This script will automatically load the best model saved from Phase 2.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_02_phase3.py
```

### Phase 4: 3 Target Ships
*This script will automatically load the best model saved from Phase 3.*
```bash
poetry run python3 robot_nav/otter_rl_train_CNNPPO_imazu_03_phase4.py
```
