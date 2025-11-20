# Installation

This guide will walk you through setting up the project environment. It is recommended to use `conda` for managing the environment and `poetry` for installing Python dependencies.

## 1. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone <your-repo-url>
cd DRL-otter-navigation
```

## 2. Set Up the Conda Environment

Activate the conda environment created for this project. The recommended environment name is `DRL-otter-nav`.

```bash
conda activate DRL-otter-nav
```

If you do not have the environment set up, you may need to create it using an appropriate environment file or a list of required packages.

## 3. Install Dependencies

This project uses `poetry` to manage its dependencies. `poetry` ensures that you have the exact versions of the packages required for the project to run correctly.

From the project root directory (`/home/hyo/DRL-otter-navigation`), run:

```bash
poetry install
```

This command reads the `pyproject.toml` file, resolves the dependencies, and installs them into your activated conda environment.

## 4. Install External Dependencies

This project has local dependencies that must be installed separately. Ensure the following projects are located at the specified paths and installed in your environment.

- **colregs-core:** A library for reward and collision risk calculation.
  - Expected Path: `/home/hyo/colregs-core/`
- **PythonVehicleSimulator:** Provides the 6-DOF USV dynamics.
  - Expected Path: `/home/hyo/PythonVehicleSimulator/`
- **ir-sim:** The 2D simulator for visualization and LiDAR.
  - Expected Path: `/home/hyo/ir-sim/`

If they are not installed, you may need to install them in editable mode, for example:

```bash
pip install -e /home/hyo/colregs-core
pip install -e /home/hyo/PythonVehicleSimulator
pip install -e /home/hyo/ir-sim
```
