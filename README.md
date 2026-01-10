# Wheel Pupper V3 Training (Colab)

This repository hosts the Google Colab notebook used for training the Reinforcement Learning (RL) policy for the **Wheeled Pupper V3** robot. This project builds upon the original Pupper V3 (legged) codebase, adapting it for a hybrid wheeled-legged locomotion platform using MuJoCo MJX for accelerated simulation on GPU.

## Key Contributions: Legged to Wheeled Conversion

This project represents a significant modification of the original quadrupedal training pipeline to support wheeled locomotion. Key engineering efforts include:

1.  **Joint Limit Unlocking**:
    *   The standard Pupper V3 has finite joint limits for all actuators.
    *   For this wheeled variant, the distal joints (ankles) were replaced with wheels. I modified the simulation configuration to set the joint limits for these specific indices (Wheel_FR, Wheel_FL, Wheel_BR, Wheel_BL) to infinity (`float('inf')`), enabling **continuous rotation** required for rolling.

2.  **Model Adaptation**:
    *   Integrated a custom MuJoCo XML model (`Wheel_pupper.xml`) that defines the wheeled morphology.
    *   Configured the simulation environment to correctly load this modified XML, ensuring that collision geometries and actuator parameters match the physical wheeled robot.

3.  **Control Logic Adjustments**:
    *   Adapted the actuator control parameters in `Colab_wheeled.ipynb` to support the different dynamic requirements of driving vs. stepping.
    *   Ensured that the RL agent can learn hybrid policies that may utilize both leg articulation and wheel rotation.

## Getting Started

The core of this project is a self-contained Google Colab notebook. You do not need to install dependencies locally.

[**Open in Google Colab**](https://colab.research.google.com/github/TundTT/colab_Wheel_pupperv3/blob/main/Colab_wheeled.ipynb)

## Prerequisites

*   **Google Account**: Required to run Colab notebooks. A **Pro** subscription is recommended to access better GPUs (A100/L4) for faster training.
*   **Weights & Biases Account**: Required for logging training metrics. You will need your API key.

## Repositories

This project involves multiple component repositories:

*   **Training Notebook** (This Repo): [colab_Wheel_pupperv3](https://github.com/TundTT/colab_Wheel_pupperv3)
*   **Core Codebase**: [mjx_Wheels_pupperv3](https://github.com/TundTT/mjx_Wheels_pupperv3) - Contains the underlying python logic, finding the `pupperv3_mjx` package.
*   **Robot Description**: [description_Wheels_pupperv3](https://github.com/TundTT/description_Wheels_pupperv3) - Contains the MuJoCo XML models (`Wheel_pupper.xml`) and meshes.

## Usage Instructions

1.  **Select GPU Runtime**:
    *   In Colab, go to **Runtime** -> **Change runtime type**.
    *   Select **T4 GPU** (Standard users) or **A100/L4 GPU** (Pro users).

2.  **Login to Weights & Biases**:
    *   Run the setup cells.
    *   When prompted, enter your W&B API key to enable experiment tracking.

3.  **Configure Training**:
    *   The notebook includes a "Training Config" section where you can adjust parameters like:
        *   `num_timesteps`: Total training steps (default is 200M, increase for better performance).
        *   `learning_rate`: Adjust based on training stability.
        *   `n_obstacles`: Add terrain complexity to the environment.

4.  **Train**:
    *   Run the "Training" cells to start the PPO learning process.
    *   Monitor progress via the printed logs or your W&B dashboard.

5.  **Visualize**:
    *   The notebook includes cells to visualize the trained policy behavior directly in the browser.

## Troubleshooting

*   **Interrupting Training**: If you cannot stop the training cell, go to **Runtime** -> **Restart session** to clear the state.
*   **CUDA/JAX Errors**: Ensure you are running on a GPU instance. The notebook attempts to automatically resolve dependency conflicts between JAX, Orbax, and CUDA.
