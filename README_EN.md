# IsaacSim-Hil-Serl

IsaacSim-Hil-Serl is a hybrid hardware-in-the-loop reinforcement learning (HIL-SERL) framework built on Isaac Sim. It aims to provide a unified training loop for real-world reinforcement learning (Real-World RL) across both high-fidelity simulation and real robots.

## Part 1: Project Setup

### 📁 Code Structure

The main code structure of the IsaacSim-Hil-Serl project is as follows:

| Directory | Description |
| :-------: | :---------: |
| `examples` | Scripts for policy training, demonstration data collection, and reward classifier training |
| `serl_launcher` | Core launch and training code for IsaacSim-HIL-SERL |
| `serl_launcher.agents` | RL agent implementations (e.g., SAC) |
| `serl_launcher.wrappers` | Wrappers and adapters for Gym environments |
| `serl_launcher.data` | Replay buffer and data storage modules |
| `serl_launcher.vision` | Vision-related models and utility functions |
| `robot_infra` | Infrastructure code for running real and simulated robots |
| `robot_infra.robot_servers` | Flask server for communicating with robots via ROS2 |
| `robot_infra.gym_env` | Gym environments for robots |
| `robot_infra.isaacsim_venvs` | Isaac Sim–based virtual environment configurations for robots |

### 💻 Runtime Environment

- **Ubuntu 22.04**
- **CUDA 12.8**
- **Python 3.11**

### 📦 Install Base Dependencies

```bash
git clone https://github.com/Incalos/IsaacSim-Hil-Serl
cd IsaacSim-Hil-Serl

# Install Foxglove, see https://foxglove.dev/download

# Install uv, see https://docs.astral.sh/uv/getting-started/installation/

# Create and activate Python environment
uv venv --python=3.11
source .venv/bin/activate
uv pip install --upgrade pip
sudo apt update && sudo apt install -y xvfb x11-utils
sudo apt install cmake build-essential

# Install PyTorch
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Sim 5.1 
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install main dependencies under dependencies/
cd dependencies/

# Install Isaac Lab under dependencies/
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab/
./isaaclab.sh --install

# Install LeIsaac under dependencies/
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac/
uv pip install -e source/leisaac

# Install LeRobot under dependencies/
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .

# Install cuRobo under dependencies/
git clone --branch v0.7.7 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
uv pip install -e . --no-build-isolation

# Install agentlace under dependencies/
git clone https://github.com/youliangtan/agentlace.git
cd agentlace/
uv pip install -e .
```

## Part 2: Real-World RL for the SO101-PickOranges Task

This part describes how to configure and train Real-World RL for the SO101 arm in the Isaac Sim environment. Note that in this context, “Real-World RL” is a **pseudo real-robot setting**: a high-fidelity Isaac Sim simulation stands in for the physical robot, allowing policy training and validation without directly operating real hardware.

### 🧩 Isaac Sim Assets Preparation

We take the open-source SO101-PickOranges task as an example. You first need to [download the USD assets](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0) and set up the scene.

Unzip the downloaded archive and place the assets under `robot_infra/isaacsim_venvs/assets`.

The `assets` directory should have the following structure:

```text
<assets>
├── robots/
│   └── so101_follower.usd
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets/
        └── objects/
```

### 🤖 Getting Familiar with the Isaac Sim Environment

In this part, Isaac Sim is treated as a “digital twin”–level real-world proxy, providing high-fidelity physics simulation and real-time control interfaces for the SO101 arm.

For the SO101 arm, we provide two control modes:

- Cartesian pose control
- Joint position control

To improve the robustness of Real-World RL, we add **domain randomization** to the environment. Press the `R` key to quickly reset the scene.

During simulation, the robot’s physical state (including joint torques, end-effector poses, camera image streams, etc.) is published via ROS2 in real time, ensuring that the data observed by the algorithm is highly consistent with real-world physics. We recommend using Foxglove Studio for visualization and debugging, to monitor ROS2 topics and send control commands.

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./2_foxglove_inspect_data.sh
```

![Foxglove Studio Visualization](./assets/foxglove.png "Foxglove Studio Visualization")

In addition to using Foxglove Studio to monitor ROS2 topics and send commands, we also provide a Flask server–based communication interface with ROS2, following the design in [Hil-Serl](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/README.md).

First, build the Flask server in your ROS2 workspace:

```bash
cd IsaacSim-Hil-Serl/robot_infra/robot_servers
colcon build
```

Then start the Isaac Sim environment and the Flask server node in order:

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
```

In a new terminal, you can run the following commands to obtain similar monitoring and interaction capabilities as Foxglove Studio:

```bash
while true; do curl -X POST http://127.0.0.1:5000/get_joint_positions; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_joint_velocities; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_joint_efforts; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_joint_forces; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_joint_torques; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_poses_quat; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_poses_euler; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_forces; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_torques; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_velocities; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_eef_jacobians; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_state; echo; done
while true; do curl -X POST http://127.0.0.1:5000/get_config; echo; done
# Reset the robot to its initial pose
curl -X POST http://127.0.0.1:5000/reset_robot
# Publish joint positions
curl -X POST http://127.0.0.1:5000/move_joints -H "Content-Type: application/json" -d '{"joint_pose":[0,0,0.2,1.0,-1.5708,0.5]}'
# Publish end-effector pose as position + RPY
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[0.02,-0.23,0.18,21.24,-0.00,-180], "gripper_state":0.1}'
# Publish end-effector pose as position + quaternion (x, y, z, w)
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[-0.16,-0.14,0.2,0.2,0.2,0.2,0.2], "gripper_state":0.1}'
```

### 🛠️ Running HIL-SERL

#### Step 1. Define the Workspace

To avoid dangerous behaviors such as collisions during random exploration in RL training, you should define the robot’s workspace according to the task characteristics **before** training.

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./4_check_robot_workspace.sh
```

Instructions:

- **Configuration file**: The program saves workspace parameters in real time to the ROS2 parameter server file `robot_infra/robot_servers/src/so101_interfaces/config/so101_params.yaml`.
- **Isaac Sim control**: Press the `r` key to reset the environment.
- **Robot control**: Only Gamepad control is provided here.

Gamepad mapping:

| Control    | Description |
| :-------: | :---------: |
| move `L` forward / backward | Translate end-effector forward/backward |
| move `L` left / right | Control the `shoulder_pan` joint (left-right swing of the arm) |
| move `R` forward / backward | Control the `wrist_flex` joint (up-down pitch of the end-effector) |
| move `R` left / right | Control the `wrist_roll` joint (rotation of the end-effector) |
| press `LB` / `LT` | Translate the end-effector up/down |
| press `RB` / `RT` | Control the `grasp` joint (open/close the gripper) |

After the workspace is defined, rebuild the Flask server so that the updated parameters are applied in the parameter server:

```bash
cd IsaacSim-Hil-Serl/robot_infra/robot_servers
colcon build
```

#### Step 2. Train the Reward Classifier

In this step, you teleoperate the arm via Gamepad and manually label keyframes to collect samples for training the reward function. The data will be stored in `examples/experiments/so101_pick_oranges/classifier_data`.

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./5_record_classifier_data.sh
```

Instructions:

- **Robot control**: Same as in [Step 1. Define the Workspace](#step-1-define-the-workspace).
- **Labeling scheme**:
  - Press `b` to start recording samples for the current episode.
  - Press `space` to mark the current attempt as **Successful** and terminate the episode; the robot returns to its initial pose.
  - If the maximum step count for an episode is exceeded, the episode terminates automatically and the robot is reset.

- **Isaac Sim control**: Press `r` to reset the environment. It is recommended to use this when the robot is reset or when something goes wrong with the task.

After data collection, run the following command to train the reward classifier. The trained weights will be stored in `examples/experiments/so101_pick_oranges/classifier_ckpt`:

```bash
# Open in a new terminal
bash ./6_train_reward_classifier.sh
```

#### Step 3. Collect Demonstrations (Demos)

Before the main training, Hil-Serl needs a set of successful demonstrations based on the trained reward classifier. These demos are also collected via Gamepad teleoperation.

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./7_record_demos.sh
```

Instructions:

- **Robot control**: Same as in [Step 1. Define the Workspace](#step-1-define-the-workspace).
- **Demo collection**:
  - Press `b` to start recording the current episode. The trained reward classifier will automatically judge whether the attempt is successful.
  - If an attempt is judged successful, the current episode terminates automatically and a new recording session starts.
  - If the maximum step count for an episode is exceeded, the episode terminates automatically and the robot is reset.

Collected demos are stored in `examples/experiments/so101_pick_oranges/demo_data`. To verify the collected demos and check for potential issues, you can replay them:

```bash
# Open in a new terminal
bash ./8_replay_demos.sh
```

#### Step 4. Train Real-World RL

Follow the training procedure in [Hil-Serl](https://github.com/rail-berkeley/hil-serl). In the early training phase, you typically rely heavily on human intervention to guide the robot to complete the task, and then gradually reduce human involvement as the learned policy improves.

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./run_learner.sh
# Open in a new terminal
bash ./run_actor.sh
```

Instructions:

- **Robot control**: Same as in [Step 1. Define the Workspace](#step-1-define-the-workspace).

#### Step 5. Validate Real-World RL

After training, you can validate the trained policy as follows:

```bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./9_val_actor.sh
```

