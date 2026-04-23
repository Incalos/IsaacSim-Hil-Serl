# IsaacSim-Hil-Serl

> **中文文档**: [README_CN.md](./README_CN.md) | **English Documentation**: README.md

This project adapts UC Berkeley RAIL Lab's [HIL-SERL](https://github.com/rail-berkeley/hil-serl) framework and innovatively integrates NVIDIA Isaac Sim as a "pre-validation platform" for policy feasibility. By validating policies inside a simulator, we greatly reduce the trial-and-error risk of Real-World Reinforcement Learning (Real-World RL) on physical robots, enabling efficient RL training deployed directly on real hardware — achieving seamless transfer from virtual verification to real-world execution.

## Part 1: Project Setup

### 📁 Repository Structure

The IsaacSim-Hil-Serl repository has the following structure:

| Directory | Description |
| :-------: | :---------: |
| `dependencies` | Local dependency libraries required by the project |
| `examples` | Scripts for workspace calibration, demonstration data collection, reward classifier training, and policy training |
| `robot_infra` | Core infrastructure code that supports both simulation and real-robot operation |
| `robot_infra.gym_env` | Gym-style environment definitions for robot tasks |
| `robot_infra.isaacsim_venvs` | IsaacSim-based robot simulation environment configuration and initialization modules |
| `robot_infra.robot_servers` | Flask-based server implementation for ROS2 <-> robot interaction |
| `serl_launcher` | Core runtime logic and shared utilities that connect modules |
| `serl_launcher.agents` | Implementations of RL agent policies |
| `serl_launcher.common` | Shared foundational modules used across the framework |
| `serl_launcher.data` | Experience replay buffers and data storage management |
| `serl_launcher.networks` | Neural network layers and architectures used in training |
| `serl_launcher.utils` | Utility scripts and helper functions |
| `serl_launcher.vision` | Vision models and related helper functions |
| `serl_launcher.wrappers` | Gym environment wrappers and adapters |

### 💻 Supported Environment

- **Ubuntu 22.04**
- **CUDA 12.8**
- **Python 3.11**

### 📦 Install Project Dependencies

```bash
git clone https://github.com/Incalos/IsaacSim-Hil-Serl
cd IsaacSim-Hil-Serl

# Install system dependencies
sudo apt update && sudo apt install -y \
    xvfb \
    x11-utils \
    cmake \
    build-essential \
    coinor-libipopt-dev \
    gfortran \
    liblapack-dev \
    pkg-config \
    swig \
    git \
    python3 \
    python3-pip \
    git-lfs \
    foxglove-studio \
    --install-recommends

# Install uv and create basic Python venv
pip install uv --user
uv venv --python=3.11
source .venv/bin/activate
uv pip install ml_collections

# Install PyTorch
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install IsaacSim 5.1
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install IsaacLab under dependencies/
cd dependencies/ && \
git clone https://github.com/isaac-sim/IsaacLab.git && \
cd IsaacLab/ && \
./isaaclab.sh --install

# Install LeIsaac under dependencies/
cd dependencies/ && \
git clone --branch v0.3.0 --depth 1 https://github.com/LightwheelAI/leisaac.git && \
cd leisaac/ && \
uv pip install -e source/leisaac

# Install LeRobot under dependencies/
cd dependencies/ && \
git clone --branch v0.4.3 --depth 1 https://github.com/huggingface/lerobot.git && \
cd lerobot && \
uv pip install -e .

# Install cuRobo under dependencies/
cd dependencies/ && \
git clone --branch v0.7.7 --depth 1 https://github.com/NVlabs/curobo.git && \
cd curobo && \
uv pip install -e . --no-build-isolation

# Install agentlace under dependencies/
cd dependencies/ && \
git clone https://github.com/youliangtan/agentlace.git && \
cd agentlace/ && \
uv pip install -e .
```

## Part 2: Real-World RL for the SO101-Grasp-Orange Task

This chapter will systematically elaborate on the complete configuration scheme and training implementation paradigm for the Real-World Reinforcement Learning (RL) of the SO101 robotic arm executing the SO101-Grasp-Orange task within the IsaacSim simulation environment.

It should be specifically noted that the Real-World RL scheme adopted in this chapter relies on simulation for implementation: by leveraging IsaacSim to achieve a high-fidelity replication of the real robot's physical scenario, the iterative training and effectiveness validation of intelligent policies are completed without directly manipulating the physical hardware.

### 🧩 Prepare IsaacSim Assets

Prior to initiating the Real-World RL training process for the SO101 robotic arm, please first download the [USD assets](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip) and complete the deployment and configuration of the simulation scene.

Extract the downloaded archive and place all the extracted asset files into the `robot_infra/isaacsim_venvs/tasks/scenes` directory to complete the path deployment for the simulation resources.

The `scenes` folder must look like this:

```shell
tasks/
├── robots/
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets/
        └── objects/
```

Enter `robot_infra/isaacsim_venvs/tasks/scenes/kitchen_with_orange/objects` and remove the redundant assets `Orange002`, `Orange003`, and `Plate` from that folder.

### 🤖 Get Familiar with IsaacSim

IsaacSim acts as a high-fidelity digital-twin environment that provides low-latency control and accurate physics for the SO101 robot arm.

For SO101 this project supports both **cartesian pose control** and **joint position control**. To improve policy robustness, **domain randomization** is integrated in the environment. Press **R** during simulation to quickly reset the environment.

During simulation, key physical states — joint torques, end-effector poses, and camera image streams — are published over ROS2 to ensure observations align with realistic physics. We recommend using Foxglove Studio for visualization and debugging to monitor ROS2 topics and send control commands.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal run:
bash ./2_foxglove_inspect_data.sh
```

After Foxglove Studio starts, you can import `examples/experiments/so101_grasp_orange/foxglove_layout.json` to load a preset visualization layout.

![Foxglove Studio Debug View](./assets/foxglove.png "Foxglove Studio Debug View")

In addition to Foxglove Studio, we provide a Flask server that bridges ROS2 for monitoring and interaction. Steps to use the Flask server:

1. Build the Flask server

Enter the ROS2 workspace and build the Flask server code:

```bash
cd robot_infra/robot_servers
colcon build
```

2. Start simulation and the server

Start IsaacSim first, then start the Flask server node in order:

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal:
bash ./3_start_robot_server.sh
```

3. Monitor and interact

Open a new terminal and run the commands below to poll topics or send commands similarly to Foxglove Studio.

```bash
while true; do curl -X POST http://127.0.0.1:5000/get_joint_positions; echo; done
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
# Reset robot to initial pose and reset IsaacSim
curl -X POST http://127.0.0.1:5000/reset_robot
# Reset IsaacSim environment
curl -X POST http://127.0.0.1:5000/reset_isaacsim
# Publish joints as positions
curl -X POST http://127.0.0.1:5000/move_joints -H "Content-Type: application/json" -d '{"joint_pose":[0.5,0.1,-0.4,0.2,1.2,0.7]}'
# Publish EEF using position + RPY
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[0.27138811349868774,-0.0001829345856094733,0.21648338437080383,0.7695847901163139,0.030466061901383457,-1.6022399150116016], "gripper_state":0.5}'
# Publish EEF using position + quaternion (x,y,z,w)
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[0.2988014817237854,-0.05197674408555031,0.18513618409633636,-0.6495405435562134,-0.5627134442329407,0.3135982155799866,0.4038655161857605], "gripper_state":1.0}'
```

### 🕹️ Teleoperate the SO101 Arm {#teleoperate-the-so101-arm}

If you plan to use an Xbox controller for teleoperation, first connect the controller and obtain its unique GUID to avoid interference from other devices. Put the GUID into `examples/experiments/so101_grasp_orange/exp_params.yaml`. Keep the device connected and run the following command to list joystick devices and GUIDs:

```bash
python3 -c "import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'; import pygame; pygame.init(); pygame.joystick.init(); [print(f'\nIndex: {i}\nName: {j.get_name()}\nGUID: {j.get_guid()}\n' + '-'*20) or j.init() for i in range(pygame.joystick.get_count()) for j in [pygame.joystick.Joystick(i)]]; pygame.quit()"
```

The mapping relationship between Xbox controller buttons and robotic arm operations is shown in the table below. The button diagram is as follows:

<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 2rem; max-width: 100%;">
  <div style="flex: 1 1 450px; min-width: 300px;">
    <table style="width: 100%; border-collapse: collapse;">
      <tr>
        <th align="center" style="padding: 8px; border: 1px solid #ddd;">Control Button</th>
        <th align="center" style="padding: 8px; border: 1px solid #ddd;">Description</th>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Move Left Joystick Forward/Backward</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Translate the end effector of the robotic arm forward and backward</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Move Left Joystick Left/Right</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the shoulder_pan joint to swing left and right</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Move Right Joystick Forward/Backward</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the wrist_flex joint to pitch the end effector up and down</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Move Right Joystick Left/Right</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the wrist_roll joint to rotate the end effector</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Press LB Button</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the end effector of the robotic arm to translate upward</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Press LT Button</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the end effector of the robotic arm to translate downward</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Press RB Button</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the grasp joint to open the gripper</td>
      </tr>
      <tr>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Press RT Button</td>
        <td align="center" style="padding: 8px; border: 1px solid #ddd;">Control the grasp joint to close the gripper</td>
      </tr>
    </table>
  </div>
  <div style="flex: 1 1 300px; min-width: 280px; text-align: center;">
    <img src="./assets/xbox.jpg" alt="Xbox Controller" style="width: 100%; height: auto; object-fit: contain;">
  </div>
</div>

### 🛠️ Running Real-World RL

#### Step 1. Define the workspace

To avoid collisions and other safety risks during the exploration phase of RL, you must define the robot's workspace limits carefully before training.

These workspace parameters are monitored and adjusted during training and automatically saved to the task-specific config file: `examples/experiments/so101_grasp_orange/exp_params.yaml`.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal
bash ./3_start_robot_server.sh
# In a new terminal
bash ./4_check_robot_workspace.sh
```

Operating Instructions:

- Robotic arm control: Refer to [Teleoperate the SO101 Arm](#teleoperate-the-so101-arm).

- Workspace definition: Determine the reasonable motion range of the robotic arm according to specific task requirements. Before formal training begins, be sure to fully verify that the robotic arm can avoid collisions at all limit positions and attitudes within the task space, so as to ensure the safety of the workspace.

#### Step 2. Train the reward classifier

Use the Xbox controller to teleoperate the robot and collect keyframes from recorded videos for manual annotation. These labeled samples are used to train a reward classifier. Collected samples are stored in `examples/experiments/so101_grasp_orange/classifier_data/`.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal
bash ./3_start_robot_server.sh
# In a new terminal
bash ./5_record_classifier_data.sh
```
Operating Instructions:

- Robotic arm control: Refer to [Teleoperate the SO101 Arm](#teleoperate-the-so101-arm).

- Annotation guidelines:

  - Start recording: press `b` to begin recording the current episode.

  - Manually mark success: press `space` to mark the current attempt as a "success"; the episode will end and the robot + IsaacSim will reset to the initial state.

  - Auto terminate and reset: if an episode exceeds the configured max steps, the attempt will auto-terminate and reset the robot and IsaacSim state.

- Reset IsaacSim: press `r` (useful if the robot is stuck or the task fails).

After collecting data, train the reward classifier. The trained weights are saved to `examples/experiments/so101_grasp_orange/classifier_ckpt/`.

```bash
# In a new terminal run:
bash ./6_train_reward_classifier.sh
```

#### Step 3. Collect demonstrations

Before training policies, collect a set of high-quality demonstration trajectories using the trained reward classifier as a guide. Demonstrations are still collected via the Xbox controller.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal
bash ./3_start_robot_server.sh
# In a new terminal
bash ./7_record_demos.sh
```

Operating Instructions:

- Robotic arm control: Refer to [Teleoperate the SO101 Arm](#teleoperate-the-so101-arm).

- Demo collection:
  
  - Press `b` to start recording an episode; the reward classifier will judge success and automatically end and restart recording for successful episodes.
  
  - Episodes that exceed max steps will auto-terminate and reset.

Demonstrations are saved to `examples/experiments/so101_grasp_orange/demo_data/`. To validate collected demos, replay them:

```bash
# In a new terminal run:
bash ./8_replay_demos.sh
```

#### Step 4. Train the policy

Following the HIL-SERL training paradigm, the early stages require intensive human intervention: operators teleoperate the robot to guide it through many successful trials. High-frequency human guidance helps the policy learn quickly and adapt.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal
bash ./3_start_robot_server.sh
# In a new terminal
bash ./run_learner.sh
# In a new terminal
bash ./run_actor.sh
```

Operating Instructions:

- Robotic arm control: Refer to [Teleoperate the SO101 Arm](#teleoperate-the-so101-arm).

#### Step 5. Validate the policy

After training, load the learned policy and evaluate it inside IsaacSim to measure task performance, motion stability, and generalization under high-fidelity physical conditions. This provides strong evidence for successful sim-to-real transfer.

```bash
cd examples/experiments/so101_grasp_orange
bash ./1_start_isaacsim_venv.sh
# In a new terminal
bash ./3_start_robot_server.sh
# In a new terminal
bash ./9_val_actor.sh
```

![SO101 Reach Orange Policy](./assets/so101_reach_orange_policy.gif)
