# IsaacSim-Hil-Serl

基于 Isaac Sim 的混合真机强化学习（HIL-SERL）框架，用于在高保真仿真和真实机器人之间搭建统一的真实世界强化学习（Real-World RL）训练闭环。

## Part 1: 项目准备

### 📁 代码结构

IsaacSim-Hil-Serl 项目的主要代码结构如下所示：

| 目录 | 说明 |
| :-------: | :---------: |
| `examples` 	| 策略训练、演示数据采集与奖励分类器训练相关脚本 |
| `serl_launcher` | IsaacSim-HIL-SERL 的核心启动与训练代码 |
| `serl_launcher.agents` | 强化学习智能体策略实现（如 SAC） |
| `serl_launcher.wrappers` | 针对 Gym 环境的包装与适配模块 |
| `serl_launcher.data` | 回放缓冲区与数据存储模块 |
| `serl_launcher.vision` | 视觉相关模型与工具函数 |
| `robot_infra` | 真实与仿真机器人运行所需的基础设施代码 |
| `robot_infra.robot_servers` | 通过 ROS2 与机器人交互的 Flask 服务端 |
| `robot_infra.gym_env` | 机器人相关的 Gym 环境定义 |
| `robot_infra.isaacsim_venvs` | 基于 Isaac Sim 的机器人虚拟环境配置 |

### 💻 项目运行环境

- **Ubuntu 22.04**
- **CUDA 12.8**
- **Python 3.11**

### 📦 安装基础依赖

```Bash
git clone https://github.com/Incalos/IsaacSim-Hil-Serl
cd IsaacSim-Hil-Serl

# 安装 Foxglove，参考 https://foxglove.dev/download

# 安装 uv，参考 https://docs.astral.sh/uv/getting-started/installation/

# 安装基础 python 环境
uv venv --python=3.11
source .venv/bin/activate
uv pip install --upgrade pip
sudo apt update && sudo apt install -y xvfb x11-utils
sudo apt install cmake build-essential

# 安装 PyTorch
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 Isaac Sim 5.1 
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 在 dependencies/ 下安装主要依赖
cd dependencies/

# 在 dependencies/ 下安装 Isaac Lab
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab/
./isaaclab.sh --install

# 在 dependencies/ 下安装 LeIsaac
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac/
uv pip install -e source/leisaac

# 在 dependencies/ 下安装 LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .

# 在 dependencies/ 下安装 cuRobo
git clone --branch v0.7.7 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
uv pip install -e . --no-build-isolation

# 在 dependencies/ 下安装 agentlace
git clone https://github.com/youliangtan/agentlace.git
cd agentlace/
uv pip install -e .
```

## Part 2: SO101-PickOranges 任务的 Real-World RL

本部分详细介绍如何在 Isaac Sim 仿真环境中配置并训练 SO101 机械臂的真实世界强化学习（Real-World RL）。需要特别说明的是，这里的 Real-World RL 为“伪真机”设置，即通过 Isaac Sim 高保真模拟真实机器人，从而在不直接操作真机的前提下完成策略训练与验证。

### 🧩 Isaac Sim 资产准备

本部分以开源的 SO101-PickOranges 任务为例，需要先[下载 USD 资源](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)并配置场景文件。

解压下载好的压缩包，并将资产放置在 `robot_infra/isaacsim_venvs/assets` 文件夹中。

`assets` 文件夹的结构如下：

```
<assets>
├── robots/
│   └── so101_follower.usd
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets/
        └── objects/
```

### 🤖 熟悉 Isaac Sim 环境

在本部分中，我们将 Isaac Sim 视作“数字孪生”层面的真实世界代理，用于为 SO101 机械臂提供高保真度的物理模拟与实时控制接口。

针对 SO101 机械臂，我们提供了笛卡尔位姿控制（cartesian pose control）与关节位置控制（joint position control）两种控制模式。为提升 Real-World RL 的鲁棒性，环境中加入了域随机化（domain randomization）策略；按下键盘的 `R` 键即可快速重置环境。

机械臂在仿真过程中的物理状态（包括关节力矩、末端位姿、相机图像流等）均通过 ROS2 实时发布，确保算法获取的数据与真实世界物理规律高度一致。推荐使用 Foxglove Studio 进行可视化调试，以实时监控 ROS2 话题并下发控制指令。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./2_foxglove_inspect_data.sh
```

![Foxglove Studio 可视化调试](./assets/foxglove.png "Foxglove Studio 可视化调试")

除了使用 Foxglove Studio 实时监控 ROS2 话题并下发控制指令外，参考 [Hil-Serl](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/README.md)，我们还提供基于 Flask Server 的方式与 ROS2 进行通讯。

首先，需在 ROS2 的工作空间内编译 Flask Server。

```Bash
cd IsaacSim-Hil-Serl/robot_infra/robot_servers
colcon build
```

之后，依次启动 Isaac Sim 环境以及 Flask Server 节点。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
```

新开一个终端，输入以下指令，也可实现与 Foxglove Studio 相似的监控与交互功能。

```Bash
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
# Robot 恢复初始位姿
curl -X POST http://127.0.0.1:5000/reset_robot
# Joints 以 position 的格式发布
curl -X POST http://127.0.0.1:5000/move_joints -H "Content-Type: application/json" -d '{"joint_pose":[0,0,0.2,1.0,-1.5708,0.5]}'
# EEF 以 position + rpy 格式发布
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[0.02,-0.23,0.18,21.24,-0.00,-180], "gripper_state":0.1}'
# EEF 以 position + quaternion (x,y,z,w)格式发布
curl -X POST http://127.0.0.1:5000/move_eef -H "Content-Type: application/json" -d '{"eef_pose":[-0.16,-0.14,0.2,0.2,0.2,0.2,0.2], "gripper_state":0.1}'
```

### 🛠️ 运行 HIL-SERL

##### Step 1. 定义工作空间

为避免在强化学习随机探索过程中发生机器人碰撞等危险情况，在训练开始之前，需要根据任务特性预先确定其工作空间。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./4_check_robot_workspace.sh
```

操作说明：

- 配置文件：程序会实时将工作空间参数保存到 ROS2 参数服务器的配置文件 `robot_infra/robot_servers/src/so101_interfaces/config/so101_params.yaml` 中。

- Isaac Sim 控制方式：按下键盘上的 `r` 键可以重置环境。

- 机械臂控制方式：此处仅提供 Gamepad 的控制方式。

| 控制按键    | 描述 |
| :-------: | :---------: |
| move `L` forward / backward | 控制末端的前后平移 |
| move `L` left / right | 控制 `shoulder_pan` 关节，机械臂的左右摆动 |
| move `R` forward / backward | 控制 `wrist_flex` 关节，末端的上下俯仰 |
| move `R` left / right | 控制 `wrist_roll` 关节，末端的旋转 |
| press `LB` / `LT` | 控制末端的上下平移 |
| press `RB` / `RT` | 控制 `grasp` 关节，末端夹爪的开合 |

工作空间定义好后，需要重新编译 Flask Server，以将最新参数传入参数服务器中。

```Bash
cd IsaacSim-Hil-Serl/robot_infra/robot_servers
colcon build
```

##### Step 2. 训练奖励分类器（Reward Classifier）

在本步骤中，我们通过 Gamepad 遥操作机械臂并人工标注关键帧，收集用于训练奖励函数的样本。样本将存放于 `examples/experiments/so101_pick_oranges/classifier_data` 文件夹中。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./5_record_classifier_data.sh
```

操作说明：

- 机械臂控制方式：参考 [Step 1. 定义工作空间](#step-1-定义工作空间)。

- 样本标注方式：

    - 按下 `b` 键开启当前回合（episode）的样本记录。
    
    - 按下 `space` 键将当前尝试标记为“成功（Successful）”并终止该回合，机器人将恢复初始位姿。
    
    - 当超过单个回合的最大步骤时，当前尝试会自动终止并重置机器人位姿。

- Isaac Sim 控制方式：按下键盘上的 `r` 键将重置环境，建议在机器人重置位姿或者任务发生异常时使用。

样本采集完成后，运行如下命令训练奖励分类器（Reward Classifier），训练得到的权重将保存在 `examples/experiments/so101_pick_oranges/classifier_ckpt` 文件夹中。

```Bash
# Open in a new terminal
bash ./6_train_reward_classifier.sh
```

##### Step 3. 收集演示数据（Demo）

在正式训练之前，Hil-Serl 需要基于前面训练好的奖励分类器（Reward Classifier）额外采集一批成功的演示数据（demo）。此步骤依然通过 Gamepad 遥操作完成。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./7_record_demos.sh
```

操作说明：

- 机械臂控制方式：参考 [Step 1. 定义工作空间](#step-1-定义工作空间)。

- Demo 收集方式：

    - 按下 `b` 键开启当前回合（episode）的样本记录，训练好的奖励分类器会自动判断当前操作是否成功。若判断成功，则自动终止当前步骤，并开启新的记录流程。
    
    - 当超过单个回合的最大步骤时，当前尝试会自动终止并重置机器人位姿。

采集好的 demo 将存放于 `examples/experiments/so101_pick_oranges/demo_data` 文件夹中。为了验证采集的 demo 是否存在错误，可以运行如下命令进行回放与验证。

```Bash
# Open in a new terminal
bash ./8_replay_demos.sh
```

##### Step 4. Real-World RL 的训练

参考 [Hil-Serl](https://github.com/rail-berkeley/hil-serl) 的训练方式，在训练前期更多依赖人工介入，引导机械臂完成任务，随着策略性能提升逐步减少人工干预。

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./run_learner.sh
# Open in a new terminal
bash ./run_actor.sh
```

操作说明：

- 机械臂控制方式：参考 [Step 1. 定义工作空间](#step-1-定义工作空间)。

##### Step 5. Real-World RL 的验证

```Bash
cd IsaacSim-Hil-Serl/examples/experiments/so101_pick_oranges
bash ./1_start_isaacsim_venv.sh
# Open in a new terminal
bash ./3_start_robot_server.sh
# Open in a new terminal
bash ./9_val_actor.sh
```