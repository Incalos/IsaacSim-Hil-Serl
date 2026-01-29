# IsaacSim-Hil-Serl

## Part 1

Part 1 详细说明如何在 IsaacSim 仿真环境中配置并测试 SO101 机械臂 Manipulation 的 Real World RL (此处 Real World 为 IsaacSim)。

### 📋 前提条件

- 建议安装 Foxglove Studio, uv。

- 运行环境为 Ubantu 22.04, CUDA 12.8, Python 3.11。

#### 🚀 步骤 1：安装基础框架

```Bash
git clone https://github.com/Incalos/IsaacSim-Hil-Serl
cd IsaacSim-Hil-Serl

# 安装基础 python 环境
uv venv --python=3.11
source .venv/bin/activate
uv pip install --upgrade pip
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
```

#### 📦 步骤 2：资产准备 (Asset Preparation)

为了运行 SO101 的任务 SO101-PickOranges，需要[下载](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0)并配置 USD 场景文件。

解压文件，将资产放置在 `robot_infra/isaacsim_venvs/so101_pick_oranges/assets` 文件夹中. 

`assets` 文件夹的结构如下:

```
<assets>
├── robots/
│   └── so101_follower.usd
└── scenes/
    └── kitchen_with_orange/
        ├── scene.usd
        ├── assets
        └── objects/
            ├── Orange001
            ├── Orange002
            ├── Orange003
            └── Plate
```

#### 🤖 步骤 3：熟悉 SO101 机械臂 Isaac Sim 仿真环境

本环境将 Isaac Sim 视作“数字孪生”层面的 Real World 代理，旨在为 SO101 机械臂提供高保真度的物理模拟与实时控制接口。

针对 SO101 机械臂，我们提供了 cartesian pose control 以及 joint position control 两种控制模式。为提升 Real World RL 的鲁棒性，我们为该环境增加了 Domain randomization 策略，按下键盘的 `R` 键即可重置该环境。

机械臂在仿真过程中的物理状态（包括关节力矩、末端位姿、相机流等）均通过 ROS2 实时发布，确保算法获取的数据与真实世界物理规律高度一致。我们推荐使用 Foxglove Studio 进行可视化调试，实时监控 ROS2 话题并下发控制指令。

```Bash
cd IsaacSim-Hil-Serl/examples/SO101/pick_oranges

bash ./start_isaacsim_venv.sh

# Open in a new terminal
bash ./foxglove_inspect_data.sh
```

![Foxglove可视化调试](./assets/foxglove.png "Foxglove可视化调试")

#### 🛠️ 步骤 4：配置并运行 HIL-SERL

##### 4.1 定义机械臂的工作空间

在开始正式训练前，需要根据具体任务确定 SO101 机械臂的工作空间。

```Bash
cd IsaacSim-Hil-Serl/examples/SO101/pick_oranges
bash ./check_robot_workspace.sh
```

操作说明：

- IsaacSim 控制方式：按下键盘上的 `b` 键开启环境；按下 `r` 键将重置环境。

- 机械臂控制方式：此处仅提供 SO101-Leader、Keyboard、Gamepad 三种控制方式，具体参考[文档](https://lightwheelai.github.io/leisaac/resources/available_devices)。

- 配置文件：脚本会实时将工作空间参数传入 ROS2 参数服务器的配置文件中。

##### 4.2 遥操收集离线示例

在 Issac Sim 中通过遥操作控制机械臂，并收集离线示例。

```Bash
cd IsaacSim-Hil-Serl/examples/SO101/pick_oranges
bash ./record_task_demos.sh
```

操作说明：

- 机械臂控制方式：使用键盘控制移动机械臂到期待的极限位置，参考[文档](https://lightwheelai.github.io/leisaac/resources/available_devices)。

- IsaacSim 控制方式：

    - 按下 `b` 键开启机械臂键盘控制模式；
    
    - 按下 `r` 键将重置环境并将当前尝试标记为 `失败 (Failed)`；
    
    - 按下 `n` 键将重置环境并将当前尝试标记为 `成功 (Successful)`;


##### 4.3 训练 Reward Classifier



##### 4.4 收集离线示例



##### 4.5 训练 Policy

此处将 Isaac Sim 视作“数字孪生”层面的 Real World 代理，故在开始训练 Policy 之前，需要启动该虚拟环境并配置 Flask Server。

```Bash
# 编译 ros2
cd IsaacSim-Hil-Serl/robot_infra/ros2_ws
colcon build

# 启动 IsaacSim
cd IsaacSim-Hil-Serl/examples/SO101/pick_oranges
bash ./start_isaacsim_venv.sh

# 另开一个终端
bash ./start_robot_server.sh
```

