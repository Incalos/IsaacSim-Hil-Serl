# Set the RMW implementation to Fast DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

#!/usr/bin/env bash
set -euo pipefail

# Get absolute path of the current script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extract experiment name from directory
EXPERIMENT_NAME=$(basename "${SCRIPT_DIR}")

# Define root and robot server paths
ROOT_DIR="${SCRIPT_DIR}/../../../"
SETUP_BASH="${ROOT_DIR}/robot_infra/robot_servers/install/setup.bash"

# Source ROS 2 environment
set +u
source "${SETUP_BASH}"
set -u

# Start the robot server with specified experiment
ros2 launch so101_interfaces start_robot_server.launch.py "experiment_name:=${EXPERIMENT_NAME}"
