#!/usr/bin/env bash
set -euo pipefail

# Set the RMW implementation to Fast DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Get absolute path of the current script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extract experiment name from directory
EXPERIMENT_NAME=$(basename "${SCRIPT_DIR}")

# Define root path
ROOT_DIR="${SCRIPT_DIR}/../../../"

# Change to root directory
cd "${ROOT_DIR}"

# Activate Python virtual environment
source .venv/bin/activate

# Launch IsaacSim environment with experiment configuration
uv run robot_infra/isaacsim_venvs/start_so101_venv.py --experiment_name="${EXPERIMENT_NAME}"
