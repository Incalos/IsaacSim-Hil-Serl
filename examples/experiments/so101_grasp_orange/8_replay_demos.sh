#!/usr/bin/env bash
set -euo pipefail

# Get absolute path of the current script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extract experiment name from directory
EXPERIMENT_NAME=$(basename "${SCRIPT_DIR}")

# Define root and robot server paths
ROOT_DIR="${SCRIPT_DIR}/../../../"

# Navigate to root directory, activate virtual environment and replay demos
cd "${ROOT_DIR}"
source .venv/bin/activate
uv run examples/replay_demos.py --exp_name="${EXPERIMENT_NAME}" --demo_file=demo_data/so101_grasp_orange_2_demos_2026-03-25_19-29-47.pkl