#!/usr/bin/env bash
set -euo pipefail

# Get absolute path of the current script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extract experiment name from directory
EXPERIMENT_NAME=$(basename "${SCRIPT_DIR}")

# Define root and robot server paths
ROOT_DIR="${SCRIPT_DIR}/../../../"

# Navigate to root directory, activate virtual environment and check robot workspace
cd "${ROOT_DIR}"
source .venv/bin/activate
uv run examples/check_workspace.py --exp_name="${EXPERIMENT_NAME}"