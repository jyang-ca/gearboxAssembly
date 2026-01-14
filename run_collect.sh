#!/bin/bash
# Wrapper script to run data collection with the correct python environment

# Path to the Isaac Lab specific python
PYTHON_EXE="/venv/isaaclab/bin/python"

# Check if python exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python interpreter not found at $PYTHON_EXE"
    exit 1
fi

echo "Running collect_data.py using $PYTHON_EXE..."

# Run the script with default arguments
# You can override these by passing arguments to this script
# e.g. ./run_collect.sh --collect_steps 100

$PYTHON_EXE scripts/collect_data.py \
    --task Template-Galaxea-Lab-External-Direct-v0 \
    --num_envs 1 \
    --headless \
    --enable_cameras \
    "$@"
