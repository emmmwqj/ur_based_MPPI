#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ROS_DISTRO:-}" ]; then
    source /opt/ros/humble/setup.bash
fi

eval "$(conda shell.bash hook)"
conda activate whole_control

cd "${SCRIPT_DIR}"
python3 run_sequential_retargeting_validation.py --with-rviz "$@"
