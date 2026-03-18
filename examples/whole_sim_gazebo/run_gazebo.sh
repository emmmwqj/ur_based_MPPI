#!/bin/bash
set -e

echo "============================================================"
echo "启动 Gazebo UR7e 仿真 (ForwardPositionController)"
echo "============================================================"

source /opt/ros/humble/setup.bash

if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ Sourced ros_ur_driver"
fi

if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then
    source ~/ur_arm/gazebo_ur_sim/install/setup.bash
    echo "✓ Sourced gazebo_ur_sim"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_POSITIONS_FILE="${SCRIPT_DIR}/config/initial_positions.yaml"

ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
    ur_type:=ur7e \
    initial_joint_controller:=forward_position_controller \
    initial_positions_file:="${INITIAL_POSITIONS_FILE}" \
    launch_rviz:=true \
    gazebo_gui:=true
