#!/bin/bash
# ============================================================================
# UR7e STORM MPC Gazebo 仿真启动脚本
# ============================================================================
#
# 用法:
#   终端 1: ./run_gazebo.sh          # 启动 Gazebo 仿真
#   终端 2: ./run_mpc.sh             # 启动 MPC 控制器
#
# ============================================================================

set -e

echo "============================================================"
echo "启动 Gazebo UR7e 仿真 (ForwardPositionController)"
echo "============================================================"

# Source ROS2 环境
source /opt/ros/humble/setup.bash

# Source UR 驱动工作空间
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ Sourced ros_ur_driver"
fi

# Source Gazebo 仿真工作空间
if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then
    source ~/ur_arm/gazebo_ur_sim/install/setup.bash
    echo "✓ Sourced gazebo_ur_sim"
fi

echo ""
echo "启动参数:"
echo "  ur_type: ur7e"
echo "  initial_joint_controller: forward_position_controller"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_POSITIONS_FILE="${SCRIPT_DIR}/config/initial_positions.yaml"

# 启动 Gazebo 仿真
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
    ur_type:=ur7e \
    initial_joint_controller:=forward_position_controller \
    initial_positions_file:="${INITIAL_POSITIONS_FILE}" \
    launch_rviz:=true \
    gazebo_gui:=true
