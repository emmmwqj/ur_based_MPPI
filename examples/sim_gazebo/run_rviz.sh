#!/bin/bash
# 启动 RViz 可视化
# 用于查看目标位置、末端执行器和障碍物

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e STORM MPC - RViz 可视化"
echo "=============================================="

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "正在 source ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

# Source UR 描述包
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
fi

echo "ROS_DISTRO: $ROS_DISTRO"
echo ""
echo "话题说明:"
echo "  /target_marker    - 红球: 目标位置"
echo "  /ee_marker        - 绿球: 末端执行器位置"
echo "  /obstacle_markers - 蓝球: 障碍物"
echo ""

# 启动 RViz
ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/reach_static.rviz"
