#!/bin/bash
# ============================================================================
# UR7e HIL SAGE RViz 可视化启动脚本
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "启动 RViz - UR7e HIL SAGE 可视化"
echo "=============================================="

# Source ROS2
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi

# Source UR 驱动 (for robot_description 等)
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ Sourced UR ROS2 Driver"
fi

echo "ROS_DISTRO: $ROS_DISTRO"
echo ""

# 检查 RViz 配置文件
RVIZ_CONFIG="$SCRIPT_DIR/config/hil_rviz.rviz"
if [ ! -f "$RVIZ_CONFIG" ]; then
    echo "警告: RViz 配置文件不存在: $RVIZ_CONFIG"
    echo "使用默认配置启动..."
    ros2 run rviz2 rviz2
else
    echo "使用配置: $RVIZ_CONFIG"
    echo ""
    ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG"
fi
