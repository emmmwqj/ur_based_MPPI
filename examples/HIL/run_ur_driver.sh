#!/bin/bash
# ============================================================================
# UR7e ROS2 驱动启动脚本
# ============================================================================
#
# 启动 UR 官方 ROS2 驱动连接真实 UR7e 机器人
#
# 前置条件:
#   1. UR7e 机器人已开机并连接网络 (IP: 192.168.56.100)
#   2. UR 示教器上运行 External Control 程序
#
# ============================================================================

set -e

echo "=============================================="
echo "UR7e ROS2 驱动启动"
echo "=============================================="

# 配置
ROBOT_IP="192.168.56.100"
UR_TYPE="ur7e"
CALIBRATION_FILE="${HOME}/ur_arm/my_robot_calibration.yaml"

# 检查网络连接
echo ""
echo "检查机器人连接..."
if ping -c 1 -W 2 $ROBOT_IP > /dev/null 2>&1; then
    echo "✓ 机器人 ($ROBOT_IP) 网络可达"
else
    echo "✗ 无法连接机器人 ($ROBOT_IP)"
    echo "  请检查："
    echo "    1. 机器人是否开机"
    echo "    2. 网络连接是否正常"
    echo "    3. IP 地址是否正确"
    exit 1
fi

# 检查标定文件
if [ -f "$CALIBRATION_FILE" ]; then
    echo "✓ 标定文件存在: $CALIBRATION_FILE"
else
    echo "⚠ 标定文件不存在: $CALIBRATION_FILE"
    echo "  将使用默认标定参数"
    CALIBRATION_FILE=""
fi

# Source ROS2 环境
echo ""
echo "配置 ROS2 环境..."
source /opt/ros/humble/setup.bash
echo "✓ ROS2 Humble"

# Source UR 驱动
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ UR ROS2 Driver"
else
    echo "✗ UR ROS2 Driver 未找到"
    echo "  请确保 ~/ur_arm/ros_ur_driver 已编译"
    exit 1
fi

# 启动驱动
echo ""
echo "=============================================="
echo "启动 UR7e ROS2 驱动"
echo "  机器人 IP: $ROBOT_IP"
echo "  机器人类型: $UR_TYPE"
echo "=============================================="
echo ""
echo "⚠️  安全提示:"
echo "  1. 确保工作区域无人员"
echo "  2. 急停按钮在可触及范围内"
echo "  3. 在 UR 示教器上启动 External Control 程序"
echo ""
echo "按 Enter 继续，Ctrl+C 取消..."
read

if [ -n "$CALIBRATION_FILE" ]; then
    ros2 launch ur_robot_driver ur_control.launch.py \
        ur_type:=$UR_TYPE \
        robot_ip:=$ROBOT_IP \
        kinematics_params_file:="$CALIBRATION_FILE" \
        initial_joint_controller:=forward_position_controller \
        launch_rviz:=false
else
    ros2 launch ur_robot_driver ur_control.launch.py \
        ur_type:=$UR_TYPE \
        robot_ip:=$ROBOT_IP \
        initial_joint_controller:=forward_position_controller \
        launch_rviz:=false
fi
