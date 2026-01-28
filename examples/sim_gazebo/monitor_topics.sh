#!/bin/bash
# ============================================================================
# ROS2 话题监控脚本
# ============================================================================

echo "============================================================"
echo "ROS2 话题监控"
echo "============================================================"

source /opt/ros/humble/setup.bash

echo ""
echo "=== 控制器列表 ==="
ros2 control list_controllers 2>/dev/null || echo "(controller_manager 未运行)"

echo ""
echo "=== 关键话题 ==="
echo ""
echo "--- /joint_states (最新一条) ---"
ros2 topic echo /joint_states --once 2>/dev/null || echo "(无数据)"

echo ""
echo "--- /forward_position_controller/commands (最新一条) ---"
ros2 topic echo /forward_position_controller/commands --once 2>/dev/null || echo "(无数据)"

echo ""
echo "=== 话题频率 ==="
echo "按 Ctrl+C 退出"
echo ""
ros2 topic hz /joint_states &
PID=$!
sleep 3
kill $PID 2>/dev/null
