#!/bin/bash
# ============================================================================
# UR7e HIL MPC 控制器启动脚本
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e STORM MPC - 硬件在环仿真 (HIL)"
echo "=============================================="

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "正在 source ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

# Source UR 驱动
if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ UR ROS2 Driver"
fi

echo "ROS_DISTRO: $ROS_DISTRO"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate storm_py310

if [[ "$CONDA_DEFAULT_ENV" != "storm_py310" ]]; then
    echo "错误: 无法激活 storm_py310 环境"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# 检查是否需要启动 RViz
LAUNCH_RVIZ=true
RVIZ_PID=""
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    fi
done

# 安全提示
echo "⚠️  安全警告:"
echo "  这将控制真实机械臂！"
echo "  确保:"
echo "    1. 工作区域无人员"
echo "    2. 急停按钮在可触及范围"
echo "    3. UR ROS2 Driver 已启动"
echo "    4. External Control 程序正在运行"
echo ""
echo "按 Enter 继续，Ctrl+C 取消..."
read

# 运行控制器
echo "启动 HIL MPC 控制器..."
echo ""
cd "$SCRIPT_DIR"

# 启动 RViz (延迟后台启动)
if $LAUNCH_RVIZ; then
    (
        sleep 3
        echo "[RViz] 启动可视化..."
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/hil_rviz.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 ur7e_hil_mpc.py "$@"

# 清理 RViz
if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ]; then
    echo ""
    echo "关闭 RViz..."
    kill $RVIZ_PID 2>/dev/null || true
fi
