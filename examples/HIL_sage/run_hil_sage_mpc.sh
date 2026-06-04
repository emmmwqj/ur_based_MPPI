#!/bin/bash
# ============================================================================
# UR7e HIL SAGE-MPPI 控制器启动脚本
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e SAGE-MPPI - 硬件在环仿真 (HIL_sage)"
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

# 激活 conda 环境。默认使用 clean_SAGE 使用的 whole_control；
# 如需沿用 HIL 环境，可运行 STORM_CONDA_ENV=storm_py310 ./run_hil_sage_mpc.sh
eval "$(conda shell.bash hook)"
STORM_CONDA_ENV="${STORM_CONDA_ENV:-whole_control}"
conda activate "$STORM_CONDA_ENV"

if [[ "$CONDA_DEFAULT_ENV" != "$STORM_CONDA_ENV" ]]; then
    echo "错误: 无法激活 $STORM_CONDA_ENV 环境"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# 检查是否需要启动 RViz
LAUNCH_RVIZ=true
RVIZ_PID=""
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    else
        PASSTHROUGH_ARGS+=("$arg")
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
echo "启动 HIL SAGE-MPPI 控制器..."
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

python3 ur7e_hil_sage_mpc.py "${PASSTHROUGH_ARGS[@]}"

# 清理 RViz
if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ]; then
    echo ""
    echo "关闭 RViz..."
    kill $RVIZ_PID 2>/dev/null || true
fi
