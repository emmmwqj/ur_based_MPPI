#!/bin/bash
# ============================================================================
# UR7e HIL SAGE-MPPI RTDE + servoJ 控制器启动脚本
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e SAGE-MPPI - HIL_sage RTDE + servoJ"
echo "=============================================="

if [ -z "$ROS_DISTRO" ]; then
    echo "正在 source ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
    echo "✓ UR ROS2 Driver"
fi

echo "ROS_DISTRO: $ROS_DISTRO"

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

echo "⚠️  安全警告:"
echo "  这将通过 RTDE servoJ 控制 UR7e / URSim。"
echo "  建议先连接 URSim：--robot-ip 192.168.56.100"
echo "  确保:"
echo "    1. 工作区域无人员"
echo "    2. 急停按钮在可触及范围"
echo "    3. URSim 或真实 UR7e 已启动并可被 RTDE 访问"
echo "    4. 如需 RViz 状态显示，UR ROS2 Driver 正在发布 /joint_states"
echo ""
echo "按 Enter 继续，Ctrl+C 取消..."
read

echo "启动 HIL SAGE-MPPI RTDE servoJ 控制器..."
echo ""
cd "$SCRIPT_DIR"

if $LAUNCH_RVIZ; then
    (
        sleep 3
        echo "[RViz] 启动可视化..."
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/hil_rviz.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 ur7e_hil_sage_mpc_rtde.py "${PASSTHROUGH_ARGS[@]}"

if $LAUNCH_RVIZ && [ -n "$RVIZ_PID" ]; then
    echo ""
    echo "关闭 RViz..."
    kill $RVIZ_PID 2>/dev/null || true
fi
