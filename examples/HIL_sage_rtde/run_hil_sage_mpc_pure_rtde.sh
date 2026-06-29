#!/bin/bash
# Pure RTDE UR7e HIL SAGE-MPPI launcher.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e SAGE-MPPI - Pure RTDE HIL"
echo "=============================================="

if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS2 Humble..."
    source /opt/ros/humble/setup.bash
fi

echo "ROS_DISTRO: $ROS_DISTRO"

eval "$(conda shell.bash hook)"
STORM_CONDA_ENV="${STORM_CONDA_ENV:-whole_control}"
conda activate "$STORM_CONDA_ENV"

if [[ "$CONDA_DEFAULT_ENV" != "$STORM_CONDA_ENV" ]]; then
    echo "error: failed to activate $STORM_CONDA_ENV"
    exit 1
fi

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

LAUNCH_RVIZ=true
RVIZ_PID=""
RSP_PID=""
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

echo "Safety warning:"
echo "  This controller sends RTDE servoJ commands directly to UR7e / URSim."
echo "  It does not require or launch the UR ROS driver."
echo "  Make sure the workspace is clear and emergency stop is reachable."
echo ""
echo "Press Enter to continue, Ctrl+C to cancel..."
read

cd "$SCRIPT_DIR"

cleanup() {
    if [ -n "$RVIZ_PID" ]; then
        echo ""
        echo "Stopping RViz..."
        kill "$RVIZ_PID" 2>/dev/null || true
    fi
    if [ -n "$RSP_PID" ]; then
        echo "Stopping robot_state_publisher..."
        kill "$RSP_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

if $LAUNCH_RVIZ; then
    ROBOT_MODEL_URDF="$SCRIPT_DIR/config/ur7e_robot_model.urdf"
    if [ ! -f "$ROBOT_MODEL_URDF" ]; then
        echo "error: robot model URDF not found: $ROBOT_MODEL_URDF"
        exit 1
    fi
    echo "[RobotModel] starting robot_state_publisher..."
    ros2 run robot_state_publisher robot_state_publisher "$ROBOT_MODEL_URDF" >/tmp/hil_sage_rtde_robot_state_publisher.log 2>&1 &
    RSP_PID=$!

    (
        sleep 3
        echo "[RViz] starting visualization..."
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/hil_rviz.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

python3 ur7e_hil_sage_mpc_pure_rtde.py "${PASSTHROUGH_ARGS[@]}"
