#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "UR7e Dynamic Primitive Ball Reaching Demo"
echo "=============================================="

source /opt/ros/humble/setup.bash
eval "$(conda shell.bash hook)"
conda activate storm_py310

LAUNCH_RVIZ=true
PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rviz" ]]; then
        LAUNCH_RVIZ=false
        continue
    fi
    PY_ARGS+=("$arg")
done

RVIZ_PID=""
cleanup() {
    if $LAUNCH_RVIZ && [[ -n "$RVIZ_PID" ]] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
}
on_interrupt() {
    cleanup
    exit 130
}
trap on_interrupt INT TERM
trap cleanup EXIT

if $LAUNCH_RVIZ; then
    (
        sleep 3
        ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/reach_dynamic_ball.rviz" 2>/dev/null
    ) &
    RVIZ_PID=$!
fi

cd "$SCRIPT_DIR"
python3 scripts/reach_dynamic_ball.py "${PY_ARGS[@]}"
