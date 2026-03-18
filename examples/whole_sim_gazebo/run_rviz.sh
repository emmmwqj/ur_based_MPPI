#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then
    source ~/ur_arm/ros_ur_driver/install/setup.bash
fi

exec ros2 run rviz2 rviz2 -d "$SCRIPT_DIR/config/reach_static.rviz"
