#!/usr/bin/env bash
# Integration tests against the built ROS2 workspace.
set -eo pipefail  # no -u: colcon setup scripts reference unset variables
cd "$(dirname "$0")/.."
source install/ros2/setup.bash
exec pytest rex_omni_ros_python/test/integration -v "$@"
