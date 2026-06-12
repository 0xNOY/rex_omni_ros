#!/usr/bin/env bash
# Integration tests against the built ROS1 workspace; starts roscore if needed.
set -eo pipefail  # no -u: setup scripts reference unset variables
cd "$(dirname "$0")/.."
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
source install/ros1/setup.bash

roscore_pid=""
if ! rosnode list >/dev/null 2>&1; then
  roscore >/tmp/rex_omni_test_roscore.log 2>&1 &
  roscore_pid=$!
  for _ in $(seq 1 40); do
    rosnode list >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

set +e
pytest rex_omni_ros_python/test/integration -v "$@"
status=$?
set -e

if [ -n "$roscore_pid" ]; then
  kill "$roscore_pid" 2>/dev/null || true
  pkill -f rosmaster 2>/dev/null || true
  pkill -f 'rosout/rosout' 2>/dev/null || true
fi
exit $status
