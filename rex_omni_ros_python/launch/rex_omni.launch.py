"""ROS2 launch file for the rex_omni service node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params = os.path.join(
        get_package_share_directory("rex_omni_ros_python"),
        "config",
        "rex_omni_ros2.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="rex_omni_ros_python",
                executable="rex_omni_server",
                name="rex_omni",
                output="screen",
                parameters=[params],
            )
        ]
    )
