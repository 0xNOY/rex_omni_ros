"""Shared helpers for ROS1/ROS2 integration tests."""

from __future__ import annotations

from typing import Any


FRAME_ID = "test_camera"


def make_image_msg(width: int = 640, height: int = 480) -> Any:
    """Build a black rgb8 sensor_msgs/Image (works for ROS1 and ROS2)."""
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.frame_id = FRAME_ID
    msg.height = height
    msg.width = width
    msg.encoding = "rgb8"
    msg.step = width * 3
    msg.data = bytes(height * width * 3)
    return msg


def make_bad_image_msg() -> Any:
    msg = make_image_msg(8, 8)
    msg.encoding = "yuv422"
    return msg
