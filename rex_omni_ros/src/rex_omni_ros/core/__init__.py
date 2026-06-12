"""ROS-agnostic core of the Rex-Omni wrapper.

This subpackage must not import any ROS module so that it stays unit-testable
without a ROS runtime. It depends only on numpy, Pillow and (lazily) vLLM.
"""
