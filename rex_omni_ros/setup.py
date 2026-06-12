# Used only by the ROS1 (catkin) build via catkin_python_setup().
# The ROS2 build installs the package with ament_python_install_package().
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(
    **generate_distutils_setup(
        packages=["rex_omni_ros", "rex_omni_ros.core"],
        package_dir={"": "src"},
    )
)
