# setup.py file for ROS package

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['door_detector_ros'],
    package_dir={'': 'src'}
)

setup(**d)
