from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l9_planning_demo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'data'), glob('data/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeid Kootbally',
    maintainer_email='zeidk@umd.edu',
    description='The L9 sampling planner on Town03 in CARLA, drawn in RViz (ENPM818Z L9).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planning_demo = l9_planning_demo.demo_node:main',
        ],
    },
)
