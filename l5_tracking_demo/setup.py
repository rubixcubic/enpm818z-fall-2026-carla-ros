from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l5_tracking_demo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeid Kootbally',
    maintainer_email='zeidk@umd.edu',
    description='Multi-object tracking in CARLA: one Kalman filter per object, '
                'gating, GNN or NN, track lifecycle (ENPM818Z L5).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector       = l5_tracking_demo.detector_node:main',
            'truth_detector = l5_tracking_demo.truth_detector_node:main',
            'tracker        = l5_tracking_demo.tracker_node:main',
            'evaluate       = l5_tracking_demo.evaluate_node:main',
            'snapshot       = l5_tracking_demo.snapshot:main',
            'spawn_traffic  = l5_tracking_demo.spawn_traffic:main',
        ],
    },
)
