from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l5_bev_demo'

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
    description='Three bird\'s-eye views from CARLA: LiDAR grid, camera IPM, '
                'semantic lift-splat (ENPM818Z L5).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'surround_rig = l5_bev_demo.surround_rig_node:main',
            'lidar_bev    = l5_bev_demo.lidar_bev_node:main',
            'ipm_bev      = l5_bev_demo.ipm_bev_node:main',
            'semantic_bev = l5_bev_demo.semantic_bev_node:main',
            'snapshot     = l5_bev_demo.snapshot:main',
        ],
    },
)
