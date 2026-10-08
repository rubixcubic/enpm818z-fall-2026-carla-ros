from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l5_box_demo'

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
    description='3D boxes from the LiDAR without learning: clusters and L-shape '
                'fitting, graded against CARLA (ENPM818Z L5).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'box_detector = l5_box_demo.box_detector_node:main',
            'snapshot     = l5_box_demo.snapshot:main',
        ],
    },
)
