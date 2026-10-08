from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l3_ekf_demo'

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
    description='Fuse CARLA GNSS and IMU with an EKF and log the NIS (ENPM818Z L3).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'truth      = l3_ekf_demo.truth_node:main',
            'logger     = l3_ekf_demo.logger_node:main',
            'ekf        = l3_ekf_demo.ekf_node:main',
            'plot_nis   = l3_ekf_demo.plot_nis:main',
        ],
    },
)
