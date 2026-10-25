from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l8_foxglove_demo'

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
        (os.path.join('share', package_name, 'foxglove'), glob('foxglove/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeid Kootbally',
    maintainer_email='zeidk@umd.edu',
    description='Every car and its CV prediction in Foxglove, live from CARLA '
                '(ENPM818Z L8 reading).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'l8_scene = l8_foxglove_demo.l8_scene_node:main',
        ],
    },
)
