from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'l10_topics_demo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeid Kootbally',
    maintainer_email='zeidk@umd.edu',
    description='The five controller input topics, live from CARLA (ENPM818Z L10).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'l10_topics = l10_topics_demo.topics_node:main',
        ],
    },
)
