from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fruit_act_anotator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryota',
    maintainer_email='ryota@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'annotator_node = fruit_act_anotator.annotator_node:main',
            'detection_node = fruit_act_anotator.detection_node:main',
            'fruit_tracker_node = fruit_act_anotator.fruit_tracker_node:main',
        ],
    },
)