from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manuel Domínguez Montero',
    maintainer_email='mandominguez97@gmail.com',
    description='Cooperative perception: on-board ArUco detection and '
                'georeferencing (PnP) and global fusion of the estimates.',
    license='private',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector_node = perception.aruco_detector_node:main',
            'fusion_node = perception.fusion_node:main',
        ],
    },
)
