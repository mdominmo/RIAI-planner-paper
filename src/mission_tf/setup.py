from setuptools import find_packages, setup

package_name = 'mission_tf'

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
    description="mission_tf",
    license='private',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_tf_node = mission_tf.mission_tf_node:main', 
        ],
    },
)
