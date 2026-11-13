# MIT License

# Copyright (c) 2025 Manuel Domínguez

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def px4_gz_positioning_setup(context, *args, **kwargs):

    num_vehicles = int(LaunchConfiguration('num_vehicles').perform(context))

    actions = []
    for i in range(1, num_vehicles + 1):
        node = Node(
            package='px4_gz_positioning',
            executable='px4_gz_positioning_node',
            name=f'px4_gz_positioning_{i}',
            output='screen',
            parameters=[{
                'gz_topic': f'/model/x500_vision_{i}/pose',
                'px4_topic': f'/px4_{i}/fmu/in/vehicle_visual_odometry',
            }]
        )
        actions.append(node)
    return actions


def perception_setup(context, *args, **kwargs):
    """Cooperative perception: one on-board ArUco detector per UAV + global fusion."""

    num_vehicles = int(LaunchConfiguration('num_vehicles').perform(context))
    marker_length = float(LaunchConfiguration('marker_length').perform(context))
    marker_ids = [int(i) for i in LaunchConfiguration('marker_ids').perform(context).split(',') if i.strip()]

    actions = []
    for i in range(1, num_vehicles + 1):
        actions.append(Node(
            package='perception',
            executable='aruco_detector_node',
            name=f'aruco_detector_{i}',
            output='screen',
            parameters=[{
                'uav_id': i,
                'marker_length': marker_length,
                'marker_ids': marker_ids,
            }]
        ))
    actions.append(Node(
        package='perception',
        executable='fusion_node',
        name='fusion_node',
        output='screen',
    ))
    return actions


def generate_launch_description():

    num_vehicles_arg = DeclareLaunchArgument(
        'num_vehicles',
        default_value='1',
        description='Número de vehículos a lanzar'
    )
    plan_type_arg = DeclareLaunchArgument(
        'plan_type',
        default_value='1',  
        description='Tipo de plan de misión'
    )
    mission_radius_arg = DeclareLaunchArgument(
        'mission_radius',
        default_value='10.0',  
        description='Radio de percepción'
    )
    mission_height_arg = DeclareLaunchArgument(
        'mission_height',
        default_value='10.0',  
        description='Altura de misión'
    )
    spatial_tol_arg = DeclareLaunchArgument(
        'spatial_tol',
        default_value='0.45',  
        description='Spatial tolerance in RRT'
    )

    marker_length_arg = DeclareLaunchArgument(
        'marker_length',
        default_value='2.0',
        description='Side of the ArUco markers [m]'
    )
    marker_ids_arg = DeclareLaunchArgument(
        'marker_ids',
        default_value='0,1,2,3,4,5',
        description='Valid task IDs (other decoded IDs are false detections)'
    )

    ld = LaunchDescription()

    ld.add_action(num_vehicles_arg)
    ld.add_action(plan_type_arg)
    ld.add_action(mission_radius_arg)
    ld.add_action(mission_height_arg)
    ld.add_action(spatial_tol_arg)
    ld.add_action(marker_length_arg)
    ld.add_action(marker_ids_arg)
    ld.add_action(OpaqueFunction(function=px4_gz_positioning_setup))
    ld.add_action(OpaqueFunction(function=perception_setup))

    ld.add_action(Node(
        package='tracking',
        executable='tracking_node',
        name='tracking_1',
        output='screen'
    ))

    ld.add_action(Node(
    package='mission',
    executable='mission_node',
    name='mission_1',
    output='screen',
    parameters=[{
        'num_vehicles': LaunchConfiguration('num_vehicles'),
        'plan_type': LaunchConfiguration('plan_type'),
        'n_points': 500,
        'mission_frame': [108.28299713134766, -94.181564331054688, 15.0],
        'mission_radius': LaunchConfiguration('mission_radius'),
        'mission_height': LaunchConfiguration('mission_height'),
        'step_size': 1.0,
        'n_steps': 2000,
        'space_coef': 0.8,
        'time_coef': 0.2,
        'avg_speed': 1.0,
        'spatial_tol': LaunchConfiguration('spatial_tol'),
        'time_tol': 100.0,
        'cylinder_height': 2.0,
        'cylinder_radius': 0.5
        }]
    ))

    return ld
