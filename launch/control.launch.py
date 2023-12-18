from launch_ros.actions import Node, PushRosNamespace

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    launch_description = LaunchDescription()
    arg = DeclareLaunchArgument('vehicle_name')
    launch_description.add_action(arg)

    package_path = get_package_share_path('position_control')
    position_controller_params_file_path = str(package_path / 'config/position_controller_params.yaml')

    group = GroupAction([
        PushRosNamespace(LaunchConfiguration('vehicle_name')),
        Node(executable='yaw_controller.py', package='position_control'),
        Node(
            executable='position_controller.py',
            package='position_control',
            parameters=[
                LaunchConfiguration(
                    'position_controller_params_file_path',
                    default=position_controller_params_file_path)
                ]
            )
    ])
    launch_description.add_action(group)
    return launch_description
