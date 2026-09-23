"""Launch one automated controller-comparison run (hardware + OptiTrack).

Brings up OptiTrack + the crazybridge (loading the config's ``pid.conf``) + the
rerun viewer (live planned-vs-actual overlay) + the ``controller_test`` node,
which flies the fixed sequence, records this config's results into
``<output_dir>/<config_name>/`` and then exits.

``config_name`` drives three things: the run folder under ``output_dir``, the
gains recorded inside it, and the rerun application id (``crazybridge/<name>``)
so the run is easy to find in the live viewer.

Drive one run per config, e.g.::

    ros2 launch crazybridge controller_test.launch.py \
        config_name:=gains_v1 \
        pid_conf_path:=/abs/path/to/pid.conf.v1 \
        output_dir:=/abs/path/to/results

The rerun viewer can live on a different machine than the one flying the drone:
run ``rerun`` on your laptop and point this launch at it with
``rerun_mode:=connect rerun_addr:=rerun+http://<laptop-ip>:9876/proxy``.

For a no-risk dry run in CrazySim, override ``uri:=udp://127.0.0.1:19850`` and
drop OptiTrack (``use_optitrack:=false``).
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            'config_name', default_value='default',
            description='Label for this run; names the output folder and the '
                        'rerun recording.'),
        DeclareLaunchArgument(
            'tracking_mode', default_value='marker',
            description="Mocap tracking mode: 'rigid_body' (6-DoF), 'marker' (single-ball 3-DoF), or 'auto'."),
        DeclareLaunchArgument(
            'pid_conf_path', default_value='',
            description='OOT gain file to load for this run (empty = auto-resolve based on tracking_mode).'),
        DeclareLaunchArgument(
            'output_dir', default_value='/tmp/ctrl_test',
            description='Parent directory; this run writes into '
                        '<output_dir>/<config_name>/.'),
        DeclareLaunchArgument(
            'uri', default_value='radio://0/100/2M',
            description='cflib URI (hardware default; sim = udp://127.0.0.1:19850).'),
        DeclareLaunchArgument(
            'use_optitrack', default_value='true',
            description='Launch the OptiTrack client (set false for sim).'),
        DeclareLaunchArgument(
            'rerun_mode', default_value='spawn',
            description="rerun attach mode: spawn|connect|save|disabled. Use "
                        "'connect' with rerun_addr for a viewer on another host."),
        DeclareLaunchArgument(
            'rerun_addr', default_value='',
            description='gRPC address of a remote rerun viewer for '
                        'rerun_mode=connect, e.g. rerun+http://10.43.100.150:9876/proxy '
                        '(empty = local default).'),
        DeclareLaunchArgument(
            'rerun_save_path', default_value='',
            description='Output .rrd path when rerun_mode=save.'),
        # Trajectory geometry / phase durations (kept identical across configs).
        DeclareLaunchArgument('takeoff_height_m', default_value='1.0'),
        DeclareLaunchArgument('goal_xyz', default_value='[0.5, 0.0, 1.0]'),
        DeclareLaunchArgument('circle_radius_m', default_value='0.5'),
        DeclareLaunchArgument('takeoff_duration_s', default_value='3.0'),
        DeclareLaunchArgument('settle_s', default_value='2.0'),
        DeclareLaunchArgument('goto_duration_s', default_value='3.0'),
        DeclareLaunchArgument('regulate_s', default_value='3.0'),
        DeclareLaunchArgument('circle_duration_s', default_value='8.0'),
        DeclareLaunchArgument('circle_settle_s', default_value='2.0'),
        DeclareLaunchArgument('land_duration_s', default_value='3.0'),
    ]

    bridge = Node(
        package='crazybridge', executable='crazybridge', name='crazybridge',
        output='screen',
        parameters=[{
            'uri': LaunchConfiguration('uri'),
            'tracking_mode': LaunchConfiguration('tracking_mode'),
            'pid_conf_path': LaunchConfiguration('pid_conf_path'),
            'load_pid_conf': True,
        }],
    )

    rerun = Node(
        package='crazybridge', executable='rerun', name='rerun',
        output='screen',
        parameters=[{
            'rerun_mode': LaunchConfiguration('rerun_mode'),
            'rerun_addr': LaunchConfiguration('rerun_addr'),
            'rerun_save_path': LaunchConfiguration('rerun_save_path'),
            # Names the recording after the config, so the run is easy to pick
            # out in the live viewer.
            'run_name': LaunchConfiguration('config_name'),
        }],
    )

    optitrack = Node(
        package='optitrack_client', executable='optitrack_client',
        name='optitrack', output='screen',
        condition=IfCondition(LaunchConfiguration('use_optitrack')),
    )

    test = Node(
        package='crazybridge', executable='controller_test',
        name='controller_test', output='screen',
        parameters=[{
            'config_name': LaunchConfiguration('config_name'),
            'output_dir': LaunchConfiguration('output_dir'),
            # Same gain file the bridge loads; recorded with the metrics.
            'pid_conf_path': LaunchConfiguration('pid_conf_path'),
            'takeoff_height_m': LaunchConfiguration('takeoff_height_m'),
            'goal_xyz': LaunchConfiguration('goal_xyz'),
            'circle_radius_m': LaunchConfiguration('circle_radius_m'),
            'takeoff_duration_s': LaunchConfiguration('takeoff_duration_s'),
            'settle_s': LaunchConfiguration('settle_s'),
            'goto_duration_s': LaunchConfiguration('goto_duration_s'),
            'regulate_s': LaunchConfiguration('regulate_s'),
            'circle_duration_s': LaunchConfiguration('circle_duration_s'),
            'circle_settle_s': LaunchConfiguration('circle_settle_s'),
            'land_duration_s': LaunchConfiguration('land_duration_s'),
        }],
    )

    return LaunchDescription(args + [bridge, rerun, optitrack, test])
