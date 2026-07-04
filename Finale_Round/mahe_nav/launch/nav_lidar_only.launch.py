"""
nav_lidar_only.launch.py
========================
Minimal launch for PURE LiDAR obstacle avoidance testing on hardware.

Launches ONLY:
  1. LiDAR analyzer    — raw /scan → /lidar/analysis
  2. Nav controller    — pure LiDAR state machine → /cmd_vel
  3. LiDAR visualiser  — real-time bird's-eye OpenCV popup

Camera nodes (aruco_detector, sign_detector, status_logger) are intentionally
NOT started. The nav_controller also has camera callbacks disabled internally.

Prerequisites:
    RPLidar driver must be running (publishing /scan).
    Wheel encoder driver must be running (publishing /r1a004/wheel_odom).

Usage:
    ros2 launch mahe_nav nav_lidar_only.launch.py
    ros2 launch mahe_nav nav_lidar_only.launch.py log_level:=debug
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level = LaunchConfiguration('log_level')

    # ── 1. LiDAR analyzer ────────────────────────────────────────────────────────
    lidar_node = Node(
        package    = 'mahe_nav',
        executable = 'lidar_analyzer',
        name       = 'lidar_analyzer',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 2. Nav controller ────────────────────────────────────────────────────────
    nav_node = Node(
        package    = 'mahe_nav',
        executable = 'nav_controller',
        name       = 'nav_controller_node',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 3. LiDAR visualiser ──────────────────────────────────────────────────────
    viz_node = Node(
        package    = 'mahe_nav',
        executable = 'lidar_viz',
        name       = 'lidar_viz',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_level',
            default_value = 'info',
            description   = 'Logging level: debug, info, warn, error'
        ),
        lidar_node,
        nav_node,
        viz_node,
    ])
