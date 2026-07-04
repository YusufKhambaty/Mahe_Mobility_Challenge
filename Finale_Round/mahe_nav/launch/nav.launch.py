"""
nav.launch.py
=============
Launches the full MAHE UGV hardware navigation stack.

Prerequisites:
    The following drivers must already be running:
        ros2 launch realsense2_camera rs_launch.py   # RealSense camera
        # BNO055 IMU driver
        # RPLidar driver
        # R1A004 wheel encoder driver

Starts:
  1.  LiDAR analyzer    — raw /scan → structured /lidar/analysis
  2.  ArUco detector    — /camera/camera/color/image_raw → /aruco/detections
  3.  Sign detector     — /camera/camera/color/image_raw → /floor_marker/detection
  4.  Status logger     — aggregates all events, publishes /mission_status
  5.  Nav controller    — state machine, publishes /cmd_vel

Usage:
    ros2 launch mahe_nav nav.launch.py
    ros2 launch mahe_nav nav.launch.py log_level:=debug
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    log_level = LaunchConfiguration('log_level')
    use_display = LaunchConfiguration('use_display')
    use_image_view = LaunchConfiguration('use_image_view')

    # ── 1. LiDAR analyzer ────────────────────────────────────────────────────────
    # Subscribes: /scan (sensor_msgs/LaserScan)
    # Publishes:  /lidar/analysis (LidarAnalysis)
    lidar_node = Node(
        package    = 'mahe_nav',
        executable = 'lidar_analyzer',
        name       = 'lidar_analyzer',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 2. ArUco detector ────────────────────────────────────────────────────────
    # Subscribes: /camera/camera/color/image_raw, /camera/camera/color/camera_info,
    #             /imu/data, /r1a004/wheel_odom
    # Publishes:  /aruco/detections, /aruco/pose_correction, /aruco/debug_image
    aruco_node = Node(
        package    = 'mahe_nav',
        executable = 'aruco_detector',
        name       = 'aruco_detector_node',
        output     = 'screen',
        emulate_tty = True,
        parameters = [{
            'marker_size_m': 0.150,
            'use_display':   use_display,
        }],
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 3. Sign detector ─────────────────────────────────────────────────────────
    # Subscribes: /camera/camera/color/image_raw, /imu/data, /aruco/detections
    # Publishes:  /floor_marker/detection (FloorMarkerDetection)
    sign_node = Node(
        package    = 'mahe_nav',
        executable = 'sign_detector',
        name       = 'sign_detector_node',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 4. Status logger ─────────────────────────────────────────────────────────
    # Subscribes: /r1a004/wheel_odom, /imu/data, /aruco/detections,
    #             /floor_marker/detection, /lidar/analysis
    # Publishes:  /mission_status (std_msgs/String JSON)
    logger_node = Node(
        package    = 'mahe_nav',
        executable = 'status_logger',
        name       = 'status_logger_node',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 5. Navigation controller ─────────────────────────────────────────────────
    # Subscribes: /r1a004/wheel_odom, /imu/data, /lidar/analysis,
    #             /aruco/detections, /floor_marker/detection, /aruco/pose_correction
    # Publishes:  /cmd_vel (geometry_msgs/Twist)
    nav_node = Node(
        package    = 'mahe_nav',
        executable = 'nav_controller',
        name       = 'nav_controller_node',
        output     = 'screen',
        emulate_tty = True,
        arguments  = ['--ros-args', '--log-level', log_level],
    )

    # ── 6. Optional Debug Image Viewer ───────────────────────────────────────
    debug_viewer_node = Node(
        package    = 'rqt_image_view',
        executable = 'rqt_image_view',
        name       = 'aruco_debug_viewer',
        arguments  = ['/aruco/debug_image'],
        condition  = IfCondition(use_image_view),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level: debug, info, warn, error'
        ),
        DeclareLaunchArgument(
            'use_display',
            default_value='false',
            description='If true, ArUco node opens a native cv2.imshow window'
        ),
        DeclareLaunchArgument(
            'use_image_view',
            default_value='false',
            description='If true, launches rqt_image_view on /aruco/debug_image'
        ),
        lidar_node,
        aruco_node,
        sign_node,
        logger_node,
        nav_node,
        debug_viewer_node,
    ])
