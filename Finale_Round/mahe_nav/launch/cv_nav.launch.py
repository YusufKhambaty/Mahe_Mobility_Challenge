"""
cv_nav.launch.py
================
Standalone CV-only launch file — launches ArUco detector, sign detector,
and status logger without the navigation controller or LiDAR analyzer.

Use this to test and debug the computer vision pipeline in isolation
on real hardware (Intel RealSense camera).

Prerequisites:
    The RealSense camera driver must be running:
        ros2 launch realsense2_camera rs_launch.py
    The BNO055 IMU driver must be running:
        ros2 launch <imu_package> <imu_launch>.py

Usage:
    ros2 launch mahe_nav cv_nav.launch.py
    ros2 launch mahe_nav cv_nav.launch.py use_display:=false
    ros2 launch mahe_nav cv_nav.launch.py log_level:=debug use_image_view:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # ─── Launch Arguments ───
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error)'
    )
    
    use_display_arg = DeclareLaunchArgument(
        'use_display',
        default_value='true',
        description='If true, enables cv2.imshow natively in the ArUco detector node'
    )
    
    use_image_view_arg = DeclareLaunchArgument(
        'use_image_view',
        default_value='true',
        description='If true, launches an rqt_image_view node subscribing to /aruco/debug_image'
    )

    # ─── Configurations ───
    log_level = LaunchConfiguration('log_level')
    use_display = LaunchConfiguration('use_display')
    use_image_view = LaunchConfiguration('use_image_view')

    # ─── ArUco Detector ───
    # Subscribes: /camera/camera/color/image_raw, /camera/camera/color/camera_info,
    #             /imu/data, /r1a004/wheel_odom
    # Publishes:  /aruco/detections, /aruco/pose_correction, /aruco/debug_image
    aruco_detector_node = Node(
        package='mahe_nav',
        executable='aruco_detector',
        name='aruco_detector_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_display': use_display
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ─── Sign Detector (Tile Counter / Floor Marker) ───
    # Subscribes: /camera/camera/color/image_raw, /imu/data, /aruco/detections
    # Publishes:  /floor_marker/detection
    sign_detector_node = Node(
        package='mahe_nav',
        executable='sign_detector',
        name='sign_detector_node',
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ─── Status Logger ───
    # Subscribes: /r1a004/wheel_odom, /imu/data, /aruco/detections,
    #             /floor_marker/detection, /lidar/analysis
    # Publishes:  /mission_status
    status_logger_node = Node(
        package='mahe_nav',
        executable='status_logger',
        name='status_logger_node',
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ─── Optional External Debug Viewer ───
    # Requires the ArUco detector to publish to /aruco/debug_image
    debug_viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='aruco_debug_viewer',
        arguments=['/aruco/debug_image'],
        condition=IfCondition(use_image_view),
    )

    return LaunchDescription([
        log_level_arg,
        use_display_arg,
        use_image_view_arg,
        
        aruco_detector_node,
        sign_detector_node,
        status_logger_node,
        debug_viewer_node,
    ])
