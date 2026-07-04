#!/usr/bin/env python3
"""
ArUco Pose Correction Node — Hardware-Ready (RealSense Camera)
==============================================================
Detects ArUco markers via the Intel RealSense color stream, computes
6-DOF pose using IPPE_SQUARE dual-hypothesis testing, and publishes
pose-correction messages for the navigation controller.

Subscriptions:
  /camera/camera/color/image_raw   — RealSense RGB image
  /camera/camera/color/camera_info — RealSense camera intrinsics
  /imu/data                        — BNO055 fused orientation
  /r1a004/wheel_odom               — Wheel odometry (yaw fallback)

Publications:
  /aruco/detections      — ArucoDetection msg per confirmed marker
  /aruco/pose_correction — Pose2D when marker is within correction range
  /aruco/debug_image     — Annotated image for remote debug viewing
"""
import os
import time
import math
import numpy as np
import cv2
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose2D
from mahe_nav_interfaces.msg import ArucoDetection

# === MARKER POSITIONS ===
MARKER_WORLD_POS = {
    2: (1.796,  0.975),
    1: (0.450, -0.441),
    3: (1.320, -1.341),
    4: (1.796, -1.875),
    5: (-0.450, 1.341),
}
MARKER_ID_REMAP = {0: 2, 1: 1, 2: 3, 3: 4, 4: 5}

ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11


def normalize_quaternion(q):
    """Normalize quaternion to prevent math domain errors."""
    norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    if norm < 1e-6: return 0.0, 0.0, 0.0, 1.0
    return q.x/norm, q.y/norm, q.z/norm, q.w/norm

def quat_to_yaw(x, y, z, w):
    """Convert ROS quaternion to Euler Yaw (Z-axis rotation)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def circular_mean(angles_rad):
    """Compute average angle safely handling the -pi to pi wrap-around."""
    if not angles_rad: return 0.0
    sin_sum = sum(math.sin(a) for a in angles_rad)
    cos_sum = sum(math.cos(a) for a in angles_rad)
    return math.atan2(sin_sum, cos_sum)


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # --- Parameters ---
        self.declare_parameter('marker_size_m', 0.150)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('confirmation_frames', 3) 
        self.declare_parameter('camera_frame', 'CAM')
        self.declare_parameter('camera_yaw_offset', 0.0)
        
        self.declare_parameter('use_display', False)
        self.declare_parameter('use_imu_smoothing', True)
        self.declare_parameter('imu_smoothing_window', 5)
        self.declare_parameter('allow_odom_fallback', True)

        self.marker_size = self.get_parameter('marker_size_m').value
        self.conf_frames = self.get_parameter('confirmation_frames').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.yaw_offset = self.get_parameter('camera_yaw_offset').value
        
        # Robustly handle 'true' string from launch files
        viz_param = self.get_parameter('use_display').value
        self.use_display = (viz_param is True or str(viz_param).lower() == 'true')
        
        self.use_smoothing = self.get_parameter('use_imu_smoothing').value
        self.smooth_win_size = self.get_parameter('imu_smoothing_window').value
        self.allow_fallback = self.get_parameter('allow_odom_fallback').value

        # --- Subsystems Setup ---
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.seen_ids_global = set()
        self.sighting_counts = defaultdict(int) 
        self.last_seen_times = defaultdict(float)

        self.imu_yaw_buffer = deque(maxlen=self.smooth_win_size)
        self.latest_odom_yaw = 0.0

        self.saved_image_ids = set()
        self.log_dir = os.path.expanduser('~/aruco_pngs')
        os.makedirs(self.log_dir, exist_ok=True)
        log_file_path = os.path.join(self.log_dir, 'aruco_log.csv')
        write_header = not os.path.exists(log_file_path)
        self.log_file = open(log_file_path, 'a')
        if write_header:
            self.log_file.write("timestamp,marker_id\n")
            self.log_file.flush()

        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None

        # --- Object Points for IPPE SQUARE (Top-Left, Top-Right, Bottom-Right, Bottom-Left) ---
        half = self.marker_size / 2.0
        self.obj_pts = np.array([
            [-half,  half, 0], 
            [ half,  half, 0], 
            [ half, -half, 0], 
            [-half, -half, 0]
        ], dtype=np.float32)

        # --- Publishers & Subscribers ---
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.pub_detection = self.create_publisher(ArucoDetection, '/aruco/detections', 10)
        self.pub_pose_correction = self.create_publisher(Pose2D, '/aruco/pose_correction', 10)
        self.pub_debug_img = self.create_publisher(Image, '/aruco/debug_image', 1)

        self.create_subscription(Imu, '/imu/data', self._imu_cb, sensor_qos)
        self.create_subscription(Odometry, '/r1a004/wheel_odom', self._odom_cb, sensor_qos)
        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value, self._info_cb, sensor_qos)
        self.create_subscription(Image, self.get_parameter('camera_topic').value, self._image_cb, sensor_qos)

        self.get_logger().info(
            f'ArUco Pose Correction Node Online | '
            f'camera={self.get_parameter("camera_topic").value} | '
            f'display={self.use_display}'
        )

    def _get_robot_yaw(self):
        """Fetch robust yaw applying smoothing and fallbacks."""
        if len(self.imu_yaw_buffer) > 0:
            return circular_mean(self.imu_yaw_buffer) if self.use_smoothing else self.imu_yaw_buffer[-1]
        
        if self.allow_fallback:
            return self.latest_odom_yaw
        return 0.0

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        nx, ny, nz, nw = normalize_quaternion(q)
        self.latest_odom_yaw = quat_to_yaw(nx, ny, nz, nw)

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        # Ignore completely uninitialized matrices
        if q.x == 0 and q.y == 0 and q.z == 0 and q.w == 0: return
        # Ignore identity quaternions (typical of /imu/raw)
        if q.x == 0 and q.y == 0 and q.z == 0 and q.w == 1: return
        
        nx, ny, nz, nw = normalize_quaternion(q)
        yaw = quat_to_yaw(nx, ny, nz, nw)
        self.imu_yaw_buffer.append(yaw)

    def _info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def _image_cb(self, msg: Image):
        if self.camera_matrix is None: return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image decode error: {e}')
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        display_frame = cv_img.copy() if (self.use_display or self.pub_debug_img.get_subscription_count() > 0) else None

        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        now = self.get_clock().now().nanoseconds * 1e-9

        # Handle expiration of stale debounces
        for k in list(self.sighting_counts.keys()):
            if now - self.last_seen_times[k] > 2.0:  # Reset if not seen for 2 seconds
                self.sighting_counts[k] = 0

        if ids is not None:
            if display_frame is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

            for i, raw_marker_id in enumerate(ids.flatten()):
                raw_mid = int(raw_marker_id)
                mid = MARKER_ID_REMAP.get(raw_mid, raw_mid)
                
                self.sighting_counts[mid] += 1
                self.last_seen_times[mid] = now
                
                if self.sighting_counts[mid] < self.conf_frames:
                    continue

                # --- 1. IPPE_SQUARE Dual Hypothesis Testing ---
                success, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                    self.obj_pts, 
                    corners[i][0].astype(np.float32), 
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if not success or not rvecs: continue

                # Extract the hypothesis with the smallest reprojection error
                best_idx = np.argmin(np.array(reproj_errs).flatten())
                rvec = rvecs[best_idx].flatten()
                tvec = tvecs[best_idx].flatten()

                # --- 2. Math & Camera Frame Extraction ---
                distance = float(np.linalg.norm(tvec))
                bearing_rad = float(math.atan2(-tvec[0], tvec[2])) # Optical frame Z forward, X right

                pose_yaw = self._get_robot_yaw()

                # --- 3. Publishing the raw detection ---
                det_msg = ArucoDetection()
                det_msg.header = msg.header
                det_msg.header.frame_id = self.camera_frame
                det_msg.marker_id = mid
                det_msg.distance = distance
                det_msg.bearing_angle_rad = bearing_rad
                det_msg.position_camera = Point(x=float(tvec[0]), y=float(tvec[1]), z=float(tvec[2]))
                det_msg.tvec = [float(v) for v in tvec]
                det_msg.rvec = [float(v) for v in rvec]

                # First ever detection 
                det_msg.first_detection = mid not in self.seen_ids_global
                if det_msg.first_detection:
                    self.seen_ids_global.add(mid)
                    
                    # --- Logging and Image Capture ---
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    self.log_file.write(f"{current_time},{mid}\n")
                    self.log_file.flush()

                    if mid not in self.saved_image_ids:
                        self.saved_image_ids.add(mid)
                        # Crop the ArUco marker with padding
                        try:
                            pts = corners[i][0].astype(int)
                            x, y, w, h = cv2.boundingRect(pts)
                            img_h, img_w = cv_img.shape[:2]
                            pad = 30
                            x1 = max(0, x - pad)
                            y1 = max(0, y - pad)
                            x2 = min(img_w, x + w + pad)
                            y2 = min(img_h, y + h + pad)
                            aruco_crop = cv_img[y1:y2, x1:x2]
                            img_path = os.path.join(self.log_dir, f'aruco_{mid}.png')
                            if aruco_crop.size > 0:
                                ok = cv2.imwrite(img_path, aruco_crop)
                                if ok:
                                    self.get_logger().info(f'Saved ArUco {mid} image: {img_path}')
                                else:
                                    self.get_logger().error(f'cv2.imwrite FAILED for {img_path}')
                            else:
                                self.get_logger().warn(f'ArUco {mid} crop was empty, skipping save')
                        except Exception as e:
                            self.get_logger().error(f'ArUco {mid} image save error: {e}')

                    self.get_logger().info(
                        f'ArUco {mid} CONFIRMED | dist={distance:.2f}m | '
                        f'bearing={math.degrees(bearing_rad):.1f}° | '
                        f'Err={reproj_errs[best_idx].flatten()[0]:.2f}'
                    )

                self.pub_detection.publish(det_msg)

                # --- 4. World Frame Alignment Correction ---
                if mid in MARKER_WORLD_POS and distance <= 0.8:
                    mx, my = MARKER_WORLD_POS[mid]
                    
                    world_angle = pose_yaw + bearing_rad + self.yaw_offset
                    corrected_x = mx - distance * math.cos(world_angle)
                    corrected_y = my - distance * math.sin(world_angle)
                    
                    corr_msg = Pose2D()
                    corr_msg.x = corrected_x
                    corr_msg.y = corrected_y
                    corr_msg.theta = pose_yaw
                    self.pub_pose_correction.publish(corr_msg)

                if display_frame is not None:
                    center = corners[i][0].mean(axis=0).astype(int)
                    cv2.putText(display_frame, f"ID:{mid} D:{distance:.2f}m", (center[0], center[1]+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Err:{reproj_errs[best_idx].flatten()[0]:.2f}", (center[0], center[1]+40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # --- Safely handle displays ---
        if display_frame is not None:
            if self.pub_debug_img.get_subscription_count() > 0:
                self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(display_frame, "bgr8"))
            if self.use_display:
                cv2.imshow("Aruco Detector Feed", display_frame)
                cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'log_file'):
            node.log_file.close()
        if node.use_display: cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
