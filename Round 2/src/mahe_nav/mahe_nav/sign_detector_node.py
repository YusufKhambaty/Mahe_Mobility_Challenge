"""
MAHE UGV Navigation: Sign Detector Node (Phase 2 CV Floor Marker Tracking)
==========================================================================
Replaces the old sign detector logic with the validated computer vision pipeline
from cv_follower_node_corrected.py. Completely rewritten to process FloorMarkerDetection.

_pipeline() is a 1:1 clone of CvFollowerNode._image_cb() from cv_follower_node_corrected_3_.py,
with the Red Tile HALT Detection block inserted after Stage 2 (ROI crop), before Stage 3
(Blue Logo-Presence Gate).
"""

import math
import time
import collections
from collections import deque, Counter
import numpy as np
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from mahe_nav_interfaces.msg import ArucoDetection, FloorMarkerDetection


# ─────────────────────────────────────────────────────────────
#  VALIDATED CONSTANTS  (do not edit without re-running calibration)
# ─────────────────────────────────────────────────────────────

# ROI — percentage-based, floor-anchored
# top=0.20 sees most of the tile without letting in excessive wall clutter
ROI_TOP_FRACTION    = 0.20
ROI_BOTTOM_FRACTION = 1.00

# Blue logo-presence gates (Test 3 validated against 73-frame CSV)
BLUE_GATE_LOGO      = 3000   # px — logo is in frame; gate all detection on this
BLUE_GATE_APPROACH  = 5000   # px — optimal detection range has begun (informational)

# HSV thresholds — HEURISTIC-VALIDATED from live Gazebo camera frames (2026-04-18)
# Green  : Measured H=59, S=149-190, V=188-194 (padded for shadows)
HSV_GREEN_LO  = np.array([45,  80,  80],  dtype=np.uint8)
HSV_GREEN_HI  = np.array([85,  255, 255], dtype=np.uint8)
# Orange : Measured H=13, S=194, V=255 (padded for shadows)
HSV_ORANGE_LO = np.array([5,   100, 120], dtype=np.uint8)
HSV_ORANGE_HI = np.array([25,  255, 255], dtype=np.uint8)
# Blue   : logo centre — presence gate + centroid anchor + BLUE arrow mode
HSV_BLUE_LO   = np.array([100, 80,  80],  dtype=np.uint8)
HSV_BLUE_HI   = np.array([140, 255, 255], dtype=np.uint8)

# Morphology — removes single-pixel HSV noise after masking
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Contour / geometry sanity guards
MIN_PETAL_AREA   = 200    # px² — below this is noise
MIN_PETAL_VECTOR = 15.0   # px  — too close to logo centre (probably the disc)
MAX_PETAL_VECTOR = 120.0  # px  — outside tile boundary (noise)

# Frame gate — reject frames captured during active rotation
MAX_ANGULAR_VEL  = 0.20   # rad/s — 0.05 was too strict, blocked frames during PD corridor centering

# Majority vote parameters (architecture design)
VOTE_WINDOW      = 7      # rolling buffer length
VOTE_THRESHOLD   = 4      # min agreements to emit a direction (5/7 was too strict)
VOTE_TIMEOUT_SEC = 3.0    # seconds → emit DIRECTION_TIMEOUT, nav must retry

# ArUco tag IDs that control the CV state machine
# State: DORMANT → GREEN (tag 3) → DORMANT (tag 4) → ORANGE (tag 5)
TAG_ID_GREEN   = 3    # Activates GREEN tracking
TAG_ID_DORMANT = 4    # Deactivates CV (go silent between green and orange lines)
TAG_ID_ORANGE  = 5    # Activates ORANGE tracking
TAG_DIST_GATE  = 3.0  # metres — tag must be within this distance to trigger mode switch

# === PHASE 2 CV: Red Tile HALT Detection ===
RED_HSV_LOWER  = np.array([0,   120, 120])
RED_HSV_UPPER  = np.array([10,  255, 255])
RED_HSV_LOWER2 = np.array([170, 120, 120])
RED_HSV_UPPER2 = np.array([180, 255, 255])

# Direction bins — world angle is CCW from +X (standard ROS/math convention)
# Arena start: robot faces +Y (north). At yaw=0 robot faces east (+X).
DIRECTION_BINS = [
    ((-22.5,   22.5),  "RIGHT"),
    (( 22.5,   67.5),  "FORWARD_RIGHT"),
    (( 67.5,  112.5),  "FORWARD"),
    ((112.5,  157.5),  "FORWARD_LEFT"),
    ((157.5,  180.1),  "LEFT"),        # 180.1 closes the edge-case at exactly ±180°
    ((-180.1, -157.5), "LEFT"),
    ((-157.5, -112.5), "BACKWARD_LEFT"),
    ((-112.5,  -67.5), "BACKWARD"),
    (( -67.5,  -22.5), "BACKWARD_RIGHT"),
]

COLLAPSE_TO_4 = {
    "FORWARD":        "FORWARD",
    "FORWARD_RIGHT":  "FORWARD",
    "FORWARD_LEFT":   "FORWARD",
    "RIGHT":          "RIGHT",
    "BACKWARD_RIGHT": "BACKWARD",
    "BACKWARD":       "BACKWARD",
    "BACKWARD_LEFT":  "BACKWARD",
    "LEFT":           "LEFT",
}


# ─────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def normalise_angle(a: float) -> float:
    """Normalise angle to (-180, 180]. Handles ±180 edge correctly."""
    a = a % 360.0          # 0 to 360
    if a > 180.0:
        a -= 360.0         # -180 to 180
    return a


def angle_to_direction_8(angle_deg: float) -> str:
    """Map a world angle (degrees, CCW from +X) to an 8-bin direction string."""
    a = normalise_angle(angle_deg)
    for (lo, hi), label in DIRECTION_BINS:
        if lo <= a < hi:
            return label
    return "UNKNOWN"


def angle_to_direction_4(angle_deg: float) -> str:
    return COLLAPSE_TO_4.get(angle_to_direction_8(angle_deg), "UNKNOWN")


def quat_to_yaw(q) -> float:
    """Extract yaw (radians) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────

class SignDetectorNode(Node):

    def __init__(self):
        super().__init__('sign_detector')

        self.bridge = CvBridge()

        # ── State ──────────────────────────────────────────────────────────
        # cv_mode starts DORMANT — activates GREEN after ArUco 3, ORANGE after ArUco 5.
        # State machine: DORMANT → GREEN (tag3) → DORMANT (tag4) → ORANGE (tag5)
        self.cv_mode         = "DORMANT"
        self.robot_yaw_rad   = 0.0
        self.angular_vel     = 0.0
        self.tile_count      = 0
        self.vote_buffer     = collections.deque(maxlen=VOTE_WINDOW)
        self.vote_start_time = None
        self.last_direction  = "NONE"

        # ── QoS ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            '/r1_mini/camera/image_raw',
            self._image_cb,
            qos_profile=sensor_qos
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom_fused',
            self._odom_cb,
            qos_profile=sensor_qos
        )
        self.sub_aruco = self.create_subscription(
            ArucoDetection,
            '/aruco/detections',
            self._aruco_cb,
            qos_profile=sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_detection = self.create_publisher(
            FloorMarkerDetection,
            '/floor_marker/detection',
            10
        )

        self.get_logger().info("Sign Detector (CV Floor Marker Mode) Active")

    # ──────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ──────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Cache robot yaw and angular velocity from fused odometry."""
        self.robot_yaw_rad = quat_to_yaw(msg.pose.pose.orientation)
        self.angular_vel   = abs(msg.twist.twist.angular.z)

    def _aruco_cb(self, msg: ArucoDetection):
        """
        CV state machine controlled by ArUco proximity.
        DORMANT → GREEN (tag 3) → DORMANT (tag 4) → ORANGE (tag 5)
        """
        if msg.distance > TAG_DIST_GATE:
            return

        tag_id = msg.marker_id

        # Tag 3: DORMANT → GREEN (activate green floor marker tracking)
        if tag_id == TAG_ID_GREEN and self.cv_mode == "DORMANT":
            self.cv_mode = "GREEN"
            self._reset_vote_buffer()
            self.get_logger().info(
                f"CV mode switched to GREEN (Tag {TAG_ID_GREEN} at {msg.distance:.2f}m)"
            )

        # Tag 4: GREEN → DORMANT (deactivate CV between green and orange lines)
        elif tag_id == TAG_ID_DORMANT and self.cv_mode == "GREEN":
            self.cv_mode = "DORMANT"
            self._reset_vote_buffer()
            self.get_logger().info(
                f"CV mode switched to DORMANT (Tag {TAG_ID_DORMANT} at {msg.distance:.2f}m)"
            )

        # Tag 5: DORMANT → ORANGE (activate orange tracking)
        elif tag_id == TAG_ID_ORANGE and self.cv_mode == "DORMANT":
            self.cv_mode = "ORANGE"
            self._reset_vote_buffer()
            self.get_logger().info(
                f"CV mode switched to ORANGE (Tag {TAG_ID_ORANGE} at {msg.distance:.2f}m)"
            )

    def _image_cb(self, msg: Image):
        """Entry point — delegates to _pipeline which contains the full CV logic."""
        self._pipeline(msg)

    # ──────────────────────────────────────────────────────────────────────
    #  PIPELINE  (1:1 clone of CvFollowerNode._image_cb from
    #             cv_follower_node_corrected_3_.py, with Red Tile HALT block
    #             inserted after Stage 2, before Stage 3)
    # ──────────────────────────────────────────────────────────────────────

    def _pipeline(self, msg: Image):
        """
        Main CV pipeline. Called on every camera frame.

        Stage 1 : Frame gate   — reject turns, enforce angular vel limit
        Stage 2 : ROI crop     — percentage-based, floor-anchored
        [RED]   : Red Tile HALT Detection — independent of cv_mode
        Stage 3 : Blue gate    — confirm logo is in frame before any colour work
        Stage 4 : Colour mask  — green / orange / blue depending on mode
        Stage 5 : Direction    — centroid vector → world angle
        Stage 6 : Vote + emit  — majority vote with timeout
        """

        # ── STAGE 1: Frame Gate ───────────────────────────────────────────
        # Reject frames during rotation — logo sweeps out of frame,
        # producing the 8/73 all-zero blind frames seen in CSV.
        if self.angular_vel > MAX_ANGULAR_VEL:
            return

        # ── Convert ───────────────────────────────────────────────────────
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ── STAGE 2: ROI Crop ─────────────────────────────────────────────
        # top=20% excludes ceiling/wall clutter while keeping tile petals visible.
        # bottom=100% recovers Y=441-479 band cut by old fixed ROI.
        roi_top = int(h * ROI_TOP_FRACTION)
        roi_hsv = hsv[roi_top:h, 0:w]

        # ── Red Tile HALT Detection ───────────────────────────────────────
        # Run independently of cv_mode — checks for massive red floor region
        red_mask  = cv2.inRange(roi_hsv, RED_HSV_LOWER,  RED_HSV_UPPER)
        red_mask2 = cv2.inRange(roi_hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
        red_mask  = cv2.bitwise_or(red_mask, red_mask2)
        red_px    = int(np.count_nonzero(red_mask))

        if red_px > 5000:
            halt_msg = FloorMarkerDetection()
            halt_msg.header.stamp = self.get_clock().now().to_msg()
            halt_msg.colour    = "RED"
            halt_msg.direction = "HALT"
            halt_msg.confidence = 1.0
            halt_msg.blue_px   = 0
            halt_msg.petal_px  = red_px
            self.pub_detection.publish(halt_msg)
            self.get_logger().info(f'[CV] RED TILE DETECTED — red_px={red_px} — publishing HALT')
            return   # skip rest of pipeline this frame

        # ── DORMANT GUARD ─────────────────────────────────────────────────
        # Skip colour pipeline when DORMANT (between green and orange sections).
        # Red tile detection above always runs regardless of mode.
        if self.cv_mode == "DORMANT":
            return

        # ── STAGE 3: Blue Logo-Presence Gate ─────────────────────────────
        # Blue is the most stable logo indicator (dominant pixel mass).
        # 3000px threshold separates logo-present from logo-absent across all 73 frames.
        blue_mask = cv2.inRange(roi_hsv, HSV_BLUE_LO, HSV_BLUE_HI)
        blue_px   = int(cv2.countNonZero(blue_mask))

        if blue_px < BLUE_GATE_LOGO:
            return  # Logo not in frame — skip, do NOT feed vote buffer

        # Log approach window opening (informational, does not gate anything)
        if blue_px >= BLUE_GATE_APPROACH:
            pass  # optimal detection range: x ≈ 1.5-2.5 m per CSV data

        # Find blue blob centroid — used as tile centre anchor for direction
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not blue_contours:
            return

        largest_blue = max(blue_contours, key=cv2.contourArea)
        bm = cv2.moments(largest_blue)
        if bm["m00"] == 0:
            return

        logo_cx = bm["m10"] / bm["m00"]
        logo_cy = bm["m01"] / bm["m00"]

        # ── STAGE 4: Active Colour Extraction ─────────────────────────────
        active_colour = self.cv_mode  # GREEN | ORANGE

        if active_colour == "GREEN":
            colour_lo, colour_hi = HSV_GREEN_LO,  HSV_GREEN_HI
        elif active_colour == "ORANGE":
            colour_lo, colour_hi = HSV_ORANGE_LO, HSV_ORANGE_HI
        else:
            return

        colour_mask = cv2.inRange(roi_hsv, colour_lo, colour_hi)

        # Morphological opening — erode then dilate to remove single-pixel HSV noise
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        petal_px = int(cv2.countNonZero(colour_mask))
        
        if active_colour == "ORANGE":
            self.get_logger().info(f"[CV ORANGE DEBUG] petal_px={petal_px}, red_px={red_px}, blue_px={blue_px}")

        if petal_px < MIN_PETAL_AREA:
            # Petal not visible — PETAL_NOT_VISIBLE, do not feed vote buffer
            return

        petal_contours, _ = cv2.findContours(
            colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not petal_contours:
            return

        largest_petal = max(petal_contours, key=cv2.contourArea)
        pm = cv2.moments(largest_petal)
        if pm["m00"] == 0:
            return

        petal_cx = pm["m10"] / pm["m00"]
        petal_cy = pm["m01"] / pm["m00"]

        # ── STAGE 5: Direction Computation ───────────────────────────────
        # Both logo and petal centroids are in ROI-space coordinates.
        # dx/dy are relative — coordinate offset doesn't matter.
        dx = petal_cx - logo_cx
        dy = petal_cy - logo_cy   # positive = petal is BELOW logo in image = south

        vector_mag = math.sqrt(dx * dx + dy * dy)

        # Geometric sanity: reject implausible vectors
        # < 15px → petal centroid inside the central disc (noise)
        # > 120px → petal is outside the tile boundary (noise)
        if not (MIN_PETAL_VECTOR <= vector_mag <= MAX_PETAL_VECTOR):
            return

        # Negate dy to convert image Y-down → standard math Y-up convention.
        # atan2(-dy, dx) then gives CCW-from-+X angle in standard math coords.
        # IMPORTANT: validated formula in calibration report wrote atan2(dy,dx)
        # without negation — verify final direction on one known-orientation tile.
        raw_angle_rad = math.atan2(-dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)

        # Add robot yaw to convert camera-relative angle → world angle.
        # robot_yaw_rad is CCW-positive from +X (standard ROS convention from /odom_fused).
        robot_yaw_deg   = math.degrees(self.robot_yaw_rad)
        world_angle_deg = normalise_angle(raw_angle_deg + robot_yaw_deg)

        direction_4 = angle_to_direction_4(world_angle_deg)
        if direction_4 == "UNKNOWN":
            return

        # ── STAGE 6: Majority Vote + Emit ─────────────────────────────────
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.vote_start_time is None:
            self.vote_start_time = now

        self.vote_buffer.append(direction_4)

        vote_counts = collections.Counter(self.vote_buffer)
        top_dir, top_count = vote_counts.most_common(1)[0]

        if top_count >= VOTE_THRESHOLD:
            # Strong consensus — emit direction
            self.tile_count += 1    # 1-indexed: first emit → tile 1
            confidence = top_count / VOTE_WINDOW
            self._emit_detection(
                direction   = top_dir,
                colour      = active_colour,
                confidence  = confidence,
                petal_cx    = petal_cx,
                petal_cy    = petal_cy,
                logo_cx     = logo_cx,
                logo_cy     = logo_cy,
                raw_angle   = raw_angle_deg,
                world_angle = world_angle_deg,
                blue_px     = blue_px,
                petal_px    = petal_px,
                header      = msg.header
            )
            self.get_logger().info(
                f"[TILE {self.tile_count}] {active_colour} → {top_dir} "
                f"(conf={confidence:.2f}, world_angle={world_angle_deg:.1f}°, "
                f"blue={blue_px}px, petal={petal_px}px)"
            )
            self._reset_vote_buffer()
            return

        # Timeout — buffer stalled, no consensus reached
        elapsed = now - self.vote_start_time
        if elapsed >= VOTE_TIMEOUT_SEC:
            self.get_logger().warn(
                f"DIRECTION_TIMEOUT after {elapsed:.1f}s — "
                f"best: {top_dir} ({top_count}/{VOTE_WINDOW}). Nav must retry."
            )
            self._emit_detection(
                direction   = "TIMEOUT",
                colour      = active_colour,
                confidence  = top_count / VOTE_WINDOW,
                petal_cx    = petal_cx,
                petal_cy    = petal_cy,
                logo_cx     = logo_cx,
                logo_cy     = logo_cy,
                raw_angle   = raw_angle_deg,
                world_angle = world_angle_deg,
                blue_px     = blue_px,
                petal_px    = petal_px,
                header      = msg.header
            )
            self._reset_vote_buffer()

    # ──────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _reset_vote_buffer(self):
        self.vote_buffer.clear()
        self.vote_start_time = None

    def _emit_detection(self, direction, colour, confidence,
                        petal_cx, petal_cy, logo_cx, logo_cy,
                        raw_angle, world_angle, blue_px, petal_px,
                        header):
        msg = FloorMarkerDetection()
        msg.header          = header
        msg.colour          = colour
        msg.direction       = direction
        msg.confidence      = float(confidence)
        msg.tile_count      = self.tile_count   # 1-indexed after increment above
        msg.petal_cx        = float(petal_cx)
        msg.petal_cy        = float(petal_cy)
        msg.logo_cx         = float(logo_cx)
        msg.logo_cy         = float(logo_cy)
        msg.raw_angle_deg   = float(raw_angle)
        msg.world_angle_deg = float(world_angle)
        msg.blue_px         = blue_px
        msg.petal_px        = petal_px
        self.pub_detection.publish(msg)



# ──────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('SystemExit raised. Graceful Stop.')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()