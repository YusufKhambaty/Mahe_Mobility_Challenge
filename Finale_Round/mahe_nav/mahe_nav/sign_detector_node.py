#!/usr/bin/env python3
"""
tile_counter.py — Floor Tile Colour Counter
============================================
Counts green and orange floor tiles as the robot drives through the arena.

Pipeline:
  - ROI crop (bottom 45%): eliminates wall logos geometrically
  - Blue pixel gate: confirms robot is over a logo tile
  - Green vs Orange pixel dominance: determines tile colour
  - 6-frame vote (5/6 threshold): confirms stable detection
  - Post-detection lockout: prevents re-counting the same tile
  - Red tile detection (two-shot): triggers HALT
  - Pending-mode logic: detection only activates after turn settles

Mode switching:
  Marker 3 → GREEN mode armed (activates after turn stops)
  Marker 4 → DORMANT
  Marker 5 → ORANGE mode armed (activates after turn stops)
"""

import os
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from mahe_nav_interfaces.msg import ArucoDetection, FloorMarkerDetection

# ── HSV Thresholds (calibrated from real arena images) ─────────────────────────
BLUE_LO        = np.array([ 98,  47,  54], dtype=np.uint8)
BLUE_HI        = np.array([119, 132, 111], dtype=np.uint8)

GREEN_LO       = np.array([ 64,  55,  77], dtype=np.uint8)
GREEN_HI       = np.array([ 81, 130, 119], dtype=np.uint8)

ORANGE_LO1     = np.array([  0,  80,  92], dtype=np.uint8)
ORANGE_HI1     = np.array([ 15, 202, 173], dtype=np.uint8)
ORANGE_LO2     = np.array([173,  60,  92], dtype=np.uint8)
ORANGE_HI2     = np.array([179, 180, 173], dtype=np.uint8)

RED_LO1        = np.array([  0, 150, 100], dtype=np.uint8)
RED_HI1        = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LO2        = np.array([170, 150, 100], dtype=np.uint8)
RED_HI2        = np.array([180, 255, 255], dtype=np.uint8)

# ── Detection Constants ─────────────────────────────────────────────────────────
ROI_FRACTION       = 0.55    # process bottom 45% — excludes wall logos
BLUE_GATE_PX       = 3000    # min blue pixels to confirm camera is over a tile
COLOUR_RATIO       = 1.25    # winning colour must beat other by this factor
MAX_ANGULAR_VEL    = 0.20    # rad/s — skip frame if robot is turning
RED_PX_THRESHOLD   = 8000    # red pixels above this = HALT zone

VOTE_FRAMES        = 6       # frames in voting window
VOTE_REQUIRED      = 5       # frames that must agree to confirm

SETTLE_FRAMES      = 20      # consecutive still frames before pending mode activates
SETTLE_VEL_THR     = 0.05    # rad/s — threshold for "robot has stopped turning"
LOCKOUT_FRAMES     = 45      # frames to ignore after a confirmed detection (~1.5s at 30Hz)
MORPH_KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# ── Colour Modes ───────────────────────────────────────────────────────────────
MODE_DORMANT = 'DORMANT'
MODE_GREEN   = 'GREEN'
MODE_ORANGE  = 'ORANGE'


class SignDetectorNode(Node):

    def __init__(self):
        super().__init__('sign_detector_node')
        self.bridge = CvBridge()

        # ── State ──────────────────────────────────────────────────────────────
        self.colour_mode       = MODE_DORMANT
        self.pending_mode      = None        # armed but waiting for turn to settle
        self.settle_counter    = 0           # frames of low angular vel since arming

        self.angular_vel       = 0.0
        self.green_count       = 0
        self.orange_count      = 0

        self.vote_buffer       = deque(maxlen=VOTE_FRAMES)
        self.lockout_remaining = 0           # frames remaining in post-detection lockout

        # Red HALT two-shot state
        self.red_seen_before   = False
        self.halt_triggered    = False

        self.declare_parameter('use_display', False)
        self.use_display = self.get_parameter('use_display').value

        # ── CSV Logging ────────────────────────────────────────────────────────
        self.log_file = None
        log_path = os.path.expanduser("~/TILELOG/tile_count_log2.csv")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
            self.log_file = open(log_path, "a")
            if write_header:
                self.log_file.write("timestamp,mode,colour,tile_num,green_px,orange_px,blue_px\n")
                self.log_file.flush()
        except Exception as e:
            self.get_logger().error(f"Log file failed: {e}")

        # ── QoS ────────────────────────────────────────────────────────────────
        sensor_qos   = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  durability=DurabilityPolicy.VOLATILE, depth=1)
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE, depth=10)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Image,          '/camera/camera/color/image_raw', self._image_cb,  sensor_qos)
        self.create_subscription(Imu,            '/imu/data',                      self._imu_cb,    sensor_qos)
        self.create_subscription(ArucoDetection, '/aruco/detections',              self._aruco_cb,  reliable_qos)

        # ── Publisher ──────────────────────────────────────────────────────────
        self.det_pub = self.create_publisher(FloorMarkerDetection,
                                             '/floor_marker/detection', reliable_qos)

        self.get_logger().info('TileCounterNode started — MODE: DORMANT')
        self.get_logger().info(f'Logs: {log_path}')

    # ── IMU Callback ───────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        self.angular_vel = abs(msg.angular_velocity.z)

        # Check if a pending mode can now activate (turn has settled)
        if self.pending_mode is not None:
            if self.angular_vel < SETTLE_VEL_THR:
                self.settle_counter += 1
            else:
                self.settle_counter = 0   # robot still turning — reset wait

            if self.settle_counter >= SETTLE_FRAMES:
                self.colour_mode    = self.pending_mode
                self.pending_mode   = None
                self.settle_counter = 0
                self.vote_buffer.clear()
                self.lockout_remaining = 0
                self.get_logger().info(f'MODE ACTIVATED → {self.colour_mode} (turn settled)')

    # ── ArUco Callback ─────────────────────────────────────────────────────────

    def _aruco_cb(self, msg: ArucoDetection):
        mid = msg.marker_id

        if mid == 3:
            # Arm GREEN — activates after turn settles
            self.pending_mode   = MODE_GREEN
            self.settle_counter = 0
            self.get_logger().info('Marker 3 → GREEN armed (waiting for turn to settle)')

        elif mid == 4:
            # Arm DORMANT — only stops green counting after U-turn fully settles
            self.pending_mode   = MODE_DORMANT
            self.settle_counter = 0
            self.get_logger().info('Marker 4 → DORMANT armed (waiting for U-turn to settle)')

        elif mid == 5:
            # Arm ORANGE — activates after turn settles
            self.pending_mode   = MODE_ORANGE
            self.settle_counter = 0
            self.get_logger().info('Marker 5 → ORANGE armed (waiting for turn to settle)')

    # ── Image Callback ─────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Bridge error: {e}')
            return

        h, w = frame.shape[:2]
        disp  = frame.copy()

        # ROI: bottom 45% only — wall logos are in the top half, floor tiles at bottom
        roi_top = int(h * ROI_FRACTION)
        roi     = frame[roi_top:, :]
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        cv2.line(disp, (0, roi_top), (w, roi_top), (0, 255, 255), 2)
        cv2.putText(disp, f'Mode: {self.colour_mode}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if self.pending_mode:
            cv2.putText(disp, f'Pending: {self.pending_mode} ({self.settle_counter}/{SETTLE_FRAMES})',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # ── STEP 1: Red HALT (always runs) ────────────────────────────────────
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LO1, RED_HI1),
            cv2.inRange(hsv, RED_LO2, RED_HI2)
        )
        red_px = int(np.sum(red_mask > 0))
        cv2.putText(disp, f'Red: {red_px}/{RED_PX_THRESHOLD}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if red_px >= RED_PX_THRESHOLD:
            self.red_seen_before = True
            cv2.putText(disp, 'RED ZONE', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return
        elif self.red_seen_before and not self.halt_triggered:
            self.halt_triggered = True
            self.get_logger().info('RED HALT — passed over red zone')
            self._publish_halt()

        if self.halt_triggered:
            cv2.putText(disp, 'HALTED', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return

        # ── STEP 2: Skip if dormant or pending ────────────────────────────────
        if self.colour_mode == MODE_DORMANT or self.pending_mode is not None:
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return

        # ── STEP 3: Skip if turning ───────────────────────────────────────────
        if self.angular_vel > MAX_ANGULAR_VEL:
            cv2.putText(disp, 'TURNING', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return

        # ── STEP 4: Skip if in post-detection lockout ─────────────────────────
        if self.lockout_remaining > 0:
            self.lockout_remaining -= 1
            cv2.putText(disp, f'Lockout: {self.lockout_remaining}', (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return

        # ── STEP 5: Blue gate — confirm we are over a floor logo tile ─────────
        blue_mask = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
        blue_px   = int(np.sum(blue_mask > 0))
        cv2.putText(disp, f'Blue: {blue_px}/{BLUE_GATE_PX}', (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if blue_px < BLUE_GATE_PX:
            self.vote_buffer.clear()   # not on a tile — reset vote
            if self.use_display:
                cv2.imshow('Tile Counter', disp)
                cv2.waitKey(1)
            return

        # ── STEP 6: Pixel dominance — which colour wins ───────────────────────
        green_mask  = cv2.morphologyEx(cv2.inRange(hsv, GREEN_LO, GREEN_HI), cv2.MORPH_OPEN, MORPH_KERNEL)
        orange_mask = cv2.morphologyEx(cv2.bitwise_or(
            cv2.inRange(hsv, ORANGE_LO1, ORANGE_HI1),
            cv2.inRange(hsv, ORANGE_LO2, ORANGE_HI2)
        ), cv2.MORPH_OPEN, MORPH_KERNEL)
        green_px  = int(np.sum(green_mask > 0))
        orange_px = int(np.sum(orange_mask > 0))

        cv2.putText(disp, f'G:{green_px} O:{orange_px}', (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if green_px > orange_px * COLOUR_RATIO:
            result = MODE_GREEN
        elif orange_px > green_px * COLOUR_RATIO:
            result = MODE_ORANGE
        else:
            result = None   # ambiguous — don't vote

        # ── STEP 7: Only vote if result matches current mode ──────────────────
        if result == self.colour_mode:
            self.vote_buffer.append(result)
        else:
            self.vote_buffer.append(None)

        cv2.putText(disp, f'Votes: {sum(1 for v in self.vote_buffer if v == self.colour_mode)}/{VOTE_FRAMES}',
                    (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ── STEP 8: Check for confirmed detection ─────────────────────────────
        if len(self.vote_buffer) == VOTE_FRAMES:
            matching = sum(1 for v in self.vote_buffer if v == self.colour_mode)
            if matching >= VOTE_REQUIRED:
                self.vote_buffer.clear()
                self.lockout_remaining = LOCKOUT_FRAMES

                if self.colour_mode == MODE_GREEN:
                    self.green_count += 1
                    count = self.green_count
                else:
                    self.orange_count += 1
                    count = self.orange_count

                self.get_logger().info(
                    f'TILE #{count} | {self.colour_mode} | '
                    f'G={green_px} O={orange_px} B={blue_px}'
                )
                self._log(self.colour_mode, count, green_px, orange_px, blue_px)
                self._publish_tile(self.colour_mode, count)

                cv2.putText(disp, f'CONFIRMED: {self.colour_mode} #{count}',
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if self.use_display:
            cv2.imshow('Tile Counter', disp)
            cv2.waitKey(1)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _log(self, colour, tile_num, green_px, orange_px, blue_px):
        if self.log_file is None:
            return
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.log_file.write(
            f'{ts},{self.colour_mode},{colour},{tile_num},{green_px},{orange_px},{blue_px}\n'
        )
        self.log_file.flush()

    def _publish_tile(self, colour: str, tile_num: int):
        msg = FloorMarkerDetection()
        msg.sign_type = f"{colour}_TILE_{tile_num}"
        msg.confidence = 1.0
        msg.distance_estimate = 0.0
        msg.pixel_width = 0.0
        msg.image_x = 0.0
        msg.image_y = 0.0
        self.det_pub.publish(msg)

    def _publish_halt(self):
        msg = FloorMarkerDetection()
        msg.sign_type = "RED_HALT"
        msg.confidence = 1.0
        msg.distance_estimate = 0.0
        msg.pixel_width = 0.0
        msg.image_x = 0.0
        msg.image_y = 0.0
        self.det_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, 'use_display', False):
            cv2.destroyAllWindows()
        if getattr(node, 'log_file', None):
            node.log_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
