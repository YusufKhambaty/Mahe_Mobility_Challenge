#!/usr/bin/env python3
"""
capture_hsv_frames.py — Run alongside Gazebo to capture robot camera frames.
=========================================================================
HOW TO USE:
  1. Launch Gazebo with the maze world + spawn the robot
  2. In a new terminal:
       source /opt/ros/humble/setup.bash
       python3 capture_hsv_frames.py
  3. Use teleop or let nav run — the script saves a frame every 0.5s
  4. Press Ctrl+C to stop
  5. Frames are saved in ~/hsv_captures/

Then run the tuner on the captured frames:
  python3 hsv_heuristic_tuner.py --dir ~/hsv_captures/
"""

import os
import time
import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image


SAVE_DIR = os.path.expanduser("~/hsv_captures")
CAPTURE_INTERVAL = 0.5  # seconds between saves


class FrameCapturer(Node):
    def __init__(self):
        super().__init__('hsv_frame_capturer')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_save_time = 0.0

        os.makedirs(SAVE_DIR, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.create_subscription(
            Image,
            '/r1_mini/camera/image_raw',
            self._image_cb,
            qos_profile=sensor_qos
        )

        self.get_logger().info(f"Capturing frames to {SAVE_DIR} (every {CAPTURE_INTERVAL}s)")
        self.get_logger().info("Drive the robot near the floor tiles, then Ctrl+C when done.")

    def _image_cb(self, msg: Image):
        now = time.time()
        if now - self.last_save_time < CAPTURE_INTERVAL:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        self.frame_count += 1
        fname = os.path.join(SAVE_DIR, f"frame_{self.frame_count:04d}.png")
        cv2.imwrite(fname, bgr)
        self.last_save_time = now

        # Quick HSV snapshot of the frame for terminal feedback
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]
        roi = hsv[int(h * 0.4):h, :]  # bottom 60% = floor area

        # Count pixels in rough color ranges
        orange_px = cv2.countNonZero(cv2.inRange(roi, np.array([0, 80, 80]), np.array([30, 255, 255])))
        green_px = cv2.countNonZero(cv2.inRange(roi, np.array([35, 60, 60]), np.array([90, 255, 255])))
        blue_px = cv2.countNonZero(cv2.inRange(roi, np.array([95, 40, 40]), np.array([140, 255, 255])))

        self.get_logger().info(
            f"Frame #{self.frame_count} saved | O={orange_px} G={green_px} B={blue_px}"
        )


def main():
    rclpy.init()
    node = FrameCapturer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Done! {node.frame_count} frames saved to {SAVE_DIR}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
