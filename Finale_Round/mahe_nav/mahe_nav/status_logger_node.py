"""
status_logger_node.py
=====================
Mission dashboard for the current MAHE UGV arena.

Subscribes to:
  /r1a004/wheel_odom   nav_msgs/Odometry
  /imu/data            sensor_msgs/Imu
  /aruco/detections    mahe_nav_interfaces/ArucoDetection
  /floor_marker/detection  mahe_nav_interfaces/FloorMarkerDetection
  /lidar/analysis      mahe_nav_interfaces/LidarAnalysis

Publishes:
  /mission_status      std_msgs/String  (JSON, 2 Hz)

Logs to:
  /tmp/mahe_ugv_mission_<timestamp>.log
"""

import math
import time
import json
import csv
import os
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from mahe_nav_interfaces.msg import ArucoDetection, FloorMarkerDetection, LidarAnalysis


# ArUco ID → human-readable command (matches nav_controller_node.py)
ARUCO_CMD = {
    1: 'TURN LEFT',
    2: 'TURN RIGHT',
    3: 'FOLLOW GREEN',
    4: 'U-TURN',
    5: 'FOLLOW ORANGE',
}

LOG_FILE = f'/tmp/mahe_ugv_mission_{int(time.time())}.log'

class MissionLogger:
    # ArUco ID → human-readable command (matches nav_controller_node.py (1-5 scheme))
    _ARUCO_ACTIONS = {
        1: 'TURN LEFT',
        2: 'TURN RIGHT',
        3: 'FOLLOW GREEN',
        4: 'U-TURN',
        5: 'FOLLOW ORANGE',
    }

    def __init__(self, log_path=os.path.expanduser('~/mission_logs/mission_log.csv')):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.lock = threading.Lock()
        self.logging_active = True

        self.aruco_count = 0
        self._aruco_logged_ids = set()

        self.floor_count = 0
        self.total_green = 0
        self.total_orange = 0

        # Create/Clear file and write legend
        with self.lock:
            try:
                # Check for existing lock files (e.g. from LibreOffice)
                lock_file = self.log_path.replace('mission_log.csv', '.~lock.mission_log.csv#')
                if os.path.exists(lock_file):
                    print(f"\a[MissionLogger] WARNING: Lock file detected at {lock_file}")
                    print(f"[MissionLogger] PLEASE CLOSE THE CSV VIEWER SO THE LOG CAN BE UPDATED!")

                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['type', 'serial_no', 'value_1', 'value_2', 'value_3'])
                    writer.writerow(['LEGEND', 'ARUCO', 'tag_id', 'timestamp', 'action_triggered'])
                    writer.writerow(['LEGEND', 'FLOOR', 'colour',  'direction', 'timestamp'])
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[MissionLogger] Baseline CSV initialized at: {self.log_path}")
            except Exception as e:
                print(f"[MissionLogger] ERROR initializing CSV: {e}")

    def _write_aruco_row(self, tag_id, timestamp, action_str):
        # Special Rule: Phantom ID 1 logging when 2 is seen
        if tag_id == 2 and 1 not in self._aruco_logged_ids:
            self.aruco_count += 1
            self._aruco_logged_ids.add(1)
            try:
                with open(self.log_path, 'a', newline='') as f:
                    csv.writer(f).writerow(['ARUCO', self.aruco_count, 1, 'UNDETECTED', 'RIGHT Arc Turn'])
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[MissionLogger] Auto-inserted Phantom Tag 1 (RIGHT Arc Turn)")
            except Exception as e:
                print(f"[MissionLogger] ERROR writing phantom ID 1: {e}")

        self.aruco_count += 1
        self._aruco_logged_ids.add(tag_id)
        try:
            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow(['ARUCO', self.aruco_count, tag_id, timestamp, action_str])
                f.flush()
                os.fsync(f.fileno())
            print(f"[MissionLogger] Saved ARUCO: ID={tag_id}, action={action_str}")
        except Exception as e:
            print(f"[MissionLogger] ERROR writing ARUCO row: {e}")
        print(f"[MissionLogger] Saved ARUCO: ID={tag_id}, action={action_str}")

    def log_aruco_first_detection(self, tag_id):
        with self.lock:
            if not self.logging_active or tag_id in self._aruco_logged_ids:
                return
            action_str = self._ARUCO_ACTIONS.get(tag_id, f'UNKNOWN_ID_{tag_id}')
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._write_aruco_row(tag_id, timestamp, action_str)

    def log_floor_marker(self, colour, direction):
        with self.lock:
            if not self.logging_active:
                return
            self.floor_count += 1
            if colour == 'GREEN':
                self.total_green += 1
            elif colour == 'ORANGE':
                self.total_orange += 1
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            try:
                with open(self.log_path, 'a', newline='') as f:
                    csv.writer(f).writerow(['FLOOR', self.floor_count, colour, direction, timestamp])
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[MissionLogger] Saved FLOOR: {colour} ({direction})")
            except Exception as e:
                print(f"[MissionLogger] ERROR writing FLOOR row: {e}")
            print(f"[MissionLogger] Saved FLOOR: {colour} ({direction})")

    def write_summary(self):
        with self.lock:
            if not self.logging_active:
                return
            self.logging_active = False
            try:
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['SUMMARY', '', 'TOTAL_GREEN',  self.total_green,  ''])
                    writer.writerow(['SUMMARY', '', 'TOTAL_ORANGE', self.total_orange, ''])
                    writer.writerow(['SUMMARY', '', 'TOTAL_TILES', self.total_green + self.total_orange, ''])
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[MissionLogger] Mission Summary Saved to CSV.")
            except Exception as e:
                print(f"[MissionLogger] ERROR writing final summary: {e}")
            print(f"[MissionLogger] Mission Summary Saved to CSV.")

class StatusLoggerNode(Node):

    def __init__(self):
        super().__init__('status_logger')

        # State
        self.pose_x   = 0.0
        self.pose_y   = 0.0
        self.pose_yaw = 0.0
        self.start_time = time.time()

        self.aruco_log     = {}   # mid → {time_s, cmd, dist}
        self.sign_log      = []   # [{type, time_s, confidence}]
        self.last_sign     = 'NONE'
        self.fsm_state     = 'UNKNOWN'
        self.junction_type = 'UNKNOWN'
        self.forward_dist  = 0.0

        # Mission Logger
        self.mission_logger = MissionLogger()

        # Log file
        try:
            self.log_file = open(LOG_FILE, 'w')
            self._write(f'=== MAHE UGV Mission Log — {time.ctime()} ===\n')
        except IOError as e:
            self.get_logger().error(f'Cannot open log file: {e}')
            self.log_file = None

        # Publishers
        self.pub_status = self.create_publisher(String, '/mission_status', 10)

        # Subscribers
        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable    = QoSProfile(depth=10)

        self.create_subscription(Odometry,      '/r1a004/wheel_odom',self._odom_cb,  reliable)
        self.create_subscription(Imu,           '/imu/data',         self._imu_cb,   reliable)
        self.create_subscription(ArucoDetection,'/aruco/detections', self._aruco_cb, best_effort)
        self.create_subscription(FloorMarkerDetection, '/floor_marker/detection', self._sign_cb, best_effort)
        self.create_subscription(LidarAnalysis, '/lidar/analysis',   self._lidar_cb, best_effort)

        # Dashboard timer — 2 Hz
        self.create_timer(0.5, self._dashboard)
        self.get_logger().info(f'Status logger active. Logging to: {LOG_FILE}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        self.pose_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _lidar_cb(self, msg: LidarAnalysis):
        self.forward_dist  = msg.forward_dist
        self.junction_type = msg.junction_type if msg.junction_type else 'UNKNOWN'

    def _aruco_cb(self, msg: ArucoDetection):
        mid     = msg.marker_id
        elapsed = time.time() - self.start_time
        cmd     = ARUCO_CMD.get(mid, f'UNKNOWN_ID_{mid}')

        # Log first detection only
        if msg.first_detection:
            self.aruco_log[mid] = {
                'dist':   round(float(msg.distance), 3),
            }
            # Log to Mission Logger
            self.mission_logger.log_aruco_first_detection(mid)
            line = (
                f'[{elapsed:7.2f}s] *** FIRST ARUCO *** '
                f'ID={mid} | CMD={cmd} | '
                f'dist={msg.distance:.2f}m | '
                f'bearing={math.degrees(msg.bearing_angle_rad):.1f}° | '
                f'pos=({self.pose_x:.2f}, {self.pose_y:.2f})'
            )
            self.get_logger().info(line)
            self._write(line)
        else:
            # Repeat detection — only log if within action threshold
            if float(msg.distance) <= 0.8:
                line = (
                    f'[{elapsed:7.2f}s] ARUCO repeat '
                    f'ID={mid} | dist={msg.distance:.2f}m | '
                    f'bearing={math.degrees(msg.bearing_angle_rad):.1f}°'
                )
                self.get_logger().debug(line)

    def _sign_cb(self, msg: FloorMarkerDetection):
        if msg.colour == "RED" and msg.direction == "HALT":
            self.mission_logger.write_summary()
            self.get_logger().info("RED TILE DETECTED - Mission Summary Saved.")
            return

        if not msg.colour or msg.colour == 'NONE':
            return
        if msg.confidence < 0.45:
            return
        if msg.direction == 'TIMEOUT':
            return

        elapsed = time.time() - self.start_time

        # Debounce same colour+direction within 2s  (allows direction CHANGES through)
        key = f'{msg.colour}:{msg.direction}'
        recent = [e for e in self.sign_log if e.get('key') == key]
        if recent and (elapsed - recent[-1]['time_s']) < 2.0:
            return

        self.last_sign = f'{msg.colour}:{msg.direction}'
        self.sign_log.append({
            'type':       msg.colour,
            'key':        key,
            'time_s':     round(elapsed, 2),
            'confidence': round(float(msg.confidence), 3),
        })

        # Log to Mission Logger
        self.mission_logger.log_floor_marker(msg.colour, msg.direction)

        line = (
            f'[{elapsed:7.2f}s] FLOOR MARKER: {msg.colour} \u2192 {msg.direction} '
            f'(conf={msg.confidence:.2f}) tile#{msg.tile_count} | '
            f'world_angle={msg.world_angle_deg:.1f}\u00b0 | '
            f'pos=({self.pose_x:.2f}, {self.pose_y:.2f})'
        )
        self.get_logger().info(line)
        self._write(line)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _dashboard(self):
        elapsed  = time.time() - self.start_time
        seen_ids = sorted(self.aruco_log.keys())

        lines = [
            '─' * 52,
            f'  T={elapsed:6.1f}s  |  pos=({self.pose_x:.2f}, {self.pose_y:.2f})  |  yaw={math.degrees(self.pose_yaw):.1f}°',
            f'  fwd={self.forward_dist:.2f}m  |  junction={self.junction_type}',
            f'  ARUCO seen: {seen_ids}',
            f'  Last CV   : {self.last_sign}',
            '─' * 52,
        ]
        for l in lines:
            self.get_logger().info(l)

        self._publish_status(elapsed, seen_ids)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write(self, text: str):
        if self.log_file:
            self.log_file.write(text + '\n')
            self.log_file.flush()

    def _publish_status(self, elapsed: float, seen_ids: list):
        msg = String()
        msg.data = json.dumps({
            'elapsed_s':    round(elapsed, 1),
            'pos':          [round(self.pose_x, 2), round(self.pose_y, 2)],
            'yaw_deg':      round(math.degrees(self.pose_yaw), 1),
            'junction':     self.junction_type,
            'forward_dist': round(self.forward_dist, 2),
            'aruco_seen':   seen_ids,
            'last_sign':    self.last_sign,
        })
        self.pub_status.publish(msg)

    def destroy_node(self):
        if self.log_file:
            self._write(f'\n=== Mission ended {time.ctime()} ===')
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StatusLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()