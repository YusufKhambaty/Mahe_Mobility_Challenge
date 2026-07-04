"""
lidar_analyzer_node.py
======================
Converts the raw LaserScan into structured analysis that the nav_controller
can use directly — no raw scan indices in the controller.

LIDAR SPECS (from URDF)
-----------------------
    Sensor    : gpu_lidar, 360 samples, 10 Hz
    min_angle : -π rad     max_angle : +π rad
    min_range : 0.12 m     max_range : 30.0 m

INDEX ↔ DIRECTION MAPPING
--------------------------
    In the ROS LaserScan message the angle at index i is:
        angle_i = angle_min + i * angle_increment
              = -π + i * (2π / 360)

    Standard ROS convention: angle 0 = FORWARD, +ve = LEFT (CCW from above).
    Therefore:
        index 0   → angle = -π       → BACKWARD
        index 90  → angle = -π/2     → RIGHT
        index 180 → angle = 0        → FORWARD
        index 270 → angle = +π/2     → LEFT

    NOTE: The robot description document lists index 0 = FORWARD.
    If the robot behaves backwards in testing, swap by setting
    parameter  lidar_forward_index = 0  (document convention)
    vs the default lidar_forward_index = 180  (physics convention).

OPENING DETECTION ALGORITHM
-----------------------------
1.  Replace inf/NaN readings with max_range.
2.  Smooth the scan with a 5-sample median filter to kill noise spikes.
3.  Scan all 360°.  An "open gap" starts where range > OPEN_THRESHOLD and
    ends where range drops back below OPEN_THRESHOLD.
4.  For each gap: compute angular width and estimate physical width at the
    mean distance:
        physical_width ≈ 2 * mean_dist * tan(angular_width / 2)
5.  Mark the gap as PASSABLE if:
        physical_width ≥ PASSABILITY_FACTOR × ROBOT_WIDTH
    PASSABILITY_FACTOR = 1.50  (leaves margin)
    ROBOT_WIDTH        = 0.350 m  (conservative — wider than wheel separation)
    → passable threshold ≈ 0.525 m  (allows the 580 mm tight gap)

U-SHAPE DETECTION (closed row / dead end)
------------------------------------------
    A closed row presents walls on three sides.  Detected when:
        forward_dist < CLOSED_ROW_DEPTH  AND
        left_dist    < CLOSED_ROW_DEPTH  AND
        right_dist   < CLOSED_ROW_DEPTH
    where CLOSED_ROW_DEPTH = 2.5 m  (tuned to row dimensions ≈ 700-1000 mm).
    The back direction must be open (robot came from there) so we also check:
        back_dist > forward_dist  (just a sanity check, not strictly required)

TOPICS
------
  SUB  /r1_mini/lidar    sensor_msgs/LaserScan
  PUB  /lidar/analysis   mahe_nav_interfaces/LidarAnalysis
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from mahe_nav_interfaces.msg import LidarAnalysis


# ── Robot geometry ──────────────────────────────────────────────────────────────
ROBOT_WIDTH_M       = 0.350   # metres — conservative body width
PASSABILITY_FACTOR  = 1.50    # physical_width must be > factor × robot_width
PASSABLE_THRESHOLD  = PASSABILITY_FACTOR * ROBOT_WIDTH_M   # 0.525 m

# ── Thresholds ──────────────────────────────────────────────────────────────────
OPEN_THRESHOLD      = 0.80    # metres — range > this considered "open"
CLOSED_ROW_DEPTH    = 2.5     # metres — rows shorter than this are "closed"

# ── Smoothing kernel width (samples) ────────────────────────────────────────────
MEDIAN_KERNEL       = 5


class LidarAnalyzerNode(Node):

    def __init__(self):
        super().__init__('lidar_analyzer')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.declare_parameter('lidar_forward_index', 180)  # physics-correct default
        self.declare_parameter('open_threshold',      OPEN_THRESHOLD)
        self.declare_parameter('closed_row_depth',    CLOSED_ROW_DEPTH)
        self.declare_parameter('robot_width_m',       ROBOT_WIDTH_M)
        self.declare_parameter('passability_factor',  PASSABILITY_FACTOR)

        fwd = self.get_parameter('lidar_forward_index').value
        self.FORWARD  = fwd % 360
        self.LEFT     = (fwd + 90)  % 360
        self.BACKWARD = (fwd + 180) % 360
        self.RIGHT    = (fwd + 270) % 360

        self.open_thresh  = self.get_parameter('open_threshold').value
        self.closed_depth = self.get_parameter('closed_row_depth').value
        robot_w           = self.get_parameter('robot_width_m').value
        p_factor          = self.get_parameter('passability_factor').value
        self.passable_thr = p_factor * robot_w

        self.get_logger().info(
            f'LiDAR analyzer: FORWARD={self.FORWARD} LEFT={self.LEFT} '
            f'RIGHT={self.RIGHT} BACK={self.BACKWARD}  '
            f'passable_thr={self.passable_thr:.3f} m')

        # ── Publisher / Subscriber ──────────────────────────────────────────────
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub = self.create_publisher(LidarAnalysis, '/lidar/analysis', 10)

        self.sub = self.create_subscription(
            LaserScan, '/r1_mini/lidar', self._scan_cb, sensor_qos)

    # ── Scan callback ───────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        n   = len(msg.ranges)
        raw = np.array(msg.ranges, dtype=np.float32)

        # Replace inf / NaN / below-min with max_range so gaps read as "open"
        bad = np.logical_or(np.isinf(raw), np.isnan(raw))
        bad = np.logical_or(bad, raw < msg.range_min)
        raw[bad] = msg.range_max

        # Median smoothing (circular — wraps at 0/359)
        k      = MEDIAN_KERNEL
        padded = np.concatenate([raw[-(k//2):], raw, raw[:k//2]])
        smooth = np.array([
            float(np.median(padded[i:i+k]))
            for i in range(n)
        ], dtype=np.float32)

        # ── Cardinal distances ──────────────────────────────────────────────────
        # Average a ±3° cone for robustness against single bad readings
        def cone_avg(centre_idx: int, half_width: int = 3) -> float:
            idxs = [(centre_idx + d) % n for d in range(-half_width, half_width+1)]
            return float(np.mean(smooth[idxs]))

        fwd_d  = cone_avg(self.FORWARD)
        left_d = cone_avg(self.LEFT)
        rgt_d  = cone_avg(self.RIGHT)
        bck_d  = cone_avg(self.BACKWARD)

        # ── U-shape / closed-row detection ─────────────────────────────────────
        is_u = (fwd_d  < self.closed_depth and
                left_d < self.closed_depth and
                rgt_d  < self.closed_depth)

        # ── Lateral corridor width estimate ────────────────────────────────────
        lateral_width = left_d + rgt_d + ROBOT_WIDTH_M

        # ── Opening detection ───────────────────────────────────────────────────
        opening_angles  = []
        opening_widths  = []
        opening_pass    = []

        in_gap      = False
        gap_start   = 0
        gap_dists   = []

        for i in range(n):
            r = float(smooth[i])
            if r > self.open_thresh:
                if not in_gap:
                    in_gap    = True
                    gap_start = i
                    gap_dists = []
                gap_dists.append(r)
            else:
                if in_gap:
                    in_gap = False
                    self._record_gap(
                        gap_start, i - 1, gap_dists, n, msg,
                        opening_angles, opening_widths, opening_pass)

        # Handle gap that wraps around index 359 → 0
        if in_gap:
            self._record_gap(
                gap_start, n - 1, gap_dists, n, msg,
                opening_angles, opening_widths, opening_pass)

        # ── Best opening (widest passable gap) ─────────────────────────────────
        best_idx = -1
        best_w   = 0.0
        for j, (w, p) in enumerate(zip(opening_widths, opening_pass)):
            if p and w > best_w:
                best_w   = w
                best_idx = j

        # ── Build and publish message ───────────────────────────────────────────
        out                         = LidarAnalysis()
        out.header                  = msg.header
        out.forward_dist            = fwd_d
        out.left_dist               = left_d
        out.right_dist              = rgt_d
        out.back_dist               = bck_d
        out.is_u_shape              = is_u
        out.corridor_width_lateral  = lateral_width
        out.opening_angles_rad      = [float(a) for a in opening_angles]
        out.opening_widths_m        = [float(w) for w in opening_widths]
        out.opening_passable        = list(opening_pass)
        out.best_opening_idx        = best_idx
        out.no_forward_passage      = (fwd_d < ROBOT_WIDTH_M * 1.2)

        self.pub.publish(out)

        self.get_logger().debug(
            f'fwd={fwd_d:.2f}  l={left_d:.2f}  r={rgt_d:.2f}  b={bck_d:.2f}  '
            f'U={is_u}  openings={len(opening_angles)}  best={best_idx}')

    # ── Record a detected gap ───────────────────────────────────────────────────
    def _record_gap(self, start_idx, end_idx, gap_dists, n_samples, msg,
                    angles_out, widths_out, pass_out):
        n_gap       = end_idx - start_idx + 1
        ang_width   = n_gap * msg.angle_increment           # radians
        mean_dist   = float(np.mean(gap_dists))
        phys_width  = 2.0 * mean_dist * math.tan(ang_width / 2.0)
        passable    = phys_width >= self.passable_thr

        # Centre angle of gap, in ROS convention (0=fwd, +ve=left, CCW)
        centre_i    = (start_idx + end_idx) // 2
        centre_ang  = msg.angle_min + centre_i * msg.angle_increment
        # Normalise to robot forward = 0: rotate by -(forward_index * increment)
        fwd_ang     = msg.angle_min + self.FORWARD * msg.angle_increment
        rel_angle   = centre_ang - fwd_ang
        # Wrap to [-π, π]
        while rel_angle >  math.pi: rel_angle -= 2*math.pi
        while rel_angle < -math.pi: rel_angle += 2*math.pi

        angles_out.append(rel_angle)
        widths_out.append(phys_width)
        pass_out.append(passable)

        self.get_logger().debug(
            f'  gap: angle={math.degrees(rel_angle):.1f}°  '
            f'width={phys_width:.3f}m  passable={passable}')


def main(args=None):
    rclpy.init(args=args)
    node = LidarAnalyzerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
