"""
lidar_viz_node.py
=================
Real-time bird's-eye OpenCV visualisation of the processed LiDAR data.

Subscribes to:
  /scan             sensor_msgs/LaserScan       (raw points for background)
  /lidar/analysis   mahe_nav_interfaces/LidarAnalysis  (processed data overlay)

Displays:
  - Raw scan points (cyan dots)
  - Cardinal direction axes (coloured lines)
  - Sector distances (numeric labels)
  - Detected openings (green = passable, red = blocked)
  - Junction type, corridor width, U-shape, cylinder bearing
  - Terminal printout of key values at 2 Hz
"""

import math
import numpy as np
import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from mahe_nav_interfaces.msg import LidarAnalysis


# ── Visualisation constants ─────────────────────────────────────────────────────
WIN_SIZE     = 700          # px — square window
SCALE        = 100          # px per metre (1 m = 100 px)
MAX_RANGE_VIZ = 3.0         # metres — clip display range
CENTER       = (WIN_SIZE // 2, WIN_SIZE // 2)

# Colours (BGR)
COL_BG         = (20, 20, 20)
COL_GRID       = (45, 45, 45)
COL_SCAN_PT    = (220, 200, 50)     # cyan-ish scan dots
COL_ROBOT      = (0, 180, 255)      # orange robot marker
COL_FWD        = (0, 255, 0)        # green  — forward
COL_LEFT       = (255, 200, 0)      # cyan   — left
COL_RIGHT      = (0, 100, 255)      # orange — right
COL_BACK       = (80, 80, 255)      # red    — backward
COL_PASSABLE   = (0, 255, 100)      # green gap
COL_BLOCKED    = (0, 0, 220)        # red gap
COL_CYLINDER   = (255, 0, 255)      # magenta — cylinder
COL_TEXT       = (220, 220, 220)
COL_WARN       = (0, 100, 255)      # orange warning

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL     = 0.42
FONT_MED       = 0.55
FONT_BIG       = 0.65


def _world_to_px(x_m, y_m):
    """Convert robot-frame (x=fwd, y=left) to pixel coords (y flipped)."""
    px = CENTER[0] + int(-y_m * SCALE)   # left in world = right on screen
    py = CENTER[1] + int(-x_m * SCALE)   # forward in world = up on screen
    return (px, py)


class LidarVizNode(Node):

    def __init__(self):
        super().__init__('lidar_viz')

        self.scan_msg = None
        self.analysis_msg = None
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(LaserScan,    '/scan',           self._scan_cb,     sensor_qos)
        self.create_subscription(LidarAnalysis, '/lidar/analysis', self._analysis_cb, sensor_qos)

        # Render at ~15 Hz
        self.create_timer(1.0 / 15.0, self._render)
        # Terminal printout at 2 Hz
        self.create_timer(0.5, self._terminal_print)

        cv2.namedWindow('LiDAR View', cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('LiDAR Visualiser started — press Q in the window to quit')

    # ── Callbacks ────────────────────────────────────────────────────────────────

    def _scan_cb(self, msg):
        with self.lock:
            self.scan_msg = msg

    def _analysis_cb(self, msg):
        with self.lock:
            self.analysis_msg = msg

    # ── Terminal dashboard ───────────────────────────────────────────────────────

    def _terminal_print(self):
        a = self.analysis_msg
        if a is None:
            return
        n_open = len(a.opening_angles_rad)
        n_pass = sum(a.opening_passable) if a.opening_passable else 0
        cyl = f'YES  bearing={math.degrees(a.cylinder_bearing_rad):.0f}°' if a.cylinder_detected else 'no'

        print(
            f'\033[96m'
            f'┌─ LiDAR ──────────────────────────────────────────┐\n'
            f'│  FWD  {a.forward_dist:5.2f}m    LEFT {a.left_dist:5.2f}m              │\n'
            f'│  BACK {a.back_dist:5.2f}m    RIGHT {a.right_dist:5.2f}m             │\n'
            f'│  Junction : {a.junction_type:<14s}  Corridor: {a.corridor_width_lateral:.2f}m │\n'
            f'│  Openings : {n_open} total, {n_pass} passable               │\n'
            f'│  U-shape  : {a.is_u_shape!s:<6s}  Cylinder: {cyl:<16s}│\n'
            f'│  Wall err : {math.degrees(a.wall_alignment_error_rad):+.1f}°                              │\n'
            f'└──────────────────────────────────────────────────┘'
            f'\033[0m'
        )

    # ── Render loop ──────────────────────────────────────────────────────────────

    def _render(self):
        canvas = np.full((WIN_SIZE, WIN_SIZE, 3), COL_BG, dtype=np.uint8)

        # ── Grid rings (1m, 2m, 3m) ─────────────────────────────────────────
        for r_m in range(1, int(MAX_RANGE_VIZ) + 1):
            r_px = int(r_m * SCALE)
            cv2.circle(canvas, CENTER, r_px, COL_GRID, 1)
            cv2.putText(canvas, f'{r_m}m', (CENTER[0] + 4, CENTER[1] - r_px + 14),
                        FONT, FONT_SMALL, COL_GRID, 1)

        # ── Axis lines ──────────────────────────────────────────────────────
        axis_len = int(MAX_RANGE_VIZ * SCALE)
        # Forward (up)
        cv2.line(canvas, CENTER, (CENTER[0], CENTER[1] - axis_len), COL_FWD, 1)
        cv2.putText(canvas, 'FWD', (CENTER[0] + 5, CENTER[1] - axis_len + 15), FONT, FONT_SMALL, COL_FWD, 1)
        # Backward (down)
        cv2.line(canvas, CENTER, (CENTER[0], CENTER[1] + axis_len), COL_BACK, 1)
        # Left (right on screen because we flip)
        cv2.line(canvas, CENTER, (CENTER[0] - axis_len, CENTER[1]), COL_LEFT, 1)
        cv2.putText(canvas, 'L', (CENTER[0] - axis_len + 5, CENTER[1] - 8), FONT, FONT_SMALL, COL_LEFT, 1)
        # Right
        cv2.line(canvas, CENTER, (CENTER[0] + axis_len, CENTER[1]), COL_RIGHT, 1)
        cv2.putText(canvas, 'R', (CENTER[0] + axis_len - 18, CENTER[1] - 8), FONT, FONT_SMALL, COL_RIGHT, 1)

        with self.lock:
            scan = self.scan_msg
            analysis = self.analysis_msg

        # ── Raw scan points ─────────────────────────────────────────────────
        LIDAR_MOUNT_OFFSET = math.pi  # Match lidar_analyzer_node LIDAR_MOUNT_REVERSED
        if scan is not None:
            n = len(scan.ranges)
            for i in range(0, n, max(1, n // 800)):   # subsample for speed
                r = scan.ranges[i]
                if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                    continue
                if r > MAX_RANGE_VIZ:
                    continue
                angle = scan.angle_min + i * scan.angle_increment + LIDAR_MOUNT_OFFSET
                # Robot frame: x=forward, y=left  → angle 0 = forward (+x)
                x_m = r * math.cos(angle)
                y_m = r * math.sin(angle)
                px, py = _world_to_px(x_m, y_m)
                if 0 <= px < WIN_SIZE and 0 <= py < WIN_SIZE:
                    cv2.circle(canvas, (px, py), 2, COL_SCAN_PT, -1)

        # ── Analysed data overlay ───────────────────────────────────────────
        if analysis is not None:
            a = analysis

            # Cardinal distance markers (thick line segments)
            for dist, angle, col, label in [
                (a.forward_dist, math.pi / 2,   COL_FWD,   f'{a.forward_dist:.2f}m'),
                (a.back_dist,   -math.pi / 2,   COL_BACK,  f'{a.back_dist:.2f}m'),
                (a.left_dist,    math.pi,        COL_LEFT,  f'{a.left_dist:.2f}m'),
                (a.right_dist,   0.0,            COL_RIGHT, f'{a.right_dist:.2f}m'),
            ]:
                d_clipped = min(dist, MAX_RANGE_VIZ)
                # angle: 0=right, pi/2=forward, pi=left in standard math
                x_m = d_clipped * math.cos(angle)
                y_m = d_clipped * math.sin(angle)
                end_px = _world_to_px(y_m, -x_m)  # swap for screen coords
                # Draw a small filled circle at the endpoint
                if 0 <= end_px[0] < WIN_SIZE and 0 <= end_px[1] < WIN_SIZE:
                    cv2.circle(canvas, end_px, 5, col, -1)
                    cv2.putText(canvas, label, (end_px[0] + 8, end_px[1] + 5),
                                FONT, FONT_SMALL, col, 1)

            # ── Openings (gaps) ─────────────────────────────────────────────
            for i, (ang, width, passable) in enumerate(zip(
                    a.opening_angles_rad, a.opening_widths_m, a.opening_passable)):
                col = COL_PASSABLE if passable else COL_BLOCKED
                # Draw a wedge line at the gap angle, length proportional to width
                gap_len = min(width, MAX_RANGE_VIZ)
                # ang is relative to forward: 0=fwd, +ve=left
                x_m = gap_len * math.cos(ang)   # forward component
                y_m = gap_len * math.sin(ang)    # left component
                end = _world_to_px(x_m, y_m)
                if 0 <= end[0] < WIN_SIZE and 0 <= end[1] < WIN_SIZE:
                    cv2.line(canvas, CENTER, end, col, 2)
                    # Small label
                    lbl = f'{width:.2f}m'
                    mid_x = (CENTER[0] + end[0]) // 2
                    mid_y = (CENTER[1] + end[1]) // 2
                    cv2.putText(canvas, lbl, (mid_x, mid_y), FONT, FONT_SMALL - 0.05, col, 1)

            # ── Cylinder detection ──────────────────────────────────────────
            if a.cylinder_detected:
                cyl_ang = a.cylinder_bearing_rad
                cyl_x = 1.5 * math.cos(cyl_ang)
                cyl_y = 1.5 * math.sin(cyl_ang)
                cyl_px = _world_to_px(cyl_x, cyl_y)
                if 0 <= cyl_px[0] < WIN_SIZE and 0 <= cyl_px[1] < WIN_SIZE:
                    cv2.circle(canvas, cyl_px, 12, COL_CYLINDER, 2)
                    cv2.putText(canvas, 'CYL', (cyl_px[0] - 12, cyl_px[1] - 15),
                                FONT, FONT_SMALL, COL_CYLINDER, 1)

            # ── Text overlay (top-left) ─────────────────────────────────────
            y_text = 25
            line_h = 22

            def put(text, col=COL_TEXT):
                nonlocal y_text
                cv2.putText(canvas, text, (12, y_text), FONT, FONT_MED, col, 1, cv2.LINE_AA)
                y_text += line_h

            junc_col = COL_WARN if a.junction_type in ('DEAD_END', 'T_JUNCTION') else COL_TEXT
            put(f'Junction: {a.junction_type}', junc_col)
            put(f'Corridor: {a.corridor_width_lateral:.2f}m')
            put(f'Wall err: {math.degrees(a.wall_alignment_error_rad):+.1f} deg')

            n_pass = sum(a.opening_passable) if a.opening_passable else 0
            put(f'Openings: {len(a.opening_angles_rad)} ({n_pass} passable)')

            if a.is_u_shape:
                put('!! U-SHAPE / DEAD END !!', COL_WARN)
            if a.no_forward_passage:
                put('!! NO FORWARD PASSAGE !!', COL_WARN)
            if a.cylinder_detected:
                put(f'CYLINDER @ {math.degrees(a.cylinder_bearing_rad):.0f} deg', COL_CYLINDER)

        # ── Robot marker ────────────────────────────────────────────────────
        cv2.circle(canvas, CENTER, 6, COL_ROBOT, -1)
        # Forward arrow
        cv2.arrowedLine(canvas, CENTER, (CENTER[0], CENTER[1] - 25), COL_ROBOT, 2, tipLength=0.4)

        # ── Show ────────────────────────────────────────────────────────────
        cv2.imshow('LiDAR View', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            self.get_logger().info('Quit requested — shutting down visualiser')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = LidarVizNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
