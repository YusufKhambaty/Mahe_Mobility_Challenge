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
2.  Smooth the scan with a 7-sample median filter to kill noise spikes.
3.  Apply seam bridging: short laser-passthrough spikes (≤12 samples,
    both neighbours < 3.0 m and within 0.4 m of each other) are filled
    with linear interpolation so segmented walls look continuous.
4.  Scan all 360°.  An "open gap" starts where range > OPEN_THRESHOLD and
    ends where range drops back below OPEN_THRESHOLD.
5.  For each gap: compute physical width via Law of Cosines between the
    two wall-edge boundary readings.
6.  Mark the gap as PASSABLE if:
        physical_width ≥ PASSABILITY_FACTOR × ROBOT_WIDTH
    PASSABILITY_FACTOR = 1.42  (leaves margin)
    ROBOT_WIDTH        = 0.390 m  (actual robot width)
    → passable threshold ≈ 0.554 m

U-SHAPE DETECTION (closed row / dead end)
------------------------------------------
    A closed row presents walls on three sides.  Detected when:
        forward_dist < CLOSED_ROW_DEPTH  AND
        left_dist    < CLOSED_ROW_DEPTH  AND
        right_dist   < CLOSED_ROW_DEPTH
    where CLOSED_ROW_DEPTH = 2.5 m  (tuned to row dimensions ≈ 700-1000 mm).

GEOMETRIC WALL CLASSIFICATION
-------------------------------
    For CORRIDOR detection: SVD line fit over ±40 samples around LEFT and
    RIGHT cardinal indices.  A sector is a "genuine wall" when:
        mean perpendicular residual < WALL_RESID_THR (0.12 m)
    Two walls form a corridor when both fit AND their normals are within
    WALL_PARALLEL_THR (0.15 rad) of each other.

TOPICS
------
  SUB  /scan             sensor_msgs/LaserScan
  PUB  /lidar/analysis   mahe_nav_interfaces/LidarAnalysis

CHANGES vs previous version
-----------------------------
  FIX 1 : Replaced old slat-bridge (flat average, wrong trigger) with
           seam-bridge using linear interpolation and depth-agreement check.
  FIX 2 : Replaced hard-threshold junction classifier with geometric SVD
           line fit.  WALL_RESID_THR raised from 0.07 → 0.12 m so slightly
           rough arena walls still register as walls.
  FIX 3 : wall_alignment_error_rad now reuses the already-computed fits
           instead of running a second, different fitting pass.
  FIX 4 : Removed dead quad_info / align_err code that was computed but
           never published and used raw degree indices ignoring FORWARD.
  FIX 5 : cardinal initialisation moved before data cleaning so FORWARD
           index is always valid when sector math runs.
"""

import math
import numpy as np
import scipy.ndimage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from mahe_nav_interfaces.msg import LidarAnalysis


# ── Robot geometry ───────────────────────────────────────────────────────────
ROBOT_WIDTH_M       = 0.390
PASSABILITY_FACTOR  = 1.15
PASSABLE_THRESHOLD  = PASSABILITY_FACTOR * ROBOT_WIDTH_M   # ≈ 0.45 m

# ── Thresholds ───────────────────────────────────────────────────────────────
OPEN_THRESHOLD      = 0.45    # m — range > this is "open"
CLOSED_ROW_DEPTH    = 2.5     # m — u-shape detection depth

# ── Smoothing ────────────────────────────────────────────────────────────────
MEDIAN_KERNEL       = 7

# ── Seam bridging ────────────────────────────────────────────────────────────
SEAM_MAX_SAMPLES    = 12      # max spike run to bridge (≈ 6° at 0.5°/sample)
SEAM_WALL_MAX       = 3.0     # both neighbours must be wall, not open air (m)
SEAM_WALL_DIFF_M    = 0.40    # two wall sides must agree in depth (m)

# ── Geometric wall classification ────────────────────────────────────────────
# FIX 2: raised from 0.07 → 0.12 so slightly rough walls still classify
WALL_RESID_THR      = 0.12    # m — mean perpendicular residual for straight wall
WALL_PARALLEL_THR   = 0.15    # rad — max normal angle difference for corridor

# ── Hardware mount correction ─────────────────────────────────────────────────
LIDAR_MOUNT_REVERSED = False


class LidarAnalyzerNode(Node):

    def __init__(self):
        super().__init__('lidar_analyzer')

        self.declare_parameter('open_threshold',      OPEN_THRESHOLD)
        self.declare_parameter('closed_row_depth',    CLOSED_ROW_DEPTH)
        self.declare_parameter('robot_width_m',       ROBOT_WIDTH_M)
        self.declare_parameter('passability_factor',  PASSABILITY_FACTOR)

        self.open_thresh  = self.get_parameter('open_threshold').value
        self.closed_depth = self.get_parameter('closed_row_depth').value
        robot_w           = self.get_parameter('robot_width_m').value
        p_factor          = self.get_parameter('passability_factor').value
        self.passable_thr = p_factor * robot_w

        self._cardinal_initialized = False
        self.FORWARD  = 0
        self.LEFT     = 0
        self.BACKWARD = 0
        self.RIGHT    = 0

        self.get_logger().info(
            f'LiDAR analyzer ready — passable_thr={self.passable_thr:.3f} m  '
            f'wall_resid_thr={WALL_RESID_THR:.3f} m')

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(LidarAnalysis, '/lidar/analysis', 10)
        self.sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)

    def _init_cardinal_from_scan(self, n, amin, inc):
        # The Gefier R1 LiDAR physical mount defies standard ROS angle_min reporting.
        # We explicitly hardcode the indices to match the physical reality:
        # Index 0 is perfectly Forward, spinning Counter-Clockwise.
        offset = n // 4
        self.FORWARD  = 0
        self.LEFT     = offset
        self.BACKWARD = 2 * offset
        self.RIGHT    = 3 * offset
        self._cardinal_initialized = True
        self.get_logger().info(
            f'LiDAR cardinal (HARDCODED FOR GEFIER R1): '
            f'n={n} FORWARD={self.FORWARD} LEFT={self.LEFT} '
            f'RIGHT={self.RIGHT} BACK={self.BACKWARD}')

    # ── Scan callback ────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        n   = len(msg.ranges)
        raw = np.array(msg.ranges, dtype=np.float32)

        # ── 1. Mount correction ───────────────────────────────────────────────
        if LIDAR_MOUNT_REVERSED:
            raw      = raw[::-1]
            eff_amin = msg.angle_max + math.pi
            while eff_amin >  math.pi: eff_amin -= 2 * math.pi
            while eff_amin <= -math.pi: eff_amin += 2 * math.pi
            eff_amin = eff_amin + (n - 1) * msg.angle_increment
            while eff_amin >  math.pi: eff_amin -= 2 * math.pi
            while eff_amin <= -math.pi: eff_amin += 2 * math.pi
            eff_inc  = msg.angle_increment
        else:
            eff_amin = msg.angle_min
            eff_inc  = msg.angle_increment

        self._eff_amin = eff_amin
        self._eff_inc  = eff_inc

        # FIX 5: initialise cardinals before any sector math
        if not self._cardinal_initialized:
            self._init_cardinal_from_scan(n, eff_amin, eff_inc)

        # ── 2. Clean: replace inf / NaN / below-min with max_range ───────────
        bad = np.logical_or(np.isinf(raw), np.isnan(raw))
        bad = np.logical_or(bad, raw < msg.range_min)
        raw[bad] = msg.range_max

        # ── 3. Median smoothing ───────────────────────────────────────────────
        smooth = scipy.ndimage.median_filter(
            raw, size=MEDIAN_KERNEL, mode='wrap').astype(np.float32)

        # ── 4. Seam bridging ──────────────────────────────────────────────────
        # Laser beams that pass through narrow wall seams produce short spikes
        # of large range surrounded by normal wall readings.  Fill with linear
        # interpolation (follows wall slope) rather than flat average.
        # A run is treated as a seam when ALL of:
        #   • run length ≤ SEAM_MAX_SAMPLES
        #   • both neighbouring wall readings ≤ SEAM_WALL_MAX
        #   • the two wall readings agree in depth within SEAM_WALL_DIFF_M
        in_spike  = False
        spk_start = 0

        for si in range(n + SEAM_MAX_SAMPLES):   # overshoot to close wrap seams
            ci = si % n
            r  = float(smooth[ci])
            if r > self.open_thresh:
                if not in_spike:
                    in_spike  = True
                    spk_start = si
            else:
                if in_spike:
                    in_spike = False
                    run_len  = si - spk_start
                    if run_len <= SEAM_MAX_SAMPLES:
                        wall_L = float(smooth[(spk_start - 1) % n])
                        wall_R = float(smooth[ci % n])
                        if (wall_L <= SEAM_WALL_MAX and
                                wall_R <= SEAM_WALL_MAX and
                                abs(wall_L - wall_R) <= SEAM_WALL_DIFF_M):
                            for fi in range(run_len):
                                t = (fi + 1) / (run_len + 1)   # 0 < t < 1
                                smooth[(spk_start + fi) % n] = (
                                    (1.0 - t) * wall_L + t * wall_R)

        # ── 5. Sector distances ───────────────────────────────────────────────
        def sector_min(start_idx: int, end_idx: int) -> float:
            if start_idx <= end_idx:
                return float(np.min(smooth[start_idx:end_idx + 1]))
            return float(np.min(
                np.concatenate([smooth[start_idx:], smooth[:end_idx + 1]])))

        ang_inc = self._eff_inc if self._eff_inc > 0 else 0.01745

        def deg2idx(deg_val: float) -> int:
            return int(abs(math.radians(deg_val)) / ang_inc)

        o_20  = deg2idx(20)
        o_110 = deg2idx(110)

        idx_fwd_right  = (self.FORWARD - o_20)        % n
        idx_fwd_left   = (self.FORWARD + o_20)        % n

        idx_left_front = (self.FORWARD + o_20 + 1)    % n
        idx_left_back  = (self.FORWARD + o_110)       % n

        idx_rgt_front  = (self.FORWARD - o_110)       % n
        idx_rgt_back   = (self.FORWARD - (o_20 + 1))  % n

        idx_back_left  = (self.FORWARD + o_110 + 1)   % n
        idx_back_rgt   = (self.FORWARD - (o_110 + 1)) % n

        fwd_d  = sector_min(idx_fwd_right,  idx_fwd_left)
        left_d = sector_min(idx_left_front, idx_left_back)
        rgt_d  = sector_min(idx_rgt_front,  idx_rgt_back)
        bck_d  = sector_min(idx_back_left,  idx_back_rgt)

        # ── 6. Geometric junction classification ──────────────────────────────
        # FIX 2: SVD line fit replaces scalar threshold lookup table.
        # FIX 2: WALL_RESID_THR = 0.12 m (was 0.07) for arena walls.
        left_fit = self._fit_wall_line(
            smooth, self.LEFT,  n, self._eff_amin, self._eff_inc)
        rgt_fit  = self._fit_wall_line(
            smooth, self.RIGHT, n, self._eff_amin, self._eff_inc)

        left_is_wall = (left_fit is not None and left_fit[2] < WALL_RESID_THR)
        rgt_is_wall  = (rgt_fit  is not None and rgt_fit[2]  < WALL_RESID_THR)

        walls_parallel = (
            left_is_wall and rgt_is_wall and
            abs(left_fit[1] - rgt_fit[1]) < WALL_PARALLEL_THR)

        if fwd_d >= 0.5:
            if left_is_wall and rgt_is_wall and walls_parallel:
                junction_type = "CORRIDOR"
            elif left_is_wall and not rgt_is_wall:
                junction_type = "RIGHT_JUNCTION"
            elif rgt_is_wall and not left_is_wall:
                junction_type = "LEFT_JUNCTION"
            elif not left_is_wall and not rgt_is_wall:
                junction_type = "CROSSROADS"
            else:
                junction_type = "UNKNOWN_OPEN"
        else:
            # Close to front wall (< 0.5m)
            if not left_is_wall and not rgt_is_wall:
                junction_type = "T_JUNCTION"
            elif not left_is_wall and rgt_is_wall:
                junction_type = "LEFT_JUNCTION"   # Left is open
            elif left_is_wall and not rgt_is_wall:
                junction_type = "RIGHT_JUNCTION"  # Right is open
            else:
                junction_type = "DEAD_END"

        # ── 7. Wall alignment error (reuse fits — FIX 3) ─────────────────────
        wall_alignment_error_rad = 0.0
        if junction_type == "CORRIDOR":
            if left_fit is not None and rgt_fit is not None:
                wall_alignment_error_rad = float(
                    (left_fit[1] + rgt_fit[1]) / 2.0)
            elif left_fit is not None:
                wall_alignment_error_rad = float(left_fit[1])
            elif rgt_fit is not None:
                wall_alignment_error_rad = float(rgt_fit[1])

        # ── 8. Threat angles ─────────────────────────────────────────────────
        def sector_argmin_angle(start_idx: int, end_idx: int) -> float:
            """Relative angle (rad, 0=fwd, +ve=left) of closest point."""
            if start_idx <= end_idx:
                local_idx = int(np.argmin(smooth[start_idx:end_idx + 1]))
                abs_idx   = start_idx + local_idx
            else:
                combined  = np.concatenate(
                    [smooth[start_idx:], smooth[:end_idx + 1]])
                local_idx = int(np.argmin(combined))
                abs_idx   = (start_idx + local_idx) % n
            raw_ang = self._eff_amin + abs_idx   * self._eff_inc
            fwd_ang = self._eff_amin + self.FORWARD * self._eff_inc
            rel     = raw_ang - fwd_ang
            while rel >  math.pi: rel -= 2 * math.pi
            while rel < -math.pi: rel += 2 * math.pi
            return rel

        fwd_threat_ang  = sector_argmin_angle(idx_fwd_right,  idx_fwd_left)
        left_threat_ang = sector_argmin_angle(idx_left_front, idx_left_back)
        rgt_threat_ang  = sector_argmin_angle(idx_rgt_front,  idx_rgt_back)
        bck_threat_ang  = sector_argmin_angle(idx_back_left,  idx_back_rgt)

        # ── 9. U-shape / closed-row detection ────────────────────────────────
        is_u = (fwd_d  < self.closed_depth and
                left_d < self.closed_depth and
                rgt_d  < self.closed_depth)

        # ── 10. Lateral corridor width ────────────────────────────────────────
        lateral_width = left_d + rgt_d

        # ── 11. Opening (gap) detection ───────────────────────────────────────
        opening_angles  = []
        opening_widths  = []
        opening_pass    = []
        opening_shapes  = []
        opening_dists   = []

        in_gap    = False
        gap_start = 0
        gap_dists = []

        # Start on a solid sample to avoid splitting a wrap-around gap
        start_i = 0
        for i in range(n):
            if float(smooth[i]) <= self.open_thresh:
                start_i = i
                break

        for offset in range(n + 1):
            if offset == n:
                if in_gap:
                    self._record_gap(
                        gap_start, start_i + n - 1, gap_dists, n,
                        opening_angles, opening_widths,
                        opening_pass, opening_shapes, opening_dists, smooth)
                break

            i           = (start_i + offset) % n
            unwrapped_i = start_i + offset
            r           = float(smooth[i])

            if r > self.open_thresh:
                if not in_gap:
                    in_gap    = True
                    gap_start = unwrapped_i
                    gap_dists = []
                gap_dists.append(r)
            else:
                if in_gap:
                    in_gap = False
                    self._record_gap(
                        gap_start, unwrapped_i - 1, gap_dists, n,
                        opening_angles, opening_widths,
                        opening_pass, opening_shapes, opening_dists, smooth)

        # ── 12. Best opening (widest passable gap) ────────────────────────────
        best_idx = -1
        best_w   = 0.0
        for j, (w, p) in enumerate(zip(opening_widths, opening_pass)):
            if p and w > best_w:
                best_w   = w
                best_idx = j

        # ── 13. Cylinder detection ────────────────────────────────────────────
        angles_all  = np.array(
            [self._eff_amin + i * self._eff_inc for i in range(n)])
        fwd_ang_val = self._eff_amin + self.FORWARD * self._eff_inc
        rel_angles  = angles_all - fwd_ang_val
        xs_all      = smooth * np.cos(rel_angles)
        ys_all      = smooth * np.sin(rel_angles)

        cylinder_detected    = False
        cylinder_bearing_rad = 0.0
        best_r2              = 0.0

        for i in range(0, n, 8):
            idx   = [(i + j) % n for j in range(15)]
            x_win = xs_all[idx]
            y_win = ys_all[idx]
            z     = x_win ** 2 + y_win ** 2
            M     = np.c_[x_win, y_win, np.ones(15)]
            try:
                p, _, _, _ = np.linalg.lstsq(M, z, rcond=None)
                xc, yc = p[0] / 2, p[1] / 2
                r_sq   = p[2] + xc ** 2 + yc ** 2
                if r_sq <= 0:
                    continue
                r_opt  = math.sqrt(r_sq)
                ri     = np.sqrt((x_win - xc) ** 2 + (y_win - yc) ** 2)
                ss_res = np.sum((ri - r_opt) ** 2)
                ss_tot = np.sum(
                    (x_win - np.mean(x_win)) ** 2 +
                    (y_win - np.mean(y_win)) ** 2)
                r2     = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                center_dist = math.hypot(xc, yc)
                if 0.05 <= r_opt <= 0.30 and center_dist < 3.0 and r2 > 0.90:
                    if r2 > best_r2:
                        best_r2              = float(r2)
                        cylinder_detected    = True
                        cylinder_bearing_rad = float(math.atan2(yc, xc))
            except Exception:
                pass

        # ── 14. Build and publish ─────────────────────────────────────────────
        out                          = LidarAnalysis()
        out.header                   = msg.header
        out.junction_type            = junction_type
        out.wall_alignment_error_rad = wall_alignment_error_rad
        out.cylinder_detected        = cylinder_detected
        out.cylinder_bearing_rad     = cylinder_bearing_rad
        out.forward_dist             = fwd_d
        out.left_dist                = left_d
        out.right_dist               = rgt_d
        out.back_dist                = bck_d
        out.forward_threat_angle     = fwd_threat_ang
        out.left_threat_angle        = left_threat_ang
        out.right_threat_angle       = rgt_threat_ang
        out.back_threat_angle        = bck_threat_ang
        out.is_u_shape               = is_u
        out.corridor_width_lateral   = lateral_width
        out.opening_angles_rad       = [float(a) for a in opening_angles]
        out.opening_widths_m         = [float(w) for w in opening_widths]
        out.opening_passable         = list(opening_pass)
        out.opening_shapes           = list(opening_shapes)
        out.opening_distances_m      = [float(d) for d in opening_dists]
        out.best_opening_idx         = best_idx
        out.no_forward_passage       = (fwd_d < ROBOT_WIDTH_M * 1.2)

        self.pub.publish(out)

        self.get_logger().debug(
            f'fwd={fwd_d:.2f}  l={left_d:.2f}  r={rgt_d:.2f}  b={bck_d:.2f}  '
            f'junc={junction_type}  U={is_u}  gaps={len(opening_angles)}  '
            f'best={best_idx}')

    # ── Geometric wall-line fitting ───────────────────────────────────────────
    def _fit_wall_line(self, smooth, center_idx, n, eff_amin, eff_inc,
                       half_width=20, min_pts=10, range_max=29.0):
        """SVD line fit over a sector centred on center_idx.

        Returns (perp_dist_m, normal_angle_rad, mean_residual_m) or None.
          perp_dist_m      : perpendicular distance from robot origin to wall
          normal_angle_rad : wall normal direction in robot frame
          mean_residual_m  : mean perpendicular residual — low = straight wall
        Only points with range < range_max * 0.95 are included (excludes
        open-air max-range readings).
        """
        pts_x  = []
        pts_y  = []
        thresh = range_max * 0.95
        for k in range(-half_width, half_width + 1):
            i   = (center_idx + k) % n
            ang = eff_amin + i * eff_inc
            r   = float(smooth[i])
            if r < thresh:
                pts_x.append(r * math.cos(ang))
                pts_y.append(r * math.sin(ang))

        if len(pts_x) < min_pts:
            return None

        xs  = np.array(pts_x, dtype=np.float32)
        ys  = np.array(pts_y, dtype=np.float32)
        cx, cy = float(xs.mean()), float(ys.mean())
        M   = np.column_stack((xs - cx, ys - cy))
        try:
            _, _, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Vt[1] = minor singular vector = wall normal direction
        nx, ny     = float(Vt[1, 0]), float(Vt[1, 1])
        perp_dist  = abs(nx * cx + ny * cy)
        normal_ang = math.atan2(ny, nx)
        residuals  = np.abs((xs - cx) * nx + (ys - cy) * ny)
        mean_resid = float(residuals.mean())
        return perp_dist, normal_ang, mean_resid

    # ── Record a detected gap ────────────────────────────────────────────────
    def _record_gap(self, start_idx, end_idx, gap_dists, n_samples,
                    angles_out, widths_out, pass_out, shapes_out, dists_out, smooth):
        idx_right_bound = (start_idx - 1) % n_samples
        idx_left_bound  = (end_idx   + 1) % n_samples

        R1 = float(smooth[idx_right_bound])
        R2 = float(smooth[idx_left_bound])

        n_gap     = end_idx - start_idx + 1
        ang_width = min((n_gap + 2) * self._eff_inc, 2 * math.pi)

        # Law of Cosines: true Euclidean gap width
        phys_width = math.sqrt(
            R1 ** 2 + R2 ** 2 - 2.0 * R1 * R2 * math.cos(ang_width))
        passable   = phys_width >= self.passable_thr

        # Gap shape classification (Dmax method)
        gap_indices = [(start_idx + k) % n_samples for k in range(n_gap)]
        if len(gap_indices) >= 3:
            X1 = R1 * math.cos(
                self._eff_amin + idx_right_bound * self._eff_inc)
            Y1 = R1 * math.sin(
                self._eff_amin + idx_right_bound * self._eff_inc)
            X2 = R2 * math.cos(
                self._eff_amin + idx_left_bound  * self._eff_inc)
            Y2 = R2 * math.sin(
                self._eff_amin + idx_left_bound  * self._eff_inc)
            baseline_len = math.hypot(X2 - X1, Y2 - Y1) + 1e-6

            dmax = 0.0
            for gi in gap_indices:
                ang_i = self._eff_amin + gi * self._eff_inc
                Xp    = smooth[gi] * math.cos(ang_i)
                Yp    = smooth[gi] * math.sin(ang_i)
                d_perp = abs(
                    (Y2 - Y1) * Xp - (X2 - X1) * Yp +
                    X2 * Y1 - Y2 * X1) / baseline_len
                dmax = max(dmax, d_perp)

            if dmax >= 0.2 * baseline_len:
                shape = 'curved'
            elif abs(R1 - R2) > 0.4:
                shape = 'asymmetric'
            else:
                shape = 'flat'
        else:
            shape = 'flat'

        # Centre angle relative to forward
        centre_i   = ((start_idx + end_idx) // 2) % n_samples
        centre_ang = self._eff_amin + centre_i    * self._eff_inc
        fwd_ang    = self._eff_amin + self.FORWARD * self._eff_inc
        rel_angle  = centre_ang - fwd_ang
        while rel_angle >  math.pi: rel_angle -= 2 * math.pi
        while rel_angle < -math.pi: rel_angle += 2 * math.pi

        angles_out.append(rel_angle)
        widths_out.append(phys_width)
        pass_out.append(passable)
        shapes_out.append(shape)
        dists_out.append(float(np.median(gap_dists)))

        self.get_logger().debug(
            f'  gap: angle={math.degrees(rel_angle):.1f}°  '
            f'width={phys_width:.3f}m  shape={shape}  passable={passable}')


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
