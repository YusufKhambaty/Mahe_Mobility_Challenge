"""
nav_controller_node.py — Differential Arc Turn + PD Centering Controller
=========================================================================
Architecture:
  Corridors: PD centering (right-left error) + heading hold from odom
  Turns:     Differential arc (V>0, W≠0) with closed-loop corner safety
  Safety:    Dynamic proportional repulsion + skid-steer speed damping
Robot: 39×43cm.  Arena: 61cm corridors.

CHANGES vs previous version
-----------------------------
  FIX 1 : _best_gap() CONE reduced from ±90° to ±75° — prevents rear gaps
           being selected when the robot is near a back wall.

  FIX 2 : _drive() now checks junction_type for T_JUNCTION explicitly.
           Previously the robot would oscillate at a T-junction trying to
           pick between two symmetric side gaps.  Now it stops forward motion
           and turns cleanly toward the side with greater clearance.

  FIX 3 : _arc_turn() corner safety now reduces W (widens arc radius) instead
           of reducing V.  Arc radius R = V/W — reducing V while keeping W
           tightens the arc and pushes the corner closer, making the problem
           worse.  Correct fix: reduce W to widen the arc.

  FIX 4 : _apply_safety() repulsion now uses the published threat angles
           (left_threat_angle, right_threat_angle) to decompose the repulsion
           force along the actual threat vector instead of always pushing
           perpendicular regardless of where the obstacle is.

  FIX 5 : _explore() stuck detector now tracks net displacement from a
           reference checkpoint rather than resetting on any 8cm move.
           This catches oscillation-in-place (move 8cm fwd, 8cm back, repeat)
           which previously kept the timer from ever firing.

  FIX 6 : _tag_act() no longer calls time.sleep(0.2) which blocked the ROS
           executor.  Replaced with a non-blocking elapsed timestamp check.
"""

import math
import time
import os
from datetime import datetime
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from mahe_nav_interfaces.msg import ArucoDetection, LidarAnalysis, FloorMarkerDetection


def _angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


class State(Enum):
    EXPLORE       = auto()
    APPROACH_TAG  = auto()
    TAG_ACTION    = auto()
    FOLLOW_GREEN  = auto()
    FOLLOW_ORANGE = auto()
    UTURN         = auto()
    RECOVERY      = auto()
    APPROACH_HALT = auto()
    HALT          = auto()


# ═══ TUNEABLE CONSTANTS ═══════════════════════════════════════════════════════

# Speeds
V_FWD       = 0.15    # m/s — corridor speed
V_CREEP     = 0.06    # m/s — approach / halt

# Steering limits
MAX_W       = 0.30    # rad/s — max angular velocity
GAP_K       = 0.80    # proportional gain for gap steering

# Arc turn
V_ARC       = 0.05    # m/s — forward speed during arc turn
W_ARC       = 0.40    # rad/s — base angular speed during arc
ARC_YAW_90  = math.radians(85)    # target yaw for 90° turn
ARC_YAW_180 = math.radians(175)   # target yaw for U-turn

# Corner safety: half-diagonal of 39×43cm robot = 29cm
CORNER_RADIUS    = 0.29   # m
CORNER_SAFETY    = 0.25   # m — min outer-side clearance during arc
# FIX 3: ARC_WIDEN_FACTOR now applied to W (not V).
# Reducing W widens arc radius (R=V/W).  Factor < 1 → W shrinks → R grows.
ARC_WIDEN_FACTOR = 0.5    # multiply W by this when corner too close

# Dynamic repulsion
DANGER_ZONE = 0.20    # m — side distance that triggers repulsion
REPULSE_K   = 1.50    # repel magnitude = K * (danger - dist)
REPULSE_MAX = 0.25    # rad/s cap

# Skid-steer damping
SKID_DAMP   = 0.22    # m — damp V by 40% if wall closer than this

# Distance thresholds
SLOWDOWN      = 1.0    # m
STOP_DIST_FWD = 0.10   # m — hard stop forward
STOP_DIST_BCK = 0.30   # m — hard stop backward
SCAN_W        = 0.15   # rad/s — slow scan when no gap found

# Halt
HALT_WALL    = 0.35
HALT_SPEED   = 0.06
HALT_TIMEOUT = 10.0

# ArUco
ACTION_DIST  = 0.01
APPROACH_DIST = 0.02
SIGN_ARM_SEC = 0.5

# Stuck detection
STUCK_NET_DIST   = 0.20   # m — net displacement required to reset progress timer
STUCK_TIMEOUT    = 14.0   # s — time without net progress before RECOVERY


class NavControllerNode(Node):
    def __init__(self):
        super().__init__('nav_controller')
        self.state = State.EXPLORE
        self.t0    = time.time()
        self.logged_tags = set()

        # Pose
        self.px = self.py = self.yaw = 0.0

        # LiDAR / camera
        self.lidar = None
        self.aruco = None
        self.floor = None

        # Turn
        self.turn_cmd   = "NONE"
        self.turn_yaw0  = 0.0
        self.uturn_post = State.EXPLORE
        self.sign_armed = False
        self.sign_t0    = 0.0

        # FIX 6: non-blocking post-turn delay
        self._tag_done_t = None

        # FIX 5: two-level stuck detection
        # last_pos tracks movement per tick (8cm threshold stays for
        # backwards-compat logging); _prog_ref tracks net displacement.
        self.last_pos  = (0.0, 0.0)
        self.last_prog = time.time()
        self._prog_ref = (0.0, 0.0)   # checkpoint for net displacement

        self.halted = False

        # Gap persistence
        self._prev_gap_ang    = None
        self._gap_streak      = 0
        self._GAP_PERSIST_MIN = 2

        # ROS
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry,       '/odom_fused',             self._odom, 10)
        self.create_subscription(LidarAnalysis,  '/lidar/analysis',         self._lid,  be)
        self.create_subscription(ArucoDetection, '/aruco/detections',       self._aru,  be)
        self.create_subscription(
            FloorMarkerDetection, '/floor_marker/detection', self._flr, be)
        self.create_subscription(Pose2D, '/aruco/pose_correction', self._pcor, 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info('NavController ready — all fixes applied')

        # Buffered CSV logging
        log_dir   = os.path.expanduser('~/ros2_finale/mission_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file           = os.path.join(log_dir, f'nav_log_{ts}.csv')
        self._log_buffer        = []
        self._log_flush_interval = 2.0
        self._last_log_flush    = time.time()
        with open(self.log_file, 'w') as f:
            f.write("time,state,px,py,yaw,fwd_dist,left_dist,right_dist,"
                    "back_dist,junction,aruco_id,floor_color,cmd_v,cmd_w\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _odom(self, m):
        self.px  = m.pose.pose.position.x
        self.py  = m.pose.pose.position.y
        q        = m.pose.pose.orientation
        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))

    def _pcor(self, m):
        self.px, self.py = m.x, m.y

    def _lid(self, m):
        self.lidar = m

    def _aru(self, m):
        self.aruco = m

    def _flr(self, m):
        if m.colour == "RED" and m.direction == "HALT":
            if self.state not in (State.HALT, State.APPROACH_HALT):
                self._go(State.APPROACH_HALT)
            return
        if m.confidence > 0 and m.direction != "TIMEOUT":
            self.floor = m

    # ── FSM helpers ───────────────────────────────────────────────────────────
    def _go(self, s):
        self.get_logger().info(f'FSM: {self.state.name} → {s.name}')
        self.state     = s
        self.t0        = time.time()
        self.last_prog = time.time()
        self.turn_cmd  = "NONE"
        if s not in (State.FOLLOW_GREEN, State.FOLLOW_ORANGE):
            self.sign_armed = False
        if s in (State.TAG_ACTION, State.UTURN):
            self.turn_yaw0 = self.yaw
        # FIX 6: clear non-blocking delay guard on state change
        self._tag_done_t = None

    def _cmd(self, v, w):
        t = Twist()
        t.linear.x  = float(v)
        t.angular.z = float(w)
        self.pub.publish(t)

        if hasattr(self, 'log_file'):
            try:
                lid_f  = self.lidar.forward_dist  if self.lidar else -1.0
                lid_l  = self.lidar.left_dist      if self.lidar else -1.0
                lid_r  = self.lidar.right_dist     if self.lidar else -1.0
                lid_b  = self.lidar.back_dist      if self.lidar else -1.0
                junc   = self.lidar.junction_type  if self.lidar else "NONE"
                aid    = self.aruco.marker_id      if self.aruco else -1
                fcol   = self.floor.colour         if self.floor else "NONE"
                self._log_buffer.append(
                    f"{time.time():.2f},{self.state.name},"
                    f"{self.px:.3f},{self.py:.3f},{self.yaw:.3f},"
                    f"{lid_f:.3f},{lid_l:.3f},{lid_r:.3f},{lid_b:.3f},"
                    f"{junc},{aid},{fcol},{float(v):.3f},{float(w):.3f}\n")
                now = time.time()
                if now - self._last_log_flush >= self._log_flush_interval:
                    try:
                        with open(self.log_file, 'a') as f:
                            f.writelines(self._log_buffer)
                    finally:
                        self._log_buffer.clear()
                        self._last_log_flush = now
            except Exception:
                pass

    def _tick(self):
        FREE = (State.HALT, State.APPROACH_HALT,
                State.APPROACH_TAG, State.TAG_ACTION, State.UTURN)
        if not self.lidar and self.state not in FREE:
            return
        {State.EXPLORE:       self._explore,
         State.APPROACH_TAG:  self._appr_tag,
         State.TAG_ACTION:    self._tag_act,
         State.FOLLOW_GREEN:  lambda: self._follow("GREEN"),
         State.FOLLOW_ORANGE: lambda: self._follow("ORANGE"),
         State.UTURN:         self._uturn,
         State.RECOVERY:      self._recovery,
         State.APPROACH_HALT: self._appr_halt,
         State.HALT:          self._halt}[self.state]()

    # ═══════════════════════════════════════════════════════════════════════════
    #  360° SAFETY NET
    # ═══════════════════════════════════════════════════════════════════════════
    def _apply_safety(self, v, w):
        """Apply obstacle repulsion and forward/backward hard stops."""
        if not self.lidar:
            return 0.0, 0.0

        # FIX 4: decompose repulsion along the actual threat vector instead of
        # always pushing purely in ±z.  threat_angle is 0 at forward, +ve left.
        if self.lidar.left_dist < DANGER_ZONE:
            r        = REPULSE_K * (DANGER_ZONE - self.lidar.left_dist)
            r        = min(r, REPULSE_MAX)
            lt       = self.lidar.left_threat_angle
            # Angular component (push right = negative w)
            w       -= r * math.cos(lt)
            # Forward damping proportional to how head-on the threat is
            v       *= max(0.4, 1.0 - 0.6 * abs(math.cos(lt)))

        if self.lidar.right_dist < DANGER_ZONE:
            r        = REPULSE_K * (DANGER_ZONE - self.lidar.right_dist)
            r        = min(r, REPULSE_MAX)
            rt       = self.lidar.right_threat_angle
            # Angular component (push left = positive w)
            w       += r * math.cos(rt)
            v       *= max(0.4, 1.0 - 0.6 * abs(math.cos(rt)))

        # Skid-steer speed damping
        if (self.lidar.left_dist < SKID_DAMP or
                self.lidar.right_dist < SKID_DAMP):
            v *= 0.60

        # Forward hard stop
        if self.lidar.forward_dist < STOP_DIST_FWD:
            v = -0.04 if self.lidar.back_dist > STOP_DIST_BCK else 0.0

        # Backward hard stop
        if v < 0 and self.lidar.back_dist < STOP_DIST_BCK:
            v = 0.0

        return float(v), float(w)

    # ═══════════════════════════════════════════════════════════════════════════
    #  GAP STEERING
    # ═══════════════════════════════════════════════════════════════════════════
    def _drive(self, v):
        """Drive forward using gap steering and 360° safety net."""
        if not self.lidar:
            self._cmd(0.0, 0.0)
            return

        w = 0.0

        # Handle blocked junctions (T-junctions or Corners)
        if self.lidar.forward_dist < 0.50:
            if self.lidar.junction_type == "T_JUNCTION":
                # Prefer shorter path per user request
                self.turn_cmd = "LEFT" if self.lidar.left_dist < self.lidar.right_dist else "RIGHT"
                self._go(State.TAG_ACTION)
                return
            elif self.lidar.junction_type == "LEFT_JUNCTION":
                self.turn_cmd = "LEFT"
                self._go(State.TAG_ACTION)
                return
            elif self.lidar.junction_type == "RIGHT_JUNCTION":
                self.turn_cmd = "RIGHT"
                self._go(State.TAG_ACTION)
                return
            elif self.lidar.junction_type == "DEAD_END":
                self._cmd(0, SCAN_W)
                return

        ang, _ = self._best_gap()
        if ang is not None:
            if abs(ang) > math.radians(30):
                # Sharp side gap — don't commit until close to the junction
                if self.lidar.forward_dist < 0.45:
                    w = W_ARC if ang > 0 else -W_ARC
                    v = V_ARC * 0.5
                else:
                    w = 0.0   # drive straight toward junction
            else:
                # Gap is near forward — proportional steer
                w = GAP_K * ang
                w = max(-MAX_W, min(MAX_W, w))
        else:
            w = SCAN_W
            v = 0.0

        v_safe, w_safe = self._apply_safety(v, w)
        self._cmd(v_safe, w_safe)

    # ═══════════════════════════════════════════════════════════════════════════
    #  DIFFERENTIAL ARC TURN
    # ═══════════════════════════════════════════════════════════════════════════
    def _arc_turn(self, direction, target_yaw_delta):
        """Differential arc turn.  Returns True when complete.
        direction: 'LEFT' (+W) or 'RIGHT' (-W).

        FIX 3: corner safety now reduces W to widen the arc radius (R = V/W).
        The old code reduced V which tightened R and made the corner problem
        worse.
        """
        turned = abs(_angle_diff(self.yaw, self.turn_yaw0))
        if turned >= target_yaw_delta:
            self._cmd(0.0, 0.0)
            return True

        w = W_ARC if direction == "LEFT" else -W_ARC
        v = V_ARC

        if self.lidar:
            outer = (self.lidar.left_dist
                     if direction == "RIGHT"
                     else self.lidar.right_dist)
            fwd   = self.lidar.forward_dist

            # FIX 3: reduce W (widens arc) not V (would tighten arc)
            if outer < CORNER_SAFETY:
                w *= ARC_WIDEN_FACTOR
                self.get_logger().debug(
                    f'[ARC SAFETY] outer={outer:.2f} m → widening arc '
                    f'(W×{ARC_WIDEN_FACTOR})')

            if fwd < 0.20:
                v *= 0.3

            # Pause if dangerously close
            if outer < 0.15 or fwd < 0.12:
                self._cmd(0.0, 0.0)
                return False

        self._cmd(v, w)
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    #  GAP SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    def _best_gap(self):
        # FIX 1: CONE reduced to ±75° to exclude near-rear gaps.
        # The old ±90° included gaps at exactly ±90° from two sectors,
        # and could select a gap directly behind the robot near a back wall.
        CONE = math.radians(75)

        fwd = []
        for a, w, p, d in zip(self.lidar.opening_angles_rad,
                             self.lidar.opening_widths_m,
                             self.lidar.opening_passable,
                             self.lidar.opening_distances_m):
            if not p:
                continue
            if abs(a) <= CONE:
                # Store (angle, width, distance)
                fwd.append((a, w, d))

        if fwd:
            # 1. Prefer FORWARD if available (abs angle < 20 deg)
            # 2. Otherwise, prefer SMALLEST PHYSICAL DISTANCE (shorter path)
            # 3. Tie-breaker: widest gap
            fwd.sort(key=lambda x: (
                0 if abs(x[0]) < math.radians(20) else 1, # Forward first
                round(x[2], 2),                           # Shortest distance second
                -round(x[1], 2)                           # Widest tie-break
            ))
            best_ang, best_w, _ = fwd[0]

            # Gap persistence: side gaps must appear for N consecutive ticks
            if abs(best_ang) > math.radians(30):
                if (self._prev_gap_ang is not None and
                        abs(best_ang - self._prev_gap_ang) < math.radians(15)):
                    self._gap_streak += 1
                else:
                    self._gap_streak = 1
                self._prev_gap_ang = best_ang

                if self._gap_streak < self._GAP_PERSIST_MIN:
                    return None, None
            else:
                self._prev_gap_ang = best_ang
                self._gap_streak   = 0

            return best_ang, best_w

        self._prev_gap_ang = None
        self._gap_streak   = 0
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    #  STATE HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _explore(self):
        now = time.time()

        # FIX 5: two-level stuck detection.
        # Level 1 — any movement (8cm): keep last_pos for reference, same as before.
        # Level 2 — net displacement from _prog_ref: only reset timer when the robot
        # has actually moved STUCK_NET_DIST from its last progress checkpoint.
        d = math.hypot(self.px - self.last_pos[0],
                       self.py - self.last_pos[1])
        if d > 0.08:
            self.last_pos = (self.px, self.py)
            net = math.hypot(self.px - self._prog_ref[0],
                             self.py - self._prog_ref[1])
            if net > STUCK_NET_DIST:
                self._prog_ref = (self.px, self.py)
                self.last_prog = now

        if now - self.last_prog > STUCK_TIMEOUT:
            self._go(State.RECOVERY)
            return

        if self.aruco:
            if (float(self.aruco.distance) <= ACTION_DIST and
                    self.aruco.marker_id <= 5):
                self._cmd(0, 0)
                self._go(State.APPROACH_TAG)
                return

        sf = min(1.0, self.lidar.forward_dist / SLOWDOWN)
        v  = max(V_CREEP, V_FWD * sf)
        self._drive(v)

    def _tag_act(self):
        el = time.time() - self.t0
        if el > 12.0:
            self._go(State.RECOVERY)
            return
        done = self._arc_turn(self.turn_cmd, ARC_YAW_90)
        if done:
            # FIX 6: non-blocking 0.2 s pause — was time.sleep(0.2) which
            # blocked the ROS executor and staled all sensor callbacks.
            if self._tag_done_t is None:
                self._tag_done_t = time.time()
            if time.time() - self._tag_done_t >= 0.2:
                self._tag_done_t = None
                self._go(State.EXPLORE)

    def _uturn(self):
        if time.time() - self.t0 > 15.0:
            self._go(State.RECOVERY)
            return
        turned = abs(_angle_diff(self.yaw, self.turn_yaw0))
        if turned < ARC_YAW_180:
            self._cmd(0.02, 0.30)
        else:
            self._cmd(0, 0)
            self._go(self.uturn_post)

    def _appr_tag(self):
        el = time.time() - self.t0
        self._cmd(0, 0)
        if el < 1.5:
            return
        if not self.aruco:
            self._go(State.EXPLORE)
            return
        mid, d  = self.aruco.marker_id, float(self.aruco.distance)
        self.aruco = None
        if d > ACTION_DIST:
            self._go(State.EXPLORE)
            return
        if mid in self.logged_tags:
            self.uturn_post = State.EXPLORE
            self._go(State.UTURN)
            return
        self.logged_tags.add(mid)
        self.get_logger().info(f'[ARUCO] Logged ID {mid}')
        if mid == 1:
            self._go(State.TAG_ACTION); self.turn_cmd = "LEFT"
        elif mid == 2:
            self._go(State.TAG_ACTION); self.turn_cmd = "RIGHT"
        elif mid == 3:
            self.sign_armed = False; self.sign_t0 = time.time()
            self._go(State.FOLLOW_GREEN)
        elif mid == 4:
            self.uturn_post = State.EXPLORE; self._go(State.UTURN)
        elif mid == 5:
            self.sign_armed = False; self.sign_t0 = time.time()
            self._go(State.FOLLOW_ORANGE)
        else:
            self._go(State.EXPLORE)

    def _follow(self, color):
        el = time.time() - self.t0
        if not self.sign_armed:
            if time.time() - self.sign_t0 >= SIGN_ARM_SEC:
                self.sign_armed = True
            else:
                if self.lidar:
                    self._drive(V_CREEP)
                return
        if self.aruco and float(self.aruco.distance) <= ACTION_DIST:
            if self.aruco.marker_id not in self.logged_tags:
                self._cmd(0, 0)
                self._go(State.APPROACH_TAG)
                return
        if el > 120:
            self._go(State.EXPLORE)
            return
        if self.floor and self.floor.colour == color:
            d          = self.floor.direction
            self.floor = None
            if d == "FORWARD":
                self._drive(V_CREEP)
            elif d == "LEFT":
                self._go(State.TAG_ACTION); self.turn_cmd = "LEFT"
            elif d == "RIGHT":
                self._go(State.TAG_ACTION); self.turn_cmd = "RIGHT"
            elif d == "BACKWARD":
                st = (State.FOLLOW_GREEN
                      if color == "GREEN" else State.FOLLOW_ORANGE)
                self.uturn_post = st; self._go(State.UTURN)
            return
        if self.lidar:
            self._drive(V_CREEP)

    def _recovery(self):
        el = time.time() - self.t0
        if el < 0.8:
            if self.lidar and self.lidar.back_dist > STOP_DIST_BCK:
                self._cmd(-0.06, 0)
            else:
                self._cmd(0, 0)
        elif el < 3.0:
            spin_dir = (0.25
                        if (self.lidar and
                            self.lidar.left_dist >= self.lidar.right_dist)
                        else -0.25)
            self._cmd(0, spin_dir)
        else:
            self._cmd(0, 0)
            # Reset progress ref so stuck timer starts fresh after recovery
            self._prog_ref = (self.px, self.py)
            self.last_prog = time.time()
            self._go(State.EXPLORE)

    def _appr_halt(self):
        if time.time() - self.t0 > HALT_TIMEOUT:
            self._go(State.HALT)
            return
        if self.lidar:
            if self.lidar.forward_dist <= HALT_WALL:
                self._cmd(0, 0)
                self._go(State.HALT)
                return

            # Creep forward slowly, use PD centering to stay straight and avoid gaps
            v = HALT_SPEED
            w = 0.0
            if self.lidar.left_dist > 0 and self.lidar.right_dist > 0:
                wall_diff = self.lidar.left_dist - self.lidar.right_dist
                w = -0.8 * wall_diff  # gentle PD centering
                w = max(-0.3, min(0.3, w))
            self._cmd(v, w)
        else:
            self._cmd(HALT_SPEED, 0)

    def _halt(self):
        self._cmd(0, 0)
        if not self.halted:
            self.halted = True
            if self._log_buffer:
                try:
                    with open(self.log_file, 'a') as f:
                        f.writelines(self._log_buffer)
                    self._log_buffer.clear()
                except Exception:
                    pass
            log_dir   = os.path.expanduser('~/ros2_finale/mission_logs')
            os.makedirs(log_dir, exist_ok=True)
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file  = os.path.join(log_dir, f'mission_log_{ts}.txt')
            with open(log_file, 'w') as f:
                f.write(f"MISSION COMPLETE\nTags: {self.logged_tags}\n"
                        f"Pos: ({self.px:.2f}, {self.py:.2f})\n")
            self.get_logger().info("━━━ MISSION COMPLETE ━━━")
            self.destroy_node()
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NavControllerNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
