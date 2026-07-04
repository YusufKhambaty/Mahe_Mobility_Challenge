import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from mahe_nav_interfaces.msg import ArucoDetection, LidarAnalysis, FloorMarkerDetection


# --- FIXED: [BUG 2] ---
ACTION_DISTANCE_THRESHOLD = 0.01  # reduced by 40% from 0.5
APPROACH_TRIGGER_DIST     = 0.02  # reduced proportionately from 0.8

# === PHASE 1: REPLACE TIME-BASED TURNS WITH YAW-TRACKED TURNS ===
def _angle_diff(a, b):
    diff = a - b
    while diff > math.pi: diff -= 2*math.pi
    while diff < -math.pi: diff += 2*math.pi
    return diff


# === PHASE 4: FSM ===
class State(Enum):
    EXPLORE       = auto()
    APPROACH_TAG  = auto()
    TAG_ACTION    = auto()
    FOLLOW_GREEN  = auto()
    FOLLOW_ORANGE = auto()
    UTURN         = auto()
    RECOVERY      = auto()
    APPROACH_HALT = auto()   # creep forward after RED tile until safe wall distance
    HALT          = auto()


# ── Exploration speeds ───────────────────────────────────────────────────────
V_MAX  = 0.30   # m/s — normal forward speed
V_MIN  = 0.10   # m/s — creep

# ── Distance thresholds ──────────────────────────────────────────────────────
SLOWDOWN_DIST  = 1.5    # m — begin ramping down
STOP_DIST      = 0.35   # m — hard-stop ahead

# ── Halt approach (RED tile) ─────────────────────────────────────────────────
HALT_WALL_DIST   = 0.35   # m — stop this far from wall
HALT_CREEP_SPEED = 0.10   # m/s — slow forward creep into halt zone
HALT_APPROACH_TIMEOUT = 10.0  # s — safety: force halt even if wall never close

# ── Exploration angular rates ────────────────────────────────────────────────
W_TURN = 0.50   # rad/s — gentle spin for gap-finding
W_SCAN = 0.35   # rad/s — very slow scan

# Turn angular rates
W_SIGN_TURN = 0.70   # rad/s
W_SIGN_SPIN = 0.40   # rad/s — pure spin (linear.x=0); higher = faster, turning radius stays zero

V_ARC                     = 0.08  # m/s  — forward creep during LEFT/RIGHT arc turns
TURN_AHEAD_SIDE_THRESH    = 0.40  # m    — minimum side clearance before starting a LEFT/RIGHT turn
TURN_DURING_FWD_ABORT     = 0.20  # m    — hard abort if wall ahead and behind during pivot
TURN_DURING_FWD_SLOW      = 0.30  # m    — halve angular rate if wall approaching during pivot
SIGN_ARM_DELAY_SEC        = 0.5   # s    — delay before consuming sign detections after tag 2/4

# === PORTED: ARUCO POSE CORRECTION === 
# Removed MARKER_WORLD_POS dict - handled in aruco_detector_node

class NavControllerNode(Node):
    def __init__(self):
        super().__init__('nav_controller')

        # === PHASE 4: FSM ===
        self.state = State.EXPLORE
        self.state_start_time = time.time()
        self.logged_tags = set()

        self.pose_x = self.pose_y = self.pose_yaw = 0.0
        self.lidar = None
        self.latest_sign = None
        self.latest_aruco = None

        self.PASSABLE_THR = 0.525

        # Turn execution / state trackers
        self.active_turn_cmd  = "NONE"   # NONE, LEFT, RIGHT
        self.uturn_post_state = State.EXPLORE
        self.indicator_blue_arrow = False
        self.last_pos = (0.0, 0.0)
        self.last_progress_time = time.time()
        self.sign_follow_armed = False
        self.sign_arm_start_time = 0.0

        # === PHASE 1: ADD PD CORRIDOR CENTERING ===
        self.last_wall_diff = 0.0
        self.last_center_time = time.time()
        
        # === PHASE 1: REPLACE TIME-BASED TURNS WITH YAW-TRACKED TURNS ===
        self.turn_start_yaw = 0.0

        # [FIX 2] Proper halt flag in __init__ instead of hasattr()
        self.has_halted = False

        # ROS 2 interfaces
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        best_effort  = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry,       '/odom_fused',       self._odom_cb,  10)
        self.create_subscription(LidarAnalysis,  '/lidar/analysis',   self._lidar_cb, best_effort)
        # self.create_subscription(SignDetection,  '/sign_detection',   self._sign_cb,  best_effort)
        self.create_subscription(ArucoDetection, '/aruco/detections', self._aruco_cb, best_effort)

        # === CV INTEGRATION: REPLACE OLD SIGN SUBSCRIBER ===
        self.latest_floor_marker = None
        self.create_subscription(
            FloorMarkerDetection,
            '/floor_marker/detection',
            self._floor_marker_cb,
            best_effort)
        
        # === PORTED: ARUCO POSE CORRECTION ===
        self.create_subscription(Pose2D, '/aruco/pose_correction', self._pose_correction_cb, 10)

        self.create_timer(0.1, self._fsm_tick)
        self.get_logger().info('NavController: Phase 4 FSM Active')


    # ── Callbacks (Pure State Modifiers) ─────────────────────────────────────

    def _odom_cb(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # === PORTED: ARUCO POSE CORRECTION ===
    def _pose_correction_cb(self, msg):
        self.pose_x = msg.x
        self.pose_y = msg.y

    def _lidar_cb(self, msg):
        self.lidar = msg

    # DEPRECATED — old SignDetection topic, no longer used
    def _sign_cb_DEPRECATED(self, msg):
        self.latest_sign = msg
        sign_type = msg.sign_type.upper() if msg.sign_type else "NONE"

        # Phase 4 Rule 1: Massive Red tile check via pixel width (>70px ~ 5000px sq)
        if sign_type == "GOAL" and (msg.pixel_width * msg.pixel_width > 5000):
            if self.state != State.HALT:
                self.get_logger().info("━━━ [STATUS] MASSIVE RED TILE DETECTED ━━━")
                self._transition(State.HALT)

    # === CV INTEGRATION: REPLACE OLD SIGN SUBSCRIBER ===
    def _floor_marker_cb(self, msg: FloorMarkerDetection):
        # Red tile HALT — highest priority, handle immediately
        if msg.colour == "RED" and msg.direction == "HALT":
            if self.state not in (State.HALT, State.APPROACH_HALT):
                self.get_logger().info("━━━ [CV] RED TILE DETECTED — APPROACHING HALT ━━━")
                self._transition(State.APPROACH_HALT)
            return
        # Store normal CV directions
        if msg.confidence > 0.0 and msg.direction != "TIMEOUT":
            self.latest_floor_marker = msg

    def _aruco_cb(self, msg: ArucoDetection):
        self.latest_aruco = msg


    # ── FSM Dispatcher & Helpers ─────────────────────────────────────────────

    def _transition(self, new_state: State):
        self.get_logger().info(f'Transition: {self.state.name} → {new_state.name}')
        self.state = new_state
        self.state_start_time = time.time()
        self.last_progress_time = time.time()
        self.active_turn_cmd = "NONE"

        if new_state not in (State.FOLLOW_GREEN, State.FOLLOW_ORANGE):
            self.sign_follow_armed = False

        # === PHASE 1: REPLACE TIME-BASED TURNS WITH YAW-TRACKED TURNS ===
        if new_state in (State.TAG_ACTION, State.UTURN):
            self.turn_start_yaw = self.pose_yaw

    def _publish_vel(self, v: float, w: float):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _move_sign(self, v: float, w: float):
        if self.lidar:
            if (self.lidar.forward_dist < TURN_DURING_FWD_ABORT
                    and self.lidar.back_dist < 0.30):
                self._publish_vel(0.0, 0.0)
                self.state_start_time = time.time() - 8.5
                return
            if self.lidar.forward_dist < TURN_DURING_FWD_SLOW:
                w *= 0.5
        self._publish_vel(v, w)

    def _fsm_tick(self):
        # [FIX 3] Expanded exclusion list — states that must run even without LiDAR data
        LIDAR_FREE_STATES = (State.HALT, State.APPROACH_HALT, State.APPROACH_TAG, State.TAG_ACTION, State.UTURN)
        if not self.lidar and self.state not in LIDAR_FREE_STATES:
            return

        match self.state:
            case State.EXPLORE:       self._handle_explore()
            case State.APPROACH_TAG:  self._handle_approach_tag()
            case State.TAG_ACTION:    self._handle_tag_action()
            case State.FOLLOW_GREEN:  self._handle_follow_green()
            case State.FOLLOW_ORANGE: self._handle_follow_orange()
            case State.UTURN:         self._handle_uturn()
            case State.RECOVERY:      self._handle_recovery()
            case State.APPROACH_HALT: self._handle_approach_halt()
            case State.HALT:          self._handle_halt()


    # ── State Handlers ───────────────────────────────────────────────────────

    def _handle_explore(self):
        # Watchdog for stuck prevention
        now  = time.time()
        dist = math.hypot(self.pose_x - self.last_pos[0], self.pose_y - self.last_pos[1])
        if dist > 0.10:
            self.last_pos = (self.pose_x, self.pose_y)
            self.last_progress_time = now
        elif (now - self.last_progress_time) > 9.0:
            self.get_logger().warn('STUCK: Timeout in EXPLORE')
            self._transition(State.RECOVERY)
            return

        # General Exploration behavior
        speed_factor = min(1.0, self.lidar.forward_dist / SLOWDOWN_DIST)
        v_cmd = max(V_MIN, V_MAX * speed_factor)

        # [FIX 6 Correction] Check ArUco for smooth deceleration
        if self.latest_aruco:
            d   = float(self.latest_aruco.distance)
            mid = self.latest_aruco.marker_id
            if mid <= 5:
                if d <= ACTION_DISTANCE_THRESHOLD:
                    # Phase 4 Rule 3: Publish zero immediately on entering APPROACH_TAG
                    self._publish_vel(0.0, 0.0)
                    self._transition(State.APPROACH_TAG)
                    return
                elif d <= APPROACH_TRIGGER_DIST:
                    v_cmd = V_MIN

        if self.lidar.forward_dist < 2.0:
            is_l_left  = self.lidar.left_dist > 1.5 and self.lidar.right_dist < 1.2
            is_l_right = self.lidar.right_dist > 1.5 and self.lidar.left_dist < 1.2
            if is_l_left or is_l_right:
                v_cmd = V_MIN

        target_angle, _ = self._select_best_gap()
        w_cmd = target_angle if target_angle is not None else W_SCAN

        # === PHASE 1: ADD PD CORRIDOR CENTERING ===
        if self.lidar.wall_alignment_error_rad != 0.0:
            if abs(self.lidar.wall_alignment_error_rad) > math.radians(3):
                w_correction = self.lidar.wall_alignment_error_rad * 0.4
                w_cmd += w_correction

        self._move(v_cmd, w_cmd)

    def _select_best_gap(self):
        FORWARD_CONE = math.radians(35)
        forward_gaps, side_gaps = [], []
        for angle, width, passable in zip(
                self.lidar.opening_angles_rad,
                self.lidar.opening_widths_m,
                self.lidar.opening_passable):
            if not passable:
                continue
            (forward_gaps if abs(angle) <= FORWARD_CONE else side_gaps).append((angle, width))
        if forward_gaps:
            return max(forward_gaps, key=lambda x: x[1])
        if side_gaps:
            return max(side_gaps, key=lambda x: x[1])
        return None, None

    def _move(self, v: float, w: float):
        if v > 0.05 and self.lidar.left_dist < 1.5 and self.lidar.right_dist < 1.5:
            # === PHASE 1: ADD PD CORRIDOR CENTERING ===
            now = time.time()
            dt = max(now - self.last_center_time, 0.01)
            diff  = self.lidar.right_dist - self.lidar.left_dist
            p_term = 0.6 * diff
            d_term = 0.3 * (diff - self.last_wall_diff) / dt
            w += p_term + d_term
            self.last_wall_diff = diff
            self.last_center_time = now
        REPULSION_DIST = 0.30
        if self.lidar.left_dist  < REPULSION_DIST: w -= 0.55
        if self.lidar.right_dist < REPULSION_DIST: w += 0.55
        w = max(min(w, W_TURN * 1.5), -W_TURN * 1.5)
        if self.lidar.forward_dist < STOP_DIST:
            v = -0.06 if self.lidar.back_dist > 0.3 else 0.0
        self._publish_vel(v, w)

    def _handle_approach_tag(self):
        elapsed = time.time() - self.state_start_time

        # Phase 4 Rule 3: Keep robot stopped during entire stabilization window
        self._publish_vel(0.0, 0.0)

        if elapsed < 1.5:
            return  # waiting for camera reading to stabilize

        if not self.latest_aruco:
            self._transition(State.EXPLORE)
            return

        mid = self.latest_aruco.marker_id
        d   = float(self.latest_aruco.distance)

        # [FIX 5] Clear latest_aruco after consuming so stale data is never re-processed
        self.latest_aruco = None

        # Only act if marker is within execution distance
        if d > ACTION_DISTANCE_THRESHOLD:
            self.get_logger().info(f'[ARUCO] ID {mid} detected at {d:.2f}m — too far. Resuming approach.')
            self._transition(State.EXPLORE)
            return

        # Phase 4 Rule 4: Re-detection guard ONLY if distance <= threshold
        if mid in self.logged_tags:
            if d <= ACTION_DISTANCE_THRESHOLD:
                self.get_logger().info(f'[ARUCO] ID {mid} already logged! Forcing 180 U-Turn!')
                self.uturn_post_state = State.EXPLORE
                self._transition(State.UTURN)
            else:
                self.get_logger().info(f'[ARUCO] ID {mid} already logged but far ({d:.2f}m). Ignoring.')
                self._transition(State.EXPLORE)
            return

        # New tag — log and dispatch
        self.logged_tags.add(mid)
        self.get_logger().info(f'[ARUCO] Logged new ID {mid} at {d:.2f}m — executing action!')

        # ID Dispatch Rules
        if mid == 1:
            self.get_logger().info(f"====== [ARUCO COMMAND] Marker {mid} explicitly triggered LEFT Arc Turn ======")
            self.active_turn_cmd = "LEFT"
            self._transition(State.TAG_ACTION)
        elif mid == 2:
            self.get_logger().info(f"====== [ARUCO COMMAND] Marker {mid} explicitly triggered RIGHT Arc Turn ======")
            self.active_turn_cmd = "RIGHT"
            self._transition(State.TAG_ACTION)
        elif mid == 3:
            self.sign_follow_armed = False
            self.sign_arm_start_time = time.time()
            self._transition(State.FOLLOW_GREEN)
        elif mid == 4:
            self.uturn_post_state = State.EXPLORE
            self._transition(State.UTURN)
        elif mid == 5:
            self.sign_follow_armed = False
            self.sign_arm_start_time = time.time()
            self._transition(State.FOLLOW_ORANGE)
        else:
            self._transition(State.EXPLORE)

    def _handle_tag_action(self):
        elapsed = time.time() - self.state_start_time

        if elapsed < 0.15 and self.lidar:
            self.get_logger().info(f"*** EXECUTING ARUCO DIFFERENTIAL {self.active_turn_cmd} TURN ***")
            self.get_logger().info(
                f'[TAG_ACTION] Clearance check -> fwd={self.lidar.forward_dist:.2f} '
                f'left={self.lidar.left_dist:.2f} right={self.lidar.right_dist:.2f}'
            )
            if self.active_turn_cmd == "RIGHT" and self.lidar.right_dist < TURN_AHEAD_SIDE_THRESH:
                self.get_logger().warn(
                    f'RIGHT turn blocked: right_dist={self.lidar.right_dist:.2f}m '
                    f'< {TURN_AHEAD_SIDE_THRESH}m'
                )
                self._transition(State.EXPLORE)
                return
            if self.active_turn_cmd == "LEFT" and self.lidar.left_dist < TURN_AHEAD_SIDE_THRESH:
                self.get_logger().warn(
                    f'LEFT turn blocked: left_dist={self.lidar.left_dist:.2f}m '
                    f'< {TURN_AHEAD_SIDE_THRESH}m'
                )
                self._transition(State.EXPLORE)
                return

        # Watchdog timeout: TAG_ACTION -> RECOVERY
        if elapsed > 10.0:
            self.get_logger().warn("TAG_ACTION Timeout! Recovering.")
            self._transition(State.RECOVERY)
            return

        # === PHASE 1: REPLACE TIME-BASED TURNS WITH YAW-TRACKED TURNS ===
        yaw_turned = abs(_angle_diff(self.pose_yaw, self.turn_start_yaw))
        
        # Explicit debugging to see that the ArUco turn is continuously commanding the differential drive
        if int(elapsed * 10) % 5 == 0:
            w_cmd = +W_SIGN_TURN if self.active_turn_cmd == "LEFT" else -W_SIGN_TURN
            self.get_logger().info(
                f"[ARUCO TURN ACTIVE] {self.active_turn_cmd} Arc | "
                f"Yaw progress: {math.degrees(yaw_turned):.1f}° / 87.0° | "
                f"Diff-Drive Cmds: linear.x={V_ARC}, angular.z={w_cmd}"
            )

        if yaw_turned < math.radians(87):
            if self.active_turn_cmd == "LEFT":
                self._move_sign(V_ARC, +W_SIGN_TURN)
            elif self.active_turn_cmd == "RIGHT":
                self._move_sign(V_ARC, -W_SIGN_TURN)
            else:
                self._transition(State.EXPLORE)
        else:
            self._publish_vel(0.0, 0.0)
            self._transition(State.EXPLORE)

    def _handle_follow_green(self):
        elapsed = time.time() - self.state_start_time

        if not self.sign_follow_armed:
            if time.time() - self.sign_arm_start_time >= SIGN_ARM_DELAY_SEC:
                self.sign_follow_armed = True
                self.latest_sign = None  # discard any sign detections accumulated during delay
            else:
                # Not yet armed — navigate using gap/lidar only, ignore sign data
                if not self.lidar:
                    return
                target_angle, _ = self._select_best_gap()
                if target_angle is not None:
                    self._move(V_MIN, target_angle)
                else:
                    self._publish_vel(V_MIN, W_SCAN)
                return

        # Check for next ArUco tag
        if (self.latest_aruco and
                float(self.latest_aruco.distance) <= ACTION_DISTANCE_THRESHOLD and
                self.latest_aruco.marker_id not in self.logged_tags):
            self._publish_vel(0.0, 0.0)
            self._transition(State.APPROACH_TAG)
            return

        # [FIX 1] FOLLOW states use gap navigation directly, NOT _handle_explore()
        # This avoids firing the EXPLORE stuck watchdog while in a FOLLOW state
        if elapsed > 120.0:
            self.get_logger().warn("FOLLOW_GREEN timeout! Falling back to EXPLORE.")
            self._transition(State.EXPLORE)
            return

        # === CV INTEGRATION: USE CV DIRECTION IN _handle_follow_green() ===
        # Try CV direction first
        if self.latest_floor_marker and self.latest_floor_marker.colour == "GREEN":
            direction = self.latest_floor_marker.direction
            self.latest_floor_marker = None   # consume it
            self.get_logger().info(f'[FOLLOW_GREEN] CV direction: {direction}')
            if direction == "FORWARD":
                self._publish_vel(V_MIN, 0.0)
            elif direction == "LEFT":
                self._publish_vel(0.0, +W_SIGN_TURN)
            elif direction == "RIGHT":
                self._publish_vel(0.0, -W_SIGN_TURN)
            elif direction == "BACKWARD":
                self.uturn_post_state = State.FOLLOW_GREEN
                self._transition(State.UTURN)
            return

        # Fallback: LiDAR gap nav if no CV signal yet
        if not self.lidar:
            return
        target_angle, _ = self._select_best_gap()
        if target_angle is not None:
            self._move(V_MIN, target_angle)
        else:
            self._publish_vel(V_MIN, W_SCAN)

    def _handle_follow_orange(self):
        elapsed = time.time() - self.state_start_time

        if not self.sign_follow_armed:
            if time.time() - self.sign_arm_start_time >= SIGN_ARM_DELAY_SEC:
                self.sign_follow_armed = True
                self.latest_sign = None  # discard any sign detections accumulated during delay
            else:
                # Not yet armed — navigate using gap/lidar only, ignore sign data
                if not self.lidar:
                    return
                target_angle, _ = self._select_best_gap()
                if target_angle is not None:
                    self._move(V_MIN, target_angle)
                else:
                    self._publish_vel(V_MIN, W_SCAN)
                return

        # Check for next ArUco tag
        if (self.latest_aruco and
                float(self.latest_aruco.distance) <= ACTION_DISTANCE_THRESHOLD and
                self.latest_aruco.marker_id not in self.logged_tags):
            self._publish_vel(0.0, 0.0)
            self._transition(State.APPROACH_TAG)
            return

        # [FIX 1] FOLLOW states use gap navigation directly, NOT _handle_explore()
        if elapsed > 120.0:
            self.get_logger().warn("FOLLOW_ORANGE timeout! Falling back to EXPLORE.")
            self._transition(State.EXPLORE)
            return

        # === CV INTEGRATION: USE CV DIRECTION IN _handle_follow_orange() ===
        # Try CV direction first
        if self.latest_floor_marker and self.latest_floor_marker.colour == "ORANGE":
            direction = self.latest_floor_marker.direction
            self.latest_floor_marker = None   # consume it
            self.get_logger().info(f'[FOLLOW_ORANGE] CV direction: {direction}')
            if direction == "FORWARD":
                self._publish_vel(V_MIN, 0.0)
            elif direction == "LEFT":
                self._publish_vel(0.0, +W_SIGN_TURN)
            elif direction == "RIGHT":
                self._publish_vel(0.0, -W_SIGN_TURN)
            elif direction == "BACKWARD":
                self.uturn_post_state = State.FOLLOW_ORANGE
                self._transition(State.UTURN)
            return

        # Fallback: LiDAR gap nav if no CV signal yet
        if not self.lidar:
            return
        target_angle, _ = self._select_best_gap()
        if target_angle is not None:
            self._move(V_MIN, target_angle)
        else:
            self._publish_vel(V_MIN, W_SCAN)

    def _handle_uturn(self):
        elapsed = time.time() - self.state_start_time
        self.indicator_blue_arrow = True

        # Watchdog timeout: UTURN -> RECOVERY
        if elapsed > 15.0:
            self.get_logger().warn("UTURN Timeout! Recovering.")
            self._transition(State.RECOVERY)
            return

        # === PHASE 1: REPLACE TIME-BASED TURNS WITH YAW-TRACKED TURNS ===
        yaw_turned = abs(_angle_diff(self.pose_yaw, self.turn_start_yaw))
        if yaw_turned < math.radians(177):
            self._move_sign(0.0, +W_SIGN_SPIN)
        else:
            self._publish_vel(0.0, 0.0)
            self.indicator_blue_arrow = False
            self._transition(self.uturn_post_state)

    def _handle_recovery(self):
        elapsed = time.time() - self.state_start_time

        if elapsed < 1.0:
            self._publish_vel(-0.20, 0.0)    # 1s reverse
        elif elapsed < 3.0:
            self._publish_vel(0.0, 0.4)      # 2s rotate / scan
        else:
            self._publish_vel(0.0, 0.0)
            self.get_logger().info("Recovery complete, returning to EXPLORE")
            self._transition(State.EXPLORE)

    def _handle_approach_halt(self):
        """Creep forward after RED tile detection until safely close to wall."""
        elapsed = time.time() - self.state_start_time

        # Safety timeout — force halt if we never reach the wall
        if elapsed > HALT_APPROACH_TIMEOUT:
            self.get_logger().info("APPROACH_HALT timeout — forcing HALT")
            self._transition(State.HALT)
            return

        # If LiDAR available, check forward distance
        if self.lidar:
            fwd = self.lidar.forward_dist
            if fwd <= HALT_WALL_DIST:
                self.get_logger().info(
                    f"━━━ HALT: Reached safe distance (fwd={fwd:.2f}m) ━━━"
                )
                self._publish_vel(0.0, 0.0)
                self._transition(State.HALT)
                return

            # Creep forward slowly, use PD centering to stay straight
            v = HALT_CREEP_SPEED
            w = 0.0
            # Basic wall centering if side data available
            if self.lidar.left_dist > 0 and self.lidar.right_dist > 0:
                wall_diff = self.lidar.left_dist - self.lidar.right_dist
                w = -0.8 * wall_diff  # gentle PD centering
                w = max(-0.3, min(0.3, w))
            self._publish_vel(v, w)
        else:
            # No LiDAR yet — creep forward blindly at minimum speed
            self._publish_vel(HALT_CREEP_SPEED, 0.0)

    def _handle_halt(self):
        # Stop the robot completely every tick
        self._publish_vel(0.0, 0.0)

        # [FIX 2] Use self.has_halted flag (set in __init__) instead of hasattr()
        # Prevents SystemExit from being raised repeatedly at 10Hz
        if not self.has_halted:
            self.has_halted = True

            with open('/tmp/mahe_nav_mission_log.txt', 'w') as f:
                f.write("MISSION COMPLETE\n")
                f.write(f"Logged Tags: {self.logged_tags}\n")
                f.write(f"Final Position: ({self.pose_x:.2f}, {self.pose_y:.2f})\n")

            self.get_logger().info("━━━ MISSION COMPLETE: Logs written. Node Halt. ━━━")
            self.destroy_node()
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NavControllerNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('SystemExit raised. Graceful Stop.')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()