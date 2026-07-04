import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from mahe_nav_interfaces.msg import ArucoDetection, SignDetection, LidarAnalysis

class State(Enum):
    INIT = auto()
    EXPLORE_FORWARD = auto()
    U_TURN_RECOVERY = auto()
    MAZE_NAVIGATE = auto()
    T_JUNCTION_SPIN = auto()
    BACKTRACK = auto()

# --- Configuration Constants ---
V_MAX = 0.40           
V_MIN = 0.12           
V_MAZE = 0.08          # Cautious speed for 580mm gap
SLOWDOWN_DIST = 1.2    
STOP_DIST = 0.28       # Standard stop
MAZE_STOP_DIST = 0.20  # Closer stop for narrow corridors
W_TURN = 0.55          

class NavControllerNode(Node):
    def __init__(self):
        super().__init__('nav_controller')
        
        self.state = State.INIT
        self.pose_x = self.pose_y = self.pose_yaw = 0.0
        self.spawn_yaw = None
        self.lidar = None
        self.sign = None
        self.aruco_seen = set()
        
        # Physical Thresholds
        self.PASSABLE_THR = 0.525
        
        # Stuck Prevention
        self.last_progress_time = time.time()
        self.last_pos = (0.0, 0.0)

        # ROS 2 Interfaces
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/odom_fused', self._odom_cb, 10)
        self.create_subscription(LidarAnalysis, '/lidar/analysis', self._lidar_cb, best_effort)
        self.create_subscription(SignDetection, '/sign_detection', self._sign_cb, best_effort)
        self.create_subscription(ArucoDetection, '/aruco/detections', self._aruco_cb, best_effort)

        self.create_timer(0.05, self._control_loop)
        self.get_logger().info('Reactive Nav Controller: Unified Version Active')

    def _odom_cb(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def _lidar_cb(self, msg): 
        self.lidar = msg

    def _sign_cb(self, msg): 
        self.sign = msg

    def _aruco_cb(self, msg):
        if msg.first_detection:
            self.aruco_seen.add(msg.marker_id)
            if msg.marker_id == 0: 
                self._transition(State.MAZE_NAVIGATE)

    def _control_loop(self):
        if not self.lidar: return
        
        if self.state == State.INIT:
            self.spawn_yaw = self.pose_yaw
            return self._transition(State.EXPLORE_FORWARD)
        
        self._check_stuck()

        if self.state == State.EXPLORE_FORWARD:
            self._handle_explore()
        elif self.state == State.U_TURN_RECOVERY:
            self._handle_u_turn()
        elif self.state == State.MAZE_NAVIGATE:
            self._handle_maze()

    def _handle_explore(self):
        # Dynamic Speed Scaling
        speed_factor = min(1.0, self.lidar.forward_dist / SLOWDOWN_DIST)
        v_cmd = max(V_MIN, V_MAX * speed_factor)
        
        # Dead-End Detection (Enter deep before turning)
        if self.lidar.is_u_shape and self.lidar.forward_dist < 0.60:
            return self._transition(State.U_TURN_RECOVERY)

        target_angle = 0.0
        if self.lidar.best_opening_idx >= 0 and self.lidar.opening_passable[self.lidar.best_opening_idx]:
            target_angle = self.lidar.opening_angles_rad[self.lidar.best_opening_idx]
        
        self._move(v_cmd, target_angle)

    def _handle_u_turn(self):
        self._move(0.0, W_TURN)
        
        # Oscillation Fix: Must turn at least 70 degrees from entry
        yaw_diff = abs(self.pose_yaw - self.spawn_yaw)
        if self.lidar.forward_dist > 2.0 and yaw_diff > 1.2: 
            self._transition(State.EXPLORE_FORWARD)

    def _handle_maze(self):
        # Logic for 580mm Narrow Gap
        valid_paths = []
        for i in range(len(self.lidar.opening_angles_rad)):
            if self.lidar.opening_widths_m[i] > self.PASSABLE_THR:
                valid_paths.append({
                    'angle': self.lidar.opening_angles_rad[i],
                    'width': self.lidar.opening_widths_m[i]
                })

        if not valid_paths:
            return self._move(0.0, W_TURN)

        # Centering Strategy: Pick the path closest to current heading
        best_path = min(valid_paths, key=lambda x: abs(x['angle']))
        
        # Steering gain for precision in tight 580mm path
        steering_gain = 1.3 if best_path['width'] < 0.65 else 1.0
        self._move(V_MAZE, best_path['angle'] * steering_gain)

    def _move(self, v, w):
        # Emergency Brake Scaling
        current_limit = MAZE_STOP_DIST if self.state == State.MAZE_NAVIGATE else STOP_DIST
        if self.lidar and self.lidar.forward_dist < current_limit:
            v = 0.0
        
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _transition(self, new_state):
        self.get_logger().info(f"Transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.last_progress_time = time.time()

    def _check_stuck(self):
        now = time.time()
        dist = math.hypot(self.pose_x - self.last_pos[0], self.pose_y - self.last_pos[1])
        
        if dist > 0.05:
            self.last_pos = (self.pose_x, self.pose_y)
            self.last_progress_time = now
        elif (now - self.last_progress_time) > 5.0:
            self.get_logger().warn("STUCK: Performing Strong Backtrack")
            self._move(-0.20, 0.0) # High power reverse to clear corners
            self.last_progress_time = now

def main(args=None):
    rclpy.init(args=args)
    node = NavControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
