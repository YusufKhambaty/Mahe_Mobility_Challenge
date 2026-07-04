# Processed LiDAR scan analysis — produced by lidar_analyzer_node
std_msgs/Header header

# Cardinal distances (metres) — corrected for min-range saturation
float32 forward_dist
float32 left_dist
float32 right_dist
float32 back_dist

# True when forward+left+right are all below CLOSED_ROW_THRESHOLD
# Indicates a dead-end / closed row (U-shape)
bool is_u_shape

# Estimated corridor width using perpendicular rays (metres)
float32 corridor_width_lateral       # left_dist + right_dist + robot_width

# List of angular openings detected in 360° scan.
# Each opening is described by its centre angle (rad, 0=fwd, +ve=left)
# and its estimated physical width (metres).
float32[] opening_angles_rad         # centre bearing of each gap
float32[] opening_widths_m           # estimated physical width
bool[]    opening_passable           # true if width > P * robot_width threshold

# Index of the widest passable opening (-1 if none)
int32 best_opening_idx

# True if the robot appears to be stuck (no passable forward motion)
bool no_forward_passage
