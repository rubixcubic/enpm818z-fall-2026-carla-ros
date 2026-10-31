# One control command (GP4 Task 1 applies it at the next tick).
std_msgs/Header header   # stamp: the frame the command answers (lock step)
float32 throttle         # 0 to max_throttle
float32 steer            # -1 to 1: steering angle / max_steer_angle, + turns LEFT
float32 brake            # 0 to 1
