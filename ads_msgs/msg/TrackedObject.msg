# One object tracked on the map (GP3 Task 4). Frame: the array's header.
string id                      # track ID, kept while the track lives
string class_id                # 'vehicle' or 'pedestrian', from the detector
geometry_msgs/Point position   # meters, map frame
geometry_msgs/Vector3 velocity # m/s, map frame
float64[16] covariance         # 4 x 4, row major, state [x, y, vx, vy]
