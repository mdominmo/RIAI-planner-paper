# Georeferenced estimate p_n^(i) of one ArUco marker made on board one UAV
# (Detección Distribuida). Sent to the global fusion module.
std_msgs/Header header          # stamp: image stamp; frame_id: world frame
uint32 uav_id
uint32 marker_id
geometry_msgs/Point position    # marker centre p_n^(i) in the world frame (ENU) [m]
float32 distance                # camera-marker distance [m]
float32 reprojection_error      # PnP mean reprojection error [px]
