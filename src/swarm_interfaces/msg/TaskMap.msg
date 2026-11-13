# Global map of fused markers (Módulo de Fusión Global): one entry per ID_n.
std_msgs/Header header
uint32[] ids
geometry_msgs/Point[] positions     # fused p_n, world frame (ENU) [m]
uint32[] num_measurements           # sum_i |M_n^(i)|
uint32[] num_uavs                   # |R_n|
