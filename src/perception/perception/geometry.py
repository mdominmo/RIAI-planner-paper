"""Geometry used by the cooperative perception (no ROS dependencies).

Frames
------
* World ``W``: ENU, the Gazebo world frame used by the rest of the stack.
* Robot ``R``: FLU body frame of the UAV (Gazebo ``base_link``).
* Camera link ``L``: Gazebo camera link (optical axis +X, image right -Y,
  image down -Z).
* Camera ``C``: OpenCV optical frame (Z forward, X right, Y down).
* Marker ``M``: centre of the ArUco square.

PX4 reports the UAV pose as position in local NED and the attitude as the
rotation FRD body -> NED. In this system the PX4 local frame is the Gazebo
world frame expressed in NED (px4_gz_positioning feeds the Gazebo pose as
visual odometry), the same assumption used by mission_node.

Georeferencing (paper, Sec. Percepción):
    ^W T_M = ^W T_R · ^R T_C · ^C T_M
"""
import bisect
import math

import cv2
import numpy as np

# NED <-> ENU and FRD <-> FLU (both are involutions).
NED_ENU = np.array([[0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0]])
FRD_FLU = np.diag([1.0, -1.0, -1.0])

# Columns: OpenCV camera axes expressed in the Gazebo camera link frame.
R_LINK_CV = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])


def rpy_to_matrix(roll, pitch, yaw):
    """Rotation matrix for fixed-axis roll/pitch/yaw (SDF/URDF convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_to_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def px4_to_world(position_ned, q_ned_frd):
    """PX4 odometry (NED position, FRD->NED quaternion w,x,y,z) -> ^W T_R."""
    p = NED_ENU @ np.asarray(position_ned, dtype=float)
    r = NED_ENU @ quat_to_matrix(*q_ned_frd) @ FRD_FLU
    return p, r


def camera_matrix(width, height, hfov):
    """Pinhole intrinsics of an ideal (Gazebo) camera with square pixels."""
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]])


def camera_extrinsics(xyz, rpy):
    """^R T_C from the camera link pose in the UAV body frame (SDF values)."""
    r = rpy_to_matrix(*rpy) @ R_LINK_CV
    return np.asarray(xyz, dtype=float), r


def marker_object_points(length):
    """Marker corners in ``M`` in ArUco order (TL, TR, BR, BL), for IPPE_SQUARE."""
    h = length / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
                    dtype=np.float32)


def solve_marker_pnp(corners, object_points, k, dist=None):
    """PnP for one marker. Returns (tvec ^C p_M, rvec, mean reprojection error px)."""
    img = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dist = np.zeros(5) if dist is None else dist
    ok, rvec, tvec = cv2.solvePnP(object_points, img, k, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, k, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img, axis=1)))
    return tvec.reshape(3), rvec.reshape(3), err


def marker_in_world(t_cm, p_wr, r_wr, t_rc, r_rc):
    """^W p_M = ^W T_R · ^R T_C · ^C p_M (only the marker position is kept)."""
    p_r = r_rc @ np.asarray(t_cm, dtype=float) + t_rc
    return r_wr @ p_r + p_wr


class PoseBuffer:
    """Short history of UAV poses to match each image with the pose at its stamp."""

    def __init__(self, horizon=2.0):
        self.horizon = horizon
        self.t = []
        self.data = []

    def add(self, t, p, r):
        if self.t and t <= self.t[-1]:
            if t < self.t[-1] - self.horizon:   # clock jumped back: restart
                self.t, self.data = [], []
            else:
                return
        self.t.append(t)
        self.data.append((p, r))
        cut = bisect.bisect_left(self.t, t - self.horizon)
        if cut:
            del self.t[:cut]
            del self.data[:cut]

    def latest_time(self):
        return self.t[-1] if self.t else None

    def lookup(self, t, max_dt):
        """Pose at ``t``: position interpolated, attitude of the nearest sample."""
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        if 0 < i < len(self.t):
            t0, t1 = self.t[i - 1], self.t[i]
            if t1 - t0 > max_dt:
                return None
            a = (t - t0) / (t1 - t0)
            p = (1 - a) * self.data[i - 1][0] + a * self.data[i][0]
            r = self.data[i - 1][1] if a < 0.5 else self.data[i][1]
            return p, r
        j = 0 if i == 0 else len(self.t) - 1
        if abs(self.t[j] - t) > max_dt:
            return None
        return self.data[j]


class MarkerFusion:
    """Global fusion: p_n = mean of every measurement p_n^(i) of marker n by any UAV.

    Keeps running sums, so the estimate is exactly the paper's average over
    sum_i |M_n^(i)| measurements without storing them.
    """

    def __init__(self):
        self.sum = {}
        self.count = {}
        self.uavs = {}

    def add(self, uav_id, marker_id, p):
        self.sum[marker_id] = self.sum.get(marker_id, np.zeros(3)) + np.asarray(p, dtype=float)
        self.count[marker_id] = self.count.get(marker_id, 0) + 1
        self.uavs.setdefault(marker_id, {})
        self.uavs[marker_id][uav_id] = self.uavs[marker_id].get(uav_id, 0) + 1

    def estimate(self, marker_id):
        return self.sum[marker_id] / self.count[marker_id]

    def targets(self, min_measurements=1):
        """[(ID_n, p_n, total measurements, |R_n|)] sorted by ID."""
        return [(n, self.estimate(n), self.count[n], len(self.uavs[n]))
                for n in sorted(self.count) if self.count[n] >= min_measurements]
