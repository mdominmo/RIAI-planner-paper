import math

import cv2
import numpy as np
import pytest

from perception import geometry as g


def _quat_from_matrix(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return (w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w))


def test_quat_roundtrip():
    r = g.rpy_to_matrix(0.1, -0.2, 2.0)
    assert np.allclose(g.quat_to_matrix(*_quat_from_matrix(r)), r)


def test_camera_looks_forward_and_down():
    # Camera pitched 0.8 rad down on a level UAV heading +X (ENU east).
    t_rc, r_rc = g.camera_extrinsics([0.12, 0.03, 0.242], [0.0, 0.8, 0.0])
    z_axis = r_rc[:, 2]
    assert z_axis[0] > 0 and z_axis[2] < 0           # forward and down
    assert np.isclose(math.atan2(-z_axis[2], z_axis[0]), 0.8)
    assert np.allclose(r_rc[:, 0], [0, -1, 0])       # image right = body right


def test_px4_level_heading_north():
    # PX4: level, yaw 0 (nose north). In ENU the body X axis must point +Y.
    p, r = g.px4_to_world([1.0, 2.0, -3.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(p, [2.0, 1.0, 3.0])
    assert np.allclose(r[:, 0], [0, 1, 0])
    assert np.allclose(r[:, 2], [0, 0, 1])


@pytest.mark.parametrize('yaw_ned', [0.0, 0.7, -2.5])
def test_end_to_end_georeferencing(yaw_ned):
    k = g.camera_matrix(1240, 720, 1.047)
    length = 2.0
    obj = g.marker_object_points(length)

    # UAV at 10 m height, yawed and slightly tilted, given as PX4 odometry.
    r_ned_frd = (np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @
                 g.rpy_to_matrix(0.03, -0.05, math.pi / 2 - yaw_ned) @ g.FRD_FLU)
    p_wr, r_wr = g.px4_to_world([5.0, -3.0, -10.0], _quat_from_matrix(r_ned_frd))
    t_rc, r_rc = g.camera_extrinsics([0.12, 0.03, 0.242], [0.0, 0.8, 0.0])

    # Marker lying flat 12 m ahead of the camera along its optical axis.
    cam_w = r_wr @ t_rc + p_wr
    r_wc = r_wr @ r_rc
    ahead = r_wc[:, 2] * 12.0
    marker_w = cam_w + ahead
    r_wm = g.rpy_to_matrix(0.0, 0.0, 0.4)
    corners_w = (r_wm @ obj.T).T + marker_w
    corners_c = (r_wc.T @ (corners_w - cam_w).T).T
    img, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), k, np.zeros(5))

    t_cm, _, err = g.solve_marker_pnp(img, obj, k)
    assert err < 1e-3
    est = g.marker_in_world(t_cm, p_wr, r_wr, t_rc, r_rc)
    assert np.allclose(est, marker_w, atol=1e-3)


def test_pose_buffer_interpolates_and_rejects_gaps():
    b = g.PoseBuffer(horizon=1.0)
    eye = np.eye(3)
    b.add(0.0, np.array([0.0, 0, 0]), eye)
    b.add(0.1, np.array([1.0, 0, 0]), eye)
    p, _ = b.lookup(0.05, max_dt=0.2)
    assert np.allclose(p, [0.5, 0, 0])
    assert b.lookup(0.5, max_dt=0.2) is None
    b.add(2.0, np.array([2.0, 0, 0]), eye)       # old samples dropped
    assert b.lookup(0.05, max_dt=0.2) is None


def test_fusion_is_the_mean_of_all_measurements():
    f = g.MarkerFusion()
    f.add(1, 3, [0, 0, 0])
    f.add(1, 3, [2, 0, 0])
    f.add(2, 3, [4, 3, 0])
    f.add(2, 5, [1, 1, 1])
    (i3, p3, n3, u3), (i5, p5, n5, u5) = f.targets()
    assert (i3, n3, u3) == (3, 3, 2) and np.allclose(p3, [2, 1, 0])
    assert (i5, n5, u5) == (5, 1, 1)
    assert [t[0] for t in f.targets(min_measurements=2)] == [3]
