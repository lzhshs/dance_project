import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from g1_dance.contact_balance_optimize import (
    compute_segment_median_support_centers,
    compute_support_centers,
    limit_root_motion,
    pd_correct_root_xy,
)


def test_support_centers_follow_contacting_feet_and_hold_previous_without_contact():
    left = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    right = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    left_contact = np.array([True, True, False, False])
    right_contact = np.array([True, False, True, False])

    centers = compute_support_centers(left, right, left_contact, right_contact)

    np.testing.assert_allclose(centers[0], [0.0, 1.0])
    np.testing.assert_allclose(centers[1], [1.0, 0.0])
    np.testing.assert_allclose(centers[2], [2.0, 2.0])
    np.testing.assert_allclose(centers[3], [2.0, 2.0])


def test_pd_correct_root_xy_moves_toward_support_with_step_limit():
    qpos = np.zeros((5, 8), dtype=np.float64)
    support = np.tile(np.array([1.0, 0.0]), (5, 1))

    corrected = pd_correct_root_xy(qpos, support, kp=0.5, kd=0.0, max_step=0.1)

    assert corrected[1, 0] > qpos[1, 0]
    assert corrected[-1, 0] < support[-1, 0]
    assert np.abs(np.diff(corrected[:, 0])).max() <= 0.1000001
    np.testing.assert_allclose(corrected[:, 1], 0.0)


def test_limit_root_motion_preserves_start_and_scales_displacement():
    qpos = np.zeros((3, 8), dtype=np.float64)
    qpos[:, 0] = [2.0, 4.0, 6.0]
    qpos[:, 1] = [1.0, 3.0, 5.0]

    limited = limit_root_motion(qpos, gain=0.25)

    np.testing.assert_allclose(limited[0, :2], qpos[0, :2])
    np.testing.assert_allclose(limited[-1, :2], [3.0, 2.0])


def test_segment_median_support_centers_ignore_sliding_within_contact_segment():
    left = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0]])
    right = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    left_contact = np.array([True, True, True, False])
    right_contact = np.array([False, False, False, False])

    centers = compute_segment_median_support_centers(left, right, left_contact, right_contact)

    np.testing.assert_allclose(centers[0], [0.4, 0.0])
    np.testing.assert_allclose(centers[1], [0.4, 0.0])
    np.testing.assert_allclose(centers[2], [0.4, 0.0])
    np.testing.assert_allclose(centers[3], [0.4, 0.0])
