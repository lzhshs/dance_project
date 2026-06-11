import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from motion_constraints import _joint_profile


def test_expressive_profile_preserves_more_upper_body_motion_than_safe_profile():
    safe = _joint_profile("left_shoulder_pitch_joint", profile="safe")
    expressive = _joint_profile("left_shoulder_pitch_joint", profile="expressive")

    assert expressive.gain > safe.gain
    assert expressive.window <= safe.window
    assert expressive.max_step <= 0.14


def test_balance_profile_is_more_conservative_for_legs_but_keeps_some_arms():
    safe_leg = _joint_profile("left_hip_pitch_joint", profile="safe")
    balance_leg = _joint_profile("left_hip_pitch_joint", profile="balance")
    balance_arm = _joint_profile("left_shoulder_pitch_joint", profile="balance")

    assert balance_leg.gain < safe_leg.gain
    assert balance_leg.max_step < safe_leg.max_step
    assert balance_arm.gain > balance_leg.gain
    assert balance_arm.gain >= 0.30


def test_showcase_profile_emphasizes_arms_more_than_legs():
    expressive_arm = _joint_profile("left_shoulder_pitch_joint", profile="expressive")
    showcase_arm = _joint_profile("left_shoulder_pitch_joint", profile="showcase")
    showcase_leg = _joint_profile("left_hip_pitch_joint", profile="showcase")

    assert showcase_arm.gain > showcase_leg.gain
    assert showcase_arm.gain >= expressive_arm.gain
    assert showcase_leg.gain < expressive_arm.gain


def test_unknown_constraint_profile_is_rejected():
    try:
        _joint_profile("left_shoulder_pitch_joint", profile="unknown")
    except ValueError as exc:
        assert "Unknown constraint profile" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown profile")
