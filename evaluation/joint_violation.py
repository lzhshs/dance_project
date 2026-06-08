"""
Joint Limit Violation Rate (JLVR)
衡量关节角度超出物理限位的帧比例
"""
import os
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import load_model_safe


def joint_limit_violation_rate(qpos_trajectory, model_path):
    """
    Joint Limit Violation Rate = 有任意关节超限的帧数 / 总帧数

    Args:
        qpos_trajectory: (T, nq) 关节角度序列
        model_path: MuJoCo XML 模型路径

    Returns:
        rate: float in [0, 1]
        details: dict with per-joint violation info
    """
    assert mujoco is not None, "mujoco required"

    model = load_model_safe(model_path)

    total_frames = len(qpos_trajectory)
    violation_count = 0
    joint_violations = {}  # joint_name -> count

    for t in range(total_frames):
        qpos = qpos_trajectory[t]
        frame_violated = False

        for j in range(model.njnt):
            if not model.jnt_limited[j]:
                continue

            jnt_name = model.joint(j).name
            q_addr = model.jnt_qposadr[j]
            jnt_type = model.jnt_type[j]

            if jnt_type == 0:  # free joint, skip
                continue
            elif jnt_type == 3:  # hinge: 1 DoF
                if q_addr < len(qpos):
                    q_val = qpos[q_addr]
                    lo, hi = model.jnt_range[j]
                    if q_val < lo or q_val > hi:
                        frame_violated = True
                        joint_violations[jnt_name] = joint_violations.get(jnt_name, 0) + 1

        if frame_violated:
            violation_count += 1

    rate = violation_count / total_frames if total_frames > 0 else 0.0

    # Per-joint violation rate
    per_joint_rate = {
        name: count / total_frames for name, count in joint_violations.items()
    }

    return rate, {
        "total_violation_frames": violation_count,
        "total_frames": total_frames,
        "per_joint_violations": joint_violations,
        "per_joint_violation_rate": per_joint_rate,
    }


def check_joint_limits_kinematic(motion_data, joint_ranges):
    """
    检查运动学数据的关节限位违反（不需要 MuJoCo 模型）

    Args:
        motion_data: (K, 151) EDGE 格式
        joint_ranges: dict {joint_idx: (lower, upper)} in radians

    Returns:
        rate: float
    """
    from stage2_retarget.utils import rot6d_to_axis_angle

    K = motion_data.shape[0]
    joint_rot6d = motion_data[:, :144].reshape(K, 24, 6)
    joint_aa = rot6d_to_axis_angle(joint_rot6d)  # (K, 24, 3)
    joint_angles = np.linalg.norm(joint_aa, axis=-1)  # (K, 24)

    violation_count = 0
    for t in range(K):
        for j_idx, (lo, hi) in joint_ranges.items():
            if joint_angles[t, j_idx] > hi:
                violation_count += 1
                break

    return violation_count / K if K > 0 else 0.0


def compute_jlvr(sim_result_dir, name, model_path):
    """
    从仿真结果计算 JLVR

    Returns:
        result: dict
    """
    import os

    qpos_path = os.path.join(sim_result_dir, f"{name}_qpos.npy")
    if not os.path.exists(qpos_path):
        qpos_path = os.path.join(sim_result_dir, f"{name}_physics_qpos.npy")

    qpos = np.load(qpos_path)
    rate, details = joint_limit_violation_rate(qpos, model_path)

    return {
        "joint_limit_violation_rate": rate,
        **details,
    }
