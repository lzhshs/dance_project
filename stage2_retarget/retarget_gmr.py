"""
Strategy B: SMPL-to-Robot 重定向 (GMR -> Unitree G1)
使用 GMR 将 SMPL 动作重定向到 Unitree G1 机器人 (37 actuated DoFs)

使用方法:
    conda activate sim
    python retarget_gmr.py --motion ../outputs/generated_motions/ballet_motion.npy \
                           --output ../outputs/retargeted_motions/strategy_b/ballet_g1_qpos.npy
"""
import argparse
import os
import sys
import numpy as np
from utils import (
    parse_edge_motion,
    rot6d_to_axis_angle,
    rot6d_to_quat_wxyz,
    scale_motion,
    SMPL_JOINT_NAMES,
)

GMR_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "GMR")

# ============================================================
# SMPL -> G1 关节映射
# G1 有 37 个 actuated DoFs
# ============================================================
SMPL_TO_G1_JOINT_MAP = {
    # SMPL idx -> G1 joint name(s)
    # 下肢
    1:  ["left_hip_pitch", "left_hip_roll", "left_hip_yaw"],     # l_hip (ball -> 3 hinge)
    2:  ["right_hip_pitch", "right_hip_roll", "right_hip_yaw"],  # r_hip
    4:  ["left_knee"],                                            # l_knee (hinge)
    5:  ["right_knee"],                                           # r_knee
    7:  ["left_ankle_pitch", "left_ankle_roll"],                  # l_ankle (2 hinge)
    8:  ["right_ankle_pitch", "right_ankle_roll"],                # r_ankle
    # 上肢
    16: ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw"],   # l_shoulder
    17: ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw"],# r_shoulder
    18: ["left_elbow"],                                           # l_elbow
    19: ["right_elbow"],                                          # r_elbow
    20: ["left_wrist_roll"],                                      # l_wrist
    21: ["right_wrist_roll"],                                     # r_wrist
    # 躯干
    3:  ["waist_yaw"],                                            # spine1
    6:  ["waist_pitch", "waist_roll"],                            # spine2
}


def decompose_ball_to_euler(axis_angle, order="xyz"):
    """
    将 ball joint 的 axis-angle 分解为多个 hinge joint 的角度
    Input: (3,) axis-angle
    Output: (3,) euler angles in specified order
    """
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(axis_angle)
    return R.as_euler(order)


def retarget_frame_manual(smpl_rot6d_frame, smpl_root_trans, g1_joint_names, g1_joint_limits):
    """
    手动映射单帧 SMPL -> G1 关节角度
    当 GMR 不可用时的 fallback

    Args:
        smpl_rot6d_frame: (24, 6) 单帧 SMPL 关节旋转
        smpl_root_trans: (3,) root translation
        g1_joint_names: list of G1 joint names (ordered)
        g1_joint_limits: dict {joint_name: (lower, upper)}

    Returns:
        g1_joints: dict {joint_name: angle}
    """
    from scipy.spatial.transform import Rotation

    g1_joints = {name: 0.0 for name in g1_joint_names}

    for smpl_idx, g1_names in SMPL_TO_G1_JOINT_MAP.items():
        # SMPL 6D -> axis-angle
        aa = rot6d_to_axis_angle(smpl_rot6d_frame[smpl_idx])  # (3,)

        if len(g1_names) == 1:
            # Hinge joint: 取 axis-angle 的角度大小
            angle = np.linalg.norm(aa)
            if aa[1] < 0:  # 简单的符号判断
                angle = -angle
            g1_joints[g1_names[0]] = angle

        elif len(g1_names) == 3:
            # Ball joint -> 3 个 hinge: 分解为 euler 角
            euler = decompose_ball_to_euler(aa, order="xyz")
            for i, name in enumerate(g1_names):
                g1_joints[name] = euler[i]

        elif len(g1_names) == 2:
            # 2-DoF joint
            euler = decompose_ball_to_euler(aa, order="xyz")
            g1_joints[g1_names[0]] = euler[0]
            g1_joints[g1_names[1]] = euler[1]

    # 应用关节限位
    for name in g1_joint_names:
        if name in g1_joint_limits:
            lo, hi = g1_joint_limits[name]
            g1_joints[name] = np.clip(g1_joints[name], lo, hi)

    return g1_joints


def retarget_with_gmr(motion_path, output_path):
    """
    使用 GMR 进行重定向
    GMR 提供了学习到的 correspondence，效果更好
    """
    sys.path.insert(0, GMR_DIR)

    try:
        from gmr.retarget import MotionRetargeter

        motion = np.load(motion_path)
        joint_rot6d, root_trans, _ = parse_edge_motion(motion)

        retargeter = MotionRetargeter(
            source="smpl",
            target="unitree_g1",
            urdf_path=os.path.join(GMR_DIR, "assets", "unitree_g1", "g1.urdf"),
        )

        g1_trajectory = []
        for t in range(len(joint_rot6d)):
            g1_qpos = retargeter.retarget(
                source_rotations=joint_rot6d[t],
                source_root_translation=root_trans[t],
            )
            g1_trajectory.append(g1_qpos)

        g1_trajectory = np.array(g1_trajectory)
        np.save(output_path, g1_trajectory)
        print(f"GMR retargeted: {g1_trajectory.shape} -> {output_path}")
        return g1_trajectory

    except ImportError:
        print("GMR not available. Using manual retargeting fallback...")
        return retarget_manual_fallback(motion_path, output_path)


def retarget_manual_fallback(motion_path, output_path, motion_scale=0.8):
    """
    手动 retargeting fallback（当 GMR 不可用时）
    """
    motion = np.load(motion_path)
    joint_rot6d, root_trans, _ = parse_edge_motion(motion)
    K = joint_rot6d.shape[0]

    # 缩放动作（G1 关节范围比人体小）
    if motion_scale != 1.0:
        joint_rot6d, root_trans = scale_motion(joint_rot6d, root_trans, motion_scale)

    # G1 关节列表和限位（简化版）
    g1_joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_pitch", "waist_roll",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll",
    ]

    g1_joint_limits = {
        "left_hip_pitch": (-1.57, 1.57),
        "left_hip_roll": (-0.5, 0.5),
        "left_hip_yaw": (-0.5, 0.5),
        "left_knee": (0.0, 2.5),
        "left_ankle_pitch": (-0.8, 0.8),
        "left_ankle_roll": (-0.3, 0.3),
        "right_hip_pitch": (-1.57, 1.57),
        "right_hip_roll": (-0.5, 0.5),
        "right_hip_yaw": (-0.5, 0.5),
        "right_knee": (0.0, 2.5),
        "right_ankle_pitch": (-0.8, 0.8),
        "right_ankle_roll": (-0.3, 0.3),
        "waist_yaw": (-1.0, 1.0),
        "waist_pitch": (-0.5, 0.5),
        "waist_roll": (-0.3, 0.3),
        "left_shoulder_pitch": (-3.14, 3.14),
        "left_shoulder_roll": (-1.5, 1.5),
        "left_shoulder_yaw": (-1.5, 1.5),
        "left_elbow": (-2.5, 0.0),
        "left_wrist_roll": (-1.0, 1.0),
        "right_shoulder_pitch": (-3.14, 3.14),
        "right_shoulder_roll": (-1.5, 1.5),
        "right_shoulder_yaw": (-1.5, 1.5),
        "right_elbow": (-2.5, 0.0),
        "right_wrist_roll": (-1.0, 1.0),
    }

    n_joints = len(g1_joint_names)
    trajectory = np.zeros((K, 7 + n_joints))  # root(7) + joints

    for t in range(K):
        # Root
        trajectory[t, 0:3] = root_trans[t]
        trajectory[t, 3:7] = rot6d_to_quat_wxyz(joint_rot6d[t, 0])  # pelvis orientation

        # Joints
        g1_joints = retarget_frame_manual(
            joint_rot6d[t], root_trans[t], g1_joint_names, g1_joint_limits
        )
        for j, name in enumerate(g1_joint_names):
            trajectory[t, 7 + j] = g1_joints[name]

    np.save(output_path, trajectory)
    print(f"Manual retargeted: {trajectory.shape} -> {output_path}")
    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Strategy B: SMPL to Unitree G1 retargeting")
    parser.add_argument("--motion", type=str, help="Single motion .npy file")
    parser.add_argument("--motion_dir", type=str, default="../outputs/generated_motions")
    parser.add_argument("--output_dir", type=str, default="../outputs/retargeted_motions/strategy_b")
    parser.add_argument("--scale", type=float, default=0.8, help="Motion amplitude scale for G1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.motion:
        name = os.path.basename(args.motion).replace("_motion.npy", "")
        retarget_with_gmr(
            args.motion,
            os.path.join(args.output_dir, f"{name}_g1_qpos.npy"),
        )
    else:
        motion_files = [f for f in os.listdir(args.motion_dir) if f.endswith("_motion.npy")]
        for f in motion_files:
            name = f.replace("_motion.npy", "")
            print(f"\nRetargeting: {name}")
            retarget_with_gmr(
                os.path.join(args.motion_dir, f),
                os.path.join(args.output_dir, f"{name}_g1_qpos.npy"),
            )
        print(f"\nDone! Retargeted {len(motion_files)} motions to G1.")


if __name__ == "__main__":
    main()
