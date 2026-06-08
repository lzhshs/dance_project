"""
Strategy B: SMPL → Unitree G1 重定向
将 EDGE 输出的 SMPL 24-joint 动作 映射到 Unitree G1 29-DOF 人形机器人

核心难点:
  - SMPL 用 ball joint (3D rotation per joint)
  - G1 用独立 hinge joints (pitch/roll/yaw 分开)
  - 需要把 SMPL 的 rotation matrix 分解为 Euler angles 映射到 G1 关节

Usage:
    conda activate sim
    python retarget_g1.py
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_retarget"))
from utils import parse_edge_motion, rot6d_to_rotmat

# G1 model path (ASCII, no Chinese characters)
G1_MODEL_DIR = "C:/temp/g1"
G1_SCENE_XML = os.path.join(G1_MODEL_DIR, "scene.xml")

# ============================================================
# SMPL → G1 关节映射
# ============================================================
# SMPL joints:  0=pelvis, 1=l_hip, 2=r_hip, 3=spine1, 4=l_knee, 5=r_knee,
#               6=spine2, 7=l_ankle, 8=r_ankle, 9=spine3, 10=l_foot, 11=r_foot,
#               12=neck, 13=l_collar, 14=r_collar, 15=head,
#               16=l_shoulder, 17=r_shoulder, 18=l_elbow, 19=r_elbow,
#               20=l_wrist, 21=r_wrist, 22=l_hand, 23=r_hand
#
# G1 joints:   hip (pitch/roll/yaw), knee, ankle (pitch/roll),
#              waist (yaw/roll/pitch),
#              shoulder (pitch/roll/yaw), elbow, wrist (roll/pitch/yaw)
#
# Mapping: SMPL ball joint rotation → decompose into G1 Euler angles

# SMPL joint index → (G1 joint names, euler_order, axis_signs)
# euler_order: 'YXZ' means first Y(pitch), then X(roll), then Z(yaw) in extrinsic
SMPL_TO_G1 = {
    # Left leg: SMPL l_hip (idx 1) → G1 left_hip_{pitch,roll,yaw}
    1: {
        "joints": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"],
        "euler": "YXZ",  # pitch=Y, roll=X, yaw=Z
        "scale": [1.0, 1.0, 1.0],
    },
    # Left knee: SMPL l_knee (idx 4) → G1 left_knee_joint
    4: {
        "joints": ["left_knee_joint"],
        "euler": "Y",  # single axis: pitch
        "scale": [1.0],
    },
    # Left ankle: SMPL l_ankle (idx 7) → G1 left_ankle_{pitch,roll}
    7: {
        "joints": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
        "euler": "YX",  # pitch=Y, roll=X
        "scale": [1.0, 1.0],
    },
    # Right leg
    2: {
        "joints": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"],
        "euler": "YXZ",
        "scale": [1.0, 1.0, 1.0],
    },
    5: {
        "joints": ["right_knee_joint"],
        "euler": "Y",
        "scale": [1.0],
    },
    8: {
        "joints": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
        "euler": "YX",
        "scale": [1.0, 1.0],
    },
    # Spine: SMPL spine1(3) + spine2(6) + spine3(9) → G1 waist_{yaw,roll,pitch}
    # We combine spine rotations into a single waist mapping
    # spine1 → waist (most important)
    3: {
        "joints": ["waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"],
        "euler": "YXZ",  # pitch=Y, roll=X, yaw=Z
        "scale": [1.0, 1.0, 1.0],
    },
    # Left arm
    16: {
        "joints": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint"],
        "euler": "YXZ",
        "scale": [1.0, 1.0, 1.0],
    },
    18: {
        "joints": ["left_elbow_joint"],
        "euler": "Y",
        "scale": [1.0],
    },
    20: {
        "joints": ["left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
        "euler": "XYZ",  # roll=X, pitch=Y, yaw=Z
        "scale": [1.0, 1.0, 1.0],
    },
    # Right arm
    17: {
        "joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint"],
        "euler": "YXZ",
        "scale": [1.0, 1.0, 1.0],
    },
    19: {
        "joints": ["right_elbow_joint"],
        "euler": "Y",
        "scale": [1.0],
    },
    21: {
        "joints": ["right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
        "euler": "XYZ",
        "scale": [1.0, 1.0, 1.0],
    },
}


def decompose_rotation(rotmat, euler_order):
    """
    Decompose 3x3 rotation matrix into Euler angles.

    Args:
        rotmat: (3, 3) rotation matrix
        euler_order: str like "YXZ" (3 axes) or "Y"/"YX" for partial extraction

    Returns:
        angles: list of floats (in radians), length = len(euler_order)
    """
    r = Rotation.from_matrix(rotmat)

    if len(euler_order) == 3:
        angles = r.as_euler(euler_order.upper(), degrees=False)
        return angles.tolist()
    elif len(euler_order) == 2:
        # For 2-axis: use a 3-axis decomposition and take first 2
        full_order = euler_order.upper() + "Z"  # append unused axis
        angles = r.as_euler(full_order, degrees=False)
        return angles[:2].tolist()
    elif len(euler_order) == 1:
        # For single axis: use axis-angle and project
        aa = r.as_rotvec()  # (3,)
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        idx = axis_map[euler_order.upper()]
        return [aa[idx]]
    else:
        return []


def retarget_to_g1(motion_path, output_path, motion_scale=0.6):
    """
    Convert EDGE SMPL motion → Unitree G1 qpos trajectory.

    Args:
        motion_path: EDGE output .npy (K, 151)
        output_path: output .npy (K, 36)  — G1 nq=36
        motion_scale: scale factor for motion amplitude (G1 has smaller range than SMPL)
    """
    import mujoco

    # 1. Load G1 model
    model = mujoco.MjModel.from_xml_path(G1_SCENE_XML)
    nq = model.nq  # 36
    print(f"  G1 model: nq={nq}, nv={model.nv}, nu={model.nu}")

    # Build joint name → qpos addr mapping
    joint_addr = {}
    for i in range(model.njnt):
        name = model.joint(i).name
        joint_addr[name] = model.jnt_qposadr[i]

    # 2. Parse EDGE motion
    motion = np.load(motion_path)
    joint_rot6d, root_trans, foot_contact = parse_edge_motion(motion)
    K = joint_rot6d.shape[0]  # number of frames

    # 3. Convert 6D → rotation matrices
    rotmats = rot6d_to_rotmat(joint_rot6d)  # (K, 24, 3, 3)

    # 4. Build G1 qpos trajectory
    qpos_traj = np.zeros((K, nq))

    # G1 root height (standing) = 0.793m, SMPL root ~ 0.91m
    height_ratio = 0.793 / 0.91

    for t in range(K):
        # Root: freejoint [pos(3) + quat(4)]
        # Scale root translation from SMPL to G1 proportions
        root_pos = root_trans[t].copy()
        root_pos[2] *= height_ratio  # scale z height
        root_pos[:2] *= height_ratio  # scale xy too
        qpos_traj[t, 0:3] = root_pos

        # Root orientation: from SMPL pelvis rotation
        root_rotmat = rotmats[t, 0]
        root_quat_xyzw = Rotation.from_matrix(root_rotmat).as_quat()
        root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
        qpos_traj[t, 3:7] = root_quat_wxyz

        # Body joints: decompose SMPL rotations into G1 Euler angles
        for smpl_idx, mapping in SMPL_TO_G1.items():
            rotmat = rotmats[t, smpl_idx]
            euler_order = mapping["euler"]
            scales = mapping["scale"]
            g1_joints = mapping["joints"]

            angles = decompose_rotation(rotmat, euler_order)

            # Map each Euler angle component to corresponding G1 joint
            for j, (joint_name, scale) in enumerate(zip(g1_joints, scales)):
                if j < len(angles):
                    addr = joint_addr[joint_name]
                    # Scale down to fit G1 range, and clip to joint limits
                    angle = angles[j] * scale * motion_scale
                    # Clip to joint range
                    jnt_idx = None
                    for ji in range(model.njnt):
                        if model.joint(ji).name == joint_name:
                            jnt_idx = ji
                            break
                    if jnt_idx is not None and model.jnt_limited[jnt_idx]:
                        lo, hi = model.jnt_range[jnt_idx]
                        angle = np.clip(angle, lo, hi)
                    qpos_traj[t, addr] = angle

    # 5. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, qpos_traj)
    print(f"  Retargeted: {motion_path} -> {output_path}")
    print(f"  Frames: {K}, G1 nq: {nq}, motion_scale: {motion_scale}")
    return qpos_traj


def simulate_g1(qpos_trajectory, output_dir, name):
    """
    Simulate G1 with kinematic root (same approach as SMPL humanoid).
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(G1_SCENE_XML)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    fps = 30
    steps_per_frame = max(1, int(1.0 / (fps * dt)))

    K = len(qpos_trajectory)
    results = {"qpos": [], "com_height": [], "fell": []}

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos_trajectory[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    for t in range(K):
        target = qpos_trajectory[t]

        # Kinematic root
        data.qpos[:3] = target[:3]
        data.qpos[3:7] = target[3:7]
        data.qvel[:6] = 0

        # Set ctrl = target joint angles (hinge joints start at qpos[7])
        for i in range(model.nu):
            jid = model.actuator_trnid[i, 0]
            qaddr = model.jnt_qposadr[jid]
            data.ctrl[i] = target[qaddr]

        for _ in range(steps_per_frame):
            data.qpos[:3] = target[:3]
            data.qpos[3:7] = target[3:7]
            data.qvel[:6] = 0
            mujoco.mj_step(model, data)

        com_h = data.subtree_com[0][2]
        results["qpos"].append(data.qpos.copy())
        results["com_height"].append(com_h)
        results["fell"].append(com_h < 0.3)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for key, val in results.items():
        np.save(os.path.join(output_dir, f"{name}_{key}.npy"), np.array(val))

    stable = sum(1 for f in results["fell"] if not f)
    print(f"  [{name}] PSR: {stable}/{K} = {100*stable/K:.1f}%")
    return results


def record_g1_video(qpos_trajectory, output_path, fps=30):
    """Record G1 video."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(G1_SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    frames = []
    for t in range(len(qpos_trajectory)):
        data.qpos[:] = qpos_trajectory[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    # Save video
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3_simulation"))
    from record_video import save_video
    save_video(frames, output_path, fps)


# ============================================================
# Main
# ============================================================
def main():
    PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MOTION_DIR = os.path.join(PROJECT, "outputs", "generated_motions")
    G1_RETARGET_DIR = os.path.join(PROJECT, "outputs", "retargeted_motions", "strategy_b_g1")
    G1_SIM_DIR = os.path.join(PROJECT, "outputs", "sim_results_g1")
    G1_VIDEO_DIR = os.path.join(PROJECT, "outputs", "videos")

    GENRES = ["ballet", "hiphop", "house"]

    print("=" * 60)
    print("  Strategy B: SMPL → Unitree G1 Retargeting + Simulation")
    print("=" * 60)

    # Ensure G1 model is available at ASCII path
    if not os.path.exists(G1_SCENE_XML):
        print(f"ERROR: G1 model not found at {G1_SCENE_XML}")
        print("Run: cp -r external/mujoco_menagerie/unitree_g1/* C:/temp/g1/")
        return

    os.makedirs(G1_RETARGET_DIR, exist_ok=True)
    os.makedirs(G1_SIM_DIR, exist_ok=True)
    os.makedirs(G1_VIDEO_DIR, exist_ok=True)

    for genre in GENRES:
        motion_path = os.path.join(MOTION_DIR, f"{genre}_motion.npy")
        if not os.path.exists(motion_path):
            print(f"\nSKIP: {motion_path} not found")
            continue

        print(f"\n{'='*40}")
        print(f"  Processing: {genre}")
        print(f"{'='*40}")

        # 1. Retarget
        print("\n  1. Retargeting SMPL → G1...")
        qpos_path = os.path.join(G1_RETARGET_DIR, f"{genre}_g1_qpos.npy")
        qpos_traj = retarget_to_g1(motion_path, qpos_path, motion_scale=0.6)

        # 2. Simulate
        print("\n  2. Physics simulation...")
        results = simulate_g1(qpos_traj, G1_SIM_DIR, genre)

        # 3. Record video (from sim qpos for physics accuracy)
        print("\n  3. Recording video...")
        sim_qpos = np.array(results["qpos"])
        record_g1_video(sim_qpos, os.path.join(G1_VIDEO_DIR, f"{genre}_g1_physics.mp4"))

        # Also record kinematic video (direct qpos)
        record_g1_video(qpos_traj, os.path.join(G1_VIDEO_DIR, f"{genre}_g1_kinematic.mp4"))

    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"{'='*60}")
    print(f"  G1 retargeted: {G1_RETARGET_DIR}")
    print(f"  G1 sim results: {G1_SIM_DIR}")
    print(f"  G1 videos: {G1_VIDEO_DIR}")


if __name__ == "__main__":
    main()
