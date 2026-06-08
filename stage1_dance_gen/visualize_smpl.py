"""
SMPL 动作可视化
将生成的 SMPL 动作渲染为视频，验证动作质量

使用方法:
    conda activate sim
    python visualize_smpl.py --motion ../outputs/generated_motions/ballet_motion.npy --output ../outputs/videos/ballet_smpl.mp4
"""
import argparse
import os
import numpy as np

try:
    import smplx
    import torch
    import trimesh
except ImportError:
    print("Need: pip install smplx torch trimesh")
    exit(1)


def rot6d_to_rotmat(rot6d):
    """6D rotation -> 3x3 rotation matrix"""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rot6d_to_axis_angle(rot6d):
    """6D rotation -> axis-angle (3,) for SMPL"""
    from scipy.spatial.transform import Rotation
    rotmat = rot6d_to_rotmat(rot6d)
    shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(rotmat_flat).as_rotvec()
    return aa.reshape(*shape, 3)


def parse_motion(motion_data):
    """
    解析 EDGE 输出
    motion: (K, 151)
    """
    K = motion_data.shape[0]
    joint_rot6d = motion_data[:, :144].reshape(K, 24, 6)
    root_trans = motion_data[:, 144:147]
    foot_contact = motion_data[:, 147:151]
    return joint_rot6d, root_trans, foot_contact


def visualize_frame(smpl_model, joint_rot6d_frame, root_trans_frame):
    """渲染单帧 SMPL mesh"""
    # 转换为 axis-angle
    body_pose_aa = rot6d_to_axis_angle(joint_rot6d_frame[1:])  # (23, 3) 去掉 root
    global_orient_aa = rot6d_to_axis_angle(joint_rot6d_frame[0:1])  # (1, 3)

    body_pose = torch.from_numpy(body_pose_aa.reshape(1, -1)).float()
    global_orient = torch.from_numpy(global_orient_aa.reshape(1, -1)).float()
    transl = torch.from_numpy(root_trans_frame.reshape(1, -1)).float()

    with torch.no_grad():
        output = smpl_model(
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )

    vertices = output.vertices[0].numpy()
    faces = smpl_model.faces
    return vertices, faces


def visualize_sequence(motion_path, output_path, smpl_model_path, sample_every=5):
    """
    可视化整个动作序列
    sample_every: 每隔几帧采样一次（减少计算量）
    """
    motion = np.load(motion_path)
    joint_rot6d, root_trans, _ = parse_motion(motion)

    print(f"Motion: {motion.shape[0]} frames")
    print(f"Sampling every {sample_every} frames -> {motion.shape[0] // sample_every} frames")

    smpl_model = smplx.create(
        model_path=os.path.dirname(smpl_model_path),
        model_type="smpl",
        gender="neutral",
    )

    # 导出为一系列 OBJ 文件（或用 pyrender 渲染视频）
    mesh_dir = output_path.replace(".mp4", "_meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    for i in range(0, len(joint_rot6d), sample_every):
        vertices, faces = visualize_frame(smpl_model, joint_rot6d[i], root_trans[i])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(os.path.join(mesh_dir, f"frame_{i:05d}.obj"))

        if (i // sample_every) % 20 == 0:
            print(f"  Frame {i}/{len(joint_rot6d)}")

    print(f"Meshes saved to: {mesh_dir}")
    print("Use Blender or MeshLab to view the meshes.")


def quick_check(motion_path):
    """快速检查动作数据是否合理"""
    motion = np.load(motion_path)
    print(f"Shape: {motion.shape}")
    print(f"Root translation range:")
    print(f"  X: [{motion[:, 144].min():.3f}, {motion[:, 144].max():.3f}]")
    print(f"  Y: [{motion[:, 145].min():.3f}, {motion[:, 145].max():.3f}]")
    print(f"  Z: [{motion[:, 146].min():.3f}, {motion[:, 146].max():.3f}]")
    print(f"Foot contact mean: {motion[:, 147:151].mean(axis=0)}")

    # 检查旋转数据是否在合理范围
    rot_data = motion[:, :144]
    print(f"Rotation data range: [{rot_data.min():.3f}, {rot_data.max():.3f}]")
    print(f"Rotation data std: {rot_data.std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SMPL dance motion")
    parser.add_argument("--motion", type=str, required=True, help="Path to motion .npy")
    parser.add_argument("--output", type=str, default="../outputs/videos/smpl_viz.mp4")
    parser.add_argument("--smpl_model", type=str, default="../body_models/smpl")
    parser.add_argument("--check", action="store_true", help="Quick data check only")
    parser.add_argument("--sample_every", type=int, default=5)
    args = parser.parse_args()

    if args.check:
        quick_check(args.motion)
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        visualize_sequence(args.motion, args.output, args.smpl_model, args.sample_every)


if __name__ == "__main__":
    main()
