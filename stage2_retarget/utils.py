"""
旋转表示转换工具函数
处理 SMPL 6D rotation / axis-angle / quaternion / rotation matrix 之间的转换
"""
import numpy as np
from scipy.spatial.transform import Rotation

# ============================================================
# SMPL 关节定义
# ============================================================
SMPL_JOINT_NAMES = [
    "pelvis",       # 0
    "l_hip",        # 1
    "r_hip",        # 2
    "spine1",       # 3
    "l_knee",       # 4
    "r_knee",       # 5
    "spine2",       # 6
    "l_ankle",      # 7
    "r_ankle",      # 8
    "spine3",       # 9
    "l_foot",       # 10
    "r_foot",       # 11
    "neck",         # 12
    "l_collar",     # 13
    "r_collar",     # 14
    "head",         # 15
    "l_shoulder",   # 16
    "r_shoulder",   # 17
    "l_elbow",      # 18
    "r_elbow",      # 19
    "l_wrist",      # 20
    "r_wrist",      # 21
    "l_hand",       # 22
    "r_hand",       # 23
]

SMPL_PARENT = [
    -1,  # 0  pelvis (root)
    0,   # 1  l_hip -> pelvis
    0,   # 2  r_hip -> pelvis
    0,   # 3  spine1 -> pelvis
    1,   # 4  l_knee -> l_hip
    2,   # 5  r_knee -> r_hip
    3,   # 6  spine2 -> spine1
    4,   # 7  l_ankle -> l_knee
    5,   # 8  r_ankle -> r_knee
    6,   # 9  spine3 -> spine2
    7,   # 10 l_foot -> l_ankle
    8,   # 11 r_foot -> r_ankle
    9,   # 12 neck -> spine3
    9,   # 13 l_collar -> spine3
    9,   # 14 r_collar -> spine3
    12,  # 15 head -> neck
    13,  # 16 l_shoulder -> l_collar
    14,  # 17 r_shoulder -> r_collar
    16,  # 18 l_elbow -> l_shoulder
    17,  # 19 r_elbow -> r_shoulder
    18,  # 20 l_wrist -> l_elbow
    19,  # 21 r_wrist -> r_elbow
    20,  # 22 l_hand -> l_wrist
    21,  # 23 r_hand -> r_wrist
]

# ============================================================
# 旋转转换函数
# ============================================================

def rot6d_to_rotmat(rot6d):
    """
    6D rotation -> 3x3 rotation matrix (Gram-Schmidt)
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

    Input: (..., 6)
    Output: (..., 3, 3)
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)


def rotmat_to_quat_wxyz(rotmat):
    """
    3x3 rotation matrix -> quaternion (w, x, y, z) — MuJoCo 格式

    Input: (..., 3, 3)
    Output: (..., 4)
    """
    shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)

    quats_xyzw = Rotation.from_matrix(rotmat_flat).as_quat()  # scipy: (x, y, z, w)
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]  # MuJoCo: (w, x, y, z)

    return quats_wxyz.reshape(*shape, 4)


def rotmat_to_axis_angle(rotmat):
    """
    3x3 rotation matrix -> axis-angle (3,)

    Input: (..., 3, 3)
    Output: (..., 3)
    """
    shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(rotmat_flat).as_rotvec()
    return aa.reshape(*shape, 3)


def rot6d_to_quat_wxyz(rot6d):
    """6D rotation -> MuJoCo quaternion (w, x, y, z)"""
    rotmat = rot6d_to_rotmat(rot6d)
    return rotmat_to_quat_wxyz(rotmat)


def rot6d_to_axis_angle(rot6d):
    """6D rotation -> axis-angle (3,)"""
    rotmat = rot6d_to_rotmat(rot6d)
    return rotmat_to_axis_angle(rotmat)


def axis_angle_to_quat_wxyz(aa):
    """
    Axis-angle -> quaternion (w, x, y, z) — MuJoCo 格式

    Input: (..., 3)
    Output: (..., 4)
    """
    shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)
    quats_xyzw = Rotation.from_rotvec(aa_flat).as_quat()  # (x, y, z, w)
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]  # (w, x, y, z)
    return quats_wxyz.reshape(*shape, 4)


def quat_wxyz_to_axis_angle(quat):
    """
    Quaternion (w, x, y, z) -> axis-angle (3,)

    Input: (..., 4)
    Output: (..., 3)
    """
    shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)
    quat_xyzw = quat_flat[:, [1, 2, 3, 0]]  # scipy format
    aa = Rotation.from_quat(quat_xyzw).as_rotvec()
    return aa.reshape(*shape, 3)


def quat_multiply(q1, q2):
    """
    四元数乘法 q1 * q2 (w, x, y, z) 格式

    Input: (..., 4), (..., 4)
    Output: (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.stack([w, x, y, z], axis=-1)


def quat_inverse(q):
    """
    四元数求逆 (w, x, y, z) 格式
    对于 unit quaternion，逆 = 共轭

    Input: (..., 4)
    Output: (..., 4)
    """
    return q * np.array([1, -1, -1, -1])


def normalize_quat(q):
    """归一化四元数"""
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


# ============================================================
# SMPL 动作解析
# ============================================================

def parse_edge_motion(motion_data):
    """
    解析 EDGE 输出的动作数据
    Input: (K, 151) numpy array
    Returns:
        joint_rot6d: (K, 24, 6) — 24 joints x 6D rotation
        root_trans: (K, 3) — root translation
        foot_contact: (K, 4) — foot contact labels
    """
    K = motion_data.shape[0]
    assert motion_data.shape[1] == 151, f"Expected 151, got {motion_data.shape[1]}"

    joint_rot6d = motion_data[:, :144].reshape(K, 24, 6)
    root_trans = motion_data[:, 144:147]
    foot_contact = motion_data[:, 147:151]

    return joint_rot6d, root_trans, foot_contact


def scale_motion(joint_rot6d, root_trans, scale_factor=1.0):
    """
    缩放动作幅度（用于降低摔倒风险）
    scale_factor < 1.0 会减小动作幅度
    """
    if scale_factor == 1.0:
        return joint_rot6d, root_trans

    # 将旋转转为 axis-angle，缩放角度，再转回 6D
    K, J = joint_rot6d.shape[:2]
    aa = rot6d_to_axis_angle(joint_rot6d)  # (K, 24, 3)
    aa_scaled = aa * scale_factor

    # 转回 rotation matrix 再取前两列作为 6D
    rotmat = Rotation.from_rotvec(aa_scaled.reshape(-1, 3)).as_matrix()
    rotmat = rotmat.reshape(K, J, 3, 3)
    scaled_rot6d = np.concatenate([rotmat[..., 0], rotmat[..., 1]], axis=-1)  # (K, 24, 6)

    # root translation 的水平位移也缩放
    scaled_trans = root_trans.copy()
    scaled_trans[:, :2] *= scale_factor  # x, y
    # z (高度) 不缩放

    return scaled_rot6d, scaled_trans
