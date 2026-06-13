"""SMPL-24 forward kinematics with hardcoded approximate rest-pose offsets.

Given axis-angle rotations (T, 24, 3) and root translation (T, 3), returns
per-frame joint world positions (T, 24, 3). The rest-pose offsets are
approximate (not from an actual SMPL body model) but close enough for
visualization.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# SMPL-24 kinematic tree. Parent of joint i (pelvis has no parent = -1).
PARENTS = np.array([-1,
                     0, 0, 0,
                     1, 2, 3,
                     4, 5, 6,
                     7, 8, 9,
                     9, 9, 12,
                     13, 14,
                     16, 17,
                     18, 19,
                     20, 21], dtype=np.int32)

# Approximate T-pose joint positions in world frame (meters, Y-up).
REST = np.array([
    [ 0.00, 0.00,  0.00],   # 0 pelvis
    [ 0.06,-0.09,  0.00],   # 1 L hip
    [-0.06,-0.09,  0.00],   # 2 R hip
    [ 0.00, 0.12,  0.00],   # 3 spine1
    [ 0.09,-0.47, -0.02],   # 4 L knee
    [-0.09,-0.47, -0.02],   # 5 R knee
    [ 0.00, 0.26,  0.00],   # 6 spine2
    [ 0.09,-0.87, -0.04],   # 7 L ankle
    [-0.09,-0.87, -0.04],   # 8 R ankle
    [ 0.00, 0.36,  0.02],   # 9 spine3
    [ 0.12,-0.93,  0.08],   # 10 L foot (toe)
    [-0.12,-0.93,  0.08],   # 11 R foot
    [ 0.00, 0.50, -0.02],   # 12 neck
    [ 0.07, 0.45, -0.01],   # 13 L collar
    [-0.07, 0.45, -0.01],   # 14 R collar
    [ 0.00, 0.66,  0.03],   # 15 head
    [ 0.18, 0.45, -0.02],   # 16 L shoulder
    [-0.18, 0.45, -0.02],   # 17 R shoulder
    [ 0.44, 0.45, -0.04],   # 18 L elbow
    [-0.44, 0.45, -0.04],   # 19 R elbow
    [ 0.67, 0.45, -0.03],   # 20 L wrist
    [-0.67, 0.45, -0.03],   # 21 R wrist
    [ 0.74, 0.45, -0.03],   # 22 L hand
    [-0.74, 0.45, -0.03],   # 23 R hand
], dtype=np.float64)

# Local offset from each joint to its parent (in parent's rest orientation).
# In T-pose, all frames are aligned with world, so local offset = REST[i]-REST[parent].
_LOCAL_OFFSET = np.zeros_like(REST)
for i in range(24):
    p = PARENTS[i]
    _LOCAL_OFFSET[i] = REST[i] - REST[p] if p >= 0 else REST[i]


# Skeleton edges for drawing
EDGES = [(p, i) for i, p in enumerate(PARENTS) if p >= 0]


def fk(poses: np.ndarray, trans: np.ndarray, return_R: bool = False):
    """poses: (T, 72) axis-angle. trans: (T, 3) root translation.
    Returns joint positions (T, 24, 3) in world frame.
    If return_R=True, also returns (T, 24, 3, 3) world rotations per joint."""
    T = poses.shape[0]
    aa = poses.reshape(T, 24, 3)
    out = np.zeros((T, 24, 3), dtype=np.float64)
    outR = np.zeros((T, 24, 3, 3), dtype=np.float64)
    for t in range(T):
        local_R = R.from_rotvec(aa[t]).as_matrix()  # (24, 3, 3)
        global_R = [None] * 24
        global_p = [None] * 24
        for i in range(24):
            p = PARENTS[i]
            if p < 0:
                global_R[i] = local_R[i]
                global_p[i] = _LOCAL_OFFSET[i].copy()
            else:
                global_R[i] = global_R[p] @ local_R[i]
                global_p[i] = global_p[p] + global_R[p] @ _LOCAL_OFFSET[i]
            out[t, i] = global_p[i]
            outR[t, i] = global_R[i]
        out[t] += trans[t]
    if return_R:
        return out, outR
    return out
