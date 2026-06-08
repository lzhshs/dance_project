"""
关键帧编排舞蹈生成器
直接输出 MuJoCo qpos 格式 (K, 79)，跳过 SMPL 中间格式。

每个体裁定义一组关键 pose，在关键帧之间用三次样条插值，
并根据音乐节拍和 onset 调制动作幅度和节奏。

使用方法:
    conda activate sim
    python generate_choreography.py --input ../music --output ../outputs/retargeted_motions/strategy_a
"""
import argparse
import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed. Run: pip install librosa")
    exit(1)

# ============================================================
# MuJoCo qpos layout for smpl_humanoid.xml (nq=79)
# ============================================================
# [  0:  7] root         (free: 3 pos + 4 quat wxyz)
# [  7: 11] spine1       (ball: 4 quat)
# [ 11: 15] spine2       (ball: 4 quat)
# [ 15: 19] spine3       (ball: 4 quat)
# [ 19: 23] neck         (ball: 4 quat)
# [ 23: 27] head         (ball: 4 quat)
# [ 27: 31] l_collar     (ball: 4 quat)
# [ 31: 35] l_shoulder   (ball: 4 quat)
# [ 35: 36] l_elbow      (hinge: 1 angle, range [-2.5, 0])
# [ 36: 40] l_wrist      (ball: 4 quat)
# [ 40: 44] r_collar     (ball: 4 quat)
# [ 44: 48] r_shoulder   (ball: 4 quat)
# [ 48: 49] r_elbow      (hinge: 1 angle, range [-2.5, 0])
# [ 49: 53] r_wrist      (ball: 4 quat)
# [ 53: 57] l_hip        (ball: 4 quat)
# [ 57: 58] l_knee       (hinge: 1 angle, range [0, 2.5])
# [ 58: 62] l_ankle      (ball: 4 quat)
# [ 62: 66] l_foot       (ball: 4 quat)
# [ 66: 70] r_hip        (ball: 4 quat)
# [ 70: 71] r_knee       (hinge: 1 angle, range [0, 2.5])
# [ 71: 75] r_ankle      (ball: 4 quat)
# [ 75: 79] r_foot       (ball: 4 quat)

NQ = 79
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z


def aa_to_quat(axis_angle):
    """Axis-angle (3,) -> MuJoCo quaternion (w,x,y,z)."""
    r = Rotation.from_rotvec(axis_angle)
    q_xyzw = r.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def euler_to_quat(roll, pitch, yaw):
    """Euler angles (xyz order) -> MuJoCo quaternion (w,x,y,z)."""
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q_xyzw = r.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two wxyz quaternions."""
    # Convert to scipy format (xyzw)
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    r = Slerp([0, 1], Rotation.from_quat([q0_xyzw, q1_xyzw]))
    q_xyzw = r([t])[0].as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def standing_pose():
    """Default standing pose (all identity quaternions)."""
    qpos = np.zeros(NQ)
    qpos[0:3] = [0, 0, 0.91]  # root position
    qpos[3:7] = IDENTITY_QUAT  # root orientation
    # All ball joints = identity
    for addr in [7, 11, 15, 19, 23, 27, 31, 36, 40, 44, 49, 53, 58, 62, 66, 71, 75]:
        qpos[addr:addr+4] = IDENTITY_QUAT
    # Hinge joints = 0
    qpos[35] = 0   # l_elbow
    qpos[48] = 0   # r_elbow
    qpos[57] = 0   # l_knee
    qpos[70] = 0   # r_knee
    return qpos


def make_pose(root_pos=None, root_euler=None,
              spine1_euler=None, spine2_euler=None, spine3_euler=None,
              neck_euler=None, head_euler=None,
              l_collar_euler=None, l_shoulder_euler=None, l_elbow=None, l_wrist_euler=None,
              r_collar_euler=None, r_shoulder_euler=None, r_elbow=None, r_wrist_euler=None,
              l_hip_euler=None, l_knee=None, l_ankle_euler=None, l_foot_euler=None,
              r_hip_euler=None, r_knee=None, r_ankle_euler=None, r_foot_euler=None):
    """Create a pose from euler angles (degrees) for readability."""
    qpos = standing_pose()

    if root_pos is not None:
        qpos[0:3] = root_pos
    if root_euler is not None:
        qpos[3:7] = euler_to_quat(*np.radians(root_euler))

    joint_map = [
        (spine1_euler, 7), (spine2_euler, 11), (spine3_euler, 15),
        (neck_euler, 19), (head_euler, 23),
        (l_collar_euler, 27), (l_shoulder_euler, 31),
        (l_wrist_euler, 36),
        (r_collar_euler, 40), (r_shoulder_euler, 44),
        (r_wrist_euler, 49),
        (l_hip_euler, 53), (l_ankle_euler, 58), (l_foot_euler, 62),
        (r_hip_euler, 66), (r_ankle_euler, 71), (r_foot_euler, 75),
    ]
    for euler, addr in joint_map:
        if euler is not None:
            qpos[addr:addr+4] = euler_to_quat(*np.radians(euler))

    # Hinge joints (in radians)
    if l_elbow is not None:
        qpos[35] = np.clip(np.radians(l_elbow), -2.5, 0)
    if r_elbow is not None:
        qpos[48] = np.clip(np.radians(r_elbow), -2.5, 0)
    if l_knee is not None:
        qpos[57] = np.clip(np.radians(l_knee), 0, 2.5)
    if r_knee is not None:
        qpos[70] = np.clip(np.radians(r_knee), 0, 2.5)

    return qpos


def interpolate_poses(pose_a, pose_b, n_frames):
    """Smoothly interpolate between two poses over n_frames."""
    result = np.zeros((n_frames, NQ))

    # Interpolation parameter with smooth ease-in-out
    for i in range(n_frames):
        raw_t = i / max(n_frames - 1, 1)
        # Smooth step (cubic ease-in-out)
        t = raw_t * raw_t * (3 - 2 * raw_t)

        # Root position: linear
        result[i, 0:3] = pose_a[0:3] * (1 - t) + pose_b[0:3] * t

        # Root orientation: slerp
        result[i, 3:7] = slerp_quat(pose_a[3:7], pose_b[3:7], t)

        # Ball joints: slerp
        ball_addrs = [7, 11, 15, 19, 23, 27, 31, 36, 40, 44, 49, 53, 58, 62, 66, 71, 75]
        for addr in ball_addrs:
            result[i, addr:addr+4] = slerp_quat(pose_a[addr:addr+4], pose_b[addr:addr+4], t)

        # Hinge joints: linear
        for addr in [35, 48, 57, 70]:
            result[i, addr] = pose_a[addr] * (1 - t) + pose_b[addr] * t

    return result


# ============================================================
# Genre-specific keyframe choreography
# ============================================================

def get_hiphop_keyframes():
    """Hiphop: bounce, body wave, arm pop."""
    poses = []

    # Pose 0: Standing bounce down (knees bent, arms in)
    poses.append(make_pose(
        root_pos=[0, 0, 0.85],
        spine1_euler=[0, 0, 0],
        l_shoulder_euler=[0, 0, 30], r_shoulder_euler=[0, 0, -30],
        l_elbow=-70, r_elbow=-70,
        l_hip_euler=[15, 0, 0], r_hip_euler=[15, 0, 0],
        l_knee=30, r_knee=30,
    ))

    # Pose 1: Bounce up, arms out to sides
    poses.append(make_pose(
        root_pos=[0, 0, 0.91],
        spine1_euler=[0, 5, 0],
        l_shoulder_euler=[0, -10, 70], r_shoulder_euler=[0, 10, -70],
        l_elbow=-30, r_elbow=-30,
        l_hip_euler=[5, 0, 0], r_hip_euler=[5, 0, 0],
        l_knee=10, r_knee=10,
    ))

    # Pose 2: Lean right, left arm up
    poses.append(make_pose(
        root_pos=[0.02, 0, 0.87],
        root_euler=[0, 0, -8],
        spine1_euler=[5, 0, -10],
        spine2_euler=[3, 0, -5],
        l_shoulder_euler=[0, -20, 90], r_shoulder_euler=[0, 15, -40],
        l_elbow=-20, r_elbow=-80,
        l_hip_euler=[10, -5, 0], r_hip_euler=[20, 5, 0],
        l_knee=15, r_knee=35,
    ))

    # Pose 3: Bounce down, arms crossed
    poses.append(make_pose(
        root_pos=[0, 0, 0.83],
        spine1_euler=[5, 0, 0],
        spine2_euler=[3, 0, 0],
        l_shoulder_euler=[20, 0, 40], r_shoulder_euler=[20, 0, -40],
        l_elbow=-90, r_elbow=-90,
        l_hip_euler=[20, 0, 0], r_hip_euler=[20, 0, 0],
        l_knee=40, r_knee=40,
    ))

    # Pose 4: Lean left, right arm up
    poses.append(make_pose(
        root_pos=[-0.02, 0, 0.87],
        root_euler=[0, 0, 8],
        spine1_euler=[5, 0, 10],
        spine2_euler=[3, 0, 5],
        l_shoulder_euler=[0, -15, 40], r_shoulder_euler=[0, 20, -90],
        l_elbow=-80, r_elbow=-20,
        l_hip_euler=[20, 5, 0], r_hip_euler=[10, -5, 0],
        l_knee=35, r_knee=15,
    ))

    # Pose 5: Bounce up, body wave starting
    poses.append(make_pose(
        root_pos=[0, 0, 0.90],
        spine1_euler=[8, 0, 0],
        spine2_euler=[-5, 0, 0],
        spine3_euler=[-3, 0, 0],
        l_shoulder_euler=[0, -5, 50], r_shoulder_euler=[0, 5, -50],
        l_elbow=-45, r_elbow=-45,
        l_hip_euler=[10, 0, 0], r_hip_euler=[10, 0, 0],
        l_knee=15, r_knee=15,
    ))

    # Pose 6: Body wave peak
    poses.append(make_pose(
        root_pos=[0, 0.01, 0.88],
        spine1_euler=[-5, 0, 0],
        spine2_euler=[10, 0, 0],
        spine3_euler=[5, 0, 0],
        neck_euler=[-5, 0, 0],
        l_shoulder_euler=[10, -5, 60], r_shoulder_euler=[10, 5, -60],
        l_elbow=-50, r_elbow=-50,
        l_hip_euler=[15, 0, 0], r_hip_euler=[15, 0, 0],
        l_knee=25, r_knee=25,
    ))

    # Pose 7: Arm pop right
    poses.append(make_pose(
        root_pos=[0, 0, 0.86],
        spine1_euler=[3, 0, -5],
        l_shoulder_euler=[0, -10, 30], r_shoulder_euler=[-30, 10, -110],
        l_elbow=-60, r_elbow=-10,
        l_hip_euler=[15, 0, 0], r_hip_euler=[15, 0, 0],
        l_knee=25, r_knee=30,
    ))

    return poses


def get_ballet_keyframes():
    """Ballet: graceful arm movements, releve, plié."""
    poses = []

    # Pose 0: First position (arms low, feet together)
    poses.append(make_pose(
        root_pos=[0, 0, 0.91],
        l_shoulder_euler=[0, -5, 25], r_shoulder_euler=[0, 5, -25],
        l_elbow=-15, r_elbow=-15,
        l_hip_euler=[0, 0, 10], r_hip_euler=[0, 0, -10],
        l_knee=0, r_knee=0,
    ))

    # Pose 1: Demi-plié (knees bent, arms opening)
    poses.append(make_pose(
        root_pos=[0, 0, 0.82],
        spine1_euler=[3, 0, 0],
        l_shoulder_euler=[0, -10, 45], r_shoulder_euler=[0, 10, -45],
        l_elbow=-10, r_elbow=-10,
        l_hip_euler=[20, 0, 10], r_hip_euler=[20, 0, -10],
        l_knee=40, r_knee=40,
        l_ankle_euler=[-10, 0, 0], r_ankle_euler=[-10, 0, 0],
    ))

    # Pose 2: Relevé (rise up, arms high fifth)
    poses.append(make_pose(
        root_pos=[0, 0, 0.95],
        spine1_euler=[-3, 0, 0],
        spine2_euler=[-2, 0, 0],
        l_shoulder_euler=[-10, -10, 100], r_shoulder_euler=[-10, 10, -100],
        l_elbow=-15, r_elbow=-15,
        l_hip_euler=[0, 0, 5], r_hip_euler=[0, 0, -5],
        l_knee=0, r_knee=0,
        l_ankle_euler=[20, 0, 0], r_ankle_euler=[20, 0, 0],
    ))

    # Pose 3: Tendu right (right leg extended, arms in second)
    poses.append(make_pose(
        root_pos=[0.01, 0, 0.90],
        spine1_euler=[0, 0, 3],
        l_shoulder_euler=[0, -10, 80], r_shoulder_euler=[0, 10, -80],
        l_elbow=-10, r_elbow=-10,
        l_hip_euler=[5, 0, 5], r_hip_euler=[-5, 0, -25],
        l_knee=5, r_knee=0,
    ))

    # Pose 4: Développé (working leg lifts, supporting arm high)
    poses.append(make_pose(
        root_pos=[0.01, 0, 0.88],
        spine1_euler=[0, 0, 5],
        spine2_euler=[-3, 0, 3],
        l_shoulder_euler=[-5, -10, 100], r_shoulder_euler=[0, 10, -70],
        l_elbow=-15, r_elbow=-15,
        l_hip_euler=[5, 0, 5], r_hip_euler=[-30, 0, -15],
        l_knee=10, r_knee=10,
    ))

    # Pose 5: Port de bras forward (bow, arms sweeping)
    poses.append(make_pose(
        root_pos=[0, 0.01, 0.85],
        spine1_euler=[15, 0, 0],
        spine2_euler=[10, 0, 0],
        neck_euler=[5, 0, 0],
        l_shoulder_euler=[20, -5, 60], r_shoulder_euler=[20, 5, -60],
        l_elbow=-15, r_elbow=-15,
        l_hip_euler=[10, 0, 5], r_hip_euler=[10, 0, -5],
        l_knee=15, r_knee=15,
    ))

    # Pose 6: Arms in third (one arm up, one side)
    poses.append(make_pose(
        root_pos=[0, 0, 0.92],
        spine1_euler=[-2, 0, -3],
        l_shoulder_euler=[-10, -10, 105], r_shoulder_euler=[0, 10, -70],
        l_elbow=-15, r_elbow=-10,
        l_hip_euler=[0, 0, 8], r_hip_euler=[3, 0, -8],
        l_knee=0, r_knee=5,
    ))

    # Pose 7: Plié in second position
    poses.append(make_pose(
        root_pos=[0, 0, 0.80],
        spine1_euler=[2, 0, 0],
        l_shoulder_euler=[0, -10, 60], r_shoulder_euler=[0, 10, -60],
        l_elbow=-10, r_elbow=-10,
        l_hip_euler=[25, 0, 15], r_hip_euler=[25, 0, -15],
        l_knee=50, r_knee=50,
        l_ankle_euler=[-8, 0, 0], r_ankle_euler=[-8, 0, 0],
    ))

    return poses


def get_house_keyframes():
    """House: jacking, stepping, stomp."""
    poses = []

    # Pose 0: Jack up (chest forward, arms back)
    poses.append(make_pose(
        root_pos=[0, 0, 0.91],
        spine1_euler=[-8, 0, 0],
        spine2_euler=[-5, 0, 0],
        l_shoulder_euler=[20, 0, 15], r_shoulder_euler=[20, 0, -15],
        l_elbow=-30, r_elbow=-30,
        l_hip_euler=[5, 0, 0], r_hip_euler=[5, 0, 0],
        l_knee=5, r_knee=5,
    ))

    # Pose 1: Jack down (chest back, arms forward)
    poses.append(make_pose(
        root_pos=[0, 0, 0.83],
        spine1_euler=[12, 0, 0],
        spine2_euler=[8, 0, 0],
        l_shoulder_euler=[-15, -5, 40], r_shoulder_euler=[-15, 5, -40],
        l_elbow=-60, r_elbow=-60,
        l_hip_euler=[25, 0, 0], r_hip_euler=[25, 0, 0],
        l_knee=35, r_knee=35,
    ))

    # Pose 2: Step left (weight on left, right lifted)
    poses.append(make_pose(
        root_pos=[-0.03, 0, 0.88],
        root_euler=[0, 0, 5],
        spine1_euler=[5, 0, 5],
        l_shoulder_euler=[-10, -5, 50], r_shoulder_euler=[10, 5, -30],
        l_elbow=-40, r_elbow=-50,
        l_hip_euler=[15, 5, 5], r_hip_euler=[10, -5, -5],
        l_knee=20, r_knee=25,
    ))

    # Pose 3: Jack up again
    poses.append(make_pose(
        root_pos=[0, 0, 0.90],
        spine1_euler=[-6, 0, 0],
        spine2_euler=[-4, 0, 0],
        l_shoulder_euler=[15, 0, 20], r_shoulder_euler=[15, 0, -20],
        l_elbow=-35, r_elbow=-35,
        l_hip_euler=[8, 0, 0], r_hip_euler=[8, 0, 0],
        l_knee=10, r_knee=10,
    ))

    # Pose 4: Step right (weight on right, left lifted)
    poses.append(make_pose(
        root_pos=[0.03, 0, 0.88],
        root_euler=[0, 0, -5],
        spine1_euler=[5, 0, -5],
        l_shoulder_euler=[10, -5, 30], r_shoulder_euler=[-10, 5, -50],
        l_elbow=-50, r_elbow=-40,
        l_hip_euler=[10, 5, 5], r_hip_euler=[15, -5, -5],
        l_knee=25, r_knee=20,
    ))

    # Pose 5: Jack down with twist
    poses.append(make_pose(
        root_pos=[0, 0, 0.84],
        root_euler=[0, 0, 8],
        spine1_euler=[10, 0, -5],
        spine2_euler=[5, 0, -3],
        l_shoulder_euler=[-10, -5, 55], r_shoulder_euler=[5, 5, -35],
        l_elbow=-45, r_elbow=-65,
        l_hip_euler=[20, 0, 5], r_hip_euler=[22, 0, -3],
        l_knee=30, r_knee=35,
    ))

    # Pose 6: Stomp right
    poses.append(make_pose(
        root_pos=[0.02, 0, 0.87],
        spine1_euler=[5, 0, -3],
        l_shoulder_euler=[5, -5, 35], r_shoulder_euler=[-5, 5, -55],
        l_elbow=-55, r_elbow=-30,
        l_hip_euler=[8, 0, 3], r_hip_euler=[18, 0, -8],
        l_knee=10, r_knee=30,
    ))

    # Pose 7: Bounce center
    poses.append(make_pose(
        root_pos=[0, 0, 0.86],
        spine1_euler=[8, 0, 0],
        spine2_euler=[5, 0, 0],
        l_shoulder_euler=[-5, -5, 45], r_shoulder_euler=[-5, 5, -45],
        l_elbow=-55, r_elbow=-55,
        l_hip_euler=[18, 0, 3], r_hip_euler=[18, 0, -3],
        l_knee=30, r_knee=30,
    ))

    return poses


def generate_dance_qpos(wav_path, output_path, genre=None, fps=30):
    """
    Generate MuJoCo qpos trajectory from music.

    The approach:
    1. Extract tempo and beat positions from audio
    2. Select genre-appropriate keyframe poses
    3. Assign keyframes to beats with some variation
    4. Interpolate smoothly between keyframes
    5. Modulate amplitude based on onset strength
    """
    y, sr = librosa.load(wav_path)
    duration = len(y) / sr
    K = int(duration * fps)

    # Audio analysis
    hop_length = sr // fps
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo)[0])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = onset_env[:K]
    if len(onset_env) < K:
        onset_env = np.pad(onset_env, (0, K - len(onset_env)))
    onset_env = onset_env / (onset_env.max() + 1e-8)

    # Chroma for variation
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma_sum = chroma.sum(axis=0)[:K]
    if len(chroma_sum) < K:
        chroma_sum = np.pad(chroma_sum, (0, K - len(chroma_sum)))

    print(f"  Audio: {duration:.1f}s, {K} frames, tempo={tempo:.1f} BPM")
    print(f"  Beats detected: {len(beat_frames)}")

    # Auto-detect genre from filename if not specified
    if genre is None:
        name = os.path.basename(wav_path).lower()
        if 'ballet' in name:
            genre = 'ballet'
        elif 'house' in name:
            genre = 'house'
        else:
            genre = 'hiphop'

    # Get keyframe poses for this genre
    if genre == 'ballet':
        keyframe_poses = get_ballet_keyframes()
    elif genre == 'house':
        keyframe_poses = get_house_keyframes()
    else:
        keyframe_poses = get_hiphop_keyframes()

    n_poses = len(keyframe_poses)
    stand = standing_pose()

    # Assign poses to beats
    # Each beat gets a keyframe, cycling through the pose library
    # We ensure a smooth cycle by including transitions back
    beat_frame_list = sorted(beat_frames[beat_frames < K].tolist())

    # Add start and end frames
    if len(beat_frame_list) == 0 or beat_frame_list[0] != 0:
        beat_frame_list.insert(0, 0)
    if beat_frame_list[-1] != K - 1:
        beat_frame_list.append(K - 1)

    # Build the full trajectory
    trajectory = np.zeros((K, NQ))

    # Assign a pose index to each beat
    # Use a pattern that creates visual variety
    np.random.seed(42)  # Reproducible but varied
    pose_sequence = []
    for i in range(len(beat_frame_list)):
        # Cycle through poses, with occasional returns to standing-like poses
        idx = i % n_poses
        pose_sequence.append(idx)

    # Interpolate between keyframes
    prev_frame = 0
    prev_pose = keyframe_poses[pose_sequence[0]]

    for beat_idx in range(1, len(beat_frame_list)):
        curr_frame = beat_frame_list[beat_idx]
        curr_pose = keyframe_poses[pose_sequence[beat_idx]]
        n_interp = curr_frame - prev_frame

        if n_interp <= 0:
            continue

        segment = interpolate_poses(prev_pose, curr_pose, n_interp)

        # Amplitude modulation: scale the deviation from standing pose
        for i in range(n_interp):
            frame_idx = prev_frame + i
            if frame_idx >= K:
                break
            amp = 0.6 + 0.4 * onset_env[min(frame_idx, len(onset_env) - 1)]

            # Blend between standing pose and choreographed pose based on amplitude
            for addr in [0, 1, 2]:  # root position
                deviation = segment[i, addr] - stand[addr]
                trajectory[frame_idx, addr] = stand[addr] + deviation * amp
            # Root quaternion
            trajectory[frame_idx, 3:7] = slerp_quat(stand[3:7], segment[i, 3:7], amp)
            # Ball joints
            for addr in [7, 11, 15, 19, 23, 27, 31, 36, 40, 44, 49, 53, 58, 62, 66, 71, 75]:
                trajectory[frame_idx, addr:addr+4] = slerp_quat(
                    stand[addr:addr+4], segment[i, addr:addr+4], amp)
            # Hinge joints
            for addr in [35, 48, 57, 70]:
                trajectory[frame_idx, addr] = segment[i, addr] * amp

        prev_frame = curr_frame
        prev_pose = curr_pose

    # Fill any remaining frames
    if prev_frame < K:
        for i in range(prev_frame, K):
            trajectory[i] = trajectory[max(0, prev_frame - 1)]

    # Post-processing: smooth the trajectory with a small Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    for j in range(NQ):
        # Don't smooth quaternion w components too aggressively
        sigma = 2.0
        trajectory[:, j] = gaussian_filter1d(trajectory[:, j], sigma=sigma)

    # Re-normalize all quaternions after smoothing
    for addr in [3, 7, 11, 15, 19, 23, 27, 31, 36, 40, 44, 49, 53, 58, 62, 66, 71, 75]:
        for i in range(K):
            q = trajectory[i, addr:addr+4]
            norm = np.linalg.norm(q)
            if norm > 1e-8:
                trajectory[i, addr:addr+4] = q / norm
            else:
                trajectory[i, addr:addr+4] = IDENTITY_QUAT

    # Ensure hinge joint limits
    trajectory[:, 35] = np.clip(trajectory[:, 35], -2.5, 0)   # l_elbow
    trajectory[:, 48] = np.clip(trajectory[:, 48], -2.5, 0)   # r_elbow
    trajectory[:, 57] = np.clip(trajectory[:, 57], 0, 2.5)    # l_knee
    trajectory[:, 70] = np.clip(trajectory[:, 70], 0, 2.5)    # r_knee

    np.save(output_path, trajectory)
    print(f"  Choreography: {trajectory.shape} -> {output_path}")
    print(f"  Genre: {genre}, root height range: "
          f"{trajectory[:, 2].min():.3f} ~ {trajectory[:, 2].max():.3f}")
    return trajectory


# Also generate the SMPL (K, 151) format for backwards compatibility
def generate_smpl_format(qpos_trajectory, output_path):
    """Convert MuJoCo qpos trajectory to EDGE-compatible (K, 151) format."""
    K = len(qpos_trajectory)
    motion = np.zeros((K, 151))

    # Initialize all joints to identity rotation [1,0,0,0,1,0]
    for j in range(24):
        motion[:, j * 6 + 0] = 1.0
        motion[:, j * 6 + 4] = 1.0

    # Root translation
    motion[:, 144:147] = qpos_trajectory[:, 0:3]

    # Convert quaternions back to 6D rotation for the joints that matter
    # This is lossy but maintains compatibility
    ball_to_smpl = {
        7: 3,    # spine1
        11: 6,   # spine2
        15: 9,   # spine3
        19: 12,  # neck
        23: 15,  # head
        27: 13,  # l_collar
        31: 16,  # l_shoulder
        36: 20,  # l_wrist
        40: 14,  # r_collar
        44: 17,  # r_shoulder
        49: 21,  # r_wrist
        53: 1,   # l_hip
        58: 7,   # l_ankle
        62: 10,  # l_foot
        66: 2,   # r_hip
        71: 8,   # r_ankle
        75: 11,  # r_foot
    }

    for qpos_addr, smpl_idx in ball_to_smpl.items():
        for t in range(K):
            q_wxyz = qpos_trajectory[t, qpos_addr:qpos_addr+4]
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            try:
                R = Rotation.from_quat(q_xyzw).as_matrix()
                motion[t, smpl_idx * 6: smpl_idx * 6 + 3] = R[:, 0]
                motion[t, smpl_idx * 6 + 3: smpl_idx * 6 + 6] = R[:, 1]
            except ValueError:
                pass  # Keep identity

    # Pelvis (root orientation)
    for t in range(K):
        q_wxyz = qpos_trajectory[t, 3:7]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        try:
            R = Rotation.from_quat(q_xyzw).as_matrix()
            motion[t, 0:3] = R[:, 0]
            motion[t, 3:6] = R[:, 1]
        except ValueError:
            pass

    # Foot contact (simple heuristic based on knee angles)
    l_knee = qpos_trajectory[:, 57]
    r_knee = qpos_trajectory[:, 70]
    motion[:, 147] = (l_knee < 0.3).astype(float)  # L heel
    motion[:, 148] = (l_knee < 0.3).astype(float)  # L toe
    motion[:, 149] = (r_knee < 0.3).astype(float)  # R heel
    motion[:, 150] = (r_knee < 0.3).astype(float)  # R toe

    np.save(output_path, motion)
    return motion


def main():
    parser = argparse.ArgumentParser(description="Keyframe choreography dance generator")
    parser.add_argument("--input", type=str, default="../music")
    parser.add_argument("--output", type=str, default="../outputs/retargeted_motions/strategy_a")
    parser.add_argument("--smpl_output", type=str, default="../outputs/generated_motions",
                        help="Also save EDGE-compatible (K,151) format")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.smpl_output:
        os.makedirs(args.smpl_output, exist_ok=True)

    wav_files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in {args.input}")
        return

    for wav_file in wav_files:
        name = wav_file.replace(".wav", "")
        print(f"\n{'='*50}")
        print(f"Generating choreography: {name}")
        print(f"{'='*50}")

        qpos_path = os.path.join(args.output, f"{name}_qpos.npy")
        qpos_traj = generate_dance_qpos(
            os.path.join(args.input, wav_file),
            qpos_path,
            genre=name,
        )

        # Also save SMPL format for evaluation compatibility
        if args.smpl_output:
            smpl_path = os.path.join(args.smpl_output, f"{name}_motion.npy")
            generate_smpl_format(qpos_traj, smpl_path)
            print(f"  SMPL format: {smpl_path}")

    print(f"\nDone! Generated choreography for {len(wav_files)} tracks.")


if __name__ == "__main__":
    main()
