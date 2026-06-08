"""
Beat Alignment Score (BAS)
衡量动作节拍与音乐节拍的对齐程度

参考: Li et al. "AI Choreographer: Music Conditioned 3D Dance Generation with AIST++" (ICCV 2021)
EDGE 变体: 使用关节加速度峰值作为运动节拍
"""
import numpy as np

try:
    import librosa
except ImportError:
    print("WARNING: librosa not installed. Music beat extraction will not work.")
    librosa = None


def extract_music_beats(wav_path):
    """
    从音乐文件中提取节拍时间点

    Returns:
        beat_times: (N,) 节拍时间（秒）
        tempo: float 估计的 BPM
    """
    assert librosa is not None, "librosa required for music beat extraction"

    y, sr = librosa.load(wav_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return beat_times, float(tempo)


def extract_motion_beats(motion_data, fps=30, use_root=False):
    """
    从动作数据中提取运动节拍（加速度峰值）

    Args:
        motion_data: (K, 151) EDGE 原始格式 或 (K, D) 关节位置
        fps: 帧率
        use_root: 如果 True，只用 root translation 计算

    Returns:
        beat_times: (M,) 运动节拍时间（秒）
        accel_magnitude: (K-2,) 加速度范数序列
    """
    from scipy.signal import find_peaks

    if motion_data.shape[1] == 151:
        # EDGE 格式: 用关节旋转数据
        if use_root:
            data = motion_data[:, 144:147]  # root translation only
        else:
            data = motion_data[:, :144]  # all joint rotations
    else:
        data = motion_data

    # 一阶差分 -> 速度
    velocity = np.diff(data, axis=0) * fps  # (K-1, D)

    # 二阶差分 -> 加速度
    acceleration = np.diff(velocity, axis=0) * fps  # (K-2, D)

    # 加速度 L2 范数
    accel_magnitude = np.linalg.norm(acceleration, axis=1)  # (K-2,)

    # 平滑（减少噪声）
    kernel_size = 5
    if len(accel_magnitude) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        accel_smoothed = np.convolve(accel_magnitude, kernel, mode="same")
    else:
        accel_smoothed = accel_magnitude

    # 寻找峰值
    height_threshold = np.mean(accel_smoothed) + 0.5 * np.std(accel_smoothed)
    min_distance = int(fps * 0.25)  # 至少间隔 0.25 秒
    peaks, properties = find_peaks(
        accel_smoothed,
        height=height_threshold,
        distance=min_distance,
    )

    # 转换为时间
    motion_beat_times = (peaks + 1) / fps  # +1 因为两次差分

    return motion_beat_times, accel_magnitude


def beat_alignment_score(music_beats, motion_beats, sigma=0.1):
    """
    计算 Beat Alignment Score

    BAS = (1/N) * Σ_i exp(-min_j(t_i^motion - t_j^music)^2 / (2σ^2))

    Args:
        music_beats: (N,) 音乐节拍时间（秒）
        motion_beats: (M,) 运动节拍时间（秒）
        sigma: Gaussian kernel 宽度（秒）

    Returns:
        score: float in [0, 1], 越高越好
    """
    if len(motion_beats) == 0 or len(music_beats) == 0:
        return 0.0

    scores = []
    for mb in motion_beats:
        # 找到最近的音乐节拍
        min_dist = np.min(np.abs(music_beats - mb))
        score = np.exp(-min_dist ** 2 / (2 * sigma ** 2))
        scores.append(score)

    return float(np.mean(scores))


def compute_bas(wav_path, motion_path, sigma=0.1, fps=30):
    """
    一站式计算 BAS

    Returns:
        result: dict with score, music_beats, motion_beats, tempo
    """
    music_beats, tempo = extract_music_beats(wav_path)
    motion = np.load(motion_path)
    motion_beats, accel = extract_motion_beats(motion, fps=fps)

    score = beat_alignment_score(music_beats, motion_beats, sigma=sigma)

    return {
        "beat_alignment_score": score,
        "tempo": tempo,
        "n_music_beats": len(music_beats),
        "n_motion_beats": len(motion_beats),
        "music_beats": music_beats,
        "motion_beats": motion_beats,
    }
