"""
Physical Stability Rate (PSR)
衡量机器人在仿真中不摔倒的帧比例
"""
import numpy as np


def physical_stability_rate(com_heights, threshold=0.3):
    """
    Physical Stability Rate = 稳定帧数 / 总帧数

    Args:
        com_heights: (T,) 每帧的质心高度 (m)
        threshold: 质心低于此高度判定为摔倒 (m)

    Returns:
        rate: float in [0, 1]
    """
    stable = np.sum(com_heights > threshold)
    total = len(com_heights)
    return float(stable / total) if total > 0 else 0.0


def first_fall_frame(com_heights, threshold=0.3):
    """找到第一次摔倒的帧索引，-1 表示从未摔倒"""
    for i, h in enumerate(com_heights):
        if h < threshold:
            return i
    return -1


def stability_over_time(com_heights, threshold=0.3, window_sec=1.0, fps=30):
    """
    滑动窗口稳定性分析

    Returns:
        times: (N,) 时间点
        rates: (N,) 每个时间点的局部稳定率
    """
    window = int(window_sec * fps)
    if window < 1:
        window = 1

    rates = []
    times = []
    for i in range(0, len(com_heights) - window + 1, window // 2):
        chunk = com_heights[i:i + window]
        rate = np.mean(chunk > threshold)
        rates.append(rate)
        times.append((i + window // 2) / fps)

    return np.array(times), np.array(rates)


def compute_psr(sim_result_dir, name, threshold=0.3, fps=30):
    """
    从仿真结果目录计算 PSR

    Returns:
        result: dict
    """
    import os
    com_path = os.path.join(sim_result_dir, f"{name}_com_height.npy")
    if not os.path.exists(com_path):
        # 尝试 physics 前缀
        com_path = os.path.join(sim_result_dir, f"{name}_physics_com_height.npy")

    com_heights = np.load(com_path)

    psr = physical_stability_rate(com_heights, threshold)
    first_fall = first_fall_frame(com_heights, threshold)
    times, windowed_rates = stability_over_time(com_heights, threshold, fps=fps)

    return {
        "physical_stability_rate": psr,
        "first_fall_frame": first_fall,
        "first_fall_time": first_fall / fps if first_fall >= 0 else -1,
        "total_frames": len(com_heights),
        "duration_sec": len(com_heights) / fps,
        "com_height_mean": float(np.mean(com_heights)),
        "com_height_min": float(np.min(com_heights)),
    }
