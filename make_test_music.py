"""
生成测试用的 WAV 文件（简单节拍音频）
用于在没有真实舞曲时测试 pipeline
"""
import numpy as np
import os
import wave

SR = 22050


def make_test_wav(name, bpm, duration=5.0, out_dir="music"):
    """Generate a simple test WAV with clicks at the given BPM."""
    os.makedirs(out_dir, exist_ok=True)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    beat_interval = 60.0 / bpm
    signal = np.zeros_like(t)

    # Base tone
    signal += 0.3 * np.sin(2 * np.pi * 220 * t)
    signal += 0.15 * np.sin(2 * np.pi * 110 * t)

    # Beat clicks (decaying sine bursts at 1000 Hz)
    for beat_time in np.arange(0, duration, beat_interval):
        idx_start = int(beat_time * SR)
        idx_end = min(idx_start + int(0.02 * SR), len(signal))
        click_t = np.linspace(0, 0.02, idx_end - idx_start)
        signal[idx_start:idx_end] += 0.7 * np.sin(2 * np.pi * 1000 * click_t) * np.exp(-click_t * 100)

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    signal_int16 = (signal * 32767).astype(np.int16)

    path = os.path.join(out_dir, f"{name}.wav")
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(signal_int16.tobytes())
    print(f"  Created {path} ({duration}s, {bpm} BPM)")


if __name__ == "__main__":
    make_test_wav("hiphop", bpm=90, duration=5.0)
    make_test_wav("ballet", bpm=60, duration=5.0)
    make_test_wav("house", bpm=128, duration=5.0)
    print("Done! 3 test WAV files created.")
