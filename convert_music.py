"""
将原始音乐文件转为 WAV 格式并截取合适长度
"""
import os
import numpy as np
import wave

# imageio-ffmpeg bundled ffmpeg path
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SRC_DIR = "C:/Users/Sijie/Documents/xwechat_files/wxid_6y6v87yp50w222_b3c4/msg/file/2026-04"
DST_DIR = "C:/Users/Sijie/Desktop/智能机器人导论proposal/project/music"

# File mapping: (source filename, target genre name)
FILES = [
    (" 月亮代表我的心 (伴奏)", "ballet"),
    (" 麦恩莉 (伴奏)", "hiphop"),
]

CLIP_DURATION = 10  # seconds
SR = 22050


def convert_m4a_to_wav(src_path, dst_path, start_sec=None, duration=CLIP_DURATION):
    """Use ffmpeg to convert m4a to wav, optionally clipping."""
    import subprocess

    # First get duration to find the middle
    probe_cmd = [FFMPEG, "-i", src_path, "-f", "null", "-"]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, errors="replace")
    stderr = result.stderr

    # Parse duration from ffmpeg output
    total_duration = None
    for line in stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            total_duration = int(h) * 3600 + int(m) * 60 + float(s)
            break

    if total_duration and start_sec is None:
        # Start from middle
        start_sec = max(0, (total_duration - duration) / 2)

    cmd = [FFMPEG, "-y", "-i", src_path]
    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-t", str(duration), "-ar", str(SR), "-ac", "1", dst_path]

    subprocess.run(cmd, capture_output=True)
    print(f"  Converted: {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
    if total_duration:
        print(f"    Total: {total_duration:.1f}s, Clipped: {start_sec:.1f}s ~ {start_sec+duration:.1f}s")


def generate_house_music(dst_path, bpm=128, duration=10):
    """Generate a synthetic house music track with four-on-the-floor beat."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    signal = np.zeros_like(t)
    beat_interval = 60.0 / bpm

    # Kick drum (four-on-the-floor)
    for beat_time in np.arange(0, duration, beat_interval):
        idx = int(beat_time * SR)
        length = min(int(0.15 * SR), len(signal) - idx)
        if length <= 0:
            continue
        kick_t = np.linspace(0, 0.15, length)
        # Exponentially decaying sine sweep (150Hz -> 50Hz)
        freq = 150 * np.exp(-kick_t * 10) + 50
        phase = np.cumsum(2 * np.pi * freq / SR)
        signal[idx:idx+length] += 0.8 * np.sin(phase) * np.exp(-kick_t * 15)

    # Hi-hat (offbeat eighth notes)
    for beat_time in np.arange(beat_interval / 2, duration, beat_interval):
        idx = int(beat_time * SR)
        length = min(int(0.03 * SR), len(signal) - idx)
        if length <= 0:
            continue
        hh_t = np.linspace(0, 0.03, length)
        noise = np.random.randn(length)
        signal[idx:idx+length] += 0.25 * noise * np.exp(-hh_t * 150)

    # Bass line (simple pattern)
    bass_notes = [55, 55, 65.4, 55]  # A1, A1, C2, A1
    beats_per_bar = 4
    bar_duration = beats_per_bar * beat_interval
    for bar_start in np.arange(0, duration, bar_duration):
        for i, note_freq in enumerate(bass_notes):
            note_start = bar_start + i * beat_interval
            idx = int(note_start * SR)
            length = min(int(beat_interval * 0.8 * SR), len(signal) - idx)
            if length <= 0:
                continue
            bass_t = np.linspace(0, beat_interval * 0.8, length)
            signal[idx:idx+length] += 0.4 * np.sin(2 * np.pi * note_freq * bass_t) * np.exp(-bass_t * 3)

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.85
    signal_int16 = (signal * 32767).astype(np.int16)

    with wave.open(dst_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(signal_int16.tobytes())

    print(f"  Generated synthetic house music: {dst_path} ({duration}s, {bpm} BPM)")


if __name__ == "__main__":
    os.makedirs(DST_DIR, exist_ok=True)

    # Convert real music files
    for src_name, genre in FILES:
        src_path = os.path.join(SRC_DIR, src_name)
        dst_path = os.path.join(DST_DIR, f"{genre}.wav")
        if os.path.exists(src_path):
            convert_m4a_to_wav(src_path, dst_path)
        else:
            print(f"  WARNING: File not found: {src_path}")

    # Generate synthetic house track
    house_path = os.path.join(DST_DIR, "house.wav")
    generate_house_music(house_path, bpm=128, duration=10)

    # Verify
    print("\nFinal music files:")
    for f in os.listdir(DST_DIR):
        if f.endswith(".wav"):
            fpath = os.path.join(DST_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f}: {size_kb:.0f} KB")
