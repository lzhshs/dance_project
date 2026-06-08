"""
Bailando 备选方案
当 Jukebox / EDGE 无法使用时，用 Bailando + librosa 替代

优点: 只需 librosa 提取特征，CPU 即可运行
缺点: 生成质量略低于 EDGE

使用方法:
    conda activate edge
    python generate_dance_bailando.py --input ../music --output ../outputs/generated_motions
"""
import argparse
import os
import sys
import numpy as np

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed. Run: pip install librosa")
    exit(1)

BAILANDO_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "Bailando")


def extract_audio_features(wav_path, sr=15360, fps=30):
    """
    用 librosa 提取音频特征（替代 Jukebox）
    返回: dict with mel, chroma, onset, tempo, beats
    """
    y, sr = librosa.load(wav_path, sr=sr)
    hop_length = sr // fps  # 每帧对应的采样数

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (T, 80)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).T  # (T, 12)

    # Onset strength
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)  # (T,)

    # Tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length).T  # (T, 20)

    # 拼接所有特征
    min_len = min(len(mel_db), len(chroma), len(onset), len(mfcc))
    features = np.concatenate([
        mel_db[:min_len],           # 80
        chroma[:min_len],           # 12
        onset[:min_len, None],      # 1
        mfcc[:min_len],             # 20
    ], axis=1)  # (T, 113)

    return {
        "features": features,
        "tempo": tempo,
        "beat_frames": beat_frames,
        "sr": sr,
        "fps": fps,
        "duration": len(y) / sr,
    }


def generate_with_bailando(wav_path, output_path):
    """
    使用 Bailando 生成舞蹈动作
    需要先安装 Bailando: git clone https://github.com/lisiyao21/Bailando.git
    """
    sys.path.insert(0, BAILANDO_DIR)

    try:
        # Bailando 的具体 API 取决于其代码结构
        # 这里提供一个框架，需要根据实际 Bailando 代码调整
        from bailando.generate import generate_dance

        audio_features = extract_audio_features(wav_path)
        motion = generate_dance(audio_features["features"])
        np.save(output_path, motion)
        print(f"  Bailando output: {motion.shape} -> {output_path}")

    except ImportError:
        print("  Bailando not installed. Generating placeholder motion...")
        # 生成占位动作用于测试 pipeline
        generate_placeholder_motion(wav_path, output_path)


def generate_placeholder_motion(wav_path, output_path, fps=30):
    """
    生成基于音乐节拍的占位动作，用于在 EDGE/Bailando 不可用时测试后续 pipeline
    输出格式与 EDGE 一致: (K, 151)

    根据音乐的 tempo 和 onset 来驱动动作频率和幅度，
    使不同音乐产生不同的动作模式。
    """
    y, sr = librosa.load(wav_path)
    duration = len(y) / sr
    K = int(duration * fps)

    # 提取音乐特征
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_freq = tempo / 60.0  # beats per second

    # Onset strength envelope (normalized)
    hop_length = sr // fps
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = onset_env[:K]
    if len(onset_env) < K:
        onset_env = np.pad(onset_env, (0, K - len(onset_env)))
    onset_env = onset_env / (onset_env.max() + 1e-8)

    motion = np.zeros((K, 151))
    t = np.linspace(0, duration, K)

    # 24 joints x 6D rotation, identity rotation: [1, 0, 0, 0, 1, 0]
    for j in range(24):
        motion[:, j * 6 + 0] = 1.0  # r1.x
        motion[:, j * 6 + 4] = 1.0  # r2.y

    # 动作幅度随 onset 变化
    amp_base = 0.15
    amp_mod = amp_base * (0.5 + 0.5 * onset_env)  # (K,)

    # 左肩 (joint 16) 和 右肩 (joint 17) — 跟随节拍摆动
    motion[:, 16 * 6 + 2] = amp_mod * np.sin(2 * np.pi * beat_freq * t)
    motion[:, 17 * 6 + 2] = -amp_mod * np.sin(2 * np.pi * beat_freq * t)

    # 左肩和右肩 — 横向也加一点
    motion[:, 16 * 6 + 1] = amp_mod * 0.3 * np.sin(2 * np.pi * beat_freq * 0.5 * t)
    motion[:, 17 * 6 + 1] = -amp_mod * 0.3 * np.sin(2 * np.pi * beat_freq * 0.5 * t)

    # 左髋 (joint 1) 和右髋 (joint 2) — 半拍频率
    motion[:, 1 * 6 + 2] = amp_mod * 0.6 * np.sin(2 * np.pi * beat_freq * 0.5 * t)
    motion[:, 2 * 6 + 2] = -amp_mod * 0.6 * np.sin(2 * np.pi * beat_freq * 0.5 * t)

    # 左膝 (joint 4) 和右膝 (joint 5) — 弯曲
    motion[:, 4 * 6 + 2] = amp_mod * 0.4 * np.abs(np.sin(2 * np.pi * beat_freq * t))
    motion[:, 5 * 6 + 2] = amp_mod * 0.4 * np.abs(np.sin(2 * np.pi * beat_freq * t + np.pi * 0.5))

    # 脊柱 (joint 3, 6, 9) — 轻微扭动
    motion[:, 3 * 6 + 2] = amp_mod * 0.2 * np.sin(2 * np.pi * beat_freq * 0.25 * t)
    motion[:, 6 * 6 + 1] = amp_mod * 0.15 * np.sin(2 * np.pi * beat_freq * 0.5 * t + 0.5)
    motion[:, 9 * 6 + 2] = amp_mod * 0.1 * np.sin(2 * np.pi * beat_freq * t + 1.0)

    # 头部 (joint 15) — 轻微点头
    motion[:, 15 * 6 + 2] = amp_mod * 0.08 * np.sin(2 * np.pi * beat_freq * t)

    # Root translation: 跟随节拍上下起伏 + 轻微前后
    motion[:, 144] = 0.01 * np.sin(2 * np.pi * beat_freq * 0.25 * t)  # x
    motion[:, 145] = 0.0  # y
    motion[:, 146] = 0.91 + 0.03 * onset_env * np.sin(2 * np.pi * beat_freq * t)  # z

    # Foot contact: 交替抬脚
    motion[:, 147] = (np.sin(2 * np.pi * beat_freq * 0.5 * t) > 0).astype(float)  # L heel
    motion[:, 148] = (np.sin(2 * np.pi * beat_freq * 0.5 * t) > 0).astype(float)  # L toe
    motion[:, 149] = (np.sin(2 * np.pi * beat_freq * 0.5 * t) < 0).astype(float)  # R heel
    motion[:, 150] = (np.sin(2 * np.pi * beat_freq * 0.5 * t) < 0).astype(float)  # R toe

    np.save(output_path, motion)
    print(f"  Placeholder motion: {motion.shape} -> {output_path}")
    print(f"    Tempo: {tempo:.1f} BPM, beat_freq: {beat_freq:.2f} Hz")
    return motion


def main():
    parser = argparse.ArgumentParser(description="Generate dance using Bailando (backup)")
    parser.add_argument("--input", type=str, default="../music")
    parser.add_argument("--output", type=str, default="../outputs/generated_motions")
    parser.add_argument("--placeholder", action="store_true",
                        help="Generate simple placeholder motions for pipeline testing")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wav_files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in {args.input}")
        return

    for wav_file in wav_files:
        name = wav_file.replace(".wav", "")
        print(f"\nProcessing: {name}")

        output_path = os.path.join(args.output, f"{name}_motion.npy")
        if args.placeholder:
            generate_placeholder_motion(
                os.path.join(args.input, wav_file), output_path
            )
        else:
            generate_with_bailando(
                os.path.join(args.input, wav_file), output_path
            )

    print(f"\nDone! Processed {len(wav_files)} tracks.")


if __name__ == "__main__":
    main()
