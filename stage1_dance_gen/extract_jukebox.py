"""
Jukebox 特征提取
在云 GPU 上运行（需要 >= 16GB VRAM）
推荐: AutoDL / Colab Pro / 实验室服务器

使用方法:
    conda activate edge
    python extract_jukebox.py --input ../music --output ../outputs/jukebox_features
"""
import argparse
import os
import numpy as np

try:
    import jukemirlib
except ImportError:
    print("ERROR: jukemirlib not installed. Run: pip install jukemirlib")
    print("NOTE: Jukebox requires >= 16GB VRAM. Use cloud GPU if needed.")
    exit(1)


def extract_features(wav_path, output_path):
    """
    提取 Jukebox 音乐特征
    输出: (T, 4800) numpy array, 30fps
    """
    print(f"Extracting features from: {wav_path}")
    reps = jukemirlib.extract(
        fpath=wav_path,
        layers=[36, 37, 38],
        downsample_target_rate=30,
    )
    np.save(output_path, reps)
    print(f"  Shape: {reps.shape} -> {output_path}")
    return reps


def main():
    parser = argparse.ArgumentParser(description="Extract Jukebox features from music")
    parser.add_argument("--input", type=str, default="../music", help="Directory of .wav files")
    parser.add_argument("--output", type=str, default="../outputs/jukebox_features")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wav_files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in {args.input}")
        return

    for wav_file in wav_files:
        name = wav_file.replace(".wav", "")
        extract_features(
            os.path.join(args.input, wav_file),
            os.path.join(args.output, f"{name}.npy"),
        )

    print(f"\nDone! Extracted features for {len(wav_files)} files.")


if __name__ == "__main__":
    main()
