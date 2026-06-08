"""
EDGE 舞蹈生成
从 Jukebox 音乐特征生成 SMPL 格式的舞蹈动作

使用方法:
    conda activate edge
    python generate_dance.py --features ../outputs/jukebox_features --output ../outputs/generated_motions

输出格式: (K, 151) numpy array
    - [:, :144]   = 24 joints x 6D rotation
    - [:, 144:147] = root translation (x, y, z)
    - [:, 147:151] = foot contact labels (4)
"""
import argparse
import os
import sys
import numpy as np
import torch

# EDGE 路径
EDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "EDGE")


def load_edge_model(checkpoint_path, device="cuda"):
    """加载 EDGE 预训练模型"""
    sys.path.insert(0, EDGE_DIR)
    try:
        from test_edge import EDGE
    except ImportError:
        # 不同版本的 EDGE 导入路径可能不同
        from args import parse_test_opt
        from EDGE import EDGE

    opt = parse_test_opt()
    opt.feature_type = "jukebox"
    model = EDGE(opt)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def stitch_clips(clips, overlap_frames):
    """
    线性插值拼接重叠区域
    clips: list of (T, D) arrays
    overlap_frames: 重叠帧数
    """
    if len(clips) == 1:
        return clips[0]

    result = clips[0]
    for i in range(1, len(clips)):
        if overlap_frames > 0 and overlap_frames < len(result):
            weights = np.linspace(1, 0, overlap_frames)[:, None]
            overlap_region = result[-overlap_frames:] * weights + clips[i][:overlap_frames] * (1 - weights)
            result = np.concatenate([result[:-overlap_frames], overlap_region, clips[i][overlap_frames:]], axis=0)
        else:
            result = np.concatenate([result, clips[i]], axis=0)
    return result


def generate_dance_from_features(music_features, model, device="cuda",
                                  clip_length=150, overlap=30):
    """
    从 Jukebox 特征生成舞蹈
    music_features: (T, 4800) Jukebox 特征
    返回: (K, 151) SMPL 动作序列
    """
    total_frames = len(music_features)
    clips = []

    # 分段生成 5 秒片段
    start = 0
    while start < total_frames:
        end = min(start + clip_length, total_frames)

        # 如果剩余帧不足一个完整片段，用最后 clip_length 帧
        if end - start < clip_length and total_frames >= clip_length:
            start = total_frames - clip_length
            end = total_frames

        clip_feat = music_features[start:end]

        # padding 如果不足 clip_length
        if len(clip_feat) < clip_length:
            pad = np.zeros((clip_length - len(clip_feat), clip_feat.shape[1]))
            clip_feat = np.concatenate([clip_feat, pad], axis=0)

        clip_feat_tensor = torch.from_numpy(clip_feat).unsqueeze(0).float().to(device)

        with torch.no_grad():
            motion = model.generate(clip_feat_tensor)  # (1, T, 151)

        clip_motion = motion.squeeze(0).cpu().numpy()

        # 如果做了 padding，去掉 padding 部分
        actual_len = end - start
        clips.append(clip_motion[:actual_len])

        if end >= total_frames:
            break
        start += clip_length - overlap

    # 拼接
    full_motion = stitch_clips(clips, overlap)
    return full_motion


def generate_dance_simple(music_feature_path, output_path, checkpoint_path, device="cuda"):
    """简化版：单个文件生成"""
    music_features = np.load(music_feature_path)
    print(f"  Music features: {music_features.shape}")

    model = load_edge_model(checkpoint_path, device)
    motion = generate_dance_from_features(music_features, model, device)

    np.save(output_path, motion)
    print(f"  Generated motion: {motion.shape} -> {output_path}")
    return motion


def main():
    parser = argparse.ArgumentParser(description="Generate dance from music features using EDGE")
    parser.add_argument("--features", type=str, default="../outputs/jukebox_features")
    parser.add_argument("--output", type=str, default="../outputs/generated_motions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="EDGE checkpoint path (default: external/EDGE/checkpoints/train-6000.pt)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(EDGE_DIR, "checkpoints", "train-6000.pt")

    os.makedirs(args.output, exist_ok=True)

    feat_files = [f for f in os.listdir(args.features) if f.endswith(".npy")]
    if not feat_files:
        print(f"No .npy feature files found in {args.features}")
        return

    for feat_file in feat_files:
        name = feat_file.replace(".npy", "")
        print(f"\nGenerating dance for: {name}")
        generate_dance_simple(
            os.path.join(args.features, feat_file),
            os.path.join(args.output, f"{name}_motion.npy"),
            args.checkpoint,
            args.device,
        )

    print(f"\nDone! Generated dances for {len(feat_files)} tracks.")


if __name__ == "__main__":
    main()
