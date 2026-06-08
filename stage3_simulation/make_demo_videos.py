"""
生成带音频的 demo 视频
包括：运动学回放视频 + 物理仿真视频，合并音乐轨道
"""
import os
import sys
import numpy as np
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_retarget"))

import mujoco
import cv2
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MODEL_PATH = "../outputs/smpl_humanoid.xml"
MOTION_DIR = "../outputs/retargeted_motions/strategy_a"
SIM_DIR = "../outputs/sim_results"
MUSIC_DIR = "../music"
VIDEO_DIR = "../outputs/videos"
GENRES = ["ballet", "hiphop", "house"]
FPS = 30
WIDTH, HEIGHT = 1280, 720


def render_trajectory(model_path, trajectory, width=WIDTH, height=HEIGHT, fps=FPS):
    """Render a qpos trajectory to a list of RGB frames."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    frames = []
    for t in range(len(trajectory)):
        nq = min(len(trajectory[t]), model.nq)
        data.qpos[:nq] = trajectory[t][:nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame.copy())
    renderer.close()
    return frames


def save_video_with_audio(frames, audio_path, output_path, fps=FPS):
    """Save video frames as mp4, then merge with audio using ffmpeg."""
    h, w = frames[0].shape[:2]

    # Step 1: Save silent video with opencv
    tmp_video = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Step 2: Merge with audio using ffmpeg
    if os.path.exists(audio_path):
        cmd = [
            FFMPEG, "-y",
            "-i", tmp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            os.remove(tmp_video)
            print(f"  Video+audio saved: {output_path}")
        else:
            # fallback: just rename silent video
            os.rename(tmp_video, output_path)
            print(f"  Video saved (no audio): {output_path}")
    else:
        os.rename(tmp_video, output_path)
        print(f"  Video saved (no audio): {output_path}")


def add_text_overlay(frame, text, position=(20, 40)):
    """Add text to frame using opencv."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame_bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    os.makedirs(VIDEO_DIR, exist_ok=True)

    for genre in GENRES:
        print(f"\n{'='*50}")
        print(f"Generating videos for: {genre}")
        print(f"{'='*50}")

        kin_qpos_path = os.path.join(MOTION_DIR, f"{genre}_qpos.npy")
        phys_qpos_path = os.path.join(SIM_DIR, f"{genre}_qpos.npy")
        kinroot_qpos_path = os.path.join(SIM_DIR, f"{genre}_kinroot_qpos.npy")
        audio_path = os.path.join(MUSIC_DIR, f"{genre}.wav")

        kin_trajectory = np.load(kin_qpos_path)

        # 1. Kinematic playback video
        print("  Rendering kinematic playback...")
        kin_frames = render_trajectory(MODEL_PATH, kin_trajectory)
        kin_frames_labeled = [add_text_overlay(f, f"{genre} - Kinematic") for f in kin_frames]
        save_video_with_audio(kin_frames_labeled, audio_path,
                              os.path.join(VIDEO_DIR, f"{genre}_kinematic.mp4"))

        # 2. Full physics video
        if os.path.exists(phys_qpos_path):
            print("  Rendering full physics...")
            phys_trajectory = np.load(phys_qpos_path)
            phys_frames = render_trajectory(MODEL_PATH, phys_trajectory)
            phys_frames_labeled = [add_text_overlay(f, f"{genre} - Full Physics") for f in phys_frames]
            save_video_with_audio(phys_frames_labeled, audio_path,
                                  os.path.join(VIDEO_DIR, f"{genre}_physics.mp4"))

        # 3. Side-by-side: kinematic-root vs full-physics
        if os.path.exists(kinroot_qpos_path) and os.path.exists(phys_qpos_path):
            print("  Rendering side-by-side comparison...")
            kinroot_trajectory = np.load(kinroot_qpos_path)
            kinroot_frames = render_trajectory(MODEL_PATH, kinroot_trajectory, width=640, height=720)
            phys_frames_half = render_trajectory(MODEL_PATH, phys_trajectory, width=640, height=720)

            T = min(len(kinroot_frames), len(phys_frames_half))
            combined_frames = []
            for i in range(T):
                left = add_text_overlay(kinroot_frames[i], "Kinematic Root", (10, 30))
                right = add_text_overlay(phys_frames_half[i], "Full Physics", (10, 30))
                combined = np.concatenate([left, right], axis=1)
                combined_frames.append(combined)

            save_video_with_audio(combined_frames, audio_path,
                                  os.path.join(VIDEO_DIR, f"{genre}_comparison.mp4"))

    print("\n\nAll videos generated!")
    for f in sorted(os.listdir(VIDEO_DIR)):
        if f.endswith(".mp4"):
            size = os.path.getsize(os.path.join(VIDEO_DIR, f)) / 1024
            print(f"  {f}: {size:.0f} KB")
