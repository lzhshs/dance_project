"""Mux an AIST++ music file onto a rendered MuJoCo video.

Usage: python -m g1_dance.add_audio <video.mp4> [music.wav]
If music.wav is omitted, parses music_id (e.g. mJB5) from the filename and
looks up the corresponding wav in artifacts/aist_data/music/. Output: <video>_audio.mp4
next to the input.
"""
import os
import re
import subprocess
import sys

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
from g1_dance.project_paths import AIST_MUSIC_DIR

MUSIC_DIR = str(AIST_MUSIC_DIR)


def main():
    video = sys.argv[1]
    if len(sys.argv) >= 3:
        wav = sys.argv[2]
    else:
        base = os.path.basename(video)
        m = re.search(r"_m([A-Z]{2}\d)_", base)
        if not m:
            sys.exit(f"Could not parse music ID from filename: {base}")
        music_id = "m" + m.group(1)
        wav = os.path.join(MUSIC_DIR, f"{music_id}.wav")
    if not os.path.exists(wav):
        sys.exit(f"Missing music file: {wav}")

    stem, ext = os.path.splitext(video)
    out = stem + "_audio.mp4"
    cmd = [
        FFMPEG, "-y",
        "-i", video,
        "-i", wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",  # stop at shortest stream (= video duration)
        out,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
