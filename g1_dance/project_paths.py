"""Project-local paths used by the reproducibility scripts."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EDGE_DIR = PROJECT_ROOT / "EDGE"
GENERATED_MOTIONS_DIR = ARTIFACTS_DIR / "generated_motions"
OPTIMIZED_MOTIONS_DIR = ARTIFACTS_DIR / "optimized_motions"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
CACHED_FEATURES_DIR = ARTIFACTS_DIR / "cached_features"
INPUTS_DIR = ARTIFACTS_DIR / "inputs"
POP_WAV = INPUTS_DIR / "pop.wav"

AIST_DATA_DIR = ARTIFACTS_DIR / "aist_data"
AIST_MOTIONS_DIR = AIST_DATA_DIR / "motions"
AIST_MUSIC_DIR = AIST_DATA_DIR / "music"

G1_XML = str(PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml")
