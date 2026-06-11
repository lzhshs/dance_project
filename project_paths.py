"""Project-local paths used by the reproducibility scripts."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

EDGE_DIR = PROJECT_ROOT / "EDGE"
GENERATED_MOTIONS_DIR = PROJECT_ROOT / "generated_motions"
OPTIMIZED_MOTIONS_DIR = PROJECT_ROOT / "optimized_motions"
VIDEOS_DIR = PROJECT_ROOT / "videos"

AIST_DATA_DIR = PROJECT_ROOT / "aist_data"
AIST_MOTIONS_DIR = AIST_DATA_DIR / "motions"
AIST_MUSIC_DIR = AIST_DATA_DIR / "music"

G1_XML = str(PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml")
