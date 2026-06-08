"""Test full physics vs kinematic-root physics comparison"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_retarget"))
from pd_controller import SimulationRunner

MODEL = "../outputs/smpl_humanoid.xml"
MOTION_DIR = "../outputs/retargeted_motions/strategy_a"
OUTPUT_DIR = "../outputs/sim_results"
GENRES = ["ballet", "hiphop", "house"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for genre in GENRES:
    trajectory = np.load(os.path.join(MOTION_DIR, f"{genre}_qpos.npy"))
    print(f"\n{'='*60}")
    print(f"{genre}: {trajectory.shape}")
    print(f"{'='*60}")

    # Full physics (no root cheating)
    runner = SimulationRunner(MODEL, kp=1000.0, kd=100.0)
    print("\n  [A] Full physics (root free):")
    results_full = runner.run_physics(trajectory, render=False, kinematic_root=False)

    # Kinematic root (for comparison)
    runner2 = SimulationRunner(MODEL, kp=1000.0, kd=100.0)
    print("\n  [B] Kinematic root (comparison):")
    results_kin = runner2.run_physics(trajectory, render=False, kinematic_root=True)

    # Save full physics results (these are the "real" ones)
    for key, val in results_full.items():
        np.save(os.path.join(OUTPUT_DIR, f"{genre}_{key}.npy"), val)

    # Save kinematic-root results separately for comparison
    for key, val in results_kin.items():
        np.save(os.path.join(OUTPUT_DIR, f"{genre}_kinroot_{key}.npy"), val)

    print(f"\n  COM height (full):    min={results_full['com_height'].min():.3f}, mean={results_full['com_height'].mean():.3f}")
    print(f"  COM height (kinroot): min={results_kin['com_height'].min():.3f}, mean={results_kin['com_height'].mean():.3f}")

print("\n\nDone! Results saved to", OUTPUT_DIR)
