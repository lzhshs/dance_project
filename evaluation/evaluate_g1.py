"""
Evaluate Unitree G1 simulation results.
Computes BAS, PSR, JLVR for Strategy B (G1 robot).
"""
import os
import sys
import numpy as np
import json
import mujoco

def load_model_safe(xml_path):
    """Load MuJoCo model, handling Chinese path issues."""
    try:
        return mujoco.MjModel.from_xml_path(xml_path)
    except (ValueError, UnicodeDecodeError):
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_string = f.read()
        return mujoco.MjModel.from_xml_string(xml_string)

# Paths
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G1_MODEL = "C:/temp/g1/scene.xml"
G1_SIM_DIR = os.path.join(PROJECT, "outputs", "sim_results_g1")
OUTPUT_JSON = os.path.join(PROJECT, "outputs", "evaluation_g1.json")


def compute_bas(qpos_traj, model, fps=30):
    """
    Body Angular Speed — mean angular velocity across all hinge joints (rad/s).
    Computed via finite differences of joint angles.
    """
    dt = 1.0 / fps
    # Extract hinge joint angles only (skip freejoint qpos 0:7)
    hinge_angles = []
    for i in range(model.njnt):
        if model.jnt_type[i] == 3:  # hinge
            qaddr = model.jnt_qposadr[i]
            hinge_angles.append(qpos_traj[:, qaddr])

    if not hinge_angles:
        return 0.0

    hinge_angles = np.array(hinge_angles).T  # (K, n_hinge)
    # Angular velocity: finite difference
    angular_vel = np.diff(hinge_angles, axis=0) / dt  # (K-1, n_hinge)
    # Mean absolute angular speed across all joints and frames
    return float(np.mean(np.abs(angular_vel)))


def compute_psr(fell_flags):
    """Pose Success Rate"""
    stable = sum(1 for f in fell_flags if not f)
    return stable / len(fell_flags)


def compute_jlvr(qpos_traj, target_traj, model):
    """Joint Limit Violation Rate"""
    violations = 0
    total = 0
    
    for t in range(len(qpos_traj)):
        qpos = qpos_traj[t]
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                qaddr = model.jnt_qposadr[i]
                jtype = model.jnt_type[i]
                
                if jtype == 0:  # free joint
                    continue
                elif jtype == 3:  # hinge
                    q = qpos[qaddr]
                    lo, hi = model.jnt_range[i]
                    if q < lo or q > hi:
                        violations += 1
                    total += 1
    
    return violations / total if total > 0 else 0.0


def evaluate_genre(genre, model):
    """Evaluate one genre."""
    qpos_path = os.path.join(G1_SIM_DIR, f"{genre}_qpos.npy")
    fell_path = os.path.join(G1_SIM_DIR, f"{genre}_fell.npy")
    target_path = os.path.join(PROJECT, "outputs", "retargeted_motions", "strategy_b_g1", f"{genre}_g1_qpos.npy")
    
    if not os.path.exists(qpos_path):
        print(f"  SKIP: {qpos_path} not found")
        return None
    
    qpos_traj = np.load(qpos_path)
    fell_flags = np.load(fell_path)
    target_traj = np.load(target_path)
    
    print(f"\n  Evaluating {genre}...")
    bas = compute_bas(qpos_traj, model)
    psr = compute_psr(fell_flags)
    jlvr = compute_jlvr(qpos_traj, target_traj, model)
    
    print(f"    BAS:  {bas:.3f}")
    print(f"    PSR:  {psr:.3f}")
    print(f"    JLVR: {jlvr:.3f}")
    
    return {"BAS": bas, "PSR": psr, "JLVR": jlvr}


def main():
    print("=" * 60)
    print("  Unitree G1 Evaluation (Strategy B)")
    print("=" * 60)
    
    model = load_model_safe(G1_MODEL)
    print(f"  G1 model: nq={model.nq}, njnt={model.njnt}")
    
    results = {}
    for genre in ["ballet", "hiphop", "house"]:
        res = evaluate_genre(genre, model)
        if res:
            results[genre] = res
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Results saved: {OUTPUT_JSON}")
    print(f"{'='*60}")
    
    # Summary table
    print("\n  Summary:")
    print("  " + "-" * 50)
    print(f"  {'Genre':<10} {'BAS':>8} {'PSR':>8} {'JLVR':>8}")
    print("  " + "-" * 50)
    for genre, metrics in results.items():
        print(f"  {genre:<10} {metrics['BAS']:>8.3f} {metrics['PSR']:>8.3f} {metrics['JLVR']:>8.3f}")
    print("  " + "-" * 50)


if __name__ == "__main__":
    main()
