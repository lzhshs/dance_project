"""Final evaluation utility for the music-to-G1 dance pipeline.

Default output:
  reports/final/final_metrics.csv
  reports/final/figures/final_stability_curve.{pdf,png}

The script evaluates the core final-project comparison:
  1. Direct retargeting from EDGE SMPL motion to G1.
  2. Offline-filtered reference.
  3. Conservative balance-safe reference.
  4. Robot-feasibility optimized reference.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from g1_dance.evaluate import beat_alignment_score, joint_limit_violation_rate
from g1_dance.motion_constraints import actuator_qpos_addresses
from g1_dance.project_paths import GENERATED_MOTIONS_DIR, INPUTS_DIR, OPTIMIZED_MOTIONS_DIR, PROJECT_ROOT
from g1_dance.retarget_smpl_to_g1 import G1_XML, load_motion
from g1_dance.retarget_v2 import build_qpos_trajectory
from g1_dance.smpl_fk import fk


FALL_PELVIS_Z = 0.35
DEFAULT_METHODS = [
    ("direct_retarget", "Direct retarget", "pkl", str(GENERATED_MOTIONS_DIR / "pop.pkl")),
    ("offline_filtered", "Offline-filtered", "npz", str(OPTIMIZED_MOTIONS_DIR / "pop_optimized.npz")),
    ("balance_safe", "Balance-safe", "npz", str(OPTIMIZED_MOTIONS_DIR / "pop_balance_safe.npz")),
    ("feasibility_optimized", "Feasibility-optimized", "npz", str(OPTIMIZED_MOTIONS_DIR / "pop_feasible.npz")),
    ("support_com", "Support/COM-aware", "npz", str(OPTIMIZED_MOTIONS_DIR / "pop_support_com.npz")),
]


@dataclass
class EvalResult:
    method_id: str
    method_name: str
    source_path: str
    fps: float
    frames: int
    duration_s: float
    stability_rate: float
    fall_time_s: Optional[float]
    mean_joint_error_rad: float
    max_joint_error_rad: float
    pelvis_z_min_m: float
    pelvis_z_mean_m: float
    motion_amplitude_rad: float
    actual_joint_limit_violation_rate: float
    boundary_joint_limit_proxy_rate: float
    beat_alignment_score: Optional[float]
    music_beats: Optional[int]
    motion_beats: Optional[int]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final G1 dance pipeline outputs")
    parser.add_argument("--motion", default=str(GENERATED_MOTIONS_DIR / "pop.pkl"), help="Source EDGE/SMPL motion .pkl")
    parser.add_argument("--wav", default=str(INPUTS_DIR / "artifacts/inputs/pop.wav"), help="Music wav for beat alignment")
    parser.add_argument("--fps", type=float, default=30.0, help="Motion/reference FPS")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "reports" / "final"))
    parser.add_argument("--skip-beat", action="store_true", help="Skip librosa beat-alignment metric")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(G1_XML)
    methods = build_methods(args.motion)

    beat_info = None
    joint_proxy = None
    poses = trans = None
    if Path(args.motion).exists():
        poses, trans = load_motion(args.motion)
        joint_proxy = joint_limit_violation_rate(model, poses, trans)
        if not args.skip_beat and args.wav and Path(args.wav).exists():
            positions = fk(poses, trans)
            beat_info = beat_alignment_score(positions, args.fps, args.wav)

    results: list[EvalResult] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray, Optional[float]]] = {}

    for method_id, method_name, source_type, source_path in methods:
        if not Path(source_path).exists():
            print(f"Skipping missing input: {source_path}")
            continue
        qpos_ref, fps = load_qpos_reference(model, source_type, source_path, args.fps, poses, trans)
        stats = simulate_reference(model, qpos_ref, fps, args.max_seconds)
        actual_violation, boundary_proxy = joint_limit_rates_from_qpos(model, qpos_ref)
        motion_amp = motion_amplitude(model, qpos_ref)

        bas = n_music = n_motion = None
        boundary_proxy_for_row = boundary_proxy
        if source_type == "pkl" and beat_info is not None:
            bas, n_music, n_motion = beat_info
        if source_type == "pkl" and joint_proxy is not None:
            boundary_proxy_for_row = float(joint_proxy[0])

        results.append(EvalResult(
            method_id=method_id,
            method_name=method_name,
            source_path=source_path,
            fps=fps,
            frames=int(qpos_ref.shape[0]),
            duration_s=float(qpos_ref.shape[0] / fps),
            stability_rate=float(stats["stability_rate"]),
            fall_time_s=stats["fall_time_s"],
            mean_joint_error_rad=float(stats["mean_joint_error_rad"]),
            max_joint_error_rad=float(stats["max_joint_error_rad"]),
            pelvis_z_min_m=float(stats["pelvis_z_min_m"]),
            pelvis_z_mean_m=float(stats["pelvis_z_mean_m"]),
            motion_amplitude_rad=motion_amp,
            actual_joint_limit_violation_rate=actual_violation,
            boundary_joint_limit_proxy_rate=boundary_proxy_for_row,
            beat_alignment_score=bas,
            music_beats=n_music,
            motion_beats=n_motion,
        ))
        curves[method_name] = (stats["time_s"], stats["pelvis_z_m"], stats["fall_time_s"])
        print_result(results[-1])

    csv_path = outdir / "final_metrics.csv"
    write_csv(csv_path, results)
    plot_path = figures / "final_stability_curve"
    write_stability_plot(plot_path, curves)
    write_markdown_summary(outdir / "final_metrics_summary.md", results, plot_path)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}.pdf")
    print(f"Wrote {outdir / 'final_metrics_summary.md'}")


def build_methods(motion_path: str) -> list[tuple[str, str, str, str]]:
    methods = []
    for method_id, method_name, source_type, source_path in DEFAULT_METHODS:
        if source_type == "pkl":
            source_path = motion_path
        methods.append((method_id, method_name, source_type, source_path))
    return methods


def load_qpos_reference(model, source_type: str, source_path: str, fps: float,
                        poses=None, trans=None) -> tuple[np.ndarray, float]:
    if source_type == "npz":
        data = np.load(source_path, allow_pickle=True)
        ref_fps = float(data["fps"][0]) if "fps" in data else fps
        return np.asarray(data["qpos_ref"], dtype=np.float64), ref_fps
    if poses is None or trans is None:
        poses, trans = load_motion(source_path)
    data = mujoco.MjData(model)
    return build_qpos_trajectory(model, data, poses, trans).astype(np.float64), fps


def simulate_reference(model, qpos_ref: np.ndarray, fps: float, max_seconds: float) -> dict:
    data = mujoco.MjData(model)
    ctrl_to_qadr = actuator_qpos_addresses(model)
    duration = min(qpos_ref.shape[0] / fps, max_seconds)
    dt = float(model.opt.timestep)
    n_steps = int(duration / dt)

    data.qpos[:] = qpos_ref[0]
    data.qpos[2] += 0.02
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    joint_err_sum = np.zeros(model.nu, dtype=np.float64)
    joint_err_count = 0
    time_log = []
    pelvis_log = []
    fall_time = None

    for step in range(n_steps):
        t_sec = step * dt
        q_ref = interp_qpos(qpos_ref, t_sec, fps)
        data.ctrl[:] = q_ref[ctrl_to_qadr]
        mujoco.mj_step(model, data)

        err = np.abs(data.qpos[ctrl_to_qadr] - q_ref[ctrl_to_qadr])
        joint_err_sum += err
        joint_err_count += 1
        pelvis_z = float(data.qpos[2])
        if step % 10 == 0:
            time_log.append(t_sec)
            pelvis_log.append(pelvis_z)
        if fall_time is None and pelvis_z < FALL_PELVIS_Z:
            fall_time = t_sec

    mean_err = joint_err_sum / max(joint_err_count, 1)
    pelvis_arr = np.asarray(pelvis_log, dtype=np.float64)
    return {
        "stability_rate": 1.0 if fall_time is None else fall_time / max(duration, 1e-9),
        "fall_time_s": fall_time,
        "mean_joint_error_rad": float(mean_err.mean()),
        "max_joint_error_rad": float(mean_err.max()),
        "pelvis_z_min_m": float(pelvis_arr.min()),
        "pelvis_z_mean_m": float(pelvis_arr.mean()),
        "time_s": np.asarray(time_log, dtype=np.float64),
        "pelvis_z_m": pelvis_arr,
    }


def interp_qpos(qpos_ref: np.ndarray, t_sec: float, fps: float) -> np.ndarray:
    idx_f = t_sec * fps
    i0 = min(int(np.floor(idx_f)), qpos_ref.shape[0] - 1)
    i1 = min(i0 + 1, qpos_ref.shape[0] - 1)
    alpha = idx_f - i0
    return (1.0 - alpha) * qpos_ref[i0] + alpha * qpos_ref[i1]


def joint_limit_rates_from_qpos(model, qpos_ref: np.ndarray) -> tuple[float, float]:
    violations = 0
    boundary_hits = 0
    total = 0
    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low >= high:
            continue
        address = int(model.jnt_qposadr[joint_id])
        values = qpos_ref[:, address]
        violations += int(np.sum((values < low) | (values > high)))
        boundary_hits += int(np.sum(np.isclose(values, low, atol=1e-4) | np.isclose(values, high, atol=1e-4)))
        total += len(values)
    return violations / max(total, 1), boundary_hits / max(total, 1)


def motion_amplitude(model, qpos_ref: np.ndarray) -> float:
    actuated = actuator_qpos_addresses(model)
    return float(np.mean(np.std(qpos_ref[:, actuated], axis=0)))


def write_csv(path: Path, results: list[EvalResult]) -> None:
    fields = list(EvalResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_stability_plot(path_stem: Path, curves: dict[str, tuple[np.ndarray, np.ndarray, Optional[float]]]) -> None:
    colors = {
        "Direct retarget": "#9467bd",
        "Offline-filtered": "#d62728",
        "Balance-safe": "#1f77b4",
        "Feasibility-optimized": "#2ca02c",
        "Support/COM-aware": "#ff7f0e",
    }
    plt.figure(figsize=(6.4, 3.2))
    for label, (times, heights, fall_time) in curves.items():
        plt.plot(times, heights, label=label, linewidth=1.6, color=colors.get(label))
        if fall_time is not None:
            plt.axvline(fall_time, color=colors.get(label), linestyle=":", linewidth=1.1, alpha=0.75)
    plt.axhline(FALL_PELVIS_Z, color="black", linestyle="--", linewidth=1.0, label="Fall threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Pelvis height (m)")
    plt.xlim(0, 10)
    plt.ylim(0, 0.95)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path_stem.with_suffix(".pdf"))
    plt.savefig(path_stem.with_suffix(".png"), dpi=220)
    plt.close()


def write_markdown_summary(path: Path, results: list[EvalResult], plot_path: Path) -> None:
    lines = [
        "# Final Evaluation Summary",
        "",
        f"Stability curve: `{plot_path.with_suffix('.pdf')}`",
        "",
        "| Method | Stability | Fall time | Mean err | Pelvis min | Motion amp | Beat align | Joint-limit proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        fall = "none" if row.fall_time_s is None else f"{row.fall_time_s:.2f}s"
        beat = "--" if row.beat_alignment_score is None else f"{row.beat_alignment_score:.4f}"
        lines.append(
            f"| {row.method_name} | {row.stability_rate*100:.1f}% | {fall} | "
            f"{row.mean_joint_error_rad:.4f} | {row.pelvis_z_min_m:.3f} | "
            f"{row.motion_amplitude_rad:.4f} | {beat} | "
            f"{row.boundary_joint_limit_proxy_rate*100:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n")


def print_result(row: EvalResult) -> None:
    fall = "none" if row.fall_time_s is None else f"{row.fall_time_s:.2f}s"
    beat = "n/a" if row.beat_alignment_score is None else f"{row.beat_alignment_score:.4f}"
    print(
        f"{row.method_name:24s} stability={row.stability_rate*100:5.1f}% "
        f"fall={fall:>6s} err={row.mean_joint_error_rad:.4f} "
        f"amp={row.motion_amplitude_rad:.4f} beat={beat}"
    )


if __name__ == "__main__":
    main()
