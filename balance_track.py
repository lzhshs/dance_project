"""Online balance tracking for optimized G1 dance references.

This stage keeps the floating base free and adds a simple feedback controller on
ankles, hips, and waist. It is intentionally conservative: the controller only
modifies position-actuator targets, so it remains within the current G1 MuJoCo
actuator model instead of using fake root pinning.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_constraints import actuator_qpos_addresses, joint_addresses
from retarget_smpl_to_g1 import G1_XML


from project_paths import VIDEOS_DIR

OUT_DIR = str(VIDEOS_DIR)
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
PELVIS_BODY = "pelvis"
FALL_PELVIS_Z = 0.35
RENDER_FPS = 30


@dataclass(frozen=True)
class BalanceParams:
    name: str
    pitch_sign: float
    roll_sign: float
    kp_tilt: float
    kd_tilt: float
    kp_com: float
    ankle_ratio: float
    hip_ratio: float
    waist_ratio: float


def main():
    parser = argparse.ArgumentParser(description="Track optimized G1 motion with online balance feedback")
    parser.add_argument("ref", help="Optimized .npz reference")
    parser.add_argument("--render", action="store_true", help="Render the selected/best run")
    parser.add_argument("--candidate", help="Candidate name to render; default searches and renders best")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    args = parser.parse_args()

    ref = np.load(args.ref, allow_pickle=True)
    qpos_ref = np.asarray(ref["qpos_ref"], dtype=np.float64)
    fps = float(ref["fps"][0]) if "fps" in ref else 30.0
    model = mujoco.MjModel.from_xml_path(G1_XML)

    candidates = candidate_params()
    if args.candidate:
        candidates = [p for p in candidates if p.name == args.candidate]
        if not candidates:
            raise SystemExit(f"Unknown candidate: {args.candidate}")

    print(f"Testing {len(candidates)} balance candidates ...")
    results = []
    for params in candidates:
        stats = run_balance(model, qpos_ref, fps, params, args.max_seconds, render_path=None)
        candidate_score = score(stats)
        results.append((candidate_score, params, stats))
        print(f"{params.name:20s} score={candidate_score:7.3f} stability={stats['stability_rate']*100:5.1f}% "
              f"fall={stats['fall_time']} err={stats['mean_joint_error']:.3f} zmin={stats['pelvis_z_min']:.2f}")

    best_score, best_params, best_stats = min(results, key=lambda item: item[0])
    print(f"\nSelected: {best_params.name} score={best_score:.3f}")
    print(json.dumps(best_stats, indent=2, ensure_ascii=False))

    if args.render:
        stem = os.path.splitext(os.path.basename(args.ref))[0]
        out = os.path.join(OUT_DIR, f"{stem}_balanced.mp4")
        stats = run_balance(model, qpos_ref, fps, best_params, args.max_seconds, render_path=out)
        print(f"Rendered {out}")
        print(json.dumps(stats, indent=2, ensure_ascii=False))


def candidate_params() -> list[BalanceParams]:
    candidates = []
    ratio_profiles = [
        ("ankle", 0.85, 0.25, 0.12),
        ("mixed", 0.65, 0.35, 0.20),
        ("hip", 0.45, 0.55, 0.25),
    ]
    for pitch_sign in (-1.0, 1.0):
        for roll_sign in (-1.0, 1.0):
            for scale in (0.7, 1.2, 1.8, 2.5, 3.4):
                for profile_name, ankle_ratio, hip_ratio, waist_ratio in ratio_profiles:
                    candidates.append(BalanceParams(
                        name=f"ps{int(pitch_sign):+d}_rs{int(roll_sign):+d}_g{scale:.1f}_{profile_name}",
                        pitch_sign=pitch_sign,
                        roll_sign=roll_sign,
                        kp_tilt=0.55 * scale,
                        kd_tilt=0.08 * scale,
                        kp_com=0.75 * scale,
                        ankle_ratio=ankle_ratio,
                        hip_ratio=hip_ratio,
                        waist_ratio=waist_ratio,
                    ))
    return candidates


def run_balance(model, qpos_ref, fps, params, max_seconds, render_path=None):
    data = mujoco.MjData(model)
    adr = joint_addresses(model)
    ctrl_to_qadr = actuator_qpos_addresses(model)
    body_ids = {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FOOT_BODY),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FOOT_BODY),
        "pelvis": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY),
    }

    data.qpos[:] = qpos_ref[0]
    data.qpos[2] += 0.03
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    if render_path:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = 4.0
        cam.azimuth = 135
        cam.elevation = -15
    else:
        cam = None

    dt = model.opt.timestep
    duration = min(qpos_ref.shape[0] / fps, max_seconds)
    n_steps = int(duration / dt)
    render_every = max(1, int(round(1.0 / (RENDER_FPS * dt))))
    err_sum = 0.0
    err_count = 0
    pelvis_z_min = float("inf")
    fall_time = None

    for step in range(n_steps):
        t_sec = step * dt
        q_ref = interp_qpos(qpos_ref, t_sec, fps)
        ctrl = q_ref[ctrl_to_qadr].copy()
        apply_balance_feedback(model, data, adr, ctrl_to_qadr, ctrl, q_ref, params, body_ids)
        data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)

        err_sum += float(np.abs(data.qpos[ctrl_to_qadr] - q_ref[ctrl_to_qadr]).mean())
        err_count += 1
        pelvis_z = float(data.xpos[body_ids["pelvis"], 2])
        pelvis_z_min = min(pelvis_z_min, pelvis_z)
        if fall_time is None and pelvis_z < FALL_PELVIS_Z:
            fall_time = t_sec
            if render_path is None:
                break

        if renderer is not None and step % render_every == 0:
            cam.lookat[:] = data.qpos[0:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    if render_path:
        imageio.mimwrite(render_path, frames, fps=RENDER_FPS, quality=8)

    stability_rate = 1.0 if fall_time is None else fall_time / max(duration, 1e-9)
    return {
        "duration": duration,
        "fall_time": fall_time,
        "stability_rate": stability_rate,
        "mean_joint_error": err_sum / max(err_count, 1),
        "pelvis_z_min": pelvis_z_min,
        "candidate": params.name,
    }


def apply_balance_feedback(model, data, adr, ctrl_to_qadr, ctrl, q_ref, params, body_ids):
    quat_wxyz = data.qpos[3:7]
    roll, pitch, _ = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)
    angvel = data.qvel[3:6]
    left = data.xpos[body_ids["left"]]
    right = data.xpos[body_ids["right"]]
    zmin = min(left[2], right[2])
    wl = np.exp(-35.0 * max(0.0, left[2] - zmin))
    wr = np.exp(-35.0 * max(0.0, right[2] - zmin))
    support = (wl * left[:2] + wr * right[:2]) / max(wl + wr, 1e-9)
    com = data.subtree_com[1, :2]
    com_error = com - support

    pitch_corr = params.pitch_sign * (
        params.kp_tilt * pitch + params.kd_tilt * angvel[1] + params.kp_com * com_error[0]
    )
    roll_corr = params.roll_sign * (
        params.kp_tilt * roll + params.kd_tilt * angvel[0] + params.kp_com * com_error[1]
    )
    pitch_corr = float(np.clip(pitch_corr, -0.34, 0.34))
    roll_corr = float(np.clip(roll_corr, -0.24, 0.24))

    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "left_ankle_pitch_joint", params.ankle_ratio * pitch_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "right_ankle_pitch_joint", params.ankle_ratio * pitch_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "left_hip_pitch_joint", -params.hip_ratio * pitch_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "right_hip_pitch_joint", -params.hip_ratio * pitch_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "waist_pitch_joint", -params.waist_ratio * pitch_corr)

    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "left_ankle_roll_joint", params.ankle_ratio * roll_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "right_ankle_roll_joint", params.ankle_ratio * roll_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "left_hip_roll_joint", -params.hip_ratio * roll_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "right_hip_roll_joint", -params.hip_ratio * roll_corr)
    add_ctrl(model, adr, ctrl_to_qadr, ctrl, "waist_roll_joint", -params.waist_ratio * roll_corr)


def add_ctrl(model, adr, ctrl_to_qadr, ctrl, joint_name, delta):
    qadr = adr[joint_name]
    actuator = int(np.where(ctrl_to_qadr == qadr)[0][0])
    ctrl[actuator] += delta


def interp_qpos(qpos_ref, t_sec, fps):
    idx_f = t_sec * fps
    i0 = min(int(np.floor(idx_f)), qpos_ref.shape[0] - 1)
    i1 = min(i0 + 1, qpos_ref.shape[0] - 1)
    alpha = idx_f - i0
    return (1.0 - alpha) * qpos_ref[i0] + alpha * qpos_ref[i1]


def score(stats):
    fall_penalty = 100.0 * (1.0 - float(stats["stability_rate"]))
    error_penalty = 4.0 * float(stats["mean_joint_error"])
    height_penalty = max(0.0, 0.55 - float(stats["pelvis_z_min"])) * 8.0
    return fall_penalty + error_penalty + height_penalty


if __name__ == "__main__":
    main()
