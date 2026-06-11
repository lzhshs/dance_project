"""Measure how often each G1 joint would be clamped during retargeting.

For every frame and every hinge joint, record the raw (pre-clamp) angle and
compare against jnt_range. Reports: violation rate per joint + total.
"""
import sys
import numpy as np
import mujoco
from retarget_smpl_to_g1 import G1_XML, load_motion, smpl_to_g1_qpos

pkl = sys.argv[1]
model = mujoco.MjModel.from_xml_path(G1_XML)

# Patch smpl_to_g1_qpos by calling it without the clamp step.
# Easiest: re-import the function logic, but it's simpler to just compute
# the clamped version and compare against joint ranges by re-running without clamp.
# For now we do something cheaper: run it and check how many values hit exactly
# the limits (proxy for "was clamped"). Works because non-clamped angles rarely
# land exactly on the limit.
poses, trans = load_motion(pkl)
qpos = smpl_to_g1_qpos(model, poses, trans)
T = qpos.shape[0]

total_frames = T
total_viol = 0
per_joint = []
for i in range(1, model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = model.jnt_range[i]
    a = int(model.jnt_qposadr[i])
    col = qpos[:, a]
    eps = 1e-4
    at_low  = np.sum(col <= lo + eps)
    at_high = np.sum(col >= hi - eps)
    rate = (at_low + at_high) / total_frames
    per_joint.append((name, rate, at_low, at_high, lo, hi))
    total_viol += (at_low + at_high)

per_joint.sort(key=lambda r: -r[1])
print(f"Joint-limit saturation rate (T={total_frames} frames):")
print(f"{'joint':30s} {'rate':>6s}  {'low':>6s} {'high':>6s}  range")
for name, rate, lo_n, hi_n, lo, hi in per_joint:
    print(f"{name:30s} {rate*100:5.1f}%  {lo_n:6d} {hi_n:6d}  [{lo:+.2f}, {hi:+.2f}]")

print(f"\nMean saturation across hinge joints: "
      f"{sum(r[1] for r in per_joint)/len(per_joint)*100:.1f}%")
