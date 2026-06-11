"""Render a single debug frame with an explicit camera to verify the robot placement."""
import pickle, numpy as np, mujoco, imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from retarget_smpl_to_g1 import load_motion, smpl_to_g1_qpos, G1_XML, SMPL_PKL

model = mujoco.MjModel.from_xml_path(G1_XML)
data  = mujoco.MjData(model)

poses, trans = load_motion(SMPL_PKL)
qpos_traj = smpl_to_g1_qpos(model, poses, trans)

frame_idx = 0
data.qpos[:] = qpos_traj[frame_idx]
mujoco.mj_forward(model, data)

print(f"Frame {frame_idx} root pos: {data.qpos[0:3]}")
print(f"Frame {frame_idx} root quat: {data.qpos[3:7]}")
print(f"Pelvis height (z): {data.qpos[2]:.3f}")

# Configure an explicit free camera that looks at the robot.
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat[:] = data.qpos[0:3]   # track robot
cam.distance = 3.5
cam.azimuth = 135
cam.elevation = -15

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera=cam)
img = renderer.render()
imageio.imwrite("/Users/lucy_lzh/dance_project/debug_frame.png", img)
print("Saved debug_frame.png")
