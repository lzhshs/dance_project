"""Inspect current motion data and model structure."""
import numpy as np, mujoco, os
PROJECT = os.path.dirname(os.path.abspath(__file__))

for name in ['ballet', 'hiphop', 'house']:
    m = np.load(os.path.join(PROJECT, f'outputs/generated_motions/{name}_motion.npy'))
    print(f'{name}: shape={m.shape}')
    trans = m[:, 144:147]
    print(f'  root_trans z: min={trans[:,2].min():.3f} max={trans[:,2].max():.3f} mean={trans[:,2].mean():.3f}')

print()
qpos = np.load(os.path.join(PROJECT, 'outputs/retargeted_motions/strategy_a/ballet_qpos.npy'))
print(f'qpos shape: {qpos.shape}')
print(f'Frame 0 root pos:  {qpos[0, 0:3]}')
print(f'Frame 0 root quat: {qpos[0, 3:7]}')
print(f'Root z range: {qpos[:, 2].min():.3f} ~ {qpos[:, 2].max():.3f}')

print()
model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT, 'outputs/smpl_humanoid.xml'))
print(f'Model: nq={model.nq}, nv={model.nv}, nu={model.nu}')
for i in range(model.njnt):
    name = model.joint(i).name
    jtype = ['free','ball','slide','hinge'][model.jnt_type[i]]
    nq = [7,4,1,1][model.jnt_type[i]]
    print(f'  [{i:2d}] {name:15s} {jtype:6s} nq={nq} qposadr={model.jnt_qposadr[i]}')
print()
print('Actuators:')
for i in range(model.nu):
    name = model.actuator(i).name
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = model.joint(jnt_id).name
    print(f'  [{i:2d}] {name:15s} -> {jnt_name}')
