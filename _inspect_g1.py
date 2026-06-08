"""Inspect Unitree G1 model structure."""
import mujoco

model = mujoco.MjModel.from_xml_path('C:/temp/g1/scene.xml')

print(f'G1 Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}, nbody={model.nbody}')
print()
print('Joints:')
for i in range(model.njnt):
    name = model.joint(i).name
    jtype = ['free','ball','slide','hinge'][model.jnt_type[i]]
    qaddr = model.jnt_qposadr[i]
    daddr = model.jnt_dofadr[i]
    lo, hi = model.jnt_range[i]
    print(f'  [{i:2d}] {name:35s} {jtype:6s} qpos={qaddr:2d} dof={daddr:2d} range=[{lo:.3f}, {hi:.3f}]')

print()
print('Actuators:')
for i in range(model.nu):
    name = model.actuator(i).name
    jid = model.actuator_trnid[i, 0]
    jname = model.joint(jid).name
    print(f'  [{i:2d}] {name:35s} -> {jname}')
