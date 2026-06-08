"""Test: generate model and verify it loads with load_model_safe"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage2_retarget'))
sys.path.insert(0, os.path.dirname(__file__))
from retarget_smplsim import generate_minimal_humanoid
from mujoco_utils import load_model_safe

model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'smpl_humanoid.xml')
generate_minimal_humanoid(model_path)

m = load_model_safe(model_path)
print(f'Model loaded: nq={m.nq}, nv={m.nv}, nu={m.nu}, njnt={m.njnt}')
print(f'Total: {m.nu} position actuators for {m.njnt} joints')

# Count ball vs hinge
ball_count = sum(1 for i in range(m.njnt) if m.jnt_type[i] == 1)
hinge_count = sum(1 for i in range(m.njnt) if m.jnt_type[i] == 3)
print(f'Joints: {ball_count} ball + {hinge_count} hinge + 1 free = {m.njnt}')
print(f'Expected actuators: {ball_count*3} (ball) + {hinge_count} (hinge) = {ball_count*3 + hinge_count}')
print('OK!')
