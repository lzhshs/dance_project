"""
Strategy A: SMPL-to-SMPL 重定向
将 EDGE 输出的 SMPL 动作 转换为 SMPLSim MuJoCo 人形体的关节轨迹

使用方法:
    conda activate sim
    python retarget_smplsim.py --motion ../outputs/generated_motions/ballet_motion.npy \
                                --output ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy
"""
import argparse
import os
import sys
import json
import numpy as np
from utils import (
    parse_edge_motion,
    rot6d_to_quat_wxyz,
    rot6d_to_axis_angle,
    axis_angle_to_quat_wxyz,
    quat_multiply,
    quat_inverse,
    normalize_quat,
    scale_motion,
    SMPL_JOINT_NAMES,
)

# SMPLSim 路径
SMPLSIM_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "SMPLSim")

# 中文路径安全加载
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import load_model_safe


def generate_smplsim_model(output_xml_path, gender="neutral", beta=None):
    """
    用 SMPLSim 生成 MuJoCo XML 人形体模型

    Args:
        output_xml_path: 输出的 XML 路径
        gender: "neutral" / "male" / "female"
        beta: SMPL shape 参数 (10,), None 则用默认体型
    """
    sys.path.insert(0, SMPLSIM_DIR)

    try:
        from smpl_sim.smpllib.smpl_mujoco import SMPL_M_Renderer
        from smpl_sim.smpllib.smpl_parser import SMPL_Parser

        if beta is None:
            beta = np.zeros(10)

        # 生成 MuJoCo 模型
        parser = SMPL_Parser(model_path="../body_models/smpl")
        parser.export_mujoco_xml(
            output_path=output_xml_path,
            gender=gender,
            beta=beta,
        )
        print(f"Generated MuJoCo model -> {output_xml_path}")

    except ImportError:
        print("SMPLSim not available. Generating minimal MuJoCo humanoid model...")
        generate_minimal_humanoid(output_xml_path)


def generate_minimal_humanoid(output_xml_path):
    """
    生成一个最小化的 MuJoCo 人形体 XML（不依赖 SMPLSim）
    使用 position actuator（内置PD控制），每个 ball joint 3个轴分别控制。
    """
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="smpl_humanoid">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <joint damping="5" armature="0.05"/>
    <geom type="capsule" condim="3" friction="1 0.5 0.5" margin="0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="10 10 0.01" rgba="0.8 0.8 0.8 1"/>

    <!-- Pelvis (root) -->
    <body name="pelvis" pos="0 0 0.91">
      <freejoint name="root"/>
      <geom name="pelvis_geom" fromto="-.1 0 0 .1 0 0" size="0.08" rgba="0.3 0.5 0.8 1"/>

      <!-- Spine -->
      <body name="spine1" pos="0 0 0.12">
        <joint name="spine1" type="ball" range="0 0.52"/>
        <geom name="spine1_geom" fromto="0 0 0 0 0 0.1" size="0.07" rgba="0.3 0.5 0.8 1"/>

        <body name="spine2" pos="0 0 0.1">
          <joint name="spine2" type="ball" range="0 0.35"/>
          <geom name="spine2_geom" fromto="0 0 0 0 0 0.1" size="0.065" rgba="0.3 0.5 0.8 1"/>

          <body name="spine3" pos="0 0 0.1">
            <joint name="spine3" type="ball" range="0 0.35"/>
            <geom name="spine3_geom" fromto="-.08 0 0 .08 0 0" size="0.06" rgba="0.3 0.5 0.8 1"/>

            <!-- Neck + Head -->
            <body name="neck" pos="0 0 0.06">
              <joint name="neck" type="ball" range="0 0.70"/>
              <geom name="neck_geom" fromto="0 0 0 0 0 0.08" size="0.04" rgba="0.3 0.5 0.8 1"/>
              <body name="head" pos="0 0 0.08">
                <joint name="head" type="ball" range="0 0.52"/>
                <geom name="head_geom" type="sphere" size="0.1" rgba="0.9 0.7 0.6 1"/>
              </body>
            </body>

            <!-- Left arm -->
            <body name="l_collar" pos="0.08 0 0.04">
              <joint name="l_collar" type="ball" range="0 0.35"/>
              <geom name="l_collar_geom" fromto="0 0 0 0.08 0 0" size="0.035" rgba="0.3 0.5 0.8 1"/>
              <body name="l_shoulder" pos="0.08 0 0">
                <joint name="l_shoulder" type="ball" range="0 2.80"/>
                <geom name="l_upper_arm" fromto="0 0 0 0.25 0 0" size="0.035" rgba="0.3 0.6 0.9 1"/>
                <body name="l_elbow" pos="0.25 0 0">
                  <joint name="l_elbow" type="hinge" axis="0 1 0" range="-2.5 0"/>
                  <geom name="l_lower_arm" fromto="0 0 0 0.22 0 0" size="0.03" rgba="0.3 0.6 0.9 1"/>
                  <body name="l_wrist" pos="0.22 0 0">
                    <joint name="l_wrist" type="ball" range="0 1.22"/>
                    <geom name="l_hand" type="sphere" size="0.04" rgba="0.9 0.7 0.6 1"/>
                  </body>
                </body>
              </body>
            </body>

            <!-- Right arm -->
            <body name="r_collar" pos="-0.08 0 0.04">
              <joint name="r_collar" type="ball" range="0 0.35"/>
              <geom name="r_collar_geom" fromto="0 0 0 -0.08 0 0" size="0.035" rgba="0.3 0.5 0.8 1"/>
              <body name="r_shoulder" pos="-0.08 0 0">
                <joint name="r_shoulder" type="ball" range="0 2.80"/>
                <geom name="r_upper_arm" fromto="0 0 0 -0.25 0 0" size="0.035" rgba="0.3 0.6 0.9 1"/>
                <body name="r_elbow" pos="-0.25 0 0">
                  <joint name="r_elbow" type="hinge" axis="0 1 0" range="-2.5 0"/>
                  <geom name="r_lower_arm" fromto="0 0 0 -0.22 0 0" size="0.03" rgba="0.3 0.6 0.9 1"/>
                  <body name="r_wrist" pos="-0.22 0 0">
                    <joint name="r_wrist" type="ball" range="0 1.22"/>
                    <geom name="r_hand" type="sphere" size="0.04" rgba="0.9 0.7 0.6 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- Left leg -->
      <body name="l_hip" pos="0.08 0 -0.05">
        <joint name="l_hip" type="ball" range="0 2.09"/>
        <geom name="l_upper_leg" fromto="0 0 0 0 0 -0.4" size="0.05" rgba="0.3 0.5 0.8 1"/>
        <body name="l_knee" pos="0 0 -0.4">
          <joint name="l_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
          <geom name="l_lower_leg" fromto="0 0 0 0 0 -0.38" size="0.04" rgba="0.3 0.6 0.9 1"/>
          <body name="l_ankle" pos="0 0 -0.38">
            <joint name="l_ankle" type="ball" range="0 0.87"/>
            <geom name="l_foot_geom" fromto="0 -0.06 0 0 0.1 0" size="0.035" rgba="0.9 0.7 0.6 1"/>
            <body name="l_foot" pos="0 0.04 0">
              <joint name="l_foot" type="ball" range="0 0.52"/>
              <geom name="l_toe" fromto="0 0 0 0 0.06 0" size="0.025" rgba="0.9 0.7 0.6 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- Right leg -->
      <body name="r_hip" pos="-0.08 0 -0.05">
        <joint name="r_hip" type="ball" range="0 2.09"/>
        <geom name="r_upper_leg" fromto="0 0 0 0 0 -0.4" size="0.05" rgba="0.3 0.5 0.8 1"/>
        <body name="r_knee" pos="0 0 -0.4">
          <joint name="r_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
          <geom name="r_lower_leg" fromto="0 0 0 0 0 -0.38" size="0.04" rgba="0.3 0.6 0.9 1"/>
          <body name="r_ankle" pos="0 0 -0.38">
            <joint name="r_ankle" type="ball" range="0 0.87"/>
            <geom name="r_foot_geom" fromto="0 -0.06 0 0 0.1 0" size="0.035" rgba="0.9 0.7 0.6 1"/>
            <body name="r_foot" pos="0 0.04 0">
              <joint name="r_foot" type="ball" range="0 0.52"/>
              <geom name="r_toe" fromto="0 0 0 0 0.06 0" size="0.025" rgba="0.9 0.7 0.6 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- position actuators: built-in PD control (kp sets stiffness, kd via joint damping)
       Ball joints need 3 actuators (one per DOF axis). ctrl = target axis-angle component.
       Hinge joints need 1 actuator. ctrl = target angle. -->
  <actuator>
    <!-- Spine (ball x3 each) -->
    <position name="spine1_x" joint="spine1" gear="1 0 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine1_y" joint="spine1" gear="0 1 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine1_z" joint="spine1" gear="0 0 1" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine2_x" joint="spine2" gear="1 0 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine2_y" joint="spine2" gear="0 1 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine2_z" joint="spine2" gear="0 0 1" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine3_x" joint="spine3" gear="1 0 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine3_y" joint="spine3" gear="0 1 0" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <position name="spine3_z" joint="spine3" gear="0 0 1" kp="300" kv="30" ctrlrange="-1.5 1.5"/>
    <!-- Neck + Head (ball x3 each) -->
    <position name="neck_x" joint="neck" gear="1 0 0" kp="150" kv="15" ctrlrange="-2 2"/>
    <position name="neck_y" joint="neck" gear="0 1 0" kp="150" kv="15" ctrlrange="-2 2"/>
    <position name="neck_z" joint="neck" gear="0 0 1" kp="150" kv="15" ctrlrange="-2 2"/>
    <position name="head_x" joint="head" gear="1 0 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="head_y" joint="head" gear="0 1 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="head_z" joint="head" gear="0 0 1" kp="100" kv="10" ctrlrange="-2 2"/>
    <!-- Left arm -->
    <position name="l_collar_x" joint="l_collar" gear="1 0 0" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="l_collar_y" joint="l_collar" gear="0 1 0" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="l_collar_z" joint="l_collar" gear="0 0 1" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="l_shoulder_x" joint="l_shoulder" gear="1 0 0" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="l_shoulder_y" joint="l_shoulder" gear="0 1 0" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="l_shoulder_z" joint="l_shoulder" gear="0 0 1" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="l_elbow" joint="l_elbow" kp="150" kv="15" ctrlrange="-2.5 0"/>
    <position name="l_wrist_x" joint="l_wrist" gear="1 0 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="l_wrist_y" joint="l_wrist" gear="0 1 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="l_wrist_z" joint="l_wrist" gear="0 0 1" kp="100" kv="10" ctrlrange="-2 2"/>
    <!-- Right arm -->
    <position name="r_collar_x" joint="r_collar" gear="1 0 0" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="r_collar_y" joint="r_collar" gear="0 1 0" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="r_collar_z" joint="r_collar" gear="0 0 1" kp="150" kv="15" ctrlrange="-1.5 1.5"/>
    <position name="r_shoulder_x" joint="r_shoulder" gear="1 0 0" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="r_shoulder_y" joint="r_shoulder" gear="0 1 0" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="r_shoulder_z" joint="r_shoulder" gear="0 0 1" kp="200" kv="20" ctrlrange="-3.5 3.5"/>
    <position name="r_elbow" joint="r_elbow" kp="150" kv="15" ctrlrange="-2.5 0"/>
    <position name="r_wrist_x" joint="r_wrist" gear="1 0 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="r_wrist_y" joint="r_wrist" gear="0 1 0" kp="100" kv="10" ctrlrange="-2 2"/>
    <position name="r_wrist_z" joint="r_wrist" gear="0 0 1" kp="100" kv="10" ctrlrange="-2 2"/>
    <!-- Left leg -->
    <position name="l_hip_x" joint="l_hip" gear="1 0 0" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="l_hip_y" joint="l_hip" gear="0 1 0" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="l_hip_z" joint="l_hip" gear="0 0 1" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="l_knee" joint="l_knee" kp="400" kv="40" ctrlrange="0 2.5"/>
    <position name="l_ankle_x" joint="l_ankle" gear="1 0 0" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="l_ankle_y" joint="l_ankle" gear="0 1 0" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="l_ankle_z" joint="l_ankle" gear="0 0 1" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="l_foot_x" joint="l_foot" gear="1 0 0" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
    <position name="l_foot_y" joint="l_foot" gear="0 1 0" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
    <position name="l_foot_z" joint="l_foot" gear="0 0 1" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
    <!-- Right leg -->
    <position name="r_hip_x" joint="r_hip" gear="1 0 0" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="r_hip_y" joint="r_hip" gear="0 1 0" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="r_hip_z" joint="r_hip" gear="0 0 1" kp="500" kv="50" ctrlrange="-3.5 3.5"/>
    <position name="r_knee" joint="r_knee" kp="400" kv="40" ctrlrange="0 2.5"/>
    <position name="r_ankle_x" joint="r_ankle" gear="1 0 0" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="r_ankle_y" joint="r_ankle" gear="0 1 0" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="r_ankle_z" joint="r_ankle" gear="0 0 1" kp="200" kv="20" ctrlrange="-2 2"/>
    <position name="r_foot_x" joint="r_foot" gear="1 0 0" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
    <position name="r_foot_y" joint="r_foot" gear="0 1 0" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
    <position name="r_foot_z" joint="r_foot" gear="0 0 1" kp="100" kv="10" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""
    os.makedirs(os.path.dirname(output_xml_path) or ".", exist_ok=True)
    with open(output_xml_path, "w") as f:
        f.write(xml_content)
    print(f"Generated minimal humanoid model -> {output_xml_path}")


def build_joint_mapping(model_xml_path):
    """
    构建 SMPL 关节名 -> MuJoCo joint 索引 的映射

    Returns:
        mapping: dict {smpl_joint_index: mujoco_joint_name}
        joint_info: dict {joint_name: {"qpos_addr": int, "type": str}}
    """
    import mujoco
    model = load_model_safe(model_xml_path)

    mujoco_joint_names = []
    joint_info = {}
    for i in range(model.njnt):
        name = model.joint(i).name
        mujoco_joint_names.append(name)
        jnt_type = model.jnt_type[i]  # 0=free, 1=ball, 2=slide, 3=hinge
        joint_info[name] = {
            "qpos_addr": model.jnt_qposadr[i],
            "dof_addr": model.jnt_dofadr[i],
            "type": ["free", "ball", "slide", "hinge"][jnt_type],
            "nq": [7, 4, 1, 1][jnt_type],  # qpos 维度
            "nv": [6, 3, 1, 1][jnt_type],  # qvel 维度
        }

    # 建立映射（SMPL joint name -> MuJoCo joint name）
    mapping = {}
    for smpl_idx, smpl_name in enumerate(SMPL_JOINT_NAMES):
        if smpl_name in mujoco_joint_names:
            mapping[smpl_idx] = smpl_name
        elif smpl_name == "pelvis":
            mapping[smpl_idx] = "root"  # root freejoint

    return mapping, joint_info


def convert_motion_to_qpos(motion_path, model_xml_path, output_path, motion_scale=1.0):
    """
    核心函数: EDGE SMPL 动作 -> MuJoCo qpos 轨迹

    Args:
        motion_path: EDGE 输出的 .npy (K, 151)
        model_xml_path: MuJoCo XML 模型路径
        output_path: 输出 qpos 轨迹 .npy
        motion_scale: 动作缩放系数 (< 1.0 减小幅度)
    """
    import mujoco

    # 1. 解析 EDGE 输出
    motion = np.load(motion_path)
    joint_rot6d, root_trans, foot_contact = parse_edge_motion(motion)
    K = joint_rot6d.shape[0]

    # 可选: 缩放动作幅度
    if motion_scale != 1.0:
        joint_rot6d, root_trans = scale_motion(joint_rot6d, root_trans, motion_scale)

    # 2. 转换旋转表示: 6D -> quaternion (w,x,y,z)
    joint_quats = rot6d_to_quat_wxyz(joint_rot6d)  # (K, 24, 4)

    # 3. 加载 MuJoCo 模型
    model = load_model_safe(model_xml_path)
    nq = model.nq

    mapping, joint_info = build_joint_mapping(model_xml_path)

    # 4. 组装 qpos 轨迹
    qpos_trajectory = np.zeros((K, nq))

    for t in range(K):
        # Root: translation + orientation
        qpos_trajectory[t, 0:3] = root_trans[t]           # root translation
        qpos_trajectory[t, 3:7] = joint_quats[t, 0]       # root quaternion (pelvis)

        # Body joints
        for smpl_idx, mj_name in mapping.items():
            if smpl_idx == 0:  # pelvis -> root, 已处理
                continue
            if mj_name not in joint_info:
                continue

            info = joint_info[mj_name]
            addr = info["qpos_addr"]

            if info["type"] == "ball":
                # Ball joint: 4 个 qpos (quaternion)
                qpos_trajectory[t, addr:addr+4] = joint_quats[t, smpl_idx]
            elif info["type"] == "hinge":
                # Hinge joint: 1 个 qpos (angle)
                # 从 axis-angle 提取绕指定轴的角度
                aa = rot6d_to_axis_angle(joint_rot6d[t, smpl_idx])
                angle = np.linalg.norm(aa)
                qpos_trajectory[t, addr] = angle

    # 5. Clamp to joint limits — enforce anatomical constraints
    clamp_count = 0
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jnt_type = model.jnt_type[j]
        q_addr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]

        if jnt_type == 3:  # hinge
            before = qpos_trajectory[:, q_addr].copy()
            qpos_trajectory[:, q_addr] = np.clip(qpos_trajectory[:, q_addr], lo, hi)
            clamp_count += int(np.any(before != qpos_trajectory[:, q_addr]))
        elif jnt_type == 1:  # ball — range means max rotation angle
            for t in range(K):
                q = qpos_trajectory[t, q_addr:q_addr+4]
                # compute rotation angle from quaternion
                w = np.clip(q[0], -1.0, 1.0)
                angle = 2.0 * np.arccos(abs(w))
                if angle > hi:
                    # scale down the rotation to max allowed angle
                    scale = hi / angle
                    # slerp towards identity: q_new = slerp(identity, q, scale)
                    axis = q[1:4]
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-8:
                        axis = axis / axis_norm
                        half_new = hi / 2.0
                        q_new = np.array([np.cos(half_new),
                                          axis[0] * np.sin(half_new),
                                          axis[1] * np.sin(half_new),
                                          axis[2] * np.sin(half_new)])
                        qpos_trajectory[t, q_addr:q_addr+4] = q_new
                        clamp_count += 1

    if clamp_count > 0:
        print(f"  Joint limit clamping: {clamp_count} corrections applied")

    np.save(output_path, qpos_trajectory)
    print(f"Converted: {motion_path} -> {output_path}")
    print(f"  Frames: {K}, qpos dim: {nq}")
    return qpos_trajectory


def main():
    parser = argparse.ArgumentParser(description="Strategy A: SMPL to SMPLSim retargeting")
    parser.add_argument("--motion", type=str, help="Single motion .npy file")
    parser.add_argument("--motion_dir", type=str, default="../outputs/generated_motions")
    parser.add_argument("--model", type=str, default="../outputs/smpl_humanoid.xml")
    parser.add_argument("--output_dir", type=str, default="../outputs/retargeted_motions/strategy_a")
    parser.add_argument("--generate_model", action="store_true", help="Generate MuJoCo humanoid model first")
    parser.add_argument("--scale", type=float, default=1.0, help="Motion amplitude scale (< 1.0 to reduce)")
    args = parser.parse_args()

    # 可选: 先生成 MuJoCo 模型
    if args.generate_model:
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        generate_smplsim_model(args.model)
        # 如果 SMPLSim 失败，会自动 fallback 到 minimal humanoid
        if not os.path.exists(args.model):
            generate_minimal_humanoid(args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.motion:
        # 处理单个文件
        name = os.path.basename(args.motion).replace("_motion.npy", "")
        convert_motion_to_qpos(
            args.motion, args.model,
            os.path.join(args.output_dir, f"{name}_qpos.npy"),
            args.scale,
        )
    else:
        # 处理目录下所有文件
        motion_files = [f for f in os.listdir(args.motion_dir) if f.endswith("_motion.npy")]
        if not motion_files:
            print(f"No motion files found in {args.motion_dir}")
            return

        for f in motion_files:
            name = f.replace("_motion.npy", "")
            convert_motion_to_qpos(
                os.path.join(args.motion_dir, f),
                args.model,
                os.path.join(args.output_dir, f"{name}_qpos.npy"),
                args.scale,
            )
        print(f"\nDone! Retargeted {len(motion_files)} motions.")


if __name__ == "__main__":
    main()
