"""
位置控制器 + 仿真运行器
使用 MuJoCo position actuators（内置PD控制），将 qpos 目标直接转为控制信号。

Ball joint 的 3 个 actuator (gear x/y/z) 分别对应 axis-angle 的 3 个分量。
Hinge joint 的 1 个 actuator 对应关节角度。

默认使用 kinematic_root=True: root 由轨迹驱动，身体关节由物理控制，
这样完全绕开了平衡控制问题，可以展示真实的动作细节。

使用方法:
    conda activate sim
    python pd_controller.py --model ../outputs/smpl_humanoid.xml \
                            --motion ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy \
                            --render
"""
import argparse
import os
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed. Run: pip install mujoco")
    exit(1)

from scipy.spatial.transform import Rotation

# 中文路径安全加载
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import load_model_safe


def quat_wxyz_to_axis_angle(quat_wxyz):
    """
    MuJoCo quaternion (w, x, y, z) -> axis-angle (3,)
    """
    # scipy uses (x, y, z, w)
    q = np.asarray(quat_wxyz, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    quat_xyzw = q[[1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_rotvec()  # (3,)


def build_ctrl_target(model, target_qpos):
    """
    Convert target qpos -> ctrl array for position actuators.

    For each actuator:
      - ball joint + gear=[1,0,0]: ctrl = axis-angle[0]
      - ball joint + gear=[0,1,0]: ctrl = axis-angle[1]
      - ball joint + gear=[0,0,1]: ctrl = axis-angle[2]
      - hinge joint:              ctrl = target angle
      - free joint:               skip (no actuator should target freejoint)
    """
    nu = model.nu
    ctrl = np.zeros(nu)

    for i in range(nu):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_type = model.jnt_type[jnt_id]  # 0=free, 1=ball, 2=slide, 3=hinge
        qpos_addr = model.jnt_qposadr[jnt_id]
        gear = model.actuator_gear[i]  # (6,), first 3 are spatial

        if jnt_type == 1:  # ball joint -> qpos has 4 values (w,x,y,z)
            q = target_qpos[qpos_addr:qpos_addr + 4]
            aa = quat_wxyz_to_axis_angle(q)  # (3,)
            # project onto gear axis (first 3 components)
            axis = gear[:3]
            ctrl[i] = float(np.dot(aa, axis))

        elif jnt_type == 3:  # hinge joint -> qpos has 1 value (angle)
            ctrl[i] = float(target_qpos[qpos_addr])

        # free joint (type 0) and slide (type 2): no action

    return ctrl


class SimulationRunner:
    """仿真运行器 — 使用 position actuator 内置PD控制"""

    def __init__(self, model_path):
        self.model = load_model_safe(model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.fps = 30
        self.steps_per_frame = max(1, int(1.0 / (self.fps * self.dt)))

    def run_kinematic(self, qpos_trajectory, render=False):
        """
        运动学回放（直接设 qpos，不走物理仿真）
        用于验证重定向是否正确，生成 ground-truth 参考视频
        """
        results = {
            "qpos": [],
            "com_height": [],
        }

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        for t in range(len(qpos_trajectory)):
            nq = min(len(qpos_trajectory[t]), self.model.nq)
            self.data.qpos[:nq] = qpos_trajectory[t][:nq]
            mujoco.mj_forward(self.model, self.data)

            results["qpos"].append(self.data.qpos.copy())
            results["com_height"].append(self.data.subtree_com[0][2])

            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                import time
                time.sleep(1.0 / self.fps)

        if viewer is not None:
            viewer.close()

        return {k: np.array(v) for k, v in results.items()}

    def run_physics(self, qpos_trajectory, render=False, fall_threshold=0.3,
                    kinematic_root=True):
        """
        物理仿真（position actuator 跟踪）

        Args:
            kinematic_root: True（默认）= root 由轨迹驱动，身体关节走物理
                            False = 完全物理（需要平衡控制，较难稳定）

        Returns:
            results: dict with qpos, com_height, fell, joint_torques
        """
        results = {
            "qpos": [],
            "com_height": [],
            "fell": [],
            "joint_torques": [],
            "contact_forces": [],
        }

        # 重置仿真状态
        mujoco.mj_resetData(self.model, self.data)

        # 初始化姿态
        nq = min(len(qpos_trajectory[0]), self.model.nq)
        self.data.qpos[:nq] = qpos_trajectory[0][:nq]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        fell_frame = -1
        for t in range(len(qpos_trajectory)):
            target_qpos = np.zeros(self.model.nq)
            nq = min(len(qpos_trajectory[t]), self.model.nq)
            target_qpos[:nq] = qpos_trajectory[t][:nq]

            if kinematic_root:
                # 运动学驱动 root（直接赋值，跳过物理积分中的自由度）
                self.data.qpos[:3] = target_qpos[:3]
                self.data.qpos[3:7] = target_qpos[3:7]
                self.data.qvel[:6] = 0  # 清零 root 速度，防止漂移

            # 计算控制信号: position actuator -> ctrl = target DOF value
            ctrl = build_ctrl_target(self.model, target_qpos)
            self.data.ctrl[:] = ctrl

            # 推进物理仿真
            for _ in range(self.steps_per_frame):
                if kinematic_root:
                    # 每个子步都重新固定 root（防止物理引擎把它推走）
                    self.data.qpos[:3] = target_qpos[:3]
                    self.data.qpos[3:7] = target_qpos[3:7]
                    self.data.qvel[:6] = 0
                    self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)

            # 记录
            com_height = self.data.subtree_com[0][2]
            fell = com_height < fall_threshold

            results["qpos"].append(self.data.qpos.copy())
            results["com_height"].append(com_height)
            results["fell"].append(fell)
            results["joint_torques"].append(self.data.qfrc_actuator.copy())
            results["contact_forces"].append(
                np.array([self.data.cfrc_ext[i][2] for i in range(self.model.nbody)])
            )

            if fell and fell_frame < 0:
                fell_frame = t
                print(f"  [Frame {t}/{len(qpos_trajectory)}] Robot fell! COM height: {com_height:.3f}m")

            if render and viewer is not None and viewer.is_running():
                viewer.sync()

        if viewer is not None:
            viewer.close()

        # 统计
        total = len(qpos_trajectory)
        stable = sum(1 for f in results["fell"] if not f)
        mode_str = "kinematic-root" if kinematic_root else "full-physics"
        print(f"  [{mode_str}] Stability: {stable}/{total} frames ({100*stable/total:.1f}%)")
        if fell_frame >= 0:
            print(f"  First fall at frame {fell_frame} ({fell_frame/self.fps:.1f}s)")

        return {k: np.array(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Position actuator simulation")
    parser.add_argument("--model", type=str, required=True, help="MuJoCo XML model")
    parser.add_argument("--motion", type=str, required=True, help="qpos trajectory .npy")
    parser.add_argument("--output", type=str, default="../outputs/sim_results/")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--kinematic", action="store_true", help="Kinematic playback only (no physics)")
    parser.add_argument("--full_physics", action="store_true", help="Full physics (no kinematic root)")
    args = parser.parse_args()

    # 加载轨迹
    trajectory = np.load(args.motion)
    print(f"Loaded trajectory: {trajectory.shape}")

    # 初始化仿真
    runner = SimulationRunner(args.model)
    print(f"Model: nq={runner.model.nq}, nv={runner.model.nv}, nu={runner.model.nu}")
    print(f"Sim dt={runner.dt}, steps_per_frame={runner.steps_per_frame}")

    # 运行
    if args.kinematic:
        print("\n--- Kinematic Playback ---")
        results = runner.run_kinematic(trajectory, render=args.render)
    else:
        kinematic_root = not args.full_physics
        mode = "full-physics" if args.full_physics else "kinematic-root"
        print(f"\n--- Physics Simulation ({mode}) ---")
        results = runner.run_physics(trajectory, render=args.render,
                                     kinematic_root=kinematic_root)

    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    name = os.path.basename(args.motion).replace("_qpos.npy", "").replace(".npy", "")
    mode_tag = "kinematic" if args.kinematic else ("physics" if args.full_physics else "kinroot")
    for key, val in results.items():
        out_path = os.path.join(args.output, f"{name}_{mode_tag}_{key}.npy")
        np.save(out_path, val)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
