"""
生成评估图表 — 用于报告和答辩
包括：
1. BAS / PSR / JLVR 柱状对比图
2. COM 高度随时间变化曲线
3. 关节力矩分布图
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

SIM_DIR = "../outputs/sim_results"
EVAL_JSON = "../outputs/evaluation_results.json"
PLOT_DIR = "../outputs/plots"
GENRES = ["hiphop", "ballet", "house"]
FPS = 30


def plot_metrics_bar():
    """BAS / PSR / JLVR 三指标柱状图"""
    with open(EVAL_JSON) as f:
        results = json.load(f)

    genres = [r["genre"] for r in results]
    bas = [r.get("beat_alignment_score", 0) or 0 for r in results]
    psr = [r.get("physical_stability_rate", 0) or 0 for r in results]
    jlvr = [r.get("joint_limit_violation_rate", 0) or 0 for r in results]

    x = np.arange(len(genres))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, bas, width, label="BAS (higher=better)", color="#4CAF50")
    bars2 = ax.bar(x, psr, width, label="PSR (higher=better)", color="#2196F3")
    bars3 = ax.bar(x + width, jlvr, width, label="JLVR (lower=better)", color="#FF5722")

    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics by Genre")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in genres])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.savefig(os.path.join(PLOT_DIR, "metrics_comparison.png"))
    plt.savefig(os.path.join(PLOT_DIR, "metrics_comparison.pdf"))
    plt.close()
    print("  Saved: metrics_comparison.png/pdf")


def plot_com_height():
    """COM 高度随时间变化曲线 — 展示摔倒时刻"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, genre in enumerate(GENRES):
        ax = axes[i]

        # Full physics COM height
        com_path = os.path.join(SIM_DIR, f"{genre}_com_height.npy")
        if os.path.exists(com_path):
            com = np.load(com_path)
            t = np.arange(len(com)) / FPS
            ax.plot(t, com, color="#2196F3", linewidth=1.5, label="Full Physics")

        # Kinematic-root COM height
        kinroot_com_path = os.path.join(SIM_DIR, f"{genre}_kinroot_com_height.npy")
        if os.path.exists(kinroot_com_path):
            com_kin = np.load(kinroot_com_path)
            t_kin = np.arange(len(com_kin)) / FPS
            ax.plot(t_kin, com_kin, color="#4CAF50", linewidth=1.5, alpha=0.7, label="Kinematic Root")

        # Fall threshold
        ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Fall threshold")
        ax.set_title(f"{genre.capitalize()}")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("COM Height (m)")
        ax.set_ylim(0, 1.2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Center of Mass Height Over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "com_height.png"))
    plt.savefig(os.path.join(PLOT_DIR, "com_height.pdf"))
    plt.close()
    print("  Saved: com_height.png/pdf")


def plot_stability_comparison():
    """Full physics vs Kinematic root 稳定率对比"""
    with open(EVAL_JSON) as f:
        results = json.load(f)

    genres = [r["genre"] for r in results]
    psr_full = [r.get("physical_stability_rate", 0) or 0 for r in results]

    # Compute kinematic-root PSR
    psr_kinroot = []
    for genre in genres:
        kinroot_com = os.path.join(SIM_DIR, f"{genre}_kinroot_com_height.npy")
        if os.path.exists(kinroot_com):
            com = np.load(kinroot_com)
            psr_kinroot.append(float(np.mean(com > 0.3)))
        else:
            psr_kinroot.append(0)

    x = np.arange(len(genres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, psr_kinroot, width, label="Kinematic Root", color="#4CAF50")
    bars2 = ax.bar(x + width/2, psr_full, width, label="Full Physics", color="#FF5722")

    ax.set_ylabel("Physical Stability Rate")
    ax.set_title("Stability: Kinematic Root vs Full Physics")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in genres])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    plt.savefig(os.path.join(PLOT_DIR, "stability_comparison.png"))
    plt.savefig(os.path.join(PLOT_DIR, "stability_comparison.pdf"))
    plt.close()
    print("  Saved: stability_comparison.png/pdf")


def plot_joint_torques():
    """关节力矩分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, genre in enumerate(GENRES):
        ax = axes[i]
        torque_path = os.path.join(SIM_DIR, f"{genre}_joint_torques.npy")
        if os.path.exists(torque_path):
            torques = np.load(torque_path)
            # RMS torque per timestep
            rms = np.sqrt(np.mean(torques**2, axis=1))
            t = np.arange(len(rms)) / FPS
            ax.plot(t, rms, color="#9C27B0", linewidth=1, alpha=0.8)
            ax.fill_between(t, 0, rms, alpha=0.2, color="#9C27B0")
            ax.set_title(f"{genre.capitalize()}")
            ax.set_xlabel("Time (s)")
            if i == 0:
                ax.set_ylabel("RMS Joint Torque")
            ax.grid(alpha=0.3)

    fig.suptitle("Joint Torque Magnitude Over Time (Full Physics)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "joint_torques.png"))
    plt.savefig(os.path.join(PLOT_DIR, "joint_torques.pdf"))
    plt.close()
    print("  Saved: joint_torques.png/pdf")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Generating evaluation plots...")
    plot_metrics_bar()
    plot_com_height()
    plot_stability_comparison()
    plot_joint_torques()
    print(f"\nAll plots saved to {PLOT_DIR}/")
