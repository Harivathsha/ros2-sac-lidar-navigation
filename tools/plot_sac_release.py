#!/usr/bin/env python3
"""Plot recorded SAC outcomes, trajectories and training logs; no simulator needed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np
import pandas as pd

MODEL_SHA = "422afbcf3ef7dd47149cf0bf3415022a2b0704dc5eae0496931c3e535d2fe27b"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify_evaluation(folder, model=None):
    """Check all 100 cases against the summary and every recorded transition."""
    folder = Path(folder)
    summary = json.loads((folder / "evaluation_summary.json").read_text())
    policy = json.loads((folder / "tested_policy.json").read_text())
    with (folder / "evaluation_episodes.csv").open(newline="") as f:
        episodes = list(csv.DictReader(f))
    with (folder / "evaluation_trajectories.csv").open(newline="") as f:
        trajectories = list(csv.DictReader(f))
    layouts = json.loads((folder / "evaluation_layouts.json").read_text())
    expected_seeds = list(range(360000, 360100))
    expected = {"complete": True, "requested_episodes": 100, "completed_episodes": 100,
                "stage": 4, "suite": "standard", "deterministic": True,
                "checkpoint_steps": 540000, "training_budget_steps": 600000,
                "selected_checkpoint": "best", "initial_heading_policy": "face_goal_v1",
                "success": 91, "collision": 9, "timeout": 0}
    for k, value in expected.items():
        if summary.get(k) != value:
            raise ValueError(f"Unexpected evaluation {k}: {summary.get(k)!r}")
    if (summary.get("seeds") != expected_seeds or len(episodes) != 100 or len(layouts) != 100
            or [int(r["seed"]) for r in episodes] != expected_seeds
            or [int(r["episode"]) for r in episodes] != list(range(1, 101))):
        raise ValueError("The evaluation must contain the exact complete 100-case seed bank.")
    if policy.get("sha256") != MODEL_SHA or policy.get("steps") != 540000:
        raise ValueError("The tested-policy identity differs from the verified 91/100 model.")
    if model is not None and sha(model) != MODEL_SHA:
        raise ValueError("Model checksum differs from the model tested in this evaluation.")
    counts = Counter(row["outcome"] for row in episodes)
    if counts != Counter(success=91, collision=9):
        raise ValueError("Episode outcomes disagree with the verified summary.")
    grouped = defaultdict(list)
    for row in trajectories:
        grouped[int(row["episode"])].append(row)
    for ep, layout in zip(episodes, layouts):
        number, seed = int(ep["episode"]), int(ep["seed"])
        rows = grouped[number]
        steps = int(ep["steps"])
        if (len(rows) != steps or [int(r["step"]) for r in rows] != list(range(1, steps + 1))
                or any(int(r["seed"]) != seed for r in rows)
                or rows[-1]["outcome"] != ep["outcome"]
                or any(r["outcome"] != "running" for r in rows[:-1])):
            raise ValueError(f"Incomplete or inconsistent trajectory for episode {number}.")
        if (int(layout["seed"]) != seed or layout.get("initial_heading_policy") != "face_goal_v1"
                or int(ep["boxes"]) != len(layout["object_layout"])
                or not 12 <= int(ep["boxes"]) <= 15 or ep["task_kind"] != "standard"):
            raise ValueError(f"Layout metadata differs in episode {number}.")
        x, y, yaw = layout["requested_start"]
        gx, gy = layout["goal"]
        angle_error = math.atan2(math.sin(yaw - math.atan2(gy-y, gx-x)),
                                 math.cos(yaw - math.atan2(gy-y, gx-x)))
        if abs(angle_error) > 1e-8:
            raise ValueError("A requested spawn heading does not face its goal.")
        for row in rows:
            values = [float(row[k]) for k in ("robot_x", "robot_y", "robot_yaw",
                "nearest_obstacle_m", "distance_to_goal_m", "commanded_linear_velocity_mps",
                "commanded_angular_velocity_radps")]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite recorded observation or command.")
            if abs(values[-2] - .22) > 1e-6 or abs(values[-1]) > 1.20001:
                raise ValueError("Recorded action differs from the fixed-speed steering contract.")
        if abs(min(float(r["nearest_obstacle_m"]) for r in rows) - float(ep["minimum_clearance_m"])) > 1e-5:
            raise ValueError("The recorded episode minimum does not match its trajectory.")
    if len(grouped) != 100 or len(trajectories) != sum(int(e["steps"]) for e in episodes):
        raise ValueError("Extra or missing evaluation trajectories.")
    if abs(summary["success_rate"] - .91) > 1e-12 or abs(summary["mean_steps"] - 240.3) > 1e-9:
        raise ValueError("Summary statistics disagree with the episode records.")
    return summary, episodes, layouts, grouped


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172338",
        "axes.titleweight": "bold", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white"})


def plot_evaluation(folder, output, model=None):
    summary, episodes, layouts, grouped = verify_evaluation(folder, model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    fig.suptitle("91% navigation success across 100 evaluation episodes", fontsize=17, weight="bold")
    labels = ["Goal reached", "Collision threshold", "Timeout"]
    values = [91, 9, 0]
    colors = ["#147d66", "#d65a39", "#94a3b8"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=.58)
    ax.set(xlim=(0, 106), xlabel="Episodes (out of 100)",
           title="Deterministic policy · 12–15 boxes · goal-facing starts")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1.4, bar.get_y() + bar.get_height()/2,
                f"{value} ({value}%)", va="center", weight="bold", color="#172338")
    fig.text(.5, -.015, "600,000-step training budget · selected checkpoint: 540,000 steps · seeds 360000–360099",
             ha="center", fontsize=9, color="#475569")
    fig.savefig(output / "evaluation_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    x = np.arange(1, 101)
    colors_per_case = [colors[0] if e["outcome"] == "success" else colors[1] for e in episodes]
    axes[0].bar(x, [int(e["steps"]) for e in episodes], color=colors_per_case, width=.82)
    axes[0].axhline(240.3, color="#475569", ls="--", lw=1, label="All-episode mean: 240.3 steps")
    axes[0].set(ylabel="Actions per episode", title="Every evaluation episode, in seed order")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].scatter(x, [float(e["minimum_clearance_m"]) for e in episodes], c=colors_per_case, s=24)
    axes[1].axhline(.22, color=colors[1], ls="--", lw=1, label="Collision threshold: 0.22 m")
    axes[1].set(xlabel="Evaluation episode (seed = 359999 + episode)", ylabel="Minimum raw LiDAR range (m)")
    axes[1].legend(handles=[Patch(color=colors[0], label="Success"), Patch(color=colors[1], label="Collision"),
                            Line2D([], [], color=colors[1], ls="--", label="Collision threshold: 0.22 m")],
                   loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#e2e8f0", alpha=.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 101)
    fig.savefig(output / "evaluation_episodes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), layout="constrained")
    for ax, ep, layout in zip(axes.flat, episodes[:6], layouts[:6]):
        ax.add_patch(Rectangle((-4.5, -4.5), 9, 9, fill=False, ec="#334155", lw=1.8))
        for bx, by, bz, yaw in layout["object_layout"].values():
            ax.add_patch(Rectangle((bx-.5, by-.5), 1, 1, fc="#64748b", ec="white", lw=.5))
        rows = grouped[int(ep["episode"])]
        sx, sy = layout["robot_x"], layout["robot_y"]
        path_x = [sx] + [float(r["robot_x"]) for r in rows]
        path_y = [sy] + [float(r["robot_y"]) for r in rows]
        color = "#2563eb" if ep["outcome"] == "success" else colors[1]
        ax.plot(path_x, path_y, color=color, lw=1.9, zorder=4)
        ax.scatter(sx, sy, marker="o", c="#147d66", s=35, ec="white", lw=.6, zorder=5)
        ax.add_patch(Circle(layout["goal"], .25, fc="#e9b24c", ec="#a76b13", lw=.6, zorder=5))
        ax.scatter(path_x[-1], path_y[-1], marker="x", c=color, s=36, lw=1.6, zorder=6)
        ax.set(xlim=(-4.7, 4.7), ylim=(-4.7, 4.7), aspect="equal", xlabel="x (m)", ylabel="y (m)",
               title=f"Seed {ep['seed']} · {ep['outcome']}\n{ep['boxes']} boxes · {ep['steps']} actions")
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.grid(alpha=.12)
    fig.suptitle("Recorded evaluation trajectories — first six cases", fontsize=16, weight="bold")
    fig.legend(handles=[Line2D([], [], c="#147d66", marker="o", ls="", label="Start"),
                        Patch(fc="#e9b24c", label="Goal region (radius 0.25 m)"),
                        Line2D([], [], c="#2563eb", label="Successful trajectory"),
                        Line2D([], [], c=colors[1], label="Collision trajectory")],
               loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.savefig(output / "evaluation_paths.png", dpi=180)
    plt.close(fig)
    return summary


def plot_training(folder, output, provenance=None):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = folder / "training_episodes.csv"
    if not path.exists():
        return []
    data = pd.read_csv(path).sort_values("total_steps", kind="stable")
    if data.empty:
        return []
    style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for stage, group in data.groupby("active_stage", sort=True):
        stage = int(stage)
        x = group.total_steps
        axes[0].plot(x, group.outcome.eq("success").rolling(50, min_periods=1).mean(),
                     label=f"Stage {stage+1}", lw=1.7)
        axes[1].plot(x, group["return"].rolling(20, min_periods=1).mean(), lw=1.5)
        axes[2].plot(x, group.steps.rolling(20, min_periods=1).mean(), lw=1.5)
    validation = folder / "validation_history.csv"
    if validation.exists():
        checks = pd.read_csv(validation)
        checks = checks[checks.complete.astype(str).str.lower().eq("true")]
        axes[0].scatter(checks.total_steps, checks.success_rate, color="#172338", marker="x", s=24,
                        zorder=5, label="Internal validation (12 cases)")
    axes[0].set(ylim=(-.03, 1.03), ylabel="Success fraction (50)", title="Recorded SAC training history")
    axes[0].legend(loc="lower right", ncol=2, fontsize=8, frameon=False)
    axes[1].set_ylabel("Mean return (20 episodes)")
    axes[2].set(ylabel="Mean actions (20 episodes)", xlabel="Cumulative training environment steps")
    changes = [(305000, "Goal-facing resets", "#8b5cf6")]
    if provenance and provenance.get("dense_started_at_steps") is not None:
        changes.append((provenance["dense_started_at_steps"], "85% dense / 15% passage", "#c76a20"))
    xmin, xmax = int(data.total_steps.min()), int(data.total_steps.max())
    for ax in axes:
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis="x", style="plain")
        for step, label, color in changes:
            if xmin <= step <= xmax:
                ax.axvline(step, color=color, lw=1.1, ls=":", alpha=.8)
    visible = [(s, label, c) for s, label, c in changes if xmin <= s <= xmax]
    if visible:
        axes[1].legend(handles=[Line2D([], [], color=c, ls=":", label=f"{s:,}: {label}") for s,label,c in visible],
                       fontsize=8, loc="best", frameon=False)
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    made = ["training_curves.png"]
    updates_path = folder / "training_updates.csv"
    if updates_path.exists():
        updates = pd.read_csv(updates_path).sort_values("total_steps", kind="stable")
        if len(updates):
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
            for ax, col, label in zip(axes, ["actor_loss", "critic_loss", "entropy_coefficient"],
                                     ["Actor loss", "Critic loss", "Entropy coefficient"]):
                values = pd.to_numeric(updates[col], errors="coerce")
                ax.plot(updates.total_steps, values, color="#2563eb", lw=1.6)
                ax.set_ylabel(label)
                ax.grid(alpha=.18)
                ax.ticklabel_format(axis="x", style="plain")
            axes[0].set_title("SAC optimization diagnostics")
            axes[-1].set_xlabel("Cumulative training environment steps")
            fig.savefig(output / "sac_diagnostics.png", dpi=180)
            plt.close(fig)
            made.append("sac_diagnostics.png")
    return made


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--training-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("docs/plots"))
    p.add_argument("--model", type=Path)
    p.add_argument("--provenance", type=Path)
    args = p.parse_args(argv)
    summary = plot_evaluation(args.evaluation_dir, args.output_dir, args.model)
    provenance = json.loads(args.provenance.read_text()) if args.provenance and args.provenance.exists() else None
    if args.training_dir:
        plot_training(args.training_dir, args.output_dir, provenance)
    print(f"Verified {summary['success']}/100 success. Figures: {args.output_dir}")


if __name__ == "__main__":
    main()
