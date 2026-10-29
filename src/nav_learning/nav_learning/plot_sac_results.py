"""Create training curves and physical-scale evaluation trajectory figures."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--evaluation-dir", type=Path)
    args = p.parse_args(argv)
    output = args.run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = args.run_dir / "training_episodes.csv"
    if not episodes_path.exists():
        raise ValueError("No completed training episodes yet.")
    data = pd.read_csv(episodes_path)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for stage, group in data.groupby("active_stage"):
        x = group.total_steps
        axes[0].plot(x, group.outcome.eq("success").rolling(50, min_periods=1).mean(), label=f"Stage {stage+1}")
        axes[1].plot(x, group["return"].rolling(20, min_periods=1).mean())
        axes[2].plot(x, group.steps.rolling(20, min_periods=1).mean())
    validation = args.run_dir / "validation_history.csv"
    if validation.exists():
        checks = pd.read_csv(validation)
        checks = checks[checks.complete.astype(str).str.lower().eq("true")]
        axes[0].scatter(checks.total_steps, checks.success_rate, c="black", s=22, marker="x", label="Deterministic validation")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set(ylabel="Success fraction", ylim=(-.03, 1.03), title="SAC: training and deterministic validation")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("Mean episode return (20)")
    axes[2].set_ylabel("Mean steps per episode (20)")
    axes[2].set_xlabel("Training environment steps")
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    updates_path = args.run_dir / "training_updates.csv"
    if updates_path.exists():
        updates = pd.read_csv(updates_path)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for ax, column, label in zip(axes, ("actor_loss", "critic_loss", "entropy_coefficient"),
                                     ("Actor loss", "Critic loss", "Entropy coefficient")):
            ax.plot(updates.total_steps, updates[column])
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axes[0].set_title("SAC optimization diagnostics")
        axes[-1].set_xlabel("Training environment steps")
        fig.savefig(output / "sac_diagnostics.png", dpi=180)
        plt.close(fig)
    if args.evaluation_dir:
        folder = args.evaluation_dir
        summary = json.loads((folder / "evaluation_summary.json").read_text())
        layouts = json.loads((folder / "evaluation_layouts.json").read_text())
        episodes = pd.read_csv(folder / "evaluation_episodes.csv")
        trajectories = pd.read_csv(folder / "evaluation_trajectories.csv")
        # Show the first six cases in seed order, including failures.
        n = min(6, len(episodes))
        if n:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, (_, ep), layout in zip(axes.flat, episodes.iloc[:n].iterrows(), layouts[:n]):
                ax.set_visible(True)
                ax.add_patch(Rectangle((-4.5, -4.5), 9, 9, fill=False, lw=2))
                for x, y, z, yaw in layout["object_layout"].values():
                    ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, color="#475569"))
                route = trajectories[trajectories.episode == ep.episode]
                ax.plot(route.robot_x, route.robot_y, color="#2563eb", lw=1.5)
                ax.scatter(*layout["requested_start"][:2], marker="o", c="#0f766e", s=25)
                ax.add_patch(Circle(layout["goal"], .25, color="#f59e0b", alpha=.8))
                ax.set(xlim=(-4.7,4.7), ylim=(-4.7,4.7), aspect="equal",
                       title=f"Seed {ep.seed}: {ep.outcome}", xlabel="x (m)", ylabel="y (m)")
            fig.suptitle(f"{'Deterministic' if summary['deterministic'] else 'Stochastic'} evaluation: "
                         f"{summary['success']}/{summary['completed_episodes']} success; "
                         f"{summary['collision']} collision; {summary['timeout']} timeout")
            fig.savefig(output / "evaluation_paths.png", dpi=180)
            plt.close(fig)
    print(f"Plots: {output}")


if __name__ == "__main__":
    main()
