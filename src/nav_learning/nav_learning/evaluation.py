"""Seeded evaluation on the same simulator, with complete outcome records."""
from pathlib import Path
import csv
from collections import Counter
import numpy as np
from nav_learning.io_utils import atomic_json, capture_rng, restore_rng


def evaluate_cases(model, env, seeds, stage, output, *, benchmark=False, deterministic=True, stop=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng_state = capture_rng()
    training_state = env.state_dict()
    records, layouts = [], []
    fields = ["episode", "seed", "step", "robot_x", "robot_y", "robot_yaw",
              "nearest_obstacle_m", "distance_to_goal_m", "commanded_linear_velocity_mps",
              "commanded_angular_velocity_radps", "outcome"]
    try:
        # Action RNG is fixed for reproducible stochastic diagnostics and never
        # advances the subsequent training stream.
        import torch
        torch.manual_seed(int(seeds[0]) + 170003)
        with (output / "evaluation_trajectories.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, seed in enumerate(seeds, 1):
                if stop is not None and stop():
                    break
                obs, layout = env.reset(seed=int(seed), options={"evaluation": True, "stage": stage, "benchmark": benchmark})
                layouts.append(layout)
                while True:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    writer.writerow({k: ({"episode": index, "seed": seed, "step": info["steps"]}).get(k, info.get(k)) for k in fields})
                    if terminated or truncated:
                        row = {"episode": index, **info["episode_metrics"]}
                        records.append(row)
                        print(f"validation {index}/{len(seeds)} seed={seed} outcome={row['outcome']} steps={row['steps']}", flush=True)
                        break
                    if stop is not None and stop():
                        break
                if stop is not None and stop():
                    break
        counts = Counter(r["outcome"] for r in records)
        passages = [r for r in records if r["task_kind"] == "passage"]
        summary = {"requested_episodes": len(seeds), "completed_episodes": len(records),
                   "complete": len(records) == len(seeds), "stage": stage,
                   "suite": "standard" if benchmark else "curriculum",
                   "deterministic": deterministic, "checkpoint_steps": int(model.num_timesteps),
                   "success": counts["success"], "collision": counts["collision"], "timeout": counts["timeout"],
                   "success_rate": counts["success"] / len(records) if records else 0.,
                   "mean_steps": float(np.mean([r["steps"] for r in records])) if records else None,
                   "passage_episodes": len(passages),
                   "passage_crossed_and_success": sum(r["outcome"] == "success" and r["passage_crossed"] for r in passages),
                   "seeds": list(map(int, seeds))}
        atomic_json(output / "evaluation_summary.json", summary)
        atomic_json(output / "evaluation_layouts.json", layouts)
        scalar_fields = ["episode", "seed", "stage", "stage_name", "boxes", "task_kind", "gap_m", "outcome", "return", "steps", "minimum_clearance_m", "path_length_m", "passage_crossed"]
        with (output / "evaluation_episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        return summary
    finally:
        env.backend.stop()
        env.active = False
        env.load_state_dict(training_state)
        restore_rng(rng_state)
        # Evaluation changes the physical simulator. Force SB3 to get a fresh
        # training reset before collecting another replay transition.
        model._last_obs = None
