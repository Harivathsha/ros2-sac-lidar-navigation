"""Fresh SAC training and complete replay-aware resume for the Jazzy robot."""
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import json
import signal
import sys
import yaml
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from nav_learning.sac_math import Settings
from nav_learning.tasks import STAGE_NAMES
from nav_learning.environment import NavigationEnv
from nav_learning.io_utils import (load_config, config_hash, code_hash, dependencies,
    atomic_json, append_csv, restore_rng, simulator_session)
from nav_learning.checkpoints import save_checkpoint, resolve_checkpoint, save_best
from nav_learning.evaluation import evaluate_cases


def default_config():
    source = Path(__file__).resolve().parents[1] / "config" / "sac_training.yaml"
    if source.exists():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("nav_learning")) / "config" / "sac_training.yaml"


def create_model(env, config, device):
    return SAC("MlpPolicy", env, seed=int(config["seed"]), device=device,
               verbose=1, **deepcopy(config["sac"]))


class TrainingLog(BaseCallback):
    def __init__(self, run, stop):
        super().__init__()
        self.run = Path(run)
        self.stop = stop
        self.windows = defaultdict(lambda: deque(maxlen=50))

    def _on_step(self):
        for info in self.locals["infos"]:
            row = info.get("episode_metrics")
            if row is not None and not row["evaluation"]:
                row = {"total_steps": self.num_timesteps, **row}
                append_csv(self.run / "training_episodes.csv", row)
                self.windows[row["active_stage"]].append(row["outcome"] == "success")
                history = self.windows[row["active_stage"]]
                print(f"steps={self.num_timesteps} stage={row['active_stage']+1}/5 "
                      f"task={row['task_kind']} boxes={row['boxes']} outcome={row['outcome']} "
                      f"length={row['steps']} return={row['return']:.2f} "
                      f"recent_success={sum(history)}/{len(history)}", flush=True)
        return not self.stop()


def validation_seeds(config, stage, round_index):
    evaluation = config["evaluation"]
    start = int(evaluation["seed_base"]) + stage * 10000 + (round_index % evaluation["banks"]) * 1000
    return list(range(start, start + int(evaluation["episodes"])))


def apply_validation(trainer, stage, steps, summary, config):
    trainer["validation_rounds"][stage] += 1
    if not summary["complete"]:
        return False
    minimum = config["curriculum"]["minimum_steps"][stage]
    return (stage < 4 and steps - trainer["stage_started"] >= minimum
            and summary["success_rate"] >= config["curriculum"]["promotion_success_rate"])


def run_training(env, model, config, run, target, trainer, stop):
    """The same driver is exercised by the offline integration tests."""
    run = Path(run)
    interval = int(config["checkpoint_every_steps"])
    callback = TrainingLog(run, stop)
    if target <= model.num_timesteps:
        raise ValueError(f"Target {target} is already reached; checkpoint contains {model.num_timesteps} steps.")
    while model.num_timesteps < target and not stop():
        boundary = min(target, (model.num_timesteps // interval + 1) * interval)
        model.learn(total_timesteps=boundary - model.num_timesteps,
                    reset_num_timesteps=False, callback=callback, log_interval=10)
        logged = model.logger.name_to_value
        alpha = (model.log_ent_coef.detach().exp() if model.log_ent_coef is not None else model.ent_coef_tensor)
        append_csv(run / "training_updates.csv", {"total_steps": int(model.num_timesteps),
                   "stage": env.stage, "gradient_updates": int(model._n_updates),
                   "replay_size": model.replay_buffer.size(),
                   "actor_loss": logged.get("train/actor_loss"), "critic_loss": logged.get("train/critic_loss"),
                   "entropy_loss": logged.get("train/ent_coef_loss"), "entropy_coefficient": float(alpha.item())})
        if stop():
            break
        evaluation_interval = config["evaluation"]["dense_every_steps" if env.stage == 4 else "every_steps"]
        stage_minimum = config["curriculum"]["minimum_steps"][env.stage]
        if (model.num_timesteps - trainer["last_validation"] >= evaluation_interval
                and model.num_timesteps - trainer["stage_started"] >= stage_minimum):
            stage = env.stage
            round_index = trainer["validation_rounds"][stage]
            seeds = validation_seeds(config, stage, round_index)
            # Dense validation uses the original standard suite: 12–15 boxes,
            # random starts/goals/headings, no passage or rehearsal weighting.
            output = run / "validation" / f"step_{model.num_timesteps:09d}_stage_{stage:02d}_round_{round_index:03d}"
            print(f"Checking deterministic deployment: stage {stage+1} {STAGE_NAMES[stage]}", flush=True)
            summary = evaluate_cases(model, env, seeds, stage, output, benchmark=stage == 4, stop=stop)
            trainer["last_validation"] = int(model.num_timesteps)
            append_csv(run / "validation_history.csv", {"total_steps": int(model.num_timesteps), **summary})
            if summary["complete"]:
                score = (summary["success_rate"], -summary["collision"], -summary["mean_steps"])
                previous = trainer["best_scores"].get(str(stage))
                if previous is None or score > tuple(previous):
                    save_best(model, run, stage, summary, config)
                    trainer["best_scores"][str(stage)] = list(score)
            if apply_validation(trainer, stage, model.num_timesteps, summary, config):
                env.stage += 1
                trainer["stage_started"] = int(model.num_timesteps)
                print(f"Advanced to stage {env.stage+1}/5: {STAGE_NAMES[env.stage]}", flush=True)
        checkpoint = save_checkpoint(model, env, run, config, trainer)
        print(f"Checkpoint: {checkpoint}", flush=True)
    return trainer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--config", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", type=Path, help="Existing SAC run directory, including replay. Never a PPO .pt.")
    p.add_argument("--target-steps", type=int, default=700000, help="Absolute run target, including steps already trained.")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.target_steps <= 0:
        raise ValueError("Target steps must be positive.")
    project = args.project_dir.expanduser().resolve()
    runtime = dependencies(args.device)
    checkpoint = saved_state = None
    if args.resume:
        if args.run_dir:
            raise ValueError("Resume uses its existing run; do not supply --run-dir.")
        run = args.resume.expanduser().resolve()
        checkpoint, saved_state = resolve_checkpoint(run)
        config = load_config(run / "config_used.yaml")
        if saved_state["config_hash"] != config_hash(config) or saved_state["code_hash"] != code_hash():
            raise ValueError("SAC code/config changed since this checkpoint. Keep this run immutable.")
        if args.config and config_hash(load_config(args.config)) != config_hash(config):
            raise ValueError("Resume must use the run's frozen configuration.")
        if args.target_steps <= saved_state["steps"]:
            raise ValueError("This checkpoint has already reached the requested target.")
    else:
        config = load_config(args.config or default_config())
        run = (args.run_dir.expanduser().resolve() if args.run_dir else
               project / "runs" / "sac" / ("fresh_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
        run.mkdir(parents=True, exist_ok=False)
        (run / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        atomic_json(run / "run_metadata.json", {"runtime": runtime, "contract": config["contract"],
                    "code_hash": code_hash(), "config_hash": config_hash(config), "fresh": True})
    torch.set_num_threads(int(config["torch_threads"]))
    stopping = [False]
    def request_stop(signum, frame):
        if not stopping[0]:
            print("\nStop requested. Finishing the active transition, then saving SAC and replay.", flush=True)
        stopping[0] = True
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = model = None
    ready_to_save = False
    trainer = {"stage_started": 0, "last_validation": 0, "validation_rounds": [0] * 5, "best_scores": {}}
    try:
        with simulator_session(project):
            try:
                env = NavigationEnv(Settings(**config["environment"]), seed_base=config["reset_seed_base"])
                if checkpoint:
                    model = SAC.load(checkpoint / "model.zip", env=env, device=args.device, force_reset=True)
                    model.load_replay_buffer(checkpoint / "replay.pkl")
                    env.load_state_dict(saved_state["environment"])
                    trainer = saved_state["trainer"]
                    if model.num_timesteps != saved_state["steps"]:
                        raise ValueError("Model and trainer step counts disagree.")
                    restore_rng(saved_state["rng"])
                    model.action_space.np_random.bit_generator.state = saved_state["action_space_rng"]
                else:
                    model = create_model(env, config, args.device)
                # SB3 queues its algorithm seed for the first VecEnv reset.
                # Task layout seeds belong to our saved episode counter instead;
                # clearing that queue prevents the first old layout being
                # repeated after every resume. The SAC/action RNGs stay seeded.
                model.get_env()._reset_seeds()
                if checkpoint is None:
                    save_checkpoint(model, env, run, config, trainer)
                ready_to_save = True
                pointer = project / "runs" / "sac" / "latest_run_path.txt"
                pointer.parent.mkdir(parents=True, exist_ok=True)
                pointer.write_text(str(run) + "\n")
                print(f"Run: {run}\nStarting at {model.num_timesteps}; absolute target {args.target_steps}; "
                      f"stage {env.stage+1}/5; fixed speed {env.settings.fixed_speed_mps} m/s", flush=True)
                run_training(env, model, config, run, args.target_steps, trainer, lambda: stopping[0])
            finally:
                try:
                    if env is not None:
                        env.backend.stop()
                    if model is not None and ready_to_save:
                        save_checkpoint(model, env, run, config, trainer)
                finally:
                    if env is not None:
                        env.close()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f"Saved at {model.num_timesteps} steps. Run: {run}", flush=True)


if __name__ == "__main__":
    main()
