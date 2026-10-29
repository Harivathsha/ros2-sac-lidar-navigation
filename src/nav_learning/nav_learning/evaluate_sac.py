"""Evaluate a frozen SAC policy; never run concurrently with training."""
from datetime import datetime
from pathlib import Path
import argparse
import json
import shutil
import signal
import torch
from stable_baselines3 import SAC
from nav_learning.sac_math import Settings
from nav_learning.environment import NavigationEnv
from nav_learning.checkpoints import resolve_checkpoint
from nav_learning.io_utils import load_config, dependencies, simulator_session, atomic_json, file_hash, code_hash, config_hash
from nav_learning.evaluation import evaluate_cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="latest", help="latest, best, or an explicit checkpoint directory")
    p.add_argument("--stage", type=int, choices=range(5), default=4)
    p.add_argument("--suite", choices=("standard", "curriculum"), default="standard")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=240000, help="First exact episode seed; no hidden level offset")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("Episodes must be positive.")
    if args.suite == "standard" and args.stage != 4:
        raise ValueError("The standard suite is the dense stage; use --suite curriculum for earlier stages.")
    dependencies(args.device)
    run = args.run_dir.expanduser().resolve()
    config = load_config(run / "config_used.yaml")
    if args.checkpoint == "best":
        root = run / "best" / f"stage_{args.stage:02d}"
        metadata = json.loads((root / "metadata.json").read_text())
        model_path = root / "model.zip"
        if (file_hash(model_path) != metadata["sha256"] or metadata["code_hash"] != code_hash()
                or config_hash(metadata["config"]) != config_hash(config)):
            raise ValueError("Best policy or environment code changed.")
    else:
        root, metadata = resolve_checkpoint(run if args.checkpoint == "latest" else args.checkpoint)
        if metadata["code_hash"] != code_hash() or metadata["config_hash"] != config_hash(config):
            raise ValueError("Environment code differs from the checkpoint.")
        model_path = root / "model.zip"
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              run / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    if output.exists():
        raise FileExistsError(output)
    stopping = [False]
    handlers = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    torch.set_num_threads(int(config["torch_threads"]))
    try:
        with simulator_session(args.project_dir):
            env = NavigationEnv(Settings(**config["environment"]), seed_base=config["reset_seed_base"])
            try:
                model = SAC.load(model_path, device=args.device)
                summary = evaluate_cases(model, env, list(range(args.seed_base, args.seed_base + args.episodes)),
                    args.stage, output, benchmark=args.suite == "standard", deterministic=not args.stochastic,
                    stop=lambda: stopping[0])
                shutil.copy2(model_path, output / "model_tested.zip")
                shutil.copy2(run / "config_used.yaml", output / "config_used.yaml")
                atomic_json(output / "tested_policy.json", {"sha256": file_hash(model_path),
                            "source": str(model_path), "steps": int(model.num_timesteps)})
            finally:
                env.close()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f"Success {summary['success']}/{summary['completed_episodes']}; "
          f"collision {summary['collision']}; timeout {summary['timeout']}; complete={summary['complete']}\nResults: {output}")


if __name__ == "__main__":
    main()
