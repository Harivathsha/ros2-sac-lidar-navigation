"""Commit SAC, replay and trainer state together; never resume PPO weights."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from nav_learning.io_utils import atomic_json, capture_rng, file_hash, code_hash, config_hash


def save_checkpoint(model, env, run, config, trainer):
    run = Path(run)
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="pending_", dir=root))
    name = f"step_{model.num_timesteps:09d}"
    # Same-step saves may happen after validation or a stop request. Use a new
    # commit name rather than overwriting a previously complete checkpoint.
    if (root / name).exists():
        name += "_" + temp.name.removeprefix("pending_")
    try:
        model.save(temp / "model.zip")
        model.save_replay_buffer(temp / "replay.pkl")
        state = {"steps": int(model.num_timesteps), "environment": env.state_dict(),
                 "trainer": trainer, "rng": capture_rng(), "config_hash": config_hash(config),
                 "action_space_rng": model.action_space.np_random.bit_generator.state,
                 "code_hash": code_hash(), "files": {n: file_hash(temp / n) for n in ("model.zip", "replay.pkl")}}
        atomic_json(temp / "state.json", state)
        os.replace(temp, root / name)
        atomic_json(run / "latest.json", {"checkpoint": f"checkpoints/{name}", "steps": int(model.num_timesteps)})
        # Keep two commits, so an interrupted save cannot remove the last good
        # replay. Deletion applies only to this new run's generated checkpoints.
        complete = sorted((p for p in root.glob("step_*") if (p / "state.json").exists()),
                          key=lambda p: p.stat().st_mtime_ns)
        keep, steps_seen = set(), set()
        for path in reversed(complete):
            steps = json.loads((path / "state.json").read_text())["steps"]
            if steps not in steps_seen and len(steps_seen) < 2:
                keep.add(path)
                steps_seen.add(steps)
        for path in complete:
            if path not in keep:
                shutil.rmtree(path)
        return root / name
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir() and (path / "latest.json").exists():
        relative = json.loads((path / "latest.json").read_text())["checkpoint"]
        target = (path / relative).resolve()
        if not target.is_relative_to(path):
            raise ValueError("Checkpoint pointer escapes its run directory.")
        path = target
    if not path.is_dir() or not (path / "state.json").is_file():
        raise ValueError("Expected a SAC run or a complete checkpoint directory.")
    state = json.loads((path / "state.json").read_text())
    for name in ("model.zip", "replay.pkl"):
        if file_hash(path / name) != state["files"][name]:
            raise ValueError(f"Incomplete or changed SAC checkpoint: {name}")
    return path, state


def save_best(model, run, stage, summary, config):
    root = Path(run) / "best" / f"stage_{stage:02d}"
    root.mkdir(parents=True, exist_ok=True)
    temp = root / "model.pending.zip"
    model.save(temp)
    os.replace(temp, root / "model.zip")
    atomic_json(root / "metadata.json", {"summary": summary, "config": config,
                                        "steps": int(model.num_timesteps),
                                        "code_hash": code_hash(), "sha256": file_hash(root / "model.zip")})
