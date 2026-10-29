"""Configuration, atomic files, simulator ownership and dependency checks."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import numpy as np
import yaml
from nav_learning.sac_math import Settings, CONTRACT


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("contract") != CONTRACT:
        raise ValueError("Wrong SAC environment contract.")
    settings = Settings(**config["environment"])
    if config["sac"]["gamma"] != settings.gamma:
        raise ValueError("SAC and shaping discounts must match.")
    if len(config["curriculum"]["minimum_steps"]) != 5:
        raise ValueError("There must be five curriculum stages.")
    if config["sac"]["learning_starts"] >= config["evaluation"]["every_steps"]:
        raise ValueError("The first validation must occur after learning has started.")
    if config["sac"]["buffer_size"] < config["sac"]["batch_size"]:
        raise ValueError("Replay buffer must fit a full minibatch.")
    for v in (config["checkpoint_every_steps"], config["evaluation"]["every_steps"],
              config["evaluation"]["dense_every_steps"], config["evaluation"]["episodes"], config["evaluation"]["banks"]):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError("Checkpoint/evaluation intervals and counts must be positive integers.")
    if not 0 < config["curriculum"]["promotion_success_rate"] <= 1:
        raise ValueError("Promotion rate must lie in (0, 1].")
    if any(not isinstance(v, int) or v < 0 for v in config["curriculum"]["minimum_steps"]):
        raise ValueError("Stage minimum steps must be nonnegative integers.")
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dependencies(device):
    import torch
    if importlib.metadata.version("stable-baselines3") != "2.7.0":
        raise RuntimeError("Install this project's pinned stable-baselines3==2.7.0.")
    if importlib.metadata.version("gymnasium") != "1.2.0":
        raise RuntimeError("Install this project's pinned gymnasium==1.2.0.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; the trainer will not silently switch devices.")
    try:
        torch.from_numpy(np.zeros((1, 22), dtype=np.float32)).to(device).cpu().numpy()
    except Exception as error:
        raise RuntimeError("The existing Torch/NumPy/device combination cannot exchange observations. Fix this before training.") from error
    return {"torch": torch.__version__, "stable_baselines3": "2.7.0", "gymnasium": "1.2.0",
            "numpy": np.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}


@contextmanager
def simulator_session(project):
    """One trainer/evaluator controls this simulator; do not change ROS signals."""
    lock_path = Path(project) / "runs" / "sac_simulator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SAC process owns the simulator. Stop it before evaluating.") from error
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        owned = not rclpy.ok()
        if owned:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        try:
            yield
        finally:
            if owned and rclpy.ok():
                rclpy.shutdown()
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in row.items() if not isinstance(v, (dict, tuple, list))}
    exists = path.exists()
    if exists:
        with path.open(newline="") as handle:
            fields = next(csv.reader(handle))
    else:
        fields = list(flat)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def capture_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist(),
            "cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    def tuples(item):
        return tuple(tuples(v) for v in item) if isinstance(item, list) else item
    random.setstate(tuples(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in state["cuda"]])
