#!/usr/bin/env python3
"""Preserve the verified 91/100 SAC release, make figures, and prune its old run.

Run from the existing project. This operates on local files only: it never
changes ROS package code, trains a model, or publishes to GitHub.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
from zipfile import ZipFile

import yaml

RUN_NAME = "goal_facing_20261028_221122_000773"
EVAL_NAME = "evaluation_goal_facing_20261029_152727_958096"
RELEASE = Path("models/sac_dense_540k")
EVAL_DEST = Path("docs/results/evaluation_100")
PROVENANCE = Path("docs/results/provenance")
MODEL_SHA = "422afbcf3ef7dd47149cf0bf3415022a2b0704dc5eae0496931c3e535d2fe27b"
PHOTO_URL = "https://raw.githubusercontent.com/Harivathsha/ros2-hybrid-ppo-navigation/main/docs/images/environment_gazebo.png"
CSV_NAMES = ("training_episodes.csv", "training_updates.csv", "validation_history.csv")
EVAL_FILES = ("evaluation_summary.json", "tested_policy.json", "evaluation_episodes.csv",
              "evaluation_trajectories.csv", "evaluation_layouts.json", "config_used.yaml")
OBSOLETE_TOOLS = ("sac_goal_facing.py", "sac_focus_finetune.py", "sac_wall_turn_update.py",
                  "diagnose_sac_heading.py", "check_sac_optimizer.py", "sac_installation.json")
GITIGNORE = """# Generated ROS workspace outputs and local experiments
build/
install/
log/
runs/
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
*.egg-info/
.coverage
core
core.*
.sac_release_pending_*/
.env
.env.*
*.bag
*.db3
*.mcap
*.mp4
*.mkv
*.mov
*.pkl
*.pt
*.pth
*.zip
!models/sac_dense_540k/best/stage_04/model.zip
"""

# README_TEXT and PLOT_SOURCE are appended to the downloadable, self-contained file.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def package_hash(package):
    digest = hashlib.sha256()
    for path in sorted(Path(package).glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_plot_module(path):
    spec = importlib.util.spec_from_file_location("sac_release_plotter", path)
    module = importlib.util.module_from_spec(spec)
    # Compile the embedded source directly to avoid retaining generated .pyc
    # files inside the publication staging tree.
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def child(root, path):
    root, path = Path(root).resolve(), Path(path)
    target = path.resolve()
    if path.is_symlink() or target == root or not target.is_relative_to(root):
        raise ValueError(f"Unsafe path outside its expected directory: {path}")
    return target


def ensure_no_symlinks(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError(f"Refusing an experiment or destination containing symlinks: {root}")


@contextmanager
def exclusive_session(project):
    lock = project / "runs/sac_simulator.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Training/evaluation is active. Let it save and exit before cleanup.") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def copy_file(source, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    if sha(source) != sha(dest):
        raise RuntimeError(f"Copy verification failed: {source}")


def native_model_steps(path):
    # SB3 stores these scalar fields as JSON. Do not unpickle Torch/replay files.
    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n == "data" or n.endswith("/data")]
        if len(names) != 1:
            raise ValueError("Unexpected SB3 model archive.")
        return int(json.loads(archive.read(names[0]))["num_timesteps"])


def source_info(project):
    package = project / "src/nav_learning/nav_learning"
    if not (package / "tasks.py").is_file() or not (project / "src/description_new/package.xml").is_file():
        raise ValueError("Use the existing ROS project containing description_new and nav_learning.")
    source = (package / "tasks.py").read_text()
    if "GOAL_FACING_RESET_V1" not in source or "DENSE_TRAINING_PHASE_V1" not in source:
        raise ValueError("The working goal-facing dense sampler is not installed. No files were removed.")
    return package, package_hash(package)


def collect_training(run, dest, dense_at):
    manifest, coverage = [], {}
    for name in CSV_NAMES:
        combined, fields = [], []
        for source, tag in [(run / "origin/prior_training" / name, "before_goal_facing"),
                            (run / name, "goal_facing")]:
            if not source.is_file():
                continue
            with source.open(newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "total_steps" not in reader.fieldnames:
                    raise ValueError(f"Training CSV has an unexpected schema: {source}")
                rows = list(reader)
            copy_file(source, dest / "raw" / tag / name)
            for field in [*reader.fieldnames, "history_source", "training_phase"]:
                if field not in fields:
                    fields.append(field)
            steps = []
            for row in rows:
                step = int(row["total_steps"])
                if not 0 <= step <= 600000:
                    raise ValueError("Training history includes progress outside the verified training budget.")
                steps.append(step)
                row["history_source"] = tag
                row["training_phase"] = ("original_headings" if tag == "before_goal_facing" else
                    "goal_facing_dense" if step > dense_at else "goal_facing")
                combined.append(row)
            manifest.append({"path": str(source.relative_to(run)), "sha256": sha(source),
                             "rows": len(rows), "first_steps": min(steps) if steps else None,
                             "last_steps": max(steps) if steps else None})
        if combined:
            combined.sort(key=lambda r: int(r["total_steps"]))
            # Keep repeated update/validation rows as recorded; interruptions can
            # legitimately produce multiple records at the same saved step.
            with (dest / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader(); writer.writerows(combined)
            coverage[name] = {"rows": len(combined),
                              "first_steps": min(int(r["total_steps"]) for r in combined),
                              "last_steps": max(int(r["total_steps"]) for r in combined)}
    if not all(name in coverage for name in CSV_NAMES):
        raise ValueError("Training episodes, updates and validation records are needed before deleting the run.")
    write_json(dest / "training_history_manifest.json", {"sources": manifest, "coverage": coverage,
        "note": "Only recorded rows are retained; no missing training data or wall time is reconstructed."})
    return coverage


def get_photo(project, dest, explicit):
    if explicit is not None:
        photo = explicit.expanduser().resolve()
        if not photo.is_file():
            raise FileNotFoundError(photo)
        copy_file(photo, dest)
    elif (project / "docs/images/environment_gazebo.png").is_file():
        copy_file(project / "docs/images/environment_gazebo.png", dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(PHOTO_URL, timeout=30) as response:
                dest.write_bytes(response.read())
        except Exception as error:
            raise RuntimeError("Could not retrieve the existing Gazebo photo. Rerun with --photo ~/Downloads/your_photo.png. No cleanup occurred.") from error
    from PIL import Image
    with Image.open(dest) as image:
        if image.format != "PNG":
            raise ValueError("Use a PNG photo (the README uses environment_gazebo.png).")
        image.verify()


def verify_native_release(project, root=None):
    root = Path(root or project)
    release = root / RELEASE
    model = release / "best/stage_04/model.zip"
    meta = read_json(release / "best/stage_04/metadata.json")
    config = yaml.safe_load((release / "config_used.yaml").read_text())
    _, digest = source_info(project)
    if (sha(model) != MODEL_SHA or meta.get("sha256") != MODEL_SHA or meta.get("steps") != 540000
            or meta.get("code_hash") != digest or config_hash(meta.get("config")) != config_hash(config)
            or native_model_steps(model) != 540000):
        raise ValueError("The release model, source or frozen configuration failed identity verification.")
    return model, config


def verify_manifest(project):
    data = read_json(project / RELEASE / "release_manifest.json")
    if data.get("model_sha256") != MODEL_SHA or data.get("success") != 91:
        raise ValueError("The existing release manifest is not this verified experiment.")
    for relative, digest in data["files"].items():
        path = child(project, project / relative)
        if sha(path) != digest:
            raise ValueError(f"A retained publication file changed: {relative}")
    verify_native_release(project)
    return data


def cleanup(project, run):
    # Verify the complete retained release again before the first removal.
    verify_manifest(project)
    removed, failed = [], []
    targets = [run] if run.exists() else []
    pointer = project / "runs/sac/latest_run_path.txt"
    if pointer.is_file():
        pointed = Path(pointer.read_text().strip()).expanduser().resolve()
        if pointed == run:
            targets.append(pointer)
    targets += [project / "tools" / name for name in OBSOLETE_TOOLS]
    targets += [project / "log"]
    targets += list((project / "src/nav_learning").rglob("__pycache__"))
    targets += [project / "tools/__pycache__", project / ".pytest_cache"]
    for path in targets:
        if not path.exists() and not path.is_symlink():
            continue
        path = child(project, path)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path.relative_to(project)))
        except OSError as error:
            failed.append({"path": str(path.relative_to(project)), "error": str(error)})
    write_json(project / PROVENANCE / "release_cleanup.json", {"removed": removed, "failed": failed,
        "retained": [str(RELEASE), "docs", "src", "build", "install", "README.md", "LICENSE",
                     "tools/install_sac_dependencies.py", "tools/plot_sac_release.py", "tools/prepare_sac_release.py"],
        "old_replay_removed": not run.exists(),
        "note": "Only the named historical run was removed; any separate recording/fresh run is retained."})
    if failed:
        raise RuntimeError("Release is preserved, but some cleanup failed. See release_cleanup.json.")
    print(f"Cleanup complete. Removed {len(removed)} old paths. The published model and result are verified.")


def prepare(project, *, apply=False, clean=False, photo=None):
    project = Path(project).expanduser().resolve()
    run = project / "runs/sac" / RUN_NAME
    with exclusive_session(project):
        package, code = source_info(project)
        marker = project / RELEASE / "release_manifest.json"
        if marker.is_file():
            verify_manifest(project)
            print("The verified 91/100 release already exists.")
            if apply and clean:
                cleanup(project, run)
            return project / RELEASE
        child(project, run)
        if not run.is_dir():
            raise FileNotFoundError(f"The source run is missing: {run}")
        ensure_no_symlinks(run)
        evaluation = run / EVAL_NAME
        model = evaluation / "model_tested.zip"
        best = run / "best/stage_04"
        config = yaml.safe_load((evaluation / "config_used.yaml").read_text())
        run_config = yaml.safe_load((run / "config_used.yaml").read_text())
        defaults = yaml.safe_load((project / "src/nav_learning/config/sac_training.yaml").read_text())
        metadata = read_json(best / "metadata.json")
        if (sha(model) != MODEL_SHA or sha(best / "model.zip") != MODEL_SHA
                or metadata.get("sha256") != MODEL_SHA or metadata.get("steps") != 540000
                or metadata.get("code_hash") != code or native_model_steps(model) != 540000
                or config_hash(metadata.get("config")) != config_hash(config)
                or config_hash(run_config) != config_hash(config) or config_hash(defaults) != config_hash(config)):
            raise ValueError("The current model/source/config differs from the verified evaluation. No cleanup occurred.")
        latest = read_json(run / "latest.json")
        latest_root = child(run, run / latest["checkpoint"])
        last_state = read_json(latest_root / "state.json")
        if (latest.get("steps") != 600000 or last_state.get("steps") != 600000
                or last_state.get("code_hash") != code or last_state.get("config_hash") != config_hash(config)):
            raise ValueError("Expected the completed 600,000-step training run. No cleanup occurred.")
        dense_setup = read_json(run / "dense_training_setup.json")
        dense_at = int(dense_setup["started_at_steps"])
        if not 325000 <= dense_at <= 540000:
            raise ValueError("Unexpected dense-training phase metadata.")
        for path in (project / RELEASE, project / "docs", project / "tools", project / "README.md", project / ".gitignore"):
            if path.exists() or path.is_symlink():
                child(project, path)
                if path.is_dir():
                    ensure_no_symlinks(path)
        print("Verified model: 540,000 steps; final training budget: 600,000 steps.")
        print("Preserve: exact tested model + native best metadata, all 100 results, training CSVs, phase provenance, source.")
        print("Generate: README, Gazebo photo, trajectory/outcome/training figures, and Git exclusions.")
        if clean:
            print("Remove after verified copying:", run)
            print("This removes old replay/checkpoints; later resume commands apply to a separate fresh recording run.")
        if not apply:
            print("Preview only. Use --apply --clean-runs to prepare and remove the named historical run.")
            return None
        with tempfile.TemporaryDirectory(prefix=".sac_release_pending_", dir=project) as temp:
            stage = Path(temp)
            plot_path = stage / "tools/plot_sac_release.py"
            plot_path.parent.mkdir(parents=True)
            plot_path.write_text(PLOT_SOURCE)
            plotter = load_plot_module(plot_path)
            plotter.verify_evaluation(evaluation, model)
            target_release = stage / RELEASE
            copy_file(model, target_release / "best/stage_04/model.zip")
            copy_file(best / "metadata.json", target_release / "best/stage_04/metadata.json")
            copy_file(run / "config_used.yaml", target_release / "config_used.yaml")
            for name in EVAL_FILES:
                copy_file(evaluation / name, stage / EVAL_DEST / name)
            for name in ("goal_facing_setup.json", "dense_training_setup.json", "run_metadata.json", "last_goal_facing_execution.json"):
                if (run / name).is_file():
                    copy_file(run / name, stage / PROVENANCE / name)
            copy_file(latest_root / "state.json", stage / PROVENANCE / "final_training_state.json")
            for name in ("config_used.yaml", "run_metadata.json"):
                if (run / "origin" / name).is_file():
                    copy_file(run / "origin" / name, stage / PROVENANCE / ("original_" + name))
            training = stage / "docs/results/training"
            training.mkdir(parents=True)
            coverage = collect_training(run, training, dense_at)
            provenance = {"training_budget_steps": 600000, "evaluated_checkpoint_steps": 540000,
                "goal_facing_started_at_steps": 305000, "dense_started_at_steps": dense_at,
                "recorded_training_episodes": coverage["training_episodes.csv"]["rows"],
                "training_coverage": coverage, "wall_clock_time_available": False,
                "initial_heading_policy": "face_goal_v1", "code_hash": code,
                "config_hash": config_hash(config), "model_sha256": MODEL_SHA,
                "evaluation_seeds": [360000, 360099], "success": 91, "collision": 9, "timeout": 0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fresh_training_note": "Current defaults use goal-facing resets from step zero. This differs from the historical run's reset/sampling sequence.",
                "model_selection": "Internal validation selected the 540k best policy before the 100-case final evaluation."}
            write_json(stage / PROVENANCE / "training_provenance.json", provenance)
            get_photo(project, stage / "docs/images/environment_gazebo.png", photo)
            plotter.plot_evaluation(stage / EVAL_DEST, stage / "docs/plots", target_release / "best/stage_04/model.zip")
            made = plotter.plot_training(training, stage / "docs/plots", provenance)
            if "training_curves.png" not in made or "sac_diagnostics.png" not in made:
                raise ValueError("Required training plots were not created. No cleanup occurred.")
            (stage / "README.md").write_text(README_TEXT)
            (stage / ".gitignore").write_text(GITIGNORE)
            verify_native_release(project, stage)
            files = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
            write_json(target_release / "release_manifest.json", {"model_sha256": MODEL_SHA, "success": 91,
                "collision": 9, "timeout": 0, "source_run": RUN_NAME, "source_evaluation": EVAL_NAME,
                "files": files})
            # Install generated files without replacing unrelated documentation.
            # Delete nothing until every installed copy has been verified.
            manifest_path = target_release / "release_manifest.json"
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path != manifest_path:
                    copy_file(path, project / path.relative_to(stage))
            copy_file(manifest_path, project / RELEASE / "release_manifest.json")
            verify_manifest(project)
        if clean:
            cleanup(project, run)
        print("READY: README.md, docs/, models/sac_dense_540k/, and tools/plot_sac_release.py")
        print("Model SHA256:", MODEL_SHA)
        print("Publish from this project directory using git add/commit/push to the new repository.")
        return project / RELEASE


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--apply", action="store_true", help="Prepare the concrete local publication files.")
    p.add_argument("--clean-runs", action="store_true", help="After verification, delete only the named historical run and obsolete helpers.")
    p.add_argument("--photo", type=Path, help="Your own PNG photo; otherwise reuse the existing README photo.")
    args = p.parse_args(argv)
    prepare(args.project_dir, apply=args.apply, clean=args.clean_runs, photo=args.photo)


# EMBEDDED_PAYLOADS

README_TEXT = '# SAC LiDAR Navigation with ROS 2\n\n**ROS 2 Jazzy · Gazebo Harmonic · Stable-Baselines3 · PyTorch**\n\nA differential-drive robot learns to reach randomly placed goals among static obstacles using Soft Actor-Critic (SAC). A separate deterministic evaluation recorded **91 successes, 9 collisions and no timeouts across 100 dense layouts**.\n\nThe experiment used a **600,000-step training budget**. The published policy is the **540,000-step checkpoint**, selected by internal validation before the final evaluation. Starts face the goal; forward speed is fixed at **0.22 m/s**.\n\n## Environment and navigation\n\n![Robot and obstacle arena in Gazebo](docs/images/environment_gazebo.png)\n\n*Gazebo screenshot of the robot and its 9 × 9 m arena.*\n\n![Recorded SAC evaluation trajectories](docs/plots/evaluation_paths.png)\n\n*Reconstruction from the first six evaluation layouts and recorded trajectories, including the collision in seed 360003. This is a data visualization.*\n\n| Item | Implementation |\n|---|---|\n| Observations | 20 minimum-pooled LiDAR sectors over 360° and two relative goal offsets in the robot frame: **22 values** |\n| Action | One continuous steering command, mapped to **−1.2 to +1.2 rad/s** |\n| Forward motion | Fixed **0.22 m/s**; one action advances **0.1 simulated seconds** |\n| Policy | Feedforward MLP with two 256-unit layers; SAC with twin critics and replay |\n| Dense task | Random start, goal and positions of **12–15 boxes**, each 1 × 1 m |\n| Initialization | Robot starts facing the goal, subject to simulator settling tolerance |\n| Goal / collision | Goal distance ≤0.25 m / minimum raw LiDAR range ≤0.22 m |\n\nThe policy chooses steering directly from LiDAR and the relative goal. Obstacle coordinates, reference routes and waypoints are not actor inputs. Known simulator geometry is used to generate reachable tasks and calculate a detour-distance **training reward**. This is a simulation experiment with static obstacles.\n\n## Training process\n\n| Stage | Practice |\n|---|---|\n| 1 | Short goals with 0–1 boxes |\n| 2 | Detours with 1–3 boxes |\n| 3 | Wider two-box passages, with 0.9–1.3 m gaps |\n| 4 | Narrower passages, with 0.65–0.95 m gaps, and mixed obstacle tasks |\n| 5 | Dense layouts; the final sampling profile uses 85% dense tasks and 15% passage practice |\n\nPromotion requires at least **10/12 deterministic validation successes**, after the stage\'s minimum training budget. Validation runs every 10,000 steps in earlier stages and every 50,000 in the dense stage. Two internal seed banks alternate. They are development cases, separate from the final 100-case bank.\n\nThe historical run began with seed-42 weights and the original heading distribution. At **305,000 steps**, it continued with goal-facing resets. Later, stage-5 sampling changed to the dense/passage mixture shown above. Weights, optimizers and valid replay transitions were retained during these distribution changes. Starting fresh with today\'s configuration is a **new experiment** and does not repeat this historical sequence exactly.\n\nThe reward combines potential-based progress shaping, +20 for reaching the goal, −20 for collision, a step cost, obstacle-proximity costs and a stuck cost. SAC uses learning rate `0.0003`, replay capacity `200000`, batch size `256`, discount `0.998`, target update factor `0.005` and automatic entropy tuning. Timeouts are truncations and retain value bootstrapping.\n\n![Recorded training success, returns and episode lengths](docs/plots/training_curves.png)\n\n*Curves come from retained training CSVs; the data manifest records their coverage. Training success uses rolling windows of 50 episodes; returns and lengths use 20. Markers identify changes to initialization and sampling.*\n\n![SAC actor loss, critic loss and entropy coefficient](docs/plots/sac_diagnostics.png)\n\n*Optimization diagnostics from saved update records. These are separate from navigation success on the evaluation cases.*\n\n## Evaluation results\n\nThe frozen 540,000-step policy was evaluated deterministically on seeds **360000–360099**, with 12–15 boxes per layout, goal-facing initialization and a 1,600-action episode limit. All 100 episodes are included.\n\n| Outcome | Episodes | Rate |\n|---|---:|---:|\n| Goal reached | 91 | **91%** |\n| Collision threshold reached | 9 | 9% |\n| Timeout | 0 | 0% |\n\nMean episode length was **240.3 actions** across all outcomes. Ninety-nine layouts had a blocked direct start-to-goal line, and the robot succeeded in 90 of those. The collision label comes from the LiDAR distance threshold; it is not a contact-sensor measurement. The 91% result applies to this model, simulator and seed bank.\n\n![Evaluation outcome counts](docs/plots/evaluation_results.png)\n\n![Episode lengths and minimum LiDAR ranges for all 100 cases](docs/plots/evaluation_episodes.png)\n\nCorner clearance and close obstacle approaches remain limitations. Further training and refinement **may** improve reliability; a higher success rate needs another measured evaluation. Moving obstacles and real-robot transfer have not been validated.\n\n## Install and build\n\nUse **Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic and Python 3.12**. If ROS is not installed, prepare its official apt repository first:\n\n```bash\nsudo apt update\nsudo apt install -y curl software-properties-common\nsudo add-apt-repository -y universe\nexport ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c \'import json,sys; print(json.load(sys.stdin)["tag_name"])\')"\nexport ROS_UBUNTU_CODENAME="$(. /etc/os-release; printf \'%s\' "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"\ncurl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${ROS_UBUNTU_CODENAME}_all.deb"\nsudo dpkg -i /tmp/ros2-apt-source.deb\nsudo apt update\nsudo apt install -y ros-jazzy-desktop\n```\n\nInstall workspace dependencies, clone and build:\n\n```bash\nsudo apt install -y git python3-venv python3-pip python3-rosdep \\\n  python3-colcon-common-extensions ros-jazzy-ros-gz ros-jazzy-xacro \\\n  ros-jazzy-robot-state-publisher ros-jazzy-rviz2\nmkdir -p ~/projects\ncd ~/projects\ngit clone https://github.com/Harivathsha/ros2-sac-lidar-navigation.git\ncd ros2-sac-lidar-navigation\nsource /opt/ros/jazzy/setup.bash\npython3 -m venv --system-site-packages .venv\ntouch .venv/COLCON_IGNORE\nsource .venv/bin/activate\npython -m pip install --upgrade pip\npython -m pip install \'torch>=2.3,<3\' --index-url https://download.pytorch.org/whl/cpu\npython tools/install_sac_dependencies.py\nif [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then\n  sudo rosdep init\nfi\nrosdep update\nrosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y\ncolcon build --symlink-install --packages-select description_new nav_learning\nsource install/setup.bash\npython -m nav_learning.check_sac --offline --device cpu\n```\n\nThis installs CPU PyTorch. On the author\'s existing CUDA laptop, use the working `~/ai_envs/robotics_ai` environment and keep its PyTorch installation. Set `SAC_DEVICE=cuda` only when `python -c \'import torch; print(torch.cuda.is_available())\'` reports `True`. The dependency installer preserves the installed Torch/NumPy builds.\n\n## Launch and test the published policy\n\nPrepare **each terminal** from the repository root:\n\n```bash\nsource /opt/ros/jazzy/setup.bash\nsource .venv/bin/activate\nsource install/setup.bash\nexport SAC_DEVICE=cpu\n```\n\nTerminal 1 — keep Gazebo running:\n\n```bash\nros2 launch description_new launch_sim.launch.py\n```\n\nTerminal 2 — optional live setup check, then the saved model:\n\n```bash\npython -u -m nav_learning.check_sac --device "$SAC_DEVICE"\npython -u -m nav_learning.evaluate_sac \\\n  --run-dir models/sac_dense_540k --checkpoint best \\\n  --stage 4 --suite standard --episodes 100 --seed-base 360000 \\\n  --output-dir "runs/published_evaluation_$(date +%Y%m%d_%H%M%S)" \\\n  --device "$SAC_DEVICE"\n```\n\nFor a short demonstration, use `--episodes 3` and a separate output directory. Reported 91/100 results belong to the full evaluation. Physics pauses between controlled actions and during updates. Run one learner or evaluator at a time. Simulator timing can cause small differences on another machine.\n\n## Fresh training and resume\n\nThis starts at **step zero with new weights and an empty replay buffer**, using the current goal-facing curriculum:\n\n```bash\nmkdir -p runs/sac\nexport SAC_RUN="$PWD/runs/sac/recording_$(date +%Y%m%d_%H%M%S)"\nprintf \'%s\\n\' "$SAC_RUN" > runs/sac/recording_run_path.txt\ntime python -u -X faulthandler -m nav_learning.train_sac \\\n  --config src/nav_learning/config/sac_training.yaml \\\n  --run-dir "$SAC_RUN" --target-steps 600000 --device "$SAC_DEVICE"\n```\n\nPress **Ctrl+C once** and wait for the saved-step message before closing the terminal. To resume this fresh run:\n\n```bash\nexport SAC_RUN="$(cat runs/sac/recording_run_path.txt)"\npython -u -X faulthandler -m nav_learning.train_sac \\\n  --resume "$SAC_RUN" --target-steps 600000 --device "$SAC_DEVICE"\n```\n\nThe target is absolute. Full checkpoints include model/optimizer state, replay, random-generator states and curriculum progress; the physical episode resets on resume. The two latest checkpoint step counts are retained. The published best policy is an evaluation artifact; new runs save their own full resume checkpoints.\n\nAfter training, evaluate the run\'s internally selected best dense policy:\n\n```bash\nexport SAC_EVAL_DIR="$SAC_RUN/evaluation_$(date +%Y%m%d_%H%M%S)"\npython -u -m nav_learning.evaluate_sac \\\n  --run-dir "$SAC_RUN" --checkpoint best \\\n  --stage 4 --suite standard --episodes 100 --seed-base 460000 \\\n  --output-dir "$SAC_EVAL_DIR" --device "$SAC_DEVICE"\npython -m nav_learning.plot_sac_results \\\n  --run-dir "$SAC_RUN" --evaluation-dir "$SAC_EVAL_DIR"\n```\n\nThe final command requires a saved stage-5 best policy. Earlier-stage checks use `--suite curriculum` and the matching zero-based `--stage 0`–`3`. New run scores are measured separately from the published 91% result.\n\n## Recreate figures and check the software\n\nThese commands need no running simulator:\n\n```bash\npython -m pytest -q src/nav_learning/test\npython tools/plot_sac_release.py \\\n  --evaluation-dir docs/results/evaluation_100 \\\n  --training-dir docs/results/training \\\n  --provenance docs/results/provenance/training_provenance.json \\\n  --model models/sac_dense_540k/best/stage_04/model.zip \\\n  --output-dir docs/plots\n```\n\n| Path | Contents |\n|---|---|\n| `src/description_new/` | Robot, arena, launch files and ROS/Gazebo bridges |\n| `src/nav_learning/` | Environment, SAC training, evaluation, configuration and tests |\n| `models/sac_dense_540k/` | Exact tested model, frozen configuration and selection metadata |\n| `docs/results/` | All 100 evaluation records, recorded training CSVs and experiment provenance |\n| `docs/images/`, `docs/plots/` | Gazebo photo and figures recreated from recorded data |\n| `runs/` | Local outputs of new training and evaluation; excluded from Git |\n\nIf packages cannot be found, build from the workspace root and source `install/setup.bash`. If `rclpy` cannot import, source Jazzy and use Python 3.12 with system site packages. If startup waits for sensors, check `/scan`, `/odom`, `/clock` and the simulator launch terminal.\n\nThe Isaac SAC reference informed the steering-only experiment design. The learner here uses maintained Stable-Baselines3 SAC and a feedforward policy; this is an adaptation to this robot and environment.\n\n**Author:** Harivathsha · **License:** Apache-2.0\n'

PLOT_SOURCE = '#!/usr/bin/env python3\n"""Plot recorded SAC outcomes, trajectories and training logs; no simulator needed."""\nfrom __future__ import annotations\n\nimport argparse\nfrom collections import Counter, defaultdict\nimport csv\nimport hashlib\nimport json\nimport math\nfrom pathlib import Path\n\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nfrom matplotlib.lines import Line2D\nfrom matplotlib.patches import Circle, Patch, Rectangle\nimport numpy as np\nimport pandas as pd\n\nMODEL_SHA = "422afbcf3ef7dd47149cf0bf3415022a2b0704dc5eae0496931c3e535d2fe27b"\n\n\ndef sha(path):\n    h = hashlib.sha256()\n    with Path(path).open("rb") as f:\n        for part in iter(lambda: f.read(1024 * 1024), b""):\n            h.update(part)\n    return h.hexdigest()\n\n\ndef verify_evaluation(folder, model=None):\n    """Check all 100 cases against the summary and every recorded transition."""\n    folder = Path(folder)\n    summary = json.loads((folder / "evaluation_summary.json").read_text())\n    policy = json.loads((folder / "tested_policy.json").read_text())\n    with (folder / "evaluation_episodes.csv").open(newline="") as f:\n        episodes = list(csv.DictReader(f))\n    with (folder / "evaluation_trajectories.csv").open(newline="") as f:\n        trajectories = list(csv.DictReader(f))\n    layouts = json.loads((folder / "evaluation_layouts.json").read_text())\n    expected_seeds = list(range(360000, 360100))\n    expected = {"complete": True, "requested_episodes": 100, "completed_episodes": 100,\n                "stage": 4, "suite": "standard", "deterministic": True,\n                "checkpoint_steps": 540000, "training_budget_steps": 600000,\n                "selected_checkpoint": "best", "initial_heading_policy": "face_goal_v1",\n                "success": 91, "collision": 9, "timeout": 0}\n    for k, value in expected.items():\n        if summary.get(k) != value:\n            raise ValueError(f"Unexpected evaluation {k}: {summary.get(k)!r}")\n    if (summary.get("seeds") != expected_seeds or len(episodes) != 100 or len(layouts) != 100\n            or [int(r["seed"]) for r in episodes] != expected_seeds\n            or [int(r["episode"]) for r in episodes] != list(range(1, 101))):\n        raise ValueError("The evaluation must contain the exact complete 100-case seed bank.")\n    if policy.get("sha256") != MODEL_SHA or policy.get("steps") != 540000:\n        raise ValueError("The tested-policy identity differs from the verified 91/100 model.")\n    if model is not None and sha(model) != MODEL_SHA:\n        raise ValueError("Model checksum differs from the model tested in this evaluation.")\n    counts = Counter(row["outcome"] for row in episodes)\n    if counts != Counter(success=91, collision=9):\n        raise ValueError("Episode outcomes disagree with the verified summary.")\n    grouped = defaultdict(list)\n    for row in trajectories:\n        grouped[int(row["episode"])].append(row)\n    for ep, layout in zip(episodes, layouts):\n        number, seed = int(ep["episode"]), int(ep["seed"])\n        rows = grouped[number]\n        steps = int(ep["steps"])\n        if (len(rows) != steps or [int(r["step"]) for r in rows] != list(range(1, steps + 1))\n                or any(int(r["seed"]) != seed for r in rows)\n                or rows[-1]["outcome"] != ep["outcome"]\n                or any(r["outcome"] != "running" for r in rows[:-1])):\n            raise ValueError(f"Incomplete or inconsistent trajectory for episode {number}.")\n        if (int(layout["seed"]) != seed or layout.get("initial_heading_policy") != "face_goal_v1"\n                or int(ep["boxes"]) != len(layout["object_layout"])\n                or not 12 <= int(ep["boxes"]) <= 15 or ep["task_kind"] != "standard"):\n            raise ValueError(f"Layout metadata differs in episode {number}.")\n        x, y, yaw = layout["requested_start"]\n        gx, gy = layout["goal"]\n        angle_error = math.atan2(math.sin(yaw - math.atan2(gy-y, gx-x)),\n                                 math.cos(yaw - math.atan2(gy-y, gx-x)))\n        if abs(angle_error) > 1e-8:\n            raise ValueError("A requested spawn heading does not face its goal.")\n        for row in rows:\n            values = [float(row[k]) for k in ("robot_x", "robot_y", "robot_yaw",\n                "nearest_obstacle_m", "distance_to_goal_m", "commanded_linear_velocity_mps",\n                "commanded_angular_velocity_radps")]\n            if not all(math.isfinite(v) for v in values):\n                raise ValueError("Non-finite recorded observation or command.")\n            if abs(values[-2] - .22) > 1e-6 or abs(values[-1]) > 1.20001:\n                raise ValueError("Recorded action differs from the fixed-speed steering contract.")\n        if abs(min(float(r["nearest_obstacle_m"]) for r in rows) - float(ep["minimum_clearance_m"])) > 1e-5:\n            raise ValueError("The recorded episode minimum does not match its trajectory.")\n    if len(grouped) != 100 or len(trajectories) != sum(int(e["steps"]) for e in episodes):\n        raise ValueError("Extra or missing evaluation trajectories.")\n    if abs(summary["success_rate"] - .91) > 1e-12 or abs(summary["mean_steps"] - 240.3) > 1e-9:\n        raise ValueError("Summary statistics disagree with the episode records.")\n    return summary, episodes, layouts, grouped\n\n\ndef style():\n    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,\n        "axes.spines.top": False, "axes.spines.right": False,\n        "axes.labelcolor": "#334155", "text.color": "#172338",\n        "axes.titleweight": "bold", "figure.facecolor": "white",\n        "axes.facecolor": "white", "savefig.facecolor": "white"})\n\n\ndef plot_evaluation(folder, output, model=None):\n    summary, episodes, layouts, grouped = verify_evaluation(folder, model)\n    output = Path(output)\n    output.mkdir(parents=True, exist_ok=True)\n    style()\n    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")\n    fig.suptitle("91% navigation success across 100 evaluation episodes", fontsize=17, weight="bold")\n    labels = ["Goal reached", "Collision threshold", "Timeout"]\n    values = [91, 9, 0]\n    colors = ["#147d66", "#d65a39", "#94a3b8"]\n    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=.58)\n    ax.set(xlim=(0, 106), xlabel="Episodes (out of 100)",\n           title="Deterministic policy · 12–15 boxes · goal-facing starts")\n    ax.set_axisbelow(True)\n    ax.grid(axis="x", color="#e2e8f0")\n    for bar, value in zip(bars, values[::-1]):\n        ax.text(value + 1.4, bar.get_y() + bar.get_height()/2,\n                f"{value} ({value}%)", va="center", weight="bold", color="#172338")\n    fig.text(.5, -.015, "600,000-step training budget · selected checkpoint: 540,000 steps · seeds 360000–360099",\n             ha="center", fontsize=9, color="#475569")\n    fig.savefig(output / "evaluation_results.png", dpi=180, bbox_inches="tight")\n    plt.close(fig)\n\n    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")\n    x = np.arange(1, 101)\n    colors_per_case = [colors[0] if e["outcome"] == "success" else colors[1] for e in episodes]\n    axes[0].bar(x, [int(e["steps"]) for e in episodes], color=colors_per_case, width=.82)\n    axes[0].axhline(240.3, color="#475569", ls="--", lw=1, label="All-episode mean: 240.3 steps")\n    axes[0].set(ylabel="Actions per episode", title="Every evaluation episode, in seed order")\n    axes[0].legend(loc="upper right", frameon=False, fontsize=9)\n    axes[1].scatter(x, [float(e["minimum_clearance_m"]) for e in episodes], c=colors_per_case, s=24)\n    axes[1].axhline(.22, color=colors[1], ls="--", lw=1, label="Collision threshold: 0.22 m")\n    axes[1].set(xlabel="Evaluation episode (seed = 359999 + episode)", ylabel="Minimum raw LiDAR range (m)")\n    axes[1].legend(handles=[Patch(color=colors[0], label="Success"), Patch(color=colors[1], label="Collision"),\n                            Line2D([], [], color=colors[1], ls="--", label="Collision threshold: 0.22 m")],\n                   loc="upper right", frameon=False, fontsize=9)\n    for ax in axes:\n        ax.grid(axis="y", color="#e2e8f0", alpha=.8)\n        ax.set_axisbelow(True)\n        ax.set_xlim(0, 101)\n    fig.savefig(output / "evaluation_episodes.png", dpi=180)\n    plt.close(fig)\n\n    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), layout="constrained")\n    for ax, ep, layout in zip(axes.flat, episodes[:6], layouts[:6]):\n        ax.add_patch(Rectangle((-4.5, -4.5), 9, 9, fill=False, ec="#334155", lw=1.8))\n        for bx, by, bz, yaw in layout["object_layout"].values():\n            ax.add_patch(Rectangle((bx-.5, by-.5), 1, 1, fc="#64748b", ec="white", lw=.5))\n        rows = grouped[int(ep["episode"])]\n        sx, sy = layout["robot_x"], layout["robot_y"]\n        path_x = [sx] + [float(r["robot_x"]) for r in rows]\n        path_y = [sy] + [float(r["robot_y"]) for r in rows]\n        color = "#2563eb" if ep["outcome"] == "success" else colors[1]\n        ax.plot(path_x, path_y, color=color, lw=1.9, zorder=4)\n        ax.scatter(sx, sy, marker="o", c="#147d66", s=35, ec="white", lw=.6, zorder=5)\n        ax.add_patch(Circle(layout["goal"], .25, fc="#e9b24c", ec="#a76b13", lw=.6, zorder=5))\n        ax.scatter(path_x[-1], path_y[-1], marker="x", c=color, s=36, lw=1.6, zorder=6)\n        ax.set(xlim=(-4.7, 4.7), ylim=(-4.7, 4.7), aspect="equal", xlabel="x (m)", ylabel="y (m)",\n               title=f"Seed {ep[\'seed\']} · {ep[\'outcome\']}\\n{ep[\'boxes\']} boxes · {ep[\'steps\']} actions")\n        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])\n        ax.grid(alpha=.12)\n    fig.suptitle("Recorded evaluation trajectories — first six cases", fontsize=16, weight="bold")\n    fig.legend(handles=[Line2D([], [], c="#147d66", marker="o", ls="", label="Start"),\n                        Patch(fc="#e9b24c", label="Goal region (radius 0.25 m)"),\n                        Line2D([], [], c="#2563eb", label="Successful trajectory"),\n                        Line2D([], [], c=colors[1], label="Collision trajectory")],\n               loc="outside lower center", ncol=4, frameon=False, fontsize=9)\n    fig.savefig(output / "evaluation_paths.png", dpi=180)\n    plt.close(fig)\n    return summary\n\n\ndef plot_training(folder, output, provenance=None):\n    folder, output = Path(folder), Path(output)\n    output.mkdir(parents=True, exist_ok=True)\n    path = folder / "training_episodes.csv"\n    if not path.exists():\n        return []\n    data = pd.read_csv(path).sort_values("total_steps", kind="stable")\n    if data.empty:\n        return []\n    style()\n    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")\n    for stage, group in data.groupby("active_stage", sort=True):\n        stage = int(stage)\n        x = group.total_steps\n        axes[0].plot(x, group.outcome.eq("success").rolling(50, min_periods=1).mean(),\n                     label=f"Stage {stage+1}", lw=1.7)\n        axes[1].plot(x, group["return"].rolling(20, min_periods=1).mean(), lw=1.5)\n        axes[2].plot(x, group.steps.rolling(20, min_periods=1).mean(), lw=1.5)\n    validation = folder / "validation_history.csv"\n    if validation.exists():\n        checks = pd.read_csv(validation)\n        checks = checks[checks.complete.astype(str).str.lower().eq("true")]\n        axes[0].scatter(checks.total_steps, checks.success_rate, color="#172338", marker="x", s=24,\n                        zorder=5, label="Internal validation (12 cases)")\n    axes[0].set(ylim=(-.03, 1.03), ylabel="Success fraction (50)", title="Recorded SAC training history")\n    axes[0].legend(loc="lower right", ncol=2, fontsize=8, frameon=False)\n    axes[1].set_ylabel("Mean return (20 episodes)")\n    axes[2].set(ylabel="Mean actions (20 episodes)", xlabel="Cumulative training environment steps")\n    changes = [(305000, "Goal-facing resets", "#8b5cf6")]\n    if provenance and provenance.get("dense_started_at_steps") is not None:\n        changes.append((provenance["dense_started_at_steps"], "85% dense / 15% passage", "#c76a20"))\n    xmin, xmax = int(data.total_steps.min()), int(data.total_steps.max())\n    for ax in axes:\n        ax.grid(alpha=.18)\n        ax.ticklabel_format(axis="x", style="plain")\n        for step, label, color in changes:\n            if xmin <= step <= xmax:\n                ax.axvline(step, color=color, lw=1.1, ls=":", alpha=.8)\n    visible = [(s, label, c) for s, label, c in changes if xmin <= s <= xmax]\n    if visible:\n        axes[1].legend(handles=[Line2D([], [], color=c, ls=":", label=f"{s:,}: {label}") for s,label,c in visible],\n                       fontsize=8, loc="best", frameon=False)\n    fig.savefig(output / "training_curves.png", dpi=180)\n    plt.close(fig)\n    made = ["training_curves.png"]\n    updates_path = folder / "training_updates.csv"\n    if updates_path.exists():\n        updates = pd.read_csv(updates_path).sort_values("total_steps", kind="stable")\n        if len(updates):\n            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")\n            for ax, col, label in zip(axes, ["actor_loss", "critic_loss", "entropy_coefficient"],\n                                     ["Actor loss", "Critic loss", "Entropy coefficient"]):\n                values = pd.to_numeric(updates[col], errors="coerce")\n                ax.plot(updates.total_steps, values, color="#2563eb", lw=1.6)\n                ax.set_ylabel(label)\n                ax.grid(alpha=.18)\n                ax.ticklabel_format(axis="x", style="plain")\n            axes[0].set_title("SAC optimization diagnostics")\n            axes[-1].set_xlabel("Cumulative training environment steps")\n            fig.savefig(output / "sac_diagnostics.png", dpi=180)\n            plt.close(fig)\n            made.append("sac_diagnostics.png")\n    return made\n\n\ndef main(argv=None):\n    p = argparse.ArgumentParser(description=__doc__)\n    p.add_argument("--evaluation-dir", type=Path, required=True)\n    p.add_argument("--training-dir", type=Path)\n    p.add_argument("--output-dir", type=Path, default=Path("docs/plots"))\n    p.add_argument("--model", type=Path)\n    p.add_argument("--provenance", type=Path)\n    args = p.parse_args(argv)\n    summary = plot_evaluation(args.evaluation_dir, args.output_dir, args.model)\n    provenance = json.loads(args.provenance.read_text()) if args.provenance and args.provenance.exists() else None\n    if args.training_dir:\n        plot_training(args.training_dir, args.output_dir, provenance)\n    print(f"Verified {summary[\'success\']}/100 success. Figures: {args.output_dir}")\n\n\nif __name__ == "__main__":\n    main()\n'

if __name__ == "__main__":
    main()
