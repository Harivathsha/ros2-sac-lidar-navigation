"""Check dependencies, real simulator timing and measured drive response."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
from nav_learning.io_utils import dependencies, simulator_session, atomic_json, code_hash
from nav_learning.sac_math import Settings, CONTRACT, steering_velocity
from nav_learning.tasks import Task


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--offline", action="store_true", help="Dependency check only; does not validate Gazebo")
    args = p.parse_args(argv)
    runtime = dependencies(args.device)
    print(runtime, flush=True)
    if args.offline:
        print("DEPENDENCIES PASSED. Gazebo has not been tested.")
        return
    from nav_learning.ros_transport import GazeboTransport
    settings = Settings()
    rows = []
    with simulator_session(args.project_dir):
        backend = GazeboTransport()
        try:
            task = Task(424200, 0, {}, (-2., 0., 0.), (2., 0.), 450, "drive_check")
            obs, info = backend.reset_task(task)
            assert obs.shape == (22,) and obs.dtype == np.float32 and np.isfinite(obs).all()
            for steering in [0.] * 10 + [0.3] * 10 + [-0.3] * 10:
                v, w = steering_velocity([steering], settings)
                obs, info = backend.transition(v, w)
                if obs.shape != (22,) or not np.isfinite(obs).all():
                    raise RuntimeError("Malformed observation.")
                if not abs(info["simulation_interval_sec"] - .1) <= .002:
                    raise RuntimeError("Incorrect simulation control interval.")
                if info["maximum_sensor_age_sec"] > .05 + 1e-6:
                    raise RuntimeError("Stale sensor data.")
                rows.append({"command_v": v, "command_w": w, **info})
            mean_speed = float(np.mean([r["measured_linear_velocity_mps"] for r in rows[10:]]))
            if not .7 * settings.fixed_speed_mps <= mean_speed <= 1.3 * settings.fixed_speed_mps:
                raise RuntimeError(f"Measured speed {mean_speed:.3f} does not track 0.22 m/s. Do not start a long run.")
            if rows[19]["measured_angular_velocity_radps"] < .15 or rows[29]["measured_angular_velocity_radps"] > -.15:
                raise RuntimeError("Measured steering response has the wrong sign or insufficient magnitude.")
        finally:
            backend.close()
    output = args.project_dir / "runs" / "sac" / ("preflight_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    atomic_json(output, {"passed": True, "contract": CONTRACT, "runtime": runtime,
                        "settings": asdict(settings), "code_hash": code_hash(), "steps": rows})
    print(f"LIVE SIMULATOR CHECK PASSED: 22 observations, steering, speed and sensor timing.\nReport: {output}")


if __name__ == "__main__":
    main()
