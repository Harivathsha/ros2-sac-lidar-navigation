"""Gymnasium interface; the ROS transport is imported only for a live run."""
from dataclasses import asdict
from collections import deque
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from nav_learning.arena import DetourDistance
from nav_learning.sac_math import Settings, reward_for_transition, steering_velocity, CONTRACT
from nav_learning.tasks import sample_task


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, settings=None, backend=None, seed_base=10000):
        super().__init__()
        self.settings = settings or Settings()
        if backend is None:
            from nav_learning.ros_transport import GazeboTransport
            backend = GazeboTransport()
        self.backend = backend
        self.observation_space = spaces.Box(-1, 1, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.seed_base = int(seed_base)
        self.episode_index = 0
        self.stage = 0
        self.evaluating = False
        self.task = None
        self.active = False
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.evaluating = bool(options.get("evaluation", False))
        stage = int(options.get("stage", self.stage))
        task_seed = int(seed) if seed is not None else self.seed_base + self.episode_index
        if not self.evaluating:
            self.episode_index += 1
        self.task = sample_task(task_seed, stage, training=not self.evaluating,
                                benchmark=bool(options.get("benchmark", False)))
        obs, info = self.backend.reset_task(self.task)
        self.detour = DetourDistance(self.task.layout, self.task.goal, self.settings.detour_clearance_m)
        self.previous_detour = self.detour.distance((info["robot_x"], info["robot_y"]))
        self.steps = 0
        self.episode_return = 0.0
        self.minimum_clearance = float(info["nearest_obstacle_m"])
        self.path_length = 0.0
        self.passage_crossed = False
        self.last_xy = (info["robot_x"], info["robot_y"])
        self.positions = deque([self.last_xy], maxlen=80)
        self.active = True
        info = {**info, **self.task.metadata(), "contract": CONTRACT}
        return np.array(obs, dtype=np.float32, copy=True), info

    def step(self, action):
        if not self.active:
            raise RuntimeError("reset() must succeed before step().")
        v, omega = steering_velocity(action, self.settings)
        obs, info = self.backend.transition(v, omega)
        self.steps += 1
        xy = (info["robot_x"], info["robot_y"])
        if self.task.kind == "passage" and not self.passage_crossed:
            boxes = list(self.task.layout.values())[:2]
            centre = (np.asarray(boxes[0][:2]) + np.asarray(boxes[1][:2])) / 2
            lateral = np.asarray(boxes[1][:2]) - np.asarray(boxes[0][:2])
            lateral /= np.linalg.norm(lateral)
            normal = np.array([lateral[1], -lateral[0]])
            previous = np.asarray(self.last_xy) - centre
            current = np.asarray(xy) - centre
            before, after = float(previous @ normal), float(current @ normal)
            if before * after <= 0 and abs(before - after) > 1e-9:
                crossing = previous + (current - previous) * before / (before - after)
                self.passage_crossed = abs(float(crossing @ lateral)) <= self.task.gap_m / 2 - .22
        self.path_length += math.dist(self.last_xy, xy)
        self.last_xy = xy
        self.positions.append(xy)
        stuck = len(self.positions) == 80 and max(math.dist(xy, p) for p in self.positions) < 0.15
        nearest = float(info["nearest_obstacle_m"])
        self.minimum_clearance = min(self.minimum_clearance, nearest)
        distance = math.dist(xy, self.task.goal)
        # Collision wins if both terminal conditions are true on the same step.
        outcome = ("collision" if nearest <= 0.22 else "success" if distance <= 0.25
                   else "timeout" if self.steps >= self.task.max_steps else "running")
        terminated = outcome in {"success", "collision"}
        truncated = outcome == "timeout"
        current_detour = self.detour.distance(xy)
        reward, parts = reward_for_transition(self.previous_detour, current_detour,
                                               nearest, stuck, outcome, self.settings,
                                               front=info["front_obstacle_m"])
        self.previous_detour = current_detour
        self.episode_return += reward
        info = {**info, "outcome": outcome, "is_success": outcome == "success",
                "distance_to_goal_m": distance, "stuck": stuck, "steps": self.steps,
                "commanded_linear_velocity_mps": v, "commanded_angular_velocity_radps": omega,
                "reward_components": parts, "detour_distance_m": current_detour}
        if terminated or truncated:
            self.backend.stop()
            self.active = False
            info["episode_metrics"] = {**self.task.metadata(), "active_stage": self.stage,
                "training_episode": self.episode_index, "outcome": outcome,
                "return": self.episode_return, "steps": self.steps,
                "minimum_clearance_m": self.minimum_clearance, "path_length_m": self.path_length,
                "passage_crossed": self.passage_crossed,
                "evaluation": self.evaluating}
        return np.array(obs, dtype=np.float32, copy=True), reward, terminated, truncated, info

    def state_dict(self):
        return {"episode_index": self.episode_index, "stage": self.stage, "seed_base": self.seed_base}

    def load_state_dict(self, state):
        self.episode_index = int(state["episode_index"])
        self.stage = int(state["stage"])
        if int(state["seed_base"]) != self.seed_base:
            raise ValueError("Training seed base changed.")

    def close(self):
        if not self.closed:
            self.closed = True
            self.active = False
            self.backend.close()
