"""Seeded tasks in the existing 9 m world, including real passage practice."""
from dataclasses import dataclass, replace
import math
import numpy as np
from nav_learning.arena import (
    ARENA_HALF_M, BOX_NAMES, BOX_HEIGHT_M, ArenaEpisode,
    point_is_clear, navigation_path_exists, sample_box_layout,
    straight_path_blocker_count, _sample_random_episode, wrap_to_pi,
)

STAGE_NAMES = ("goals", "detours", "wide_passages", "mixed_passages", "dense")
MAX_STEPS = (450, 700, 900, 1100, 1600)


@dataclass(frozen=True)
class Task:
    seed: int
    stage: int
    layout: dict
    start: tuple
    goal: tuple
    max_steps: int
    kind: str
    gap_m: float | None = None

    def metadata(self):
        return {"seed": self.seed, "stage": self.stage, "stage_name": STAGE_NAMES[self.stage],
                "object_layout": self.layout, "requested_start": self.start, "goal": self.goal,
                "boxes": len(self.layout), "max_episode_steps": self.max_steps,
                "task_kind": self.kind, "gap_m": self.gap_m,
                "initial_heading_policy": START_HEADING_POLICY,
                "direct_blockers": straight_path_blocker_count(self.start[:2], self.goal, self.layout)}


def _random_task(seed, stage, rng):
    counts = ((0, 1), (1, 3), (2, 4), (4, 8), (12, 15))[stage]
    distances = ((0.8, 2.4), (2, 4), (2, 5), (2.5, 7), (3, 10))[stage]
    blocker_probability = (0, 0.65, 0.6, 0.7, 0.8)[stage]
    for _ in range(100):
        try:
            layout = sample_box_layout(rng, int(rng.integers(counts[0], counts[1] + 1)))
        except RuntimeError:
            continue
        require_blocker = rng.random() < blocker_probability
        for _ in range(300):
            start = tuple(rng.uniform(-3.8, 3.8, 2))
            goal = tuple(rng.uniform(-3.8, 3.8, 2))
            if not distances[0] <= math.dist(start, goal) <= distances[1]:
                continue
            if not point_is_clear(start, layout, 0.6) or not point_is_clear(goal, layout, 0.5):
                continue
            if require_blocker and not straight_path_blocker_count(start, goal, layout):
                continue
            if not navigation_path_exists(start, goal, layout):
                continue
            bearing = math.atan2(goal[1] - start[1], goal[0] - start[0])
            spread = (0.5, 1.2, math.pi, math.pi, math.pi)[stage]
            yaw = wrap_to_pi(bearing + rng.uniform(-spread, spread))
            return Task(seed, stage, layout, (*start, yaw), goal, MAX_STEPS[stage], "random")
    raise RuntimeError(f"Could not sample stage {stage} for seed {seed}.")


def _passage(seed, stage, rng):
    # Two real boxes straddle a line from the start to the goal. Quarter turns
    # preserve the axis-aligned box geometry; lateral jitter varies approaches.
    low, high = (0.9, 1.3) if stage == 2 else (0.65, 0.95)
    gap = float(rng.uniform(low, high))
    angle = int(rng.integers(4)) * math.pi / 2
    c, s = math.cos(angle), math.sin(angle)
    centre = rng.uniform(-1.1, 1.1, 2)

    def world(x, y):
        return (float(centre[0] + c * x - s * y), float(centre[1] + s * x + c * y))

    layout = {name: (*world(0, sign * (0.5 + gap / 2)), BOX_HEIGHT_M / 2, 0.0)
              for name, sign in zip(BOX_NAMES[:2], (-1, 1))}
    start = world(-float(rng.uniform(1.5, 2.1)), float(rng.uniform(-0.35, 0.35)))
    goal = world(float(rng.uniform(1.4, 2.0)), float(rng.uniform(-0.2, 0.2)))
    yaw = wrap_to_pi(angle + float(rng.uniform(-0.5, 0.5)))
    if not navigation_path_exists(start, goal, layout):
        raise RuntimeError("Generated passage failed the footprint reachability check.")
    return Task(seed, stage, layout, (*start, yaw), goal, MAX_STEPS[stage], "passage", gap)


def _sample_task_unaligned(seed, stage, *, training=True, benchmark=False):
    if isinstance(stage, bool) or stage not in range(5):
        raise ValueError("stage must be 0..4.")
    if not 0 <= int(seed) < 2**32:
        raise ValueError("Task seed must fit uint32.")
    rng = np.random.default_rng(int(seed))
    if benchmark:
        if stage != 4:
            raise ValueError("The standard benchmark is the dense stage.")
        episode = _sample_random_episode(int(seed), 2)
        return Task(int(seed), 4, episode.layout, episode.start, episode.goal, 1600, "standard")
    if stage >= 2 and rng.random() < (0.5 if stage in (2, 3) else 0.2):
        return _passage(int(seed), stage, rng)
    task_stage = stage
    if training and stage > 0 and rng.random() < 0.15:
        task_stage = int(rng.integers(0, stage))
    task = _random_task(int(seed), task_stage, rng)
    # Keep the actual task stage for diagnostics; trainer records its active
    # curriculum stage separately. Rehearsal never advances a training counter.
    return task


# GOAL_FACING_RESET_V1: preserve each seeded task, then change only spawn yaw.
START_HEADING_POLICY = "face_goal_v1"


def sample_task(seed, stage, *, training=True, benchmark=False):
    if training and not benchmark and stage == 4:
        if not 0 <= int(seed) < 2**32:
            raise ValueError("Task seed must fit uint32.")
        reserved = (240000 <= int(seed) < 240100 or 360000 <= int(seed) < 360100
                    or 660000 <= int(seed) < 660050 or 560000 <= int(seed) < 560012
                    or 561000 <= int(seed) < 561012)
        if reserved:
            raise ValueError("An evaluation seed entered training; collection was stopped.")
        rng = np.random.default_rng(int(seed))
        if rng.random() < 0.85:
            # Same layout distribution as dense evaluation, using fresh training seeds.
            task = _sample_task_unaligned(seed, 4, training=False, benchmark=True)
        else:
            # Retain explicit narrow-passage practice at the existing gap widths.
            task = _passage(int(seed), 4, rng)
    else:
        task = _sample_task_unaligned(seed, stage, training=training, benchmark=benchmark)
    x, y = task.start[:2]
    yaw = math.atan2(task.goal[1] - y, task.goal[0] - x)
    return replace(task, start=(x, y, yaw))

# DENSE_TRAINING_PHASE_V1: sampling only; no reward or action override.
