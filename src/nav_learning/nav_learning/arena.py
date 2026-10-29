"""Arena geometry retained from the Jazzy project. No learning algorithm lives here.

The legacy random-layout sampler is retained only for the reproducible standard
benchmark. New SAC training uses tasks.py. Geometry may shape rewards, but no
route or waypoint is included in the policy observation or steering command.
"""
from __future__ import annotations
import math
import heapq
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import numpy as np
WORLD_NAME = 'rl_square_arena'
ARENA_SIZE_M = 9.0
ARENA_HALF_M = ARENA_SIZE_M / 2.0
WALL_THICKNESS_M = 0.25
WALL_HEIGHT_M = 0.90
BOX_SIZE_M = 1.0
BOX_HEIGHT_M = 0.80
BOX_HALF_M = BOX_SIZE_M / 2.0
BOX_COUNT = 15
BOX_NAMES = tuple(f'arena_box_{i:02d}' for i in range(BOX_COUNT))
MIN_BOX_GAP_M = 0.65
MIN_BOX_WALL_GAP_M = 0.30
GOAL_RADIUS_M = 0.25
GOAL_SIZE_M = 2.0 * GOAL_RADIUS_M
ROBOT_FOOTPRINT_RADIUS_M = 0.22
GOAL_SUCCESS_RADIUS_M = GOAL_RADIUS_M
START_CLEARANCE_M = 0.50
GOAL_CLEARANCE_M = 0.50
GOAL_EDGE_CLEARANCE_M = 0.10
PATH_GRID_RESOLUTION_M = 0.10
PATH_CLEARANCE_M = ROBOT_FOOTPRINT_RADIUS_M + 0.03
Pose = tuple[float, float, float, float]



COLLISION_DISTANCE_M = 0.22

@dataclass(frozen=True)
class StageDefinition:
    size_m: float
    distance_limits_m: tuple[float, float]
    box_count_limits: tuple[int, int]
    max_steps: int
    blocked_path_probability: float

STAGES = {
    0: StageDefinition(3.0, (0.8, 2.2), (0, 2), 450, 0.0),
    1: StageDefinition(6.0, (2.0, 5.0), (5, 7), 900, 0.5),
    2: StageDefinition(9.0, (3.0, 10.0), (12, 15), 1600, 0.8),
}

DETOUR_PHASE_NAMES = ("one_box", "two_boxes", "three_boxes", "full_stage2")

DETOUR_PHASE_COUNT = 4

def detour_stage_definition(phase: int) -> StageDefinition:
    """Training-only practice in the same 6 m patch; benchmarks stay unchanged."""
    if isinstance(phase, bool) or phase not in (0, 1, 2):
        raise ValueError("Detour practice phase must be 0, 1 or 2.")
    return StageDefinition(6.0, ((2.0, 3.2), (2.0, 4.0), (2.0, 5.0))[phase],
                           (phase + 1, phase + 1), 900, 1.0)

def parked_box_pose(index: int) -> Pose:
    """Unused static boxes stay outside the walls and away from robot staging."""
    return (-7.0 + index, -8.0, BOX_HEIGHT_M / 2.0, 0.0)

@dataclass(frozen=True)
class ArenaEpisode:
    layout: dict[str, Pose]
    start: tuple[float, float, float]
    goal: tuple[float, float]
    straight_path_blocked: bool
    layout_sampling_attempt: int
    straight_path_blocker_count: int = 0
    curriculum_level: int = 2
    region_center: tuple[float, float] = (0.0, 0.0)
    region_size_m: float = ARENA_SIZE_M
    practice_phase: int = -1
    task_kind: str = "random"
    mirrored: bool = False

def _check_layout(layout: Mapping[str, Pose]) -> None:
    for name, pose in layout.items():
        if name not in BOX_NAMES:
            raise ValueError(f'Unknown arena box: {name}')
        if len(pose) != 4 or not np.isfinite(pose).all():
            raise ValueError('Box poses must contain finite x, y, z, yaw.')
        if abs(pose[3]) > 1e-9:
            raise ValueError('This geometry contract uses axis-aligned boxes.')

def point_is_clear(point: tuple[float, float], layout: Mapping[str, Pose],
                   clearance_m: float = START_CLEARANCE_M) -> bool:
    """Conservative square clearance around a point, including arena walls."""
    if not np.isfinite(point).all() or not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Point and clearance must be finite; clearance nonnegative.')
    x, y = point
    if max(abs(x), abs(y)) > ARENA_HALF_M - clearance_m:
        return False
    return all(max(abs(x - p[0]), abs(y - p[1])) > BOX_HALF_M + clearance_m
               for p in layout.values())

def goal_region_reached(robot_xy: tuple[float, float], goal_xy: tuple[float, float]) -> bool:
    """Success means the robot CENTRE reaches the small circular marker."""
    if not np.isfinite((*robot_xy, *goal_xy)).all():
        raise ValueError('Goal and robot coordinates must be finite.')
    return math.dist(robot_xy, goal_xy) <= GOAL_SUCCESS_RADIUS_M

def straight_path_blocker_count(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                                layout: Mapping[str, Pose],
                                clearance_m: float = PATH_CLEARANCE_M) -> int:
    """Count boxes intersecting the swept straight-line footprint; reset metadata only."""
    _check_layout(layout)
    if not np.isfinite((*start_xy, *goal_xy, clearance_m)).all() or clearance_m < 0:
        raise ValueError('Segment coordinates and nonnegative clearance must be finite.')
    start = np.asarray(start_xy, dtype=float)
    delta = np.asarray(goal_xy, dtype=float) - start
    half = BOX_HALF_M + clearance_m
    count = 0
    for pose in layout.values():
        centre = np.asarray(pose[:2])
        low, high = centre - half, centre + half
        enter, leave = 0.0, 1.0
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-12:
                if not low[axis] <= start[axis] <= high[axis]:
                    enter, leave = 1.0, 0.0
                    break
            else:
                a = (low[axis] - start[axis]) / delta[axis]
                b = (high[axis] - start[axis]) / delta[axis]
                enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
        if enter <= leave:
            count += 1
    return count

def straight_path_is_clear(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                           layout: Mapping[str, Pose],
                           clearance_m: float = PATH_CLEARANCE_M) -> bool:
    if not point_is_clear(start_xy, layout, clearance_m) or not point_is_clear(goal_xy, layout, clearance_m):
        return False
    return straight_path_blocker_count(start_xy, goal_xy, layout, clearance_m) == 0

def navigation_path_exists(start_xy: tuple[float, float], goal_xy: tuple[float, float],
                           layout: Mapping[str, Pose],
                           resolution_m: float = PATH_GRID_RESOLUTION_M,
                           region_center: tuple[float, float] = (0.0, 0.0),
                           region_size_m: float = ARENA_SIZE_M) -> bool:
    """Boolean reachability only: no route, waypoint or action is returned.

    Inflate by the robot margin AND half a grid-cell diagonal. This makes the
    four-connected cell-centre test conservative between neighbouring cells.
    """
    _check_layout(layout)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('resolution_m must be positive and finite.')
    if not np.isfinite((*region_center, region_size_m)).all() or not 0 < region_size_m <= ARENA_SIZE_M:
        raise ValueError('Region must be finite and fit inside the arena.')
    if max(abs(v) for v in region_center) + region_size_m / 2 > ARENA_HALF_M + 1e-9:
        raise ValueError('Region extends beyond the arena.')
    if not point_is_clear(start_xy, layout, PATH_CLEARANCE_M) or not point_is_clear(goal_xy, layout, PATH_CLEARANCE_M):
        return False
    clearance = PATH_CLEARANCE_M + resolution_m * math.sqrt(2.0) / 2.0
    limit = region_size_m / 2 - clearance
    if limit <= 0:
        return False
    coordinates = np.arange(-limit, limit + 1e-9, resolution_m)
    xx, yy = np.meshgrid(coordinates, coordinates)
    xx += region_center[0]
    yy += region_center[1]
    blocked = np.zeros(xx.shape, dtype=bool)
    for pose in layout.values():
        blocked |= (np.abs(xx - pose[0]) <= BOX_HALF_M + clearance) & (np.abs(yy - pose[1]) <= BOX_HALF_M + clearance)

    def cell(point):
        column, row = (int(round((v - origin + limit) / resolution_m))
                       for v, origin in zip(point, region_center))
        if 0 <= row < len(coordinates) and 0 <= column < len(coordinates):
            return row, column
        return None

    source, target = cell(start_xy), cell(goal_xy)
    if source is None or target is None or blocked[source] or blocked[target]:
        return False
    if source == target:
        return True
    pending = deque([source])
    visited = np.zeros(blocked.shape, dtype=bool)
    visited[source] = True
    while pending:
        row, col = pending.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(coordinates) and 0 <= nc < len(coordinates)):
                continue
            if blocked[nr, nc] or visited[nr, nc]:
                continue
            if (nr, nc) == target:
                return True
            visited[nr, nc] = True
            pending.append((nr, nc))
    return False

def sample_box_layout(rng: np.random.Generator, count: int = BOX_COUNT,
                      region_center: tuple[float, float] = (0.0, 0.0),
                      region_size_m: float = ARENA_SIZE_M) -> dict[str, Pose]:
    """Continuously sample boxes; no fixed obstacle grid or fixed corridor."""
    if not 0 <= count <= BOX_COUNT:
        raise ValueError('Invalid active box count.')
    # The patch edge is a sampling boundary, not an additional physical wall.
    limit = region_size_m / 2 - BOX_HALF_M - 0.15
    for _ in range(40):
        layout = {}
        for name in BOX_NAMES[:count]:
            for _ in range(600):
                x, y = (float(v) for v in rng.uniform(-limit, limit, 2) + region_center)
                if max(abs(x), abs(y)) > ARENA_HALF_M - BOX_HALF_M - MIN_BOX_WALL_GAP_M:
                    continue
                if all(max(abs(x - p[0]), abs(y - p[1])) >= BOX_SIZE_M + MIN_BOX_GAP_M
                       for p in layout.values()):
                    layout[name] = (x, y, BOX_HEIGHT_M / 2.0, 0.0)
                    break
            else:
                break
        if len(layout) == count:
            return layout
    raise RuntimeError('Could not pack the requested boxes with safe gaps.')

def _sample_random_episode(seed: int, curriculum_level: int = 0,
                   practice_phase: int | None = None) -> ArenaEpisode:
    """Random start, goal and obstacles; the same seed reproduces the task.

    Reachability is checked within the spawning patch. The policy receives
    no planned path, waypoint, or action from this reset-time check.
    """
    if curriculum_level not in STAGES:
        raise ValueError('Curriculum level must be 0, 1 or 2.')
    rng = np.random.default_rng(int(seed))
    if practice_phase is not None and curriculum_level != 1:
        raise ValueError("Detour practice is defined only for Stage 2 (level 1).")
    stage = (STAGES[curriculum_level] if practice_phase is None
             else detour_stage_definition(practice_phase))
    for attempt in range(1, 101):
        centre_limit = ARENA_HALF_M - stage.size_m / 2
        centre = tuple(float(v) for v in rng.uniform(-centre_limit, centre_limit, 2))
        if curriculum_level == 0:
            count = int(rng.choice((0, 1, 2), p=(0.25, 0.50, 0.25)))
        else:
            count = int(rng.integers(stage.box_count_limits[0], stage.box_count_limits[1] + 1))
        try:
            layout = sample_box_layout(rng, count, centre, stage.size_m)
        except RuntimeError:
            continue
        require_blocker = rng.random() < stage.blocked_path_probability
        limit = stage.size_m / 2 - 0.35
        for _ in range(500):
            start_xy = tuple(float(v) for v in rng.uniform(-limit, limit, 2) + centre)
            goal_xy = tuple(float(v) for v in rng.uniform(-limit, limit, 2) + centre)
            distance = math.dist(start_xy, goal_xy)
            if not stage.distance_limits_m[0] <= distance <= stage.distance_limits_m[1]:
                continue
            if not point_is_clear(start_xy, layout, START_CLEARANCE_M) or not point_is_clear(goal_xy, layout, GOAL_CLEARANCE_M):
                continue
            blockers = straight_path_blocker_count(start_xy, goal_xy, layout)
            if require_blocker and blockers == 0:
                continue
            if not navigation_path_exists(start_xy, goal_xy, layout,
                                          region_center=centre, region_size_m=stage.size_m):
                continue
            # Initially most headings are within 60 degrees of the goal;
            # include fully random headings too. Later stages use all headings.
            bearing = math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
            yaw = (bearing + rng.uniform(-math.pi / 3, math.pi / 3)
                   if ((curriculum_level == 0 and rng.random() < 0.75)
                       or practice_phase == 0)
                   else rng.uniform(-math.pi, math.pi))
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            return ArenaEpisode(layout, (*start_xy, float(yaw)), goal_xy,
                                blockers > 0, attempt, blockers,
                                curriculum_level, centre, stage.size_m,
                                -1 if practice_phase is None else practice_phase)
    raise RuntimeError(f'No connected stage {curriculum_level + 1} task for seed {seed}.')

def _disc_segment_is_clear(start, end, layout, radius):
    """Check a circular footprint along a straight segment (including corners).

    Near a box corner, radial clearance differs from square inflation. Check
    endpoint-to-box and box-corner-to-segment distances, plus intersections.
    This is only a reward helper; it does not change episode termination.
    """
    if any(max(abs(v) for v in point) > ARENA_HALF_M - radius for point in (start, end)):
        return False
    if straight_path_blocker_count(start, end, layout, clearance_m=0) > 0:
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    for bx, by, _, _ in layout.values():
        for x, y in (start, end):
            distance = math.hypot(max(abs(x - bx) - BOX_HALF_M, 0),
                                  max(abs(y - by) - BOX_HALF_M, 0))
            if distance < radius:
                return False
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            x, y = bx + sx * BOX_HALF_M, by + sy * BOX_HALF_M
            fraction = (0.0 if length_squared == 0 else
                        float(np.clip(((x - start[0]) * dx + (y - start[1]) * dy) / length_squared, 0, 1)))
            if math.hypot(x - start[0] - fraction * dx, y - start[1] - fraction * dy) < radius:
                return False
    return True

class DetourDistance:
    """Training reward helper: distance around boxes with a safety margin.

    Nodes are the goal and corners of inflated boxes. Dijkstra computes their
    distances to the goal once per reset. distance() returns ONE NUMBER; no
    route, next waypoint, preferred turn or action is exposed to the policy.
    This is a geometric heuristic, not a shortest-time differential-drive plan.
    """

    def __init__(self, layout: Mapping[str, Pose], goal: tuple[float, float],
                 clearance_m: float = 0.30):
        _check_layout(layout)
        if not math.isfinite(clearance_m) or clearance_m < PATH_CLEARANCE_M:
            raise ValueError("Detour clearance must be at least the reachability margin.")
        self.layout, self.goal, self.clearance = dict(layout), goal, clearance_m
        self.limit = ARENA_HALF_M - clearance_m
        self.corner_offset = BOX_HALF_M + clearance_m + 0.001
        points = [goal]
        for x, y, _, _ in layout.values():
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                corner = (x + dx * self.corner_offset, y + dy * self.corner_offset)
                if self._clear(corner):
                    points.append(corner)
        self.points = points
        edges = [[] for _ in points]
        for i, a in enumerate(points):
            for j in range(i):
                b = points[j]
                if self._visible(a, b):
                    length = math.dist(a, b)
                    edges[i].append((j, length))
                    edges[j].append((i, length))
        self.costs = [float("inf")] * len(points)
        self.costs[0] = 0.0
        pending = [(0.0, 0)]
        while pending:
            cost, index = heapq.heappop(pending)
            if cost != self.costs[index]:
                continue
            for neighbour, length in edges[index]:
                candidate = cost + length
                if candidate < self.costs[neighbour]:
                    self.costs[neighbour] = candidate
                    heapq.heappush(pending, (candidate, neighbour))

    def _clear(self, point):
        return point_is_clear(point, self.layout, self.clearance)

    def _visible(self, a, b):
        return straight_path_is_clear(a, b, self.layout, self.clearance)

    def _safe_distance(self, point):
        return min((math.dist(point, target) + cost
                    for target, cost in zip(self.points, self.costs)
                    if math.isfinite(cost) and self._visible(point, target)),
                   default=float("inf"))

    def distance(self, point: tuple[float, float]) -> float:
        """Remain defined when the robot enters the preferred safety margin.

        Project to safe space, checking the escape segment against the original
        collision margin. A larger escape cost discourages hugging corners.
        A bounded fallback handles disconnected/numerically degenerate states.
        """
        if self._clear(point):
            return min(20.0, self._safe_distance(point))
        x, y = point
        exits = [(float(np.clip(x, -self.limit + 0.001, self.limit - 0.001)),
                  float(np.clip(y, -self.limit + 0.001, self.limit - 0.001)))]
        for bx, by, _, _ in self.layout.values():
            if max(abs(x - bx), abs(y - by)) <= self.corner_offset:
                exits.extend(((bx - self.corner_offset, y), (bx + self.corner_offset, y),
                              (x, by - self.corner_offset), (x, by + self.corner_offset)))
        candidates = [3.0 * math.dist(point, exit_point) + self._safe_distance(exit_point)
                      for exit_point in exits if self._clear(exit_point)
                      and _disc_segment_is_clear(point, exit_point, self.layout, COLLISION_DISTANCE_M)]
        return min(20.0, min(candidates, default=20.0))

def wrap_to_pi(angle: float) -> float:

    return math.atan2(math.sin(angle), math.cos(angle))

def yaw_from_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:

    numerator = 2 * (w * z + x * y)
    denominator = 1 - 2 * (y * y + z * z)
    yaw = float(math.atan2(numerator, denominator))

    return yaw

def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:

    x = 0.0
    y = 0.0
    z = float(math.sin(yaw / 2))
    w = float(math.cos(yaw / 2))

    return x, y, z, w

def relative_goal(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
) -> tuple[float, float]:


    dx = goal_x - robot_x
    dy = goal_y - robot_y

    distance = math.hypot(dx, dy)
    world_bearing = math.atan2(dy, dx)
    relative_bearing = wrap_to_pi(world_bearing - robot_yaw)

    return distance, relative_bearing
