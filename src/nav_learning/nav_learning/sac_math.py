"""Versioned, ROS-independent observation, steering and reward contract."""
from dataclasses import dataclass, asdict
import math
import numpy as np

CONTRACT = "hv_bot_sac_steering22_v1"
LIDAR_BINS = 20
OBSERVATION_SIZE = 22
CONTROL_INTERVAL_SEC = 0.1
COLLISION_DISTANCE_M = 0.22
GOAL_RADIUS_M = 0.25
LIDAR_CAP_M = 8.0
GOAL_SCALE_M = 13.0


class InvalidScanError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    fixed_speed_mps: float = 0.22
    max_angular_speed_radps: float = 1.2
    gamma: float = 0.998
    detour_clearance_m: float = 0.25
    potential_scale: float = 1.0
    success_reward: float = 20.0
    collision_penalty: float = 20.0
    step_penalty: float = 0.01
    near_distance_m: float = 0.28
    near_penalty: float = 0.05
    front_distance_m: float = 0.6
    front_penalty: float = 0.05
    stuck_penalty: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All environment settings must be finite.")
        if not 0 < self.fixed_speed_mps <= 0.22:
            raise ValueError("This robot profile supports fixed speeds in (0, 0.22] m/s.")
        if not 0 < self.max_angular_speed_radps <= 1.2:
            raise ValueError("This profile uses angular speeds in (0, 1.2] rad/s.")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0, 1).")
        if not 0.25 <= self.detour_clearance_m < 0.325:
            raise ValueError("Detour margin must fit the 0.65 m passages.")
        if not COLLISION_DISTANCE_M < self.near_distance_m < 0.325:
            raise ValueError("Near threshold must fit a centred 0.65 m passage.")
        if self.front_distance_m <= COLLISION_DISTANCE_M:
            raise ValueError("Forward risk threshold must exceed collision distance.")
        if any(v < 0 for k, v in asdict(self).items() if k.endswith(('reward', 'penalty', 'scale'))):
            raise ValueError("Reward magnitudes must be nonnegative.")


def steering_velocity(action, settings):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (1,) or not np.isfinite(action).all():
        raise ValueError("Steering action must contain one finite value.")
    return settings.fixed_speed_mps, float(np.clip(action[0], -1, 1)) * settings.max_angular_speed_radps


def scan_sectors(ranges, angle_min, angle_increment, range_min=0.15):
    """Conservative min pooling; bin 10 is centred on the forward axis.

    Preserve all 360 degrees. A few invalid rays are treated as occupied;
    grossly malformed scans stop the run instead of entering replay.
    """
    rays = np.asarray(ranges, dtype=np.float64)
    if rays.ndim != 1 or len(rays) < LIDAR_BINS:
        raise InvalidScanError("Expected a full 360 degree LaserScan.")
    if not np.isfinite([angle_min, angle_increment, range_min]).all() or angle_increment <= 0:
        raise InvalidScanError("Invalid scan angle/range metadata.")
    if angle_increment * (len(rays) - 1) < 6.1:
        raise InvalidScanError("The 22-input contract requires 360 degree coverage.")
    if not 0 < range_min < COLLISION_DISTANCE_M:
        raise InvalidScanError("Laser minimum range must be below collision threshold.")
    invalid = np.isnan(rays) | np.isneginf(rays)
    if invalid.mean() > 0.1:
        raise InvalidScanError("Over 10% of LiDAR rays are invalid.")
    clean = np.clip(np.nan_to_num(rays, nan=range_min, neginf=range_min,
                                  posinf=LIDAR_CAP_M), range_min, LIDAR_CAP_M)
    angles = angle_min + np.arange(len(rays)) * angle_increment
    bins = np.floor((angles + math.pi + math.pi / LIDAR_BINS)
                    / (2 * math.pi / LIDAR_BINS)).astype(int) % LIDAR_BINS
    if len(np.unique(bins)) != LIDAR_BINS:
        raise InvalidScanError("Some LiDAR sectors have no samples.")
    pooled = np.full(LIDAR_BINS, LIDAR_CAP_M)
    np.minimum.at(pooled, bins, clean)
    front_mask = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= math.radians(25)
    front = float(clean[front_mask].min())
    return (pooled / LIDAR_CAP_M).astype(np.float32), float(clean.min()), front


def build_observation(sectors, robot_x, robot_y, yaw, goal):
    sectors = np.asarray(sectors, dtype=np.float32)
    if sectors.shape != (20,) or not np.isfinite(sectors).all():
        raise ValueError("Expected twenty finite LiDAR sectors.")
    if not np.isfinite([robot_x, robot_y, yaw, *goal]).all():
        raise ValueError("Nonfinite robot/goal pose.")
    dx, dy = goal[0] - robot_x, goal[1] - robot_y
    # Body-frame offsets convey the goal direction, including the robot yaw.
    gx = math.cos(yaw) * dx + math.sin(yaw) * dy
    gy = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return np.concatenate((np.clip(sectors, 0, 1),
                           np.clip([gx / GOAL_SCALE_M, gy / GOAL_SCALE_M], -1, 1))).astype(np.float32)


def reward_for_transition(previous_detour, current_detour, nearest, stuck, outcome, settings, *, front=None):
    """Discount-consistent potential shaping with explicit true terminals.

    The time limit is an external truncation: retain its next potential and
    bootstrap it in SAC. There is no separate timeout-terminal penalty.
    """
    if outcome not in {"running", "success", "collision", "timeout"}:
        raise ValueError("Unknown outcome.")
    if not np.isfinite([previous_detour, current_detour, nearest]).all():
        raise ValueError("Nonfinite reward input.")
    if front is not None and not math.isfinite(front):
        raise ValueError("Nonfinite forward range.")
    old_phi = -float(np.clip(previous_detour, 0, 20))
    next_phi = 0.0 if outcome in {"success", "collision"} else -float(np.clip(current_detour, 0, 20))
    parts = {
        "potential": settings.potential_scale * (settings.gamma * next_phi - old_phi),
        "step": -settings.step_penalty,
        "near": -settings.near_penalty * float(np.clip(
            (settings.near_distance_m - nearest) /
            (settings.near_distance_m - COLLISION_DISTANCE_M), 0, 1)),
        "front": (-settings.front_penalty * float(np.clip(
            (settings.front_distance_m - front) /
            (settings.front_distance_m - COLLISION_DISTANCE_M), 0, 1))) if front is not None else 0.,
        "stuck": -settings.stuck_penalty if stuck else 0.0,
        "success": settings.success_reward if outcome == "success" else 0.0,
        "collision": -settings.collision_penalty if outcome == "collision" else 0.0,
    }
    return float(sum(parts.values())), parts
