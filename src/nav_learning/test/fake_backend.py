"""Offline test double only: ideal unicycle motion and analytic ray casting.

This is not Gazebo and does not establish learning performance or drive limits.
"""
import math
import numpy as np
from nav_learning.sac_math import scan_sectors, build_observation


class FakeBackend:
    def __init__(self):
        self.angles = np.linspace(-math.pi, math.pi, 360)
        self.buffer = np.zeros(22, dtype=np.float32)
        self.closed = False
        self.forced_nearest = None

    def reset_task(self, task):
        self.task = task
        self.x, self.y, self.yaw = map(float, task.start)
        self.v = self.w = 0.
        return self.read()

    def read(self):
        directions = np.stack((np.cos(self.angles + self.yaw), np.sin(self.angles + self.yaw)), axis=1)
        p = np.array([self.x, self.y])
        with np.errstate(divide="ignore", invalid="ignore"):
            sides = np.where(directions > 0, 4.5, -4.5)
            wall = np.min((sides - p) / directions, axis=1)
            ranges = np.maximum(wall, .15)
            for bx, by, *_ in self.task.layout.values():
                a = (np.array([bx-.5, by-.5]) - p) / directions
                b = (np.array([bx+.5, by+.5]) - p) / directions
                enter = np.minimum(a, b).max(axis=1)
                leave = np.maximum(a, b).min(axis=1)
                hit = (leave >= np.maximum(enter, 0))
                ranges = np.where(hit, np.minimum(ranges, np.maximum(enter, .15)), ranges)
        sectors, nearest, front = scan_sectors(ranges, -math.pi, 2*math.pi/359)
        self.buffer[:] = build_observation(sectors, self.x, self.y, self.yaw, self.task.goal)
        if self.forced_nearest is not None:
            nearest = self.forced_nearest
        return self.buffer, {"robot_x": self.x, "robot_y": self.y, "robot_yaw": self.yaw,
                "nearest_obstacle_m": nearest, "front_obstacle_m": front,
                "measured_linear_velocity_mps": self.v, "measured_angular_velocity_radps": self.w,
                "simulation_interval_sec": .1, "maximum_sensor_age_sec": .01}

    def transition(self, v, w):
        if abs(w) < 1e-8:
            self.x += v * .1 * math.cos(self.yaw)
            self.y += v * .1 * math.sin(self.yaw)
        else:
            end = self.yaw + w * .1
            self.x += v / w * (math.sin(end) - math.sin(self.yaw))
            self.y += v / w * (math.cos(self.yaw) - math.cos(end))
            self.yaw = end
        self.v, self.w = v, w
        return self.read()

    def stop(self):
        self.v = self.w = 0.

    def close(self):
        self.stop()
        self.closed = True
