"""Verified synchronous Gazebo transport, independent of the SAC learner.

Retains the original run_to_sim_time protocol and fresh-sensor checks.
Robot/world assets are not changed by this module.
"""
from __future__ import annotations
import math
import threading
import time
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from nav_learning.arena import BOX_NAMES, WORLD_NAME, parked_box_pose, quaternion_from_yaw, yaw_from_quaternion, wrap_to_pi
from nav_learning.sac_math import (LIDAR_BINS, CONTROL_INTERVAL_SEC, COLLISION_DISTANCE_M,
                                   InvalidScanError, scan_sectors, build_observation)
SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"
SET_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"
CONTROL_WORLD_SERVICE = f"/world/{WORLD_NAME}/control"
CONTROL_MODE = "gazebo_run_to_sim_time_v1"
MAX_SENSOR_AGE_SEC = 0.05  # The existing scan and odometry publishers run at 30 Hz.
CLOCK_TOLERANCE_NS = 2_000_000
STARTUP_TIMEOUT_SEC = 8.0
FRESH_SENSOR_TIMEOUT_SEC = 2.0
STEP_SENSOR_TIMEOUT_SEC = 5.0
NANOSECONDS_PER_SECOND = 1_000_000_000
ROBOT_NAME = "hv_bot"
GOAL_MARKER_NAME = "goal_marker"
ROBOT_RESET_Z_M = 0.20
GOAL_MARKER_Z_M = 0.012
SET_POSE_TIMEOUT_SEC = 2.0
RESET_SENSOR_TIMEOUT_SEC = 2.0
MAX_RESET_RETRIES = 3
RESET_POSITION_TOLERANCE_M = 0.20
RESET_YAW_TOLERANCE_RAD = 0.30

class GazeboTransport(Node):
    def __init__(self):
        super().__init__("sac_navigation")
        self._sensor_condition = threading.Condition()
        self._latest_scan = self._latest_odom = None
        self._scan_count = self._odom_count = 0
        self._scan_stamp_ns = self._odom_stamp_ns = self._clock_stamp_ns = None
        self._paused_stamp_ns = None
        self._last_step_timing = {}
        self._closed = False
        self._episode_active = False
        self.goal_xy = None
        self.cmd_vel_publisher = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.scan_subscription = self.create_subscription(LaserScan, SCAN_TOPIC, self._scan_callback, qos_profile_sensor_data)
        self.odom_subscription = self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, qos_profile_sensor_data)
        self.clock_subscription = self.create_subscription(Clock, "/clock", self._clock_callback, qos_profile_sensor_data)
        self.set_pose_client = self.create_client(SetEntityPose, SET_POSE_SERVICE)
        self.control_world_client = self.create_client(ControlWorld, CONTROL_WORLD_SERVICE)
        self._executor = SingleThreadedExecutor(context=self.context)
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()
        try:
            self._startup_check()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        """Convert a ROS timestamp into one integer."""

        if not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
            raise ValueError("Stamp must have 'sec' and 'nanosec' attributes")

        if not isinstance(stamp.sec, int) or not isinstance(stamp.nanosec, int):
            raise ValueError("'sec' and 'nanosec' must be integers")

        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _scan_callback(self, message: LaserScan) -> None:
        """Cache the newest LiDAR message."""

        with self._sensor_condition:
            self._latest_scan = message
            self._scan_count += 1
            self._scan_stamp_ns = self._stamp_to_nanoseconds(message.header.stamp)
            self._sensor_condition.notify_all()

    def _odom_callback(self, message: Odometry) -> None:
        """Cache the newest odometry message."""

        with self._sensor_condition:
            self._latest_odom = message
            self._odom_count += 1
            self._odom_stamp_ns = self._stamp_to_nanoseconds(message.header.stamp)
            self._sensor_condition.notify_all()

    def _clock_callback(self, message: Clock) -> None:
        with self._sensor_condition:
            self._clock_stamp_ns = self._stamp_to_nanoseconds(message.clock)
            self._sensor_condition.notify_all()

    @property
    def simulation_time_ns(self) -> int:
        with self._sensor_condition:
            if self._clock_stamp_ns is None:
                raise RuntimeError("No /clock message. Check the Gazebo clock bridge.")
            return self._clock_stamp_ns

    def _control_world(self, *, pause: bool, run_to_ns: int | None = None) -> None:
        """Request native Gazebo control; the response acknowledges acceptance."""
        request = ControlWorld.Request()
        request.world_control.pause = pause
        if run_to_ns is not None:
            request.world_control.run_to_sim_time.sec = run_to_ns // NANOSECONDS_PER_SECOND
            request.world_control.run_to_sim_time.nanosec = run_to_ns % NANOSECONDS_PER_SECOND
        future = self.control_world_client.call_async(request)
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(STEP_SENSOR_TIMEOUT_SEC):
            future.cancel()
            raise TimeoutError(f"Gazebo control request timed out: {CONTROL_WORLD_SERVICE}")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Gazebo rejected the world-control request.")

    def pause_simulation(self) -> None:
        """Pause physics and drain the last in-flight /clock publications."""
        self._control_world(pause=True)
        deadline = time.monotonic() + STEP_SENSOR_TIMEOUT_SEC
        previous = None
        while time.monotonic() < deadline:
            time.sleep(0.02)
            current = self.simulation_time_ns
            if current == previous:
                self._paused_stamp_ns = current
                return
            previous = current
        raise TimeoutError("Gazebo did not stop advancing after pause.")

    def assert_simulation_time_unchanged(self, expected_ns: int) -> None:
        """Catch accidental Play clicks or a broken pause/step bridge."""
        if abs(self.simulation_time_ns - expected_ns) > CLOCK_TOLERANCE_NS:
            raise RuntimeError(
                "Simulation advanced outside an environment step. Leave Gazebo paused "
                "while training; run the SAC simulator check before restarting."
            )

    def _advance_simulation(self, duration_sec: float) -> tuple[LaserScan, Odometry]:
        """Advance to an absolute sim timestamp, where Gazebo pauses itself.

        run_to_sim_time is enforced by Gazebo, not a wall-clock sleep. Sensor
        delivery may lag; timestamps must be recent and belong to this step.
        """
        if not math.isfinite(duration_sec) or duration_sec <= MAX_SENSOR_AGE_SEC:
            raise ValueError("Step duration must exceed the sensor age allowance.")
        if self._paused_stamp_ns is None:
            raise RuntimeError("Simulation has not been paused and synchronized.")
        self.assert_simulation_time_unchanged(self._paused_stamp_ns)
        start_ns = self._paused_stamp_ns
        target_ns = start_ns + int(round(duration_sec * NANOSECONDS_PER_SECOND))
        self._control_world(pause=False, run_to_ns=target_ns)
        with self._sensor_condition:
            arrived = self._sensor_condition.wait_for(
                lambda: self._clock_stamp_ns is not None and self._clock_stamp_ns >= target_ns,
                timeout=STEP_SENSOR_TIMEOUT_SEC,
            )
        if not arrived:
            raise TimeoutError("Gazebo failed to reach the requested simulation time.")
        scan, odom = self._wait_for_sensor_time(
            target_ns - int(MAX_SENSOR_AGE_SEC * NANOSECONDS_PER_SECOND),
            STEP_SENSOR_TIMEOUT_SEC,
        )
        end_ns = self.simulation_time_ns
        if abs(end_ns - target_ns) > CLOCK_TOLERANCE_NS:
            raise RuntimeError("Gazebo overshot the requested step; check world-control support.")
        sensor_ns = [self._stamp_to_nanoseconds(m.header.stamp) for m in (scan, odom)]
        if min(sensor_ns) <= start_ns or max(sensor_ns) > end_ns + CLOCK_TOLERANCE_NS:
            raise RuntimeError("Sensor timestamps do not belong to the completed action step.")
        self._paused_stamp_ns = end_ns
        self._last_step_timing = {
            "simulation_interval_sec": (end_ns - start_ns) / NANOSECONDS_PER_SECOND,
            "maximum_sensor_age_sec": max(0.0, (end_ns - min(sensor_ns)) / NANOSECONDS_PER_SECOND),
        }
        return scan, odom

    def _wait_for_first_sensors(
        self,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait until both sensor streams have produced a message."""

        with self._sensor_condition:

            success = self._sensor_condition.wait_for(
                lambda: self._latest_scan is not None and self._latest_odom is not None,
                timeout=timeout_sec
            )
            if not success:
                raise TimeoutError("Timeout waiting for first scan and odometry messages.")

            return self._latest_scan, self._latest_odom

    def _wait_for_sensor_time(
        self,
        target_stamp_ns: int,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait until both sensor streams reach a simulation timestamp."""

        with self._sensor_condition:
            success = self._sensor_condition.wait_for(
                lambda: (
                    self._latest_scan is not None
                    and self._latest_odom is not None
                    and self._scan_stamp_ns is not None
                    and self._odom_stamp_ns is not None
                    and self._scan_stamp_ns >= target_stamp_ns
                    and self._odom_stamp_ns >= target_stamp_ns
                ),
                timeout=timeout_sec,
            )

            if not success:
                raise TimeoutError(
                    "Timeout waiting for the action control interval."
                )

            return self._latest_scan, self._latest_odom

    def _wait_for_fresh_sensors(
        self,
        previous_scan_count: int,
        previous_odom_count: int,
        timeout_sec: float,
    ) -> tuple[LaserScan, Odometry]:
        """Wait for scan and odometry newer than captured counters."""

        with self._sensor_condition:
            success = self._sensor_condition.wait_for(
                lambda: self._scan_count > previous_scan_count and self._odom_count > previous_odom_count,
                timeout=timeout_sec
            )
            if not success:
                raise TimeoutError("Timeout waiting for fresh scan and odometry messages.")

            return self._latest_scan, self._latest_odom

    def _startup_check(self) -> None:
        """Verify essential simulation interfaces before training."""

        if not self.control_world_client.wait_for_service(timeout_sec=STARTUP_TIMEOUT_SEC):
            raise TimeoutError(
                f"Missing {CONTROL_WORLD_SERVICE}. Rebuild description_new and restart its launch."
            )
        if not self.set_pose_client.wait_for_service(
            timeout_sec=STARTUP_TIMEOUT_SEC
        ):
            raise TimeoutError(
                f"Timeout waiting for set-pose service {SET_POSE_SERVICE}."
            )

        # A previous trainer intentionally leaves physics paused. Bootstrap only
        # while no episode is active, then retain ownership of simulation time.
        self.publish_stop()
        self._control_world(pause=False)
        first_scan, _ = self._wait_for_first_sensors(STARTUP_TIMEOUT_SEC)
        if len(first_scan.ranges) < LIDAR_BINS:
            raise InvalidScanError(
                f"Initial scan has fewer than {LIDAR_BINS} rays."
            )
        # Do not validate scan contents before reset(). A previous interrupted
        # run may have left the robot at a geometrically invalid pose.
        # reset() teleports the robot to a valid start and then performs the
        # complete LiDAR validation.

        with self._sensor_condition:
            scan_count = self._scan_count
            odom_count = self._odom_count
            scan_stamp_ns = self._scan_stamp_ns
            odom_stamp_ns = self._odom_stamp_ns

        fresh_scan, fresh_odom = self._wait_for_fresh_sensors(
            scan_count,
            odom_count,
            FRESH_SENSOR_TIMEOUT_SEC,
        )

        fresh_scan_stamp_ns = self._stamp_to_nanoseconds(fresh_scan.header.stamp)
        fresh_odom_stamp_ns = self._stamp_to_nanoseconds(fresh_odom.header.stamp)
        if fresh_scan_stamp_ns == scan_stamp_ns:
            raise RuntimeError("Fresh scan timestamp did not change.")
        if fresh_odom_stamp_ns == odom_stamp_ns:
            raise RuntimeError("Fresh odometry timestamp did not change.")
        with self._sensor_condition:
            if not self._sensor_condition.wait_for(
                lambda: self._clock_stamp_ns is not None, timeout=STARTUP_TIMEOUT_SEC,
            ):
                raise TimeoutError("Missing /clock. Check the clock bridge.")
        self.pause_simulation()

        self.get_logger().info(
            "Navigation environment startup check passed: set-pose service, "
            "world control, /clock, LiDAR, and odometry are ready; physics is paused."
        )

    def _set_entity_pose(
        self,
        entity_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
    ) -> None:
        """Teleport one Gazebo model and verify the service response."""

        request = SetEntityPose.Request()

        request.entity.name = entity_name
        request.entity.type = Entity.MODEL

        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(z)

        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        request.pose.orientation.x = qx
        request.pose.orientation.y = qy
        request.pose.orientation.z = qz
        request.pose.orientation.w = qw

        future = self.set_pose_client.call_async(request)
        deadline = time.monotonic() + SET_POSE_TIMEOUT_SEC
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError(
                    f"Timeout waiting for set-pose service response for {entity_name}."
                )
            time.sleep(0.01)

        service_exception = future.exception()
        if service_exception is not None:
            raise RuntimeError(
                f"SetEntityPose service call failed for {entity_name}: "
                f"{service_exception}"
            ) from service_exception

        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"Gazebo rejected the pose requested for '{entity_name}'."
            )

    def _publish_velocity_command(
        self,
        linear_velocity_mps: float,
        angular_velocity_radps: float,
    ) -> None:
        """Publish one already validated differential-drive command."""

        if not np.isfinite(
            [linear_velocity_mps, angular_velocity_radps]
        ).all():
            raise ValueError(
                "Velocity command must contain finite values."
            )

        message = Twist()
        message.linear.x = float(linear_velocity_mps)
        message.angular.z = float(angular_velocity_radps)

        self.cmd_vel_publisher.publish(message)

    def _read_state(self, scan, odom):
        if self.goal_xy is None:
            raise RuntimeError("No active goal.")
        sectors, nearest, front = scan_sectors(scan.ranges, float(scan.angle_min),
                                               float(scan.angle_increment), float(scan.range_min))
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        x, y = float(p.x), float(p.y)
        yaw = yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))
        obs = build_observation(sectors, x, y, yaw, self.goal_xy)
        info = {"robot_x": x, "robot_y": y, "robot_yaw": yaw,
                "nearest_obstacle_m": nearest, "front_obstacle_m": front,
                "measured_linear_velocity_mps": float(odom.twist.twist.linear.x),
                "measured_angular_velocity_radps": float(odom.twist.twist.angular.z)}
        if not np.isfinite(list(info.values())).all():
            raise RuntimeError("Nonfinite odometry or range measurement.")
        return obs, info

    def reset_task(self, task):
        self._episode_active = False
        last_error = None
        for attempt in range(1, MAX_RESET_RETRIES + 1):
            try:
                self.pause_simulation()
                self.stop()
                self._set_entity_pose(ROBOT_NAME, 0., -6., ROBOT_RESET_Z_M, 0.)
                for index, name in enumerate(BOX_NAMES):
                    self._set_entity_pose(name, *task.layout.get(name, parked_box_pose(index)))
                self.goal_xy = tuple(task.goal)
                self._set_entity_pose(GOAL_MARKER_NAME, *task.goal, GOAL_MARKER_Z_M, 0.)
                self._set_entity_pose(ROBOT_NAME, *task.start[:2], ROBOT_RESET_Z_M, task.start[2])
                self.stop()
                scan, odom = self._advance_simulation(0.40)
                obs, info = self._read_state(scan, odom)
                if math.dist((info["robot_x"], info["robot_y"]), task.start[:2]) > RESET_POSITION_TOLERANCE_M:
                    raise RuntimeError("Robot did not reach its requested reset position.")
                if abs(wrap_to_pi(info["robot_yaw"] - task.start[2])) > RESET_YAW_TOLERANCE_RAD:
                    raise RuntimeError("Robot did not reach its requested reset yaw.")
                if info["nearest_obstacle_m"] <= COLLISION_DISTANCE_M:
                    raise RuntimeError("Reset began inside the collision margin.")
                if abs(info["measured_linear_velocity_mps"]) > .03 or abs(info["measured_angular_velocity_radps"]) > .08:
                    raise RuntimeError("Robot velocity did not settle after reset.")
                self._episode_active = True
                return obs, {**info, "reset_attempt": attempt, "control_mode": CONTROL_MODE}
            except Exception as error:
                last_error = error
                self.get_logger().warning(f"Reset {attempt}/{MAX_RESET_RETRIES}: {error}")
        raise RuntimeError(f"Reset failed for task seed {task.seed}.") from last_error

    def transition(self, linear, angular):
        if not self._episode_active:
            raise RuntimeError("reset_task() must succeed before transition().")
        try:
            self._publish_velocity_command(linear, angular)
            # Deliver the command while physics remains paused.
            time.sleep(.002)
            scan, odom = self._advance_simulation(CONTROL_INTERVAL_SEC)
            obs, info = self._read_state(scan, odom)
            return obs, {**info, **self._last_step_timing}
        except Exception:
            self.stop()
            self._episode_active = False
            try:
                if self.context.ok():
                    self.pause_simulation()
            except Exception:
                pass
            raise

    def stop(self):
        # SIGINT must not attempt publication into a dead rclpy context.
        try:
            if self.context.ok():
                self.cmd_vel_publisher.publish(Twist())
        except Exception:
            if self.context.ok():
                raise

    def publish_stop(self):
        self.stop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._episode_active = False
        try:
            self.stop()
            try:
                if self.context.ok():
                    self.pause_simulation()
            except Exception:
                pass
        finally:
            try:
                self._executor.shutdown(timeout_sec=1.)
            finally:
                self._executor_thread.join(timeout=1.)
                self.destroy_node()
