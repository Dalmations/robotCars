# car_tools/picarx_path_follower.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

from model import Path, TargetPoint
from car_tools.motor_controller import MotorController
from typing import Callable, Optional


TickCallback = Callable[[float, float, float], None]  # x, y, yaw (grid units, radians) Callback for PurePursuit

@dataclass
class FollowerConfig:
    """
    Pure Pursuit parameters in *grid units* (NOT meters).

    - lookahead: how far ahead (in grid units) to chase a point
    - wheelbase: effective wheelbase in grid units (tune)
    - v: estimated speed in grid units / second (tune)
    - dt: control tick duration in seconds
    """
    lookahead: float = 3.0              # grid units
    wheelbase: float = 2.0              # grid units (effective)
    v: float = 2.0                      # grid units per second (effective)
    dt: float = 0.10                    # seconds per control update

    max_steer_deg: float = 35.0         # clamp to match MotorConfig.max_steer_deg
    goal_tolerance: float = 1.0         # grid units to final waypoint
    speed_cmd: int = 80                 # motor_controller speed (0..100-ish)


class Pose2D:
    __slots__ = ("x", "y", "yaw")

    def __init__(self, x: float, y: float, yaw: float):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)  # radians

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)


class PurePursuitFollower:
    """
    Matlab-style Pure Pursuit in grid coordinates.

    IMPORTANT: This currently uses dead-reckoning pose updates based on an
    *estimated* v (grid units/sec) and the commanded steering angle.
    For real accuracy, later we will replace pose updates with camera/odometry.
    """

    def __init__(self, motor: MotorController, cfg: Optional[FollowerConfig] = None):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()
        

    # def follow(self, path: Path, start_pose: Optional[Pose2D] = None) -> None:
    def follow(self, path: Path, start_pose: Optional[Pose2D] = None, on_tick: Optional[TickCallback] = None) -> None:
        wps = path.waypoints
        if len(wps) < 2:
            return

        # Waypoints as (x, y) in grid units
        pts: List[Tuple[float, float]] = [(wp.x, wp.y) for wp in wps]
        goal = pts[-1]

        # Init pose
        if start_pose is None:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            yaw0 = math.atan2(y1 - y0, x1 - x0)
            pose = Pose2D(x0, y0, yaw0)
        else:
            pose = start_pose.copy()

        # Configure speed (open loop)
        self.motor.set_speed(self.cfg.speed_cmd)

        try:
            while True:
                if on_tick is not None:
                    on_tick(pose.x, pose.y, pose.yaw)
                if self._dist((pose.x, pose.y), goal) <= self.cfg.goal_tolerance:
                    break

                target = self._lookahead_point(pose, pts, self.cfg.lookahead)
                steer_deg = self._pure_pursuit_steer_deg(pose, target)

                # Clamp to safe steering
                steer_deg = max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, steer_deg))

                # Command hardware
                self.motor.set_steering(steer_deg)
                self.motor.forward_for(self.cfg.dt)

                # Dead-reckoning pose update (grid bicycle model)
                pose = self._update_pose(pose, steer_deg, self.cfg.v, self.cfg.wheelbase, self.cfg.dt)

        finally:
            self.motor.set_steering(0.0)
            self.motor.stop()
            self.motor.mark_reached()

    # -------------------------
    # Pure Pursuit math
    # -------------------------
    def _pure_pursuit_steer_deg(self, pose: Pose2D, target: Tuple[float, float]) -> float:
        """
        Pure Pursuit steering:
          alpha = angle between heading and target direction
          delta = atan(2L*sin(alpha)/Ld)

        Returns steering in degrees (for PiCar-X servo).
        """
        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y

        # Target heading in global
        target_heading = math.atan2(dy, dx)
        alpha = self._wrap_angle(target_heading - pose.yaw)

        Ld = max(1e-6, math.hypot(dx, dy))
        delta_rad = math.atan2(2.0 * self.cfg.wheelbase * math.sin(alpha), Ld)
        return math.degrees(delta_rad)

    def _lookahead_point(
        self,
        pose: Pose2D,
        pts: List[Tuple[float, float]],
        lookahead: float,
    ) -> Tuple[float, float]:
        """
        Pick the first waypoint at least lookahead distance from the current pose.
        If none, use final goal.
        """
        px, py = pose.x, pose.y
        for x, y in pts:
            if math.hypot(x - px, y - py) >= lookahead:
                return (x, y)
        return pts[-1]

    def _update_pose(self, pose: Pose2D, steer_deg: float, v: float, L: float, dt: float) -> Pose2D:
        """
        Bicycle model update in grid units:
          x += v cos(yaw) dt
          y += v sin(yaw) dt
          yaw += v/L * tan(delta) dt
        """
        delta = math.radians(steer_deg)
        x = pose.x + v * math.cos(pose.yaw) * dt
        y = pose.y + v * math.sin(pose.yaw) * dt
        yaw = pose.yaw + (v / max(1e-6, L)) * math.tan(delta) * dt
        return Pose2D(x, y, self._wrap_angle(yaw))

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a
