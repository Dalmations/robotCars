from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from model import Path, Pose
from car_tools.motor_controller import MotorController
from coordination.shared_map import SharedMap


@dataclass
class FollowerConfig:
    dt: float = 0.10
    lookahead: float = 6.0
    wheelbase: float = 5.0
    goal_tolerance: float = 2.0
    max_run_seconds: float = 45.0

    pose_frame: str = "grid"
    steer_sign: float = 1.0
    max_steer_deg: float = 35.0

    steer_alpha: float = 0.25
    steer_deadband_deg: float = 2.0
    steer_rate_limit_deg_per_tick: float = 12.0

    dock_distance_grid: float = 12.0
    dock_min_lookahead_grid: float = 6.0


class PathFollower:
    def __init__(self, motor: MotorController, cfg: Optional[FollowerConfig] = None):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()
        self._filtered_steer_deg = 0.0

    def reset(self) -> None:
        self._filtered_steer_deg = 0.0
        if hasattr(self.motor, "reset_reached"):
            self.motor.reset_reached()

    def compute_lookahead(self, goal_distance: float) -> float:
        lookahead = float(self.cfg.lookahead)
        if goal_distance < self.cfg.dock_distance_grid:
            return max(self.cfg.dock_min_lookahead_grid, min(lookahead, 0.8 * goal_distance))
        return lookahead

    def lookahead_point(self, path: Path, pose: Pose, lookahead_distance: float) -> Tuple[float, float]:
        waypoints = path.waypoints
        if not waypoints:
            return (pose.x, pose.y)
        if len(waypoints) == 1:
            return (waypoints[0].x, waypoints[0].y)

        px, py = pose.x, pose.y
        d2 = [(wp.x - px) ** 2 + (wp.y - py) ** 2 for wp in waypoints]
        i0 = min(range(len(d2)), key=d2.__getitem__)

        dist_acc = 0.0
        for i in range(i0, len(waypoints) - 1):
            x0, y0 = waypoints[i].x, waypoints[i].y
            x1, y1 = waypoints[i + 1].x, waypoints[i + 1].y
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-9:
                continue

            if dist_acc + seg >= lookahead_distance:
                remaining = lookahead_distance - dist_acc
                u = float(remaining / seg)
                tx = x0 + u * (x1 - x0)
                ty = y0 + u * (y1 - y0)
                return (tx, ty)

            dist_acc += seg

        return (waypoints[-1].x, waypoints[-1].y)

    def pure_pursuit_delta(self, pose: Pose, target_x: float, target_y: float) -> float:
        dx = target_x - pose.x
        dy = target_y - pose.y

        path_angle = math.atan2(dy, dx)
        alpha = self._wrap_angle(path_angle - pose.theta)

        lookahead_distance = max(1e-6, self._dist(pose.x, pose.y, target_x, target_y))
        wheelbase = max(1e-6, float(self.cfg.wheelbase))

        return math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead_distance)

    def pure_pursuit_steer_deg(self, path: Path, pose: Pose, goal_distance: float) -> float:
        lookahead = self.compute_lookahead(goal_distance)
        tx, ty = self.lookahead_point(path, pose, lookahead)
        delta = self.pure_pursuit_delta(pose, target_x=tx, target_y=ty)
        steer_deg = math.degrees(delta) * float(self.cfg.steer_sign)
        return self._clamp_steer(steer_deg)

    def filter_steering(self, target_deg: float) -> float:
        if abs(target_deg) < self.cfg.steer_deadband_deg:
            target_deg = 0.0

        blended = (1.0 - self.cfg.steer_alpha) * self._filtered_steer_deg + self.cfg.steer_alpha * target_deg
        delta = blended - self._filtered_steer_deg
        delta = max(-self.cfg.steer_rate_limit_deg_per_tick, min(self.cfg.steer_rate_limit_deg_per_tick, delta))

        self._filtered_steer_deg = self._clamp_steer(self._filtered_steer_deg + delta)
        return self._filtered_steer_deg

    def steering_command(self, path: Path, pose: Pose, goal_distance: float) -> float:
        return self.filter_steering(self.pure_pursuit_steer_deg(path, pose, goal_distance))

    def _clamp_steer(self, steer_deg: float) -> float:
        return max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, steer_deg))

    @staticmethod
    def _dist(x0: float, y0: float, x1: float, y1: float) -> float:
        return float(math.hypot(x1 - x0, y1 - y0))

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a