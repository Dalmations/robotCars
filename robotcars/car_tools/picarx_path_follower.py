from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from model import Path, Pose
from car_tools.motor_controller import MotorController
from coordination.shared_map import SharedMap


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def estimate_step_cells_for_duration(
    duration_s: float,
    *,
    step_seconds: float,
    speed: int,
    speed_ref: int,
) -> float:
    base_step = float(duration_s) / float(max(1e-6, step_seconds))
    speed_ref = max(1.0, float(speed_ref))
    speed_cmd = float(max(0, min(100, int(speed))))
    return base_step * (speed_cmd / speed_ref)


def estimate_ackermann_yaw_delta(step_cells: float, steer_deg: float, wheelbase: float) -> float:
    wheelbase = max(1e-6, float(wheelbase))
    steer_rad = math.radians(float(steer_deg))
    return float(step_cells) * math.tan(steer_rad) / wheelbase


def integrate_dead_reckoning(
    *,
    shared_map: SharedMap,
    car_id: int,
    forward_step: float,
    yaw_delta: float,
) -> Pose:
    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is None:
        pose_world = Pose(0.0, 0.0, 0.0)

    step = float(forward_step)
    dtheta = float(yaw_delta)
    theta_mid = float(pose_world.theta) + 0.5 * dtheta

    next_pose = Pose(
        x=float(pose_world.x + step * math.cos(theta_mid)),
        y=float(pose_world.y + step * math.sin(theta_mid)),
        theta=float(wrap_angle(float(pose_world.theta) + dtheta)),
    )
    shared_map.set_pose(car_id, next_pose)
    return next_pose

@dataclass
class FollowerConfig:
    lookahead: float = 6.0
    wheelbase: float = 5.0
    odom_wheelbase: Optional[float] = None
    pivot_odom_wheelbase: Optional[float] = None
    goal_tolerance: float = 2.0

    steer_sign: float = 1.0
    max_steer_deg: float = 35.0

    steer_alpha: float = 0.25
    steer_deadband_deg: float = 2.0
    steer_rate_limit_deg_per_tick: float = 12.0
    heading_lookahead_gain: float = 0.012
    heading_lookahead_max_scale: float = 1.75

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

    def sync_to_motor_steering(self) -> None:
        self._filtered_steer_deg = float(self.motor.get_applied_steering_deg())

    def odom_wheelbase(self) -> float:
        # Dead-reckoning usually needs a calibrated "effective" wheelbase that
        # differs a bit from the steering model wheelbase.
        if self.cfg.odom_wheelbase is not None:
            return max(1e-6, float(self.cfg.odom_wheelbase))
        return max(1e-6, float(self.cfg.wheelbase))

    def odom_wheelbase_for_mode(self, drive_mode: str) -> float:
        if drive_mode in {"pivot_turn", "escape_pivot"} and self.cfg.pivot_odom_wheelbase is not None:
            return max(1e-6, float(self.cfg.pivot_odom_wheelbase))
        return self.odom_wheelbase()

    def _compute_lookahead(self, goal_distance: float) -> float:
        lookahead = float(self.cfg.lookahead)
        if goal_distance < self.cfg.dock_distance_grid:
            return max(self.cfg.dock_min_lookahead_grid, min(lookahead, 0.8 * goal_distance))
        return lookahead

    def _lookahead_point(self, path: Path, pose: Pose, lookahead_distance: float) -> Tuple[float, float]:
        waypoints = path.waypoints
        if not waypoints:
            return (pose.x, pose.y)
        if len(waypoints) == 1:
            return (waypoints[0].x, waypoints[0].y)

        px, py = pose.x, pose.y
        best_dist2 = float("inf")
        best_i = 0
        best_proj = (waypoints[0].x, waypoints[0].y)

        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i].x, waypoints[i].y
            x1, y1 = waypoints[i + 1].x, waypoints[i + 1].y
            dx = x1 - x0
            dy = y1 - y0
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue

            u = ((px - x0) * dx + (py - y0) * dy) / seg2
            u = max(0.0, min(1.0, float(u)))
            proj = (x0 + u * dx, y0 + u * dy)
            dist2 = (proj[0] - px) ** 2 + (proj[1] - py) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_proj = proj

        dist_acc = 0.0
        start_x, start_y = best_proj
        for i in range(best_i, len(waypoints) - 1):
            x0, y0 = (start_x, start_y) if i == best_i else (waypoints[i].x, waypoints[i].y)
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

    def _pure_pursuit_delta(self, pose: Pose, target_x: float, target_y: float) -> float:
        path_angle = math.atan2(target_y - pose.y, target_x - pose.x)
        alpha = wrap_angle(path_angle - pose.theta)

        lookahead_distance = max(1e-6, self._dist(pose.x, pose.y, target_x, target_y))
        wheelbase = max(1e-6, float(self.cfg.wheelbase))

        return math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead_distance)

    def tracking_geometry(self, path: Path, pose: Pose, goal_distance: float) -> Tuple[float, float, float]:
        lookahead = self._compute_lookahead(goal_distance)
        target_x, target_y = self._lookahead_point(path, pose, lookahead)
        heading_error_rad = wrap_angle(math.atan2(target_y - pose.y, target_x - pose.x) - pose.theta)
        heading_error_deg = math.degrees(heading_error_rad)

        if abs(heading_error_deg) > 20.0:
            if goal_distance > self.cfg.dock_distance_grid:
                lookahead_scale = min(
                    float(self.cfg.heading_lookahead_max_scale),
                    1.0 + float(self.cfg.heading_lookahead_gain) * (abs(heading_error_deg) - 20.0),
                )
                adjusted_lookahead = lookahead * lookahead_scale
            else:
                tighten_scale = max(0.55, 1.0 - 0.004 * (abs(heading_error_deg) - 20.0))
                adjusted_lookahead = max(float(self.cfg.dock_min_lookahead_grid), lookahead * tighten_scale)
            target_x, target_y = self._lookahead_point(path, pose, adjusted_lookahead)
            heading_error_rad = wrap_angle(math.atan2(target_y - pose.y, target_x - pose.x) - pose.theta)
            heading_error_deg = math.degrees(heading_error_rad)
        return target_x, target_y, heading_error_deg

    def tracking_command(
        self,
        path: Path,
        pose: Pose,
        goal_distance: float,
        *,
        steer_cap_deg: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Return target x/y, heading error in degrees, and filtered steer in degrees."""
        target_x, target_y, heading_error_deg = self.tracking_geometry(path, pose, goal_distance)
        delta = self._pure_pursuit_delta(pose, target_x=target_x, target_y=target_y)
        target_steer_deg = self._clamp_steer(math.degrees(delta) * float(self.cfg.steer_sign))
        if steer_cap_deg is not None:
            steer_cap = min(abs(float(steer_cap_deg)), float(self.cfg.max_steer_deg))
            target_steer_deg = max(-steer_cap, min(steer_cap, target_steer_deg))
            filtered = self._filter_steering(target_steer_deg)
            filtered = max(-steer_cap, min(steer_cap, filtered))
            self._filtered_steer_deg = filtered
            return (
                target_x,
                target_y,
                heading_error_deg,
                filtered,
            )
        return (
            target_x,
            target_y,
            heading_error_deg,
            self._filter_steering(target_steer_deg),
        )

    def _filter_steering(self, target_deg: float) -> float:
        if abs(target_deg) < self.cfg.steer_deadband_deg:
            target_deg = 0.0

        blended = (1.0 - self.cfg.steer_alpha) * self._filtered_steer_deg + self.cfg.steer_alpha * target_deg
        delta = blended - self._filtered_steer_deg
        delta = max(-self.cfg.steer_rate_limit_deg_per_tick, min(self.cfg.steer_rate_limit_deg_per_tick, delta))

        self._filtered_steer_deg = self._clamp_steer(self._filtered_steer_deg + delta)
        return self._filtered_steer_deg

    def _clamp_steer(self, steer_deg: float) -> float:
        return max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, steer_deg))

    @staticmethod
    def _dist(x0: float, y0: float, x1: float, y1: float) -> float:
        return float(math.hypot(x1 - x0, y1 - y0))
