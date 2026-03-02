# car_tools/picarx_path_follower.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np

from model import Path, Pose
from car_tools.motor_controller import MotorController
from coordination.shared_map import SharedMap


@dataclass
class FollowerConfig:
    dt: float = 0.10                 # control loop seconds
    lookahead: float = 6.0           # grid units (tune)
    wheelbase: float = 5.0           # grid units (effective; tune)
    goal_tolerance: float = 2.0      # grid units
    max_run_seconds: float = 45.0

    # Alignment knobs
    pose_frame: str = "grid"         # grid for A* paths, world in SLAM/world coords
    steer_sign: float = 1.0          # set -1.0 if steering is mirrored
    max_steer_deg: float = 35.0      # PiCar-X examples use ~35 deg


class PathFollower:
    def __init__(self, motor: MotorController, cfg: Optional[FollowerConfig] = None):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()

    def follow_with_slam(
        self,
        path: Path,
        shared_map: SharedMap,
        car_id: int = 0,
        camera: Any = None,
        slam_detector: Any = None,
    ) -> None:
        """
        Closed-loop pure pursuit using pose from SLAM (via shared_map).
        """
        if len(path.waypoints) < 2:
            return

        t0 = time.time()
        last_pose_world: Optional[Pose] = None

        try:
            while True:
                if (time.time() - t0) > self.cfg.max_run_seconds:
                    break

                # --- Update pose from SLAM ---
                if camera is not None and slam_detector is not None:
                    frame = self._grab_frame(camera)
                    if frame is not None:
                        pose = slam_detector.tick(frame)
                        if pose is not None:
                            last_pose_world = pose  # slam publishes into shared_map too

                # Get pose in the frame we want to drive in
                pose = shared_map.get_pose(car_id, frame=self.cfg.pose_frame)  # type: ignore[arg-type]
                if pose is None and last_pose_world is not None:
                    # Fallback: if shared_map not updated for some reason
                    if self.cfg.pose_frame == "world":
                        pose = last_pose_world
                    else:
                        # Fallback world->grid
                        if hasattr(shared_map, "world_to_grid_f"):
                            gx, gy = shared_map.world_to_grid_f(last_pose_world.x, last_pose_world.y)  # type: ignore[attr-defined]
                            pose = Pose(x=float(gx), y=float(gy), theta=float(last_pose_world.theta))

                if pose is None:
                    time.sleep(self.cfg.dt)
                    continue

                # --- Goal check ---
                goal = path.waypoints[-1]
                if self._dist(pose.x, pose.y, goal.x, goal.y) <= self.cfg.goal_tolerance:
                    self.motor.mark_reached()
                    break

                # --- Pure Pursuit target ---
                tx, ty = self._lookahead_point(path, pose, self.cfg.lookahead)
                delta_rad = self._pure_pursuit_delta(pose, target_x=tx, target_y=ty)

                steer_deg = float(np.rad2deg(delta_rad))
                steer_deg = float(np.clip(self.cfg.steer_sign * steer_deg, -self.cfg.max_steer_deg, self.cfg.max_steer_deg))

                self.motor.set_steering(steer_deg)
                self.motor.forward_for(self.cfg.dt)

        finally:
            self.motor.stop()

    # ---------------- Camera helpers ----------------

    @staticmethod
    def _grab_frame(camera: Any):
        if hasattr(camera, "read"):
            out = camera.read()
            if isinstance(out, tuple) and len(out) == 2:
                ret, frame = out
                return frame if ret else None
            return out
        if hasattr(camera, "capture_array"):
            return camera.capture_array()
        return None

    # ---------------- Pure Pursuit ----------------

    def _lookahead_point(self, path: Path, pose: Pose, Ld: float) -> Tuple[float, float]:
        """
        Pick a lookahead point starting from the car's closest waypoint index
        and move forward until distance >= Ld.
        """
        wps = path.waypoints
        if not wps:
            return (pose.x, pose.y)
        if len(wps) == 1:
            return (wps[0].x, wps[0].y)

        # Find closest waypoint index
        px, py = pose.x, pose.y
        d2 = [(wp.x - px) ** 2 + (wp.y - py) ** 2 for wp in wps]
        i0 = int(np.argmin(d2))

        # Walk forward accumulating arc-length
        dist_acc = 0.0
        for i in range(i0, len(wps) - 1):
            x0, y0 = wps[i].x, wps[i].y
            x1, y1 = wps[i + 1].x, wps[i + 1].y
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-9:
                continue

            if dist_acc + seg >= Ld:
                # Interpolate within this segment
                remaining = Ld - dist_acc
                u = float(remaining / seg)
                tx = x0 + u * (x1 - x0)
                ty = y0 + u * (y1 - y0)
                return (tx, ty)

            dist_acc += seg

        # If path shorter than lookahead, use final waypoint
        return (wps[-1].x, wps[-1].y)

    def _pure_pursuit_delta(self, pose: Pose, target_x: float, target_y: float) -> float:
        dx = target_x - pose.x
        dy = target_y - pose.y

        path_angle = math.atan2(dy, dx)
        alpha = self._wrap_angle(path_angle - pose.theta)

        Ld = max(1e-6, self._dist(pose.x, pose.y, target_x, target_y))
        L = max(1e-6, float(self.cfg.wheelbase))

        return math.atan2(2.0 * L * math.sin(alpha), Ld)

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