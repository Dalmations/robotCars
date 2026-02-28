# car_tools/picarx_path_follower.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

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


class PathFollower:
    def __init__(self, motor: MotorController, cfg: Optional[FollowerConfig] = None):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()

    def follow(self, path: Path) -> None:
        # point-to-point headings
        wps = path.waypoints
        if len(wps) < 2:
            return
        for i in range(1, len(wps)):
            prev = wps[i - 1]
            cur = wps[i]
            dx = cur.x - prev.x
            dy = cur.y - prev.y
            if dx == 0 and dy == 0:
                continue
            heading = math.atan2(dy, dx)
            steer_deg = float(np.rad2deg(heading))
            self.motor.set_steering(steer_deg)
            self.motor.step_forward()
        self.motor.mark_reached()

    def follow_with_slam(
        self,
        path: Path,
        shared_map: SharedMap,
        car_id: int = 0,
        camera=None,
        slam_detector=None,
    ) -> None:
        if len(path.waypoints) < 2:
            return

        t0 = time.time()
        last_pose: Optional[Pose] = None

        try:
            while True:
                if (time.time() - t0) > self.cfg.max_run_seconds:
                    break

                # Update pose from SLAM
                if camera is not None and slam_detector is not None:
                    frame = camera.read()
                    if frame is not None:
                        pose = slam_detector.tick(frame)
                        if pose is not None:
                            last_pose = pose

                pose = shared_map.poses.get(car_id) or last_pose
                if pose is None:
                    time.sleep(self.cfg.dt)
                    continue

                # Stop if near goal
                goal = path.waypoints[-1]
                if self._dist(pose.x, pose.y, goal.x, goal.y) <= self.cfg.goal_tolerance:
                    self.motor.mark_reached()
                    break

                # Pure Pursuit target
                target = self._lookahead_point(path, pose, self.cfg.lookahead)
                delta_rad = self._pure_pursuit_delta(pose, target_x=target[0], target_y=target[1])

                steer_deg = float(np.rad2deg(delta_rad))
                self.motor.set_steering(steer_deg)
                self.motor.forward_for(self.cfg.dt)

        finally:
            self.motor.stop()


    # Pure Pursuit 
    def _lookahead_point(self, path: Path, pose: Pose, Ld: float) -> tuple[float, float]:
        x, y = pose.x, pose.y
        best = (path.waypoints[-1].x, path.waypoints[-1].y)

        # Find the first waypoint at least Ld away; if none, use last waypoint
        for wp in path.waypoints:
            if self._dist(x, y, wp.x, wp.y) >= Ld:
                return (wp.x, wp.y)
        return best

    def _pure_pursuit_delta(self, pose: Pose, target_x: float, target_y: float) -> float:
        dx = target_x - pose.x
        dy = target_y - pose.y

        path_angle = math.atan2(dy, dx)
        alpha = self._wrap_angle(path_angle - pose.theta)

        Ld = max(1e-6, self._dist(pose.x, pose.y, target_x, target_y))
        L = max(1e-6, float(self.cfg.wheelbase))

        # Bicycle pure pursuit:
        # delta = atan2(2*L*sin(alpha), Ld)
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