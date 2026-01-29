# picarx_path_follower.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from model import Path, TargetPoint
from car_tools.motor_controller import MotorController


@dataclass
class FollowerConfig:
    """
    Maps grid waypoint deltas to steering angles.
    With no odometry, we assume:
      - each waypoint-to-waypoint move is one step
      - we steer toward the delta direction
    """
    # angle = atan2(dy, dx) mapped into steering range.
    # <Tune>
    heading_to_steer_gain: float = 5  # degrees of steering per radian of heading


class PathFollower:
    def __init__(self, motor: MotorController, cfg: FollowerConfig | None = None):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()

    def follow(self, path: Path) -> None:
        """
        Drives the car along path.waypoints in order.
        """
        wps = path.waypoints
        if len(wps) < 2:
            return

        # Iterate segments
        for i in range(1, len(wps)):
            prev = wps[i - 1]
            cur = wps[i]
            dx = cur.x - prev.x
            dy = cur.y - prev.y

            # If duplicate waypoint, skip
            if dx == 0 and dy == 0:
                continue

            # Convert heading into a steering command
            heading = math.atan2(dy, dx)  # radians
            steer_deg = heading * self.cfg.heading_to_steer_gain
            self.motor.set_steering(steer_deg)
            self.motor.step_forward()

        # Stop and mark done
        # self.motor.set_steering(0.0)
        self.motor.mark_reached()
