# car_tools/cameria_input.py
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional

from vilib import Vilib  # SunFounder vision library

from model import Observations, Obstacle, Pose


@dataclass
class CameraConfig:
    vflip: bool = False
    hflip: bool = False
    display_local: bool = False
    display_web: bool = True

    # First usable “obstacle detection”: color blob detection
    obstacle_color: str = "red"

    # Placeholder projection from camera detection -> grid obstacle in front of car
    forward_distance: float = 4.0  # grid units ahead of the car
    obstacle_radius: float = 2.0   # grid units

    # Debounce new obstacle_id
    min_seconds_between_reports: float = 0.5


class PiCarXCamera:
    """
    SunFounder Vilib:
      - start camera
      - enable detection
      - poll detection results into Observations
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()
        self._started = False
        self._last_report_t = 0.0
        self._seq = 0

    def start(self) -> None:
        if self._started:
            return

        Vilib.camera_start(vflip=self.cfg.vflip, hflip=self.cfg.hflip)
        Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)
        Vilib.color_detect(self.cfg.obstacle_color)

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            Vilib.color_detect("close") # "close" disables
        except Exception:
            pass
        try:
            Vilib.camera_close()
        except Exception:
            pass
        self._started = False

    def poll_observations(self, car_pose: Pose, car_id: int = 0) -> Observations:
        """
        If the configured color is detected, create a synthetic obstacle in front of the car.

        TODO: replace this with calibration / depth / multi-view fusion to place obstacles
        in correct grid positions.
        """
        now = time.time()

        # SunFounder keys: color_n, color_x, color_y, color_w, color_h
        color_n = int(Vilib.detect_obj_parameter.get("color_n", 0) or 0)

        if color_n <= 0:
            return Observations(obstacles=[])

        if (now - self._last_report_t) < self.cfg.min_seconds_between_reports:
            return Observations(obstacles=[])

        self._last_report_t = now
        self._seq += 1

        # Place obstacle forward along car heading (theta)
        ox = car_pose.x + self.cfg.forward_distance * math.cos(car_pose.theta)
        oy = car_pose.y + self.cfg.forward_distance * math.sin(car_pose.theta)

        ob = Obstacle(
            obstacle_id=f"cam_{car_id}_{self._seq}",
            x=float(ox),
            y=float(oy),
            radius=float(self.cfg.obstacle_radius),
            is_moving=True,  # TODO: to classify moving or not
        )
        return Observations(obstacles=[ob])
