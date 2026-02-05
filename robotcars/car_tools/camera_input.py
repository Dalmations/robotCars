# car_tools/cameria_input.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from vilib import Vilib  # SunFounder vision library

from model import Observations, Obstacle, Pose


@dataclass
class CameraConfig:
    vflip: bool = False
    hflip: bool = False

    # Optional live stream / local preview (SunFounder recommends web=True for headless)
    display_local: bool = False
    display_web: bool = True

    # First usable “obstacle detection”: color blob detection
    obstacle_color: str = "red"  # red/orange/yellow/green/blue/purple or "close"

    # Placeholder projection from camera detection -> grid obstacle in front of car
    forward_distance: float = 4.0  # grid units ahead of the car
    obstacle_radius: float = 2.0   # grid units

    # Debounce so we don't spam a new obstacle_id every frame
    min_seconds_between_reports: float = 0.5


class PiCarXCamera:
    """
    Thin wrapper around SunFounder Vilib to:
      - start camera
      - enable detection
      - poll detection results into Observations

    Uses Vilib.camera_start(), Vilib.display(), Vilib.color_detect(), and reads
    Vilib.detect_obj_parameter (as in SunFounder examples). :contentReference[oaicite:1]{index=1}
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
        Vilib.color_detect(self.cfg.obstacle_color)  # "close" disables

        self._started = True

    def stop(self) -> None:
        # Vilib has camera_close() per docs. :contentReference[oaicite:2]{index=2}
        if not self._started:
            return
        try:
            Vilib.color_detect("close")
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

        Later step: replace this with calibration / depth / multi-view fusion to place obstacles
        in correct grid positions.
        """
        now = time.time()

        # SunFounder keys for color detection include: color_n, color_x, color_y, color_w, color_h :contentReference[oaicite:3]{index=3}
        color_n = int(Vilib.detect_obj_parameter.get("color_n", 0) or 0)

        if color_n <= 0:
            return Observations(obstacles=[])

        if (now - self._last_report_t) < self.cfg.min_seconds_between_reports:
            return Observations(obstacles=[])

        self._last_report_t = now
        self._seq += 1

        # Placeholder world placement: forward along car heading (theta)
        ox = car_pose.x + self.cfg.forward_distance * __import__("math").cos(car_pose.theta)
        oy = car_pose.y + self.cfg.forward_distance * __import__("math").sin(car_pose.theta)

        ob = Obstacle(
            obstacle_id=f"cam_{car_id}_{self._seq}",
            x=float(ox),
            y=float(oy),
            radius=float(self.cfg.obstacle_radius),
            is_moving=True,  # conservative default; we can classify later
        )
        return Observations(obstacles=[ob])
