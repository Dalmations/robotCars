from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from vilib import Vilib

from model import Observations, Obstacle, Pose


@dataclass
class CameraConfig:
    vflip: bool = False
    hflip: bool = False
    display_local: bool = False
    display_web: bool = True

    obstacle_color: str = "red"

    frame_size: Tuple[int, int] = (640, 480)

    forward_distance: float = 4.0
    obstacle_radius: float = 2.0

    min_seconds_between_reports: float = 0.5

    # avoid the imencode crash by waiting for first frame
    startup_wait_seconds: float = 2.5


class PiCarXCamera:
    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()
        self._started = False
        self._last_report_t = 0.0
        self._seq = 0

    def start(self) -> None:
        if self._started:
            return

        Vilib.camera_start(vflip=self.cfg.vflip, hflip=self.cfg.hflip, size=self.cfg.frame_size)
        Vilib.color_detect(self.cfg.obstacle_color)

        # Wait until Vilib.img is a real numpy frame before enabling display/web
        t0 = time.time()
        while True:
            img = getattr(Vilib, "img", None)
            if isinstance(img, np.ndarray) and img.size > 0:
                # ensure flask_img is also a numpy array before web streaming thread starts
                try:
                    Vilib.flask_img = img
                except Exception:
                    pass
                break

            if time.time() - t0 > self.cfg.startup_wait_seconds:
                break
            time.sleep(0.05)

        # Now it's safe to start local/web display
        Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)

        self._started = True

    def stop(self) -> None:
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

    def read(self) -> Optional[np.ndarray]:
        """
        Returns a BGR/RGB frame from Vilib
        """
        if not self._started:
            return None

        img = getattr(Vilib, "img", None)
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None

        return img.copy()

    def poll_observations(self, car_pose: Pose, car_id: int = 0) -> Observations:
        now = time.time()

        color_n = int(Vilib.detect_obj_parameter.get("color_n", 0) or 0)
        if color_n <= 0:
            return Observations(obstacles=[])

        if (now - self._last_report_t) < self.cfg.min_seconds_between_reports:
            return Observations(obstacles=[])

        self._last_report_t = now
        self._seq += 1

        ox = car_pose.x + self.cfg.forward_distance * math.cos(car_pose.theta)
        oy = car_pose.y + self.cfg.forward_distance * math.sin(car_pose.theta)

        ob = Obstacle(
            obstacle_id=f"cam_{car_id}_{self._seq}",
            x=float(ox),
            y=float(oy),
            radius=float(self.cfg.obstacle_radius),
            is_moving=True,
        )
        return Observations(obstacles=[ob])