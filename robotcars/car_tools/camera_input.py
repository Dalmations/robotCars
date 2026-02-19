from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from vilib import Vilib  # SunFounder vision library

from model import Observations, Obstacle, Pose


@dataclass
class CameraConfig:
    # Vilib camera + streaming
    vflip: bool = False
    hflip: bool = False
    display_local: bool = False
    display_web: bool = True

    # Detection (easy first obstacle proxy)
    obstacle_color: str = "red"

    # Frame source for SLAM (prefer Picamera2 on Raspberry Pi)
    frame_size: Tuple[int, int] = (640, 480)
    use_picamera2: bool = True

    # Placeholder projection from detection -> grid obstacle in front of car
    forward_distance: float = 4.0
    obstacle_radius: float = 2.0

    # Debounce new obstacle_id
    min_seconds_between_reports: float = 0.5


class PiCarXCamera:
    """
    Combined camera stack:
      1) Vilib: starts camera + optional web stream + color detection parameters
      2) Frame stream: provides real BGR frames for SLAM (Picamera2 preferred, OpenCV fallback)

    Public API:
      - start()/stop()
      - read_bgr() -> np.ndarray | None
      - poll_observations(pose, car_id) -> Observations
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()

        self._started = False
        self._last_report_t = 0.0
        self._seq = 0

    # --------------------
    # Lifecycle
    # --------------------
    def start(self) -> None:
        if self._started:
            return

        # Start Vilib (detection + optional MJPG stream)
        Vilib.camera_start(vflip=self.cfg.vflip, hflip=self.cfg.hflip)
        Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)
        Vilib.color_detect(self.cfg.obstacle_color)

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        # Stop detection
        try:
            Vilib.color_detect("close")
        except Exception:
            pass

        # Stop Vilib camera
        try:
            Vilib.camera_close()
        except Exception:
            pass


        self._started = False

    def read(self) -> Optional[np.ndarray]:
        """
        Returns a frame for vSLAM / OpenCV processing.
        """
        if not self._started:
            return None

        img = Vilib.take_photo('test.jpg')
        img = cv2.imread('test.jpg')
        return img

    # --------------------
    # Vilib-based obstacle observations
    # --------------------
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

    # Optional: quick debug dump
    def debug_dump_detect_params(self) -> str:
        try:
            return json.dumps(Vilib.detect_obj_parameter)
        except Exception:
            return "{}"
