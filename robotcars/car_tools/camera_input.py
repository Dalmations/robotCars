from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np
import cv2
from vilib import Vilib


@dataclass
class CameraConfig:
    display_local: bool = False
    display_web: bool = False

    frame_size: Tuple[int, int] = (640, 480)
    frame_rate: int = 30
    startup_wait_seconds: float = 2.5

    camera_controls: Optional[Dict[str, Any]] = None


class PiCarXCamera:
    """
    Vilib read() to capture frames, and handed to OpenCV for SLAM.
    Warms up camera on start.
    Includes ultrasonic read.
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        Vilib.camera_start(size=self.cfg.frame_size)

        controls: Dict[str, Any] = {"FrameRate": int(self.cfg.frame_rate)}
        if self.cfg.camera_controls:
            controls.update(self.cfg.camera_controls)
        try:
            Vilib.set_controls(controls)
        except Exception:
            pass

        self._wait_for_first_frame()
        Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)
        self._started = True

    def _wait_for_first_frame(self) -> None:
        t0 = time.time()
        while True:
            img = getattr(Vilib, "img", None)
            if isinstance(img, np.ndarray) and img.size > 0:
                try:
                    Vilib.flask_img = img
                except Exception:
                    pass
                return

            if time.time() - t0 > self.cfg.startup_wait_seconds:
                return
            time.sleep(0.05)

    def stop(self) -> None:
        if not self._started:
            return
        try:
            Vilib.imshow_flag = False
            Vilib.web_display_flag = False
        except Exception:
            pass

        try:
            Vilib.camera_close()
        except Exception:
            pass

        self._started = False

    def read(self) -> Optional[np.ndarray]:
        """
        Returns a copied frame.
        """
        if not self._started:
            return None

        img = getattr(Vilib, "img", None)
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None

        frame = np.array(img, copy=True)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        return frame

def read_ultrasonic_cm(motor: Any) -> Optional[float]:
    """
    Return ultrasonic distance in cm, or None for invalid/unavailable readings.
    """
    px = getattr(motor, "px", None)
    if px is None:
        return None

    d = float(px.get_distance())
    if not np.isfinite(d) or d <= 0.0 or d > 500.0:
        return None
    return d


def ultrasonic_to_countdown(dist_cm: Optional[float]) -> int:
    """
    Map ultrasonic distance to countdown levels:
      d > 40cm  -> 30
      d <= 40cm -> 10
      d < 20cm  -> 1
    """
    if dist_cm is None:
        return 30

    d = float(dist_cm)
    if d < 20.0:
        return 1
    if d <= 40.0:
        return 10
    return 30