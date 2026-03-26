from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from vilib import Vilib
except Exception:  # pragma: no cover
    Vilib = None


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
    Vilib-backed camera provider for the PiCar-X runtime.

    The test-drive build uses this path first because it matches the robot's
    native camera stack and avoids the OpenCV/GStreamer allocation issues that
    can happen with direct VideoCapture startup on the Pi.
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()
        self._started = False

    def start(self) -> None:
        if self._started or Vilib is None or np is None:
            return

        try:
            Vilib.camera_start(size=self.cfg.frame_size)
            controls: Dict[str, Any] = {"FrameRate": int(self.cfg.frame_rate)}
            if self.cfg.camera_controls:
                controls.update(self.cfg.camera_controls)
            try:
                Vilib.set_controls(controls)
            except Exception:
                pass

            self._wait_for_first_frame()
            try:
                Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)
            except Exception:
                pass
            self._started = True
        except Exception:
            self._started = False

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
        if not self._started or Vilib is None:
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

    def read(self) -> Optional[Any]:
        if not self._started or Vilib is None or np is None:
            return None

        img = getattr(Vilib, "img", None)
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None

        frame = np.array(img, copy=True)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def is_opened(self) -> bool:
        return self._started

    def get_frame(self) -> Optional[Any]:
        return self.read()

    def frame_size(self) -> tuple[int, int]:
        width, height = self.cfg.frame_size
        return (max(1, int(width)), max(1, int(height)))

    def release(self) -> None:
        self.stop()


class OpenCvFrameProvider:
    """
    Small OpenCV-backed frame source for test-drive builds.

    Returns BGR frames, or None when the camera is unavailable.
    This remains as a fallback for non-Pi environments where Vilib is missing.
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
    ):
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self._cap = None

        if cv2 is None:
            return

        cap = cv2.VideoCapture(self.device_index)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        self._cap = cap

    def is_opened(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def get_frame(self) -> Optional[Any]:
        if not self.is_opened():
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def frame_size(self) -> tuple[int, int]:
        if not self.is_opened():
            return (self.width, self.height)

        width = int(round(float(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        return (
            max(1, width if width > 0 else self.width),
            max(1, height if height > 0 else self.height),
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def read_ultrasonic_cm(motor: Any) -> Optional[float]:
    """
    Return ultrasonic distance in cm, or None for invalid/unavailable readings.
    """
    px = getattr(motor, "px", None)
    if px is None:
        return None

    try:
        d = float(px.get_distance())
    except Exception:
        return None

    if not math.isfinite(d) or d <= 0.0 or d > 500.0:
        return None
    return d
