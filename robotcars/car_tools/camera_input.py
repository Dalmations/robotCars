from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class CameraStreamConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480


class OpenCVCameraStream:
    def __init__(self, cfg: Optional[CameraStreamConfig] = None):
        self.cfg = cfg or CameraStreamConfig()
        self.cap = None

        if cv2 is None:
            return

        try:
            self.cap = cv2.VideoCapture(int(self.cfg.device_index))
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.width))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.height))
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.release()

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.cap is None:
            return
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = None


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
