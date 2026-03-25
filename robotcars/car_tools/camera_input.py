from __future__ import annotations

import math
from typing import Any, Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class OpenCvFrameProvider:
    """
    Small OpenCV-backed frame source for test-drive builds.

    Returns BGR frames, or None when the camera is unavailable.
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
