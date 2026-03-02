from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import cv2
from vilib import Vilib


ColorOrder = Literal["rgb", "bgr"]


@dataclass
class CameraConfig:
    display_local: bool = False
    display_web: bool = False

    frame_size: Tuple[int, int] = (640, 480)

    # Reduce capture FPS to lower CPU load
    # Vilib defaults preview_config.controls = {'FrameRate': 60}
    frame_rate: int = 30

    # Wait for first frame
    startup_wait_seconds: float = 2.5

    # Color-order handling:
    # Vilib often provides RGB; if your stream appears blue-tinted, set source_color_order="bgr".
    source_color_order: ColorOrder = "rgb"
    output_color_order: ColorOrder = "rgb"

    # Optional diagnostics
    debug_color_stats: bool = False
    debug_color_stats_period_s: float = 2.0


class PiCarXCamera:
    """
    Vilib is internally configured as RGB888 + capture_array(), so frames are RGB.
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg or CameraConfig()
        self._started = False
        self.color_order: ColorOrder = self.cfg.output_color_order
        self._last_color_print_t: float = 0.0

    def start(self) -> None:
        if self._started:
            return

        Vilib.camera_start(size=self.cfg.frame_size)

        # Set frame rate to reduce load
        try:
            Vilib.set_controls({"FrameRate": int(self.cfg.frame_rate)})
        except Exception:
            pass

        # Wait until Vilib.img becomes a valid numpy array
        t0 = time.time()
        while True:
            img = getattr(Vilib, "img", None)
            if isinstance(img, np.ndarray) and img.size > 0:
                try:
                    Vilib.flask_img = img
                except Exception:
                    pass
                break

            if time.time() - t0 > self.cfg.startup_wait_seconds:
                break
            time.sleep(0.05)

        # Start local/web display
        Vilib.display(local=self.cfg.display_local, web=self.cfg.display_web)

        self._started = True

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
        Returns a frame
        """
        if not self._started:
            return None

        img = getattr(Vilib, "img", None)
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None

        # Ensure uint8 contiguous
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(img)

        if self.cfg.source_color_order != self.cfg.output_color_order:
            if self.cfg.source_color_order == "rgb" and self.cfg.output_color_order == "bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.cfg.source_color_order == "bgr" and self.cfg.output_color_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.cfg.debug_color_stats and frame.ndim == 3 and frame.shape[2] >= 3:
            now = time.time()
            if now - self._last_color_print_t >= max(0.2, self.cfg.debug_color_stats_period_s):
                self._last_color_print_t = now
                if self.cfg.output_color_order == "rgb":
                    r = float(frame[:, :, 0].mean())
                    g = float(frame[:, :, 1].mean())
                    b = float(frame[:, :, 2].mean())
                else:
                    b = float(frame[:, :, 0].mean())
                    g = float(frame[:, :, 1].mean())
                    r = float(frame[:, :, 2].mean())
                blue_ratio = b / max(1.0, 0.5 * (r + g))
                print(f"[camera] mean_rgb=({r:.1f},{g:.1f},{b:.1f}) blue_ratio={blue_ratio:.2f}")

        return frame
