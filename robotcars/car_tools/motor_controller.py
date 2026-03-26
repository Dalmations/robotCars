# car_tools/motor_controller.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from picarx import Picarx  # SunFounder PiCar-X library


@dataclass
class MotorConfig:
    max_steer_deg: float = 35.0
    steer_gain: float = 1.0
    steer_sign: float = 1.0
    steer_offset_deg: float = 0.0

    # Motion
    speed: int = 50                    # default drive speed

    # Safety / timing
    settle_seconds: float = 0.02       # servo settle pause


class MotorController:
    """
    Hardware motor driver for PiCar-X.

    - set_steering(angle_deg): sends the steering command to the servo
    - forward_for(seconds): drives forward for duration, optionally stops
    - stop(): stop motors
    """

    def __init__(self, cfg: Optional[MotorConfig] = None):
        self.cfg = cfg or MotorConfig()
        self.px = Picarx()
        self._reached_target = False
        self._applied_steer_deg: float = 0.0

        # Make sure we start stopped
        self.stop()

    def reset_reached(self) -> None:
        self._reached_target = False

    def set_steering(self, angle_deg: float) -> None:
        """
        angle_deg is the desired steering angle in degrees from controller.
        This function applies calibration and sends the steering command to the servo.
        """
        desired = float(angle_deg)
        cmd = self.cfg.steer_sign * (desired * self.cfg.steer_gain) + float(self.cfg.steer_offset_deg)
        cmd = max(-float(self.cfg.max_steer_deg), min(float(self.cfg.max_steer_deg), cmd))
        self.px.set_dir_servo_angle(cmd)
        self._applied_steer_deg = cmd
        if self.cfg.settle_seconds > 0:  # Brief servo settle pause
            time.sleep(self.cfg.settle_seconds)

    def set_speed(self, speed: int) -> None:
        self.cfg.speed = int(max(0, min(100, speed)))

    def forward_for(self, seconds: float, *, speed: Optional[int] = None) -> None:
        if seconds <= 0:
            return
        spd = self.cfg.speed if speed is None else int(max(0, min(100, speed)))  # Use configured drive speed

        self.px.forward(spd)
        time.sleep(float(seconds))

    def backward_for(self, seconds: float, *, speed: Optional[int] = None) -> None:
        if seconds <= 0:
            return
        spd = self.cfg.speed if speed is None else int(max(0, min(100, speed)))  # Use configured drive speed
        self.px.backward(spd)

        time.sleep(float(seconds))

    def stop(self) -> None:
        self.px.stop()

    def mark_reached(self) -> None:
        self._reached_target = True

    def get_applied_steering_deg(self) -> float:
        return float(self._applied_steer_deg)
