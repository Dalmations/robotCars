# motor_controller.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from picarx import Picarx  # SunFounder PiCar-X library https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/install_all_modules.html

from model import TargetPoint


@dataclass
class MotorConfig:
    # Steering
    max_steer_deg: float = 35.0     # PiCar-X examples use ~35 deg
    steer_gain: float = 1.0        # multiply desired steering angle

    # Motion
    speed: int = 80                # 0..100-ish (library uses percent-like speed)
    step_seconds: float = 0.18     # time to move ~1 grid cell (tune for your car & cell size)

    # Safety
    settle_seconds: float = 0.02   # brief pause after steering changes


class MotorController:
    """
    Hardware motor driver for PiCar-X.

    - set_steering(angle_deg): sets steering servo
    - step_forward(): drive forward for a fixed duration
    - stop(): stop motors
    """
    def __init__(self, cfg: Optional[MotorConfig] = None):
        self.cfg = cfg or MotorConfig()
        self.px = Picarx()
        self._reached_target = False

    def set_steering(self, angle_deg: float) -> None:
        # Clamp and apply gain
        a = float(angle_deg) * self.cfg.steer_gain
        a = max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, a))
        self.px.set_dir_servo_angle(a)
        time.sleep(self.cfg.settle_seconds)

    def step_forward(self) -> None:
        self.px.forward(self.cfg.speed)
        time.sleep(self.cfg.step_seconds)
        self.px.forward(0) #  stop between steps for predictability

    def stop(self) -> None:
        self.px.stop()

    def mark_reached(self) -> None:
        self._reached_target = True

    def at_target(self) -> bool:
        return self._reached_target
