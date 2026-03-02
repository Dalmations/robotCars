# car_tools/motor_controller.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from picarx import Picarx  # SunFounder PiCar-X library


@dataclass
class MotorConfig:
    max_steer_deg: float = 35.0
    steer_gain: float = 1.0            # multiplies desired steering
    steer_sign: float = 1.0            # set -1.0 if left/right are flipped
    steer_offset_deg: float = 0.0      # add to command to make "0" go straight

    # Smoothing (0 disables)
    steering_slew_deg_per_s: float = 0.0  # e.g. 180.0 = max 180 deg/s change

    # Motion
    speed: int = 50                    # 0..100
    step_seconds: float = 0.20         # "one step" time (used by step_forward)

    # Safety / timing
    settle_seconds: float = 0.02       # pause after steering changes
    brake_between_steps: bool = True   # stop motor between forward_for calls


class MotorController:
    """
    Hardware motor driver for PiCar-X.

    - set_steering(angle_deg): applies sign/gain/offset/clamp then sets servo
    - forward_for(seconds): drives forward for duration, optionally stops
    - step_forward(): forward_for(step_seconds)
    - stop(): stop motors
    """

    def __init__(self, cfg: Optional[MotorConfig] = None):
        self.cfg = cfg or MotorConfig()
        self.px = Picarx()
        self._reached_target = False

        # for slew limiting
        self._last_servo_cmd_deg: float = 0.0
        self._last_servo_time: float = time.time()

        # Make sure we start stopped
        self.stop()

    def reset_reached(self) -> None:
        self._reached_target = False

    def set_steering(self, angle_deg: float) -> None:
        """
        angle_deg is the desired steering angle in degrees from controller.
        This function applies calibration and clamps to hardware limits.
        """
        desired = float(angle_deg)

        # Apply sign + gain + offset
        cmd = self.cfg.steer_sign * (desired * self.cfg.steer_gain) + float(self.cfg.steer_offset_deg)

        # Clamp to servo limits
        cmd = max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, cmd))

        # Slew limit if enabled
        if self.cfg.steering_slew_deg_per_s > 0:
            now = time.time()
            dt = max(1e-6, now - self._last_servo_time)
            max_delta = float(self.cfg.steering_slew_deg_per_s) * dt
            cmd = max(self._last_servo_cmd_deg - max_delta, min(self._last_servo_cmd_deg + max_delta, cmd))
            self._last_servo_time = now
            self._last_servo_cmd_deg = cmd
        else:
            self._last_servo_cmd_deg = cmd
            self._last_servo_time = time.time()

        self.px.set_dir_servo_angle(cmd)
        if self.cfg.settle_seconds > 0:
            time.sleep(self.cfg.settle_seconds)

    def set_speed(self, speed: int) -> None:
        self.cfg.speed = int(max(0, min(100, speed)))

    def forward_for(self, seconds: float, *, speed: Optional[int] = None) -> None:
        if seconds <= 0:
            return
        spd = self.cfg.speed if speed is None else int(max(0, min(100, speed)))

        self.px.forward(spd)
        time.sleep(float(seconds))

        if self.cfg.brake_between_steps:
            self.stop()

    def backward_for(self, seconds: float, *, speed: Optional[int] = None) -> None:
        if seconds <= 0:
            return
        spd = self.cfg.speed if speed is None else int(max(0, min(100, speed)))
        self.px.backward(spd)

        time.sleep(float(seconds))

        if self.cfg.brake_between_steps:
            self.stop()

    def step_forward(self) -> None:
        self.forward_for(self.cfg.step_seconds)

    def stop(self) -> None:
        self.px.stop()

    def mark_reached(self) -> None:
        self._reached_target = True

    def at_target(self) -> bool:
        return self._reached_target