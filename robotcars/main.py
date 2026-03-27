# Config builder now! Consider renaming.
# Builds the shared map, motor, follower, and a fixed shape path, then follows it once.
from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
from car_tools.movement import build_square_route, route_to_path
from car_tools.motor_controller import MotorConfig, MotorController
from coordination.shared_map import SharedMap
from model import Path, Pose
from test_drive_path import LoopConfig, drive_path

def build_shared_map() -> SharedMap:
    shared_map = SharedMap()
    shared_map.configure_grid(
        size=(30, 30),
        resolution=1.0,
        origin_world=(-10.0, -10.0),
    )
    shared_map.set_pose(Pose(0.0, 0.0, 0.0))
    return shared_map


def build_motor() -> MotorController:
    return MotorController(MotorConfig(
        max_steer_deg=35.0,                        # hardware steering clamp
        steer_gain=1.0,                           # steering gain calibration
        steer_sign=1.0,                           # flip to -1.0 if left/right are mirrored
        steer_offset_deg=0.0,                     # trim so commanded 0 drives straight
        speed=26,                                 # default drive speed
        settle_seconds=0.01,                      # servo settle pause
    ))

def build_loop_config() -> LoopConfig:
    return LoopConfig(
        cm_per_grid= 20,                           # centimeters per cell
    )


def build_route(shared_map: SharedMap) -> list[tuple[int, int]]:
    start_grid = shared_map.get_car_grid_position(0)
    route = build_square_route(start_grid, side_cells=6)
    return [
        *route,
    ]


def build_path(shared_map: SharedMap) -> Path:
    return route_to_path(build_route(shared_map))