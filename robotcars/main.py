# Main entry point for the drive-path demo.
# Builds the shared map, motor, follower, and a fixed shape path, then follows it once.
from __future__ import annotations

import time
from typing import Callable, Optional

import cv2

from car_tools.camera_input import CameraConfig, OpenCvFrameProvider, PiCarXCamera
from car_tools.movement import build_square_route, route_to_path
from car_tools.motor_controller import MotorConfig, MotorController
from car_tools.obstacle_detection import (
    CameraIntrinsics,
    ConservativeCorrectionConfig,
    ConservativePoseEstimator,
    MonocularVSLAM,
    VslamConfig,
)
from car_tools.picarx_path_follower import FollowerConfig, PathFollower
from coordination.shared_map import SharedMap
from model import Path, Pose
from test_drive_path import LoopConfig, drive_path

from model import Path, Pose
from test_drive_path import LoopConfig, drive_path


def build_shared_map() -> SharedMap:
    shared_map = SharedMap()
    shared_map.configure_grid(
        size=(30, 30),
        resolution=1.0,
        origin_world=(-10.0, -10.0),
    )
    shared_map.set_pose(0, Pose(0.0, 0.0, 0.0))
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


def build_follower(motor: MotorController) -> PathFollower:
    return PathFollower(motor, FollowerConfig(
        lookahead=5.0,                            # pure pursuit lookahead distance
        wheelbase=1.0,                           # front to back wheel wheelbase
        goal_tolerance=0.6,                       # goal reached radius
        steer_sign=1.0,                           # follower steering sign
        steer_alpha=0.25,                         # steering smoother
        steer_deadband_deg=2.0,                   # ignore tiny steer changes
        steer_rate_limit_deg_per_tick=12.0,       # max steer change
        dock_distance_grid=8.0,                   # near goal threshold
        dock_min_lookahead_grid=1.5,              # minimum dock lookahead
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


def build_visual_test_stack(
    shared_map: SharedMap,
    *,
    car_id: int = 0,
) -> tuple[Optional[ConservativePoseEstimator], Optional[Callable[[], object]], Optional[Callable[[], None]]]:
    frame_provider = PiCarXCamera(CameraConfig(
        display_local=False,
        display_web=False,
        frame_size=(640, 480),
        frame_rate=30,
    ))
    frame_provider.start()
    if not frame_provider.is_opened():
        frame_provider.release()
        frame_provider = OpenCvFrameProvider(device_index=0, width=640, height=480)
    if not frame_provider.is_opened():
        frame_provider.release()
        return None, None, None

    frame_w, frame_h = frame_provider.frame_size()
    focal_px = 0.9 * max(frame_w, frame_h)
    intrinsics = CameraIntrinsics(
        fx=float(focal_px),
        fy=float(focal_px),
        cx=0.5 * float(frame_w),
        cy=0.5 * float(frame_h),
    )

    visual_localizer = MonocularVSLAM(
        intrinsics,
        shared_map,
        car_id=car_id,
        cfg=VslamConfig(
            debug_draw_keypoints=True,
            debug_draw_matches=False,
            publish_pose_to_shared_map=False,
            pose_ema_alpha=0.15,
        ),
    )
    pose_estimator = ConservativePoseEstimator(
        shared_map,
        car_id=car_id,
        visual_localizer=visual_localizer,
        cfg=ConservativeCorrectionConfig(
            min_cycles_between_corrections=6,
            min_seconds_between_corrections=0.75,
            min_translation_between_corrections=1.0,
            min_heading_change_between_corrections_deg=10.0,
            position_agreement_threshold=0.8,
            heading_agreement_threshold_deg=10.0,
            correction_alpha=0.2,
            min_visual_confidence=0.6,
        ),
    )
    return pose_estimator, frame_provider.get_frame, frame_provider.release


def main() -> None:
    shared_map = build_shared_map()
    motor = build_motor()
    follower = build_follower(motor)
    loop_cfg = build_loop_config()
    path = build_path(shared_map)
    pose_estimator, visual_frame_provider, close_frame_provider = build_visual_test_stack(shared_map)

    try:
        route = [(int(round(wp.x)), int(round(wp.y))) for wp in path.waypoints]
        print("Drive route:", route)
        if pose_estimator is None:
            print("Visual correction test build: camera unavailable, using dead reckoning only.")
        else:
            print("Visual correction test build: conservative obstacle-detection corrections enabled.")
        ok = drive_path(
            path,
            shared_map=shared_map,
            follower=follower,
            motor=motor,
            loop_cfg=loop_cfg,
            timeout_s=180.0,
            debug_show_grid=True,
            debug_show_visual=pose_estimator is not None,
            pose_estimator=pose_estimator,
            visual_frame_provider=visual_frame_provider,
        )
        print(f"Completed path to {route[-1]}:", ok)
        time.sleep(0.5)
    finally:
        if close_frame_provider is not None:
            close_frame_provider()
        motor.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
