# Main entry point for the drive-path demo.
# Builds the shared map, motor, follower, and a fixed shape path, then follows it once.
from __future__ import annotations

import time

import cv2

from car_tools.camera_input import CameraStreamConfig, OpenCVCameraStream
from car_tools.movement import build_equilateral_triangle_route, build_square_route, route_to_path
from car_tools.motor_controller import MotorConfig, MotorController
from car_tools.obstacle_detection import (
    CameraIntrinsics,
    MonocularVSLAM,
    OriginBoxConfig,
    VslamConfig,
)
from car_tools.picarx_path_follower import FollowerConfig, PathFollower
from coordination.localization_manager import LocalizationConfig, LocalizationManager
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
    return shared_map


def build_localization(shared_map: SharedMap) -> LocalizationManager:
    intr = CameraIntrinsics(
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
    )
    origin_box = OriginBoxConfig(
        enabled=True,
        marker_id=0,
        marker_size_world=0.5,
        marker_world_x=0.0,
        marker_world_y=0.0,
        marker_world_yaw_deg=0.0,
        camera_mount_forward=0.0,
        camera_mount_left=0.0,
        camera_pan_sign=1.0,
    )
    marker_localizer = MonocularVSLAM(
        intr,
        cfg=VslamConfig(
            debug_draw_frame=True,
            input_color_order="bgr",
            origin_box=origin_box,
        ),
    )
    return LocalizationManager(
        shared_map=shared_map,
        car_id=0,
        initial_pose=Pose(0.0, 0.0, 0.0),
        marker_localizer=marker_localizer,
        cfg=LocalizationConfig(
            marker_blend_alpha=0.35,
            marker_snap_distance=1.5,
            marker_snap_heading_deg=35.0,
            marker_snap_consistency_frames=3,
        ),
    )


def build_camera_stream() -> OpenCVCameraStream:
    return OpenCVCameraStream(CameraStreamConfig(
        device_index=0,
        width=640,
        height=480,
    ))


def initialize_pose_from_marker(
    *,
    localization: LocalizationManager,
    camera_stream: OpenCVCameraStream,
    timeout_s: float = 3.0,
) -> bool:
    t0 = time.time()
    while (time.time() - t0) < float(timeout_s):
        frame = camera_stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        pose = localization.initialize_from_marker_frame(
            frame,
            camera_pan_deg=0.0,
        )
        if pose is not None:
            return True
        time.sleep(0.05)
    return False


def build_motor() -> MotorController:
    return MotorController(MotorConfig(
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
    # route = build_equilateral_triangle_route(start_grid, side_cells=6)
    route = build_square_route(start_grid, side_cells=6)
    return [
        *route,
    ]


def build_path(shared_map: SharedMap) -> Path:
    return route_to_path(build_route(shared_map))


def main() -> None:
    shared_map = build_shared_map()
    localization = build_localization(shared_map)
    camera_stream = build_camera_stream()
    motor = build_motor()
    follower = build_follower(motor)
    loop_cfg = build_loop_config()
    marker_locked = initialize_pose_from_marker(
        localization=localization,
        camera_stream=camera_stream,
        timeout_s=float(loop_cfg.startup_marker_lock_timeout_s),
    )
    print(f"Startup marker lock: {marker_locked}")
    path = build_path(shared_map)

    try:
        route = [(int(round(wp.x)), int(round(wp.y))) for wp in path.waypoints]
        print("Drive route:", route)
        ok = drive_path(
            path,
            shared_map=shared_map,
            localization=localization,
            follower=follower,
            motor=motor,
            loop_cfg=loop_cfg,
            timeout_s=180.0,
            debug_show_grid=True,
            marker_frame_provider=camera_stream.get_frame,
            camera_pan_provider=lambda: 0.0,
        )
        print(f"Completed path to {route[-1]}:", ok)
        time.sleep(0.5)
    finally:
        motor.stop()
        camera_stream.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
