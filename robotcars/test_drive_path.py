from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from car_tools.camera_input import (
    PiCarXCamera,
    CameraConfig,
    read_ultrasonic_cm,
    ultrasonic_to_countdown,
)
from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Path


@dataclass
class LoopConfig:
    cm_per_grid: float = 50.0
    ahead_cm_min: float = 5.0
    ahead_cm_max: float = 120.0
    ultra_clear_cm: float = 60.0
    ultra_timer_period_s: float = 0.1

    min_pose_conf_for_replan: float = 0.55
    blurry_hold_pause_s: float = 0.20

    log_period_s: float = 1.0


def _estimate_translation_step_cells(follower: PathFollower, motor: MotorController) -> float:
    return float(follower.cfg.dt) / float(max(1e-6, motor.cfg.step_seconds))


def _tick_ultrasonic(
    *,
    motor: MotorController,
    shared_map: SharedMap,
    loop_cfg: LoopConfig,
    ultra_countdown: int,
    car_id: int,
) -> tuple[Optional[float], int]:
    dist_cm = read_ultrasonic_cm(motor)
    ultra_countdown = max(0, ultra_countdown - 1)
    ultra_countdown = min(ultra_countdown, ultrasonic_to_countdown(dist_cm))

    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is not None:
        shared_map.add_ultra_obstacle(
            pose_world=pose_world,
            dist_cm=dist_cm,
            cm_per_grid=loop_cfg.cm_per_grid,
            ahead_cm_min=loop_cfg.ahead_cm_min,
            ahead_cm_max=loop_cfg.ahead_cm_max,
            ultra_clear_cm=loop_cfg.ultra_clear_cm,
        )

    return dist_cm, ultra_countdown


def _show_slam_debug_frame(slam: MonocularVSLAM) -> None:
    frame = slam.get_debug_matches_frame()
    if frame is None:
        frame = slam.get_debug_keypoints_frame()
    if frame is None:
        return
    cv2.imshow("SLAM debug", frame)
    cv2.waitKey(1)


def _log_drive_status(
    *,
    pose_grid,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    dist_cm: Optional[float],
    ultra_countdown: int,
    pose_conf: float,
    pose_inliers: int,
    pose_blurry: bool,
    pose_contrast: float,
    prefix: str = "",
) -> None:
    ultra_str = "None" if dist_cm is None else f"{round(dist_cm, 1)}cm"
    print(
        f"{prefix}"
        f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
        f"goal=({goal_xy_grid[0]},{goal_xy_grid[1]}) d={d_goal:.1f} "
        f"ultra={ultra_str} ultra_countdown={ultra_countdown} "
        f"pose_conf={pose_conf:.2f} inliers={pose_inliers} blurry={pose_blurry} "
        f"ctr={pose_contrast:.1f}"
    )


def drive_to_goal(
    goal_xy_grid: Tuple[int, int],
    *,
    shared_map: SharedMap,
    planner: MovementPlanner,
    follower: PathFollower,
    motor: MotorController,
    slam: MonocularVSLAM,
    camera: PiCarXCamera,
    loop_cfg: LoopConfig,
    car_id: int = 0,
    timeout_s: float = 180.0,
    debug_show_keypoints: bool = False,
) -> bool:
    goal_xy_grid = (int(goal_xy_grid[0]), int(goal_xy_grid[1]))

    current_path: Optional[Path] = None
    latest_ultra_cm: Optional[float] = None
    ultra_countdown = ultrasonic_to_countdown(None)

    last_ultra_tick_t = 0.0
    last_log_t = 0.0
    t0 = time.time()

    follower.reset()

    try:
        while (time.time() - t0) < timeout_s:
            now = time.time()

            if (now - last_ultra_tick_t) >= loop_cfg.ultra_timer_period_s:
                last_ultra_tick_t = now
                latest_ultra_cm, ultra_countdown = _tick_ultrasonic(
                    motor=motor,
                    shared_map=shared_map,
                    loop_cfg=loop_cfg,
                    ultra_countdown=ultra_countdown,
                    car_id=car_id,
                )

            frame = camera.read()
            if frame is None:
                motor.stop()
                time.sleep(0.05)
                continue

            slam.tick(frame, translation_step=_estimate_translation_step_cells(follower, motor))

            if debug_show_keypoints:
                _show_slam_debug_frame(slam)

            pose_grid = shared_map.get_pose(car_id, frame="grid")
            pose_world = shared_map.get_pose(car_id, frame="world")
            if pose_grid is None or pose_world is None:
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue

            d_goal = math.hypot(goal_xy_grid[0] - pose_grid.x, goal_xy_grid[1] - pose_grid.y)
            if d_goal <= follower.cfg.goal_tolerance:
                motor.stop()
                motor.mark_reached()
                return True

            pose_conf, pose_high_conf, pose_blurry, pose_inliers, pose_contrast = slam.slam_quality(
                min_confidence=loop_cfg.min_pose_conf_for_replan
            )

            if current_path is not None and pose_blurry and not pose_high_conf:
                motor.stop()
                time.sleep(max(0.01, loop_cfg.blurry_hold_pause_s))
                if (now - last_log_t) >= loop_cfg.log_period_s:
                    last_log_t = now
                    _log_drive_status(
                        pose_grid=pose_grid,
                        goal_xy_grid=goal_xy_grid,
                        d_goal=d_goal,
                        dist_cm=latest_ultra_cm,
                        ultra_countdown=ultra_countdown,
                        pose_conf=pose_conf,
                        pose_inliers=pose_inliers,
                        pose_blurry=pose_blurry,
                        pose_contrast=pose_contrast,
                        prefix="hold=pose_recovery ",
                    )
                continue

            needs_path = current_path is None or ultra_countdown <= 0
            if needs_path:
                if pose_high_conf or current_path is None:
                    current_path = planner.plan_to_target(
                        target=TargetPoint(float(goal_xy_grid[0]), float(goal_xy_grid[1])),
                        shared_map=shared_map,
                        target_frame="grid",
                    )
                    ultra_countdown = max(1, ultrasonic_to_countdown(latest_ultra_cm))
                else:
                    motor.stop()
                    time.sleep(max(0.01, loop_cfg.blurry_hold_pause_s))
                    continue

            if current_path is None or len(current_path.waypoints) < 2:
                current_path = None
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue

            steer_deg = follower.steering_command(current_path, pose_grid, d_goal)
            motor.set_steering(steer_deg)
            motor.forward_for(follower.cfg.dt, speed=int(motor.cfg.speed))

            now = time.time()
            if (now - last_log_t) >= loop_cfg.log_period_s:
                last_log_t = now
                _log_drive_status(
                    pose_grid=pose_grid,
                    goal_xy_grid=goal_xy_grid,
                    d_goal=d_goal,
                    dist_cm=latest_ultra_cm,
                    ultra_countdown=ultra_countdown,
                    pose_conf=pose_conf,
                    pose_inliers=pose_inliers,
                    pose_blurry=pose_blurry,
                    pose_contrast=pose_contrast,
                )

    finally:
        motor.stop()

    return False


def main() -> None:
    shared_map = SharedMap()
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-2.0, -2.0))

    cam = PiCarXCamera(CameraConfig(
        display_local=False,
        display_web=False,
        frame_size=(640, 480),
        source_color_order="rgb",
        output_color_order="rgb",
        frame_rate=30,
        debug_color_stats=True,
        camera_controls={"Saturation": 0.80},
    ))
    cam.start()

    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)
    slam = MonocularVSLAM(
        intr,
        shared_map,
        car_id=0,
        cfg=VslamConfig(
            input_color_order=cam.color_order,
            debug_draw_keypoints=True,
            debug_draw_matches=True,
            translation_step=0.2,
            forward_sign=-1.0,
            pose_ema_alpha=0.25,
            gray_use_clahe=True,
            gray_clahe_clip_limit=2.5,
            gray_clahe_tile_size=8,
            auto_white_balance=True,
        ),
    )

    motor = MotorController(MotorConfig(
        speed=26,
        step_seconds=0.40,
        brake_between_steps=False,
        steering_slew_deg_per_s=90.0,
        steer_sign=1.0,
        steer_offset_deg=0.0,
        max_steer_deg=35.0,
        settle_seconds=0.01,
    ))

    follower = PathFollower(motor, FollowerConfig(
        dt=0.12,
        lookahead=14.0,
        wheelbase=2.2,
        goal_tolerance=3.0,
        pose_frame="grid",
        steer_sign=1.0,
        max_steer_deg=motor.cfg.max_steer_deg,
        steer_alpha=0.25,
        steer_deadband_deg=2.0,
        steer_rate_limit_deg_per_tick=12.0,
        dock_distance_grid=12.0,
        dock_min_lookahead_grid=6.0,
    ))

    loop_cfg = LoopConfig(
        cm_per_grid=50.0,
        ultra_timer_period_s=0.1,
    )

    planner = MovementPlanner(
        planning_cfg=PlanningConfig(
            include_slam_points=False,
            inflation_radius_cells=0,
            simplify_path=True,
            nudge_start_goal=True,
        ),
        world_size=(50, 50),
    )

    try:
        t_warm = time.time()
        while time.time() - t_warm < 1.0:
            frame = cam.read()
            if frame is not None:
                slam.tick(frame, translation_step=0.0)
            time.sleep(0.02)

        print("Drive (2,2) -> (45,45)...")
        ok = drive_to_goal(
            (45, 45),
            shared_map=shared_map,
            planner=planner,
            follower=follower,
            motor=motor,
            slam=slam,
            camera=cam,
            loop_cfg=loop_cfg,
            timeout_s=180.0,
            debug_show_keypoints=True,
        )
        print("Reached (45,45):", ok)

        square = [(45, 45), (5, 45), (5, 5), (45, 5), (45, 45)]
        print("Drive square:", square)
        for goal in square[1:]:
            ok = drive_to_goal(
                goal,
                shared_map=shared_map,
                planner=planner,
                follower=follower,
                motor=motor,
                slam=slam,
                camera=cam,
                loop_cfg=loop_cfg,
                timeout_s=180.0,
                debug_show_keypoints=True,
            )
            print(f"Reached {goal}:", ok)
            time.sleep(0.5)

    finally:
        motor.stop()
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass