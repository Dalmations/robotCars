from __future__ import annotations

import math
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

# import cv2

from car_tools.camera_input import read_ultrasonic_cm
from coordination.shared_map import SharedMap
from car_tools.motor_controller import MotorController
from car_tools.picarx_path_follower import (
    PurePursuitFollower,
)

from model import Path, Pose, TargetPoint


@dataclass
class LoopConfig:
    cm_per_grid: float = 50.0
    action_tick_s: float = 0.10
    ultra_stop_cm: float = 20.0
    pivot_turn_heading_deg: float = 45.0
    pivot_turn_exit_deg: float = 20.0
    pivot_turn_steer_deg: float = 30.0
    pivot_turn_settle_s: float = 0.12
    pivot_turn_deg_per_s: float = 12.0
    pivot_turn_cells_per_deg: float = 0.05


@dataclass
class DriveCommand:
    target_x: float
    target_y: float
    heading_error_deg: float
    commanded_steer_deg: float
    drive_mode: str
    speed: int
    odom_steer_deg: float = 0.0


# class DrivePoseEstimator(Protocol):
#     def get_pose(self, *, frame: str = "world") -> Pose: ...
#     def propagate_dead_reckoning(self, *, forward_step: float, yaw_delta: float) -> Pose: ...
#     def maybe_correct_from_obstacle_detection(
#         self,
#         frame_rgb_or_bgr: Optional[Any] = None,
#         *,
#         now_s: Optional[float] = None,
#     ) -> Any: ...


def _current_leg_path(path: Path, waypoint_idx: int) -> Path:
    if path is None or not path.waypoints:
        return Path(waypoints=[])

    end_idx = max(0, min(int(waypoint_idx), len(path.waypoints) - 1))
    start_idx = max(0, end_idx - 1)
    if start_idx == end_idx:
        return Path(waypoints=[path.waypoints[end_idx]])
    return Path(waypoints=[path.waypoints[start_idx], path.waypoints[end_idx]])


def _heading_error_to_point_deg(pose: Pose, target: TargetPoint) -> float:
    return math.degrees(
        math.atan2(
            math.sin(math.atan2(float(target.y) - float(pose.y), float(target.x) - float(pose.x)) - float(pose.theta)),
            math.cos(math.atan2(float(target.y) - float(pose.y), float(target.x) - float(pose.x)) - float(pose.theta)),
        )
    )


def _pivot_reverse_duration_s(
    *,
    loop_cfg: LoopConfig,
    follower: PurePursuitFollower,
    drive_speed: int,
    heading_error_deg: float,
) -> float:
    remaining_error_deg = max(
        0.0,
        abs(float(heading_error_deg)) - float(loop_cfg.pivot_turn_exit_deg),
    )
    if remaining_error_deg <= 1e-6:
        return 0.0

    yaw_deg_per_s = float(loop_cfg.pivot_turn_deg_per_s) * float(drive_speed)
    if yaw_deg_per_s <= 1e-6:
        return float(loop_cfg.action_tick_s)

    return max(
        float(loop_cfg.action_tick_s),
        remaining_error_deg / yaw_deg_per_s,
    )


def _pivot_reverse_yaw_delta_rad(
    *,
    loop_cfg: LoopConfig,
    follower: PurePursuitFollower,
    drive_speed: int,
    heading_error_deg: float,
    duration_s: float,
) -> float:
    yaw_deg_per_s = float(loop_cfg.pivot_turn_deg_per_s) * float(drive_speed)
    yaw_deg = min(
        max(0.0, abs(float(heading_error_deg)) - float(loop_cfg.pivot_turn_exit_deg)),
        yaw_deg_per_s * float(duration_s),
    )
    direction_sign = 1.0 if float(heading_error_deg) >= 0.0 else -1.0
    return math.radians(direction_sign * yaw_deg)


def _pivot_reverse_step_cells(
    *,
    loop_cfg: LoopConfig,
    yaw_delta_rad: float,
) -> float:
    yaw_deg = abs(math.degrees(float(yaw_delta_rad)))
    return -yaw_deg * max(0.0, float(loop_cfg.pivot_turn_cells_per_deg))


def _path_length(path: Path) -> float:
    if path is None or len(path.waypoints) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path.waypoints) - 1):
        p0 = path.waypoints[i]
        p1 = path.waypoints[i + 1]
        total += math.hypot(float(p1.x) - float(p0.x), float(p1.y) - float(p0.y))
    return total


def _path_progress(path: Path, pose: Pose) -> float:
    if path is None or len(path.waypoints) < 2:
        return 0.0

    px, py = float(pose.x), float(pose.y)
    best_dist2 = float("inf")
    best_progress = 0.0
    progress_acc = 0.0

    for i in range(len(path.waypoints) - 1):
        p0 = path.waypoints[i]
        p1 = path.waypoints[i + 1]
        x0, y0 = float(p0.x), float(p0.y)
        x1, y1 = float(p1.x), float(p1.y)
        dx = x1 - x0
        dy = y1 - y0
        seg2 = dx * dx + dy * dy
        seg_len = math.hypot(dx, dy)
        if seg2 < 1e-12 or seg_len < 1e-9:
            continue

        u = ((px - x0) * dx + (py - y0) * dy) / seg2
        u = max(0.0, min(1.0, float(u)))
        proj_x = x0 + u * dx
        proj_y = y0 + u * dy
        dist2 = (proj_x - px) ** 2 + (proj_y - py) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_progress = progress_acc + u * seg_len

        progress_acc += seg_len

    return best_progress


def _tick_ultrasonic(
    *,
    motor: MotorController,
    shared_map: SharedMap,
    loop_cfg: LoopConfig,
) -> Optional[float]:
    dist_cm = read_ultrasonic_cm(motor)

    pose_world = shared_map.get_pose(frame="world")
    if pose_world is not None:
        shared_map.add_ultra_obstacle(
            pose_world=pose_world,
            dist_cm=dist_cm,
            cm_per_grid=loop_cfg.cm_per_grid,
        )

    return dist_cm


def _current_pose_grid(
    *,
    shared_map: SharedMap,
    follower: PurePursuitFollower,
) -> Pose:
    pose_grid = shared_map.get_pose(frame="grid")
    if pose_grid is not None:
        return pose_grid

    # if pose_estimator is not None:
    #     return pose_estimator.get_pose(frame="grid")

    return follower.integrate_dead_reckoning(
        shared_map=shared_map,
        forward_step=0.0,
        yaw_delta=0.0,
    )


def _propagate_pose(
    *,
    shared_map: SharedMap,
    follower: PurePursuitFollower,
    forward_step: float,
    yaw_delta: float,
) -> None:
    # if pose_estimator is None:
    follower.integrate_dead_reckoning(
        shared_map=shared_map,
        forward_step=forward_step,
        yaw_delta=yaw_delta,
    )
    return

    # pose_estimator.propagate_dead_reckoning(
    #     forward_step=forward_step,
    #     yaw_delta=yaw_delta,
    # )


# def _maybe_apply_visual_correction(
#     *,
#     pose_estimator: Optional[DrivePoseEstimator],
#     visual_frame_provider: Optional[Callable[[], Any]],
#     now_s: float,
# ) -> None:
#     if pose_estimator is None:
#         return

#     correction = pose_estimator.maybe_correct_from_obstacle_detection(
#         frame_provider=visual_frame_provider,
#         now_s=now_s,
#     )
#     if not correction.attempted:
#         return

#     print(
#         "visual_corr "
#         f"accepted={int(correction.accepted)} "
#         f"reason={correction.reason} "
#         f"head_err={correction.heading_error_deg:.1f} "
#         f"conf={correction.visual_confidence:.2f}"
#     )


def _log_drive_status(
    *,
    pose_grid: Pose,
    waypoint_idx: int,
    waypoint_total: int,
    goal_xy_grid: tuple[int, int],
    d_goal: float,
    progress: float,
    total_progress: float,
    dist_cm: Optional[float],
    path_len: int,
    target_xy: tuple[float, float],
    heading_error_deg: float,
    commanded_steer_deg: float,
    applied_steer_deg: float,
    drive_mode: str,
    speed: int,
) -> None:
    ultra_str = "None" if dist_cm is None else f"{round(dist_cm, 1)}cm"
    print(
        f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
        f"wp={waypoint_idx}/{waypoint_total} "
        f"goal=({goal_xy_grid[0]},{goal_xy_grid[1]}) d={d_goal:.1f} "
        f"prog={progress:.1f}/{total_progress:.1f} "
        f"ultra={ultra_str} path_n={path_len} "
        f"target=({target_xy[0]:.1f},{target_xy[1]:.1f}) "
        f"head_err={heading_error_deg:.1f} "
        f"cmd_steer={commanded_steer_deg:.1f} applied_steer={applied_steer_deg:.1f} "
        f"mode={drive_mode} speed={speed}"
    )

# def build_visual_test_stack(
#     shared_map: SharedMap,
# ) -> tuple[Optional[ConservativePoseEstimator], Optional[PiCarXCamera]]:
#     frame_provider = PiCarXCamera(CameraConfig(
#         display_local=True,
#         display_web=False,
#         frame_size=(640, 480),
#         frame_rate=30,
#     ))
#     frame_provider.start()

#     frame_w, frame_h = frame_provider.cfg.frame_size
#     focal_px = 0.9 * max(frame_w, frame_h)
#     intrinsics = CameraIntrinsics(
#         fx=float(focal_px),
#         fy=float(focal_px),
#         cx=0.5 * float(frame_w),
#         cy=0.5 * float(frame_h),
#     )

#     visual_localizer = MonocularVSLAM(
#         intrinsics,
#         cfg=VslamConfig(
#             pose_ema_alpha=0.15,
#         ),
#     )
#     pose_estimator = ConservativePoseEstimator(
#         shared_map,
#         visual_localizer=visual_localizer,
#         cfg=ConservativeCorrectionConfig(
#             min_cycles_between_corrections=6,
#             min_seconds_between_corrections=0.75,
#             min_translation_between_corrections=1.0,
#             min_heading_change_between_corrections_deg=10.0,
#             heading_agreement_threshold_deg=10.0,
#             correction_alpha=0.2,
#             min_visual_confidence=0.6,
#         ),
#     )
#     return pose_estimator, frame_provider

def drive_path(
    path: Path,
    *,
    shared_map: SharedMap,
    follower: PurePursuitFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    timeout_s: float = 180.0,
) -> bool:
    if path is None or len(path.waypoints) < 2:
        raise ValueError("drive_path requires a Path with at least two waypoints")
    
    # pose_estimator, frame_provider = build_visual_test_stack(shared_map)

    total_progress = _path_length(path)
    best_progress = 0.0

    last_ultra_tick_t = 0.0
    latest_ultra_cm: Optional[float] = None
    t0 = time.time()
    current_wp_idx = 1
    pivot_active = False
    follower.reset()

    try:
        while (time.time() - t0) < timeout_s:
            now = time.time()
            if (now - last_ultra_tick_t) >= loop_cfg.action_tick_s:
                last_ultra_tick_t = now
                latest_ultra_cm = _tick_ultrasonic(
                    motor=motor,
                    shared_map=shared_map,
                    loop_cfg=loop_cfg,
                )

            pose_grid = _current_pose_grid(
                shared_map=shared_map,
                follower=follower,
                # pose_estimator=pose_estimator,
            )

            active_wp = path.waypoints[current_wp_idx]
            goal_xy_grid = (int(round(active_wp.x)), int(round(active_wp.y)))
            d_goal = math.hypot(goal_xy_grid[0] - pose_grid.x, goal_xy_grid[1] - pose_grid.y)
            best_progress = max(best_progress, _path_progress(path, pose_grid))
            if not pivot_active and d_goal <= float(follower.cfg.goal_tolerance):
                if current_wp_idx >= len(path.waypoints) - 1:
                    motor.stop()
                    motor.mark_reached()
                    return True

                current_wp_idx += 1
                active_wp = path.waypoints[current_wp_idx]
                goal_xy_grid = (int(round(active_wp.x)), int(round(active_wp.y)))
                d_goal = math.hypot(goal_xy_grid[0] - pose_grid.x, goal_xy_grid[1] - pose_grid.y)
                pivot_active = abs(_heading_error_to_point_deg(pose_grid, active_wp)) > float(loop_cfg.pivot_turn_heading_deg)

            active_wp = path.waypoints[current_wp_idx]
            tracking_path = _current_leg_path(path, current_wp_idx)
            target_x, target_y, heading_error_deg, steer_deg = follower.tracking_command(
                tracking_path,
                pose_grid,
                d_goal,
                steer_cap_deg=follower.cfg.max_steer_deg,
            )
            motor.set_steering(0) #steer_deg

            drive_mode = "track"
            drive_speed = int(motor.cfg.speed)
            odom_step_cells = 0.0
            odom_steer_deg = motor.get_applied_steering_deg()

            if latest_ultra_cm is not None and latest_ultra_cm <= float(loop_cfg.ultra_stop_cm):
                motor.stop()
                drive_mode = "paused_obstacle"
                drive_speed = 0
                time.sleep(float(loop_cfg.action_tick_s))
            else:
                if pivot_active:
                    heading_error_deg = _heading_error_to_point_deg(pose_grid, active_wp)
                    target_x = float(active_wp.x)
                    target_y = float(active_wp.y)
                    if abs(float(heading_error_deg)) <= float(loop_cfg.pivot_turn_exit_deg):
                        pivot_active = False
                        follower.sync_to_motor_steering()

                if pivot_active:
                    direction_sign = 1.0 if float(heading_error_deg) >= 0.0 else -1.0
                    steer_abs = min(
                        abs(float(loop_cfg.pivot_turn_steer_deg)),
                        float(follower.cfg.max_steer_deg),
                    )
                    pivot_duration_s = _pivot_reverse_duration_s(
                        loop_cfg=loop_cfg,
                        follower=follower,
                        drive_speed=drive_speed,
                        heading_error_deg=heading_error_deg,
                    )
                    drive_mode = "pivot_turn_reverse"
                    motor.set_steering(-direction_sign * steer_abs)
                    extra_pivot_settle_s = max(
                        0.0,
                        float(loop_cfg.pivot_turn_settle_s) - float(getattr(motor.cfg, "settle_seconds", 0.0)),
                    )
                    if extra_pivot_settle_s > 1e-6:
                        time.sleep(extra_pivot_settle_s)
                    odom_steer_deg = motor.get_applied_steering_deg()
                    steer_deg = odom_steer_deg
                    yaw_delta = _pivot_reverse_yaw_delta_rad(
                        loop_cfg=loop_cfg,
                        follower=follower,
                        drive_speed=drive_speed,
                        heading_error_deg=heading_error_deg,
                        duration_s=pivot_duration_s,
                    )
                    odom_step_cells = _pivot_reverse_step_cells(
                        loop_cfg=loop_cfg,
                        yaw_delta_rad=yaw_delta,
                    )
                    motor.backward_for(pivot_duration_s, speed=drive_speed)
                    follower.sync_to_motor_steering()
                else:
                    odom_step_cells = follower.estimate_step_cells_for_duration(
                        float(loop_cfg.action_tick_s),
                        action_tick_s=float(loop_cfg.action_tick_s),
                        speed=drive_speed,
                        speed_ref=int(motor.cfg.speed),
                    )
                    odom_steer_deg = motor.get_applied_steering_deg()
                    motor.forward_for(float(loop_cfg.action_tick_s), speed=drive_speed)
                    yaw_delta = follower.estimate_ackermann_yaw_delta(
                        odom_step_cells,
                        odom_steer_deg,
                    )
                _propagate_pose(
                    shared_map=shared_map,
                    follower=follower,
                    # pose_estimator=pose_estimator,
                    forward_step=odom_step_cells,
                    yaw_delta=yaw_delta,
                )
                # _maybe_apply_visual_correction(
                #     pose_estimator=pose_estimator,
                #     visual_frame_provider=None if frame_provider is None else frame_provider.read,
                #     now_s=time.time(),
                # )

            command = DriveCommand(
                target_x=target_x,
                target_y=target_y,
                heading_error_deg=heading_error_deg,
                commanded_steer_deg=steer_deg,
                drive_mode=drive_mode,
                speed=drive_speed,
                odom_steer_deg=odom_steer_deg,
            )

            _log_drive_status(
                pose_grid=pose_grid,
                waypoint_idx=current_wp_idx,
                waypoint_total=max(1, len(path.waypoints) - 1),
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                progress=best_progress,
                total_progress=total_progress,
                dist_cm=latest_ultra_cm,
                path_len=len(path.waypoints),
                target_xy=(command.target_x, command.target_y),
                heading_error_deg=command.heading_error_deg,
                commanded_steer_deg=command.commanded_steer_deg,
                applied_steer_deg=command.odom_steer_deg,
                drive_mode=command.drive_mode,
                speed=command.speed,
            )

            # # Debug grid
            # live_pose = shared_map.get_pose(frame="grid")
            # live_info = [
            #     (
            #         f"mode={command.drive_mode} d={d_goal:.2f} "
            #         f"head_err={command.heading_error_deg:.1f} "
            #         f"cmd={command.commanded_steer_deg:.1f} app={command.odom_steer_deg:.1f}"
            #     ),
            #     (
            #         f"ultra={'None' if latest_ultra_cm is None else f'{latest_ultra_cm:.1f}cm'} "
            #         f"path_n={len(path.waypoints)} speed={command.speed}"
            #     ),
            # ]
            # if live_pose is not None:
            #     live_info.append(
            #         f"pose=({live_pose.x:.2f},{live_pose.y:.2f},{live_pose.theta:.2f})"
            #     )
            # frame = shared_map.render_grid_debug_view(
            #     path=path,
            #     target=active_wp,
            #     control_target=TargetPoint(command.target_x, command.target_y),
            #     cell_px=14,
            #     info_lines=live_info,
            # )
            # cv2.imshow("Planning debug", frame)
            # cv2.waitKey(1)
    finally:
        motor.stop()
    return False

def start_path(
    path: Path,
    *,
    shared_map: SharedMap,
    follower: PurePursuitFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    timeout_s: float = 180.0,
) -> bool:
    if follower.shape == "circle":
        follower.follow(path)
    else:
        drive_path(
            path,
            shared_map=shared_map,
            follower=follower,
            motor=motor,
            loop_cfg=loop_cfg,
            timeout_s=timeout_s,
        )
    return True
