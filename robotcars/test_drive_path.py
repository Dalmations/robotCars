from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from car_tools.camera_input import (
    read_ultrasonic_cm,
    ultrasonic_to_countdown,
)
from coordination.shared_map import SharedMap
from car_tools.movement import (
    MovementPlanner,
    PlanningConfig,
    build_equilateral_triangle_route,
    clamp,
)
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import (
    PathFollower,
    FollowerConfig,
    estimate_ackermann_yaw_delta,
    estimate_step_cells_for_duration,
    integrate_dead_reckoning,
)
from model import TargetPoint, Path, Pose


@dataclass
class LoopConfig:
    cm_per_grid: float = 50.0
    ahead_cm_min: float = 5.0
    ahead_cm_max: float = 120.0
    ultra_timer_period_s: float = 0.1
    ultra_last_valid_ttl_s: float = 0.35
    ultra_stop_cm: float = 20.0
    ultra_caution_cm: float = 40.0

    pause_after_stop_s: float = 0.20
    close_obstacle_replan_cm: float = 15.0
    stuck_timeout_s: float = 3.0
    stuck_reverse_s: float = 0.30
    stuck_reverse_speed_scale: float = 0.6
    hard_turn_heading_deg: float = 70.0
    hard_turn_duration_scale: float = 0.55
    hard_turn_speed_scale: float = 0.75
    heading_steer_gain: float = 0.55
    min_forward_ultra_countdown: int = 10


@dataclass
class UltrasonicState:
    dist_cm: Optional[float]
    countdown: int
    last_valid_cm: Optional[float]
    last_valid_t: float


@dataclass
class DriveCommand:
    target_x: float
    target_y: float
    heading_error_deg: float
    steer_deg: float
    drive_mode: str
    drive_speed: int
    drive_duration_s: float
    odom_step_cells: float = 0.0
    odom_yaw_deg: float = 0.0
    odom_steer_deg: float = 0.0
    yaw_delta: float = 0.0


def _tick_ultrasonic(
    *,
    motor: MotorController,
    shared_map: SharedMap,
    loop_cfg: LoopConfig,
    ultra_state: UltrasonicState,
    now: float,
    car_id: int,
) -> UltrasonicState:
    raw_dist_cm = read_ultrasonic_cm(motor)
    dist_cm = raw_dist_cm
    last_valid_cm = ultra_state.last_valid_cm
    last_valid_t = ultra_state.last_valid_t

    if raw_dist_cm is not None:
        last_valid_cm = raw_dist_cm
        last_valid_t = float(now)
    elif last_valid_cm is not None and (float(now) - float(last_valid_t)) <= float(loop_cfg.ultra_last_valid_ttl_s):
        dist_cm = last_valid_cm
    else:
        dist_cm = None

    ultra_countdown = ultrasonic_to_countdown(
        dist_cm,
        stop_cm=float(loop_cfg.ultra_stop_cm),
        caution_cm=float(loop_cfg.ultra_caution_cm),
    )

    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is not None and dist_cm is not None:
        shared_map.add_ultra_obstacle(
            pose_world=pose_world,
            dist_cm=dist_cm,
            cm_per_grid=loop_cfg.cm_per_grid,
            ahead_cm_min=loop_cfg.ahead_cm_min,
            ahead_cm_max=loop_cfg.ahead_cm_max,
        )

    return UltrasonicState(
        dist_cm=dist_cm,
        countdown=ultra_countdown,
        last_valid_cm=last_valid_cm,
        last_valid_t=last_valid_t,
    )


def _ensure_pose(*, shared_map: SharedMap, car_id: int) -> Optional[Pose]:
    pose_grid = shared_map.get_pose(car_id, frame="grid")
    if pose_grid is not None:
        return pose_grid

    integrate_dead_reckoning(
        shared_map=shared_map,
        car_id=car_id,
        forward_step=0.0,
        yaw_delta=0.0,
    )
    return shared_map.get_pose(car_id, frame="grid")


def _show_grid_debug_frame(
    *,
    shared_map: SharedMap,
    current_path: Optional[Path],
    target: TargetPoint,
    car_id: int,
) -> None:
    frame = shared_map.render_grid_debug_view(
        car_id=car_id,
        path=current_path,
        target=target,
        cell_px=14,
    )
    cv2.imshow("Planning debug", frame)
    cv2.waitKey(1)


def _plan_goal_path(
    *,
    planner: MovementPlanner,
    shared_map: SharedMap,
    goal_target: TargetPoint,
    debug_show_grid: bool,
    car_id: int,
) -> Path:
    path = planner.plan_to_target(
        target=goal_target,
        shared_map=shared_map,
        target_frame="grid",
    )
    if debug_show_grid:
        _show_grid_debug_frame(
            shared_map=shared_map,
            current_path=path,
            target=goal_target,
            car_id=car_id,
        )
    return path


def _ensure_path(
    *,
    current_path: Optional[Path],
    planner: MovementPlanner,
    shared_map: SharedMap,
    goal_target: TargetPoint,
    debug_show_grid: bool,
    car_id: int,
) -> Optional[Path]:
    if current_path is not None:
        return current_path
    return _plan_goal_path(
        planner=planner,
        shared_map=shared_map,
        goal_target=goal_target,
        debug_show_grid=debug_show_grid,
        car_id=car_id,
    )


def _recover_from_stuck(
    *,
    motor: MotorController,
    loop_cfg: LoopConfig,
    latest_ultra_cm: Optional[float],
    last_motion_t: float,
) -> tuple[bool, float]:
    motor.stop()

    close_obstacle = latest_ultra_cm is not None and latest_ultra_cm <= loop_cfg.close_obstacle_replan_cm
    if close_obstacle and (time.time() - last_motion_t) >= loop_cfg.stuck_timeout_s:
        motor.set_steering(0.0)
        reverse_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.stuck_reverse_speed_scale))))
        motor.backward_for(loop_cfg.stuck_reverse_s, speed=reverse_speed)
        motor.stop()
        return True, time.time()

    time.sleep(max(0.01, loop_cfg.pause_after_stop_s))
    return False, last_motion_t


def _apply_reverse_recovery(
    *,
    shared_map: SharedMap,
    car_id: int,
    motor: MotorController,
    loop_cfg: LoopConfig,
) -> None:
    reverse_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.stuck_reverse_speed_scale))))
    integrate_dead_reckoning(
        shared_map=shared_map,
        car_id=car_id,
        forward_step=-estimate_step_cells_for_duration(
            loop_cfg.stuck_reverse_s,
            step_seconds=float(motor.cfg.step_seconds),
            speed=reverse_speed,
            speed_ref=int(motor.cfg.speed),
        ),
        yaw_delta=0.0,
    )


def _maybe_hold_or_reverse(
    *,
    current_path: Optional[Path],
    pose_grid: Pose,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    ultra_state: UltrasonicState,
    shared_map: SharedMap,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    car_id: int,
    last_motion_t: float,
) -> tuple[Optional[Path], float, bool]:
    if current_path is None or len(current_path.waypoints) < 2:
        recovered, last_motion_t = _recover_from_stuck(
            motor=motor,
            loop_cfg=loop_cfg,
            latest_ultra_cm=ultra_state.dist_cm,
            last_motion_t=last_motion_t,
        )
        if recovered:
            _apply_reverse_recovery(
                shared_map=shared_map,
                car_id=car_id,
                motor=motor,
                loop_cfg=loop_cfg,
            )
            ultra_state.countdown = 0
        else:
            time.sleep(follower.cfg.dt)
        return None, last_motion_t, True

    close_obstacle = ultra_state.dist_cm is not None and ultra_state.dist_cm <= loop_cfg.close_obstacle_replan_cm
    if close_obstacle:
        recovered, last_motion_t = _recover_from_stuck(
            motor=motor,
            loop_cfg=loop_cfg,
            latest_ultra_cm=ultra_state.dist_cm,
            last_motion_t=last_motion_t,
        )
        if recovered:
            _apply_reverse_recovery(
                shared_map=shared_map,
                car_id=car_id,
                motor=motor,
                loop_cfg=loop_cfg,
            )
        ultra_state.countdown = 0
        return None, last_motion_t, True

    if ultra_state.countdown < int(loop_cfg.min_forward_ultra_countdown):
        motor.stop()
        _log_drive_status(
            pose_grid=pose_grid,
            goal_xy_grid=goal_xy_grid,
            d_goal=d_goal,
            dist_cm=ultra_state.dist_cm,
            ultra_countdown=ultra_state.countdown,
            path_len=len(current_path.waypoints),
            target_xy=(pose_grid.x, pose_grid.y),
            heading_error_deg=0.0,
            odom_step_cells=0.0,
            odom_yaw_deg=0.0,
            odom_steer_deg=0.0,
            drive_mode="ultra_hold",
            drive_speed=0,
            drive_duration_s=0.0,
            prefix="dead_reckon ",
        )
        time.sleep(max(loop_cfg.pause_after_stop_s, loop_cfg.ultra_timer_period_s))
        return None, last_motion_t, True

    return current_path, last_motion_t, False


def _build_drive_command(
    *,
    current_path: Path,
    pose_grid: Pose,
    d_goal: float,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
) -> DriveCommand:
    target_x, target_y, heading_error_deg, steer_deg = follower.tracking_command(current_path, pose_grid, d_goal)

    drive_duration_s = float(follower.cfg.dt)
    drive_speed = int(motor.cfg.speed)
    drive_mode = "track"
    if abs(heading_error_deg) >= float(loop_cfg.hard_turn_heading_deg):
        drive_duration_s *= float(loop_cfg.hard_turn_duration_scale)
        drive_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.hard_turn_speed_scale))))
        drive_mode = "hard_turn"
        steer_deg = clamp(
            float(loop_cfg.heading_steer_gain) * float(heading_error_deg),
            -float(follower.cfg.max_steer_deg),
            float(follower.cfg.max_steer_deg),
        )

    return DriveCommand(
        target_x=target_x,
        target_y=target_y,
        heading_error_deg=heading_error_deg,
        steer_deg=steer_deg,
        drive_mode=drive_mode,
        drive_speed=drive_speed,
        drive_duration_s=drive_duration_s,
    )


def _apply_drive_command(
    *,
    command: DriveCommand,
    shared_map: SharedMap,
    follower: PathFollower,
    motor: MotorController,
    car_id: int,
) -> DriveCommand:
    command.odom_step_cells = estimate_step_cells_for_duration(
        command.drive_duration_s,
        step_seconds=float(motor.cfg.step_seconds),
        speed=command.drive_speed,
        speed_ref=int(motor.cfg.speed),
    )
    motor.set_steering(command.steer_deg)
    command.odom_steer_deg = motor.get_applied_steering_deg()
    command.yaw_delta = estimate_ackermann_yaw_delta(
        command.odom_step_cells,
        command.odom_steer_deg,
        float(follower.cfg.wheelbase),
    )
    command.odom_yaw_deg = math.degrees(command.yaw_delta)

    motor.forward_for(command.drive_duration_s, speed=command.drive_speed)
    integrate_dead_reckoning(
        shared_map=shared_map,
        car_id=car_id,
        forward_step=command.odom_step_cells,
        yaw_delta=command.yaw_delta,
    )
    return command


def _log_drive_status(
    *,
    pose_grid: Pose,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    dist_cm: Optional[float],
    ultra_countdown: int,
    path_len: int,
    target_xy: Tuple[float, float],
    heading_error_deg: float,
    odom_step_cells: float,
    odom_yaw_deg: float,
    odom_steer_deg: float,
    drive_mode: str,
    drive_speed: int,
    drive_duration_s: float,
    prefix: str = "",
) -> None:
    ultra_str = "None" if dist_cm is None else f"{round(dist_cm, 1)}cm"
    print(
        f"{prefix}"
        f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
        f"goal=({goal_xy_grid[0]},{goal_xy_grid[1]}) d={d_goal:.1f} "
        f"ultra={ultra_str} ultra_countdown={ultra_countdown} "
        f"path_n={path_len} target=({target_xy[0]:.1f},{target_xy[1]:.1f}) "
        f"head_err={heading_error_deg:.1f} "
        f"mode={drive_mode} spd={drive_speed} dur={drive_duration_s:.2f} "
        f"odom_step={odom_step_cells:.2f} odom_yaw={odom_yaw_deg:.1f} steer={odom_steer_deg:.1f} "
    )


def drive_to_goal(
    goal_xy_grid: Tuple[int, int],
    *,
    shared_map: SharedMap,
    planner: MovementPlanner,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    car_id: int = 0,
    timeout_s: float = 180.0,
    debug_show_grid: bool = False,
) -> bool:
    goal_xy_grid = (int(goal_xy_grid[0]), int(goal_xy_grid[1]))
    goal_target = TargetPoint(float(goal_xy_grid[0]), float(goal_xy_grid[1]))

    current_path: Optional[Path] = None
    ultra_state = UltrasonicState(
        dist_cm=None,
        countdown=ultrasonic_to_countdown(
            None,
            stop_cm=float(loop_cfg.ultra_stop_cm),
            caution_cm=float(loop_cfg.ultra_caution_cm),
        ),
        last_valid_cm=None,
        last_valid_t=0.0,
    )

    last_ultra_tick_t = 0.0
    t0 = time.time()
    last_motion_t = t0

    follower.reset()

    try:
        while (time.time() - t0) < timeout_s:
            now = time.time()
            # Check ultrasonic every ultra_timer_period_s
            if (now - last_ultra_tick_t) >= loop_cfg.ultra_timer_period_s:
                last_ultra_tick_t = now
                ultra_state = _tick_ultrasonic(
                    motor=motor,
                    shared_map=shared_map,
                    loop_cfg=loop_cfg,
                    ultra_state=ultra_state,
                    now=now,
                    car_id=car_id,
                )

            pose_grid = _ensure_pose(shared_map=shared_map, car_id=car_id)
            if pose_grid is None:
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue
            
            # Compute distance to goal
            d_goal = math.hypot(goal_xy_grid[0] - pose_grid.x, goal_xy_grid[1] - pose_grid.y)
            if d_goal <= follower.cfg.goal_tolerance:
                motor.stop()
                motor.mark_reached()
                return True

            current_path = _ensure_path(
                current_path=current_path,
                planner=planner,
                shared_map=shared_map,
                goal_target=goal_target,
                debug_show_grid=debug_show_grid,
                car_id=car_id,
            )
            current_path, last_motion_t, should_continue = _maybe_hold_or_reverse(
                current_path=current_path,
                pose_grid=pose_grid,
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                ultra_state=ultra_state,
                shared_map=shared_map,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
                car_id=car_id,
                last_motion_t=last_motion_t,
            )
            if should_continue:
                continue

            assert current_path is not None
            path_len = len(current_path.waypoints) if current_path is not None else 0
            command = _build_drive_command(
                current_path=current_path,
                pose_grid=pose_grid,
                d_goal=d_goal,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
            )
            command = _apply_drive_command(
                command=command,
                shared_map=shared_map,
                follower=follower,
                motor=motor,
                car_id=car_id,
            )

            _log_drive_status(
                pose_grid=pose_grid,
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                dist_cm=ultra_state.dist_cm,
                ultra_countdown=ultra_state.countdown,
                path_len=path_len,
                target_xy=(command.target_x, command.target_y),
                heading_error_deg=command.heading_error_deg,
                odom_step_cells=command.odom_step_cells,
                odom_yaw_deg=command.odom_yaw_deg,
                odom_steer_deg=command.odom_steer_deg,
                drive_mode=command.drive_mode,
                drive_speed=command.drive_speed,
                drive_duration_s=command.drive_duration_s,
                prefix="dead_reckon ",
            )
            last_motion_t = time.time()

    finally:
        motor.stop()

    return False


def main() -> None:
    shared_map = SharedMap()
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-7.0, -7.0))
    shared_map.set_pose(0, Pose(0.0, 0.0, 0.0))

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
        lookahead=8.0,
        wheelbase=2.2,
        goal_tolerance=1.25,
        steer_sign=1.0,
        max_steer_deg=motor.cfg.max_steer_deg,
        steer_alpha=0.25,
        steer_deadband_deg=2.0,
        steer_rate_limit_deg_per_tick=12.0,
        dock_distance_grid=8.0,
        dock_min_lookahead_grid=1.5,
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
        start_grid = shared_map.get_car_grid_position(0)
        route = build_equilateral_triangle_route(start_grid, side_cells=6)
        print("Drive route:", route)
        for goal in route[1:]:
            ok = drive_to_goal(
                goal,
                shared_map=shared_map,
                planner=planner,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
                timeout_s=180.0,
                debug_show_grid=True,
            )
            print(f"Reached {goal}:", ok)
            time.sleep(0.5)

    finally:
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
