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
    build_square_route,
    clamp,
)
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import (
    PathFollower,
    FollowerConfig,
    estimate_step_cells_for_duration,
    integrate_dead_reckoning,
    wrap_angle,
)
from model import TargetPoint, Path, Pose


@dataclass
class LoopConfig:
    cm_per_grid: float = 50.0                    # centimeters per cell
    ahead_cm_min: float = 5.0                    # closest mapped obstacle
    ahead_cm_max: float = 120.0                  # farthest mapped obstacle
    action_tick_s: float = 0.10                  # control loop tick
    ultra_stop_cm: float = 20.0                  # immediate obstacle stop
    ultra_caution_cm: float = 40.0               # obstacle caution band

    hard_turn_heading_deg: float = 60.0          # start hard turns
    pivot_turn_heading_deg: float = 90.0         # start pivot turns
    hard_turn_duration_scale: float = 0.55       # shorten hard turns
    hard_turn_speed_scale: float = 0.75          # slow hard turns
    pivot_turn_duration_s: float = 0.50          # pivot phase duration
    pivot_turn_speed_scale: float = 0.75         # pivot speed scale
    pivot_turn_steer_min_deg: float = 8.0        # minimum pivot steer
    pivot_turn_steer_gain: float = 0.14          # pivot steer gain
    escape_pivot_forward_scale: float = 0.65     # escape forward fraction
    escape_pivot_reverse_scale: float = 1.0      # escape reverse fraction
    heading_steer_gain: float = 0.32             # heading error gain


@dataclass
class UltrasonicState:
    dist_cm: Optional[float]
    countdown: int


@dataclass
class DriveCommand:
    target_x: float
    target_y: float
    heading_error_deg: float
    steer_deg: float
    drive_mode: str
    drive_duration_s: float
    odom_step_cells: float = 0.0
    odom_yaw_deg: float = 0.0
    odom_steer_deg: float = 0.0
    yaw_delta: float = 0.0


@dataclass
class MotionPhase:
    motion: str
    steer_deg: float
    speed: int
    ticks_remaining: int


@dataclass
class ActionState:
    drive_mode: str
    target_x: float
    target_y: float
    heading_error_deg: float
    path_len: int
    phases: list[MotionPhase]
    pivot_direction: float = 0.0
    replan_after: bool = False


def _duration_to_ticks(duration_s: float, tick_s: float) -> int:
    return max(1, int(math.ceil(max(0.0, float(duration_s)) / max(1e-6, float(tick_s)))))


def _tick_ultrasonic(
    *,
    motor: MotorController,
    shared_map: SharedMap,
    loop_cfg: LoopConfig,
    car_id: int,
) -> UltrasonicState:
    raw_dist_cm = read_ultrasonic_cm(motor)
    dist_cm = raw_dist_cm

    ultra_countdown = ultrasonic_to_countdown(
        dist_cm,
        stop_cm=float(loop_cfg.ultra_stop_cm),
        caution_cm=float(loop_cfg.ultra_caution_cm),
    )

    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is not None:
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
    )


def _direction_to_goal_sign(*, pose_grid: Pose, goal_xy_grid: Tuple[int, int]) -> float:
    heading_error = math.degrees(
        wrap_angle(math.atan2(goal_xy_grid[1] - pose_grid.y, goal_xy_grid[0] - pose_grid.x) - pose_grid.theta)
    )
    return 1.0 if heading_error >= 0.0 else -1.0


def _pivot_direction_sign(
    *,
    current_path: Optional[Path],
    pose_grid: Pose,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    follower: PathFollower,
) -> tuple[float, Tuple[float, float], float]:
    if current_path is None or len(current_path.waypoints) < 2:
        sign = _direction_to_goal_sign(pose_grid=pose_grid, goal_xy_grid=goal_xy_grid)
        return sign, (float(goal_xy_grid[0]), float(goal_xy_grid[1])), 0.0

    target_x, target_y, heading_error_deg = follower.tracking_geometry(current_path, pose_grid, d_goal)
    sign = 1.0 if heading_error_deg >= 0.0 else -1.0
    return sign, (target_x, target_y), heading_error_deg


def _build_pivot_action(
    *,
    current_path: Optional[Path],
    pose_grid: Pose,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    drive_mode: str,
    reverse_scale: float,
    forward_scale: float,
    replan_after: bool,
) -> ActionState:
    direction_sign, target_xy, heading_error_deg = _pivot_direction_sign(
        current_path=current_path,
        pose_grid=pose_grid,
        goal_xy_grid=goal_xy_grid,
        d_goal=d_goal,
        follower=follower,
    )
    pivot_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.pivot_turn_speed_scale))))
    tick_s = float(loop_cfg.action_tick_s)
    phases: list[MotionPhase] = []

    reverse_ticks = _duration_to_ticks(float(loop_cfg.pivot_turn_duration_s) * float(reverse_scale), tick_s)
    if reverse_ticks > 0:
        phases.append(MotionPhase("reverse", 0.0, pivot_speed, reverse_ticks))

    forward_ticks = _duration_to_ticks(float(loop_cfg.pivot_turn_duration_s) * float(forward_scale), tick_s)
    if forward_ticks > 0:
        phases.append(MotionPhase("forward", 0.0, pivot_speed, forward_ticks))

    return ActionState(
        drive_mode=drive_mode,
        target_x=target_xy[0],
        target_y=target_xy[1],
        heading_error_deg=heading_error_deg,
        path_len=0 if current_path is None else len(current_path.waypoints),
        phases=phases,
        pivot_direction=float(direction_sign),
        replan_after=replan_after,
    )


def _build_motion_action(
    *,
    current_path: Path,
    pose_grid: Pose,
    d_goal: float,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
) -> ActionState:
    target_x, target_y, heading_error_deg, steer_deg = follower.tracking_command(
        current_path,
        pose_grid,
        d_goal,
        steer_cap_deg=follower.cfg.max_steer_deg,
    )

    drive_duration_s = float(loop_cfg.action_tick_s)
    drive_speed = int(motor.cfg.speed)
    drive_mode = "track"
    near_goal = d_goal <= float(follower.cfg.dock_distance_grid)
    # If heading error is very large, pivot turn to get heading aligned
    if abs(heading_error_deg) >= float(loop_cfg.pivot_turn_heading_deg) or (
        near_goal and abs(heading_error_deg) >= float(loop_cfg.hard_turn_heading_deg)
    ):
        return _build_pivot_action(
            current_path=current_path,
            pose_grid=pose_grid,
            goal_xy_grid=(int(round(target_x)), int(round(target_y))),
            d_goal=d_goal,
            follower=follower,
            motor=motor,
            loop_cfg=loop_cfg,
            drive_mode="pivot_turn",
            reverse_scale=1.0,
            forward_scale=float(loop_cfg.escape_pivot_forward_scale) if near_goal else 1.0,
            replan_after=False,
        )
    # If heading error is moderately large, do a hard turn
    elif abs(heading_error_deg) >= float(loop_cfg.hard_turn_heading_deg):
        drive_duration_s *= float(loop_cfg.hard_turn_duration_scale)
        drive_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.hard_turn_speed_scale))))
        drive_mode = "hard_turn"
        steer_deg = clamp(
            float(loop_cfg.heading_steer_gain) * float(heading_error_deg),
            -follower.cfg.max_steer_deg,
            follower.cfg.max_steer_deg,
        )

    return ActionState(
        target_x=target_x,
        target_y=target_y,
        heading_error_deg=heading_error_deg,
        drive_mode=drive_mode,
        path_len=len(current_path.waypoints),
        phases=[MotionPhase("forward", steer_deg, drive_speed, _duration_to_ticks(drive_duration_s, loop_cfg.action_tick_s))],
    )


def _apply_action_tick(
    *,
    action: ActionState,
    shared_map: SharedMap,
    current_path: Optional[Path],
    pose_grid: Pose,
    goal_xy_grid: Tuple[int, int],
    d_goal: float,
    follower: PathFollower,
    motor: MotorController,
    loop_cfg: LoopConfig,
    car_id: int,
) -> tuple[Optional[ActionState], DriveCommand, bool]:
    phase = action.phases[0]
    tick_s = float(loop_cfg.action_tick_s)
    target_x = action.target_x
    target_y = action.target_y
    heading_error_deg = action.heading_error_deg
    steer_deg = phase.steer_deg

    if current_path is not None and len(current_path.waypoints) >= 2:
        target_x, target_y, heading_error_deg = follower.tracking_geometry(current_path, pose_grid, d_goal)
    else:
        target_x = float(goal_xy_grid[0])
        target_y = float(goal_xy_grid[1])
        heading_error_deg = math.degrees(
            wrap_angle(math.atan2(target_y - pose_grid.y, target_x - pose_grid.x) - pose_grid.theta)
        )

    if action.drive_mode in {"pivot_turn", "escape_pivot"}:
        steer_abs = clamp(
            float(loop_cfg.pivot_turn_steer_gain) * abs(float(heading_error_deg)),
            float(loop_cfg.pivot_turn_steer_min_deg),
            float(follower.cfg.max_steer_deg),
        )
        steer_sign = -float(action.pivot_direction) if phase.motion == "reverse" else float(action.pivot_direction)
        steer_deg = steer_sign * steer_abs
    elif action.drive_mode == "hard_turn":
        steer_deg = clamp(
            float(loop_cfg.heading_steer_gain) * float(heading_error_deg),
            -float(follower.cfg.max_steer_deg),
            float(follower.cfg.max_steer_deg),
        )
    elif action.drive_mode == "track":
        target_x, target_y, heading_error_deg, steer_deg = follower.tracking_command(
            current_path if current_path is not None else Path(waypoints=[TargetPoint(target_x, target_y)]),
            pose_grid,
            d_goal,
            steer_cap_deg=follower.cfg.max_steer_deg,
        )

    odom_step_mag = estimate_step_cells_for_duration(
        tick_s,
        action_tick_s=tick_s,
        speed=phase.speed,
        speed_ref=int(motor.cfg.speed),
    )
    motor.set_steering(steer_deg)
    odom_steer_deg = motor.get_applied_steering_deg()
    signed_step = odom_step_mag if phase.motion == "forward" else -odom_step_mag
    yaw_delta = follower.estimate_ackermann_yaw_delta(
        signed_step,
        odom_steer_deg,
    )
    if phase.motion == "forward":
        motor.forward_for(tick_s, speed=phase.speed)
    else:
        motor.backward_for(tick_s, speed=phase.speed)
    integrate_dead_reckoning(
        shared_map=shared_map,
        car_id=car_id,
        forward_step=signed_step,
        yaw_delta=yaw_delta,
    )

    command = DriveCommand(
        target_x=target_x,
        target_y=target_y,
        heading_error_deg=heading_error_deg,
        steer_deg=steer_deg,
        drive_mode=f"{action.drive_mode}_{phase.motion}" if len(action.phases) > 1 else action.drive_mode,
        drive_duration_s=tick_s,
        odom_step_cells=signed_step,
        odom_yaw_deg=math.degrees(yaw_delta),
        odom_steer_deg=odom_steer_deg,
        yaw_delta=yaw_delta,
    )

    phase.ticks_remaining -= 1
    if phase.ticks_remaining <= 0:
        action.phases.pop(0)

    should_replan = False
    if not action.phases:
        # If just finished a pivot turn, sync the follower heading to the actual motor
        if action.drive_mode in {"pivot_turn", "escape_pivot"}:
            follower.sync_to_motor_steering()
        should_replan = bool(action.replan_after)
        return None, command, should_replan

    return action, command, False


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
    drive_mode: str,
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
        f"mode={drive_mode} "
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
    )

    last_ultra_tick_t = 0.0
    t0 = time.time()
    active_action: Optional[ActionState] = None
    follower.reset()

    try:
        while (time.time() - t0) < timeout_s:
            now = time.time()
            # Check ultrasonic every action tick
            if (now - last_ultra_tick_t) >= loop_cfg.action_tick_s:
                last_ultra_tick_t = now
                ultra_state = _tick_ultrasonic(
                    motor=motor,
                    shared_map=shared_map,
                    loop_cfg=loop_cfg,
                    car_id=car_id,
                )

            pose_grid = shared_map.get_pose(car_id, frame="grid")
            if pose_grid is None:
                integrate_dead_reckoning(
                    shared_map=shared_map,
                    car_id=car_id,
                    forward_step=0.0,
                    yaw_delta=0.0,
                )
                pose_grid = shared_map.get_pose(car_id, frame="grid")
            
            # Compute distance to goal
            d_goal = math.hypot(goal_xy_grid[0] - pose_grid.x, goal_xy_grid[1] - pose_grid.y)
            if d_goal <= follower.cfg.goal_tolerance:
                motor.stop()
                motor.mark_reached()
                return True

            if active_action is None:
                if current_path is None:
                    current_path = planner.plan_to_target(
                        target=goal_target,
                        shared_map=shared_map,
                        target_frame="grid",
                    )
                    if debug_show_grid:
                        frame = shared_map.render_grid_debug_view(
                            car_id=car_id,
                            path=current_path,
                            target=goal_target,
                            cell_px=14,
                        )
                        cv2.imshow("Planning debug", frame)
                        cv2.waitKey(1)

                # pivot turn on obstacle
                close_obstacle = ultra_state.dist_cm is not None and ultra_state.dist_cm <= loop_cfg.ultra_stop_cm
                if current_path is None or len(current_path.waypoints) < 2 or close_obstacle:
                    active_action = _build_pivot_action(
                        current_path=current_path,
                        pose_grid=pose_grid,
                        goal_xy_grid=goal_xy_grid,
                        d_goal=d_goal,
                        follower=follower,
                        motor=motor,
                        loop_cfg=loop_cfg,
                        drive_mode="escape_pivot",
                        reverse_scale=float(loop_cfg.escape_pivot_reverse_scale),
                        forward_scale=float(loop_cfg.escape_pivot_forward_scale),
                        replan_after=True,
                    )
                
                if active_action is None:
                    active_action = _build_motion_action(
                        current_path=current_path,
                        pose_grid=pose_grid,
                        d_goal=d_goal,
                        follower=follower,
                        motor=motor,
                        loop_cfg=loop_cfg,
                    )

            path_len = active_action.path_len
            active_action, command, should_replan = _apply_action_tick(
                action=active_action,
                shared_map=shared_map,
                current_path=current_path,
                pose_grid=pose_grid,
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
                car_id=car_id,
            )
            if should_replan:
                current_path = None

            _log_drive_status(
                pose_grid=pose_grid,
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                dist_cm=ultra_state.dist_cm,
                ultra_countdown=ultra_state.countdown,
                path_len=path_len,
                target_xy=(command.target_x, command.target_y),
                heading_error_deg=command.heading_error_deg,
                drive_mode=command.drive_mode,
                prefix="dead_reckon ",
            )

            if debug_show_grid:
                live_pose = shared_map.get_pose(car_id, frame="grid")
                live_path_len = 0 if current_path is None else len(current_path.waypoints)
                live_info = [
                    (
                        f"mode={command.drive_mode} d={d_goal:.2f} "
                        f"head_err={command.heading_error_deg:.1f} steer={command.odom_steer_deg:.1f}"
                    ),
                    (
                        f"ultra={'None' if ultra_state.dist_cm is None else f'{ultra_state.dist_cm:.1f}cm'} "
                        f"countdown={ultra_state.countdown} path_n={live_path_len}"
                    ),
                ]
                if live_pose is not None:
                    live_info.append(
                        f"pose=({live_pose.x:.2f},{live_pose.y:.2f},{live_pose.theta:.2f})"
                    )
                frame = shared_map.render_grid_debug_view(
                    car_id=car_id,
                    path=current_path,
                    target=goal_target,
                    control_target=TargetPoint(command.target_x, command.target_y),
                    cell_px=14,
                    info_lines=live_info,
                )
                cv2.imshow("Planning debug", frame)
                cv2.waitKey(1)
    finally:
        motor.stop()

    return False


def main() -> None:
    shared_map = SharedMap()
    shared_map.configure_grid(
        size=(30, 30),
        resolution=1.0,
    )
    shared_map.set_pose(0, Pose(0.0, 0.0, 0.0))

    planner = MovementPlanner(
        planning_cfg=PlanningConfig(
            include_slam_points=False,            # ignore SLAM points
            inflation_radius_cells=0,             # no obstacle inflation
            nudge_start_goal=True,                # shift blocked start/end
        )
    )

    motor = MotorController(MotorConfig(
        speed=26,                                 # default drive speed
        settle_seconds=0.01,                      # servo settle pause
    ))

    follower = PathFollower(motor, FollowerConfig(
        lookahead=5.0,                            # pure pursuit lookahead distance
        wheelbase=0.7,                            # front to back wheel wheelbase
        goal_tolerance=0.6,                       # goal reached radius
        steer_sign=1.0,                           # follower steering sign
        steer_alpha=0.25,                         # steering smoother
        steer_deadband_deg=2.0,                   # ignore tiny steer changes
        steer_rate_limit_deg_per_tick=12.0,       # max steer change
        dock_distance_grid=8.0,                   # near goal threshold
        dock_min_lookahead_grid=1.5,              # minimum dock lookahead
    ))

    loop_cfg = LoopConfig(
        cm_per_grid=14,                           # centimeters per cell
    )

    try:
        start_grid = shared_map.get_car_grid_position(0)
        # route = build_equilateral_triangle_route(start_grid, side_cells=6)
        # route = build_square_route(start_grid, side_cells=6)
        route = [
            (start_grid[0], start_grid[1]),
            (start_grid[0] + 6, start_grid[1])
        ]
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
