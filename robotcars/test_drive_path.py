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
from car_tools.movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from model import TargetPoint, Path, Pose


@dataclass
class LoopConfig:
    cm_per_grid: float = 50.0
    ahead_cm_min: float = 5.0
    ahead_cm_max: float = 120.0
    ultra_timer_period_s: float = 0.1

    pause_after_stop_s: float = 0.20
    close_obstacle_replan_cm: float = 15.0
    stuck_timeout_s: float = 3.0
    stuck_reverse_s: float = 0.30
    stuck_reverse_speed_scale: float = 0.6


def _estimate_translation_step_cells_for_duration(
    duration_s: float,
    motor: MotorController,
    *,
    speed: Optional[int] = None,
) -> float:
    base_step = float(duration_s) / float(max(1e-6, motor.cfg.step_seconds))
    speed_ref = max(1.0, float(motor.cfg.speed))
    speed_cmd = float(motor.cfg.speed if speed is None else max(0, min(100, int(speed))))
    return base_step * (speed_cmd / speed_ref)


def _estimate_translation_step_cells(follower: PathFollower, motor: MotorController) -> float:
    return _estimate_translation_step_cells_for_duration(follower.cfg.dt, motor, speed=int(motor.cfg.speed))


def _estimate_ackermann_yaw_delta(step_cells: float, steer_deg: float, follower: PathFollower) -> float:
    wheelbase = max(1e-6, float(follower.cfg.wheelbase))
    steer_rad = math.radians(float(steer_deg))
    return float(step_cells) * math.tan(steer_rad) / wheelbase


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _integrate_dead_reckoning(
    *,
    shared_map: SharedMap,
    car_id: int,
    forward_step: float,
    yaw_delta: float,
) -> Pose:
    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is None:
        pose_world = Pose(0.0, 0.0, 0.0)

    step = float(forward_step)
    dtheta = float(yaw_delta)
    theta_mid = float(pose_world.theta) + 0.5 * dtheta

    next_pose = Pose(
        x=float(pose_world.x + step * math.cos(theta_mid)),
        y=float(pose_world.y + step * math.sin(theta_mid)),
        theta=float(_wrap_angle(float(pose_world.theta) + dtheta)),
    )
    shared_map.set_pose(car_id, next_pose)
    return next_pose


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
        )

    return dist_cm, ultra_countdown


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


def _log_drive_status(
    *,
    pose_grid,
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
    latest_ultra_cm: Optional[float] = None
    ultra_countdown = ultrasonic_to_countdown(None)

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
                latest_ultra_cm, ultra_countdown = _tick_ultrasonic(
                    motor=motor,
                    shared_map=shared_map,
                    loop_cfg=loop_cfg,
                    ultra_countdown=ultra_countdown,
                    car_id=car_id,
                )

            pose_grid = shared_map.get_pose(car_id, frame="grid")
            if pose_grid is None:
                _integrate_dead_reckoning(
                    shared_map=shared_map,
                    car_id=car_id,
                    forward_step=0.0,
                    yaw_delta=0.0,
                )
                pose_grid = shared_map.get_pose(car_id, frame="grid")
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

            path_blocked = current_path is not None and planner.is_obstructed(current_path, shared_map)

            # Recalculate when bootstrapping, when the ultrasonic demands it, or when the path is now blocked.
            needs_path = current_path is None or ultra_countdown <= 0 or path_blocked
            if needs_path:
                current_path = _plan_goal_path(
                    planner=planner,
                    shared_map=shared_map,
                    goal_target=goal_target,
                    debug_show_grid=debug_show_grid,
                    car_id=car_id,
                )
                ultra_countdown = max(1, ultrasonic_to_countdown(latest_ultra_cm))

            if current_path is None or len(current_path.waypoints) < 2:
                current_path = None
                recovered, last_motion_t = _recover_from_stuck(
                    motor=motor,
                    loop_cfg=loop_cfg,
                    latest_ultra_cm=latest_ultra_cm,
                    last_motion_t=last_motion_t,
                )
                if recovered:
                    reverse_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.stuck_reverse_speed_scale))))
                    _integrate_dead_reckoning(
                        shared_map=shared_map,
                        car_id=car_id,
                        forward_step=-_estimate_translation_step_cells_for_duration(
                            loop_cfg.stuck_reverse_s,
                            motor,
                            speed=reverse_speed,
                        ),
                        yaw_delta=0.0,
                    )
                    ultra_countdown = 0
                    continue
                time.sleep(follower.cfg.dt)
                continue

            close_obstacle = latest_ultra_cm is not None and latest_ultra_cm <= loop_cfg.close_obstacle_replan_cm
            if close_obstacle:
                current_path = None
                recovered, last_motion_t = _recover_from_stuck(
                    motor=motor,
                    loop_cfg=loop_cfg,
                    latest_ultra_cm=latest_ultra_cm,
                    last_motion_t=last_motion_t,
                )
                if recovered:
                    reverse_speed = max(1, int(round(float(motor.cfg.speed) * float(loop_cfg.stuck_reverse_speed_scale))))
                    _integrate_dead_reckoning(
                        shared_map=shared_map,
                        car_id=car_id,
                        forward_step=-_estimate_translation_step_cells_for_duration(
                            loop_cfg.stuck_reverse_s,
                            motor,
                            speed=reverse_speed,
                        ),
                        yaw_delta=0.0,
                    )
                ultra_countdown = 0
                continue

            steer_deg = follower.steering_command(current_path, pose_grid, d_goal)
            lookahead = follower.compute_lookahead(d_goal)
            target_x, target_y = follower.lookahead_point(current_path, pose_grid, lookahead)
            heading_error_deg = math.degrees(
                _wrap_angle(math.atan2(target_y - pose_grid.y, target_x - pose_grid.x) - pose_grid.theta)
            )
            path_len = len(current_path.waypoints) if current_path is not None else 0
            step_cells = _estimate_translation_step_cells(follower, motor)
            motor.set_steering(steer_deg)
            applied_steer_deg = motor.get_applied_steering_deg()
            expected_yaw_deg = math.degrees(_estimate_ackermann_yaw_delta(step_cells, applied_steer_deg, follower))

            _log_drive_status(
                pose_grid=pose_grid,
                goal_xy_grid=goal_xy_grid,
                d_goal=d_goal,
                dist_cm=latest_ultra_cm,
                ultra_countdown=ultra_countdown,
                path_len=path_len,
                target_xy=(target_x, target_y),
                heading_error_deg=heading_error_deg,
                odom_step_cells=step_cells,
                odom_yaw_deg=expected_yaw_deg,
                odom_steer_deg=applied_steer_deg,
                prefix="dead_reckon ",
            )

            motor.forward_for(follower.cfg.dt, speed=int(motor.cfg.speed))
            yaw_delta = _estimate_ackermann_yaw_delta(step_cells, applied_steer_deg, follower)
            _integrate_dead_reckoning(
                shared_map=shared_map,
                car_id=car_id,
                forward_step=step_cells,
                yaw_delta=yaw_delta,
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
        lookahead=14.0,
        wheelbase=2.2,
        goal_tolerance=1.25,
        pose_frame="grid",
        steer_sign=1.0,
        max_steer_deg=motor.cfg.max_steer_deg,
        steer_alpha=0.25,
        steer_deadband_deg=2.0,
        steer_rate_limit_deg_per_tick=12.0,
        dock_distance_grid=12.0,
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
        route = [(10, 10), (5, 10), (5, 5), (10, 5), (10, 10)]
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
