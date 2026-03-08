# main.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from car_tools.camera_input import PiCarXCamera, CameraConfig
from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Pose, Path  # Obstacle optional


# ---------------- Ultrasonic helpers ----------------

def read_ultrasonic_cm(motor: MotorController) -> Optional[float]:
    """
    Returns distance in cm, or None if invalid/unavailable.

    PiCar-X examples use px.ultrasonic.read().
    DeepWiki notes px.get_distance() is a wrapper around ultrasonic.read() and returns cm
    """
    px = getattr(motor, "px", None)
    if px is None:
        return None
    d = float(px.get_distance())
    if not np.isfinite(d) or d <= 0 or d > 500:
        return None
    return d


@dataclass
class ReplanConfig:
    # replan cadence
    replan_period_s: float = 3.0
    replan_min_interval_s: float = 1 # replan once a second if close

    # ultrasonic thresholds (cm)
    # SunFounder obstacle avoidance lesson uses SafeDistance=40, DangerDistance=20
    ultra_replan_cm: float = 40.0       # "something ahead" => replan soon
    ultra_emergency_cm: float = 20.0    # "too close" => stop/back up + replan

    # map injection: cm-to-grid scaling
    cm_per_grid: float = 50.0

    # how far ahead to drop the virtual obstacle (cm from ultrasonic sensor)
    ahead_cm_min: float = 5.0
    ahead_cm_max: float = 120.0

    # if no obstacle for a while, clear the virtual obstacle
    ultra_clear_cm: float = 60.0

    # if emergency, back up a bit
    emergency_backup_s: float = 1
    emergency_wait_high_conf_timeout_s: float = 3.0
    emergency_wait_tick_s: float = 0.05

    # pose-confidence gating
    min_pose_conf_for_replan: float = 0.55
    blurry_hold_after_no_replan_s: float = 3.0
    blurry_hold_pause_s: float = 0.20


# ---------------- Driving loop ----------------

def drive_to_goal_with_sparse_replan(
    goal_xy_grid: Tuple[int, int],
    *,
    shared_map: SharedMap,
    planner: MovementPlanner,
    follower: PathFollower,
    motor: MotorController,
    slam: MonocularVSLAM,
    camera: PiCarXCamera,
    replan_cfg: ReplanConfig,
    car_id: int = 0,
    timeout_s: float = 180.0,
    debug_show_keypoints: bool = False,
) -> bool:
    goal_gx, goal_gy = int(goal_xy_grid[0]), int(goal_xy_grid[1])

    # Keep one "virtual obstacle" entry updated
    ULTRA_OB_ID = "ultra_front"
    steer_filt = 0.0
    steer_alpha = 0.25      # 0.2–0.35 good
    steer_deadband = 2.0    # degrees; ignore tiny noise
    steer_rate_limit = 12.0 # deg per control tick max change

    current_path: Optional[Path] = None
    emergency_replan_required = False
    last_plan_t: float = 0.0
    last_ultra_t: float = 0.0

    t0 = time.time()
    last_print = 0.0

    def slam_step_per_tick() -> float:
        # motor.step_seconds ~ time to travel 1 grid cell
        return float(follower.cfg.dt) / float(max(1e-6, motor.cfg.step_seconds))

    def maybe_update_ultra_obstacle(pose_world: Pose, dist_cm: Optional[float]) -> bool:
        """
        Updates shared_map.obstacles[ULTRA_OB_ID] if dist is close enough.
        Returns True if an obstacle is considered "present".
        """
        nonlocal last_ultra_t

        if dist_cm is None:
            # no update; keep last obstacle unless it's stale
            return False

        # Clear far readings
        if dist_cm >= replan_cfg.ultra_clear_cm:
            if ULTRA_OB_ID in shared_map.obstacles:
                shared_map.obstacles.pop(ULTRA_OB_ID, None)
            return False

        # Replan range, inject an obstacle ahead
        if dist_cm <= replan_cfg.ultra_replan_cm:
            last_ultra_t = time.time()

            # Convert distance->cells (clamped in cm first, then converted).
            d_cm = float(np.clip(dist_cm, replan_cfg.ahead_cm_min, replan_cfg.ahead_cm_max))
            d_cells = d_cm / max(1e-6, replan_cfg.cm_per_grid)

            # Place obstacle "ahead" along heading in WORLD(nav) frame
            ox = float(pose_world.x + d_cells * math.cos(pose_world.theta))
            oy = float(pose_world.y + d_cells * math.sin(pose_world.theta))

            # Fixed car footprint (not exposed as config): 20cm long x 14cm wide.
            # Use half-diagonal as a conservative collision radius.
            footprint_radius_cm = math.hypot(10.0, 7.0)
            r_world = (footprint_radius_cm / max(1e-6, replan_cfg.cm_per_grid)) * float(shared_map.grid.resolution)

            # Store as a simple duck-typed object with x,y,radius,obstacle_id
            class _TmpObstacle:
                __slots__ = ("obstacle_id", "x", "y", "radius")
                def __init__(self, obstacle_id: str, x: float, y: float, radius: float):
                    self.obstacle_id = obstacle_id
                    self.x = x
                    self.y = y
                    self.radius = radius

            shared_map.obstacles[ULTRA_OB_ID] = _TmpObstacle(ULTRA_OB_ID, ox, oy, r_world)
            return True

        return False

    def need_replan(dist_cm: Optional[float], obstacle_present: bool) -> bool:
        nonlocal last_plan_t

        now = time.time()
        since_plan = now - last_plan_t

        # 1) No path yet
        if current_path is None or len(current_path.waypoints) < 2:
            return True

        # 2) Periodic replan
        if since_plan >= replan_cfg.replan_period_s:
            return True

        # 3) Immediate-ish replan when ultrasonic sees something close
        # throttle to avoid replanning every dt due to noisy readings
        if dist_cm is not None and dist_cm <= replan_cfg.ultra_replan_cm:
            if since_plan >= replan_cfg.replan_min_interval_s:
                return True

        # 4) If current path is blocked by the current occupancy grid
        if shared_map.is_path_blocked(current_path):
            if since_plan >= replan_cfg.replan_min_interval_s:
                return True

        return False

    def slam_quality() -> Tuple[float, bool, bool, int, float, float]:
        """
        Returns:
          confidence, high_confidence, blurry, inlier_count, contrast, blue_ratio
        """
        try:
            status = slam.get_status()
        except Exception:
            return 1.0, True, False, 0, 0.0, 1.0

        conf = float(getattr(status, "confidence", 0.0))
        high_conf = bool(
            getattr(status, "high_confidence", False)
            and conf >= float(replan_cfg.min_pose_conf_for_replan)
        )
        blurry = bool(getattr(status, "blurry", False))
        inliers = int(getattr(status, "inlier_count", 0))
        contrast = float(getattr(status, "contrast", 0.0))
        blue_ratio = float(getattr(status, "blue_ratio", 1.0))
        return conf, high_conf, blurry, inliers, contrast, blue_ratio

    def wait_for_high_conf_slam(timeout_s: float) -> bool:
        """
        Hold position, keep SLAM updating, and wait until status becomes high-confidence.
        """
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            fr = camera.read()
            if fr is None:
                time.sleep(max(0.01, replan_cfg.emergency_wait_tick_s))
                continue

            try:
                slam.tick(fr, translation_step=0.0)
            except TypeError:
                slam.cfg.translation_step = 0.0
                slam.tick(fr)

            if debug_show_keypoints:
                dbg = slam.get_debug_keypoints_frame()
                if dbg is not None:
                    cv2.imshow("SLAM keypoints", dbg)
                    cv2.waitKey(1)

            _conf, high_conf, _blurry, _inliers, _contrast, _blue_ratio = slam_quality()
            if high_conf:
                return True

            time.sleep(max(0.01, replan_cfg.emergency_wait_tick_s))

        return False

    try:
        while (time.time() - t0) < timeout_s:
            # ---- SLAM update ----
            frame = camera.read()
            if frame is None:
                motor.stop()
                time.sleep(0.05)
                continue

            step = slam_step_per_tick()
            try:
                slam.tick(frame, translation_step=step)
            except TypeError:
                slam.cfg.translation_step = float(step)
                slam.tick(frame)

            if debug_show_keypoints:
                dbg = slam.get_debug_keypoints_frame()
                if dbg is not None:
                    cv2.imshow("SLAM keypoints", dbg)
                    cv2.waitKey(1)

            # ---- Pose ----
            pose_grid = shared_map.get_pose(car_id, frame="grid")
            pose_world = shared_map.get_pose(car_id, frame="world")
            if pose_grid is None or pose_world is None:
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue

            # ---- Goal check ----
            d_goal = math.hypot(goal_gx - pose_grid.x, goal_gy - pose_grid.y)
            if d_goal <= follower.cfg.goal_tolerance:
                motor.stop()
                motor.mark_reached()
                return True

            pose_conf, pose_high_conf, pose_blurry, pose_inliers, pose_contrast, pose_blue_ratio = slam_quality()
            now = time.time()
            since_plan = (now - last_plan_t) if last_plan_t > 0 else 0.0
            hold_for_pose_recovery = (
                current_path is not None
                and pose_blurry
                and (not pose_high_conf)
                and since_plan >= replan_cfg.blurry_hold_after_no_replan_s
            )

            if hold_for_pose_recovery:
                motor.stop()
                time.sleep(max(0.01, replan_cfg.blurry_hold_pause_s))
                if now - last_print > 1.0:
                    last_print = now
                    print(
                        f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
                        f"goal=({goal_gx},{goal_gy}) d={d_goal:.1f} "
                        f"pose_conf={pose_conf:.2f} inliers={pose_inliers} blurry={pose_blurry} "
                        f"ctr={pose_contrast:.1f} blue={pose_blue_ratio:.2f} "
                        f"hold=pose_recovery replan_in={(replan_cfg.replan_period_s - since_plan):.1f}s"
                    )
                continue

            # ---- Ultrasonic check ----
            dist_cm = read_ultrasonic_cm(motor)

            # Emergency behavior
            if dist_cm is not None and dist_cm <= replan_cfg.ultra_emergency_cm:
                emergency_replan_required = True
                motor.stop()
                # back up a touch to create space
                if replan_cfg.emergency_backup_s > 0:
                    try:
                        motor.backward_for(replan_cfg.emergency_backup_s, speed=max(15, motor.cfg.speed // 2))
                    except Exception:
                        pass

                motor.stop()

                # Use fresh ultrasonic + pose after backing up.
                dist_after_backup = read_ultrasonic_cm(motor)
                if dist_after_backup is None:
                    dist_after_backup = dist_cm

                # Wait for high-confidence SLAM before we commit to a new path.
                got_high_conf = wait_for_high_conf_slam(replan_cfg.emergency_wait_high_conf_timeout_s)
                pose_world_after = shared_map.get_pose(car_id, frame="world")
                if pose_world_after is not None:
                    _ = maybe_update_ultra_obstacle(pose_world_after, dist_after_backup)
                else:
                    _ = maybe_update_ultra_obstacle(pose_world, dist_after_backup)

                if got_high_conf:
                    goal = TargetPoint(float(goal_gx), float(goal_gy))
                    start_plan_t = time.time()
                    current_path = planner.repath_to_target(
                        current_path=current_path,
                        target=goal,
                        shared_map=shared_map,
                        target_frame="grid",
                        force_replan=True,
                    )
                    last_plan_t = time.time()
                    print(f"time to plan = {start_plan_t-last_plan_t}")
                    emergency_replan_required = False

                # Always skip motion this loop after an emergency event.
                continue

            obstacle_present = maybe_update_ultra_obstacle(pose_world, dist_cm)

            # ---- Replan
            wants_replan = need_replan(dist_cm, obstacle_present)
            can_replan = pose_high_conf or (current_path is None)
            if wants_replan and can_replan:
                goal = TargetPoint(float(goal_gx), float(goal_gy))
                current_path = planner.repath_to_target(
                    current_path=current_path,
                    target=goal,
                    shared_map=shared_map,
                    target_frame="grid",
                )
                last_plan_t = time.time()

            if emergency_replan_required:
                motor.stop()
                if pose_high_conf:
                    goal = TargetPoint(float(goal_gx), float(goal_gy))
                    current_path = planner.repath_to_target(
                        current_path=current_path,
                        target=goal,
                        shared_map=shared_map,
                        target_frame="grid",
                        force_replan=True,
                    )
                    last_plan_t = time.time()
                    emergency_replan_required = False
                else:
                    time.sleep(max(0.01, replan_cfg.emergency_wait_tick_s))
                    continue

            if current_path is None or len(current_path.waypoints) < 2:
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue

            # ---- Track current path ----
            # Slightly reduce lookahead near goal for docking
            Ld = float(follower.cfg.lookahead)
            if d_goal < 12.0:
                Ld = max(6.0, min(Ld, 0.8 * d_goal))

            tx, ty = follower._lookahead_point(current_path, pose_grid, Ld)
            delta = follower._pure_pursuit_delta(pose_grid, target_x=tx, target_y=ty)
            steer_deg = math.degrees(delta)

            # Slow down when steering is large
            steer_abs = abs(steer_deg)
            maxs = float(motor.cfg.max_steer_deg)
            base = int(motor.cfg.speed)
            min_spd = max(15, int(0.55 * base))
            spd = int(base - (base - min_spd) * min(1.0, steer_abs / max(1e-6, maxs)))

            steer_cmd = steer_deg

            # deadband (ignore tiny noise)
            if abs(steer_cmd) < steer_deadband:
                steer_cmd = 0.0

            # EMA smoothing on steering command
            steer_filt = (1.0 - steer_alpha) * steer_filt + steer_alpha * steer_cmd

            # rate limit (per tick)
            dmax = float(steer_rate_limit)
            steer_filt = max(steer_cmd - dmax, min(steer_cmd + dmax, steer_filt))

            # Use a constant speed while tuning (speed modulation causes jerk)
            spd = int(motor.cfg.speed)

            motor.set_steering(steer_filt)
            motor.forward_for(follower.cfg.dt, speed=spd)

            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                print(
                    f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
                    f"goal=({goal_gx},{goal_gy}) d={d_goal:.1f} "
                    f"ultra={None if dist_cm is None else round(dist_cm,1)}cm "
                    f"pose_conf={pose_conf:.2f} inliers={pose_inliers} blurry={pose_blurry} "
                    f"ctr={pose_contrast:.1f} blue={pose_blue_ratio:.2f} "
                    f"replan_in={(replan_cfg.replan_period_s - (now - last_plan_t)):.1f}s"
                )

    finally:
        motor.stop()

    return False


# ---------------- Main ----------------

def main() -> None:
    # Shared map / grid alignment
    shared_map = SharedMap()
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-2.0, -2.0))

    # Camera (Vilib)
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

    # SLAM
    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)
    slam = MonocularVSLAM(
        intr, shared_map, car_id=0,
        cfg=VslamConfig(
            input_color_order=cam.color_order,
            debug_draw_keypoints=True,
            debug_draw_matches=True,
            translation_step=0.2,     # still overridden per tick
            forward_sign=-1.0,        # IMPORTANT: you observed x decreasing
            pose_ema_alpha=0.25,      # reduces jerk a lot
            gray_use_clahe=True,
            gray_clahe_clip_limit=2.5,
            gray_clahe_tile_size=8,
            auto_white_balance=True,
        )
    )

    motor = MotorController(MotorConfig(
        speed=26,
        step_seconds=0.40,           # tune later
        brake_between_steps=False,   # continuous motion
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
    ))

    replan_cfg = ReplanConfig(
        replan_period_s=3.0,
        replan_min_interval_s=0.9,
        ultra_replan_cm=40.0,
        ultra_emergency_cm=20.0,
        cm_per_grid=50.0,
        emergency_backup_s=0.20,
    )

    # Planner (inflate obstacles; simplify path)
    planner = MovementPlanner(planning_cfg=PlanningConfig(
        include_slam_points=False,
        inflation_radius_cells=0,
        simplify_path=True,
        nudge_start_goal=True,
    ), world_size=(50, 50))

    try:
        # Warm-up (let camera exposure settle)
        t_warm = time.time()
        while time.time() - t_warm < 1.0:
            fr = cam.read()
            if fr is not None:
                try:
                    slam.tick(fr, translation_step=0.0)
                except TypeError:
                    slam.cfg.translation_step = 0.0
                    slam.tick(fr)
            time.sleep(0.02)

        print("Drive (2,2) -> (45,45) with sparse replanning + ultrasonic triggers...")
        ok = drive_to_goal_with_sparse_replan(
            (45, 45),
            shared_map=shared_map,
            planner=planner,
            follower=follower,
            motor=motor,
            slam=slam,
            camera=cam,
            replan_cfg=replan_cfg,
            timeout_s=180.0,
            debug_show_keypoints=True,
        )
        print("Reached (45,45):", ok)

        # Square (inner square)
        square = [(45, 45), (5, 45), (5, 5), (45, 5), (45, 45)]
        print("Drive square:", square)
        for goal in square[1:]:
            ok = drive_to_goal_with_sparse_replan(
                goal,
                shared_map=shared_map,
                planner=planner,
                follower=follower,
                motor=motor,
                slam=slam,
                camera=cam,
                replan_cfg=replan_cfg,
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
