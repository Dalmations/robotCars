# main.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from camera_input import PiCarXCamera, CameraConfig
from coordination.shared_map import SharedMap
from movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Pose, Path  # Obstacle optional


# ---------------- Ultrasonic helpers ----------------

def read_ultrasonic_cm(motor: MotorController) -> Optional[float]:
    """
    Returns distance in cm, or None if invalid/unavailable.

    PiCar-X examples use px.ultrasonic.read(). :contentReference[oaicite:2]{index=2}
    DeepWiki notes px.get_distance() is a wrapper around ultrasonic.read() and returns cm. :contentReference[oaicite:3]{index=3}
    """
    px = getattr(motor, "px", None)
    if px is None:
        return None

    # Prefer get_distance() if present
    if hasattr(px, "get_distance"):
        try:
            d = float(px.get_distance())
            if not np.isfinite(d) or d <= 0 or d > 500:
                return None
            return d
        except Exception:
            pass

    # Fallback: ultrasonic.read()
    try:
        u = getattr(px, "ultrasonic", None)
        if u is not None and hasattr(u, "read"):
            d = float(u.read())
            if not np.isfinite(d) or d <= 0 or d > 500:
                return None
            return d
    except Exception:
        pass

    return None


@dataclass
class ReplanConfig:
    # replan cadence
    replan_period_s: float = 3.0
    replan_min_interval_s: float = 0.7  # avoid thrashing on noisy readings

    # ultrasonic thresholds (cm)
    # SunFounder obstacle avoidance lesson uses SafeDistance=40, DangerDistance=20. :contentReference[oaicite:4]{index=4}
    ultra_replan_cm: float = 40.0       # "something ahead" => replan soon
    ultra_emergency_cm: float = 20.0    # "too close" => stop/back up + replan

    # map injection: cm-to-grid scaling
    cm_per_grid: float = 10.0           # TUNE: how many cm is 1 grid cell
    obstacle_radius_cells: float = 2.0  # inflated by planner too; keep small-ish

    # how far ahead to drop the virtual obstacle (min/max in grid cells)
    ahead_cells_min: float = 2.0
    ahead_cells_max: float = 6.0

    # if no obstacle for a while, clear the virtual obstacle
    ultra_clear_cm: float = 60.0

    # optional: if emergency, back up a bit
    emergency_backup_s: float = 0.25


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

    current_path: Optional[Path] = None
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

        # If it's within "replan range", inject an obstacle ahead
        if dist_cm <= replan_cfg.ultra_replan_cm:
            last_ultra_t = time.time()

            # Convert distance->cells (clamped)
            d_cells = dist_cm / max(1e-6, replan_cfg.cm_per_grid)
            d_cells = float(np.clip(d_cells, replan_cfg.ahead_cells_min, replan_cfg.ahead_cells_max))

            # Place obstacle "ahead" along heading in WORLD(nav) frame
            ox = float(pose_world.x + d_cells * math.cos(pose_world.theta))
            oy = float(pose_world.y + d_cells * math.sin(pose_world.theta))
            r_world = float(replan_cfg.obstacle_radius_cells) * float(shared_map.grid.resolution)

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

            # ---- Ultrasonic check ----
            dist_cm = read_ultrasonic_cm(motor)

            # Emergency behavior
            if dist_cm is not None and dist_cm <= replan_cfg.ultra_emergency_cm:
                motor.stop()
                # back up a touch to create space
                if replan_cfg.emergency_backup_s > 0:
                    try:
                        motor.backward_for(replan_cfg.emergency_backup_s, speed=max(15, motor.cfg.speed // 2))
                    except Exception:
                        pass

            obstacle_present = maybe_update_ultra_obstacle(pose_world, dist_cm)

            # ---- Replan (sparsely) ----
            if need_replan(dist_cm, obstacle_present):
                goal = TargetPoint(float(goal_gx), float(goal_gy))
                current_path = planner.plan_to_target(goal, shared_map, target_frame="grid")
                last_plan_t = time.time()

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

            motor.set_steering(steer_deg)
            motor.forward_for(follower.cfg.dt, speed=spd)

            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                print(
                    f"pose_g=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f}) "
                    f"goal=({goal_gx},{goal_gy}) d={d_goal:.1f} "
                    f"ultra={None if dist_cm is None else round(dist_cm,1)}cm "
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
        vflip=False,
        hflip=False,
        display_local=False,
        display_web=False,
        frame_size=(640, 480),
        output_color_order="rgb",
        frame_rate=30,
        copy_on_read=True,
    ))
    cam.start()

    # SLAM
    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)
    slam = MonocularVSLAM(
        intr,
        shared_map,
        car_id=0,
        cfg=VslamConfig(
            input_color_order=cam.color_order,
            debug_draw_keypoints=False,
            debug_draw_matches=False,
            translation_step=0.2,  # overridden each tick
        )
    )

    # Planner (inflate obstacles; simplify path)
    planner = MovementPlanner(planning_cfg=PlanningConfig(
        include_slam_points=False,
        inflation_radius_cells=2,
        simplify_path=True,
        nudge_start_goal=True,
    ), world_size=(50, 50))

    # Motor (smooth motion: don't brake between dt steps)
    motor = MotorController(MotorConfig(
        speed=30,
        step_seconds=0.35,            # tune
        brake_between_steps=False,    # smoother than stop-start
        steering_slew_deg_per_s=140.0,
        steer_sign=1.0,
        steer_offset_deg=0.0,
        max_steer_deg=35.0,
        settle_seconds=0.02,
    ))

    follower = PathFollower(motor, FollowerConfig(
        dt=0.10,          # less CPU and less twitch; also 10Hz ultrasonic is fine :contentReference[oaicite:5]{index=5}
        lookahead=12.0,
        wheelbase=2.5,
        goal_tolerance=3.0,
        pose_frame="grid",
        steer_sign=1.0,
        max_steer_deg=motor.cfg.max_steer_deg,
    ))

    replan_cfg = ReplanConfig(
        replan_period_s=3.0,
        ultra_replan_cm=40.0,
        ultra_emergency_cm=20.0,
        cm_per_grid=10.0,            # tune to your floor scale
        obstacle_radius_cells=2.0,
    )

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
            debug_show_keypoints=False,
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
                debug_show_keypoints=False,
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