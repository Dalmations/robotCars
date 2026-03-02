# main.py
from __future__ import annotations

import time
import math
from typing import Tuple, Optional

import cv2

from camera_input import PiCarXCamera, CameraConfig
from coordination.shared_map import SharedMap
from movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Pose


def drive_to_grid_goal(
    goal: Tuple[int, int],
    *,
    shared_map: SharedMap,
    planner: MovementPlanner,
    follower: PathFollower,
    motor: MotorController,
    slam: MonocularVSLAM,
    camera: PiCarXCamera,
    car_id: int = 0,
    timeout_s: float = 180.0,
    debug_show_keypoints: bool = False,
) -> bool:
    gx, gy = int(goal[0]), int(goal[1])
    t0 = time.time()

    # Tie SLAM translation scale to your motor timing:
    # step_seconds ~ time to travel 1 grid cell at current speed (rough)
    def slam_step_per_tick() -> float:
        return float(follower.cfg.dt) / float(max(1e-6, motor.cfg.step_seconds))

    last_print = 0.0

    while (time.time() - t0) < timeout_s:
        # 1) Read latest frame from Vilib (RGB by default)
        frame = camera.read()
        if frame is None:
            motor.stop()
            time.sleep(0.05)
            continue

        # 2) SLAM tick (scale per tick)
        step = slam_step_per_tick()
        try:
            # If you added tick(..., translation_step=...) use this
            slam.tick(frame, translation_step=step)
        except TypeError:
            # Otherwise, update config dynamically
            slam.cfg.translation_step = float(step)
            slam.tick(frame)

        if debug_show_keypoints:
            dbg = slam.get_debug_keypoints_frame()
            if dbg is not None:
                cv2.imshow("SLAM keypoints", dbg)
                cv2.waitKey(1)

        # 3) Pose in grid frame
        pose: Optional[Pose] = shared_map.get_pose(car_id, frame="grid")
        if pose is None:
            motor.stop()
            time.sleep(follower.cfg.dt)
            continue

        # 4) Goal check
        d_goal = math.hypot(float(gx) - pose.x, float(gy) - pose.y)
        if d_goal <= follower.cfg.goal_tolerance:
            motor.stop()
            motor.mark_reached()
            return True

        # 5) Replan each tick (lets you react to obstacle updates)
        path = planner.plan_to_target(TargetPoint(float(gx), float(gy)), shared_map, target_frame="grid")
        if path is None or len(path.waypoints) < 2:
            motor.stop()
            time.sleep(follower.cfg.dt)
            continue

        # 6) Pure pursuit step
        # Optional: reduce lookahead near goal so you "dock" better
        Ld = float(follower.cfg.lookahead)
        if d_goal < 12.0:
            Ld = max(6.0, min(Ld, 0.8 * d_goal))

        tx, ty = follower._lookahead_point(path, pose, Ld)
        delta = follower._pure_pursuit_delta(pose, target_x=tx, target_y=ty)
        steer_deg = math.degrees(delta)

        # 7) Slow down on large steering to reduce overshoot
        steer_abs = abs(steer_deg)
        maxs = float(motor.cfg.max_steer_deg)
        base = int(motor.cfg.speed)
        min_spd = max(15, int(0.55 * base))
        spd = int(base - (base - min_spd) * min(1.0, steer_abs / max(1e-6, maxs)))

        motor.set_steering(steer_deg)
        motor.forward_for(follower.cfg.dt, speed=spd)

        # lightweight status print
        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(f"pose(grid)=({pose.x:.1f},{pose.y:.1f},{pose.theta:.2f}) goal=({gx},{gy}) d={d_goal:.1f} steer={steer_deg:.1f} spd={spd}")

    motor.stop()
    return False


def main() -> None:
    # ---------------- Shared map ----------------
    shared_map = SharedMap()

    # Map SLAM world (0,0) to grid (2,2) at startup:
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-2.0, -2.0))

    # ---------------- Camera (Vilib) ----------------
    cam = PiCarXCamera(CameraConfig(
        vflip=False,
        hflip=False,
        display_local=False,
        display_web=False,
        frame_size=(640, 480),
        output_color_order="rgb",   # Vilib is RGB internally :contentReference[oaicite:5]{index=5}
        frame_rate=30,
        copy_on_read=True,
    ))
    cam.start()

    # ---------------- SLAM ----------------
    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)

    slam_cfg = VslamConfig(
        translation_step=0.2,              # will be overridden per tick
        input_color_order=cam.color_order, # "rgb" from Vilib :contentReference[oaicite:6]{index=6}
        debug_draw_keypoints=False,
        debug_draw_matches=False,
        # (keep other defaults)
    )
    slam = MonocularVSLAM(intr, shared_map, car_id=0, cfg=slam_cfg)

    # Warm up SLAM (let exposure settle + fill feature cache)
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

    # ---------------- Planner ----------------
    planning_cfg = PlanningConfig(
        include_slam_points=False,     # keep False unless you add filtering
        inflation_radius_cells=2,      # good start for 8-connected real robot
        simplify_path=True,
        nudge_start_goal=True,
    )
    planner = MovementPlanner(planning_cfg=planning_cfg, world_size=(50, 50))

    # ---------------- Motor + Follower ----------------
    motor = MotorController(MotorConfig(
        speed=30,                 # tuned: start slow
        step_seconds=0.35,         # tune: ~time to move 1 grid cell at this speed
        steer_sign=1.0,            # flip to -1.0 if mirrored
        steer_offset_deg=0.0,      # tune straightness
        steer_gain=1.0,
        max_steer_deg=35.0,
        steering_slew_deg_per_s=140.0,
        brake_between_steps=True,
        settle_seconds=0.02,
    ))

    follower = PathFollower(motor, FollowerConfig(
        dt=0.05,
        lookahead=12.0,           # tuned for stability on 8-connected
        wheelbase=2.5,            # less aggressive turning than larger values
        goal_tolerance=3.0,
        max_run_seconds=9999.0,
        pose_frame="grid",
        steer_sign=1.0,           # keep neutral; MotorController owns sign
        max_steer_deg=motor.cfg.max_steer_deg,
    ))

    try:
        # ---------------- Task 1: (2,2) -> (45,45) ----------------
        print("Driving to (45,45)...")
        ok = drive_to_grid_goal(
            (45, 45),
            shared_map=shared_map,
            planner=planner,
            follower=follower,
            motor=motor,
            slam=slam,
            camera=cam,
            timeout_s=180.0,
            debug_show_keypoints=True,
        )
        print("Reached (45,45):", ok)

        # ---------------- Task 2: Square ----------------
        # (Use inner square to avoid edges)
        square = [(45, 45), (5, 45), (5, 5), (45, 5), (45, 45)]
        print("Driving square:", square)

        for goal in square[1:]:
            ok = drive_to_grid_goal(
                goal,
                shared_map=shared_map,
                planner=planner,
                follower=follower,
                motor=motor,
                slam=slam,
                camera=cam,
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