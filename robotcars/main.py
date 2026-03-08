# main.py
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import cv2

from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.camera_input import PiCarXCamera, CameraConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Pose


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def drive_to_grid_goal(
    *,
    goal_gx: int,
    goal_gy: int,
    shared_map: SharedMap,
    planner: MovementPlanner,
    follower: PathFollower,
    motor: MotorController,
    slam: MonocularVSLAM,
    camera: PiCarXCamera,
    car_id: int = 0,
    timeout_s: float = 90.0,
    debug_show_keypoints: bool = True,
) -> bool:
    """
    Replan every 5s and execute pure-pursuit step.
    This is the simplest “closed loop” to combine SLAM pose + A*.
    """
    t0 = time.time()
    goal = TargetPoint(x=float(goal_gx), y=float(goal_gy))
    repathCount = 0
    current_path = None
    while (time.time() - t0) < timeout_s:
        # 1) SLAM update
        frame = camera.read()
        if frame is not None:
            slam.tick(frame)

            if debug_show_keypoints:
                dbg = slam.get_debug_keypoints_frame()
                if dbg is not None:
                    cv2.imshow("SLAM keypoints", dbg)
                    cv2.waitKey(1)

        # Pose in grid frame
        pose: Optional[Pose] = shared_map.get_pose(car_id, frame="grid")
        if pose is None:
            time.sleep(follower.cfg.dt)
            continue

        # Check goal
        if math.hypot(goal.x - pose.x, goal.y - pose.y) <= follower.cfg.goal_tolerance:
            motor.stop()
            motor.mark_reached()
            return True

        # Re plan path to goal every 5s
        if repathCount * 5 > (time.time() - t0):
            repathCount += 1
            current_path = planner.repath_to_target(
                current_path=current_path,
                target=goal,
                shared_map=shared_map,
                target_frame="grid",
            )

            if current_path is None or len(current_path.waypoints) < 2:
                # Nothing to follow safely
                motor.stop()
                time.sleep(follower.cfg.dt)
                continue

            # Pure pursuit steer
            tx, ty = follower._lookahead_point(current_path, pose, follower.cfg.lookahead)
            delta_rad = follower._pure_pursuit_delta(pose, target_x=tx, target_y=ty)
            steer_deg = math.degrees(delta_rad)
            # MotorController handles sign/gain/offset/clamp
            motor.set_steering(steer_deg)
            motor.forward_for(follower.cfg.dt)

    motor.stop()
    return False


# ---------------- Main behaviors ----------------

def main() -> None:
    shared_map = SharedMap()
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-2.0, -2.0))

    # grid = np.zeros((50, 50), dtype=np.int32)
    # grid[20:30, 25:27] = 1  # a vertical wall
    # shared_map.set_static_occupancy_grid(grid)

    # --- Camera ---
    cam = PiCarXCamera(CameraConfig(
        display_web=False,
        display_local=False,
        frame_size=(640, 480),
        camera_controls={"Saturation": 0.80},
    ))
    cam.start()

    # --- SLAM ---
    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)

    slam_cfg = VslamConfig(
        translation_step=1.0,
        debug_draw_keypoints=True,  # set True if you want to see keypoints
        debug_draw_matches=False,
    )
    slam = MonocularVSLAM(intr, shared_map, car_id=0, cfg=slam_cfg)

    # Warm up SLAM a few frames
    for _ in range(10):
        fr = cam.read()
        if fr is not None:
            slam.tick(fr)
        time.sleep(0.05)

    # --- Planner ---
    planning_cfg = PlanningConfig(
        include_slam_points=False,
        inflation_radius_cells=0,      # with 50cm cells, extra inflation is overly conservative
        simplify_path=True,
        nudge_start_goal=True,
    )
    planner = MovementPlanner(planning_cfg=planning_cfg, world_size=(50, 50))

    # --- Motor + Follower ---
    motor = MotorController(MotorConfig(
        speed=45,                 # start slow
        steer_sign=1.0,           # flip to -1.0 if mirrored
        steer_offset_deg=0.0,     # tune so set_steering(0) drives straight
        steer_gain=1.0,
        max_steer_deg=35.0,
        brake_between_steps=True,
    ))

    follower = PathFollower(motor, FollowerConfig(
        dt=0.50,
        lookahead=12.0,            # good start for 8-connected + simplified path
        wheelbase=2.5,            # effective parameter (tune)
        goal_tolerance=2.0,
        max_run_seconds=9999.0,
        pose_frame="grid",
        steer_sign=1.0,           # keep neutral; MotorController owns sign
        max_steer_deg=motor.cfg.max_steer_deg,
    ))

    try:
        # ------------------
        # Task 1: (2,2) -> (45,45)
        # ------------------
        print("Driving to (45,45)...")
        ok = drive_to_grid_goal(
            goal_gx=45, goal_gy=45,
            shared_map=shared_map,
            planner=planner,
            follower=follower,
            motor=motor,
            slam=slam,
            camera=cam,
            timeout_s=120.0,
            debug_show_keypoints=True,
        )
        print("Reached (45,45):", ok)

        # ------------------
        # Task 2: drive a square
        #   We'll use a safe square inside bounds:
        #   (45,45) -> (5,45) -> (5,5) -> (45,5) -> (45,45)
        # ------------------
        square = [(45, 45), (5, 45), (5, 5), (45, 5), (45, 45)]
        print("Driving square corners:", square)

        for (gx, gy) in square[1:]:
            ok = drive_to_grid_goal(
                goal_gx=gx, goal_gy=gy,
                shared_map=shared_map,
                planner=planner,
                follower=follower,
                motor=motor,
                slam=slam,
                camera=cam,
                timeout_s=120.0,
                debug_show_keypoints=True,
            )
            print(f"Reached ({gx},{gy}):", ok)
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
