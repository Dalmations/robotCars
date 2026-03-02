# main.py
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import cv2

from coordination.shared_map import SharedMap
from movement import MovementPlanner, PlanningConfig
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from model import TargetPoint, Pose


# ---------------- Camera adapters ----------------

class CameraBase:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCVCamera(CameraBase):
    def __init__(self, device: int = 0, size: Tuple[int, int] = (640, 480)):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self) -> None:
        self.cap.release()


class Picamera2Camera(CameraBase):
    def __init__(self, size: Tuple[int, int] = (640, 480)):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (int(size[0]), int(size[1]))}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.2)

    def read(self) -> Optional[np.ndarray]:
        return self.picam2.capture_array()  # RGB

    def close(self) -> None:
        try:
            self.picam2.stop()
        except Exception:
            pass


def init_camera(prefer_picamera2: bool = True, size: Tuple[int, int] = (640, 480)) -> Tuple[CameraBase, str]:
    """
    Returns (camera, color_order) where color_order is "rgb" or "bgr" for SLAM config.
    """
    if prefer_picamera2:
        try:
            cam = Picamera2Camera(size=size)
            return cam, "rgb"
        except Exception:
            pass

    cam = OpenCVCamera(device=0, size=size)
    return cam, "bgr"


# ---------------- Control loop: replan + pure pursuit step ----------------

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
    camera: CameraBase,
    car_id: int = 0,
    timeout_s: float = 90.0,
    debug_show_keypoints: bool = False,
) -> bool:
    """
    Replan every tick and execute one pure-pursuit step.
    This is the simplest “closed loop” to combine SLAM pose + A*.
    """
    t0 = time.time()
    goal = TargetPoint(x=float(goal_gx), y=float(goal_gy))

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

        # 2) Pose in grid frame (because our path is in grid coords)
        pose: Optional[Pose] = shared_map.get_pose(car_id, frame="grid")
        if pose is None:
            time.sleep(follower.cfg.dt)
            continue

        # 3) Check goal
        if math.hypot(goal.x - pose.x, goal.y - pose.y) <= follower.cfg.goal_tolerance:
            motor.stop()
            motor.mark_reached()
            return True

        # 4) Plan path to goal (grid coords)
        path = planner.plan_to_target(goal, shared_map, target_frame="grid")

        if path is None or len(path.waypoints) < 2:
            # Nothing to follow safely
            motor.stop()
            time.sleep(follower.cfg.dt)
            continue

        # 5) Pure pursuit control step
        tx, ty = follower._lookahead_point(path, pose, follower.cfg.lookahead)
        delta_rad = follower._pure_pursuit_delta(pose, target_x=tx, target_y=ty)
        steer_deg = math.degrees(delta_rad)

        # MotorController handles sign/gain/offset/clamp
        motor.set_steering(steer_deg)
        motor.forward_for(follower.cfg.dt)

    motor.stop()
    return False


# ---------------- Main behaviors ----------------

def main() -> None:
    # --- Shared map & grid config ---
    shared_map = SharedMap()

    # We want SLAM world (0,0) -> grid (2,2) at startup.
    # With resolution=1: origin_world = (-2, -2) achieves that.
    shared_map.configure_grid(size=(50, 50), resolution=1.0, origin_world=(-2.0, -2.0))

    # grid = np.zeros((50, 50), dtype=np.int32)
    # grid[20:30, 25:27] = 1  # a vertical wall
    # shared_map.set_static_occupancy_grid(grid)

    # --- Camera ---
    camera, color_order = init_camera(prefer_picamera2=True, size=(640, 480))

    # --- SLAM ---
    # NOTE: These intrinsics are placeholders. For best results, calibrate your camera.
    intr = CameraIntrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0)

    slam_cfg = VslamConfig(
        translation_step=1.0,
        input_color_order=color_order,
        debug_draw_keypoints=False,  # set True if you want to see keypoints
        debug_draw_matches=False,
    )
    slam = MonocularVSLAM(intr, shared_map, car_id=0, cfg=slam_cfg)

    # Warm up SLAM a few frames
    for _ in range(10):
        fr = camera.read()
        if fr is not None:
            slam.tick(fr)
        time.sleep(0.05)

    # --- Planner ---
    planning_cfg = PlanningConfig(
        include_slam_points=False,     # keep False unless you add filtering
        inflation_radius_cells=2,      # recommended start for real robot
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
        dt=0.10,
        lookahead=8.0,            # good start for 8-connected + simplified path
        wheelbase=4.0,            # effective parameter (tune)
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
            camera=camera,
            timeout_s=120.0,
            debug_show_keypoints=False,
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
                camera=camera,
                timeout_s=120.0,
                debug_show_keypoints=False,
            )
            print(f"Reached ({gx},{gy}):", ok)
            time.sleep(0.5)

    finally:
        motor.stop()
        camera.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass