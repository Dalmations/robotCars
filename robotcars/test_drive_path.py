# test_drive_path.py
from __future__ import annotations

from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner
from model import TargetPoint

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig
from car_tools.camera_input import PiCarXCamera, CameraConfig
from car_tools.obstacle_detection import VslamObstacleDetector, CameraIntrinsics


def main() -> None:
    shared_map = SharedMap()
    planner = MovementPlanner(planning_cfg=None, world_size=(50, 50))

    motor = MotorController(MotorConfig(speed=35, step_seconds=0.18))
    follower = PathFollower(
        motor,
        FollowerConfig(dt=0.10, lookahead=6.0, wheelbase=5.0, goal_tolerance=2.0),
    )

    cam = PiCarXCamera(CameraConfig(
        display_web=False,
        display_local=False,
        obstacle_color="red",
        frame_size=(640, 480),
    ))
    cam.start()

    intr = CameraIntrinsics(fx=628.0, fy=642.0, cx=320.0, cy=240.0)
    det = VslamObstacleDetector(intr=intr, shared_map=shared_map, car_id=0)

    target = TargetPoint(x=45.0, y=45.0)

    try:
        path = planner.plan_to_target(target=target, shared_map=shared_map)
        print(f"Planned path waypoints: {len(path.waypoints)}")

        follower.follow_with_slam(
            path=path,
            shared_map=shared_map,
            car_id=0,
            camera=cam,
            slam_detector=det,
        )
    finally:
        cam.stop()
        motor.stop()


if __name__ == "__main__":
    main()