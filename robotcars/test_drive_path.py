# test_drive_path.py
from __future__ import annotations

from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner
from model import TargetPoint, Pose

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.camera_input import PiCarXCamera, CameraConfig
from car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig

def main() -> None:
    shared_map = SharedMap()
    planner = MovementPlanner(planning_cfg=None, world_size=(50, 50))
    camera = PiCarXCamera(CameraConfig(display_local=False, display_web=True, obstacle_color="red"))
    camera.start()

    def on_tick(x: float, y: float, yaw: float) -> None:
        # follower gives pose in grid units; SharedMap expects Pose
        pose = Pose(x=x, y=y, theta=yaw)
        obs = camera.poll_observations(pose, car_id=0)
        if obs.obstacles:
            shared_map.merge_observations(car_id=0, obs=obs, pose=pose)

    motor = MotorController(MotorConfig(speed=80, step_seconds=0.18))
    follower = PurePursuitFollower(motor, FollowerConfig())
    target = TargetPoint(x=45.0, y=45.0) # static target for testing

    try:
        path = planner.plan_to_target(target=target, shared_map=shared_map)
        print(f"Planned path waypoints: {len(path.waypoints)}")
        follower.follow(path, on_tick=on_tick)
    finally:
        motor.stop()


if __name__ == "__main__":
    main()