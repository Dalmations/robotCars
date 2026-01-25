# test_drive_path.py
from __future__ import annotations

from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner
from model import TargetPoint

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig


def main() -> None:
    shared_map = SharedMap()  # empty grid unless you merge obstacles in
    planner = MovementPlanner(planning_cfg=None, world_size=(50, 50))

    motor = MotorController(MotorConfig(speed=25, step_seconds=0.18))
    follower = PathFollower(motor, FollowerConfig(heading_to_steer_gain=22.0))

    # Same test target you used before (grid coords)
    target = TargetPoint(x=45.0, y=45.0)

    try:
        path = planner.plan_to_target(target=target, shared_map=shared_map)
        print(f"Planned path waypoints: {len(path.waypoints)}")
        follower.follow(path)
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
