# car.py (minimal "movement test" variant)
from __future__ import annotations

from config import Config
from coordination.shared_map import SharedMap
from model import TargetPoint

from movement import MovementPlanner
from motor_controller import MotorController, MotorConfig
from picarx_path_follower import PathFollower, FollowerConfig


class CarAgent:
    def __init__(self, car_id: int, planner: MovementPlanner, motor: MotorController, follower: PathFollower):
        self.car_id = car_id
        self.planner = planner
        self.motor = motor
        self.follower = follower
        self._target: TargetPoint | None = None

    @classmethod
    def from_config(cls, car_id: int, cfg: Config) -> "CarAgent":
        planner = MovementPlanner(planning_cfg=cfg.planning, world_size=(50, 50))
        motor = MotorController(MotorConfig())
        follower = PathFollower(motor, FollowerConfig())
        return cls(car_id=car_id, planner=planner, motor=motor, follower=follower)

    def set_target(self, target: TargetPoint) -> None:
        self._target = target

    def step(self, shared_map: SharedMap) -> None:
        if self._target is None:
            return
        path = self.planner.plan_to_target(target=self._target, shared_map=shared_map)
        self.follower.follow(path)

    def at_target(self) -> bool:
        return self.motor.at_target()