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


# from config import Config
# from types import TargetPoint, Observations
# from network_bus import NetworkBus
# from shared_map import SharedMap

# from camera_input import CameraInput
# from obstacle_detection import ObstacleDetector
# from obstacle_classifier import ObstacleClassifier
# from movement import MovementPlanner
# from obstacle_algorithm import ObstacleHandler
# from trajectory import TrajectoryEstimator
# from motor_controller import MotorController

# class CarAgent:
#     def __init__(
#         self,
#         car_id: int,
#         camera: CameraInput,
#         detector: ObstacleDetector,
#         classifier: ObstacleClassifier,
#         planner: MovementPlanner,
#         obstacle_handler: ObstacleHandler,
#         trajectory: TrajectoryEstimator,
#         motor: MotorController,
#         network: NetworkBus,
#     ):
#         self.car_id = car_id
#         self.camera = camera
#         self.detector = detector
#         self.classifier = classifier
#         self.planner = planner
#         self.obstacle_handler = obstacle_handler
#         self.trajectory = trajectory
#         self.motor = motor
#         self.network = network

#         self._target: TargetPoint | None = None

#     @classmethod
#     def from_config(cls, car_id: int, cfg: Config, network: NetworkBus) -> "CarAgent":
#         return cls(
#             car_id=car_id,
#             camera=CameraInput(cfg.camera),
#             detector=ObstacleDetector(cfg.vision),
#             classifier=ObstacleClassifier(cfg.vision),
#             planner=MovementPlanner(cfg.planning),
#             obstacle_handler=ObstacleHandler(cfg.planning),
#             trajectory=TrajectoryEstimator(cfg.motion),
#             motor=MotorController(cfg.motor),
#             network=network,
#         )

#     def set_target(self, target: TargetPoint) -> None:
#         self._target = target

#     def perceive(self) -> Observations:
#         frame = self.camera.capture()
#         raw = self.detector.detect(frame)
#         return self.classifier.classify(raw)

#     def step(self, shared_map: SharedMap) -> None:
#         if self._target is None:
#             return

#         path = self.planner.plan_to_target(target=self._target, shared_map=shared_map)

#         if self.planner.is_obstructed(path, shared_map):
#             path, self._target = self.obstacle_handler.resolve(
#                 target=self._target,
#                 proposed_path=path,
#                 shared_map=shared_map
#             )

#         traj = self.trajectory.estimate(path)
#         self.motor.execute(traj)

#     def at_target(self) -> bool:
#         return self.motor.at_target()
