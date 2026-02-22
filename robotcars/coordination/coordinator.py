from typing import List
from model import Scenario, PoseDict, TargetPointDict

from car_tools.car import CarAgent
from network_bus import NetworkBus
from shared_map import SharedMap
from speech_input.speech_processing import SpeechProcessor, SpeechResult
from speech_input.bucketing import ScenarioBucketer
from speech_input.feedback import ChildFeedback
from localization import LocalizationEngine
from point_assignment import PointAssigner
from config import Config

class MultiCarCoordinator:
    def __init__(
        self,
        cars: List[CarAgent],
        network: NetworkBus,
        shared_map: SharedMap,
        speech: SpeechProcessor,
        bucketer: ScenarioBucketer,
        feedback: ChildFeedback,
        localization: LocalizationEngine,
        assigner: PointAssigner,
        cfg: Config,
    ):
        self.cars = cars
        self.network = network
        self.shared_map = shared_map
        self.speech = speech
        self.bucketer = bucketer
        self.feedback = feedback
        self.localization = localization
        self.assigner = assigner
        self.cfg = cfg

    def run_forever(self) -> None:
        while True:
            speech_result: SpeechResult = self._get_valid_speech()
            scenario: Scenario = self.bucketer.choose_scenario(speech_result.intent, speech_result.parameters)

            # main loop: repeat until all cars reach assigned points
            self._run_scenario_loop(scenario)

            self.feedback.celebrate_success()
            # flowchart loops back to Start/Power On state

    def _get_valid_speech(self) -> SpeechResult:
        while True:
            result = self.speech.listen_and_process()
            if result.accepted:
                return result
            self.feedback.try_again()

    def _run_scenario_loop(self, scenario: Scenario) -> None:
        # Clear/initialize shared world
        self.shared_map.reset_for_scenario(scenario)

        while True:
            poses: PoseDict = self.localization.estimate_poses(self.cars, self.network)
            targets: TargetPointDict = self.assigner.assign_points(scenario, poses)

            for car in self.cars:
                car.set_target(targets[car.car_id])

            # Per-car sensing -> merge into shared map -> broadcast
            for car in self.cars:
                obs = car.perceive()  # camera + detect + classify
                self.shared_map.merge_observations(car.car_id, obs, poses[car.car_id])
            self.network.broadcast_map(self.shared_map.snapshot())

            # Per-car planning/execution
            for car in self.cars:
                car.step(shared_map=self.shared_map)

            if all(car.at_target() for car in self.cars):
                return
