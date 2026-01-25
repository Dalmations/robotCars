from config import Config
from coordination.coordinator import MultiCarCoordinator
from coordination.network_bus import NetworkBus
from coordination.shared_map import SharedMap
from speech_input.speech_processing import SpeechProcessor
from speech_input.bucketing import ScenarioBucketer
from speech_input.feedback import ChildFeedback
from coordination.localization import LocalizationEngine
from coordination.point_assignment import PointAssigner
from car_tools.car import CarAgent

def main() -> None:
    cfg = Config.load()

    network = NetworkBus()
    shared_map = SharedMap()

    speech = SpeechProcessor(cfg.speech)
    bucketer = ScenarioBucketer(cfg.therapy)
    feedback = ChildFeedback(cfg.feedback)

    localization = LocalizationEngine(cfg.localization)
    assigner = PointAssigner(cfg.assignment)

    cars = [CarAgent.from_config(car_id=i, cfg=cfg, network=network) for i in range(cfg.fleet_size)]

    coordinator = MultiCarCoordinator(
        cars=cars,
        network=network,
        shared_map=shared_map,
        speech=speech,
        bucketer=bucketer,
        feedback=feedback,
        localization=localization,
        assigner=assigner,
        cfg=cfg,
    )

    coordinator.run_forever()  # loops back to Start after Celebrate success

if __name__ == "__main__":
    main()
