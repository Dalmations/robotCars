from dataclasses import dataclass

@dataclass
class Config:
    fleet_size: int = 3
    speech: object = None
    therapy: object = None
    feedback: object = None
    localization: object = None
    assignment: object = None
    camera: object = None
    vision: object = None
    motion: object = None
    motor: object = None

    @staticmethod
    def load() -> "Config":
        # Replace with YAML/JSON/env loading later
        return Config()
