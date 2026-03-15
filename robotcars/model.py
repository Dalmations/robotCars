# model.py
from typing import Dict, Any, List, Optional


class SpeechResult:
    def __init__(
        self,
        accepted: bool,
        intent: str,
        parameters: Dict[str, Any],
        confidence: float,
    ):
        self.accepted = accepted
        self.intent = intent
        self.parameters = parameters
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"SpeechResult(accepted={self.accepted}, "
            f"intent={self.intent!r}, confidence={self.confidence:.2f})"
        )


class Pose:
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"


PoseDict = Dict[int, Pose]


class TargetPoint:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"TargetPoint(x={self.x:.2f}, y={self.y:.2f})"


TargetPointDict = Dict[int, TargetPoint]


class Obstacle:
    def __init__(
        self,
        obstacle_id: str,
        x: float,
        y: float,
        radius: float,
        is_moving: bool,
    ):
        self.obstacle_id = obstacle_id
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.is_moving = bool(is_moving)

    def __repr__(self) -> str:
        return (
            f"Obstacle(id={self.obstacle_id!r}, "
            f"x={self.x:.2f}, y={self.y:.2f}, "
            f"r={self.radius:.2f}, moving={self.is_moving})"
        )



class Path:
    def __init__(self, waypoints: List[TargetPoint]):
        self.waypoints = waypoints

    def __repr__(self) -> str:
        return f"Path(n_waypoints={len(self.waypoints)})"