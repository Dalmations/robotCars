# model.py
from typing import Dict, Any, List, Optional

class Pose:
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"



class TargetPoint:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"TargetPoint(x={self.x:.2f}, y={self.y:.2f})"


class Obstacle:
    def __init__(
        self,
        obstacle_id: str,
        x: float,
        y: float,
        radius: float,
    ):
        self.obstacle_id = obstacle_id
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)

    def __repr__(self) -> str:
        return (
            f"Obstacle(id={self.obstacle_id!r}, "
            f"x={self.x:.2f}, y={self.y:.2f}, "
            f"r={self.radius:.2f}"
        )



class Path:
    def __init__(self, waypoints: List[TargetPoint]):
        self.waypoints = waypoints

    def __repr__(self) -> str:
        return f"Path(n_waypoints={len(self.waypoints)})"