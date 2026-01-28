# shared_map.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from model import Observations, Pose, Scenario, TargetPoint, Path, Obstacle


GridPoint = Tuple[int, int]


@dataclass
class SharedMap:
    """
    Flowchart node: 'Update Shared Map (network)... Broadcast map to all cars'
    For now: stores obstacles but does NOT create any unless merged from observations.
    """
    obstacles: Dict[str, Obstacle] = field(default_factory=dict)

    _static_grid: Optional[np.ndarray] = None # For debug

    def reset_for_scenario(self, scenario: Scenario) -> None:
        self.obstacles.clear()

    def merge_observations(self, car_id: int, obs: Observations, pose: Pose) -> None:
        # TODO: de-dup logic; for now just overwrite by obstacle_id
        for ob in obs.obstacles:
            self.obstacles[ob.obstacle_id] = ob

    def snapshot(self) -> "SharedMap":
        # shallow copy is enough for now
        snap = SharedMap()
        snap.obstacles = dict(self.obstacles)
        return snap

    # --- Planning interfaces used by MovementPlanner/ObstacleHandler ---

    def get_car_grid_position(self) -> GridPoint:
        """
        Placeholder: replace with per-car localization pose -> grid conversion.
        For testing, keep it fixed or inject from car/localization later.
        """
        return (2, 2)
    
    # for static debug
    def set_static_occupancy_grid(self, grid: np.ndarray) -> None:
        self._static_grid = grid

    def to_occupancy_grid(self, size: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """
        Returns occupancy grid. IMPORTANT: if no obstacles merged, grid is empty.
        """
        # for debug
        if self._static_grid is not None:
            return self._static_grid
        w, h = size
        grid = np.zeros((w, h), dtype=np.int32)

        for ob in self.obstacles.values():
            x, y = int(round(ob.x)), int(round(ob.y))
            r = max(1, int(round(ob.radius)))
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            grid[x0:x1, y0:y1] = 1

        return grid

    # These are called by obstacle_algorithm.py in the earlier architecture
    def get_blocking_obstacle(self, proposed_path: Path) -> Optional[Obstacle]:
        grid = self.to_occupancy_grid()
        for wp in proposed_path.waypoints:
            x, y = int(round(wp.x)), int(round(wp.y))
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 1:
                # Return *some* obstacle that occupies this cell (rough match)
                # TODO: spatial index for accurate matching
                for ob in self.obstacles.values():
                    if int(round(ob.x)) == x and int(round(ob.y)) == y:
                        return ob
                return next(iter(self.obstacles.values()), None)
        return None

    def is_path_blocked(self, path: Path) -> bool:
        return self.get_blocking_obstacle(path) is not None

    def repath_around(self, proposed_path: Path) -> Path:
        # Placeholder: true repath should be done in MovementPlanner with updated constraints.
        return proposed_path

    def nearest_unobstructed_point(self, target: TargetPoint) -> TargetPoint:
        # Simple spiral search outward
        grid = self.to_occupancy_grid()
        tx, ty = int(round(target.x)), int(round(target.y))
        w, h = grid.shape
        for radius in range(1, 10):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = tx + dx, ty + dy
                    if 0 <= x < w and 0 <= y < h and grid[x, y] == 0:
                        return TargetPoint(x=float(x), y=float(y))
        return target

    def plan_path_to(self, target: TargetPoint) -> Path:
        # planner normally owns this.
        from virtual_world import astar
        start = self.get_car_grid_position()
        grid = self.to_occupancy_grid()
        goal = (int(round(target.x)), int(round(target.y)))
        raw = astar(grid, start, goal) or [start]
        return Path(waypoints=[TargetPoint(float(x), float(y)) for x, y in raw])
    
    
