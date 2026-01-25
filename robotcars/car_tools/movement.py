# movement.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from model import Path, TargetPoint
from coordination.shared_map import SharedMap
from virtualworld import astar, VirtualWorld, create_virtual_world

GridPoint = Tuple[int, int]


@dataclass
class MovementPlanner:
    """
    Flowchart node: 'Movement Algorithm (per car) • Plan path to assigned point'
    Uses A* over an occupancy grid coming from SharedMap.
    """
    planning_cfg: object
    world_size: tuple[int, int] = (50, 50)

    def plan_to_target(self, target: TargetPoint, shared_map: SharedMap) -> Path:
        start = shared_map.get_car_grid_position()
        goal = (int(round(target.x)), int(round(target.y)))

        grid = shared_map.to_occupancy_grid(size=self.world_size)
        raw_path = astar(grid, start, goal)

        if raw_path is None:
            # fallback: empty-world plan (guarantees a path if goal in bounds)
            empty = np.zeros(self.world_size, dtype=np.int32)
            raw_path = astar(empty, start, goal) or [start]

        waypoints = [TargetPoint(x=float(x), y=float(y)) for (x, y) in raw_path]
        return Path(waypoints=waypoints)

    def is_obstructed(self, path: Path, shared_map: SharedMap) -> bool:
        # Conservatively: if any waypoint is on an obstacle cell, call it obstructed.
        grid = shared_map.to_occupancy_grid(size=self.world_size)
        for wp in path.waypoints:
            x, y = int(round(wp.x)), int(round(wp.y))
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                if grid[x, y] == 1:
                    return True
        return False
