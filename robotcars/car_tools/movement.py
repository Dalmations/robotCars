# movement.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import math

from model import Path, TargetPoint
from coordination.shared_map import SharedMap
from virtualworld import astar

GridPoint = Tuple[int, int]


@dataclass
class MovementPlanner:
    """
    Flowchart node: 'Movement Algorithm (per car) • Plan path to assigned point'
    Uses A* over an occupancy grid coming from SharedMap.
    """
    planning_cfg: object = None
    world_size: tuple[int, int] = (25,25)

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

    def plan_formation(self, shape : str) -> Path:
        match shape:
            case 'circle':
                # with open('path_points.txt','w') as f:
                #     for pair in self.generate_circle_points():
                #         f.write(f'{pair}\n')
                return Path([TargetPoint(x, y) for (x, y) in self.generate_circle_points()])
            # TODO: Add square & hexagon path
            case _:
                return Path([])

    def generate_circle_points(self, num_points=100):
        center_x = self.world_size[0] / 2
        center_y = self.world_size[1] / 2
        radius = self.world_size[0] / 2
        points = []
        
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            
            x = center_x + radius * math.cos(theta)
            y = center_y - radius * math.sin(theta)
            
            points.append((x, y))

        return points
