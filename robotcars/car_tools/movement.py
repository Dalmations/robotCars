# movement.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import math

from model import Path, TargetPoint
from coordination.shared_map import SharedMap
from virtualworld import astar

GridPoint = Tuple[int, int]

def clamp(self, value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


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
            case 'square':
                start_grid = shared_map.get_car_grid_position(0)
                route = self.build_square_route(start_grid, side_cells=6)
                return self.route_to_path(route)
            case 'hexagon':
                start_grid = shared_map.get_car_grid_position(0)
                route = self.build_hexagon_route(start_grid, side_cells=6)
                return self.route_to_path(route)
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
    
    def build_square_route(self, start_xy_grid: GridPoint, *, side_cells: int = 6) -> list[GridPoint]:
        side = max(3, int(side_cells))
        sx, sy = int(start_xy_grid[0]), int(start_xy_grid[1])
        return [
            (sx, sy),
            (sx + side, sy),
            (sx + side, sy + side),
            (sx, sy + side),
            (sx, sy),
        ]


    def build_hexagon_route(self, start_xy_grid: GridPoint, *, side_cells: int = 6) -> list[GridPoint]:
        side = max(3, int(side_cells))
        height = max(2, int(round(side * math.sqrt(3.0) * 0.5)))
        sx, sy = int(start_xy_grid[0]), int(start_xy_grid[1])
        return [
            (sx, sy),
            (sx + side, sy),
            (sx + int(side * 3 / 2), sy + int(height / 2)),
            (sx + side, sy + height),
            (sx, sy + height),
            (sx - int(side / 2), sy + int(height / 2)),
            (sx, sy),
        ]

    def route_to_path(self, route_xy_grid: Iterable[GridPoint]) -> Path:
        waypoints = [
            TargetPoint(x=float(int(point[0])), y=float(int(point[1])))
            for point in route_xy_grid
        ]
        return Path(waypoints=waypoints)