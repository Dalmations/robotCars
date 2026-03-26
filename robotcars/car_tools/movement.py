from __future__ import annotations
import math
from typing import Iterable, Iterable, Tuple
import math

from model import Path, TargetPoint


GridPoint = Tuple[int, int]

def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))

def plan_formation(shared_map: SharedMap, shape : str) -> Path:
    match shape:
        case 'circle':
            route = generate_circle_points()
            return route_to_path(route)
        case 'square':
            start_grid = shared_map.get_car_grid_position(0)
            route = build_square_route(start_grid, side_cells=6)
            return route_to_path(route)
        case 'hexagon':
            start_grid = shared_map.get_car_grid_position(0)
            route = build_hexagon_route(start_grid, side_cells=6)
            return route_to_path(route)
        case _:
            return Path([])

def generate_circle_points(world_size, num_points=100):
    center_x = world_size[0] / 2
    center_y = world_size[1] / 2
    radius = world_size[0] / 2
    points = []
    
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        
        x = center_x + radius * math.cos(theta)
        y = center_y - radius * math.sin(theta)
        
        points.append((x, y))

    return points

def build_square_route(start_xy_grid: GridPoint, *, side_cells: int = 6) -> list[GridPoint]:
    side = max(3, int(side_cells))
    sx, sy = int(start_xy_grid[0]), int(start_xy_grid[1])
    return [
        (sx, sy),
        (sx + side, sy),
        (sx + side, sy + side),
        (sx, sy + side),
        (sx, sy),
    ]


def build_hexagon_route(start_xy_grid: GridPoint, *, side_cells: int = 6) -> list[GridPoint]:
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

def route_to_path(route_xy_grid: Iterable[GridPoint]) -> Path:
    waypoints = [
        TargetPoint(x=float(point[0]), y=float(point[1]))
        for point in route_xy_grid
    ]
    return Path(waypoints=waypoints)