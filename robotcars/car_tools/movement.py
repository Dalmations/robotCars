from __future__ import annotations

import math
from typing import Iterable, Tuple

from model import Path, TargetPoint


GridPoint = Tuple[int, int]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def route_to_path(route_xy_grid: Iterable[GridPoint]) -> Path:
    waypoints = [
        TargetPoint(x=float(int(point[0])), y=float(int(point[1])))
        for point in route_xy_grid
    ]
    return Path(waypoints=waypoints)


def build_equilateral_triangle_route(start_xy_grid: GridPoint, *, side_cells: int = 6) -> list[GridPoint]:
    side = max(3, int(side_cells))
    height = max(2, int(round(side * math.sqrt(3.0) * 0.5)))
    sx, sy = int(start_xy_grid[0]), int(start_xy_grid[1])
    return [
        (sx, sy),
        (sx + side, sy),
        (sx + side // 2, sy + height),
        (sx, sy),
    ]


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
