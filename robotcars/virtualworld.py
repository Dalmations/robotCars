# virtual_world.py
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, List, Optional, Tuple

import numpy as np

Grid = np.ndarray
Point = Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class VirtualWorld:
    """
    Holds an occupancy grid (0=free, 1=obstacle) and an optional height map.
    By default, the world is EMPTY (no obstacles) unless you add them explicitly.
    """
    grid: Grid
    height: Optional[Grid] = None

    @property
    def size(self) -> Tuple[int, int]:
        w, h = self.grid.shape
        return w, h

    def in_bounds(self, p: Point) -> bool:
        x, y = p
        w, h = self.size
        return 0 <= x < w and 0 <= y < h

    def is_free(self, p: Point) -> bool:
        x, y = p
        return self.grid[x, y] == 0

    def add_rect_obstacle(self, x0: int, x1: int, y0: int, y1: int) -> None:
        """In-place add an axis-aligned rectangular obstacle."""
        self.grid[x0:x1, y0:y1] = 1
        if self.height is not None:
            self.height[self.grid == 1] += 3.0



def heuristic(a: Point, b: Point) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def iter_neighbors_8() -> Iterable[Tuple[int, int, float]]:
    """
    8-connected grid: dx, dy, step_cost.
    """
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    for dx, dy in moves:
        yield dx, dy, float(np.hypot(dx, dy))


def astar(grid: Grid, start: Point, goal: Point) -> Optional[List[Point]]:
    """
    grid: 2D numpy array, 0 = free, 1 = obstacle
    start, goal: (x, y)
    Returns: list of (x, y) points, or None if no path.
    """
    w, h = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sx, sy] == 1 or grid[gx, gy] == 1:
        return None

    open_set = []
    heappush(open_set, (heuristic(start, goal), 0.0, start, None))

    came_from: dict[Point, Optional[Point]] = {}
    g_score: dict[Point, float] = {start: 0.0}
    visited: set[Point] = set()

    while open_set:
        _, cost, current, parent = heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent

        if current == goal:
            path: List[Point] = []
            node: Optional[Point] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, step_cost in iter_neighbors_8():
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if grid[nx, ny] == 1:
                continue

            new_cost = cost + step_cost
            nxt = (nx, ny)
            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                f = new_cost + heuristic(nxt, goal)
                heappush(open_set, (f, new_cost, nxt, current))

    return None
