# virtualworld.py
from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, List, Optional, Tuple

import numpy as np

Grid = np.ndarray
Point = Tuple[int, int]  # (x, y)
SQRT2 = math.sqrt(2.0)


def heuristic_octile(a: Point, b: Point) -> float:
    """
    Best-practice heuristic for 8-connected grids with costs:
      straight: 1, diagonal: sqrt(2)
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # D=1, D2=sqrt(2)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def iter_neighbors_8() -> Iterable[Tuple[int, int, float]]:
    """
    8-connected grid: dx, dy, step_cost.
    """
    moves = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
    ]
    for dx, dy, c in moves:
        yield dx, dy, c


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    *,
    allow_corner_cutting: bool = False,
    heuristic_weight: float = 1.0,
) -> Optional[List[Point]]:
    """
    grid: 2D numpy array, 0 = free, 1 = obstacle
    start, goal: (x, y)
    Returns: list of (x, y) points, or None if no path.
    blocks diagonal step if either adjacent cardinal cell is occupied
    heuristic_weight:
      1.0 => optimal A*
      >1.0 => weighted A* faster
    """
    w, h = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sx, sy] == 1 or grid[gx, gy] == 1:
        return None

    hw = float(max(1.0, heuristic_weight))

    open_heap: List[Tuple[float, float, Point]] = []
    heappush(open_heap, (hw * heuristic_octile(start, goal), 0.0, start))

    came_from: dict[Point, Optional[Point]] = {start: None}
    g_score: dict[Point, float] = {start: 0.0}
    closed: set[Point] = set()

    while open_heap:
        _f, g, cur = heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            # reconstruct
            path: List[Point] = []
            node: Optional[Point] = cur
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return path

        cx, cy = cur
        for dx, dy, step_cost in iter_neighbors_8():
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if grid[nx, ny] == 1:
                continue

            # prevent diagonal corner-cutting
            if not allow_corner_cutting and dx != 0 and dy != 0:
                if grid[cx + dx, cy] == 1 or grid[cx, cy + dy] == 1:
                    continue

            ng = g + step_cost
            nxt = (nx, ny)
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = cur
                nf = ng + hw * heuristic_octile(nxt, goal)
                heappush(open_heap, (nf, ng, nxt))

    return None