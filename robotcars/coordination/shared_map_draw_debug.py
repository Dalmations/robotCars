# coordination/shared_map_draw_debug.py
from __future__ import annotations

from typing import List, Tuple
import numpy as np

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def rasterize_segments_to_grid(
    segments: List[Segment],
    grid_size: tuple[int, int] = (50, 50),
    thickness_cells: int = 1,
) -> np.ndarray:
    """
    Convert drawn obstacle segments (world coords 0..W, 0..H) to occupancy grid.
    Uses a simple DDA rasterization along each segment.

    Returns grid with 0=free, 1=obstacle.
    """
    w, h = grid_size
    grid = np.zeros((w, h), dtype=np.int32)

    def clamp_cell(x: int, y: int) -> tuple[int, int]:
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    def stamp(x: int, y: int) -> None:
        # draw a small square for thickness
        for dx in range(-thickness_cells, thickness_cells + 1):
            for dy in range(-thickness_cells, thickness_cells + 1):
                xx, yy = clamp_cell(x + dx, y + dy)
                grid[xx, yy] = 1

    for (ax, ay), (bx, by) in segments:
        x0, y0 = int(round(ax)), int(round(ay))
        x1, y1 = int(round(bx)), int(round(by))

        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)

        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + t * dx))
            y = int(round(y0 + t * dy))
            x, y = clamp_cell(x, y)
            stamp(x, y)

    return grid
