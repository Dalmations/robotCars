# movement.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List

import numpy as np

from model import Path, TargetPoint
from coordination.shared_map import SharedMap
from virtualworld import astar

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


GridPoint = Tuple[int, int]
FrameName = Literal["grid", "world"]


@dataclass
class PlanningConfig:
    # Path search strategy
    algorithm: Literal["astar", "weighted_astar"] = "weighted_astar"
    astar_heuristic_weight: float = 1.25

    # If True, project SLAM map points into grid as obstacles (usually noisy unless filtered)
    include_slam_points: bool = False
    slam_points_radius_cells: int = 0
    slam_points_max: int = 1500

    # Inflate obstacles by this many cells (robot footprint + pose noise margin)
    inflation_radius_cells: int = 1

    # If start/goal lands in obstacle, nudge to nearest free cell
    nudge_start_goal: bool = True

    # Reduce waypoint count (removes collinear points)
    simplify_path: bool = True

    # Cap for waypoint count
    max_waypoints: int = 800


@dataclass
class MovementPlanner:
    """
    Uses A* / weighted A* over occupancy grid from SharedMap.
    Output path waypoints are in grid coordinates by default.
    """
    planning_cfg: object = field(default_factory=PlanningConfig)
    world_size: Optional[tuple[int, int]] = None  # if None, uses shared_map.grid.size

    def plan_to_target(
        self,
        target: TargetPoint,
        shared_map: SharedMap,
        *,
        target_frame: FrameName = "grid",
    ) -> Path:
        cfg = self._cfg()

        # Determine grid size
        size = self.world_size if self.world_size is not None else getattr(shared_map, "grid").size
        w, h = int(size[0]), int(size[1])

        # Start always in grid
        start = shared_map.get_car_grid_position()

        # Goal interpretation
        if target_frame == "world":
            gx, gy = shared_map.world_to_grid(target.x, target.y)
            goal = (int(gx), int(gy))
        else:
            goal = (int(round(target.x)), int(round(target.y)))

        start = self._clamp_point(start, w, h)
        goal = self._clamp_point(goal, w, h)

        # Occupancy grid
        grid = shared_map.to_occupancy_grid(
            size=size,
            include_slam_points=cfg.include_slam_points,
            slam_points_radius_cells=cfg.slam_points_radius_cells,
            slam_points_max=cfg.slam_points_max,
        )

        # Inflate obstacles for car size margin
        if cfg.inflation_radius_cells > 0:
            grid = self._inflate_grid(grid, radius=int(cfg.inflation_radius_cells))

        # If start/goal are blocked, nudge them
        if cfg.nudge_start_goal:
            start = self._nudge_free(grid, start)
            goal = self._nudge_free(grid, goal)

        if start == goal:
            return Path(waypoints=[TargetPoint(x=float(start[0]), y=float(start[1]))])

        heuristic_weight = self._heuristic_weight(cfg)
        raw_path = astar(grid, start, goal, heuristic_weight=heuristic_weight)

        if raw_path is None:
            # Fallback: empty-world plan (only if the goal is in bounds)
            empty = np.zeros((w, h), dtype=np.int32)
            raw_path = astar(empty, start, goal, heuristic_weight=heuristic_weight) or [start]

        # Convert to waypoints
        waypoints = [TargetPoint(x=float(x), y=float(y)) for (x, y) in raw_path]

        # Simplify for smoother pursuit
        if cfg.simplify_path:
            waypoints = self._simplify_waypoints(waypoints)

        # Cap length
        if cfg.max_waypoints > 0 and len(waypoints) > cfg.max_waypoints:
            waypoints = waypoints[: cfg.max_waypoints]

        return Path(waypoints=waypoints)

    def repath_to_target(
        self,
        current_path: Optional[Path],
        target: TargetPoint,
        shared_map: SharedMap,
        *,
        target_frame: FrameName = "grid",
        force_replan: bool = False,
    ) -> Path:
        """
        Replan only when needed; otherwise keep following the current path.
        """
        if not force_replan and current_path is not None and len(current_path.waypoints) >= 1:
            if not self.is_obstructed(current_path, shared_map):
                return current_path
        return self.plan_to_target(target=target, shared_map=shared_map, target_frame=target_frame)

    def is_obstructed(self, path: Path, shared_map: SharedMap) -> bool:
        cfg = self._cfg()
        size = self.world_size if self.world_size is not None else getattr(shared_map, "grid").size
        grid = shared_map.to_occupancy_grid(
            size=size,
            include_slam_points=cfg.include_slam_points,
            slam_points_radius_cells=cfg.slam_points_radius_cells,
            slam_points_max=cfg.slam_points_max,
        )
        if cfg.inflation_radius_cells > 0:
            grid = self._inflate_grid(grid, radius=int(cfg.inflation_radius_cells))

        for wp in path.waypoints:
            x, y = int(round(wp.x)), int(round(wp.y))
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                if grid[x, y] == 1:
                    return True
        return False


    def _cfg(self) -> PlanningConfig:
        pc = self.planning_cfg
        if isinstance(pc, PlanningConfig):
            return pc

        out = PlanningConfig()
        algo = str(getattr(pc, "algorithm", out.algorithm)).strip().lower()
        if algo in {"astar", "weighted_astar"}:
            out.algorithm = algo  # type: ignore[assignment]
        out.astar_heuristic_weight = float(getattr(pc, "astar_heuristic_weight", out.astar_heuristic_weight))
        if out.astar_heuristic_weight < 1.0:
            out.astar_heuristic_weight = 1.0
        out.include_slam_points = bool(getattr(pc, "include_slam_points", out.include_slam_points))
        out.slam_points_radius_cells = int(getattr(pc, "slam_points_radius_cells", out.slam_points_radius_cells))
        out.slam_points_max = int(getattr(pc, "slam_points_max", out.slam_points_max))
        out.inflation_radius_cells = int(getattr(pc, "inflation_radius_cells", out.inflation_radius_cells))
        out.nudge_start_goal = bool(getattr(pc, "nudge_start_goal", out.nudge_start_goal))
        out.simplify_path = bool(getattr(pc, "simplify_path", out.simplify_path))
        out.max_waypoints = int(getattr(pc, "max_waypoints", out.max_waypoints))
        return out

    @staticmethod
    def _heuristic_weight(cfg: PlanningConfig) -> float:
        if cfg.algorithm == "astar":
            return 1.0
        return max(1.0, float(cfg.astar_heuristic_weight))

    @staticmethod
    def _clamp_point(p: GridPoint, w: int, h: int) -> GridPoint:
        x, y = int(p[0]), int(p[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return (x, y)

    @staticmethod
    def _nudge_free(grid: np.ndarray, p: GridPoint, max_radius: int = 10) -> GridPoint:
        """
        If p is occupied, spiral-search outwards for nearest free cell.
        """
        x0, y0 = p
        w, h = grid.shape
        if 0 <= x0 < w and 0 <= y0 < h and grid[x0, y0] == 0:
            return p

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = x0 + dx, y0 + dy
                    if 0 <= x < w and 0 <= y < h and grid[x, y] == 0:
                        return (x, y)
        return p

    @staticmethod
    def _inflate_grid(grid: np.ndarray, radius: int) -> np.ndarray:
        """
        Dilate occupied cells by 'radius' grid cells.
        """
        if radius <= 0:
            return grid

        g = (grid.astype(np.uint8) > 0).astype(np.uint8)

        if cv2 is not None:
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            g2 = cv2.dilate(g, kernel, iterations=1)
            return (g2 > 0).astype(np.int32)

        # Fallback dilation (slow but works)
        w, h = g.shape
        out = g.copy()
        for x in range(w):
            for y in range(h):
                if g[x, y]:
                    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
                    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
                    out[x0:x1, y0:y1] = 1
        return out.astype(np.int32)

    @staticmethod
    def _simplify_waypoints(wps: List[TargetPoint]) -> List[TargetPoint]:
        """
        Remove consecutive duplicates + collinear points in a grid path.
        """
        if len(wps) <= 2:
            return wps

        # Remove duplicates
        compact: List[TargetPoint] = [wps[0]]
        for p in wps[1:]:
            if int(round(p.x)) == int(round(compact[-1].x)) and int(round(p.y)) == int(round(compact[-1].y)):
                continue
            compact.append(p)

        if len(compact) <= 2:
            return compact

        def direction(a: TargetPoint, b: TargetPoint) -> Tuple[int, int]:
            dx = int(round(b.x - a.x))
            dy = int(round(b.y - a.y))
            # normalize to -1/0/1 for grid moves
            dx = 0 if dx == 0 else (1 if dx > 0 else -1)
            dy = 0 if dy == 0 else (1 if dy > 0 else -1)
            return dx, dy

        simplified: List[TargetPoint] = [compact[0]]
        prev_dir = direction(compact[0], compact[1])

        for i in range(1, len(compact) - 1):
            cur_dir = direction(compact[i], compact[i + 1])
            if cur_dir == prev_dir:
                # skip middle point on straight segment
                continue
            simplified.append(compact[i])
            prev_dir = cur_dir

        simplified.append(compact[-1])
        return simplified
