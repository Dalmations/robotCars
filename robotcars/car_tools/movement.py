from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List

import numpy as np

from model import Path, Pose, TargetPoint
from coordination.shared_map import SharedMap
from virtualworld import astar

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


GridPoint = Tuple[int, int]
FrameName = Literal["grid", "world"]


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def estimate_step_cells_for_duration(
    duration_s: float,
    *,
    step_seconds: float,
    speed: int,
    speed_ref: int,
) -> float:
    base_step = float(duration_s) / float(max(1e-6, step_seconds))
    speed_ref = max(1.0, float(speed_ref))
    speed_cmd = float(max(0, min(100, int(speed))))
    return base_step * (speed_cmd / speed_ref)


def estimate_ackermann_yaw_delta(step_cells: float, steer_deg: float, wheelbase: float) -> float:
    wheelbase = max(1e-6, float(wheelbase))
    steer_rad = math.radians(float(steer_deg))
    return float(step_cells) * math.tan(steer_rad) / wheelbase


def integrate_dead_reckoning(
    *,
    shared_map: SharedMap,
    car_id: int,
    forward_step: float,
    yaw_delta: float,
) -> Pose:
    pose_world = shared_map.get_pose(car_id, frame="world")
    if pose_world is None:
        pose_world = Pose(0.0, 0.0, 0.0)

    step = float(forward_step)
    dtheta = float(yaw_delta)
    theta_mid = float(pose_world.theta) + 0.5 * dtheta

    next_pose = Pose(
        x=float(pose_world.x + step * math.cos(theta_mid)),
        y=float(pose_world.y + step * math.sin(theta_mid)),
        theta=float(wrap_angle(float(pose_world.theta) + dtheta)),
    )
    shared_map.set_pose(car_id, next_pose)
    return next_pose


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


@dataclass
class PlanningConfig:
    algorithm: Literal["astar", "weighted_astar"] = "weighted_astar"
    astar_heuristic_weight: float = 1.25

    include_slam_points: bool = False
    slam_points_radius_cells: int = 0
    slam_points_max: int = 1500

    inflation_radius_cells: int = 1
    nudge_start_goal: bool = True
    simplify_path: bool = True
    max_waypoints: int = 800


@dataclass
class MovementPlanner:
    """
    A* / weighted A* planner over the shared occupancy grid.
    """
    planning_cfg: PlanningConfig | object = field(default_factory=PlanningConfig)
    world_size: Optional[tuple[int, int]] = None

    def plan_to_target(
        self,
        target: TargetPoint,
        shared_map: SharedMap,
        *,
        target_frame: FrameName = "grid",
    ) -> Path:
        cfg = self._cfg()

        size = self.world_size if self.world_size is not None else getattr(shared_map, "grid").size
        w, h = int(size[0]), int(size[1])

        start = self._clamp_point(shared_map.get_car_grid_position(), w, h)

        if target_frame == "world":
            goal = shared_map.world_to_grid(target.x, target.y)
        else:
            goal = (int(round(target.x)), int(round(target.y)))
        goal = self._clamp_point(goal, w, h)

        grid = self._build_grid(shared_map, size=size, cfg=cfg)

        if cfg.nudge_start_goal:
            start = self._nudge_free(grid, start)
            goal = self._nudge_free(grid, goal)

        if start == goal:
            return Path(waypoints=[TargetPoint(x=float(start[0]), y=float(start[1]))])

        raw_path = astar(
            grid,
            start,
            goal,
            heuristic_weight=self._heuristic_weight(cfg),
        )

        # Real robot behavior: if there is no safe path, do not ignore the map.
        if raw_path is None:
            return Path(waypoints=[TargetPoint(x=float(start[0]), y=float(start[1]))])

        waypoints = [TargetPoint(x=float(x), y=float(y)) for (x, y) in raw_path]

        if cfg.simplify_path:
            waypoints = self._simplify_waypoints(waypoints)

        if cfg.max_waypoints > 0 and len(waypoints) > cfg.max_waypoints:
            waypoints = waypoints[: cfg.max_waypoints]

        return Path(waypoints=waypoints)

    def is_obstructed(self, path: Path, shared_map: SharedMap) -> bool:
        cfg = self._cfg()
        size = self.world_size if self.world_size is not None else getattr(shared_map, "grid").size
        grid = self._build_grid(shared_map, size=size, cfg=cfg)

        for wp in path.waypoints:
            x, y = int(round(wp.x)), int(round(wp.y))
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 1:
                return True
        return False

    def _build_grid(self, shared_map: SharedMap, *, size: tuple[int, int], cfg: PlanningConfig) -> np.ndarray:
        grid = shared_map.to_occupancy_grid(
            size=size,
            include_slam_points=cfg.include_slam_points,
            slam_points_radius_cells=cfg.slam_points_radius_cells,
            slam_points_max=cfg.slam_points_max,
        )
        if cfg.inflation_radius_cells > 0:
            grid = self._inflate_grid(grid, radius=int(cfg.inflation_radius_cells))
        return grid

    def _cfg(self) -> PlanningConfig:
        pc = self.planning_cfg
        if isinstance(pc, PlanningConfig):
            return pc

        out = PlanningConfig()
        algo = str(getattr(pc, "algorithm", out.algorithm)).strip().lower()
        if algo in {"astar", "weighted_astar"}:
            out.algorithm = algo  # type: ignore[assignment]

        out.astar_heuristic_weight = max(1.0, float(getattr(pc, "astar_heuristic_weight", out.astar_heuristic_weight)))
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
        return 1.0 if cfg.algorithm == "astar" else max(1.0, float(cfg.astar_heuristic_weight))

    @staticmethod
    def _clamp_point(p: GridPoint, w: int, h: int) -> GridPoint:
        x, y = int(p[0]), int(p[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return (x, y)

    @staticmethod
    def _nudge_free(grid: np.ndarray, p: GridPoint, max_radius: int = 10) -> GridPoint:
        """
        If `p` is occupied, search outward in expanding squares for the nearest free cell.
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
        if radius <= 0:
            return grid

        g = (grid.astype(np.uint8) > 0).astype(np.uint8)

        if cv2 is not None:
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            g2 = cv2.dilate(g, kernel, iterations=1)
            return (g2 > 0).astype(np.int32)

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
        if len(wps) <= 2:
            return wps

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
            dx = 0 if dx == 0 else (1 if dx > 0 else -1)
            dy = 0 if dy == 0 else (1 if dy > 0 else -1)
            return dx, dy

        simplified: List[TargetPoint] = [compact[0]]
        prev_dir = direction(compact[0], compact[1])

        for i in range(1, len(compact) - 1):
            cur_dir = direction(compact[i], compact[i + 1])
            if cur_dir == prev_dir:
                continue
            simplified.append(compact[i])
            prev_dir = cur_dir

        simplified.append(compact[-1])
        return simplified
