from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

from model import Observations, Pose, Scenario, TargetPoint, Path, Obstacle


GridPoint = Tuple[int, int]
FrameName = Literal["world", "grid"]


@dataclass
class GridConfig:
    """
    Mapping between continuous SLAM/world coordinates and discrete planning grid cells.
    """
    size: Tuple[int, int] = (50, 50)
    resolution: float = 1.0
    origin_world: Tuple[float, float] = (0.0, 0.0)

    def center_cell(self) -> Tuple[float, float]:
        return (self.size[0] / 2.0, self.size[1] / 2.0)

    def set_center_world(self, center_world: Tuple[float, float]) -> None:
        cx_cell, cy_cell = self.center_cell()
        ox = float(center_world[0]) - cx_cell * float(self.resolution)
        oy = float(center_world[1]) - cy_cell * float(self.resolution)
        self.origin_world = (ox, oy)


@dataclass
class SharedMap:
    """
    Shared state for planning and localization.

    - `poses` are stored in world coordinates.
    - `obstacles` are stored in world coordinates.
    - `map_points` are sparse SLAM points stored as world-space tuples.
    """
    obstacles: Dict[str, Obstacle] = field(default_factory=dict)
    poses: Dict[int, Pose] = field(default_factory=dict)
    map_points: List[Tuple[float, float, float]] = field(default_factory=list)

    _static_grid: Optional[np.ndarray] = None
    grid: GridConfig = field(default_factory=GridConfig)

    # ---------------- Pose + SLAM points ----------------

    def set_pose(self, car_id: int, pose: Pose) -> None:
        self.poses[car_id] = pose

    def get_pose(self, car_id: int = 0, frame: FrameName = "world") -> Optional[Pose]:
        pose = self.poses.get(car_id)
        if pose is None:
            return None
        if frame == "world":
            return pose
        gx, gy = self.world_to_grid_f(pose.x, pose.y)
        return Pose(x=float(gx), y=float(gy), theta=float(pose.theta))

    def add_map_points(self, pts3: np.ndarray, *, max_points: int = 5000, stride: int = 1) -> None:
        if pts3 is None or not isinstance(pts3, np.ndarray) or pts3.size == 0:
            return

        pts3 = np.asarray(pts3, dtype=np.float32).reshape(-1, 3)
        if stride > 1:
            pts3 = pts3[::stride]

        self.map_points.extend((float(x), float(y), float(z)) for x, y, z in pts3)

        if max_points > 0 and len(self.map_points) > max_points:
            self.map_points = self.map_points[-max_points:]

    def merge_slam_update(
        self,
        car_id: int,
        pose: Pose,
        new_map_points: Optional[object] = None,
        max_points: int = 5000,
    ) -> None:
        self.set_pose(car_id, pose)

        if new_map_points is None:
            return

        if isinstance(new_map_points, np.ndarray):
            self.add_map_points(new_map_points, max_points=max_points)
            return

        try:
            self._append_map_points_from_iterable(new_map_points, max_points=max_points)
        except Exception:
            return

    def _append_map_points_from_iterable(self, pts: object, *, max_points: int) -> None:
        for point in pts:  # type: ignore[assignment]
            x, y, z = point
            self.map_points.append((float(x), float(y), float(z)))

        if max_points > 0 and len(self.map_points) > max_points:
            self.map_points = self.map_points[-max_points:]

    def reset_for_scenario(self, _scenario: Scenario) -> None:
        self.obstacles.clear()
        self.poses.clear()
        self.map_points.clear()

    def merge_observations(self, car_id: int, obs: Observations, pose: Pose) -> None:
        self.set_pose(car_id, pose)
        for ob in obs.obstacles:
            self.obstacles[ob.obstacle_id] = ob

    def snapshot(self) -> "SharedMap":
        snap = SharedMap()
        snap.obstacles = dict(self.obstacles)
        snap.poses = dict(self.poses)
        snap.map_points = list(self.map_points)
        snap._static_grid = self._static_grid.copy() if self._static_grid is not None else None
        snap.grid = GridConfig(
            size=self.grid.size,
            resolution=self.grid.resolution,
            origin_world=self.grid.origin_world,
        )
        return snap

    # ---------------- World <-> Grid ----------------

    def configure_grid(
        self,
        *,
        size: Tuple[int, int] = (50, 50),
        resolution: float = 1.0,
        origin_world: Optional[Tuple[float, float]] = None,
        center_world: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.grid.size = size
        self.grid.resolution = float(resolution)

        if origin_world is not None:
            self.grid.origin_world = (float(origin_world[0]), float(origin_world[1]))
        if center_world is not None:
            self.grid.set_center_world((float(center_world[0]), float(center_world[1])))

    def world_to_grid_f(self, x: float, y: float) -> Tuple[float, float]:
        ox, oy = self.grid.origin_world
        r = float(self.grid.resolution)
        gx = (float(x) - ox) / r
        gy = (float(y) - oy) / r
        return gx, gy

    def world_to_grid(self, x: float, y: float, *, clamp: bool = True) -> GridPoint:
        gx_f, gy_f = self.world_to_grid_f(x, y)
        gx = int(round(gx_f))
        gy = int(round(gy_f))
        if clamp:
            w, h = self.grid.size
            gx = max(0, min(w - 1, gx))
            gy = max(0, min(h - 1, gy))
        return (gx, gy)

    def grid_to_world_f(self, gx: float, gy: float) -> Tuple[float, float]:
        ox, oy = self.grid.origin_world
        r = float(self.grid.resolution)
        x = ox + float(gx) * r
        y = oy + float(gy) * r
        return x, y

    def get_car_grid_position(self, car_id: int = 0) -> GridPoint:
        pose = self.poses.get(car_id)
        if pose is None:
            cx, cy = self.grid.center_cell()
            return (int(round(cx)), int(round(cy)))
        return self.world_to_grid(pose.x, pose.y)

    # ---------------- Occupancy ----------------

    def set_static_occupancy_grid(self, grid: np.ndarray) -> None:
        self._static_grid = np.array(grid, copy=True)

    def to_occupancy_grid(
        self,
        size: Optional[Tuple[int, int]] = None,
        *,
        include_slam_points: bool = False,
        slam_points_radius_cells: int = 0,
        slam_points_max: int = 1500,
    ) -> np.ndarray:
        if self._static_grid is not None:
            return np.array(self._static_grid, copy=True)

        w, h = size if size is not None else self.grid.size
        grid = np.zeros((w, h), dtype=np.int32)

        for obstacle in self.obstacles.values():
            gx, gy = self.world_to_grid(obstacle.x, obstacle.y)
            r_cells = int(round(float(obstacle.radius) / max(1e-6, float(self.grid.resolution))))
            r_cells = max(0, r_cells)

            x0, x1 = max(0, gx - r_cells), min(w, gx + r_cells + 1)
            y0, y1 = max(0, gy - r_cells), min(h, gy + r_cells + 1)
            grid[x0:x1, y0:y1] = 1

        if include_slam_points and self.map_points:
            points = self.map_points[-slam_points_max:] if slam_points_max > 0 else self.map_points
            radius = int(slam_points_radius_cells)

            for xw, yw, _zw in points:
                gx, gy = self.world_to_grid(xw, yw, clamp=False)
                if 0 <= gx < w and 0 <= gy < h:
                    if radius <= 0:
                        grid[gx, gy] = 1
                    else:
                        x0, x1 = max(0, gx - radius), min(w, gx + radius + 1)
                        y0, y1 = max(0, gy - radius), min(h, gy + radius + 1)
                        grid[x0:x1, y0:y1] = 1

        return grid

    def get_blocking_obstacle(self, proposed_path: Path) -> Optional[Obstacle]:
        for waypoint in proposed_path.waypoints:
            gx = int(round(waypoint.x))
            gy = int(round(waypoint.y))
            obstacle = self._explicit_obstacle_at_grid_cell(gx, gy)
            if obstacle is not None:
                return obstacle
        return None

    def _explicit_obstacle_at_grid_cell(self, gx: int, gy: int) -> Optional[Obstacle]:
        for obstacle in self.obstacles.values():
            ox, oy = self.world_to_grid(obstacle.x, obstacle.y)
            if ox == gx and oy == gy:
                return obstacle
        return None

    def is_path_blocked(self, path: Path) -> bool:
        grid = self.to_occupancy_grid()
        for waypoint in path.waypoints:
            gx = int(round(waypoint.x))
            gy = int(round(waypoint.y))
            if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1] and grid[gx, gy] == 1:
                return True
        return False

    def add_ultra_obstacle(
        self,
        *,
        pose_world: Pose,
        dist_cm: Optional[float],
        obstacle_id: str = "ultra_front",
        cm_per_grid: float = 50.0,
        ahead_cm_min: float = 5.0,
        ahead_cm_max: float = 120.0,
    ) -> bool:
        """
        Insert a front ultrasonic obstacle in world coordinates.
        """

        d_cm = float(np.clip(dist_cm, ahead_cm_min, ahead_cm_max))
        d_cells = d_cm / max(1e-6, float(cm_per_grid))

        ox = float(pose_world.x + d_cells * math.cos(pose_world.theta))
        oy = float(pose_world.y + d_cells * math.sin(pose_world.theta))

        footprint_radius_cm = math.hypot(10.0, 7.0)
        r_world = (footprint_radius_cm / max(1e-6, float(cm_per_grid))) * float(self.grid.resolution)

        self.obstacles[obstacle_id] = Obstacle(
            obstacle_id=obstacle_id,
            x=ox,
            y=oy,
            radius=float(r_world),
            is_moving=False,
        )
        return True

    def nearest_unobstructed_point(self, target: TargetPoint) -> TargetPoint:
        grid = self.to_occupancy_grid()
        tx, ty = int(round(target.x)), int(round(target.y))
        w, h = grid.shape
        for radius in range(1, 10):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = tx + dx, ty + dy
                    if 0 <= x < w and 0 <= y < h and grid[x, y] == 0:
                        return TargetPoint(x=float(x), y=float(y))
        return target