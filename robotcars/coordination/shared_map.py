# shared_map.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np

from model import Observations, Pose, TargetPoint, Path, Obstacle

import math
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


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


@dataclass
class SharedMap:
    """
    'Update Shared Map Broadcast map to all cars'
    Stores obstacles but does not create any unless merged from observations.
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

    def reset(self) -> None:
        self.obstacles.clear()
        self.poses.clear()
        self.map_points.clear()

    # ---------------- World <-> Grid ----------------
    def configure_grid(
        self,
        *,
        size: Tuple[int, int] = (50, 50),
        resolution: float = 1.0,
        origin_world: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.grid.size = size
        self.grid.resolution = float(resolution)
        self.grid.origin_world = (
            float(origin_world[0]),
            float(origin_world[1]),
        )

    def world_to_grid_f(self, x: float, y: float) -> Tuple[float, float]:
        r = float(self.grid.resolution)
        ox, oy = self.grid.origin_world
        gx = (float(x) - float(ox)) / r
        gy = (float(y) - float(oy)) / r
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
        r = float(self.grid.resolution)
        ox, oy = self.grid.origin_world
        x = float(ox) + float(gx) * r
        y = float(oy) + float(gy) * r
        return x, y

    def get_car_grid_position(self, car_id: int = 0) -> GridPoint:
        pose = self.poses.get(car_id)
        if pose is None:
            return (0, 0)
        return self.world_to_grid(pose.x, pose.y)

    # ---------------- Occupancy ----------------

    


    # --- Used by MovementPlanner/ObstacleHandler ---
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

    def add_ultra_obstacle(
        self,
        *,
        pose_world: Pose,
        dist_cm: Optional[float],
        obstacle_id: str = "ultra_front",
        cm_per_grid: float = 50.0,
    ) -> bool:
        """
        Insert a front ultrasonic obstacle in world coordinates.
        """
        if dist_cm is None or not np.isfinite(dist_cm):
            self.obstacles.pop(obstacle_id, None)
            return False

        d_cm = float(dist_cm)
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
        )
        return True
    
