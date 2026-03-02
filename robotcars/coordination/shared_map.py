# shared_map.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Iterable

import numpy as np

from model import Observations, Pose, Scenario, TargetPoint, Path, Obstacle


GridPoint = Tuple[int, int]
FrameName = Literal["world", "grid"]


@dataclass
class GridConfig:
    """
    Defines mapping between continuous world coordinates (SLAM output)
    and discrete grid coordinates (for A*).

    world_to_grid:
      gx = (x - origin_world_x) / resolution
      gy = (y - origin_world_y) / resolution

    - resolution: world units per grid cell
    - origin_world: world coordinate of grid cell (0,0)
    """
    size: Tuple[int, int] = (50, 50)
    resolution: float = 1.0
    origin_world: Tuple[float, float] = (0.0, 0.0)

    def center_cell(self) -> Tuple[float, float]:
        # For size=(50,50), this returns (25,25)
        return (self.size[0] / 2.0, self.size[1] / 2.0)

    def set_center_world(self, center_world: Tuple[float, float]) -> None:
        """
        Configure origin_world so that world center_world maps to the grid center cell.
        """
        cx_cell, cy_cell = self.center_cell()
        ox = float(center_world[0]) - cx_cell * float(self.resolution)
        oy = float(center_world[1]) - cy_cell * float(self.resolution)
        self.origin_world = (ox, oy)


@dataclass
class SharedMap:
    """
    Stores obstacles, poses, and SLAM map points.
    """
    obstacles: Dict[str, Obstacle] = field(default_factory=dict)
    poses: Dict[int, Pose] = field(default_factory=dict)

    # Store SLAM map points as a flat list of tuple
    map_points: List[Tuple[float, float, float]] = field(default_factory=list)
    _static_grid: Optional[np.ndarray] = None

    # World<->grid mapping
    grid: GridConfig = field(default_factory=GridConfig)

    # ---------------- Pose + SLAM points ----------------

    def set_pose(self, car_id: int, pose: Pose) -> None:
        """Pose is assumed to be in world frame from SLAM."""
        self.poses[car_id] = pose

    def get_pose(self, car_id: int = 0, frame: FrameName = "world") -> Optional[Pose]:
        p = self.poses.get(car_id)
        if p is None:
            return None
        if frame == "world":
            return p
        gx, gy = self.world_to_grid_f(p.x, p.y)
        return Pose(x=float(gx), y=float(gy), theta=float(p.theta))

    def add_map_points(self, pts3: np.ndarray, *, max_points: int = 5000, stride: int = 1) -> None:
        """
        Accepts Nx3 np.ndarray from SLAM and stores as List[Tuple[float,float,float]].
        """
        if pts3 is None:
            return
        if not isinstance(pts3, np.ndarray):
            return
        if pts3.size == 0:
            return

        pts3 = np.asarray(pts3, dtype=np.float32).reshape(-1, 3)
        if stride > 1:
            pts3 = pts3[::stride]

        self.map_points.extend((float(x), float(y), float(z)) for x, y, z in pts3)

        if max_points > 0 and len(self.map_points) > max_points:
            self.map_points = self.map_points[-max_points:]

    def reset_for_scenario(self, scenario: Scenario) -> None:
        self.obstacles.clear()
        self.poses.clear()
        self.map_points.clear()

    def merge_observations(self, car_id: int, obs: Observations, pose: Pose) -> None:
        # pose assumed to be world, unless your pipeline uses grid already.
        self.poses[car_id] = pose
        for ob in obs.obstacles:
            self.obstacles[ob.obstacle_id] = ob

    def merge_slam_update(
        self,
        car_id: int,
        pose: Pose,
        new_map_points: Optional[object] = None,
        max_points: int = 5000,
    ) -> None:
        """
        Backwards-compatible: accepts either:
          - list of (x,y,z) tuples
          - Nx3 numpy array
        """
        self.poses[car_id] = pose

        if new_map_points is None:
            return

        if isinstance(new_map_points, np.ndarray):
            self.add_map_points(new_map_points, max_points=max_points)
            return

        # assume iterable of tuples
        try:
            for p in new_map_points:  # type: ignore[assignment]
                x, y, z = p
                self.map_points.append((float(x), float(y), float(z)))
        except Exception:
            return

        if max_points > 0 and len(self.map_points) > max_points:
            self.map_points = self.map_points[-max_points:]

    def snapshot(self) -> "SharedMap":
        snap = SharedMap()
        snap.obstacles = dict(self.obstacles)
        snap.poses = dict(self.poses)
        snap.map_points = list(self.map_points)
        snap._static_grid = self._static_grid.copy() if self._static_grid is not None else None
        snap.grid = GridConfig(size=self.grid.size, resolution=self.grid.resolution, origin_world=self.grid.origin_world)
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
        """
        Configure mapping. SLAM starting (0,0):
            shared_map.configure_grid(size=(50,50), resolution=1.0, center_world=(0.0,0.0))

         SLAM world (0,0) -> grid center (25,25).
        """
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
        """
        Returns car position in grid coords, derived from stored world pose.
        """
        p = self.poses.get(car_id)
        if p is None:
            # center cell by default
            cx, cy = self.grid.center_cell()
            return (int(round(cx)), int(round(cy)))
        return self.world_to_grid(p.x, p.y)

    def set_static_occupancy_grid(self, grid: np.ndarray) -> None:
        self._static_grid = grid

    def to_occupancy_grid(
        self,
        size: Optional[Tuple[int, int]] = None,
        *,
        include_slam_points: bool = False,
        slam_points_radius_cells: int = 0,
        slam_points_max: int = 1500,
    ) -> np.ndarray:
        """
        Returns occupancy grid.
        """
        if self._static_grid is not None:
            return self._static_grid

        w, h = size if size is not None else self.grid.size
        grid = np.zeros((w, h), dtype=np.int32)

        # Obstacles stored in world frame
        for ob in self.obstacles.values():
            gx, gy = self.world_to_grid(ob.x, ob.y)
            r = max(1, int(round(ob.radius)))
            x0, x1 = max(0, gx - r), min(w, gx + r + 1)
            y0, y1 = max(0, gy - r), min(h, gy + r + 1)
            grid[x0:x1, y0:y1] = 1

        if include_slam_points and self.map_points:
            pts = self.map_points[-slam_points_max:] if slam_points_max > 0 else self.map_points
            r = int(slam_points_radius_cells)

            for xw, yw, _zw in pts:
                gx, gy = self.world_to_grid(xw, yw, clamp=False)
                if 0 <= gx < w and 0 <= gy < h:
                    if r <= 0:
                        grid[gx, gy] = 1
                    else:
                        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
                        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
                        grid[x0:x1, y0:y1] = 1

        return grid

    def get_blocking_obstacle(self, proposed_path: Path) -> Optional[Obstacle]:
        grid = self.to_occupancy_grid()
        for wp in proposed_path.waypoints:
            x, y = int(round(wp.x)), int(round(wp.y))
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 1:
                for ob in self.obstacles.values():
                    gx, gy = self.world_to_grid(ob.x, ob.y)
                    if gx == x and gy == y:
                        return ob
                return next(iter(self.obstacles.values()), None)
        return None

    def is_path_blocked(self, path: Path) -> bool:
        return self.get_blocking_obstacle(path) is not None

    def repath_around(self, proposed_path: Path) -> Path:
        return proposed_path

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

    def plan_path_to(self, target: TargetPoint) -> Path:
        """
        Target is interpreted in grid coords, A* runs on a grid.
        Plan to a world target, convert with:
            gx,gy = shared_map.world_to_grid(world_x, world_y)
            shared_map.plan_path_to(TargetPoint(float(gx), float(gy)))
        """
        from virtualworld import astar
        start = self.get_car_grid_position()
        grid = self.to_occupancy_grid()
        goal = (int(round(target.x)), int(round(target.y)))
        raw = astar(grid, start, goal) or [start]
        return Path(waypoints=[TargetPoint(float(x), float(y)) for x, y in raw])