from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Literal
import numpy as np

from model import Pose, TargetPoint, Path, Obstacle

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
            is_moving=False,
        )
        return True

    def render_grid_debug_view(
        self,
        *,
        car_id: int = 0,
        path: Optional[Path] = None,
        target: Optional[TargetPoint] = None,
        control_target: Optional[TargetPoint] = None,
        size: Optional[Tuple[int, int]] = None,
        include_slam_points: bool = False,
        slam_points_max: int = 1000,
        cell_px: int = 14,
        margin_px: int = 24,
        info_lines: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Render a 2D planning-grid view showing occupancy, pose, path, and target.
        """
        size = size if size is not None else self.grid.size
        w, h = int(size[0]), int(size[1])
        cell_px = max(8, int(cell_px))
        margin_px = max(8, int(margin_px))

        grid = self.to_occupancy_grid(size=size, include_slam_points=include_slam_points)

        free_color = np.array((242, 244, 246), dtype=np.uint8)
        obstacle_color = np.array((70, 85, 165), dtype=np.uint8)
        grid_line_color = (214, 220, 226)
        path_color = (66, 180, 255)
        target_color = (0, 210, 255)
        control_target_color = (0, 165, 255)
        pose_color = (46, 184, 92)
        slam_color = (148, 148, 148)
        text_color = (40, 40, 40)
        heading_line_color = (110, 135, 165)

        text_rows = list(info_lines or [])
        header_px = max(margin_px, 10 + 18 * max(1, len(text_rows) + 1))
        canvas_h = h * cell_px + header_px + margin_px
        canvas_w = w * cell_px + 2 * margin_px
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        for gx in range(w):
            for gy in range(h):
                x0 = margin_px + gx * cell_px
                y0 = header_px + (h - 1 - gy) * cell_px
                color = obstacle_color if grid[gx, gy] else free_color
                canvas[y0:y0 + cell_px, x0:x0 + cell_px] = color

        def grid_to_px(x_grid: float, y_grid: float) -> Tuple[int, int]:
            px = int(round(margin_px + (float(x_grid) + 0.5) * cell_px))
            py = int(round(header_px + (h - float(y_grid) - 0.5) * cell_px))
            return px, py

        def fill_cell(x_grid: int, y_grid: int, color: Tuple[int, int, int], pad: int = 2) -> None:
            if not (0 <= x_grid < w and 0 <= y_grid < h):
                return
            x0 = margin_px + x_grid * cell_px + pad
            y0 = header_px + (h - 1 - y_grid) * cell_px + pad
            x1 = margin_px + (x_grid + 1) * cell_px - pad
            y1 = header_px + (h - y_grid) * cell_px - pad
            if x1 <= x0 or y1 <= y0:
                return
            canvas[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)

        if include_slam_points and self.map_points:
            points = self.map_points[-slam_points_max:] if slam_points_max > 0 else self.map_points
            for xw, yw, _zw in points:
                gx, gy = self.world_to_grid(xw, yw, clamp=False)
                if 0 <= gx < w and 0 <= gy < h:
                    if cv2 is not None:
                        cv2.circle(canvas, grid_to_px(gx, gy), max(1, cell_px // 6), slam_color, -1, lineType=cv2.LINE_AA)
                    else:
                        fill_cell(gx, gy, slam_color, pad=max(3, cell_px // 3))

        if cv2 is not None:
            for gx in range(w + 1):
                x = margin_px + gx * cell_px
                cv2.line(canvas, (x, header_px), (x, header_px + h * cell_px), grid_line_color, 1, lineType=cv2.LINE_AA)
            for gy in range(h + 1):
                y = header_px + gy * cell_px
                cv2.line(canvas, (margin_px, y), (margin_px + w * cell_px, y), grid_line_color, 1, lineType=cv2.LINE_AA)

        if path is not None and path.waypoints:
            if cv2 is not None and len(path.waypoints) >= 2:
                pts = np.array([grid_to_px(wp.x, wp.y) for wp in path.waypoints], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    path_color,
                    thickness=max(2, cell_px // 4),
                    lineType=cv2.LINE_AA,
                )
                for wp in path.waypoints:
                    cv2.circle(canvas, grid_to_px(wp.x, wp.y), max(2, cell_px // 5), path_color, -1, lineType=cv2.LINE_AA)
            else:
                for wp in path.waypoints:
                    fill_cell(int(round(wp.x)), int(round(wp.y)), path_color, pad=max(2, cell_px // 5))

        if target is not None:
            if cv2 is not None:
                cv2.drawMarker(
                    canvas,
                    grid_to_px(target.x, target.y),
                    target_color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=max(10, cell_px),
                    thickness=max(1, cell_px // 5),
                    line_type=cv2.LINE_AA,
                )
            else:
                fill_cell(int(round(target.x)), int(round(target.y)), target_color, pad=max(2, cell_px // 5))

        pose_grid = self.get_pose(car_id, frame="grid")
        if pose_grid is not None:
            center = grid_to_px(pose_grid.x, pose_grid.y)
            if cv2 is not None and control_target is not None:
                cv2.line(
                    canvas,
                    center,
                    grid_to_px(control_target.x, control_target.y),
                    heading_line_color,
                    max(1, cell_px // 8),
                    lineType=cv2.LINE_AA,
                )
            if cv2 is not None:
                arrow_len = max(10, int(round(1.6 * cell_px)))
                tip = (
                    int(round(center[0] + arrow_len * math.cos(pose_grid.theta))),
                    int(round(center[1] - arrow_len * math.sin(pose_grid.theta))),
                )
                cv2.circle(canvas, center, max(3, cell_px // 3), pose_color, -1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(
                    canvas,
                    center,
                    tip,
                    pose_color,
                    thickness=max(2, cell_px // 5),
                    tipLength=0.35,
                    line_type=cv2.LINE_AA,
                )
            else:
                fill_cell(int(round(pose_grid.x)), int(round(pose_grid.y)), pose_color, pad=max(2, cell_px // 5))

        if control_target is not None:
            if cv2 is not None:
                cv2.drawMarker(
                    canvas,
                    grid_to_px(control_target.x, control_target.y),
                    control_target_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=max(10, cell_px),
                    thickness=max(1, cell_px // 5),
                    line_type=cv2.LINE_AA,
                )
            else:
                fill_cell(int(round(control_target.x)), int(round(control_target.y)), control_target_color, pad=max(2, cell_px // 5))

        if cv2 is not None:
            info = [f"car={car_id}"]
            if pose_grid is not None:
                info.append(f"pose=({pose_grid.x:.1f},{pose_grid.y:.1f},{pose_grid.theta:.2f})")
            if target is not None:
                info.append(f"goal=({target.x:.0f},{target.y:.0f})")
            if control_target is not None:
                info.append(f"ctrl=({control_target.x:.1f},{control_target.y:.1f})")
            if path is not None:
                info.append(f"wps={len(path.waypoints)}")
            text_rows = ["  ".join(info)] + text_rows
            for idx, line in enumerate(text_rows):
                cv2.putText(
                    canvas,
                    line,
                    (margin_px, 18 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    1,
                    lineType=cv2.LINE_AA,
                )

        return canvas
