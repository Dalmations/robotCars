# car_tools/obstacle_detection.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from vilib import Vilib

from model import Pose, Obstacle, Observations
from coordination.shared_map import SharedMap


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class SlamConfig:
    min_inliers: int = 60
    keyframe_every: int = 10
    max_matches: int = 500
    ransac_thresh: float = 1.0
    max_new_points_per_kf: int = 800


@dataclass
class RangeConfig:
    object_width_m: float = 0.05
    min_box_w_px: int = 12


@dataclass
class WorldConfig:
    grid_scale_m_per_cell: float = 0.05
    obstacle_radius_grid: float = 2.0


class MonoVSLAM:
    def __init__(self, intr: CameraIntrinsics, cfg: Optional[SlamConfig] = None):
        self.intr = intr
        self.K = intr.K()
        self.cfg = cfg or SlamConfig()

        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._prev_kp = None
        self._prev_des = None

        self._frame_idx = 0
        self._last_kf_idx = 0
        self._kf_kp = None
        self._kf_des = None
        self._kf_Twc = np.eye(4, dtype=np.float64)

        self.Twc = np.eye(4, dtype=np.float64)
        self._new_points: List[Tuple[float, float, float]] = []

    def process(self, frame_bgr: np.ndarray) -> Tuple[bool, np.ndarray, List[Tuple[float, float, float]]]:
        self._new_points = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or kp is None or len(kp) < 200:
            return False, self.Twc, []

        if self._prev_des is None:
            self._prev_kp, self._prev_des = kp, des
            return False, self.Twc, []

        matches = self.matcher.match(self._prev_des, des)
        if len(matches) < 80:
            self._prev_kp, self._prev_des = kp, des
            return False, self.Twc, []

        matches = sorted(matches, key=lambda m: m.distance)[: self.cfg.max_matches]
        pts1 = np.float64([self._prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.cfg.ransac_thresh)
        if E is None or mask is None:
            self._prev_kp, self._prev_des = kp, des
            return False, self.Twc, []

        inliers = int(mask.sum())
        if inliers < self.cfg.min_inliers:
            self._prev_kp, self._prev_des = kp, des
            return False, self.Twc, []

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        Tdelta = np.eye(4, dtype=np.float64)
        Tdelta[:3, :3] = R
        Tdelta[:3, 3] = t.reshape(3)

        self.Twc = self.Twc @ Tdelta
        self._frame_idx += 1

        if (self._frame_idx - self._last_kf_idx) >= self.cfg.keyframe_every:
            self._triangulate_from_keyframe(kp, des, self.Twc)
            self._last_kf_idx = self._frame_idx
            self._kf_kp, self._kf_des, self._kf_Twc = kp, des, self.Twc.copy()

        self._prev_kp, self._prev_des = kp, des
        return True, self.Twc, self._new_points

    def _triangulate_from_keyframe(self, kp, des, Twc: np.ndarray) -> None:
        if self._kf_des is None:
            self._kf_kp, self._kf_des, self._kf_Twc = kp, des, Twc.copy()
            return

        matches = self.matcher.match(self._kf_des, des)
        if len(matches) < 80:
            return

        matches = sorted(matches, key=lambda m: m.distance)[:self.cfg.max_new_points_per_kf]
        pts1 = np.float64([self._kf_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kp[m.trainIdx].pt for m in matches])

        P1 = self.K @ self._Rt_from_Twc(self._kf_Twc)
        P2 = self.K @ self._Rt_from_Twc(Twc)

        Xh = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X = (Xh[:3, :] / (Xh[3:4, :] + 1e-9)).T

        for i in range(X.shape[0]):
            xyz = X[i]
            if np.isfinite(xyz).all():
                self._new_points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

    @staticmethod
    def _Rt_from_Twc(Twc: np.ndarray) -> np.ndarray:
        Rt = np.zeros((3, 4), dtype=np.float64)
        Rt[:3, :3] = Twc[:3, :3]
        Rt[:3, 3] = Twc[:3, 3]
        return Rt


class VslamObstacleDetector:
    def __init__(
        self,
        intr: CameraIntrinsics,
        shared_map: SharedMap,
        slam_cfg: Optional[SlamConfig] = None,
        range_cfg: Optional[RangeConfig] = None,
        world_cfg: Optional[WorldConfig] = None,
        car_id: int = 0,
    ):
        self.intr = intr
        self.shared_map = shared_map
        self.slam_cfg = slam_cfg or SlamConfig()
        self.slam = MonoVSLAM(intr, slam_cfg)
        self.range_cfg = range_cfg or RangeConfig()
        self.world_cfg = world_cfg or WorldConfig()
        self.car_id = car_id
        self._seq = 0

    def tick(self, frame_bgr: np.ndarray) -> Optional[Pose]:
        ok, Twc, new_pts = self.slam.process(frame_bgr)
        if not ok:
            return None

        pose = self._pose_from_Twc(Twc)
        self.shared_map.merge_slam_update(self.car_id, pose, new_pts)

        obs = self._rgb_obstacle_observation(Twc)
        if obs is not None:
            self.shared_map.merge_observations(self.car_id, obs, pose)

        return pose

    def _pose_from_Twc(self, Twc: np.ndarray) -> Pose:
        x_m = float(Twc[0, 3])
        y_m = float(Twc[1, 3])
        yaw = float(math.atan2(Twc[1, 0], Twc[0, 0]))

        x_grid = x_m / self.world_cfg.grid_scale_m_per_cell
        y_grid = y_m / self.world_cfg.grid_scale_m_per_cell
        return Pose(x=float(x_grid), y=float(y_grid), theta=float(yaw))

    def _rgb_obstacle_observation(self, Twc: np.ndarray) -> Optional[Observations]:
        color_n = int(Vilib.detect_obj_parameter.get("color_n", 0) or 0)
        if color_n <= 0:
            return None

        box_w = int(Vilib.detect_obj_parameter.get("color_w", 0) or 0)
        if box_w < self.range_cfg.min_box_w_px:
            return None

        u = float(Vilib.detect_obj_parameter.get("color_x", 0) or 0)
        v = float(Vilib.detect_obj_parameter.get("color_y", 0) or 0)

        z_m = (self.intr.fx * self.range_cfg.object_width_m) / float(box_w)

        x_cam = (u - self.intr.cx) * z_m / self.intr.fx
        y_cam = (v - self.intr.cy) * z_m / self.intr.fy
        Pc = np.array([x_cam, y_cam, z_m], dtype=np.float64).reshape(3, 1)

        Pw = (Twc[:3, :3] @ Pc) + Twc[:3, 3:4]
        ox_grid = float(Pw[0, 0] / self.world_cfg.grid_scale_m_per_cell)
        oy_grid = float(Pw[1, 0] / self.world_cfg.grid_scale_m_per_cell)

        self._seq += 1
        ob = Obstacle(
            obstacle_id=f"rgb_{self.car_id}_{self._seq}",
            x=ox_grid,
            y=oy_grid,
            radius=float(self.world_cfg.obstacle_radius_grid),
            is_moving=True,
        )
        return Observations(obstacles=[ob])
