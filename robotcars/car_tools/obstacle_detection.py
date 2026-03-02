# car_tools/obstacle_detection.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from model import Pose
from coordination.shared_map import SharedMap


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class VslamConfig:
    max_corners: int = 2000
    quality_level: float = 0.01
    min_distance: int = 10
    orb_nfeatures: int = 2000

    ratio_test: float = 0.75
    keep_best: int = 800

    ransac_prob: float = 0.999
    ransac_thresh: float = 1.0
    min_inliers: int = 30

    # Monocular scale is unknown; set a small constant step to get “working” XY.
    # Later, replace with scale from wheel odom / known object / stereo.
    translation_step: float = 1.0

    # Triangulation filters
    min_parallax_w: float = 0.005


class _Frame:
    __slots__ = ("img_gray", "kps_xy", "des", "pose_Tcw")

    def __init__(self, img_gray: np.ndarray, kps_xy: np.ndarray, des: np.ndarray, pose_Tcw: np.ndarray):
        self.img_gray = img_gray
        self.kps_xy = kps_xy          # Nx2 pixel coords
        self.des = des                # Nx32 ORB descriptors
        self.pose_Tcw = pose_Tcw      # 4x4 transform (camera wrt world) in our convention


def _add_ones(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=xy.dtype)], axis=1)


def _triangulate(Tcw1: np.ndarray, Tcw2: np.ndarray, K: np.ndarray, pts1_px: np.ndarray, pts2_px: np.ndarray) -> np.ndarray:
    # Use OpenCV triangulation (more compact than the repo’s manual SVD A-matrix loop)
    P1 = K @ Tcw1[:3, :]
    P2 = K @ Tcw2[:3, :]
    pts4 = cv2.triangulatePoints(P1, P2, pts1_px.T, pts2_px.T).T
    pts4 = pts4 / np.maximum(1e-9, pts4[:, 3:4])
    return pts4  # Nx4


class MonocularVSLAM:
    """
    Feature-based monocular VO/SLAM front-end inspired by LearnOpenCV’s Monocular SLAM pipeline:
      - Detect corners -> ORB descriptors
      - Match consecutive frames
      - Estimate relative motion (E, recoverPose)
      - Integrate pose
      - Triangulate sparse map points
    """

    def __init__(
        self,
        intr: CameraIntrinsics,
        shared_map: SharedMap,
        car_id: int = 0,
        cfg: Optional[VslamConfig] = None,
    ):
        self.cfg = cfg or VslamConfig()
        self.K = intr.K()
        self.shared_map = shared_map
        self.car_id = car_id

        self.orb = cv2.ORB_create(nfeatures=self.cfg.orb_nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._last: Optional[_Frame] = None

        # We maintain Tcw (camera pose wrt world). Start identity.
        self._Tcw = np.eye(4, dtype=np.float64)

        # Publish an initial pose
        self._publish_pose()

    def tick(self, frame_bgr_or_rgb: np.ndarray) -> Optional[Pose]:
        gray = self._to_gray(frame_bgr_or_rgb)
        pts_xy, des = self._extract(gray)
        if pts_xy is None or des is None or len(pts_xy) < 50:
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        cur = _Frame(gray, pts_xy, des, self._Tcw.copy())

        if self._last is None:
            self._last = cur
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        ok, idx_cur, idx_last, R, t = self._estimate_motion(cur, self._last)
        if not ok:
            self._last = cur
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Scale translation (monocular ambiguity)
        t = t.reshape(3)
        t_norm = float(np.linalg.norm(t))
        if t_norm > 1e-9:
            t = (t / t_norm) * float(self.cfg.translation_step)

        # Build relative transform: T_cur_last
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        # Repo does: f1.pose = Rt @ f2.pose (their convention) :contentReference[oaicite:5]{index=5}
        # We maintain Tcw; integrate as: Tcw_new = T_cur_last @ Tcw_last
        self._Tcw = T @ self._Tcw

        # Triangulate sparse points for mapping/debug
        pts_cur = cur.kps_xy[idx_cur].astype(np.float64)
        pts_last = self._last.kps_xy[idx_last].astype(np.float64)
        pts4 = _triangulate(self._Tcw, self._last.pose_Tcw, self.K, pts_cur, pts_last)

        good = (np.abs(pts4[:, 3]) > self.cfg.min_parallax_w) & (pts4[:, 2] > 0)
        pts3 = pts4[good][:, :3]
        if pts3.shape[0] > 0:
            self.shared_map.add_map_points(pts3)

        self._last = cur
        self._publish_pose()
        return self.shared_map.poses.get(self.car_id)

    # ---------- Internals ----------

    def _publish_pose(self) -> None:
        # Convert Tcw into a 2D pose (x,y,theta). We’ll treat world axes:
        # x = Tcw[0,3], y = Tcw[1,3], theta from yaw of rotation.
        x = float(self._Tcw[0, 3])
        y = float(self._Tcw[1, 3])
        theta = self._yaw_from_R(self._Tcw[:3, :3])
        self.shared_map.set_pose(self.car_id, Pose(x=x, y=y, theta=theta))

    @staticmethod
    def _yaw_from_R(R: np.ndarray) -> float:
        # yaw from rotation matrix (Z-up-ish assumption). Works “okay” for small pitch/roll.
        return float(math.atan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        # Try BGR->GRAY first; if image is RGB, result is still usable for ORB
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _extract(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
        )
        if pts is None:
            return None, None

        kps = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 20) for p in pts]
        kps, des = self.orb.compute(gray, kps)
        if des is None or len(kps) == 0:
            return None, None

        xy = np.array([kp.pt for kp in kps], dtype=np.float32)
        return xy, des

    def _estimate_motion(
        self,
        cur: _Frame,
        last: _Frame,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        matches = self.bf.knnMatch(cur.des, last.des, k=2)

        good = []
        idx_cur = []
        idx_last = []
        for m, n in matches:
            if m.distance < self.cfg.ratio_test * n.distance:
                idx_cur.append(m.queryIdx)
                idx_last.append(m.trainIdx)
                good.append(m)

        if len(good) < 8:
            return False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1))

        # Keep best-N
        if len(good) > self.cfg.keep_best:
            order = np.argsort([m.distance for m in good])[: self.cfg.keep_best]
            idx_cur = [idx_cur[i] for i in order]
            idx_last = [idx_last[i] for i in order]

        pts1 = cur.kps_xy[np.array(idx_cur)].astype(np.float64)
        pts2 = last.kps_xy[np.array(idx_last)].astype(np.float64)

        E, inliers = cv2.findEssentialMat(
            pts1,
            pts2,
            self.K,
            method=cv2.RANSAC,
            prob=self.cfg.ransac_prob,
            threshold=self.cfg.ransac_thresh,
        )
        if E is None or inliers is None:
            return False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1))

        inliers = inliers.reshape(-1).astype(bool)
        if int(inliers.sum()) < self.cfg.min_inliers:
            return False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1))

        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)

        idx_cur_in = np.array(idx_cur, dtype=np.int64)[inliers]
        idx_last_in = np.array(idx_last, dtype=np.int64)[inliers]

        return True, idx_cur_in, idx_last_in, R, t