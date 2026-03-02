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
    translation_step: float = 1.0

    # Triangulation filters
    # Typical values: 0.001 ~ 0.01 depending on resolution/FOV/motion.
    min_parallax_w: float = 0.003

    min_depth: float = 0.05
    max_depth: float = 200.0

    # Input frames:
    input_color_order: str = "rgb"  # "bgr" or "rgb"

    # Debug drawing
    debug_draw_keypoints: bool = False
    debug_draw_matches: bool = False
    debug_max_match_draw: int = 80


class _Frame:
    __slots__ = ("img_gray", "kps_xy", "des", "pose_Tcw")

    def __init__(self, img_gray: np.ndarray, kps_xy: np.ndarray, des: np.ndarray, pose_Tcw: np.ndarray):
        self.img_gray = img_gray
        self.kps_xy = kps_xy          # Nx2 pixel coords
        self.des = des                # Nx32 ORB descriptors
        self.pose_Tcw = pose_Tcw      # 4x4 world->camera (Tcw)


def _add_ones(xyz: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,4) homogeneous."""
    return np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)


def _triangulate_Xw(
    Tcw1: np.ndarray,
    Tcw2: np.ndarray,
    K: np.ndarray,
    pts1_px: np.ndarray,
    pts2_px: np.ndarray
) -> np.ndarray:
    """
    Triangulate world points Xw (N,3) given two world->camera poses (Tcw1, Tcw2) and matching pixels.

    Using projection matrices:
      P = K [R|t] where [R|t] maps world -> camera
    """
    P1 = K @ Tcw1[:3, :]
    P2 = K @ Tcw2[:3, :]

    pts4 = cv2.triangulatePoints(P1, P2, pts1_px.T, pts2_px.T).T  # (N,4)
    w = pts4[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    Xw = pts4[:, :3] / w
    return Xw


class MonocularVSLAM:
    """
    Feature-based monocular VO/SLAM front-end:
      - corners -> ORB descriptors
      - match consecutive frames
      - estimate relative motion (E, recoverPose)
      - integrate pose (maintaining Tcw = world->camera)
      - triangulate sparse map points (world coords)
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

        # Maintain Tcw (world->camera). Start at identity.
        self._Tcw = np.eye(4, dtype=np.float64)

        # Debug buffers (BGR images for easy imshow/imwrite)
        self._dbg_keypoints_bgr: Optional[np.ndarray] = None
        self._dbg_matches_bgr: Optional[np.ndarray] = None

        self._publish_pose()

    # Debug

    def get_debug_keypoints_frame(self) -> Optional[np.ndarray]:
        """Latest grayscale-with-keypoints frame (BGR image)."""
        return None if self._dbg_keypoints_bgr is None else self._dbg_keypoints_bgr

    def get_debug_matches_frame(self) -> Optional[np.ndarray]:
        """Latest inlier match visualization (BGR image)."""
        return None if self._dbg_matches_bgr is None else self._dbg_matches_bgr

    # Main Tick()
    def tick(self, frame_bgr_or_rgb: np.ndarray) -> Optional[Pose]:
        gray = self._to_gray(frame_bgr_or_rgb)
        pts_xy, des = self._extract(gray)

        # Refresh keypoint debug view
        if self.cfg.debug_draw_keypoints and pts_xy is not None and len(pts_xy) > 0:
            self._dbg_keypoints_bgr = self._render_keypoints(gray, pts_xy)
        elif self.cfg.debug_draw_keypoints:
            self._dbg_keypoints_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Not enough features
        if pts_xy is None or des is None or len(pts_xy) < 50:
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Create current frame with current pose estimate (will be updated if motion succeeds)
        cur = _Frame(gray, pts_xy, des, self._Tcw.copy())

        if self._last is None:
            self._last = cur
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Estimate motion last -> cur
        ok, idx_cur, idx_last, R, t = self._estimate_motion(last=self._last, cur=cur)

        if not ok:
            # If tracking fails, reset reference to current frame.
            # Pose stays as last good pose.
            self._last = cur
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Scale translation (monocular ambiguity)
        t = t.reshape(3)
        t_norm = float(np.linalg.norm(t))
        if t_norm > 1e-12:
            t = (t / t_norm) * float(self.cfg.translation_step)

        # Build relative transform T_cur_last (camera last -> camera cur), but in SE(3) form.
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t

        # Integrate pose:
        # If Tcw maps world->lastCam, then world->curCam = (last->cur) @ (world->last)
        self._Tcw = T_rel @ self._Tcw
        cur.pose_Tcw = self._Tcw.copy()
        if self.cfg.debug_draw_matches:
            self._dbg_matches_bgr = self._render_inlier_matches(self._last, cur, idx_last, idx_cur)

        # Triangulate sparse points for mapping/debug
        pts_cur = cur.kps_xy[idx_cur].astype(np.float64)
        pts_last = self._last.kps_xy[idx_last].astype(np.float64)

        # Parallax (normalized image coords) filter
        parallax_norm = self._parallax_norm(pts_last, pts_cur)
        parallax_ok = parallax_norm > float(self.cfg.min_parallax_w)

        Xw = _triangulate_Xw(cur.pose_Tcw, self._last.pose_Tcw, self.K, pts_cur, pts_last)

        # Positive depth in both cameras filter + finite + depth range
        depth_ok = self._in_front_of_both(cur.pose_Tcw, self._last.pose_Tcw, Xw)

        good = parallax_ok & depth_ok
        Xw_good = Xw[good]

        if Xw_good.shape[0] > 0:
            self.shared_map.add_map_points(Xw_good)

        self._last = cur
        self._publish_pose()
        return self.shared_map.poses.get(self.car_id)

    # ---------------- Internals ----------------

    def _publish_pose(self) -> None:
        """
        We maintain Tcw = world->camera.
        The camera position in world coordinates is Twc translation, where Twc = inv(Tcw).
        """
        Twc = self._invert_se3(self._Tcw)
        x = float(Twc[0, 3])
        y = float(Twc[1, 3])
        theta = self._yaw_from_R(Twc[:3, :3])
        self.shared_map.set_pose(self.car_id, Pose(x=x, y=y, theta=theta))

    @staticmethod
    def _invert_se3(T: np.ndarray) -> np.ndarray:
        """Fast inverse for SE(3) transform."""
        R = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=T.dtype)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

    @staticmethod
    def _yaw_from_R(R: np.ndarray) -> float:
        # yaw from rotation matrix (Z-up-ish assumption). Works “okay” for small pitch/roll.
        return float(math.atan2(R[1, 0], R[0, 0]))

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            g = img
        else:
            order = (self.cfg.input_color_order or "bgr").lower().strip()
            if order == "rgb":
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ensure contiguous uint8 for OpenCV feature extractors
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(g)

    def _extract(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
        )
        if pts is None:
            return None, None

        kps = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 20.0) for p in pts]
        kps, des = self.orb.compute(gray, kps)
        if des is None or kps is None or len(kps) == 0:
            return None, None

        xy = np.array([kp.pt for kp in kps], dtype=np.float32)
        return xy, des

    def _estimate_motion(
        self,
        last: _Frame,
        cur: _Frame,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate motion from last -> cur.
        Returns:
          ok, idx_cur_inliers, idx_last_inliers, R, t
        where R,t map points in last camera coords into cur camera coords.
        """
        # IMPORTANT: match last -> cur so queryIdx refers to last, trainIdx refers to cur
        matches_knn = self.bf.knnMatch(last.des, cur.des, k=2)

        idx_last: List[int] = []
        idx_cur: List[int] = []
        dists: List[float] = []

        for pair in matches_knn:
            # Robustness: some descriptors can return <2 neighbors
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.ratio_test * n.distance:
                idx_last.append(m.queryIdx)
                idx_cur.append(m.trainIdx)
                dists.append(float(m.distance))

        if len(idx_last) < 8:
            return False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1))

        # Keep best-N
        if len(idx_last) > self.cfg.keep_best:
            order = np.argsort(dists)[: self.cfg.keep_best]
            idx_last = [idx_last[i] for i in order]
            idx_cur = [idx_cur[i] for i in order]

        pts_last = last.kps_xy[np.array(idx_last)].astype(np.float64)
        pts_cur = cur.kps_xy[np.array(idx_cur)].astype(np.float64)

        E, inliers = cv2.findEssentialMat(
            pts_last,
            pts_cur,
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

        pts_last_in = pts_last[inliers]
        pts_cur_in = pts_cur[inliers]

        # recoverPose returns R,t such that: X_cur = R * X_last + t
        _, R, t, _ = cv2.recoverPose(E, pts_last_in, pts_cur_in, self.K)

        idx_last_in = np.array(idx_last, dtype=np.int64)[inliers]
        idx_cur_in = np.array(idx_cur, dtype=np.int64)[inliers]

        return True, idx_cur_in, idx_last_in, R, t

    # Triangulation filters

    def _parallax_norm(self, pts1_px: np.ndarray, pts2_px: np.ndarray) -> np.ndarray:
        """
        Approx parallax magnitude in *normalized image coordinates*.
        This is cheap and works well as a sanity filter.
        """
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        p1 = np.column_stack(((pts1_px[:, 0] - cx) / fx, (pts1_px[:, 1] - cy) / fy))
        p2 = np.column_stack(((pts2_px[:, 0] - cx) / fx, (pts2_px[:, 1] - cy) / fy))
        return np.linalg.norm(p2 - p1, axis=1)

    def _in_front_of_both(self, Tcw1: np.ndarray, Tcw2: np.ndarray, Xw: np.ndarray) -> np.ndarray:
        """
        Keep points with positive depth in both cameras + reasonable depth range.
        Depth is the Z coordinate in each camera frame when using Tcw (world->camera).
        """
        Xw_h = _add_ones(Xw).T  # (4,N)

        Xc1 = (Tcw1 @ Xw_h).T  # (N,4)
        Xc2 = (Tcw2 @ Xw_h).T

        z1 = Xc1[:, 2]
        z2 = Xc2[:, 2]

        finite = np.isfinite(z1) & np.isfinite(z2)
        in_front = (z1 > self.cfg.min_depth) & (z2 > self.cfg.min_depth)
        not_too_far = (z1 < self.cfg.max_depth) & (z2 < self.cfg.max_depth)

        return finite & in_front & not_too_far

    # Debug rendering

    @staticmethod
    def _render_keypoints(gray: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """
        Return BGR image: grayscale background + green keypoints.
        (BGR output makes cv2.imshow / drawMatches behave nicely.)
        """
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Draw small circles (cheaper than building KeyPoint objects)
        for (x, y) in pts_xy:
            cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"kps: {len(pts_xy)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _render_inlier_matches(self, last: _Frame, cur: _Frame, idx_last: np.ndarray, idx_cur: np.ndarray) -> np.ndarray:
        """
        Draw inlier matches (last -> cur) side-by-side.
        """
        # Subsample for display if needed
        if idx_last.shape[0] > self.cfg.debug_max_match_draw:
            sel = np.linspace(0, idx_last.shape[0] - 1, self.cfg.debug_max_match_draw).astype(np.int64)
            idx_last = idx_last[sel]
            idx_cur = idx_cur[sel]

        kp_last = [cv2.KeyPoint(float(x), float(y), 20.0) for (x, y) in last.kps_xy]
        kp_cur = [cv2.KeyPoint(float(x), float(y), 20.0) for (x, y) in cur.kps_xy]

        matches: List[cv2.DMatch] = []
        for il, ic in zip(idx_last.tolist(), idx_cur.tolist()):
            matches.append(cv2.DMatch(_queryIdx=int(il), _trainIdx=int(ic), _distance=0.0))

        vis = cv2.drawMatches(
            last.img_gray, kp_last,
            cur.img_gray, kp_cur,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            vis,
            f"inliers: {len(matches)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return vis