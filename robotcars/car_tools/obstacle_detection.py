from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

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

    min_parallax_w: float = 0.003
    min_depth: float = 0.05
    max_depth: float = 200.0

    min_median_flow_px: float = 1.0
    min_triangulation_flow_px: float = 1.5

    debug_draw_keypoints: bool = True
    debug_draw_matches: bool = True
    debug_max_match_draw: int = 80

    forward_sign: float = 1.0
    pose_ema_alpha: float = 0.2

    confidence_ema_alpha: float = 0.25
    confidence_good_threshold: float = 0.55
    confidence_min_features: int = 140
    confidence_min_inliers: int = 45
    confidence_min_sharpness: float = 45.0
    confidence_min_contrast: float = 25.0

    gray_source: Literal["luma", "green", "y_channel"] = "green"
    gray_flat_field_correction: bool = True
    gray_flat_field_sigma: float = 28.0
    gray_flat_field_strength: float = 0.85
    gray_use_clahe: bool = True
    gray_clahe_clip_limit: float = 2.2
    gray_clahe_tile_size: int = 8
    gray_gamma: float = 1.0
    gray_unsharp_amount: float = 0.0

    auto_white_balance: bool = True
    max_channel_gain: float = 1.8


@dataclass
class VslamStatus:
    tracking_ok: bool = False
    feature_count: int = 0
    inlier_count: int = 0
    sharpness: float = 0.0
    contrast: float = 0.0
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    confidence: float = 0.0
    high_confidence: bool = False
    blurry: bool = True
    median_flow_px: float = 0.0


@dataclass
class _MotionEstimate:
    ok: bool
    idx_cur: np.ndarray
    idx_last: np.ndarray
    R: np.ndarray
    t: np.ndarray
    median_flow_px: float = 0.0


class _Frame:
    __slots__ = ("img_gray", "kps_xy", "des", "pose_Tcw")

    def __init__(self, img_gray: np.ndarray, kps_xy: np.ndarray, des: np.ndarray, pose_Tcw: np.ndarray):
        self.img_gray = img_gray
        self.kps_xy = kps_xy
        self.des = des
        self.pose_Tcw = pose_Tcw


def _add_ones(xyz: np.ndarray) -> np.ndarray:
    return np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)


def _triangulate_Xw(
    Tcw1: np.ndarray,
    Tcw2: np.ndarray,
    K: np.ndarray,
    pts1_px: np.ndarray,
    pts2_px: np.ndarray,
) -> np.ndarray:
    P1 = K @ Tcw1[:3, :]
    P2 = K @ Tcw2[:3, :]
    pts4 = cv2.triangulatePoints(P1, P2, pts1_px.T, pts2_px.T).T
    w = np.where(np.abs(pts4[:, 3:4]) < 1e-12, 1e-12, pts4[:, 3:4])
    return pts4[:, :3] / w


class MonocularVSLAM:
    """
    Feature-based monocular VO/SLAM front-end.

    Important behavior:
    - `recoverPose()` gives translation direction only.
    - We therefore gate translation by observed image motion before applying `translation_step`.
    - This prevents the pose from "walking away" while the car is stationary.
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
        self._pose_filt: Optional[Pose] = None
        self._status = VslamStatus()
        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

        self._Tcw = np.eye(4, dtype=np.float64)

        self._dbg_keypoints_bgr: Optional[np.ndarray] = None
        self._dbg_matches_bgr: Optional[np.ndarray] = None

        self._publish_pose()

    # ---------------- Public diagnostics ----------------

    def get_debug_keypoints_frame(self) -> Optional[np.ndarray]:
        return None if self._dbg_keypoints_bgr is None else self._dbg_keypoints_bgr.copy()

    def get_debug_matches_frame(self) -> Optional[np.ndarray]:
        return None if self._dbg_matches_bgr is None else self._dbg_matches_bgr.copy()

    def get_status(self) -> VslamStatus:
        s = self._status
        return VslamStatus(
            tracking_ok=bool(s.tracking_ok),
            feature_count=int(s.feature_count),
            inlier_count=int(s.inlier_count),
            sharpness=float(s.sharpness),
            contrast=float(s.contrast),
            mean_r=float(s.mean_r),
            mean_g=float(s.mean_g),
            mean_b=float(s.mean_b),
            confidence=float(s.confidence),
            high_confidence=bool(s.high_confidence),
            blurry=bool(s.blurry),
            median_flow_px=float(s.median_flow_px),
        )

    def slam_quality(self, *, min_confidence: float = 0.55) -> Tuple[float, bool, bool, int, float]:
        s = self.get_status()
        conf = float(s.confidence)
        high_conf = bool(s.high_confidence and conf >= float(min_confidence))
        return conf, high_conf, bool(s.blurry), int(s.inlier_count), float(s.contrast)

    # ---------------- Main update ----------------

    def tick(self, frame_rgb: np.ndarray, translation_step: Optional[float] = None) -> Optional[Pose]:
        gray, frame_stats = self._to_gray(frame_rgb)
        pts_xy, des = self._extract(gray)

        sharpness = self._frame_sharpness(gray)
        contrast = self._frame_contrast(gray)
        feature_count = 0 if pts_xy is None else int(len(pts_xy))

        if self.cfg.debug_draw_keypoints:
            self._dbg_keypoints_bgr = (
                self._render_keypoints(gray, pts_xy)
                if pts_xy is not None and len(pts_xy) > 0
                else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            )
        else:
            self._dbg_keypoints_bgr = None

        if pts_xy is None or des is None or len(pts_xy) < 50:
            self._dbg_matches_bgr = None
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                median_flow_px=0.0,
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        cur = _Frame(gray, pts_xy, des, self._Tcw.copy())

        if self._last is None:
            self._last = cur
            self._dbg_matches_bgr = None
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                median_flow_px=0.0,
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        motion = self._estimate_motion(last=self._last, cur=cur)
        if not motion.ok:
            self._last = cur
            self._dbg_matches_bgr = None
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                median_flow_px=0.0,
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        step = float(self.cfg.translation_step if translation_step is None else translation_step)
        stationary = motion.median_flow_px < float(self.cfg.min_median_flow_px) or step <= 0.0

        if stationary:
            T_rel = np.eye(4, dtype=np.float64)
        else:
            t = motion.t.reshape(3)
            t_norm = float(np.linalg.norm(t))
            if t_norm > 1e-12:
                t = (t / t_norm) * step
            else:
                t = np.zeros(3, dtype=np.float64)

            T_rel = np.eye(4, dtype=np.float64)
            T_rel[:3, :3] = motion.R
            T_rel[:3, 3] = t

        self._Tcw = T_rel @ self._Tcw
        cur.pose_Tcw = self._Tcw.copy()

        if self.cfg.debug_draw_matches:
            self._dbg_matches_bgr = self._render_inlier_matches(self._last, cur, motion.idx_last, motion.idx_cur)
        else:
            self._dbg_matches_bgr = None

        if (not stationary) and motion.median_flow_px >= float(self.cfg.min_triangulation_flow_px):
            pts_cur = cur.kps_xy[motion.idx_cur].astype(np.float64)
            pts_last = self._last.kps_xy[motion.idx_last].astype(np.float64)

            parallax_norm = self._parallax_norm(pts_last, pts_cur)
            parallax_ok = parallax_norm > float(self.cfg.min_parallax_w)

            Xw = _triangulate_Xw(cur.pose_Tcw, self._last.pose_Tcw, self.K, pts_cur, pts_last)
            depth_ok = self._in_front_of_both(cur.pose_Tcw, self._last.pose_Tcw, Xw)

            good = parallax_ok & depth_ok
            Xw_good = Xw[good]

            if Xw_good.shape[0] > 0:
                Xnav = np.column_stack((Xw_good[:, 2], -Xw_good[:, 0], -Xw_good[:, 1])).astype(np.float32)
                self.shared_map.add_map_points(Xnav)

        self._update_status(
            tracking_ok=True,
            feature_count=feature_count,
            inlier_count=int(motion.idx_cur.shape[0]),
            sharpness=sharpness,
            contrast=contrast,
            mean_r=frame_stats["mean_r"],
            mean_g=frame_stats["mean_g"],
            mean_b=frame_stats["mean_b"],
            median_flow_px=motion.median_flow_px,
        )
        self._last = cur
        self._publish_pose()
        return self.shared_map.poses.get(self.car_id)

    # ---------------- Pose publishing ----------------

    def _publish_pose(self) -> None:
        """
        Convert world->camera pose into a planar nav pose:
        - x: forward
        - y: left
        - theta: heading of the camera forward axis
        """
        Twc = self._invert_se3(self._Tcw)
        p = Twc[:3, 3]
        Rwc = Twc[:3, :3]

        s = float(self.cfg.forward_sign)
        x_raw = float(s * p[2])
        y_raw = float(s * (-p[0]))

        fwd = Rwc[:, 2]
        theta_raw = float(math.atan2(-s * fwd[0], s * fwd[2])) if (abs(fwd[0]) + abs(fwd[2]) > 1e-9) else 0.0
        raw_pose = Pose(x=x_raw, y=y_raw, theta=theta_raw)

        alpha = float(self.cfg.pose_ema_alpha)
        if alpha <= 0.0 or self._pose_filt is None:
            self._pose_filt = raw_pose
        else:
            xf = (1.0 - alpha) * self._pose_filt.x + alpha * raw_pose.x
            yf = (1.0 - alpha) * self._pose_filt.y + alpha * raw_pose.y

            c0, s0 = math.cos(self._pose_filt.theta), math.sin(self._pose_filt.theta)
            c1, s1 = math.cos(raw_pose.theta), math.sin(raw_pose.theta)
            cf = (1.0 - alpha) * c0 + alpha * c1
            sf = (1.0 - alpha) * s0 + alpha * s1
            thf = math.atan2(sf, cf)

            self._pose_filt = Pose(x=float(xf), y=float(yf), theta=float(thf))

        self.shared_map.set_pose(self.car_id, self._pose_filt)

    @staticmethod
    def _invert_se3(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=T.dtype)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

    # ---------------- Status ----------------

    def _update_status(
        self,
        *,
        tracking_ok: bool,
        feature_count: int,
        inlier_count: int,
        sharpness: float,
        contrast: float,
        mean_r: float,
        mean_g: float,
        mean_b: float,
        median_flow_px: float,
    ) -> None:
        f_ref = max(1.0, float(self.cfg.confidence_min_features))
        i_ref = max(1.0, float(self.cfg.confidence_min_inliers))
        s_ref = max(1e-6, float(self.cfg.confidence_min_sharpness))
        c_ref = max(1e-6, float(self.cfg.confidence_min_contrast))

        feat_score = float(np.clip(float(feature_count) / f_ref, 0.0, 1.0))
        inlier_score = float(np.clip(float(inlier_count) / i_ref, 0.0, 1.0))
        sharp_score = float(np.clip(float(sharpness) / s_ref, 0.0, 1.0))
        contrast_score = float(np.clip(float(contrast) / c_ref, 0.0, 1.0))

        raw_conf = 0.25 * feat_score + 0.45 * inlier_score + 0.20 * sharp_score + 0.10 * contrast_score
        if not tracking_ok:
            raw_conf *= 0.35

        alpha = float(np.clip(self.cfg.confidence_ema_alpha, 0.0, 1.0))
        conf = raw_conf if alpha <= 0.0 else ((1.0 - alpha) * float(self._status.confidence) + alpha * raw_conf)

        self._status = VslamStatus(
            tracking_ok=bool(tracking_ok),
            feature_count=int(feature_count),
            inlier_count=int(inlier_count),
            sharpness=float(sharpness),
            contrast=float(contrast),
            mean_r=float(mean_r),
            mean_g=float(mean_g),
            mean_b=float(mean_b),
            confidence=float(conf),
            high_confidence=bool(tracking_ok and conf >= float(self.cfg.confidence_good_threshold)),
            blurry=bool(sharpness < float(self.cfg.confidence_min_sharpness)),
            median_flow_px=float(median_flow_px),
        )

    # ---------------- Image preprocessing ----------------

    @staticmethod
    def _frame_sharpness(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        small = cv2.resize(gray, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(small, cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def _frame_contrast(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        return float(np.std(gray))

    def _to_gray(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        stats = {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0}

        if img.ndim == 2:
            gray = img
        else:
            color = img
            if color.dtype != np.uint8:
                color = np.clip(color, 0, 255).astype(np.uint8)
            color = np.ascontiguousarray(color)

            r = color[:, :, 0].astype(np.float32)
            g_ch = color[:, :, 1].astype(np.float32)
            b = color[:, :, 2].astype(np.float32)

            stats = {
                "mean_r": float(r.mean()),
                "mean_g": float(g_ch.mean()),
                "mean_b": float(b.mean()),
            }

            gray_source = str(self.cfg.gray_source).lower()
            if self.cfg.auto_white_balance and gray_source != "green":
                color = self._gray_world_balance(color, max_gain=float(self.cfg.max_channel_gain))

            if gray_source == "green":
                gray = color[:, :, 1]
            elif gray_source == "y_channel":
                ycc = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)
                gray = ycc[:, :, 0]
            else:
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        if self.cfg.gray_flat_field_correction:
            gray = self._flat_field_correct(
                gray,
                sigma=float(self.cfg.gray_flat_field_sigma),
                strength=float(self.cfg.gray_flat_field_strength),
            )

        gamma = float(self.cfg.gray_gamma)
        if abs(gamma - 1.0) > 1e-3:
            gray = cv2.LUT(gray, self._gamma_lut_for(gamma))

        if self.cfg.gray_use_clahe:
            tile = max(2, int(self.cfg.gray_clahe_tile_size))
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=float(self.cfg.gray_clahe_clip_limit),
                    tileGridSize=(tile, tile),
                )
            gray = self._clahe.apply(gray)

        amount = float(self.cfg.gray_unsharp_amount)
        if amount > 1e-3:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
            gray = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)

        return np.ascontiguousarray(gray), stats

    @staticmethod
    def _flat_field_correct(gray: np.ndarray, *, sigma: float, strength: float) -> np.ndarray:
        if gray.size == 0:
            return gray

        sigma = max(0.0, float(sigma))
        strength = float(np.clip(strength, 0.0, 1.0))
        if sigma <= 1e-3 or strength <= 1e-3:
            return gray

        src = gray.astype(np.float32)
        illum = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mean_illum = float(np.mean(illum))
        if mean_illum <= 1e-6:
            return gray

        corrected = src * (mean_illum / (illum + 1.0))
        corrected = np.clip(corrected, 0.0, 255.0)
        if strength < 0.999:
            corrected = (1.0 - strength) * src + strength * corrected

        return corrected.astype(np.uint8)

    def _gamma_lut_for(self, gamma: float) -> np.ndarray:
        gamma = max(1e-3, float(gamma))
        if self._gamma_lut is not None and abs(gamma - self._gamma_lut_value) < 1e-6:
            return self._gamma_lut

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        y = np.power(x, 1.0 / gamma)
        lut = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)

        self._gamma_lut = lut
        self._gamma_lut_value = gamma
        return lut

    @staticmethod
    def _gray_world_balance(img: np.ndarray, *, max_gain: float) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] < 3:
            return img

        out = img.astype(np.float32)
        r = out[:, :, 0]
        g = out[:, :, 1]
        b = out[:, :, 2]

        mr = float(np.mean(r))
        mg = float(np.mean(g))
        mb = float(np.mean(b))
        mean_all = (mr + mg + mb) / 3.0

        def gain(channel_mean: float) -> float:
            if channel_mean <= 1e-6:
                return 1.0
            return float(np.clip(mean_all / channel_mean, 0.6, max(1.0, max_gain)))

        gr = gain(mr)
        gg = gain(mg)
        gb = gain(mb)
        out[:, :, 0] *= gr
        out[:, :, 1] *= gg
        out[:, :, 2] *= gb

        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    # ---------------- Feature extraction + motion ----------------

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

    def _estimate_motion(self, last: _Frame, cur: _Frame) -> _MotionEstimate:
        matches_knn = self.bf.knnMatch(last.des, cur.des, k=2)

        idx_last: List[int] = []
        idx_cur: List[int] = []
        dists: List[float] = []

        for pair in matches_knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.ratio_test * n.distance:
                idx_last.append(m.queryIdx)
                idx_cur.append(m.trainIdx)
                dists.append(float(m.distance))

        if len(idx_last) < 8:
            return _MotionEstimate(False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1)), 0.0)

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
            return _MotionEstimate(False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1)), 0.0)

        inliers = inliers.reshape(-1).astype(bool)
        if int(inliers.sum()) < self.cfg.min_inliers:
            return _MotionEstimate(False, np.array([]), np.array([]), np.eye(3), np.zeros((3, 1)), 0.0)

        pts_last_in = pts_last[inliers]
        pts_cur_in = pts_cur[inliers]
        _, R, t, _ = cv2.recoverPose(E, pts_last_in, pts_cur_in, self.K)

        idx_last_in = np.array(idx_last, dtype=np.int64)[inliers]
        idx_cur_in = np.array(idx_cur, dtype=np.int64)[inliers]

        flow = np.linalg.norm(pts_cur_in - pts_last_in, axis=1)
        median_flow = float(np.median(flow)) if flow.size > 0 else 0.0

        return _MotionEstimate(True, idx_cur_in, idx_last_in, R, t, median_flow)

    # ---------------- Triangulation filters ----------------

    def _parallax_norm(self, pts1_px: np.ndarray, pts2_px: np.ndarray) -> np.ndarray:
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        p1 = np.column_stack(((pts1_px[:, 0] - cx) / fx, (pts1_px[:, 1] - cy) / fy))
        p2 = np.column_stack(((pts2_px[:, 0] - cx) / fx, (pts2_px[:, 1] - cy) / fy))
        return np.linalg.norm(p2 - p1, axis=1)

    def _in_front_of_both(self, Tcw1: np.ndarray, Tcw2: np.ndarray, Xw: np.ndarray) -> np.ndarray:
        Xw_h = _add_ones(Xw).T
        Xc1 = (Tcw1 @ Xw_h).T
        Xc2 = (Tcw2 @ Xw_h).T

        z1 = Xc1[:, 2]
        z2 = Xc2[:, 2]

        finite = np.isfinite(z1) & np.isfinite(z2)
        in_front = (z1 > self.cfg.min_depth) & (z2 > self.cfg.min_depth)
        not_too_far = (z1 < self.cfg.max_depth) & (z2 < self.cfg.max_depth)

        return finite & in_front & not_too_far

    # ---------------- Debug rendering ----------------

    @staticmethod
    def _render_keypoints(gray: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y in pts_xy:
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
        if idx_last.shape[0] == 0 or idx_cur.shape[0] == 0:
            return cv2.cvtColor(cur.img_gray, cv2.COLOR_GRAY2BGR)

        if idx_last.shape[0] > self.cfg.debug_max_match_draw:
            sel = np.linspace(0, idx_last.shape[0] - 1, self.cfg.debug_max_match_draw).astype(np.int64)
            idx_last = idx_last[sel]
            idx_cur = idx_cur[sel]

        pts_last = last.kps_xy[idx_last]
        pts_cur = cur.kps_xy[idx_cur]

        kp_last = [cv2.KeyPoint(float(x), float(y), 20.0) for x, y in pts_last]
        kp_cur = [cv2.KeyPoint(float(x), float(y), 20.0) for x, y in pts_cur]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(kp_last))]

        vis = cv2.drawMatches(
            last.img_gray,
            kp_last,
            cur.img_gray,
            kp_cur,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            vis,
            f"inliers: {len(matches)} flow_med_px: {self._status.median_flow_px:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return vis