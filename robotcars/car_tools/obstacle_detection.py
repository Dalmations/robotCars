# car_tools/obstacle_detection.py
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

    # Triangulation filters
    # Typical values: 0.001 ~ 0.01 depending on resolution/FOV/motion.
    min_parallax_w: float = 0.003

    min_depth: float = 0.05
    max_depth: float = 200.0

    # Debug drawing
    debug_draw_keypoints: bool = True
    debug_draw_matches: bool = True
    debug_max_match_draw: int = 80
    # If forward motion makes pose.x decrease, set forward_sign = -1.0
    forward_sign: float = 1.0

    # Pose smoothing (EMA). 0 disables smoothing, 0.1-0.3 typical.
    pose_ema_alpha: float = 0.2

    # Input frame color order from camera pipeline ("rgb" or "bgr")
    input_color_order: str = "rgb"

    # Confidence gating thresholds
    confidence_ema_alpha: float = 0.25
    confidence_good_threshold: float = 0.55
    confidence_min_features: int = 140
    confidence_min_inliers: int = 45
    confidence_min_sharpness: float = 45.0
    confidence_min_contrast: float = 25.0

    # Preprocessing for low-light / low-contrast scenes
    # "green" is robust to RGB/BGR confusion and reduces chroma artifacts from bad AWB/lens shading.
    gray_source: Literal["luma", "green", "y_channel"] = "green"
    gray_flat_field_correction: bool = True
    gray_flat_field_sigma: float = 28.0
    gray_flat_field_strength: float = 0.85
    gray_use_clahe: bool = True
    gray_clahe_clip_limit: float = 2.2
    gray_clahe_tile_size: int = 8
    gray_gamma: float = 1.0
    gray_unsharp_amount: float = 0.0

    # Blue-cast correction (simple gray-world balancing)
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
    blue_ratio: float = 1.0
    confidence: float = 0.0
    high_confidence: bool = False
    blurry: bool = True


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
        self._pose_filt: Optional[Pose] = None
        self._status = VslamStatus()
        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

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

    def get_status(self) -> VslamStatus:
        """Return the latest SLAM tracking/quality status."""
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
            blue_ratio=float(s.blue_ratio),
            confidence=float(s.confidence),
            high_confidence=bool(s.high_confidence),
            blurry=bool(s.blurry),
        )

    # Main Tick()
    def tick(self, frame_bgr_or_rgb: np.ndarray, translation_step: Optional[float] = None) -> Optional[Pose]:
        gray, frame_stats = self._to_gray(frame_bgr_or_rgb)
        pts_xy, des = self._extract(gray)
        sharpness = self._frame_sharpness(gray)
        contrast = self._frame_contrast(gray)
        feature_count = 0 if pts_xy is None else int(len(pts_xy))

        # Refresh keypoint debug view
        if self.cfg.debug_draw_keypoints and pts_xy is not None and len(pts_xy) > 0:
            self._dbg_keypoints_bgr = self._render_keypoints(gray, pts_xy)
        elif self.cfg.debug_draw_keypoints:
            self._dbg_keypoints_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Not enough features
        if pts_xy is None or des is None or len(pts_xy) < 50:
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                blue_ratio=frame_stats["blue_ratio"],
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Create current frame with current pose estimate (will be updated if motion succeeds)
        cur = _Frame(gray, pts_xy, des, self._Tcw.copy())

        if self._last is None:
            self._last = cur
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                blue_ratio=frame_stats["blue_ratio"],
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Estimate motion last -> cur
        ok, idx_cur, idx_last, R, t = self._estimate_motion(last=self._last, cur=cur)

        if not ok:
            # If tracking fails, reset reference to current frame.
            # Pose stays as last good pose.
            self._last = cur
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=frame_stats["mean_r"],
                mean_g=frame_stats["mean_g"],
                mean_b=frame_stats["mean_b"],
                blue_ratio=frame_stats["blue_ratio"],
            )
            self._publish_pose()
            return self.shared_map.poses.get(self.car_id)

        # Scale translation (monocular ambiguity)
        t = t.reshape(3)
        t_norm = float(np.linalg.norm(t))

        step = float(self.cfg.translation_step if translation_step is None else translation_step)
        if t_norm > 1e-12 and step > 0.0:
            t = (t / t_norm) * step
        else:
            t = np.zeros(3, dtype=np.float64)

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
            # Store points in the same nav frame as pose:
            # nav_x = forward = Z
            # nav_y = left    = -X
            # nav_z = up      = -Y
            Xnav = np.column_stack((Xw_good[:, 2], -Xw_good[:, 0], -Xw_good[:, 1])).astype(np.float32)
            self.shared_map.add_map_points(Xnav)

        self._update_status(
            tracking_ok=True,
            feature_count=feature_count,
            inlier_count=int(idx_cur.shape[0]),
            sharpness=sharpness,
            contrast=contrast,
            mean_r=frame_stats["mean_r"],
            mean_g=frame_stats["mean_g"],
            mean_b=frame_stats["mean_b"],
            blue_ratio=frame_stats["blue_ratio"],
        )
        self._last = cur
        self._publish_pose()
        return self.shared_map.poses.get(self.car_id)

    # ---------------- Internals ----------------

    def _publish_pose(self) -> None:
        """
        We maintain Tcw = world->camera where the initial camera frame follows OpenCV convention:
        +X right, +Y down, +Z forward.

        Publish a planar nav pose:
        x = forward  (± world Z)
        y = left     (∓ world X)
        theta from camera forward axis projected into XZ plane.

        Also apply optional EMA smoothing to reduce jerk.
        """
        Twc = self._invert_se3(self._Tcw)
        p = Twc[:3, 3]
        Rwc = Twc[:3, :3]

        s = float(self.cfg.forward_sign)

        # Planar position (forward/left). Flip signs together to keep frame right-handed.
        x_raw = float(s * p[2])         # forward
        y_raw = float(s * (-p[0]))      # left

        # Heading: camera forward axis in world
        fwd = Rwc[:, 2]
        # Use same sign flip so "forward" direction stays consistent
        theta_raw = float(math.atan2(-s * fwd[0], s * fwd[2])) if (abs(fwd[0]) + abs(fwd[2]) > 1e-9) else 0.0

        pose_raw = Pose(x=x_raw, y=y_raw, theta=theta_raw)

        a = float(self.cfg.pose_ema_alpha)
        if a <= 0.0 or self._pose_filt is None:
            self._pose_filt = pose_raw
        else:
            # EMA on x,y and circular EMA on theta
            xf = (1.0 - a) * self._pose_filt.x + a * pose_raw.x
            yf = (1.0 - a) * self._pose_filt.y + a * pose_raw.y

            # circular blend
            c0, s0 = math.cos(self._pose_filt.theta), math.sin(self._pose_filt.theta)
            c1, s1 = math.cos(pose_raw.theta), math.sin(pose_raw.theta)
            cf = (1.0 - a) * c0 + a * c1
            sf = (1.0 - a) * s0 + a * s1
            thf = math.atan2(sf, cf)

            self._pose_filt = Pose(x=float(xf), y=float(yf), theta=float(thf))

        self.shared_map.set_pose(self.car_id, self._pose_filt)

    @staticmethod
    def _invert_se3(T: np.ndarray) -> np.ndarray:
        """Fast inverse for SE(3) transform."""
        R = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=T.dtype)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

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
        blue_ratio: float,
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

        a = float(np.clip(self.cfg.confidence_ema_alpha, 0.0, 1.0))
        conf = raw_conf if a <= 0.0 else ((1.0 - a) * float(self._status.confidence) + a * raw_conf)

        self._status = VslamStatus(
            tracking_ok=bool(tracking_ok),
            feature_count=int(feature_count),
            inlier_count=int(inlier_count),
            sharpness=float(sharpness),
            contrast=float(contrast),
            mean_r=float(mean_r),
            mean_g=float(mean_g),
            mean_b=float(mean_b),
            blue_ratio=float(blue_ratio),
            confidence=float(conf),
            high_confidence=bool(tracking_ok and conf >= float(self.cfg.confidence_good_threshold)),
            blurry=bool(sharpness < float(self.cfg.confidence_min_sharpness)),
        )

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

    @staticmethod
    def _yaw_from_R(R: np.ndarray) -> float:
        # yaw from rotation matrix (Z-up-ish assumption). Works “okay” for small pitch/roll.
        return float(math.atan2(R[1, 0], R[0, 0]))

    def _to_gray(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        stats = {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0, "blue_ratio": 1.0}

        if img.ndim == 2:
            g = img
        else:
            c = img
            if c.dtype != np.uint8:
                c = np.clip(c, 0, 255).astype(np.uint8)
            c = np.ascontiguousarray(c)

            order = str(self.cfg.input_color_order).lower()
            if order == "bgr":
                b = c[:, :, 0].astype(np.float32)
                g_ch = c[:, :, 1].astype(np.float32)
                r = c[:, :, 2].astype(np.float32)
            else:
                r = c[:, :, 0].astype(np.float32)
                g_ch = c[:, :, 1].astype(np.float32)
                b = c[:, :, 2].astype(np.float32)

            mean_r = float(r.mean())
            mean_g = float(g_ch.mean())
            mean_b = float(b.mean())
            blue_ratio = float(mean_b / max(1.0, 0.5 * (mean_r + mean_g)))
            stats = {"mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b, "blue_ratio": blue_ratio}

            gray_source = str(self.cfg.gray_source).lower()
            # If we only use the green channel, skip RGB balancing to avoid injecting color gain artifacts.
            if self.cfg.auto_white_balance and gray_source != "green":
                c = self._gray_world_balance(c, order=order, max_gain=float(self.cfg.max_channel_gain))

            if gray_source == "green":
                # Channel index 1 is green for both RGB and BGR layouts.
                g = c[:, :, 1]
            elif gray_source == "y_channel":
                if order == "bgr":
                    ycc = cv2.cvtColor(c, cv2.COLOR_BGR2YCrCb)
                else:
                    ycc = cv2.cvtColor(c, cv2.COLOR_RGB2YCrCb)
                g = ycc[:, :, 0]
            else:
                if order == "bgr":
                    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                else:
                    g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)

        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)

        if self.cfg.gray_flat_field_correction:
            g = self._flat_field_correct(
                g,
                sigma=float(self.cfg.gray_flat_field_sigma),
                strength=float(self.cfg.gray_flat_field_strength),
            )

        gamma = float(self.cfg.gray_gamma)
        if abs(gamma - 1.0) > 1e-3:
            lut = self._gamma_lut_for(gamma)
            g = cv2.LUT(g, lut)

        if self.cfg.gray_use_clahe:
            tile = max(2, int(self.cfg.gray_clahe_tile_size))
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=float(self.cfg.gray_clahe_clip_limit),
                    tileGridSize=(tile, tile),
                )
            g = self._clahe.apply(g)

        amount = float(self.cfg.gray_unsharp_amount)
        if amount > 1e-3:
            blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0, sigmaY=1.0)
            g = cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)

        return np.ascontiguousarray(g), stats

    @staticmethod
    def _flat_field_correct(gray: np.ndarray, *, sigma: float, strength: float) -> np.ndarray:
        """
        Remove slow illumination gradients (vignette / yellow-center-pink-edge style shading)
        by dividing by a heavily blurred illumination estimate.
        """
        if gray.size == 0:
            return gray

        s = max(0.0, float(sigma))
        k = float(np.clip(strength, 0.0, 1.0))
        if s <= 1e-3 or k <= 1e-3:
            return gray

        g = gray.astype(np.float32)
        illum = cv2.GaussianBlur(g, (0, 0), sigmaX=s, sigmaY=s)
        m = float(np.mean(illum))
        if m <= 1e-6:
            return gray

        corrected = g * (m / (illum + 1.0))
        corrected = np.clip(corrected, 0.0, 255.0)
        if k < 0.999:
            corrected = (1.0 - k) * g + k * corrected

        return corrected.astype(np.uint8)

    def _gamma_lut_for(self, gamma: float) -> np.ndarray:
        g = max(1e-3, float(gamma))
        if self._gamma_lut is not None and abs(g - self._gamma_lut_value) < 1e-6:
            return self._gamma_lut

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        y = np.power(x, 1.0 / g)
        lut = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
        self._gamma_lut = lut
        self._gamma_lut_value = g
        return lut

    @staticmethod
    def _gray_world_balance(img: np.ndarray, *, order: str, max_gain: float) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] < 3:
            return img

        out = img.astype(np.float32)

        if order == "bgr":
            b = out[:, :, 0]
            g = out[:, :, 1]
            r = out[:, :, 2]
        else:
            r = out[:, :, 0]
            g = out[:, :, 1]
            b = out[:, :, 2]

        mr = float(np.mean(r))
        mg = float(np.mean(g))
        mb = float(np.mean(b))
        m = (mr + mg + mb) / 3.0

        def gain(ch_mean: float) -> float:
            if ch_mean <= 1e-6:
                return 1.0
            return float(np.clip(m / ch_mean, 0.6, max(1.0, max_gain)))

        gr = gain(mr)
        gg = gain(mg)
        gb = gain(mb)

        if order == "bgr":
            out[:, :, 0] *= gb
            out[:, :, 1] *= gg
            out[:, :, 2] *= gr
        else:
            out[:, :, 0] *= gr
            out[:, :, 1] *= gg
            out[:, :, 2] *= gb

        return np.clip(out, 0.0, 255.0).astype(np.uint8)

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
        # match last -> cur so queryIdx refers to last, trainIdx refers to cur
        matches_knn = self.bf.knnMatch(last.des, cur.des, k=2)

        idx_last: List[int] = []
        idx_cur: List[int] = []
        dists: List[float] = []

        for pair in matches_knn:
            # Pair matches to new frame
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
