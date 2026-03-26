from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Literal

import cv2
import numpy as np

from car_tools.picarx_path_follower import integrate_pose, wrap_angle
from model import Pose
from coordination.shared_map import SharedMap


def _copy_pose(pose: Pose) -> Pose:
    return Pose(x=float(pose.x), y=float(pose.y), theta=float(pose.theta))


def _pose_distance(a: Pose, b: Pose) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


def _heading_difference(a: float, b: float) -> float:
    return abs(wrap_angle(float(a) - float(b)))


def _blend_pose(base_pose: Pose, correction_pose: Pose, alpha: float) -> Pose:
    blend = float(np.clip(float(alpha), 0.0, 1.0))
    if blend <= 0.0:
        return _copy_pose(base_pose)
    if blend >= 1.0:
        return _copy_pose(correction_pose)

    return Pose(
        x=float((1.0 - blend) * float(base_pose.x) + blend * float(correction_pose.x)),
        y=float((1.0 - blend) * float(base_pose.y) + blend * float(correction_pose.y)),
        theta=float(wrap_angle(float(base_pose.theta) + blend * wrap_angle(float(correction_pose.theta) - float(base_pose.theta)))),
    )


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
    # Feature extraction
    extractor: Literal["orb", "gftt_orb"] = "orb"
    max_corners: int = 2000
    quality_level: float = 0.01
    min_distance: int = 10
    orb_nfeatures: int = 2500

    # Matching
    ratio_test: float = 0.75
    keep_best: int = 800

    # Geometry
    ransac_prob: float = 0.999
    ransac_thresh: float = 1.0
    min_inliers: int = 12          # essential-matrix inliers
    min_pose_inliers: int = 10     # recoverPose support
    max_rotation_deg: float = 45.0
    translation_step: float = 1.0

    # Motion gating
    min_median_flow_px: float = 0.75
    min_triangulation_flow_px: float = 1.25
    min_inlier_coverage: float = 0.02
    stale_frame_delta_mean: float = 0.35
    max_weak_hold_frames: int = 3

    # Triangulation filters
    min_parallax_w: float = 0.003
    min_depth: float = 0.05
    max_depth: float = 200.0

    # Debug
    debug_draw_keypoints: bool = True
    debug_draw_matches: bool = True
    debug_max_match_draw: int = 80

    # Pose frame publishing
    forward_sign: float = 1.0
    pose_ema_alpha: float = 0.2
    publish_pose_to_shared_map: bool = False

    # Confidence
    confidence_ema_alpha: float = 0.25
    confidence_good_threshold: float = 0.55
    confidence_min_features: int = 140
    confidence_min_inliers: float = 20.0
    confidence_min_sharpness: float = 45.0
    confidence_min_contrast: float = 25.0

    # Image preprocessing
    input_color_order: Literal["rgb", "bgr"] = "rgb"
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
    frame_delta_mean: float = 0.0
    inlier_coverage: float = 0.0
    gate_reason: str = "startup"

    match_count: int = 0
    kept_match_count: int = 0
    essential_inlier_count: int = 0
    recover_pose_count: int = 0

    requested_step: float = 0.0
    applied_step: float = 0.0
    translation_enabled: bool = False
    rotation_deg: float = 0.0

    raw_pose_x: float = 0.0
    raw_pose_y: float = 0.0
    raw_pose_theta: float = 0.0
    filtered_pose_x: float = 0.0
    filtered_pose_y: float = 0.0
    filtered_pose_theta: float = 0.0


@dataclass
class ConservativeCorrectionConfig:
    # How frugal to be about running the visual front-end at all.
    min_cycles_between_corrections: int = 4
    min_seconds_between_corrections: float = 0.4
    min_translation_between_corrections: float = 0.75
    min_heading_change_between_corrections_deg: float = 8.0

    # Visual pose must agree with dead reckoning to be trusted.
    position_agreement_threshold: float = 0.75
    heading_agreement_threshold_deg: float = 12.0

    # Accepted corrections are blended in gradually instead of snapping.
    correction_alpha: float = 0.25
    min_visual_confidence: float = 0.55


@dataclass
class ConservativeCorrectionResult:
    attempted: bool
    accepted: bool
    reason: str
    dead_reckoning_pose: Pose
    obstacle_pose: Optional[Pose] = None
    fused_pose: Optional[Pose] = None
    position_error: float = 0.0
    heading_error_deg: float = 0.0
    visual_confidence: float = 0.0


@dataclass
class _MotionEstimate:
    ok: bool
    idx_cur: np.ndarray
    idx_last: np.ndarray
    R: np.ndarray
    t: np.ndarray

    median_flow_px: float = 0.0
    inlier_coverage: float = 0.0
    frame_delta_mean: float = 0.0

    match_count: int = 0
    kept_match_count: int = 0
    essential_inlier_count: int = 0
    recover_pose_count: int = 0
    rotation_deg: float = 0.0
    reason: str = "startup"


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
    Feature-based monocular VO front-end with simple planar odometry fusion.

    Key behavior:
    - When recoverPose support is healthy, visual odometry supplies yaw.
    - Translation magnitude comes from the externally provided step estimate.
    - Brief weak-frame gaps can keep propagating pose from odometry.
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
        self._raw_pose_latest = Pose(0.0, 0.0, 0.0)
        self._status = VslamStatus()

        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

        self._Tcw = np.eye(4, dtype=np.float64)
        self._weak_hold_count: int = 0

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
        raw_pose = self._raw_pose_latest
        filt_pose = self._pose_filt if self._pose_filt is not None else raw_pose
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
            frame_delta_mean=float(s.frame_delta_mean),
            inlier_coverage=float(s.inlier_coverage),
            gate_reason=str(s.gate_reason),
            match_count=int(s.match_count),
            kept_match_count=int(s.kept_match_count),
            essential_inlier_count=int(s.essential_inlier_count),
            recover_pose_count=int(s.recover_pose_count),
            requested_step=float(s.requested_step),
            applied_step=float(s.applied_step),
            translation_enabled=bool(s.translation_enabled),
            rotation_deg=float(s.rotation_deg),
            raw_pose_x=float(raw_pose.x),
            raw_pose_y=float(raw_pose.y),
            raw_pose_theta=float(raw_pose.theta),
            filtered_pose_x=float(filt_pose.x),
            filtered_pose_y=float(filt_pose.y),
            filtered_pose_theta=float(filt_pose.theta),
        )

    def get_pose_estimate(self, *, filtered: bool = True) -> Pose:
        pose = self._pose_filt if filtered and self._pose_filt is not None else self._raw_pose_latest
        return _copy_pose(pose)

    def set_pose_estimate(
        self,
        pose: Pose,
        *,
        reset_filter: bool = True,
        sync_last_frame: bool = True,
        publish: bool = False,
    ) -> Pose:
        pose_copy = _copy_pose(pose)
        self._Tcw = self._planar_pose_to_Tcw(pose_copy)
        self._raw_pose_latest = pose_copy

        if reset_filter or self._pose_filt is None:
            self._pose_filt = _copy_pose(pose_copy)
        else:
            self._pose_filt = _blend_pose(self._pose_filt, pose_copy, float(self.cfg.pose_ema_alpha))

        if sync_last_frame and self._last is not None:
            self._last.pose_Tcw = self._Tcw.copy()

        if publish or bool(self.cfg.publish_pose_to_shared_map):
            self.shared_map.set_pose(self.car_id, self.get_pose_estimate(filtered=True))

        return self.get_pose_estimate(filtered=True)

    def slam_quality(self, *, min_confidence: float = 0.55) -> Tuple[float, bool, bool, int, float]:
        s = self.get_status()
        conf = float(s.confidence)
        high_conf = bool(s.high_confidence and conf >= float(min_confidence))
        return conf, high_conf, bool(s.blurry), int(s.inlier_count), float(s.contrast)

    # ---------------- Main update ----------------

    def tick(
        self,
        frame_rgb_or_bgr: np.ndarray,
        translation_step: Optional[float] = None,
        odom_yaw_delta: Optional[float] = None,
    ) -> Optional[Pose]:
        gray, frame_stats = self._to_gray(frame_rgb_or_bgr)
        pts_xy, des = self._extract(gray)
        requested_step = float(self.cfg.translation_step if translation_step is None else translation_step)
        odom_yaw_delta = 0.0 if odom_yaw_delta is None else float(odom_yaw_delta)

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
            return self._handle_weak_frame(
                cur=None,
                tracking_reason="too_few_features",
                feature_count=feature_count,
                sharpness=sharpness,
                contrast=contrast,
                frame_stats=frame_stats,
                requested_step=requested_step,
                odom_yaw_delta=odom_yaw_delta,
            )

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
                frame_delta_mean=0.0,
                inlier_coverage=0.0,
                gate_reason="bootstrap_no_last",
                requested_step=requested_step,
            )
            self._publish_pose(update_filter=False)
            self._annotate_debug_frames()
            return self.get_pose_estimate(filtered=True)

        frame_delta_mean = self._frame_delta_mean(self._last.img_gray, cur.img_gray)
        if frame_delta_mean < float(self.cfg.stale_frame_delta_mean):
            return self._handle_weak_frame(
                cur=cur,
                tracking_reason="stale_frame",
                feature_count=feature_count,
                sharpness=sharpness,
                contrast=contrast,
                frame_stats=frame_stats,
                requested_step=requested_step,
                odom_yaw_delta=odom_yaw_delta,
                frame_delta_mean=frame_delta_mean,
            )

        motion = self._estimate_motion(last=self._last, cur=cur, frame_delta_mean=frame_delta_mean)
        if not motion.ok:
            return self._handle_weak_frame(
                cur=cur,
                tracking_reason=motion.reason,
                feature_count=feature_count,
                sharpness=sharpness,
                contrast=contrast,
                frame_stats=frame_stats,
                requested_step=requested_step,
                odom_yaw_delta=odom_yaw_delta,
                median_flow_px=motion.median_flow_px,
                frame_delta_mean=motion.frame_delta_mean,
                inlier_coverage=motion.inlier_coverage,
                match_count=motion.match_count,
                kept_match_count=motion.kept_match_count,
                essential_inlier_count=motion.essential_inlier_count,
                recover_pose_count=motion.recover_pose_count,
                rotation_deg=motion.rotation_deg,
            )

        # Rotation validity is stricter than before: recoverPose support must be healthy.
        rotation_valid = motion.recover_pose_count >= int(self.cfg.min_pose_inliers)

        # Translation is a separate decision from rotation.
        translation_enabled = rotation_valid and abs(float(requested_step)) > 1e-9

        if not rotation_valid:
            return self._handle_weak_frame(
                cur=cur,
                tracking_reason="too_few_pose_inliers",
                feature_count=feature_count,
                sharpness=sharpness,
                contrast=contrast,
                frame_stats=frame_stats,
                requested_step=requested_step,
                odom_yaw_delta=odom_yaw_delta,
                median_flow_px=motion.median_flow_px,
                frame_delta_mean=motion.frame_delta_mean,
                inlier_coverage=motion.inlier_coverage,
                match_count=motion.match_count,
                kept_match_count=motion.kept_match_count,
                essential_inlier_count=motion.essential_inlier_count,
                recover_pose_count=motion.recover_pose_count,
                rotation_deg=motion.rotation_deg,
            )

        applied_step = 0.0
        gate_reason = "rotation_only"

        self._weak_hold_count = 0
        vo_yaw_delta = self._yaw_delta_from_relative_rotation(motion.R)
        self._apply_planar_motion(forward_step=requested_step, yaw_delta=vo_yaw_delta)
        applied_step = abs(float(requested_step))
        gate_reason = "tracking_ok" if translation_enabled else "rotation_only_zero_step"
        cur.pose_Tcw = self._Tcw.copy()

        if self.cfg.debug_draw_matches:
            self._dbg_matches_bgr = self._render_inlier_matches(self._last, cur, motion.idx_last, motion.idx_cur)
        else:
            self._dbg_matches_bgr = None

        if translation_enabled and motion.median_flow_px >= float(self.cfg.min_triangulation_flow_px):
            pts_cur = cur.kps_xy[motion.idx_cur].astype(np.float64)
            pts_last = self._last.kps_xy[motion.idx_last].astype(np.float64)

            parallax_norm = self._parallax_norm(pts_last, pts_cur)
            parallax_ok = parallax_norm > float(self.cfg.min_parallax_w)

            Xw = _triangulate_Xw(cur.pose_Tcw, self._last.pose_Tcw, self.K, pts_cur, pts_last)
            depth_ok = self._in_front_of_both(cur.pose_Tcw, self._last.pose_Tcw, Xw)

            good = parallax_ok & depth_ok
            Xw_good = Xw[good]

            if Xw_good.shape[0] > 0:
                # nav frame: x=forward(z), y=left(-x), z=up(-y)
                Xnav = np.column_stack((Xw_good[:, 2], -Xw_good[:, 0], -Xw_good[:, 1])).astype(np.float32)
                self.shared_map.add_map_points(Xnav)

        self._update_status(
            tracking_ok=True,
            feature_count=feature_count,
            inlier_count=int(motion.recover_pose_count),
            sharpness=sharpness,
            contrast=contrast,
            mean_r=frame_stats["mean_r"],
            mean_g=frame_stats["mean_g"],
            mean_b=frame_stats["mean_b"],
            median_flow_px=motion.median_flow_px,
            frame_delta_mean=motion.frame_delta_mean,
            inlier_coverage=motion.inlier_coverage,
            gate_reason=gate_reason,
            match_count=motion.match_count,
            kept_match_count=motion.kept_match_count,
            essential_inlier_count=motion.essential_inlier_count,
            recover_pose_count=motion.recover_pose_count,
            requested_step=requested_step,
            applied_step=applied_step,
            translation_enabled=translation_enabled,
            rotation_deg=motion.rotation_deg,
        )

        self._last = cur
        self._publish_pose()
        self._annotate_debug_frames()
        return self.get_pose_estimate(filtered=True)

    # ---------------- Pose publishing ----------------

    def _publish_pose(self, *, update_filter: bool = True) -> None:
        raw_pose = self._planar_pose_from_Tcw(self._Tcw)
        self._raw_pose_latest = raw_pose

        if update_filter or self._pose_filt is None:
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

        if bool(self.cfg.publish_pose_to_shared_map):
            self.shared_map.set_pose(self.car_id, self.get_pose_estimate(filtered=True))

    def _apply_planar_motion(self, *, forward_step: float, yaw_delta: float) -> None:
        pose = self._planar_pose_from_Tcw(self._Tcw)
        next_pose = integrate_pose(
            pose,
            forward_step=forward_step,
            yaw_delta=yaw_delta,
        )
        self._Tcw = self._planar_pose_to_Tcw(next_pose)

    def _planar_pose_from_Tcw(self, Tcw: np.ndarray) -> Pose:
        Twc = self._invert_se3(Tcw)
        p = Twc[:3, 3]
        Rwc = Twc[:3, :3]

        s = float(self.cfg.forward_sign)
        x_raw = float(s * p[2])
        y_raw = float(s * (-p[0]))

        fwd = Rwc[:, 2]
        theta_raw = float(math.atan2(-s * fwd[0], s * fwd[2])) if (abs(fwd[0]) + abs(fwd[2]) > 1e-9) else 0.0
        return Pose(x=x_raw, y=y_raw, theta=theta_raw)

    @staticmethod
    def _planar_pose_to_Twc(pose: Pose) -> np.ndarray:
        theta = float(pose.theta)
        c = float(math.cos(theta))
        s = float(math.sin(theta))

        Twc = np.eye(4, dtype=np.float64)
        Twc[:3, :3] = np.array(
            [
                [c, 0.0, -s],
                [0.0, 1.0, 0.0],
                [s, 0.0, c],
            ],
            dtype=np.float64,
        )
        Twc[:3, 3] = np.array([-float(pose.y), 0.0, float(pose.x)], dtype=np.float64)
        return Twc

    def _planar_pose_to_Tcw(self, pose: Pose) -> np.ndarray:
        return self._invert_se3(self._planar_pose_to_Twc(pose))

    @staticmethod
    def _invert_se3(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        Tinv = np.eye(4, dtype=T.dtype)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return wrap_angle(a)

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
        frame_delta_mean: float,
        inlier_coverage: float,
        gate_reason: str = "tracking_ok",
        match_count: int = 0,
        kept_match_count: int = 0,
        essential_inlier_count: int = 0,
        recover_pose_count: int = 0,
        requested_step: float = 0.0,
        applied_step: float = 0.0,
        translation_enabled: bool = False,
        rotation_deg: float = 0.0,
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
            frame_delta_mean=float(frame_delta_mean),
            inlier_coverage=float(inlier_coverage),
            gate_reason=str(gate_reason),
            match_count=int(match_count),
            kept_match_count=int(kept_match_count),
            essential_inlier_count=int(essential_inlier_count),
            recover_pose_count=int(recover_pose_count),
            requested_step=float(requested_step),
            applied_step=float(applied_step),
            translation_enabled=bool(translation_enabled),
            rotation_deg=float(rotation_deg),
        )

    def _handle_weak_frame(
        self,
        *,
        cur: Optional[_Frame],
        tracking_reason: str,
        feature_count: int,
        sharpness: float,
        contrast: float,
        frame_stats: dict,
        requested_step: float,
        odom_yaw_delta: float = 0.0,
        median_flow_px: float = 0.0,
        frame_delta_mean: float = 0.0,
        inlier_coverage: float = 0.0,
        match_count: int = 0,
        kept_match_count: int = 0,
        essential_inlier_count: int = 0,
        recover_pose_count: int = 0,
        rotation_deg: float = 0.0,
    ) -> Optional[Pose]:
        hold_limit = max(0, int(self.cfg.max_weak_hold_frames))
        hold_tracking = bool(self._status.tracking_ok) and self._weak_hold_count < hold_limit
        odom_step = float(requested_step)
        odom_motion = abs(odom_step) > 1e-9 or abs(float(odom_yaw_delta)) > 1e-9

        if hold_tracking:
            self._weak_hold_count += 1
            tracking_ok = True
            if odom_motion:
                self._apply_planar_motion(forward_step=odom_step, yaw_delta=float(odom_yaw_delta))
                gate_reason = f"odom_hold_{tracking_reason}"
            else:
                gate_reason = f"hold_{tracking_reason}"
        else:
            self._weak_hold_count = 0
            tracking_ok = False
            gate_reason = tracking_reason
            if cur is not None:
                self._last = cur

        self._dbg_matches_bgr = None
        self._update_status(
            tracking_ok=tracking_ok,
            feature_count=feature_count,
            inlier_count=int(max(essential_inlier_count, recover_pose_count)),
            sharpness=sharpness,
            contrast=contrast,
            mean_r=frame_stats["mean_r"],
            mean_g=frame_stats["mean_g"],
            mean_b=frame_stats["mean_b"],
            median_flow_px=median_flow_px,
            frame_delta_mean=frame_delta_mean,
            inlier_coverage=inlier_coverage,
            gate_reason=gate_reason,
            match_count=match_count,
            kept_match_count=kept_match_count,
            essential_inlier_count=essential_inlier_count,
            recover_pose_count=recover_pose_count,
            requested_step=requested_step,
            applied_step=abs(odom_step) if hold_tracking and odom_motion else 0.0,
            translation_enabled=bool(hold_tracking and abs(odom_step) > 1e-9),
            rotation_deg=rotation_deg,
        )
        self._publish_pose(update_filter=False)
        self._annotate_debug_frames()
        return self.get_pose_estimate(filtered=True)

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

    @staticmethod
    def _frame_delta_mean(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
        if prev_gray.shape != cur_gray.shape:
            return float("inf")
        diff = cv2.absdiff(prev_gray, cur_gray)
        return float(np.mean(diff))

    def _to_gray(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        stats = {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0}

        if img.ndim == 2:
            gray = img
        else:
            color = img
            if color.dtype != np.uint8:
                color = np.clip(color, 0, 255).astype(np.uint8)
            color = np.ascontiguousarray(color)

            if self.cfg.input_color_order == "bgr":
                b = color[:, :, 0].astype(np.float32)
                g_ch = color[:, :, 1].astype(np.float32)
                r = color[:, :, 2].astype(np.float32)
            else:
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
                color = self._gray_world_balance(
                    color,
                    order=str(self.cfg.input_color_order).lower(),
                    max_gain=float(self.cfg.max_channel_gain),
                )

            if gray_source == "green":
                gray = color[:, :, 1]
            elif gray_source == "y_channel":
                if self.cfg.input_color_order == "bgr":
                    ycc = cv2.cvtColor(color, cv2.COLOR_BGR2YCrCb)
                else:
                    ycc = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)
                gray = ycc[:, :, 0]
            else:
                if self.cfg.input_color_order == "bgr":
                    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
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
        mean_all = (mr + mg + mb) / 3.0

        def gain(channel_mean: float) -> float:
            if channel_mean <= 1e-6:
                return 1.0
            return float(np.clip(mean_all / channel_mean, 0.6, max(1.0, max_gain)))

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

    # ---------------- Feature extraction + motion ----------------

    def _extract(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.cfg.extractor == "gftt_orb":
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

        kps, des = self.orb.detectAndCompute(gray, None)
        if des is None or kps is None or len(kps) == 0:
            return None, None

        if len(kps) > self.cfg.max_corners:
            order = np.argsort([-kp.response for kp in kps])[: self.cfg.max_corners]
            kps = [kps[i] for i in order]
            des = des[np.array(order, dtype=np.int64)]

        xy = np.array([kp.pt for kp in kps], dtype=np.float32)
        return xy, des

    def _estimate_motion(self, last: _Frame, cur: _Frame, frame_delta_mean: float) -> _MotionEstimate:
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

        match_count = len(idx_last)
        if match_count < 8:
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=match_count,
                reason="too_few_matches",
            )

        if len(idx_last) > self.cfg.keep_best:
            order = np.argsort(dists)[: self.cfg.keep_best]
            idx_last = [idx_last[i] for i in order]
            idx_cur = [idx_cur[i] for i in order]

        kept_match_count = len(idx_last)
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
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                reason="essential_failed",
            )

        inliers = inliers.reshape(-1).astype(bool)
        essential_inlier_count = int(inliers.sum())
        if essential_inlier_count < int(self.cfg.min_inliers):
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                essential_inlier_count=essential_inlier_count,
                reason="too_few_inliers",
            )

        pts_last_in = pts_last[inliers]
        pts_cur_in = pts_cur[inliers]

        pose_solution = self._recover_pose_from_essential(E, pts_last_in, pts_cur_in)
        if pose_solution is None:
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                essential_inlier_count=essential_inlier_count,
                reason="recover_pose_failed",
            )

        recover_pose_count, R, t, pose_mask = pose_solution
        recover_pose_count = int(recover_pose_count)

        if recover_pose_count <= 0 or pose_mask is None:
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                essential_inlier_count=essential_inlier_count,
                recover_pose_count=recover_pose_count,
                reason="recover_pose_failed",
            )

        pose_mask = pose_mask.reshape(-1).astype(bool)
        if int(pose_mask.sum()) <= 0:
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                essential_inlier_count=essential_inlier_count,
                recover_pose_count=recover_pose_count,
                reason="recover_pose_failed",
            )

        idx_last_in = np.array(idx_last, dtype=np.int64)[inliers][pose_mask]
        idx_cur_in = np.array(idx_cur, dtype=np.int64)[inliers][pose_mask]
        pts_last_pose = pts_last_in[pose_mask]
        pts_cur_pose = pts_cur_in[pose_mask]

        flow = np.linalg.norm(pts_cur_pose - pts_last_pose, axis=1)
        median_flow = float(np.median(flow)) if flow.size > 0 else 0.0
        inlier_coverage = self._point_coverage(pts_cur_pose, cur.img_gray.shape)
        rotation_deg = self._rotation_angle_deg(R)

        if rotation_deg > float(self.cfg.max_rotation_deg):
            return _MotionEstimate(
                False,
                np.array([]),
                np.array([]),
                np.eye(3),
                np.zeros((3, 1)),
                median_flow_px=median_flow,
                inlier_coverage=inlier_coverage,
                frame_delta_mean=frame_delta_mean,
                match_count=match_count,
                kept_match_count=kept_match_count,
                essential_inlier_count=essential_inlier_count,
                recover_pose_count=recover_pose_count,
                rotation_deg=rotation_deg,
                reason="rotation_too_large",
            )

        return _MotionEstimate(
            True,
            idx_cur_in,
            idx_last_in,
            R,
            t,
            median_flow_px=median_flow,
            inlier_coverage=inlier_coverage,
            frame_delta_mean=frame_delta_mean,
            match_count=match_count,
            kept_match_count=kept_match_count,
            essential_inlier_count=essential_inlier_count,
            recover_pose_count=recover_pose_count,
            rotation_deg=rotation_deg,
            reason="ok",
        )

    def _recover_pose_from_essential(
        self,
        E: np.ndarray,
        pts_last_in: np.ndarray,
        pts_cur_in: np.ndarray,
    ) -> Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        best_solution: Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = None
        best_score: Optional[Tuple[int, int]] = None

        for candidate in self._essential_candidates(E):
            try:
                pose_count, R, t, pose_mask = cv2.recoverPose(candidate, pts_last_in, pts_cur_in, self.K)
            except cv2.error:
                continue

            if pose_mask is None:
                continue

            pose_count = int(pose_count)
            pose_support = int(pose_mask.reshape(-1).astype(bool).sum())
            score = (pose_support, pose_count)
            if best_score is None or score > best_score:
                best_score = score
                best_solution = (pose_count, R, t, pose_mask)

        return best_solution

    @staticmethod
    def _essential_candidates(E: np.ndarray) -> List[np.ndarray]:
        E = np.asarray(E, dtype=np.float64)
        if E.size == 9:
            return [E.reshape(3, 3)]

        candidates: List[np.ndarray] = []
        if E.ndim != 2:
            return candidates

        if E.shape[1] == 3 and E.shape[0] % 3 == 0:
            for row in range(0, E.shape[0], 3):
                candidates.append(E[row:row + 3, :])
        elif E.shape[0] == 3 and E.shape[1] % 3 == 0:
            for col in range(0, E.shape[1], 3):
                candidates.append(E[:, col:col + 3])

        return [candidate for candidate in candidates if candidate.shape == (3, 3)]

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
    def _rotation_angle_deg(R: np.ndarray) -> float:
        trace_val = float(np.trace(R))
        cos_theta = float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0))
        return float(math.degrees(math.acos(cos_theta)))

    @staticmethod
    def _yaw_delta_from_relative_rotation(R: np.ndarray) -> float:
        return float(math.atan2(float(R[0, 2] - R[2, 0]), float(R[0, 0] + R[2, 2])))

    @staticmethod
    def _point_coverage(points_xy: np.ndarray, image_shape: Tuple[int, int]) -> float:
        if points_xy.size == 0:
            return 0.0
        h, w = int(image_shape[0]), int(image_shape[1])
        if h <= 0 or w <= 0:
            return 0.0

        min_xy = np.min(points_xy, axis=0)
        max_xy = np.max(points_xy, axis=0)

        box_w = max(1.0, float(max_xy[0] - min_xy[0]))
        box_h = max(1.0, float(max_xy[1] - min_xy[1]))
        return float((box_w * box_h) / float(w * h))

    def _annotate_debug_frames(self) -> None:
        lines = self._debug_overlay_lines()
        if self._dbg_keypoints_bgr is not None:
            self._draw_debug_overlay(self._dbg_keypoints_bgr, lines)
        if self._dbg_matches_bgr is not None:
            self._draw_debug_overlay(self._dbg_matches_bgr, lines)

    def _debug_overlay_lines(self) -> List[str]:
        s = self.get_status()
        return [
            f"gate={s.gate_reason} track={int(s.tracking_ok)} conf={s.confidence:.2f} blurry={int(s.blurry)}",
            f"feat={s.feature_count} match={s.match_count}/{s.kept_match_count} e_inl={s.essential_inlier_count} pose_inl={s.recover_pose_count}",
            f"flow={s.median_flow_px:.2f} frame_d={s.frame_delta_mean:.2f} cov={s.inlier_coverage:.3f} rot={s.rotation_deg:.1f}",
            f"step={s.requested_step:.2f}->{s.applied_step:.2f} trans={int(s.translation_enabled)}",
            f"raw=({s.raw_pose_x:.2f},{s.raw_pose_y:.2f},{s.raw_pose_theta:.2f}) filt=({s.filtered_pose_x:.2f},{s.filtered_pose_y:.2f},{s.filtered_pose_theta:.2f})",
        ]

    @staticmethod
    def _draw_debug_overlay(img: np.ndarray, lines: List[str]) -> None:
        y = 20
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

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
        return vis


class ConservativePoseEstimator:
    """
    Dead-reckoning-first pose estimator with conservative, occasional visual corrections.

    Dead reckoning always propagates the authoritative pose. Visual localization is only
    called after enough elapsed motion/time, then gated against the dead-reckoning pose.
    """

    def __init__(
        self,
        shared_map: SharedMap,
        *,
        car_id: int = 0,
        visual_localizer: Optional[MonocularVSLAM] = None,
        cfg: Optional[ConservativeCorrectionConfig] = None,
    ):
        self.shared_map = shared_map
        self.car_id = car_id
        self.visual_localizer = visual_localizer
        self.cfg = cfg or ConservativeCorrectionConfig()

        self._pending_translation = 0.0
        self._pending_yaw = 0.0
        self._cycles_since_correction = 0
        self._last_correction_time_s: Optional[float] = None
        self._last_result = ConservativeCorrectionResult(
            attempted=False,
            accepted=False,
            reason="startup",
            dead_reckoning_pose=self.get_pose(frame="world"),
        )

        if self.visual_localizer is not None:
            self.visual_localizer.cfg.publish_pose_to_shared_map = False
            self.visual_localizer.set_pose_estimate(self.get_pose(frame="world"))

    def get_pose(self, *, frame: Literal["world", "grid"] = "world") -> Pose:
        pose_world = self.shared_map.get_pose(self.car_id, frame="world")
        if pose_world is None:
            pose_world = Pose(0.0, 0.0, 0.0)
            self.shared_map.set_pose(self.car_id, pose_world)

        if frame == "world":
            return _copy_pose(pose_world)

        pose_grid = self.shared_map.get_pose(self.car_id, frame="grid")
        if pose_grid is not None:
            return pose_grid

        gx, gy = self.shared_map.world_to_grid_f(pose_world.x, pose_world.y)
        return Pose(x=float(gx), y=float(gy), theta=float(pose_world.theta))

    def get_last_result(self) -> ConservativeCorrectionResult:
        return ConservativeCorrectionResult(
            attempted=bool(self._last_result.attempted),
            accepted=bool(self._last_result.accepted),
            reason=str(self._last_result.reason),
            dead_reckoning_pose=_copy_pose(self._last_result.dead_reckoning_pose),
            obstacle_pose=None if self._last_result.obstacle_pose is None else _copy_pose(self._last_result.obstacle_pose),
            fused_pose=None if self._last_result.fused_pose is None else _copy_pose(self._last_result.fused_pose),
            position_error=float(self._last_result.position_error),
            heading_error_deg=float(self._last_result.heading_error_deg),
            visual_confidence=float(self._last_result.visual_confidence),
        )

    def get_debug_keypoints_frame(self) -> Optional[np.ndarray]:
        if self.visual_localizer is None:
            return None
        return self.visual_localizer.get_debug_keypoints_frame()

    def get_debug_matches_frame(self) -> Optional[np.ndarray]:
        if self.visual_localizer is None:
            return None
        return self.visual_localizer.get_debug_matches_frame()

    def propagate_dead_reckoning(self, *, forward_step: float, yaw_delta: float) -> Pose:
        pose_world = self.get_pose(frame="world")
        next_pose = integrate_pose(
            pose_world,
            forward_step=forward_step,
            yaw_delta=yaw_delta,
        )
        self.shared_map.set_pose(self.car_id, next_pose)

        self._pending_translation += float(forward_step)
        self._pending_yaw += float(yaw_delta)
        self._cycles_since_correction += 1
        return next_pose

    def should_run_obstacle_detection(self, *, now_s: Optional[float] = None, force: bool = False) -> bool:
        if self.visual_localizer is None:
            return False
        if force:
            return True
        if self._cycles_since_correction < max(1, int(self.cfg.min_cycles_between_corrections)):
            return False

        enough_motion = (
            abs(float(self._pending_translation)) >= float(self.cfg.min_translation_between_corrections)
            or abs(math.degrees(float(self._pending_yaw))) >= float(self.cfg.min_heading_change_between_corrections_deg)
        )
        if not enough_motion:
            return False

        if now_s is None or self._last_correction_time_s is None:
            return True
        return (float(now_s) - float(self._last_correction_time_s)) >= float(self.cfg.min_seconds_between_corrections)

    def maybe_correct_from_obstacle_detection(
        self,
        frame_rgb_or_bgr: Optional[np.ndarray] = None,
        *,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        now_s: Optional[float] = None,
        force: bool = False,
    ) -> ConservativeCorrectionResult:
        dead_reckoning_pose = self.get_pose(frame="world")

        if self.visual_localizer is None:
            self._last_result = ConservativeCorrectionResult(
                attempted=False,
                accepted=False,
                reason="visual_disabled",
                dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
                fused_pose=_copy_pose(dead_reckoning_pose),
            )
            return self.get_last_result()

        if not self.should_run_obstacle_detection(now_s=now_s, force=force):
            self._last_result = ConservativeCorrectionResult(
                attempted=False,
                accepted=False,
                reason="not_needed",
                dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
                fused_pose=_copy_pose(dead_reckoning_pose),
            )
            return self.get_last_result()

        if frame_rgb_or_bgr is None and frame_provider is not None:
            try:
                frame_rgb_or_bgr = frame_provider()
            except Exception:
                self._last_result = ConservativeCorrectionResult(
                    attempted=True,
                    accepted=False,
                    reason="frame_provider_error",
                    dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
                    fused_pose=_copy_pose(dead_reckoning_pose),
                )
                return self.get_last_result()

        if frame_rgb_or_bgr is None:
            self._last_result = ConservativeCorrectionResult(
                attempted=True,
                accepted=False,
                reason="no_frame",
                dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
                fused_pose=_copy_pose(dead_reckoning_pose),
            )
            return self.get_last_result()

        pending_translation = float(self._pending_translation)
        pending_yaw = float(self._pending_yaw)
        self._pending_translation = 0.0
        self._pending_yaw = 0.0
        self._cycles_since_correction = 0
        self._last_correction_time_s = None if now_s is None else float(now_s)

        try:
            obstacle_pose = self.visual_localizer.tick(
                frame_rgb_or_bgr,
                translation_step=pending_translation,
                odom_yaw_delta=pending_yaw,
            )
            status = self.visual_localizer.get_status()
        except Exception:
            fused_pose = _copy_pose(dead_reckoning_pose)
            self.shared_map.set_pose(self.car_id, fused_pose)
            self.visual_localizer.set_pose_estimate(fused_pose, reset_filter=True, sync_last_frame=True)
            self._last_result = ConservativeCorrectionResult(
                attempted=True,
                accepted=False,
                reason="visual_exception",
                dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
                fused_pose=fused_pose,
            )
            return self.get_last_result()

        accepted = False
        reason = "tracking_not_ok" if obstacle_pose is None else "accepted"
        position_error = 0.0
        heading_error_deg = 0.0
        fused_pose = _copy_pose(dead_reckoning_pose)

        if obstacle_pose is None:
            obstacle_pose_copy = None
        else:
            obstacle_pose_copy = _copy_pose(obstacle_pose)
            position_error = _pose_distance(dead_reckoning_pose, obstacle_pose_copy)
            heading_error_deg = math.degrees(
                _heading_difference(dead_reckoning_pose.theta, obstacle_pose_copy.theta)
            )

            if not bool(status.tracking_ok):
                reason = f"tracking_{status.gate_reason}"
            elif float(status.confidence) < float(self.cfg.min_visual_confidence):
                reason = "low_confidence"
            elif position_error > float(self.cfg.position_agreement_threshold):
                reason = "position_mismatch"
            elif heading_error_deg > float(self.cfg.heading_agreement_threshold_deg):
                reason = "heading_mismatch"
            else:
                accepted = True
                fused_pose = _blend_pose(dead_reckoning_pose, obstacle_pose_copy, float(self.cfg.correction_alpha))

        self.shared_map.set_pose(self.car_id, fused_pose)
        self.visual_localizer.set_pose_estimate(fused_pose, reset_filter=True, sync_last_frame=True)

        self._last_result = ConservativeCorrectionResult(
            attempted=True,
            accepted=accepted,
            reason=reason,
            dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
            obstacle_pose=obstacle_pose_copy,
            fused_pose=_copy_pose(fused_pose),
            position_error=float(position_error),
            heading_error_deg=float(heading_error_deg),
            visual_confidence=float(status.confidence),
        )
        return self.get_last_result()
