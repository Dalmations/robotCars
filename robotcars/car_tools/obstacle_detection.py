# Conservative monocular VO helpers.
# Keeps dead reckoning primary and uses sparse feature tracking plus essential-matrix yaw as a conservative correction source.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Literal

import cv2
import numpy as np

from model import Pose
from coordination.shared_map import SharedMap


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def integrate_pose(
    pose: Pose,
    *,
    forward_step: float,
    yaw_delta: float,
) -> Pose:
    step = float(forward_step)
    dtheta = float(yaw_delta)
    theta_mid = float(pose.theta) + 0.5 * dtheta
    return Pose(
        x=float(pose.x + step * math.cos(theta_mid)),
        y=float(pose.y + step * math.sin(theta_mid)),
        theta=float(wrap_angle(float(pose.theta) + dtheta)),
    )


def _copy_pose(pose: Pose) -> Pose:
    return Pose(x=float(pose.x), y=float(pose.y), theta=float(pose.theta))


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


def _blend_heading_pose(base_pose: Pose, correction_pose: Pose, alpha: float) -> Pose:
    blend = float(np.clip(float(alpha), 0.0, 1.0))
    if blend <= 0.0:
        return _copy_pose(base_pose)
    if blend >= 1.0:
        return Pose(
            x=float(base_pose.x),
            y=float(base_pose.y),
            theta=float(correction_pose.theta),
        )

    return Pose(
        x=float(base_pose.x),
        y=float(base_pose.y),
        theta=float(
            wrap_angle(
                float(base_pose.theta)
                + blend * wrap_angle(float(correction_pose.theta) - float(base_pose.theta))
            )
        ),
    )


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[np.ndarray] = None

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def dist(self) -> Optional[np.ndarray]:
        if self.dist_coeffs is None:
            return None
        arr = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        return arr


@dataclass
class VslamConfig:
    # Debug
    debug_draw_frame: bool = True

    # Pose frame
    forward_sign: float = 1.0
    pose_ema_alpha: float = 0.2

    # Conservative VO
    vo_max_corners: int = 220
    vo_quality_level: float = 0.01
    vo_min_distance: float = 8.0
    vo_block_size: int = 7
    vo_lk_window_size: int = 21
    vo_lk_max_level: int = 3
    vo_min_tracked_features: int = 30
    vo_min_pose_inliers: int = 16
    vo_min_median_flow_px: float = 0.75
    vo_ransac_threshold_px: float = 1.5
    vo_max_rotation_deg: float = 35.0

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
    pose_valid: bool = False
    sharpness: float = 0.0
    contrast: float = 0.0
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    confidence: float = 0.0
    high_confidence: bool = False
    blurry: bool = True

    gate_reason: str = "startup"

    feature_count: int = 0
    inlier_count: int = 0
    match_count: int = 0
    kept_match_count: int = 0
    essential_inlier_count: int = 0
    recover_pose_count: int = 0

    median_flow_px: float = 0.0
    frame_delta_mean: float = 0.0
    inlier_coverage: float = 0.0
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
    min_cycles_between_corrections: int = 4
    min_seconds_between_corrections: float = 0.4
    min_translation_between_corrections: float = 0.75
    min_heading_change_between_corrections_deg: float = 8.0

    heading_agreement_threshold_deg: float = 12.0

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
    heading_error_deg: float = 0.0
    visual_confidence: float = 0.0


@dataclass
class _FrameState:
    gray: np.ndarray
    pts_xy: np.ndarray


class MonocularVSLAM:
    """
    Conservative monocular VO localizer.

    The caller seeds the pose from dead reckoning. Each visual update tracks
    sparse features between frames and only applies a yaw correction when the
    recovered relative pose looks reliable enough to trust.
    """

    def __init__(
        self,
        intr: CameraIntrinsics,
        cfg: Optional[VslamConfig] = None,
    ):
        self.cfg = cfg or VslamConfig()
        self.K = intr.K()
        self.dist_coeffs = intr.dist()

        # Runtime state is populated incrementally as frames are processed.
        self._raw_pose_latest: Optional[Pose] = None
        self._pose_filt: Optional[Pose] = None
        self._last_frame: Optional[_FrameState] = None
        self._status = VslamStatus()

        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

        self._Tcw = np.eye(4, dtype=np.float64)

        self._dbg_frame_bgr: Optional[np.ndarray] = None

    # ---------------- Public diagnostics ----------------

    def get_status(self) -> VslamStatus:
        return VslamStatus(**vars(self._status))

    def get_pose_estimate(self, *, filtered: bool = True) -> Pose:
        pose = self._pose_filt if filtered and self._pose_filt is not None else self._raw_pose_latest
        return _copy_pose(pose)

    def set_pose_estimate(
        self,
        pose: Pose,
        *,
        reset_filter: bool = True,
    ) -> Pose:
        pose_copy = _copy_pose(pose)
        self._Tcw = self._planar_pose_to_Tcw(pose_copy)
        self._raw_pose_latest = pose_copy

        if reset_filter or self._pose_filt is None:
            self._pose_filt = _copy_pose(pose_copy)
        else:
            self._pose_filt = _blend_pose(self._pose_filt, pose_copy, float(self.cfg.pose_ema_alpha))

        return self.get_pose_estimate(filtered=True)

    def tick(
        self,
        frame_rgb_or_bgr: np.ndarray,
        *,
        pose_prior: Optional[Pose] = None,
        requested_step: float = 0.0,
        odom_yaw_delta: float = 0.0,
    ) -> Optional[Pose]:
        gray = self._to_gray(frame_rgb_or_bgr)
        mean_r, mean_g, mean_b = self._frame_color_stats(frame_rgb_or_bgr)
        sharpness = self._frame_sharpness(gray)
        contrast = self._frame_contrast(gray)

        if self.cfg.debug_draw_frame:
            self._dbg_frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            self._dbg_frame_bgr = None

        if self._last_frame is None:
            feature_count = self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                gate_reason="bootstrap",
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        last = self._last_frame
        prev_pts, cur_pts = self._track_features(last.gray, gray, last.pts_xy)
        feature_count = int(prev_pts.shape[0])
        frame_delta_mean = self._frame_delta_mean(last.gray, gray)

        if feature_count < int(self.cfg.vo_min_tracked_features):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                frame_delta_mean=frame_delta_mean,
                gate_reason="too_few_tracks",
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        flow_px = np.linalg.norm(cur_pts - prev_pts, axis=1)
        median_flow_px = float(np.median(flow_px)) if flow_px.size else 0.0
        if median_flow_px < float(self.cfg.vo_min_median_flow_px):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="low_flow",
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        E, essential_mask = cv2.findEssentialMat(
            prev_pts,
            cur_pts,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=float(self.cfg.vo_ransac_threshold_px),
        )
        if E is None:
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="essential_failed",
                match_count=feature_count,
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        E = np.asarray(E, dtype=np.float64)
        if E.ndim == 3:
            E = E[0]
        elif E.ndim == 2 and E.shape[0] > 3:
            E = E[:3, :3]
        if E.shape != (3, 3):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=0,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="essential_invalid",
                match_count=feature_count,
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        essential_keep = (
            essential_mask.reshape(-1).astype(bool)
            if essential_mask is not None
            else np.ones(prev_pts.shape[0], dtype=bool)
        )
        prev_in = prev_pts[essential_keep]
        cur_in = cur_pts[essential_keep]
        essential_inlier_count = int(prev_in.shape[0])
        if essential_inlier_count < int(self.cfg.vo_min_pose_inliers):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=essential_inlier_count,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="too_few_essential_inliers",
                match_count=feature_count,
                kept_match_count=essential_inlier_count,
                essential_inlier_count=essential_inlier_count,
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        recover_count, R, _t, pose_mask = cv2.recoverPose(E, prev_in, cur_in, self.K)
        recover_pose_count = int(recover_count)
        if recover_pose_count < int(self.cfg.vo_min_pose_inliers):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=recover_pose_count,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="too_few_pose_inliers",
                match_count=feature_count,
                kept_match_count=essential_inlier_count,
                essential_inlier_count=essential_inlier_count,
                recover_pose_count=recover_pose_count,
                requested_step=requested_step,
            )
            self._annotate_debug_frames()
            return None

        pose_keep = (
            pose_mask.reshape(-1).astype(bool)
            if pose_mask is not None
            else np.ones(prev_in.shape[0], dtype=bool)
        )
        prev_pose_pts = prev_in[pose_keep]
        cur_pose_pts = cur_in[pose_keep]
        rotation_deg = math.degrees(self._yaw_delta_from_rotation(R))
        if not np.isfinite(rotation_deg) or abs(rotation_deg) > float(self.cfg.vo_max_rotation_deg):
            self._refresh_last_frame(gray)
            self._update_status(
                tracking_ok=False,
                feature_count=feature_count,
                inlier_count=recover_pose_count,
                sharpness=sharpness,
                contrast=contrast,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                median_flow_px=median_flow_px,
                frame_delta_mean=frame_delta_mean,
                gate_reason="rotation_outlier",
                match_count=feature_count,
                kept_match_count=essential_inlier_count,
                essential_inlier_count=essential_inlier_count,
                recover_pose_count=recover_pose_count,
                requested_step=requested_step,
                rotation_deg=rotation_deg,
            )
            self._annotate_debug_frames()
            return None

        base_pose = pose_prior
        if base_pose is None:
            base_pose = self._pose_filt or self._raw_pose_latest or Pose(0.0, 0.0, 0.0)

        # Dead reckoning owns translation. VO only contributes a yaw residual
        # relative to the odometry turn over the same correction window.
        visual_yaw_delta = math.radians(rotation_deg)
        yaw_residual = wrap_angle(visual_yaw_delta - float(odom_yaw_delta))
        next_pose = Pose(
            x=float(base_pose.x),
            y=float(base_pose.y),
            theta=float(wrap_angle(float(base_pose.theta) + yaw_residual)),
        )
        pose = _copy_pose(next_pose)
        self._Tcw = self._planar_pose_to_Tcw(pose)
        self._raw_pose_latest = _copy_pose(pose)
        self._pose_filt = _copy_pose(pose)

        self._refresh_last_frame(gray)
        if self._dbg_frame_bgr is not None:
            self._draw_feature_tracks(self._dbg_frame_bgr, prev_pose_pts, cur_pose_pts)

        self._update_status(
            tracking_ok=True,
            feature_count=feature_count,
            inlier_count=recover_pose_count,
            sharpness=sharpness,
            contrast=contrast,
            mean_r=mean_r,
            mean_g=mean_g,
            mean_b=mean_b,
            median_flow_px=median_flow_px,
            frame_delta_mean=frame_delta_mean,
            inlier_coverage=self._point_coverage(cur_pose_pts, gray.shape),
            gate_reason="vo_tracking",
            match_count=feature_count,
            kept_match_count=essential_inlier_count,
            essential_inlier_count=essential_inlier_count,
            recover_pose_count=recover_pose_count,
            requested_step=requested_step,
            applied_step=abs(float(requested_step)),
            translation_enabled=bool(abs(float(requested_step)) > 1e-9),
            rotation_deg=rotation_deg,
        )
        self._annotate_debug_frames()
        return pose

    def _planar_pose_from_Tcw(self, Tcw: np.ndarray) -> Pose:
        Twc = self._invert_se3(Tcw)
        p = Twc[:3, 3]
        Rwc = Twc[:3, :3]

        s = float(self.cfg.forward_sign)
        # Project to floor.
        x_raw = float(s * p[2])
        y_raw = float(s * (-p[0]))

        # Camera forward heading.
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
        median_flow_px: float = 0.0,
        frame_delta_mean: float = 0.0,
        inlier_coverage: float = 0.0,
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
        flow_ref = max(1e-6, float(self.cfg.vo_min_median_flow_px))

        feat_score = float(np.clip(float(feature_count) / f_ref, 0.0, 1.0))
        inlier_score = float(np.clip(float(inlier_count) / i_ref, 0.0, 1.0))
        sharp_score = float(np.clip(float(sharpness) / s_ref, 0.0, 1.0))
        contrast_score = float(np.clip(float(contrast) / c_ref, 0.0, 1.0))
        flow_score = float(np.clip(float(median_flow_px) / flow_ref, 0.0, 1.0))

        raw_conf = (
            0.30 * feat_score
            + 0.35 * inlier_score
            + 0.15 * sharp_score
            + 0.10 * contrast_score
            + 0.10 * flow_score
        )
        if not tracking_ok:
            raw_conf *= 0.20

        alpha = float(np.clip(self.cfg.confidence_ema_alpha, 0.0, 1.0))
        conf = raw_conf if alpha <= 0.0 else ((1.0 - alpha) * float(self._status.confidence) + alpha * raw_conf)

        raw_pose = self._raw_pose_latest if self._raw_pose_latest is not None else self._planar_pose_from_Tcw(self._Tcw)
        filt_pose = self._pose_filt if self._pose_filt is not None else raw_pose

        self._status = VslamStatus(
            tracking_ok=bool(tracking_ok),
            pose_valid=bool(self._raw_pose_latest is not None or self._pose_filt is not None),
            sharpness=float(sharpness),
            contrast=float(contrast),
            mean_r=float(mean_r),
            mean_g=float(mean_g),
            mean_b=float(mean_b),
            feature_count=int(feature_count),
            inlier_count=int(inlier_count),
            confidence=float(conf),
            high_confidence=bool(tracking_ok and conf >= float(self.cfg.confidence_good_threshold)),
            blurry=bool(sharpness < float(self.cfg.confidence_min_sharpness)),
            gate_reason=str(gate_reason),
            match_count=int(match_count),
            kept_match_count=int(kept_match_count),
            essential_inlier_count=int(essential_inlier_count),
            recover_pose_count=int(recover_pose_count),
            median_flow_px=float(median_flow_px),
            frame_delta_mean=float(frame_delta_mean),
            inlier_coverage=float(inlier_coverage),
            requested_step=float(requested_step),
            applied_step=float(applied_step),
            translation_enabled=bool(translation_enabled),
            rotation_deg=float(rotation_deg),
            raw_pose_x=float(raw_pose.x),
            raw_pose_y=float(raw_pose.y),
            raw_pose_theta=float(raw_pose.theta),
            filtered_pose_x=float(filt_pose.x),
            filtered_pose_y=float(filt_pose.y),
            filtered_pose_theta=float(filt_pose.theta),
        )

    def _refresh_last_frame(self, gray: np.ndarray) -> int:
        pts_xy = self._extract_features(gray)
        self._last_frame = _FrameState(
            gray=np.ascontiguousarray(gray),
            pts_xy=pts_xy,
        )
        return int(pts_xy.shape[0])

    def _extract_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.cfg.vo_max_corners),
            qualityLevel=float(self.cfg.vo_quality_level),
            minDistance=float(self.cfg.vo_min_distance),
            blockSize=int(self.cfg.vo_block_size),
        )
        if pts is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    def _track_features(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if prev_pts.size == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        win = max(3, int(self.cfg.vo_lk_window_size))
        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts.reshape(-1, 1, 2),
            None,
            winSize=(win, win),
            maxLevel=int(self.cfg.vo_lk_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        keep = status.reshape(-1).astype(bool)
        prev_kept = prev_pts[keep]
        cur_kept = np.asarray(next_pts, dtype=np.float32).reshape(-1, 2)[keep]
        return prev_kept.astype(np.float32), cur_kept.astype(np.float32)

    @staticmethod
    def _frame_delta_mean(prev_gray: np.ndarray, gray: np.ndarray) -> float:
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.mean(diff))

    @staticmethod
    def _yaw_delta_from_rotation(R: np.ndarray) -> float:
        return wrap_angle(float(math.atan2(-float(R[0, 2]), float(R[2, 2]))))

    @staticmethod
    def _point_coverage(pts_xy: np.ndarray, image_shape: Tuple[int, ...]) -> float:
        if pts_xy.shape[0] < 2:
            return 0.0

        min_xy = np.min(pts_xy, axis=0)
        max_xy = np.max(pts_xy, axis=0)
        width = max(0.0, float(max_xy[0] - min_xy[0]))
        height = max(0.0, float(max_xy[1] - min_xy[1]))
        image_area = max(1.0, float(image_shape[0] * image_shape[1]))
        return float(np.clip((width * height) / image_area, 0.0, 1.0))

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

    def _frame_color_stats(self, img: np.ndarray) -> Tuple[float, float, float]:
        if img.ndim == 2:
            mean_gray = float(np.mean(img)) if img.size else 0.0
            return mean_gray, mean_gray, mean_gray

        color = np.ascontiguousarray(img)
        if color.dtype != np.uint8:
            color = np.clip(color, 0, 255).astype(np.uint8)

        if str(self.cfg.input_color_order).lower() == "bgr":
            mean_b = float(np.mean(color[:, :, 0]))
            mean_g = float(np.mean(color[:, :, 1]))
            mean_r = float(np.mean(color[:, :, 2]))
        else:
            mean_r = float(np.mean(color[:, :, 0]))
            mean_g = float(np.mean(color[:, :, 1]))
            mean_b = float(np.mean(color[:, :, 2]))
        return mean_r, mean_g, mean_b

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            gray = img
        else:
            color = np.ascontiguousarray(img)
            if color.dtype != np.uint8:
                color = np.clip(color, 0, 255).astype(np.uint8)
            color = np.ascontiguousarray(color)

            gray_source = str(self.cfg.gray_source).lower()
            if self.cfg.auto_white_balance and gray_source != "green":
                color = self._gray_world_balance(
                    color,
                    order=str(self.cfg.input_color_order).lower(),
                    max_gain=float(self.cfg.max_channel_gain),
                )

            gray_source = str(self.cfg.gray_source).lower()
            if gray_source == "green":
                gray = color[:, :, 1]
            elif gray_source == "y_channel":
                code = (
                    cv2.COLOR_BGR2YCrCb
                    if self.cfg.input_color_order == "bgr"
                    else cv2.COLOR_RGB2YCrCb
                )
                gray = cv2.cvtColor(color, code)[:, :, 0]
            else:
                code = (
                    cv2.COLOR_BGR2GRAY
                    if self.cfg.input_color_order == "bgr"
                    else cv2.COLOR_RGB2GRAY
                )
                gray = cv2.cvtColor(color, code)

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        if self.cfg.gray_flat_field_correction:
            gray = self._flat_field_correct(
                gray,
                sigma=float(self.cfg.gray_flat_field_sigma),
                strength=float(self.cfg.gray_flat_field_strength),
            )

        if self.cfg.gray_use_clahe:
            tile = max(1, int(self.cfg.gray_clahe_tile_size))
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=float(self.cfg.gray_clahe_clip_limit),
                    tileGridSize=(tile, tile),
                )
            gray = self._clahe.apply(gray)

        if abs(float(self.cfg.gray_gamma) - 1.0) > 1e-6:
            gray = cv2.LUT(gray, self._gamma_lut_for(float(self.cfg.gray_gamma)))

        amount = float(self.cfg.gray_unsharp_amount)
        if amount > 1e-6:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
            gray = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)

        return np.ascontiguousarray(gray)

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

    # ---------------- Debug rendering ----------------

    def _annotate_debug_frames(self) -> None:
        lines = self._debug_overlay_lines()
        if self._dbg_frame_bgr is not None:
            self._draw_debug_overlay(self._dbg_frame_bgr, lines)

    def _debug_overlay_lines(self) -> List[str]:
        s = self.get_status()
        filt_pose = self._pose_filt if self._pose_filt is not None else self._planar_pose_from_Tcw(self._Tcw)
        return [
            f"gate={s.gate_reason} track={int(s.tracking_ok)} conf={s.confidence:.2f}",
            f"feat={s.feature_count} inliers={s.inlier_count} kept={s.kept_match_count} recov={s.recover_pose_count}",
            f"flow={s.median_flow_px:.2f} delta={s.frame_delta_mean:.1f} cov={s.inlier_coverage:.2f}",
            f"sharp={s.sharpness:.1f} contrast={s.contrast:.1f} rot={s.rotation_deg:.1f}",
            f"pose=({filt_pose.x:.2f},{filt_pose.y:.2f},{filt_pose.theta:.2f})",
        ]

    @staticmethod
    def _draw_feature_tracks(img: np.ndarray, prev_pts: np.ndarray, cur_pts: np.ndarray) -> None:
        count = min(int(prev_pts.shape[0]), int(cur_pts.shape[0]))
        for i in range(count):
            x0, y0 = prev_pts[i]
            x1, y1 = cur_pts[i]
            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))
            cv2.line(img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, p1, 2, (0, 200, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_debug_overlay(img: np.ndarray, lines: List[str]) -> None:
        y = 20
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18


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
        visual_localizer: Optional[MonocularVSLAM] = None,
        cfg: Optional[ConservativeCorrectionConfig] = None,
    ):
        self.shared_map = shared_map
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
            self.visual_localizer.set_pose_estimate(self.get_pose(frame="world"))

    def get_pose(self, *, frame: Literal["world", "grid"] = "world") -> Pose:
        pose_world = self.shared_map.get_pose(frame="world")
        if pose_world is None:
            pose_world = Pose(0.0, 0.0, 0.0)
            self.shared_map.set_pose(pose_world)

        if frame == "world":
            return _copy_pose(pose_world)

        pose_grid = self.shared_map.get_pose(frame="grid")
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
            heading_error_deg=float(self._last_result.heading_error_deg),
            visual_confidence=float(self._last_result.visual_confidence),
        )

    def propagate_dead_reckoning(self, *, forward_step: float, yaw_delta: float) -> Pose:
        pose_world = self.get_pose(frame="world")
        next_pose = integrate_pose(
            pose_world,
            forward_step=forward_step,
            yaw_delta=yaw_delta,
        )
        self.shared_map.set_pose(next_pose)

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
        pending_cycles = int(self._cycles_since_correction)
        pending_last_correction_time = self._last_correction_time_s

        obstacle_pose = self.visual_localizer.tick(
            frame_rgb_or_bgr,
            pose_prior=dead_reckoning_pose,
            requested_step=pending_translation,
            odom_yaw_delta=pending_yaw,
        )
        status = self.visual_localizer.get_status()

        accepted = False
        reason = "tracking_not_ok" if obstacle_pose is None else "accepted"
        heading_error_deg = 0.0
        fused_pose = _copy_pose(dead_reckoning_pose)

        if obstacle_pose is None:
            obstacle_pose_copy = None
            self._pending_translation = pending_translation
            self._pending_yaw = pending_yaw
            self._cycles_since_correction = pending_cycles
            self._last_correction_time_s = pending_last_correction_time
        else:
            obstacle_pose_copy = _copy_pose(obstacle_pose)
            heading_error_deg = math.degrees(
                _heading_difference(dead_reckoning_pose.theta, obstacle_pose_copy.theta)
            )

            # Heading-only VO corrections share dead-reckoned x/y with the
            # motion prior, so acceptance is based on tracking quality,
            # confidence, and heading agreement.
            if not bool(status.tracking_ok):
                reason = f"tracking_{status.gate_reason}"
            elif float(status.confidence) < float(self.cfg.min_visual_confidence):
                reason = "low_confidence"
            elif heading_error_deg > float(self.cfg.heading_agreement_threshold_deg):
                reason = "heading_mismatch"
            else:
                accepted = True
                fused_pose = _blend_heading_pose(
                    dead_reckoning_pose,
                    obstacle_pose_copy,
                    float(self.cfg.correction_alpha),
                )

            if accepted:
                self._pending_translation = 0.0
                self._pending_yaw = 0.0
                self._cycles_since_correction = 0
                self._last_correction_time_s = None if now_s is None else float(now_s)
            else:
                self._pending_translation = pending_translation
                self._pending_yaw = pending_yaw
                self._cycles_since_correction = pending_cycles
                self._last_correction_time_s = pending_last_correction_time

        self.shared_map.set_pose(fused_pose)
        self.visual_localizer.set_pose_estimate(fused_pose, reset_filter=True)

        self._last_result = ConservativeCorrectionResult(
            attempted=True,
            accepted=accepted,
            reason=reason,
            dead_reckoning_pose=_copy_pose(dead_reckoning_pose),
            obstacle_pose=obstacle_pose_copy,
            fused_pose=_copy_pose(fused_pose),
            heading_error_deg=float(heading_error_deg),
            visual_confidence=float(status.confidence),
        )
        return self.get_last_result()
