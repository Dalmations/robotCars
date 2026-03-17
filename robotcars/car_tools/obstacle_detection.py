# Conservative monocular VO helpers for intermittent yaw correction.
# Keeps dead reckoning primary, uses OpenCV sparse tracking plus essential-matrix rotation as a gated assist.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

from model import Pose

def _wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad

def _point_coverage(points_xy: np.ndarray, image_shape: tuple[int, int]) -> float:
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


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace_val = float(np.trace(R))
    cos_theta = float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_theta)))


def _yaw_from_relative_rotation(R: np.ndarray) -> float:
    return float(math.atan2(float(R[0, 2] - R[2, 0]), float(R[0, 0] + R[2, 2])))


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[tuple[float, ...]] = None

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
        arr = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        return None if arr.size == 0 else arr


@dataclass
class ImagePreprocessConfig:
    input_color_order: Literal["rgb", "bgr"] = "rgb"
    gray_source: Literal["luma", "green", "y_channel"] = "green"
    undistort: bool = True

    auto_white_balance: bool = True
    max_channel_gain: float = 1.8

    flat_field_correction: bool = True
    flat_field_sigma: float = 28.0
    flat_field_strength: float = 0.85

    use_clahe: bool = True
    clahe_clip_limit: float = 2.2
    clahe_tile_size: int = 8

    gamma: float = 1.0
    unsharp_amount: float = 0.0


@dataclass
class VisualOdometryConfig:
    preprocess: ImagePreprocessConfig = field(default_factory=ImagePreprocessConfig)

    attempt_interval_s: float = 2.0
    min_baseline_distance: float = 0.50
    min_baseline_yaw_deg: float = 6.0

    max_corners: int = 400
    quality_level: float = 0.01
    min_distance_px: int = 10
    block_size: int = 7
    feature_mask_top_fraction: float = 0.0

    min_keyframe_features: int = 120
    min_current_features: int = 120
    min_tracked_points: int = 60

    lk_win_size: int = 21
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    lk_epsilon: float = 0.01
    lk_max_error_px: float = 20.0
    use_forward_backward_check: bool = True
    max_forward_backward_error_px: float = 1.5

    ransac_prob: float = 0.999
    ransac_threshold_px: float = 1.0
    min_inliers: int = 35
    min_inlier_ratio: float = 0.35
    min_median_flow_px: float = 3.0
    min_inlier_coverage: float = 0.08
    max_median_epipolar_error_px: float = 1.5
    max_p90_epipolar_error_px: float = 3.0
    max_rotation_deg: float = 25.0
    max_yaw_residual_deg: float = 12.0

    min_sharpness: float = 45.0
    min_contrast: float = 20.0

    confidence_good_threshold: float = 0.70
    yaw_correction_gain: float = 0.60
    max_applied_yaw_correction_deg: float = 8.0

    cm_per_world_unit: float = 1.0
    ultrasonic_max_range_cm: float = 120.0
    ultrasonic_forward_tolerance_cm: float = 15.0
    ultrasonic_lateral_tolerance_cm: float = 12.0
    ultrasonic_close_range_cm: float = 40.0
    ultrasonic_close_yaw_limit_deg: float = 5.0


@dataclass
class VisualYawUpdate:
    attempted: bool = False
    accepted: bool = False
    keyframe_replaced: bool = False
    reason: str = "startup"

    confidence: float = 0.0
    high_confidence: bool = False

    keyframe_age_s: float = 0.0
    baseline_distance: float = 0.0
    baseline_yaw_deg: float = 0.0

    sharpness: float = 0.0
    contrast: float = 0.0
    feature_count: int = 0
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    median_flow_px: float = 0.0
    inlier_coverage: float = 0.0
    median_epipolar_error_px: float = 0.0
    p90_epipolar_error_px: float = 0.0
    rotation_deg: float = 0.0

    visual_yaw_delta_deg: float = 0.0
    dead_reckoned_yaw_delta_deg: float = 0.0
    yaw_residual_deg: float = 0.0
    applied_yaw_correction_deg: float = 0.0

    corrected_pose: Optional[Pose] = None
    ultrasonic_distance_cm: Optional[float] = None
    ultrasonic_rejected: bool = False


@dataclass
class _Keyframe:
    gray: np.ndarray
    points_xy: np.ndarray
    pose: Pose
    timestamp_s: float


@dataclass
class _MotionEstimate:
    ok: bool
    reason: str
    tracked_cur: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    median_flow_px: float = 0.0
    inlier_coverage: float = 0.0
    median_epipolar_error_px: float = 0.0
    p90_epipolar_error_px: float = 0.0
    rotation_deg: float = 0.0


@dataclass
class _UpdateContext:
    sharpness: float = 0.0
    contrast: float = 0.0
    ultrasonic_distance_cm: Optional[float] = None
    keyframe_age_s: float = 0.0
    baseline_distance: float = 0.0
    baseline_yaw_deg: float = 0.0
    feature_count: int = 0
    confidence: float = 0.0
    high_confidence: bool = False
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    median_flow_px: float = 0.0
    inlier_coverage: float = 0.0
    median_epipolar_error_px: float = 0.0
    p90_epipolar_error_px: float = 0.0
    rotation_deg: float = 0.0


class ConservativeVisualOdometry:
    """
    Sparse-feature monocular VO that only proposes bounded yaw corrections.

    Dead reckoning remains the primary pose source. This helper keeps a visual
    keyframe, attempts a relative-motion solve at a low rate, and returns a
    correction only when the update passes strict visual and ultrasonic gates.
    """

    def __init__(self, intrinsics: CameraIntrinsics, cfg: Optional[VisualOdometryConfig] = None):
        self.intr = intrinsics
        self.cfg = cfg or VisualOdometryConfig()
        self.K = intrinsics.K()

        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

        self._keyframe: Optional[_Keyframe] = None
        self._last_attempt_time_s: Optional[float] = None

    def reset(self) -> None:
        self._keyframe = None
        self._last_attempt_time_s = None

    def _make_update(
        self,
        context: _UpdateContext,
        *,
        reason: str,
        attempted: bool,
        accepted: bool = False,
        keyframe_replaced: bool = False,
        visual_yaw_delta_deg: float = 0.0,
        dead_reckoned_yaw_delta_deg: float = 0.0,
        yaw_residual_deg: float = 0.0,
        applied_yaw_correction_deg: float = 0.0,
        corrected_pose: Optional[Pose] = None,
        ultrasonic_rejected: bool = False,
    ) -> VisualYawUpdate:
        return VisualYawUpdate(
            attempted=attempted,
            accepted=accepted,
            keyframe_replaced=keyframe_replaced,
            reason=reason,
            confidence=context.confidence,
            high_confidence=context.high_confidence,
            keyframe_age_s=context.keyframe_age_s,
            baseline_distance=context.baseline_distance,
            baseline_yaw_deg=context.baseline_yaw_deg,
            sharpness=context.sharpness,
            contrast=context.contrast,
            feature_count=context.feature_count,
            tracked_count=context.tracked_count,
            inlier_count=context.inlier_count,
            inlier_ratio=context.inlier_ratio,
            median_flow_px=context.median_flow_px,
            inlier_coverage=context.inlier_coverage,
            median_epipolar_error_px=context.median_epipolar_error_px,
            p90_epipolar_error_px=context.p90_epipolar_error_px,
            rotation_deg=context.rotation_deg,
            visual_yaw_delta_deg=visual_yaw_delta_deg,
            dead_reckoned_yaw_delta_deg=dead_reckoned_yaw_delta_deg,
            yaw_residual_deg=yaw_residual_deg,
            applied_yaw_correction_deg=applied_yaw_correction_deg,
            corrected_pose=corrected_pose,
            ultrasonic_distance_cm=context.ultrasonic_distance_cm,
            ultrasonic_rejected=ultrasonic_rejected,
        )

    @staticmethod
    def _apply_motion_to_context(
        context: _UpdateContext,
        motion: _MotionEstimate,
        *,
        confidence: float = 0.0,
        high_confidence: bool = False,
    ) -> None:
        context.confidence = float(confidence)
        context.high_confidence = bool(high_confidence)
        context.tracked_count = int(motion.tracked_count)
        context.inlier_count = int(motion.inlier_count)
        context.inlier_ratio = float(motion.inlier_ratio)
        context.median_flow_px = float(motion.median_flow_px)
        context.inlier_coverage = float(motion.inlier_coverage)
        context.median_epipolar_error_px = float(motion.median_epipolar_error_px)
        context.p90_epipolar_error_px = float(motion.p90_epipolar_error_px)
        context.rotation_deg = float(motion.rotation_deg)

    def update(
        self,
        frame_rgb_or_bgr: np.ndarray,
        dead_reckoned_pose: Pose,
        timestamp_s: float,
        ultrasonic_distance_cm: Optional[float] = None,
    ) -> VisualYawUpdate:
        gray = self._prepare_gray(frame_rgb_or_bgr)
        sharpness = self._frame_sharpness(gray)
        contrast = self._frame_contrast(gray)
        context = _UpdateContext(
            sharpness=sharpness,
            contrast=contrast,
            ultrasonic_distance_cm=ultrasonic_distance_cm,
        )

        if self._keyframe is None:
            update = self._bootstrap_keyframe(
                gray=gray,
                pose=dead_reckoned_pose,
                timestamp_s=timestamp_s,
                context=context,
                reason="bootstrap",
            )
            return update

        context.keyframe_age_s = max(0.0, float(timestamp_s) - float(self._keyframe.timestamp_s))
        context.baseline_distance = math.hypot(
            float(dead_reckoned_pose.x) - float(self._keyframe.pose.x),
            float(dead_reckoned_pose.y) - float(self._keyframe.pose.y),
        )
        context.baseline_yaw_deg = abs(
            math.degrees(_wrap_angle(float(dead_reckoned_pose.theta) - float(self._keyframe.pose.theta)))
        )

        if self._last_attempt_time_s is not None:
            dt_attempt = float(timestamp_s) - float(self._last_attempt_time_s)
            if dt_attempt < float(self.cfg.attempt_interval_s):
                return self._make_update(
                    context,
                    attempted=False,
                    accepted=False,
                    reason="wait_interval",
                )

        if (
            context.baseline_distance < float(self.cfg.min_baseline_distance)
            and context.baseline_yaw_deg < float(self.cfg.min_baseline_yaw_deg)
        ):
            return self._make_update(
                context,
                attempted=False,
                accepted=False,
                reason="insufficient_baseline",
            )

        self._last_attempt_time_s = float(timestamp_s)

        if sharpness < float(self.cfg.min_sharpness):
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="blurry_frame",
            )

        if contrast < float(self.cfg.min_contrast):
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="low_contrast",
            )

        current_points = self._detect_features(gray)
        context.feature_count = 0 if current_points is None else int(current_points.shape[0])
        if context.feature_count < int(self.cfg.min_current_features):
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="too_few_features",
            )

        motion = self._estimate_motion(self._keyframe.gray, self._keyframe.points_xy, gray)
        confidence = self._compute_confidence(
            sharpness=sharpness,
            contrast=contrast,
            feature_count=context.feature_count,
            tracked_count=motion.tracked_count,
            inlier_count=motion.inlier_count,
            inlier_ratio=motion.inlier_ratio,
            median_flow_px=motion.median_flow_px,
            inlier_coverage=motion.inlier_coverage,
            median_epipolar_error_px=motion.median_epipolar_error_px,
        )
        high_confidence = bool(motion.ok and confidence >= float(self.cfg.confidence_good_threshold))
        self._apply_motion_to_context(context, motion, confidence=confidence, high_confidence=high_confidence)

        if not motion.ok:
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason=motion.reason,
            )

        if not high_confidence:
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="low_confidence",
            )

        visual_yaw_delta = _yaw_from_relative_rotation(motion.rotation_matrix)
        dr_yaw_delta = _wrap_angle(float(dead_reckoned_pose.theta) - float(self._keyframe.pose.theta))
        yaw_residual = _wrap_angle(visual_yaw_delta - dr_yaw_delta)
        yaw_residual_deg = math.degrees(yaw_residual)
        if abs(yaw_residual_deg) > float(self.cfg.max_yaw_residual_deg):
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="yaw_residual_too_large",
                visual_yaw_delta_deg=math.degrees(visual_yaw_delta),
                dead_reckoned_yaw_delta_deg=math.degrees(dr_yaw_delta),
                yaw_residual_deg=yaw_residual_deg,
            )

        correction_limit_deg = float(self.cfg.max_applied_yaw_correction_deg)
        if self._is_valid_ultrasonic_cm(ultrasonic_distance_cm) and float(ultrasonic_distance_cm) <= float(self.cfg.ultrasonic_close_range_cm):
            correction_limit_deg = min(correction_limit_deg, float(self.cfg.ultrasonic_close_yaw_limit_deg))

        applied_yaw_correction_deg = float(np.clip(
            float(self.cfg.yaw_correction_gain) * yaw_residual_deg,
            -float(correction_limit_deg),
            float(correction_limit_deg),
        ))
        corrected_theta = _wrap_angle(float(dead_reckoned_pose.theta) + math.radians(applied_yaw_correction_deg))
        corrected_pose = Pose(float(dead_reckoned_pose.x), float(dead_reckoned_pose.y), corrected_theta)

        ultrasonic_rejected = not self._ultrasonic_consistent(
            pose_dead_reckoned=dead_reckoned_pose,
            corrected_pose=corrected_pose,
            ultrasonic_distance_cm=ultrasonic_distance_cm,
        )
        if ultrasonic_rejected:
            return self._make_update(
                context,
                attempted=True,
                accepted=False,
                reason="ultrasonic_conflict",
                visual_yaw_delta_deg=math.degrees(visual_yaw_delta),
                dead_reckoned_yaw_delta_deg=math.degrees(dr_yaw_delta),
                yaw_residual_deg=yaw_residual_deg,
                applied_yaw_correction_deg=applied_yaw_correction_deg,
                corrected_pose=corrected_pose,
                ultrasonic_rejected=True,
            )

        replacement_points = current_points
        if replacement_points is None or replacement_points.shape[0] < int(self.cfg.min_keyframe_features):
            replacement_points = motion.tracked_cur.astype(np.float32)

        self._set_keyframe(
            gray=gray,
            points_xy=replacement_points,
            pose=corrected_pose,
            timestamp_s=timestamp_s,
        )

        return self._make_update(
            context,
            attempted=True,
            accepted=True,
            keyframe_replaced=True,
            reason="accepted",
            visual_yaw_delta_deg=math.degrees(visual_yaw_delta),
            dead_reckoned_yaw_delta_deg=math.degrees(dr_yaw_delta),
            yaw_residual_deg=yaw_residual_deg,
            applied_yaw_correction_deg=applied_yaw_correction_deg,
            corrected_pose=corrected_pose,
        )

    def _bootstrap_keyframe(
        self,
        *,
        gray: np.ndarray,
        pose: Pose,
        timestamp_s: float,
        context: _UpdateContext,
        reason: str,
    ) -> VisualYawUpdate:
        points = self._detect_features(gray)
        feature_count = 0 if points is None else int(points.shape[0])
        context.feature_count = feature_count
        if feature_count < int(self.cfg.min_keyframe_features):
            return self._make_update(
                context,
                attempted=False,
                accepted=False,
                reason="bootstrap_too_few_features",
            )

        self._set_keyframe(gray=gray, points_xy=points, pose=pose, timestamp_s=timestamp_s)
        return self._make_update(
            context,
            attempted=False,
            accepted=False,
            keyframe_replaced=True,
            reason=reason,
        )

    def _set_keyframe(self, *, gray: np.ndarray, points_xy: np.ndarray, pose: Pose, timestamp_s: float) -> None:
        self._keyframe = _Keyframe(
            gray=np.ascontiguousarray(gray),
            points_xy=np.ascontiguousarray(points_xy.astype(np.float32).reshape(-1, 2)),
            pose=Pose(float(pose.x), float(pose.y), float(pose.theta)),
            timestamp_s=float(timestamp_s),
        )

    def _estimate_motion(
        self,
        keyframe_gray: np.ndarray,
        keyframe_points_xy: np.ndarray,
        current_gray: np.ndarray,
    ) -> _MotionEstimate:
        tracked_prev, tracked_cur = self._track_keyframe_points(
            keyframe_gray=keyframe_gray,
            keyframe_points_xy=keyframe_points_xy,
            current_gray=current_gray,
        )
        tracked_count = 0 if tracked_prev is None else int(tracked_prev.shape[0])
        if tracked_prev is None or tracked_cur is None or tracked_count < int(self.cfg.min_tracked_points):
            return _MotionEstimate(
                ok=False,
                reason="too_few_tracks",
                tracked_count=tracked_count,
            )

        flow = np.linalg.norm(tracked_cur - tracked_prev, axis=1)
        median_flow_px = float(np.median(flow)) if flow.size > 0 else 0.0
        if median_flow_px < float(self.cfg.min_median_flow_px):
            return _MotionEstimate(
                ok=False,
                reason="insufficient_parallax",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                median_flow_px=median_flow_px,
            )

        E, ransac_mask = cv2.findEssentialMat(
            tracked_prev,
            tracked_cur,
            self.K,
            method=cv2.RANSAC,
            prob=float(self.cfg.ransac_prob),
            threshold=float(self.cfg.ransac_threshold_px),
        )
        if E is None or ransac_mask is None:
            return _MotionEstimate(
                ok=False,
                reason="essential_failed",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                median_flow_px=median_flow_px,
            )

        ransac_mask = ransac_mask.reshape(-1).astype(bool)
        pts_prev_ransac = tracked_prev[ransac_mask]
        pts_cur_ransac = tracked_cur[ransac_mask]
        if pts_prev_ransac.shape[0] < int(self.cfg.min_inliers):
            return _MotionEstimate(
                ok=False,
                reason="too_few_ransac_inliers",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                inlier_count=int(pts_prev_ransac.shape[0]),
                inlier_ratio=float(pts_prev_ransac.shape[0]) / max(1.0, float(tracked_count)),
                median_flow_px=median_flow_px,
            )

        best_E: Optional[np.ndarray] = None
        best_R: Optional[np.ndarray] = None
        best_mask: Optional[np.ndarray] = None
        best_support = -1
        for candidate in self._essential_candidates(E):
            try:
                _pose_count, R, _t, pose_mask = cv2.recoverPose(candidate, pts_prev_ransac, pts_cur_ransac, self.K)
            except cv2.error:
                continue

            if pose_mask is None:
                continue

            pose_support = int(pose_mask.reshape(-1).astype(bool).sum())
            if pose_support > best_support:
                best_support = pose_support
                best_E = candidate
                best_R = R
                best_mask = pose_mask.reshape(-1).astype(bool)

        if best_E is None or best_R is None or best_mask is None or best_support <= 0:
            return _MotionEstimate(
                ok=False,
                reason="recover_pose_failed",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                inlier_count=int(pts_prev_ransac.shape[0]),
                inlier_ratio=float(pts_prev_ransac.shape[0]) / max(1.0, float(tracked_count)),
                median_flow_px=median_flow_px,
            )

        inlier_prev = pts_prev_ransac[best_mask]
        inlier_cur = pts_cur_ransac[best_mask]
        inlier_count = int(inlier_prev.shape[0])
        inlier_ratio = float(inlier_count) / max(1.0, float(tracked_count))
        if inlier_count < int(self.cfg.min_inliers):
            return _MotionEstimate(
                ok=False,
                reason="too_few_pose_inliers",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_flow_px=median_flow_px,
            )

        if inlier_ratio < float(self.cfg.min_inlier_ratio):
            return _MotionEstimate(
                ok=False,
                reason="low_inlier_ratio",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_flow_px=median_flow_px,
            )

        inlier_coverage = _point_coverage(inlier_cur, current_gray.shape)
        if inlier_coverage < float(self.cfg.min_inlier_coverage):
            return _MotionEstimate(
                ok=False,
                reason="low_inlier_coverage",
                tracked_cur=tracked_cur,
                tracked_count=tracked_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_flow_px=median_flow_px,
                inlier_coverage=inlier_coverage,
            )

        epipolar_errors_px = self._epipolar_errors_px(best_E, inlier_prev, inlier_cur)
        median_epi = float(np.median(epipolar_errors_px)) if epipolar_errors_px.size > 0 else float("inf")
        p90_epi = float(np.percentile(epipolar_errors_px, 90)) if epipolar_errors_px.size > 0 else float("inf")
        rotation_deg = _rotation_angle_deg(best_R)
        if median_epi > float(self.cfg.max_median_epipolar_error_px) or p90_epi > float(self.cfg.max_p90_epipolar_error_px):
            return _MotionEstimate(
                ok=False,
                reason="epipolar_error_too_large",
                tracked_cur=tracked_cur,
                rotation_matrix=best_R,
                tracked_count=tracked_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_flow_px=median_flow_px,
                inlier_coverage=inlier_coverage,
                median_epipolar_error_px=median_epi,
                p90_epipolar_error_px=p90_epi,
                rotation_deg=rotation_deg,
            )

        if rotation_deg > float(self.cfg.max_rotation_deg):
            return _MotionEstimate(
                ok=False,
                reason="rotation_too_large",
                tracked_cur=tracked_cur,
                rotation_matrix=best_R,
                tracked_count=tracked_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_flow_px=median_flow_px,
                inlier_coverage=inlier_coverage,
                median_epipolar_error_px=median_epi,
                p90_epipolar_error_px=p90_epi,
                rotation_deg=rotation_deg,
            )

        return _MotionEstimate(
            ok=True,
            reason="ok",
            tracked_cur=tracked_cur,
            rotation_matrix=best_R,
            tracked_count=tracked_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            median_flow_px=median_flow_px,
            inlier_coverage=inlier_coverage,
            median_epipolar_error_px=median_epi,
            p90_epipolar_error_px=p90_epi,
            rotation_deg=rotation_deg,
        )

    def _track_keyframe_points(
        self,
        *,
        keyframe_gray: np.ndarray,
        keyframe_points_xy: np.ndarray,
        current_gray: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if keyframe_points_xy is None or keyframe_points_xy.size == 0:
            return None, None

        lk_points = keyframe_points_xy.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            keyframe_gray,
            current_gray,
            lk_points,
            None,
            winSize=(int(self.cfg.lk_win_size), int(self.cfg.lk_win_size)),
            maxLevel=int(self.cfg.lk_max_level),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.cfg.lk_max_iterations),
                float(self.cfg.lk_epsilon),
            ),
        )
        if next_pts is None or status is None:
            return None, None

        valid = status.reshape(-1).astype(bool)
        if err is not None:
            valid &= err.reshape(-1) <= float(self.cfg.lk_max_error_px)

        next_xy = next_pts.reshape(-1, 2)
        h, w = int(current_gray.shape[0]), int(current_gray.shape[1])
        valid &= np.isfinite(next_xy).all(axis=1)
        valid &= (next_xy[:, 0] >= 0.0) & (next_xy[:, 0] < float(w))
        valid &= (next_xy[:, 1] >= 0.0) & (next_xy[:, 1] < float(h))

        if bool(self.cfg.use_forward_backward_check):
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray,
                keyframe_gray,
                next_pts,
                None,
                winSize=(int(self.cfg.lk_win_size), int(self.cfg.lk_win_size)),
                maxLevel=int(self.cfg.lk_max_level),
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(self.cfg.lk_max_iterations),
                    float(self.cfg.lk_epsilon),
                ),
            )
            if back_pts is None or back_status is None:
                return None, None

            back_valid = back_status.reshape(-1).astype(bool)
            fb_err = np.linalg.norm(back_pts.reshape(-1, 2) - keyframe_points_xy, axis=1)
            valid &= back_valid
            valid &= fb_err <= float(self.cfg.max_forward_backward_error_px)

        tracked_prev = keyframe_points_xy[valid].astype(np.float64)
        tracked_cur = next_xy[valid].astype(np.float64)
        if tracked_prev.shape[0] == 0:
            return None, None
        return tracked_prev, tracked_cur

    def _prepare_gray(self, img: np.ndarray) -> np.ndarray:
        cfg = self.cfg.preprocess
        src = img
        if src.ndim == 3 and cfg.undistort and self.intr.dist() is not None:
            src = cv2.undistort(src, self.K, self.intr.dist())

        if src.ndim == 2:
            gray = src
        else:
            color = np.ascontiguousarray(src)
            if color.dtype != np.uint8:
                color = np.clip(color, 0, 255).astype(np.uint8)

            if cfg.auto_white_balance and str(cfg.gray_source).lower() != "green":
                color = self._gray_world_balance(
                    color,
                    order=str(cfg.input_color_order).lower(),
                    max_gain=float(cfg.max_channel_gain),
                )

            gray_source = str(cfg.gray_source).lower()
            if gray_source == "green":
                gray = color[:, :, 1]
            elif gray_source == "y_channel":
                code = cv2.COLOR_BGR2YCrCb if cfg.input_color_order == "bgr" else cv2.COLOR_RGB2YCrCb
                gray = cv2.cvtColor(color, code)[:, :, 0]
            else:
                code = cv2.COLOR_BGR2GRAY if cfg.input_color_order == "bgr" else cv2.COLOR_RGB2GRAY
                gray = cv2.cvtColor(color, code)

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        if cfg.flat_field_correction:
            gray = self._flat_field_correct(
                gray,
                sigma=float(cfg.flat_field_sigma),
                strength=float(cfg.flat_field_strength),
            )

        if cfg.use_clahe:
            tile = max(1, int(cfg.clahe_tile_size))
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(tile, tile))
            gray = self._clahe.apply(gray)

        if abs(float(cfg.gamma) - 1.0) > 1e-6:
            gray = cv2.LUT(gray, self._gamma_lut_for(float(cfg.gamma)))

        amount = float(cfg.unsharp_amount)
        if amount > 1e-6:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
            gray = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)

        return np.ascontiguousarray(gray)

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        mask = None
        if float(self.cfg.feature_mask_top_fraction) > 0.0:
            h, w = gray.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            y0 = int(round(float(h) * float(self.cfg.feature_mask_top_fraction)))
            mask[y0:, :] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.cfg.max_corners),
            qualityLevel=float(self.cfg.quality_level),
            minDistance=float(self.cfg.min_distance_px),
            mask=mask,
            blockSize=int(self.cfg.block_size),
        )
        if pts is None:
            return None
        return np.ascontiguousarray(pts.reshape(-1, 2).astype(np.float32))

    def _compute_confidence(
        self,
        *,
        sharpness: float,
        contrast: float,
        feature_count: int,
        tracked_count: int,
        inlier_count: int,
        inlier_ratio: float,
        median_flow_px: float,
        inlier_coverage: float,
        median_epipolar_error_px: float,
    ) -> float:
        feature_score = np.clip(float(feature_count) / max(1.0, float(self.cfg.min_current_features)), 0.0, 1.0)
        track_score = np.clip(float(tracked_count) / max(1.0, float(self.cfg.min_tracked_points)), 0.0, 1.0)
        inlier_score = np.clip(float(inlier_count) / max(1.0, float(self.cfg.min_inliers)), 0.0, 1.0)
        ratio_score = np.clip(float(inlier_ratio) / max(1e-6, float(self.cfg.min_inlier_ratio)), 0.0, 1.0)
        flow_score = np.clip(float(median_flow_px) / max(1e-6, float(self.cfg.min_median_flow_px)), 0.0, 1.0)
        coverage_score = np.clip(float(inlier_coverage) / max(1e-6, float(self.cfg.min_inlier_coverage)), 0.0, 1.0)
        sharp_score = np.clip(float(sharpness) / max(1e-6, float(self.cfg.min_sharpness)), 0.0, 1.0)
        contrast_score = np.clip(float(contrast) / max(1e-6, float(self.cfg.min_contrast)), 0.0, 1.0)
        epi_score = np.clip(1.0 - (float(median_epipolar_error_px) / max(1e-6, float(self.cfg.max_median_epipolar_error_px))), 0.0, 1.0)

        confidence = (
            0.12 * feature_score
            + 0.12 * track_score
            + 0.22 * inlier_score
            + 0.14 * ratio_score
            + 0.10 * flow_score
            + 0.10 * coverage_score
            + 0.10 * sharp_score
            + 0.05 * contrast_score
            + 0.05 * epi_score
        )
        return float(np.clip(confidence, 0.0, 1.0))

    def _ultrasonic_consistent(
        self,
        *,
        pose_dead_reckoned: Pose,
        corrected_pose: Pose,
        ultrasonic_distance_cm: Optional[float],
    ) -> bool:
        if not self._is_valid_ultrasonic_cm(ultrasonic_distance_cm):
            return True

        dist_cm = float(ultrasonic_distance_cm)
        if dist_cm > float(self.cfg.ultrasonic_max_range_cm):
            return True

        cm_per_unit = max(1e-6, float(self.cfg.cm_per_world_unit))
        forward_units = dist_cm / cm_per_unit

        obs_x = float(pose_dead_reckoned.x) + forward_units * math.cos(float(pose_dead_reckoned.theta))
        obs_y = float(pose_dead_reckoned.y) + forward_units * math.sin(float(pose_dead_reckoned.theta))

        dx = obs_x - float(corrected_pose.x)
        dy = obs_y - float(corrected_pose.y)
        c = math.cos(float(corrected_pose.theta))
        s = math.sin(float(corrected_pose.theta))

        forward_prime_cm = (dx * c + dy * s) * cm_per_unit
        lateral_prime_cm = (-dx * s + dy * c) * cm_per_unit

        if forward_prime_cm <= 0.0:
            return False

        if abs(forward_prime_cm - dist_cm) > float(self.cfg.ultrasonic_forward_tolerance_cm):
            return False

        if abs(lateral_prime_cm) > float(self.cfg.ultrasonic_lateral_tolerance_cm):
            return False

        return True

    @staticmethod
    def _is_valid_ultrasonic_cm(dist_cm: Optional[float]) -> bool:
        return dist_cm is not None and math.isfinite(float(dist_cm)) and float(dist_cm) > 0.0

    @staticmethod
    def _essential_candidates(E: np.ndarray) -> list[np.ndarray]:
        E = np.asarray(E, dtype=np.float64)
        if E.size == 9:
            return [E.reshape(3, 3)]

        candidates: list[np.ndarray] = []
        if E.ndim != 2:
            return candidates

        if E.shape[1] == 3 and E.shape[0] % 3 == 0:
            for row in range(0, E.shape[0], 3):
                candidates.append(E[row:row + 3, :])
        elif E.shape[0] == 3 and E.shape[1] % 3 == 0:
            for col in range(0, E.shape[1], 3):
                candidates.append(E[:, col:col + 3])
        return [candidate for candidate in candidates if candidate.shape == (3, 3)]

    def _epipolar_errors_px(self, E: np.ndarray, pts1_px: np.ndarray, pts2_px: np.ndarray) -> np.ndarray:
        Kinv = np.linalg.inv(self.K)
        F = Kinv.T @ E @ Kinv

        x1 = np.column_stack((pts1_px[:, 0], pts1_px[:, 1], np.ones((pts1_px.shape[0],), dtype=np.float64)))
        x2 = np.column_stack((pts2_px[:, 0], pts2_px[:, 1], np.ones((pts2_px.shape[0],), dtype=np.float64)))
        Fx1 = (F @ x1.T).T
        Ftx2 = (F.T @ x2.T).T
        numer = np.abs(np.sum(x2 * Fx1, axis=1))

        denom1 = np.sqrt(np.maximum(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2, 1e-12))
        denom2 = np.sqrt(np.maximum(Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2, 1e-12))
        return 0.5 * (numer / denom1 + numer / denom2)

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


__all__ = [
    "CameraIntrinsics",
    "ImagePreprocessConfig",
    "VisualOdometryConfig",
    "VisualYawUpdate",
    "ConservativeVisualOdometry",
]
