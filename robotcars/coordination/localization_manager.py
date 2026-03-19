from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from car_tools.obstacle_detection import MonocularVSLAM, VslamStatus
from coordination.shared_map import FrameName, SharedMap
from model import Pose


@dataclass
class LocalizationConfig:
    marker_blend_alpha: float = 0.35
    marker_snap_distance: float = 1.5
    marker_snap_heading_deg: float = 35.0
    marker_snap_consistency_frames: int = 3
    marker_snap_position_tolerance: float = 0.75
    marker_snap_heading_tolerance_deg: float = 12.0


@dataclass
class LocalizationStatus:
    pose_source: Literal["init", "dr", "marker", "fused"] = "init"
    marker_visible: bool = False
    marker_pose_valid: bool = False
    marker_confidence: float = 0.0
    marker_gate_reason: str = "startup"
    marker_snap_count: int = 0
    correction_distance: float = 0.0
    correction_heading_deg: float = 0.0
    seconds_since_marker_fix: float = -1.0


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def predict_dead_reckoning_pose(
    pose_world: Pose,
    *,
    forward_step: float,
    yaw_delta: float,
) -> Pose:
    step = float(forward_step)
    dtheta = float(yaw_delta)
    theta_mid = float(pose_world.theta) + 0.5 * dtheta

    return Pose(
        x=float(pose_world.x + step * math.cos(theta_mid)),
        y=float(pose_world.y + step * math.sin(theta_mid)),
        theta=float(wrap_angle(float(pose_world.theta) + dtheta)),
    )


class LocalizationManager:
    """Owns shared pose publication."""

    def __init__(
        self,
        *,
        shared_map: SharedMap,
        car_id: int = 0,
        initial_pose: Optional[Pose] = None,
        marker_localizer: Optional[MonocularVSLAM] = None,
        cfg: Optional[LocalizationConfig] = None,
    ):
        self.shared_map = shared_map
        self.car_id = int(car_id)
        self.cfg = cfg or LocalizationConfig()
        self.marker_localizer = marker_localizer
        self._status = LocalizationStatus()
        self._pending_marker_pose: Optional[Pose] = None
        self._pending_marker_count = 0
        self._last_marker_fix_t: Optional[float] = None

        start_pose = initial_pose if initial_pose is not None else Pose(0.0, 0.0, 0.0)
        self._publish_pose(start_pose, source="init")

    def get_pose(self, *, frame: FrameName = "world") -> Optional[Pose]:
        return self.shared_map.get_pose(self.car_id, frame=frame)

    def get_status(self) -> LocalizationStatus:
        s = self._status
        return LocalizationStatus(
            pose_source=str(s.pose_source),
            marker_visible=bool(s.marker_visible),
            marker_pose_valid=bool(s.marker_pose_valid),
            marker_confidence=float(s.marker_confidence),
            marker_gate_reason=str(s.marker_gate_reason),
            marker_snap_count=int(s.marker_snap_count),
            correction_distance=float(s.correction_distance),
            correction_heading_deg=float(s.correction_heading_deg),
            seconds_since_marker_fix=float(s.seconds_since_marker_fix),
        )

    def get_marker_status(self) -> Optional[VslamStatus]:
        if self.marker_localizer is None:
            return None
        return self.marker_localizer.get_status()

    def predict_dead_reckoning(self, *, forward_step: float, yaw_delta: float) -> Pose:
        pose_world = self.get_pose(frame="world")
        if pose_world is None:
            pose_world = Pose(0.0, 0.0, 0.0)

        next_pose = predict_dead_reckoning_pose(
            pose_world,
            forward_step=forward_step,
            yaw_delta=yaw_delta,
        )
        self._refresh_fix_age()
        self._publish_pose(next_pose, source="dr")
        return next_pose

    def initialize_from_marker_frame(
        self,
        frame_rgb_or_bgr: Optional[np.ndarray],
        *,
        camera_pan_deg: Optional[float] = None,
    ) -> Optional[Pose]:
        marker_pose = self._marker_pose_from_frame(
            frame_rgb_or_bgr,
            camera_pan_deg=camera_pan_deg,
        )
        if marker_pose is None:
            return None
        if not self._accept_pending_marker_pose(marker_pose):
            return None

        self._status.correction_distance = 0.0
        self._status.correction_heading_deg = 0.0
        self._mark_marker_fix()
        self._publish_pose(marker_pose, source="marker")
        self._clear_pending_marker_pose()
        return marker_pose

    def correct_from_marker_frame(
        self,
        frame_rgb_or_bgr: Optional[np.ndarray],
        *,
        camera_pan_deg: Optional[float] = None,
    ) -> Optional[Pose]:
        marker_pose = self._marker_pose_from_frame(
            frame_rgb_or_bgr,
            camera_pan_deg=camera_pan_deg,
        )
        if marker_pose is None:
            self._refresh_fix_age()
            return self.get_pose(frame="world")

        current_pose = self.get_pose(frame="world")
        if current_pose is None:
            self._status.correction_distance = 0.0
            self._status.correction_heading_deg = 0.0
            self._mark_marker_fix()
            self._publish_pose(marker_pose, source="marker")
            self._clear_pending_marker_pose()
            return marker_pose

        fused_pose, source = self._fuse_marker_pose(current_pose, marker_pose)
        if fused_pose is None:
            self._refresh_fix_age()
            return current_pose
        self._status.correction_distance = math.hypot(
            float(fused_pose.x) - float(current_pose.x),
            float(fused_pose.y) - float(current_pose.y),
        )
        self._status.correction_heading_deg = abs(
            math.degrees(wrap_angle(float(fused_pose.theta) - float(current_pose.theta)))
        )
        self._mark_marker_fix()
        self._publish_pose(fused_pose, source=source)
        if source == "marker":
            self._clear_pending_marker_pose()
        return fused_pose

    def step(
        self,
        *,
        forward_step: float,
        yaw_delta: float,
        frame_rgb_or_bgr: Optional[np.ndarray] = None,
        camera_pan_deg: Optional[float] = None,
    ) -> Pose:
        pose = self.predict_dead_reckoning(
            forward_step=forward_step,
            yaw_delta=yaw_delta,
        )
        if frame_rgb_or_bgr is None:
            return pose
        corrected = self.correct_from_marker_frame(
            frame_rgb_or_bgr,
            camera_pan_deg=camera_pan_deg,
        )
        return pose if corrected is None else corrected

    def _publish_pose(
        self,
        pose: Pose,
        *,
        source: Literal["init", "dr", "marker", "fused"],
    ) -> None:
        self.shared_map.set_pose(self.car_id, pose)
        self._status.pose_source = source
        self._refresh_fix_age()

    def _marker_pose_from_frame(
        self,
        frame_rgb_or_bgr: Optional[np.ndarray],
        *,
        camera_pan_deg: Optional[float] = None,
    ) -> Optional[Pose]:
        if self.marker_localizer is None or frame_rgb_or_bgr is None:
            return None

        marker_pose = self.marker_localizer.tick(
            frame_rgb_or_bgr,
            camera_pan_deg=camera_pan_deg,
        )
        marker_status = self.marker_localizer.get_status()
        self._status.marker_visible = bool(marker_status.origin_visible)
        self._status.marker_pose_valid = bool(marker_status.pose_valid)
        self._status.marker_confidence = float(marker_status.confidence)
        self._status.marker_gate_reason = str(marker_status.gate_reason)
        if marker_pose is None or not marker_status.pose_valid:
            return None
        return marker_pose

    def _fuse_marker_pose(
        self,
        predicted_pose: Pose,
        marker_pose: Pose,
    ) -> tuple[Optional[Pose], Literal["marker", "fused"]]:
        dx = float(marker_pose.x) - float(predicted_pose.x)
        dy = float(marker_pose.y) - float(predicted_pose.y)
        dist = math.hypot(dx, dy)
        dtheta_deg = abs(math.degrees(wrap_angle(float(marker_pose.theta) - float(predicted_pose.theta))))

        if (
            dist < float(self.cfg.marker_snap_distance)
            and dtheta_deg < float(self.cfg.marker_snap_heading_deg)
        ):
            self._clear_pending_marker_pose()
            return self._blend_pose(predicted_pose, marker_pose), "fused"

        if not self._accept_pending_marker_pose(marker_pose):
            return None, "marker"

        return marker_pose, "marker"

    def _blend_pose(self, predicted_pose: Pose, marker_pose: Pose) -> Pose:
        alpha = float(self.cfg.marker_blend_alpha)
        x = (1.0 - alpha) * float(predicted_pose.x) + alpha * float(marker_pose.x)
        y = (1.0 - alpha) * float(predicted_pose.y) + alpha * float(marker_pose.y)

        c0, s0 = math.cos(float(predicted_pose.theta)), math.sin(float(predicted_pose.theta))
        c1, s1 = math.cos(float(marker_pose.theta)), math.sin(float(marker_pose.theta))
        cf = (1.0 - alpha) * c0 + alpha * c1
        sf = (1.0 - alpha) * s0 + alpha * s1

        return Pose(x=x, y=y, theta=math.atan2(sf, cf))

    def _accept_pending_marker_pose(self, marker_pose: Pose) -> bool:
        if self._pending_marker_pose is None:
            self._pending_marker_pose = marker_pose
            self._pending_marker_count = 1
        else:
            if self._poses_are_consistent(self._pending_marker_pose, marker_pose):
                self._pending_marker_pose = marker_pose
                self._pending_marker_count += 1
            else:
                self._pending_marker_pose = marker_pose
                self._pending_marker_count = 1

        self._status.marker_snap_count = self._pending_marker_count
        return self._pending_marker_count >= max(1, int(self.cfg.marker_snap_consistency_frames))

    def _clear_pending_marker_pose(self) -> None:
        self._pending_marker_pose = None
        self._pending_marker_count = 0
        self._status.marker_snap_count = 0

    def _mark_marker_fix(self) -> None:
        self._last_marker_fix_t = time.monotonic()
        self._refresh_fix_age()

    def _refresh_fix_age(self) -> None:
        if self._last_marker_fix_t is None:
            self._status.seconds_since_marker_fix = -1.0
            return
        self._status.seconds_since_marker_fix = max(0.0, time.monotonic() - self._last_marker_fix_t)

    def _poses_are_consistent(self, pose_a: Pose, pose_b: Pose) -> bool:
        dist = math.hypot(float(pose_b.x) - float(pose_a.x), float(pose_b.y) - float(pose_a.y))
        dtheta_deg = abs(math.degrees(wrap_angle(float(pose_b.theta) - float(pose_a.theta))))
        return (
            dist <= float(self.cfg.marker_snap_position_tolerance)
            and dtheta_deg <= float(self.cfg.marker_snap_heading_tolerance_deg)
        )
