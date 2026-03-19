from __future__ import annotations

import math
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


@dataclass
class LocalizationStatus:
    pose_source: Literal["init", "dead_reckoning", "marker", "marker_fused"] = "init"
    marker_visible: bool = False
    marker_pose_valid: bool = False
    marker_confidence: float = 0.0
    marker_gate_reason: str = "startup"


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
        self._publish_pose(next_pose, source="dead_reckoning")
        return next_pose

    def correct_from_marker_frame(
        self,
        frame_rgb_or_bgr: Optional[np.ndarray],
        *,
        camera_pan_deg: Optional[float] = None,
    ) -> Optional[Pose]:
        if self.marker_localizer is None or frame_rgb_or_bgr is None:
            return self.get_pose(frame="world")

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
            return self.get_pose(frame="world")

        current_pose = self.get_pose(frame="world")
        if current_pose is None:
            self._publish_pose(marker_pose, source="marker")
            return marker_pose

        fused_pose, source = self._fuse_marker_pose(current_pose, marker_pose)
        self._publish_pose(fused_pose, source=source)
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
        source: Literal["init", "dead_reckoning", "marker", "marker_fused"],
    ) -> None:
        self.shared_map.set_pose(self.car_id, pose)
        self._status.pose_source = source

    def _fuse_marker_pose(
        self,
        predicted_pose: Pose,
        marker_pose: Pose,
    ) -> tuple[Pose, Literal["marker", "marker_fused"]]:
        dx = float(marker_pose.x) - float(predicted_pose.x)
        dy = float(marker_pose.y) - float(predicted_pose.y)
        dist = math.hypot(dx, dy)
        dtheta_deg = abs(math.degrees(wrap_angle(float(marker_pose.theta) - float(predicted_pose.theta))))

        if (
            dist >= float(self.cfg.marker_snap_distance)
            or dtheta_deg >= float(self.cfg.marker_snap_heading_deg)
        ):
            return marker_pose, "marker"

        alpha = float(self.cfg.marker_blend_alpha)
        x = (1.0 - alpha) * float(predicted_pose.x) + alpha * float(marker_pose.x)
        y = (1.0 - alpha) * float(predicted_pose.y) + alpha * float(marker_pose.y)

        c0, s0 = math.cos(float(predicted_pose.theta)), math.sin(float(predicted_pose.theta))
        c1, s1 = math.cos(float(marker_pose.theta)), math.sin(float(marker_pose.theta))
        cf = (1.0 - alpha) * c0 + alpha * c1
        sf = (1.0 - alpha) * s0 + alpha * s1

        return Pose(x=x, y=y, theta=math.atan2(sf, cf)), "marker_fused"
