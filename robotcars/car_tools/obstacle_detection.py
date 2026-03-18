from __future__ import annotations

import math
from dataclasses import dataclass, field
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
class OriginBoxConfig:
    enabled: bool = False
    dictionary_name: str = "DICT_4X4_50"
    marker_id: int = 0
    marker_size_world: float = 1.0
    marker_world_x: float = 0.0
    marker_world_y: float = 0.0
    marker_world_yaw_deg: float = 0.0

    camera_mount_forward: float = 0.0
    camera_mount_left: float = 0.0
    camera_pan_sign: float = 1.0

    min_area_ratio: float = 0.001
    max_range_world: float = 80.0

    pan_track_gain: float = 1.0
    max_pan_step_deg: float = 10.0


@dataclass
class VslamConfig:
    # Debug
    debug_draw_frame: bool = True

    # Pose frame publishing
    forward_sign: float = 1.0
    pose_ema_alpha: float = 0.2

    # Detection quality
    blur_sharpness_threshold: float = 45.0

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

    # Global landmark origin
    origin_box: OriginBoxConfig = field(default_factory=OriginBoxConfig)


@dataclass
class VslamStatus:
    tracking_ok: bool = False
    pose_valid: bool = False
    sharpness: float = 0.0
    contrast: float = 0.0
    confidence: float = 0.0
    high_confidence: bool = False
    blurry: bool = True

    gate_reason: str = "startup"
    origin_area_ratio: float = 0.0

    origin_visible: bool = False
    origin_pose_used: bool = False
    origin_marker_id: int = -1
    origin_range_world: float = 0.0
    origin_bearing_cam_deg: float = 0.0
    origin_bearing_body_deg: float = 0.0
    origin_pan_error_deg: float = 0.0
    origin_pan_suggest_deg: float = 0.0
    origin_pose_x: float = 0.0
    origin_pose_y: float = 0.0
    origin_pose_theta: float = 0.0

@dataclass
class OriginBoxObservation:
    marker_id: int
    corners_px: np.ndarray
    center_px: Tuple[float, float]
    area_ratio: float
    range_world: float
    bearing_cam_deg: float
    bearing_body_deg: float
    pan_error_deg: float
    pan_suggest_deg: float
    pose_world: Pose


class MonocularVSLAM:
    """
    Marker-based global-origin localizer.

    The global pose comes from a known ArUco marker on the origin box.
    The caller may optionally pass the current camera pan angle so body
    bearing and suggested pan corrections stay consistent with vehicle pose.
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
        self.dist_coeffs = intr.dist()
        self.shared_map = shared_map
        self.car_id = car_id

        self._pose_filt: Optional[Pose] = None
        self._status = VslamStatus()

        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_lut_value: float = -1.0

        self._Tcw = np.eye(4, dtype=np.float64)

        self._dbg_frame_bgr: Optional[np.ndarray] = None
        self._origin_obs: Optional[OriginBoxObservation] = None
        self._aruco_dict = None
        self._aruco_params = None
        self._aruco_detector = None

        self._init_origin_box_detector()

        self._publish_pose()

    # ---------------- Public diagnostics ----------------

    def get_debug_frame(self) -> Optional[np.ndarray]:
        return None if self._dbg_frame_bgr is None else self._dbg_frame_bgr.copy()

    def get_status(self) -> VslamStatus:
        s = self._status
        return VslamStatus(
            tracking_ok=bool(s.tracking_ok),
            pose_valid=bool(s.pose_valid),
            sharpness=float(s.sharpness),
            contrast=float(s.contrast),
            confidence=float(s.confidence),
            high_confidence=bool(s.high_confidence),
            blurry=bool(s.blurry),
            gate_reason=str(s.gate_reason),
            origin_area_ratio=float(s.origin_area_ratio),
            origin_visible=bool(s.origin_visible),
            origin_pose_used=bool(s.origin_pose_used),
            origin_marker_id=int(s.origin_marker_id),
            origin_range_world=float(s.origin_range_world),
            origin_bearing_cam_deg=float(s.origin_bearing_cam_deg),
            origin_bearing_body_deg=float(s.origin_bearing_body_deg),
            origin_pan_error_deg=float(s.origin_pan_error_deg),
            origin_pan_suggest_deg=float(s.origin_pan_suggest_deg),
            origin_pose_x=float(s.origin_pose_x),
            origin_pose_y=float(s.origin_pose_y),
            origin_pose_theta=float(s.origin_pose_theta),
        )

    def slam_quality(self, *, min_confidence: float = 0.55) -> Tuple[float, bool, bool, bool, float]:
        s = self.get_status()
        conf = float(s.confidence)
        high_conf = bool(s.high_confidence and conf >= float(min_confidence))
        return conf, high_conf, bool(s.blurry), bool(s.pose_valid), float(s.contrast)

    # ---------------- Main update ----------------

    def tick(
        self,
        frame_rgb_or_bgr: np.ndarray,
        camera_pan_deg: Optional[float] = None,
    ) -> Optional[Pose]:
        gray = self._to_gray(frame_rgb_or_bgr)
        camera_pan_deg = 0.0 if camera_pan_deg is None else float(camera_pan_deg)
        origin_obs = self._detect_origin_box(gray, camera_pan_deg=camera_pan_deg)

        sharpness = self._frame_sharpness(gray)
        contrast = self._frame_contrast(gray)

        if self.cfg.debug_draw_frame:
            self._dbg_frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            self._dbg_frame_bgr = None

        if origin_obs is None:
            self._record_origin_status(None, False)
            self._status = VslamStatus(
                tracking_ok=False,
                pose_valid=False,
                sharpness=sharpness,
                contrast=contrast,
                confidence=0.0,
                high_confidence=False,
                blurry=bool(sharpness < float(self.cfg.blur_sharpness_threshold)),
                gate_reason="origin_not_visible",
                origin_area_ratio=0.0,
            )
            self._publish_pose(update_filter=False)
            self._annotate_debug_frames()
            return self.shared_map.poses.get(self.car_id)

        self._Tcw = self._planar_pose_to_Tcw(origin_obs.pose_world)
        self._record_origin_status(origin_obs, True)
        self._status = VslamStatus(
            tracking_ok=True,
            pose_valid=True,
            sharpness=sharpness,
            contrast=contrast,
            confidence=1.0,
            high_confidence=True,
            blurry=bool(sharpness < float(self.cfg.blur_sharpness_threshold)),
            gate_reason="origin_visible",
            origin_area_ratio=float(origin_obs.area_ratio),
        )
        self._publish_pose()
        self._annotate_debug_frames()
        return self.shared_map.poses.get(self.car_id)

    # ---------------- Pose publishing ----------------

    def _publish_pose(self, *, update_filter: bool = True) -> None:
        raw_pose = self._planar_pose_from_Tcw(self._Tcw)

        if update_filter or self._pose_filt is None:
            alpha = float(self.cfg.pose_ema_alpha)
            if alpha <= 0.0 or self._pose_filt is None:
                self._pose_filt = raw_pose
            else:
                # Smooth marker jitter.
                xf = (1.0 - alpha) * self._pose_filt.x + alpha * raw_pose.x
                yf = (1.0 - alpha) * self._pose_filt.y + alpha * raw_pose.y

                c0, s0 = math.cos(self._pose_filt.theta), math.sin(self._pose_filt.theta)
                c1, s1 = math.cos(raw_pose.theta), math.sin(raw_pose.theta)
                cf = (1.0 - alpha) * c0 + alpha * c1
                sf = (1.0 - alpha) * s0 + alpha * s1
                thf = math.atan2(sf, cf)

                self._pose_filt = Pose(x=float(xf), y=float(yf), theta=float(thf))

        self.shared_map.set_pose(self.car_id, self._pose_filt)

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

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _wrap_deg(a: float) -> float:
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def _init_origin_box_detector(self) -> None:
        self._aruco_dict = None
        self._aruco_params = None
        self._aruco_detector = None

        cfg = self.cfg.origin_box
        if not cfg.enabled:
            return

        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            return

        dict_id = getattr(aruco, str(cfg.dictionary_name), None)
        if dict_id is None:
            return

        try:
            self._aruco_dict = aruco.getPredefinedDictionary(dict_id)
        except Exception:
            self._aruco_dict = None
            return

        if hasattr(aruco, "DetectorParameters"):
            try:
                self._aruco_params = aruco.DetectorParameters()
            except Exception:
                self._aruco_params = None

        if self._aruco_params is None and hasattr(aruco, "DetectorParameters_create"):
            try:
                self._aruco_params = aruco.DetectorParameters_create()
            except Exception:
                self._aruco_params = None

        if hasattr(aruco, "ArucoDetector"):
            try:
                self._aruco_detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
            except Exception:
                self._aruco_detector = None

    def _detect_origin_box(self, gray: np.ndarray, *, camera_pan_deg: float) -> Optional[OriginBoxObservation]:
        cfg = self.cfg.origin_box
        self._origin_obs = None
        if not cfg.enabled or self._aruco_dict is None or gray.size == 0:
            return None

        corners: List[np.ndarray]
        ids: Optional[np.ndarray]
        try:
            if self._aruco_detector is not None:
                corners, ids, _rejected = self._aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _rejected = cv2.aruco.detectMarkers(  # type: ignore[attr-defined]
                    gray,
                    self._aruco_dict,
                    parameters=self._aruco_params,
                )
        except Exception:
            return None

        if ids is None or len(ids) == 0 or not corners:
            return None

        h, w = gray.shape[:2]
        image_area = float(max(1, h * w))
        best_obs: Optional[OriginBoxObservation] = None
        best_area = -1.0

        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            if int(marker_id) != int(cfg.marker_id):
                continue

            pts = np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2)
            if pts.shape[0] != 4:
                continue

            area_ratio = float(abs(cv2.contourArea(pts.astype(np.float32))) / image_area)
            if area_ratio < float(cfg.min_area_ratio):
                continue

            pose_solution = self._estimate_origin_marker_pose(pts)
            if pose_solution is None:
                continue

            rvec, tvec = pose_solution
            obs = self._origin_observation_from_marker(
                corners_px=pts,
                rvec=rvec,
                tvec=tvec,
                image_shape=gray.shape,
                camera_pan_deg=camera_pan_deg,
            )
            if obs is None:
                continue

            if area_ratio > best_area:
                best_obs = obs
                best_area = area_ratio

        self._origin_obs = best_obs
        return best_obs

    def _estimate_origin_marker_pose(self, corners_px: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cfg = self.cfg.origin_box
        half = 0.5 * float(cfg.marker_size_world)
        obj_pts = np.array(
            [
                [-half, +half, 0.0],
                [+half, +half, 0.0],
                [+half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        img_pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 1, 2)

        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                self.K,
                self.dist_coeffs,
                flags=flags,
            )
        except cv2.error:
            return None

        if not ok:
            return None

        return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)

    def _origin_observation_from_marker(
        self,
        *,
        corners_px: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        image_shape: Tuple[int, int],
        camera_pan_deg: float,
    ) -> Optional[OriginBoxObservation]:
        cfg = self.cfg.origin_box

        tz = float(tvec[2])
        if not np.isfinite(tz) or tz <= 1e-6:
            return None

        try:
            R_cm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        except cv2.error:
            return None

        # Invert marker pose.
        R_mc = R_cm.T
        cam_pos_marker = (-R_mc @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)).reshape(3)

        range_world = float(math.hypot(float(cam_pos_marker[2]), float(cam_pos_marker[0])))
        if not np.isfinite(range_world) or range_world <= 1e-6:
            return None
        if range_world > float(cfg.max_range_world):
            return None

        marker_yaw = math.radians(float(cfg.marker_world_yaw_deg))
        cam_forward_local = float(cam_pos_marker[2])
        cam_left_local = float(-cam_pos_marker[0])
        # Marker frame to world.
        cam_x_world = float(cfg.marker_world_x + cam_forward_local * math.cos(marker_yaw) - cam_left_local * math.sin(marker_yaw))
        cam_y_world = float(cfg.marker_world_y + cam_forward_local * math.sin(marker_yaw) + cam_left_local * math.cos(marker_yaw))

        cam_fwd_marker = R_mc[:, 2]
        cam_heading_local = float(math.atan2(float(-cam_fwd_marker[0]), float(cam_fwd_marker[2])))
        camera_heading_world = self._wrap_angle(marker_yaw + cam_heading_local)

        pan_body = math.radians(float(cfg.camera_pan_sign) * float(camera_pan_deg))
        # Remove camera pan.
        car_theta_world = self._wrap_angle(camera_heading_world - pan_body)

        mount_forward = float(cfg.camera_mount_forward)
        mount_left = float(cfg.camera_mount_left)
        # Shift to car center.
        car_x_world = float(cam_x_world - mount_forward * math.cos(car_theta_world) + mount_left * math.sin(car_theta_world))
        car_y_world = float(cam_y_world - mount_forward * math.sin(car_theta_world) - mount_left * math.cos(car_theta_world))

        center = np.mean(np.asarray(corners_px, dtype=np.float64).reshape(-1, 2), axis=0)
        cx_ref = float(self.K[0, 2]) if self.K.shape == (3, 3) else (float(image_shape[1]) * 0.5)
        fx_ref = max(1e-6, float(self.K[0, 0]))

        bearing_cam_deg = float(math.degrees(math.atan2(float(-tvec[0]), tz)))
        bearing_body_deg = float(self._wrap_deg(float(cfg.camera_pan_sign) * float(camera_pan_deg) + bearing_cam_deg))
        pan_error_deg = float(math.degrees(math.atan2(cx_ref - float(center[0]), fx_ref)))
        # Clamp pan command.
        pan_suggest_deg = float(
            np.clip(
                float(cfg.pan_track_gain) * pan_error_deg,
                -float(cfg.max_pan_step_deg),
                float(cfg.max_pan_step_deg),
            )
        )

        return OriginBoxObservation(
            marker_id=int(cfg.marker_id),
            corners_px=np.asarray(corners_px, dtype=np.float32).reshape(-1, 2),
            center_px=(float(center[0]), float(center[1])),
            area_ratio=float(abs(cv2.contourArea(np.asarray(corners_px, dtype=np.float32))) / max(1, image_shape[0] * image_shape[1])),
            range_world=range_world,
            bearing_cam_deg=bearing_cam_deg,
            bearing_body_deg=bearing_body_deg,
            pan_error_deg=pan_error_deg,
            pan_suggest_deg=pan_suggest_deg,
            pose_world=Pose(x=car_x_world, y=car_y_world, theta=car_theta_world),
        )

    def _record_origin_status(self, obs: Optional[OriginBoxObservation], used: bool) -> None:
        self._origin_obs = obs
        self._status.origin_visible = bool(obs is not None)
        self._status.origin_pose_used = bool(obs is not None and used)

        if obs is None:
            self._status.origin_marker_id = -1
            self._status.origin_range_world = 0.0
            self._status.origin_bearing_cam_deg = 0.0
            self._status.origin_bearing_body_deg = 0.0
            self._status.origin_pan_error_deg = 0.0
            self._status.origin_pan_suggest_deg = 0.0
            self._status.origin_pose_x = 0.0
            self._status.origin_pose_y = 0.0
            self._status.origin_pose_theta = 0.0
            return

        self._status.origin_marker_id = int(obs.marker_id)
        self._status.origin_range_world = float(obs.range_world)
        self._status.origin_bearing_cam_deg = float(obs.bearing_cam_deg)
        self._status.origin_bearing_body_deg = float(obs.bearing_body_deg)
        self._status.origin_pan_error_deg = float(obs.pan_error_deg)
        self._status.origin_pan_suggest_deg = float(obs.pan_suggest_deg)
        self._status.origin_pose_x = float(obs.pose_world.x)
        self._status.origin_pose_y = float(obs.pose_world.y)
        self._status.origin_pose_theta = float(obs.pose_world.theta)

    # ---------------- Status ----------------

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

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            gray = img
        else:
            color = img
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
            self._draw_origin_observation(self._dbg_frame_bgr)
            self._draw_debug_overlay(self._dbg_frame_bgr, lines)

    def _debug_overlay_lines(self) -> List[str]:
        s = self.get_status()
        filt_pose = self._pose_filt if self._pose_filt is not None else self._planar_pose_from_Tcw(self._Tcw)
        return [
            f"gate={s.gate_reason} origin={int(s.origin_visible)} used={int(s.origin_pose_used)} conf={s.confidence:.2f}",
            f"range={s.origin_range_world:.2f} bear_cam={s.origin_bearing_cam_deg:.1f} bear_body={s.origin_bearing_body_deg:.1f}",
            f"pan_err={s.origin_pan_error_deg:.1f} pan_suggest={s.origin_pan_suggest_deg:.1f}",
            f"sharp={s.sharpness:.1f} contrast={s.contrast:.1f} area={s.origin_area_ratio:.3f}",
            f"pose=({filt_pose.x:.2f},{filt_pose.y:.2f},{filt_pose.theta:.2f})",
        ]

    @staticmethod
    def _draw_debug_overlay(img: np.ndarray, lines: List[str]) -> None:
        y = 20
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    def _draw_origin_observation(self, img: np.ndarray) -> None:
        obs = self._origin_obs
        if img is None or obs is None:
            return

        pts_xy = np.asarray(obs.corners_px, dtype=np.int32).reshape(-1, 2)
        cx, cy = int(round(obs.center_px[0])), int(round(obs.center_px[1]))

        pts = pts_xy.reshape(-1, 1, 2)
        if pts.shape[0] >= 4:
            cv2.polylines(img, [pts], True, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.circle(img, (cx, cy), 4, (0, 200, 255), -1, cv2.LINE_AA)
