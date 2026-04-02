from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from model import Path, Pose
from coordination.shared_map import SharedMap
import time
from car_tools.motor_controller import MotorController
from typing import Callable, Optional
from coordination.shared_map import SharedMap


TickCallback = Callable[[float, float, float], None]  # x, y, yaw (grid units, radians) Callback for PurePursuit
if TYPE_CHECKING:
    from car_tools.motor_controller import MotorController

@dataclass
class FollowerConfig:
    """
    Pure Pursuit parameters in *grid units* (NOT meters).

    - lookahead: how far ahead (in grid units) to chase a point
    - wheelbase: effective wheelbase in grid units (tune)
    - v: estimated speed in grid units / second (tune)
    - dt: control tick duration in seconds
    """
    lookahead: float = 3.0              # grid units
    wheelbase: float = 2.0              # grid units 
    v: float = 7.75                      # grid units per second
    dt: float = 0.10                    # seconds per control update

    max_steer_deg: float = 35.0         # clamp to match MotorConfig.max_steer_deg
    goal_tolerance: float = 3.0         # grid units to final waypoint
    speed_cmd: int = 100                 # motor_controller speed (0..100)

    safe_dist: float = 40.0        # cm: Start slowing down
    stop_dist: float = 10.0        # cm: Complete stop

    steer_sign: float = 1.0                    # follower steering sign

    steer_alpha: float = 0.25                  # steering blend factor
    steer_deadband_deg: float = 2.0            # ignore tiny steer
    steer_rate_limit_deg_per_tick: float = 12.0  # max steer change
    heading_lookahead_gain: float = 0.012
    heading_lookahead_max_scale: float = 1.75

    dock_distance_grid: float = 12.0           # near goal threshold
    dock_min_lookahead_grid: float = 6.0       # minimum dock lookahead
    straight: bool = False

    pivot_turn_heading_deg: float = 45.0
    pivot_turn_exit_deg: float = 20.0
    pivot_turn_steer_deg: float = 30.0
    pivot_turn_settle_s: float = 0.12
    pivot_turn_deg_per_s: float = 12.0
    pivot_turn_cells_per_deg: float = 0.05


class PurePursuitFollower:
    """
    Matlab-style Pure Pursuit in grid coordinates.

    IMPORTANT: This currently uses dead-reckoning pose updates based on an
    *estimated* v (grid units/sec) and the commanded steering angle.
    For real accuracy, later we will replace pose updates with camera/odometry.
    """

    def __init__(self, motor: MotorController, cfg: Optional[FollowerConfig] = None, params: dict={}):
        self.motor = motor
        self.cfg = cfg or FollowerConfig()
        self.params = params
        self._filtered_steer_deg = 0.0
        self.shape = None
        

    # def follow(self, path: Path, start_pose: Optional[Pose] = None) -> None:
    # Use for circle
    def follow(self, path: Path, start_pose: Optional[Pose] = None, on_tick: Optional[TickCallback] = None) -> None:
        wps = path.waypoints
        if len(wps) < 2:
            return

        # Waypoints as (x, y) in grid units
        pts: List[Tuple[float, float]] = [(wp.x, wp.y) for wp in wps]
        goal = pts[-1]

        # Init pose
        if start_pose is None:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            yaw0 = math.atan2(y1 - y0, x1 - x0)
            pose = Pose(x0, y0, yaw0)
        else:
            pose = start_pose.copy()

        # Configure speed (open loop)
        self.motor.set_speed(self.cfg.speed_cmd)
        print(f'Goal: {goal}')
        f = open('poses.txt', 'w')
        loop_start = time.perf_counter()
        try:
            while True:
                f.write(f'{pose.x, pose.y}  {self._dist((pose.x, pose.y), goal)}\n')
                print(f'{pose.x, pose.y}  {self._dist((pose.x, pose.y), goal)}\n')
                if on_tick is not None:
                    on_tick(pose.x, pose.y, pose.theta)
                
                # break from following if within tolerance of goal and loop has been running for more than 2 seconds
                # without the time condition, this breaks when start and goal positions are the same
                if self._dist((pose.x, pose.y), goal) <= self.cfg.goal_tolerance and time.perf_counter()-loop_start > 2:
                    break

                scalar = self._get_speed_scaler()
                scaled_speed = int(self.cfg.speed_cmd*scalar)
                self.motor.set_speed(scaled_speed)

                if self.motor.cfg.speed > 0:
                    target = self._lookahead_point(pose, pts, self.cfg.lookahead)
                    steer_deg = self._pure_pursuit_steer_deg(pose, target)

                    # Clamp to safe steering
                    steer_deg = max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, steer_deg))

                    # Command hardware
                    self.motor.set_steering(steer_deg)
                    self.motor.forward_for(self.cfg.dt)

                    # Dead-reckoning pose update (grid bicycle model)
                    pose = self._update_pose(pose, steer_deg, scaled_speed/12.9, self.cfg.wheelbase, self.cfg.dt)
                else:
                    time.sleep(self.cfg.dt)

        finally:
            f.close()
            self.motor.set_steering(0.0)
            self.motor.mark_reached()

    def _get_speed_scaler(self) -> float:
            """
            Returns a multiplier between 0.0 and 1.0 based on ultrasonic data.
            """
            distance = self.motor.px.get_distance()
            if distance < 0:
                 distance = 100
            #print(distance)
            if distance > self.cfg.safe_dist:
                return 1.0
            if distance <= self.cfg.stop_dist:
                return 0.0
            # Linear interpolation: (dist - stop) / (safe - stop)
            return (distance - self.cfg.stop_dist) / (self.cfg.safe_dist - self.cfg.stop_dist)
    
    def update_params(self, shape):
        self.shape = shape
        if shape in self.params:
            self.cfg.wheelbase = self.params[shape]['wheelbase']
            self.cfg.pivot_turn_heading_deg = self.params[shape]['pivot_turn_heading_deg']
            self.cfg.straight = self.params[shape]['straight']
            self.cfg.pivot_turn_deg_per_s = self.params[shape]['pivot_turn_deg_per_s']

    # -------------------------
    # Pure Pursuit math
    # -------------------------
    def _pure_pursuit_steer_deg(self, pose: Pose, target: Tuple[float, float]) -> float:
        """
        Pure Pursuit steering:
          alpha = angle between heading and target direction
          delta = atan(2L*sin(alpha)/Ld)

        Returns steering in degrees (for PiCar-X servo).
        """
        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y

        # Target heading in global
        target_heading = math.atan2(dy, dx)
        alpha = self._wrap_angle(target_heading - pose.theta)

        Ld = max(1e-6, math.hypot(dx, dy))
        delta_rad = math.atan2(2.0 * self.cfg.wheelbase * math.sin(alpha), Ld)
        return math.degrees(delta_rad)

    def _lookahead_point(
        self,
        pose: Pose,
        pts: List[Tuple[float, float]],
        lookahead: float,
    ) -> Tuple[float, float]:
        """
        Pick the first waypoint at least lookahead distance from the current pose.
        If none, use final goal.
        """
        px, py = pose.x, pose.y
        for x, y in pts:
            if math.hypot(x - px, y - py) >= lookahead:
                return (x, y)
        return pts[-1]

    def _update_pose(self, pose: Pose, steer_deg: float, v: float, L: float, dt: float) -> Pose:
        """
        Bicycle model update in grid units:
          x += v cos(yaw) dt
          y += v sin(yaw) dt
          yaw += v/L * tan(delta) dt
        """
        delta = math.radians(steer_deg)
        x = pose.x + v * math.cos(pose.theta) * dt
        y = pose.y + v * math.sin(pose.theta) * dt
        yaw = pose.theta + (v / max(1e-6, L)) * math.tan(delta) * dt
        return Pose(x, y, self._wrap_angle(yaw))

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a
    
    def reset(self) -> None:
        self._filtered_steer_deg = 0.0
        if hasattr(self.motor, "reset_reached"):
            self.motor.reset_reached()

    def sync_to_motor_steering(self) -> None:
        self._filtered_steer_deg = float(self.motor.get_applied_steering_deg())
    
    def estimate_ackermann_yaw_delta(self, step_cells: float, steer_deg: float) -> float:
        steer_rad = math.radians(float(steer_deg))
        return float(step_cells) * math.tan(steer_rad) / self.cfg.wheelbase

    def _compute_lookahead(self, goal_distance: float) -> float:
        lookahead = float(self.cfg.lookahead)  # Start with path lookahead
        if goal_distance < self.cfg.dock_distance_grid:  # Use near-goal distance threshold
            return max(self.cfg.dock_min_lookahead_grid, min(lookahead, 0.8 * goal_distance))  # Use shorter dock lookahead
        return lookahead

    def _lookahead_target(self, path: Path, pose: Pose, lookahead_distance: float) -> Tuple[float, float]:
        waypoints = path.waypoints
        if not waypoints:
            return (pose.x, pose.y)
        if len(waypoints) == 1:
            return (waypoints[0].x, waypoints[0].y)

        px, py = pose.x, pose.y
        best_dist2 = float("inf")
        best_i = 0
        best_proj = (waypoints[0].x, waypoints[0].y)

        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i].x, waypoints[i].y
            x1, y1 = waypoints[i + 1].x, waypoints[i + 1].y
            dx = x1 - x0
            dy = y1 - y0
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                continue

            u = ((px - x0) * dx + (py - y0) * dy) / seg2
            u = max(0.0, min(1.0, float(u)))
            proj = (x0 + u * dx, y0 + u * dy)
            dist2 = (proj[0] - px) ** 2 + (proj[1] - py) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_proj = proj

        dist_acc = 0.0
        start_x, start_y = best_proj
        for i in range(best_i, len(waypoints) - 1):
            x0, y0 = (start_x, start_y) if i == best_i else (waypoints[i].x, waypoints[i].y)
            x1, y1 = waypoints[i + 1].x, waypoints[i + 1].y
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-9:
                continue

            if dist_acc + seg >= lookahead_distance:
                remaining = lookahead_distance - dist_acc
                u = float(remaining / seg)
                tx = x0 + u * (x1 - x0)
                ty = y0 + u * (y1 - y0)
                return (tx, ty)

            dist_acc += seg

        return (waypoints[-1].x, waypoints[-1].y)

    def _pure_pursuit_delta(self, pose: Pose, target_x: float, target_y: float) -> float:
        path_angle = math.atan2(target_y - pose.y, target_x - pose.x)
        alpha = self._wrap_angle(path_angle - pose.theta)

        lookahead_distance = math.hypot(target_x - pose.x, target_y - pose.y)
        wheelbase = max(1e-6, float(self.cfg.wheelbase))  # Use model wheelbase

        return math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead_distance)

    def tracking_geometry(self, path: Path, pose: Pose, goal_distance: float) -> Tuple[float, float, float]:
        lookahead = self._compute_lookahead(goal_distance)
        target_x, target_y = self._lookahead_target(path, pose, lookahead)
        heading_error_rad = self._wrap_angle(math.atan2(target_y - pose.y, target_x - pose.x) - pose.theta)
        heading_error_deg = math.degrees(heading_error_rad)

        if abs(heading_error_deg) > 20.0:
            if goal_distance > self.cfg.dock_distance_grid:
                lookahead_scale = min(
                    float(self.cfg.heading_lookahead_max_scale),
                    1.0 + float(self.cfg.heading_lookahead_gain) * (abs(heading_error_deg) - 20.0),
                )
                adjusted_lookahead = lookahead * lookahead_scale
            else:
                tighten_scale = max(0.55, 1.0 - 0.004 * (abs(heading_error_deg) - 20.0))
                adjusted_lookahead = max(float(self.cfg.dock_min_lookahead_grid), lookahead * tighten_scale)
            target_x, target_y = self._lookahead_target(path, pose, adjusted_lookahead)
            heading_error_rad = self._wrap_angle(math.atan2(target_y - pose.y, target_x - pose.x) - pose.theta)
            heading_error_deg = math.degrees(heading_error_rad)
        return target_x, target_y, heading_error_deg

    def tracking_command(
        self,
        path: Path,
        pose: Pose,
        goal_distance: float,
        *,
        steer_cap_deg: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Return target x/y, heading error in degrees, and filtered steer in degrees."""
        target_x, target_y, heading_error_deg = self.tracking_geometry(path, pose, goal_distance)
        delta = self._pure_pursuit_delta(pose, target_x=target_x, target_y=target_y)
        target_steer_deg = self._clamp_steer(math.degrees(delta) * float(self.cfg.steer_sign))  # Apply follower steering sign
        if steer_cap_deg is not None:
            steer_cap = min(abs(float(steer_cap_deg)), float(self.cfg.max_steer_deg))
            target_steer_deg = max(-steer_cap, min(steer_cap, target_steer_deg))
            filtered = self._filter_steering(target_steer_deg)
            filtered = max(-steer_cap, min(steer_cap, filtered))
            self._filtered_steer_deg = filtered
            return (
                target_x,
                target_y,
                heading_error_deg,
                filtered,
            )
        return (
            target_x,
            target_y,
            heading_error_deg,
            self._filter_steering(target_steer_deg),
        )

    def _filter_steering(self, target_deg: float) -> float:
        if abs(target_deg) < self.cfg.steer_deadband_deg:  # Ignore very small corrections
            target_deg = 0.0

        blended = (1.0 - self.cfg.steer_alpha) * self._filtered_steer_deg + self.cfg.steer_alpha * target_deg  # Blend toward new target
        delta = blended - self._filtered_steer_deg
        delta = max(-self.cfg.steer_rate_limit_deg_per_tick, min(self.cfg.steer_rate_limit_deg_per_tick, delta))  # Limit per-tick steering change

        self._filtered_steer_deg = self._clamp_steer(self._filtered_steer_deg + delta)
        return self._filtered_steer_deg

    def _clamp_steer(self, steer_deg: float) -> float:
        return max(-self.cfg.max_steer_deg, min(self.cfg.max_steer_deg, steer_deg))

    def integrate_dead_reckoning(
        self,
        shared_map: SharedMap,
        forward_step: float,
        yaw_delta: float,
    ) -> Pose:
        pose_world = shared_map.get_pose(frame="world")
        if pose_world is None:
            pose_world = Pose(0.0, 0.0, 0.0)
        step = float(forward_step)
        dtheta = float(yaw_delta)
        theta_mid = float(pose_world.theta) + 0.5 * dtheta
        next_pose = Pose(
            x=float(pose_world.x + step * math.cos(theta_mid)),
            y=float(pose_world.y + step * math.sin(theta_mid)),
            theta=float(self._wrap_angle(float(pose_world.theta) + dtheta)),
        )
        shared_map.set_pose(next_pose)
        return next_pose
    
    def estimate_step_cells_for_duration(
        self,
        duration_s: float,
        action_tick_s: float,
        speed: int,
        speed_ref: int,
    ) -> float:
        tick_count = float(duration_s) / float(max(1e-6, action_tick_s))
        ref_step = tick_count / 5.0
        speed_ref = max(1.0, float(speed_ref))
        speed_cmd = float(max(0, min(100, int(speed))))
        return ref_step * (speed_cmd / speed_ref)