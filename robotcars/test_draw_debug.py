# test_draw_debug.py
from __future__ import annotations

from coordination.shared_map import SharedMap
from car_tools.movement import MovementPlanner
from model import TargetPoint

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PathFollower, FollowerConfig

from car_tools.draw_debug import TurtleObstacleGoalDrawer
from coordination.shared_map_draw_debug import rasterize_segments_to_grid


def main() -> None:
    # Draw obstacles + goal
    drawer = TurtleObstacleGoalDrawer(world_size=(50, 50), pixels_per_cell=12)
    segments, goal = drawer.run()

    if goal is None:
        print("No goal set. Exiting.")
        return

    # Load obstacles into shared map as an occupancy grid
    obstacle_grid = rasterize_segments_to_grid(segments, grid_size=(50, 50), thickness_cells=1)

    shared_map = SharedMap()
    shared_map.set_static_occupancy_grid(obstacle_grid)

    # Plan to goal (grid coords)
    planner = MovementPlanner(planning_cfg=None, world_size=(50, 50))

    target = TargetPoint(x=float(round(goal[0])), y=float(round(goal[1])))

    motor = MotorController(MotorConfig(speed=80, step_seconds=0.18))
    follower = PathFollower(motor, FollowerConfig(heading_to_steer_gain=22.0))

    try:
        path = planner.plan_to_target(target=target, shared_map=shared_map)
        print(f"Planned path waypoints: {len(path.waypoints)}")
        follower.follow(path)
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
