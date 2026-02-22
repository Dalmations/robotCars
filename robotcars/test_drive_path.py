from __future__ import annotations

import time

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.camera_input import PiCarXCamera, CameraConfig
from coordination.shared_map import SharedMap
from car_tools.obstacle_detection import VslamObstacleDetector, CameraIntrinsics


def main() -> None:
    shared_map = SharedMap()

    cam = PiCarXCamera(CameraConfig(
        display_web=False,      # turn on if you want browser view
        display_local=False,
        obstacle_color="red",
        frame_size=(640, 480),
    ))
    cam.start()

    intr = CameraIntrinsics(
        fx=628.0, fy=642.0,
        cx=320.0, cy=240.0
    )
    det = VslamObstacleDetector(intr=intr, shared_map=shared_map, car_id=0)

    motor = MotorController(MotorConfig(
        speed=35,              # start low
        step_seconds=0.0,      # not used here
    ))

    steer_deg = 18.0          # circle tightness: 10..25 deg usually
    duration_s = 12.0         # how long to drive

    print("Starting circle. Ctrl+C to stop.")
    motor.set_steering(steer_deg)

    t0 = time.time()
    last_print = 0.0

    try:
        motor.px.forward(motor.cfg.speed)

        while True:
            now = time.time()
            if now - t0 >= duration_s:
                break

            frame = cam.read()
            if frame is not None:
                pose = det.tick(frame)
                if pose is not None and (now - last_print) > 0.5:
                    last_print = now
                    print(
                        f"pose: x={pose.x:.2f} y={pose.y:.2f} theta={pose.theta:.2f}  "
                        f"obs={len(shared_map.obstacles)} pts={len(shared_map.map_points)}"
                    )

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()
        cam.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()