from car_tools.camera_input import PiCarXCamera, CameraConfig
from car_tools.obstacle_detection import VslamObstacleDetector, CameraIntrinsics
from coordination.shared_map import SharedMap

shared_map = SharedMap()

cam = PiCarXCamera(CameraConfig(display_web=True, obstacle_color="red"))
cam.start()

intr = CameraIntrinsics(
    fx=628.0,
    fy=642.0,
    cx=cam.cfg.frame_size[0] / 2.0,  # 320 for 640x480
    cy=cam.cfg.frame_size[1] / 2.0,  # 240 for 640x480
)

det = VslamObstacleDetector(intr=intr, shared_map=shared_map, car_id=0)

try:
    while True:
        frame = cam.read_bgr()
        if frame is None:
            continue
        pose = det.tick(frame)
        # shared_map.poses[0] updates when pose is valid
        # shared_map.map_points grows
        # shared_map.obstacles fills
finally:
    cam.stop()
