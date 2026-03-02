from car_tools.camera_input import PiCarXCamera, CameraConfig
from car_tools.obstacle_detection import MonocularVSLAM, CameraIntrinsics, VslamConfig
from coordination.shared_map import SharedMap

shared_map = SharedMap()

cam = PiCarXCamera(CameraConfig(display_web=False))
cam.start()

intr = CameraIntrinsics(
    fx=628.0,
    fy=642.0,
    cx=cam.cfg.frame_size[0] / 2.0,  # 320 for 640x480
    cy=cam.cfg.frame_size[1] / 2.0,  # 240 for 640x480
)

cfg = VslamConfig(
    debug_draw_keypoints=True,
    input_color_order="rgb",
)

slam = MonocularVSLAM(intr=intr, shared_map=shared_map, car_id=0, cfg=cfg)

# shared_map.poses[0] updates when pose is valid
# shared_map.map_points grows
# shared_map.obstacles fills
try:
    while True:
        frame = cam.read()
        if frame is None:
            continue
        pose = slam.tick(frame)
        if pose is not None:
            print("pose:", pose.x, pose.y, pose.theta, "obstacles:", len(shared_map.obstacles), "pts:", len(shared_map.map_points))
finally:
    cam.stop()
