import queue
import os
import threading

from car_tools.movement import plan_formation
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig

from coordination.shared_map import SharedMap
from main import (
    build_shared_map,
    build_loop_config,
)

from test_drive_path import LoopConfig, start_path
from speech_input.processor import handle_input
from picarx.stt import Vosk


from comms.MQTTClient import MQTTClient
from typing_extensions import override
class FollowerClient(MQTTClient):
    def __init__(self, id, broker_ip, broker_port=1883):
        super().__init__(id, broker_ip, broker_port)
        self.message_q = queue.Queue()
        self.busy = threading.Event()

    @override
    def handle_message(self, client, topic, msg):
        if self.busy.is_set():
            return
        print(f'{self.id} received: {topic} - {msg}')
        self.message_q.put(msg)

IDENTITY = os.uname().nodename
PARAMS = {
    'strawberry': {
        'circle': {
            'wheelbase':2.0,
            'pivot_turn_heading_deg': 90.0
        }
    },
    'blueberry': {
        'circle': {
            'wheelbase':2.5,
            'pivot_turn_heading_deg': 90.0
        }
    },
    'raspberry': {
        'circle': {
            'wheelbase':2.5,
            'pivot_turn_heading_deg': 90.0
        }
    }
}

def build_follower(motor: MotorController) -> PurePursuitFollower:
    follower = PurePursuitFollower(motor, FollowerConfig(
        lookahead=5.0,                            # pure pursuit lookahead distance
        wheelbase=1.0,                           # front to back wheel wheelbase
        goal_tolerance=0.6,                       # goal reached radius
        steer_sign=1.0,                           # follower steering sign
        steer_alpha=0.25,                         # steering smoother
        steer_deadband_deg=2.0,                   # ignore tiny steer changes
        steer_rate_limit_deg_per_tick=12.0,       # max steer change
        dock_distance_grid=8.0,                   # near goal threshold
        dock_min_lookahead_grid=1.5,              # minimum dock lookahead
    ), PARAMS[IDENTITY])
    return follower

def follower_main():
    fc = FollowerClient(IDENTITY, "10.229.180.83")
    fc.start() 
    motor = MotorController(MotorConfig(speed=80))
    follower = build_follower(motor)
    shared_map = build_shared_map()
    loop_cfg = build_loop_config()
    while True:
        try:
            msg = fc.message_q.get()
            fc.busy.set()
            shape = msg['message']
            path = plan_formation(shared_map, shape)
            follower.update_params(shape)
            start_path(
                path,
                shared_map=shared_map,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
                timeout_s=180.0,
            )
            motor.stop()
        finally:
            motor.stop()
            fc.busy.clear()

def leader_main():
    fc = FollowerClient(IDENTITY, 'localhost')
    fc.start()
    motor = MotorController(MotorConfig(speed=80))
    follower = build_follower(motor)
    vosk = Vosk(language="en-us")
    shared_map = build_shared_map()
    loop_cfg = build_loop_config()
    while True:
        try:
            print('Listening')
            phrase = vosk.listen(stream=False)
            print(phrase)
            if not phrase:
               continue
            shape, robots = handle_input(phrase)
            if not robots:
                fc.publish_broadcast({'message':shape})
            else:
                for robot in robots:
                    fc.publish_to_robot(robot, {'message':shape})
            try:
                fc.message_q.get(timeout=3.0)
            except Exception:
                continue
            path = plan_formation(shared_map, shape)
            follower.update_params(shape)
            start_path(
                path,
                shared_map=shared_map,
                follower=follower,
                motor=motor,
                loop_cfg=loop_cfg,
                timeout_s=180.0,
            )
            motor.stop()
        finally:
            motor.stop()

if IDENTITY=='strawberry':
    leader_main()
else:
    follower_main()
