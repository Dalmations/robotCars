import queue
import os
import threading

from car_tools.movement import plan_formation
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig, PathFollower, PathFollowerConfig

from coordination.shared_map import SharedMap
from main import (
    build_shared_map,
    build_follower,
    build_loop_config,
)

from test_drive_path import LoopConfig, drive_path
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

def follower_main():
    fc = FollowerClient(IDENTITY, "10.229.180.83")
    fc.start() 
    motor = MotorController(MotorConfig(speed=80))
    follower = PurePursuitFollower(motor, FollowerConfig(), PARAMS[IDENTITY])
    
    pathFollower = build_follower(motor)
    shared_map = build_shared_map()
    loop_cfg = build_loop_config()
    while True:
        try:
            msg = fc.message_q.get()
            fc.busy.set()
            shape = msg['message']
            path = plan_formation(shared_map, shape)
            follower.update_params(shape)
            # follower.follow(path)
            # TODO: Try drive_path() with circle and merge PurePursuitFollower and PathFollower in picarx_path_follower.py
            ok = drive_path(
                path,
                shared_map=shared_map,
                follower=pathFollower,
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
    follower = PurePursuitFollower(motor, FollowerConfig(), PARAMS[IDENTITY])
    vosk = Vosk(language="en-us")

    pathFollower = build_follower(motor)
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
            # follower.follow(path)
            # TODO: Try drive_path() with circle and merge PurePursuitFollower and PathFollower in picarx_path_follower.py
            ok = drive_path(
                path,
                shared_map=shared_map,
                follower=pathFollower,
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
