import queue
import os
import threading

from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.oled_display import display_text
from car_tools.shape_paths import drive_in_shape

from speech_input.processor import handle_input
from speech_input.shapes import SHAPES
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

def follower_main():
    #fc = FollowerClient(IDENTITY, "192.168.4.1")
    #fc.start() 
    motor = MotorController()
    while True:
        try:
            #msg = fc.message_q.get()
            #fc.busy.set()
            i = input('1 for circle, 2 for square, 3 for hexagon')
            s = {'1':'circle', '2':'square','3':'hexagon'}
            #shape = msg['message']
            shape = s[i]
            drive_in_shape(shape, motor)
            motor.stop()
        finally:
            motor.stop()
            fc.busy.clear()

def leader_main():
    fc = FollowerClient(IDENTITY, 'localhost')
    fc.start()
    motor = MotorController()
    vosk = Vosk(language="en-us")
    display_text('Listening')
    while True:
        try:
            print('Listening')
            phrase = vosk.listen(stream=False)
            print(phrase)
            if not phrase:
               continue
            shape, robots = handle_input(phrase)
            if shape not in SHAPES:
                display_text(f'Try again! Received: {phrase}')
                continue
            else:
                display_text('processing ' + shape + '...')
            if not robots:
                fc.publish_broadcast({'message':shape})
            else:
                for robot in robots:
                    fc.publish_to_robot(robot, {'message':shape})
            try:
                fc.message_q.get(timeout=3.0)
            except Exception:
                continue
            drive_in_shape(shape, motor)
            display_text(shape + ' completed!')
            motor.stop()
        finally:
            motor.stop()

if IDENTITY=='lugnut':
    leader_main()
else:
    follower_main()
