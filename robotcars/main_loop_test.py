import queue
import os

from car_tools.movement import plan_formation
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig
# from speech_input.test_phrase_to_bucket import classify
from speech_input.classification_MVP import classify
from picarx.stt import Vosk


from comms.MQTTClient import MQTTClient
from typing_extensions import override
class FollowerClient(MQTTClient):
    def __init__(self, id, broker_ip, broker_port=1883):
        super().__init__(id, broker_ip, broker_port)
        self.message_q = queue.Queue()

    @override
    def handle_message(self, client, topic, msg):
        print(f'{self.id} received: {topic} - {msg}')
        self.message_q.put(msg)

IDENTITY = os.uname().nodename
PARAMS = {
    'strawberry': {
        'circle': {
            'wheelbase':2.0
        }
    },
    'blueberry': {
        'circle': {
            'wheelbase':2.5
        }
    }
}

def follower_main():
    fc = FollowerClient(IDENTITY, "10.229.180.83")
    fc.start() 
    motor = MotorController(MotorConfig(speed=80))
    follower = PurePursuitFollower(motor, FollowerConfig(), PARAMS[IDENTITY])
    while True:
        try:
            msg = fc.message_q.get()
            shape = msg['message']
            path = plan_formation(shape)
            follower.update_params(shape)
            follower.follow(path)
            motor.stop()
        finally:
            motor.stop()

def leader_main():
    fc = FollowerClient(IDENTITY, 'localhost')
    fc.start()
    motor = MotorController(MotorConfig(speed=80))
    follower = PurePursuitFollower(motor, FollowerConfig(), PARAMS[IDENTITY])
    vosk = Vosk(language="en-us")
    while True:
        try:
            print('Listening')
            phrase = vosk.listen(stream=False)
            print(phrase)
            if not phrase:
               continue
            shape = classify(phrase)
            fc.publish_broadcast({'message':shape})
            fc.message_q.get()
            path = plan_formation(shape)
            follower.update_params(shape)
            follower.follow(path)
            motor.stop()
        finally:
            motor.stop()

if IDENTITY=='strawberry':
    leader_main()
else:
    follower_main()
