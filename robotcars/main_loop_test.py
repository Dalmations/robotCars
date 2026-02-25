import queue

from car_tools.movement import MovementPlanner
from car_tools.motor_controller import MotorController, MotorConfig
from car_tools.picarx_path_follower import PurePursuitFollower, FollowerConfig


from comms.MQTTClient import MQTTClient
from typing import override
class FollowerClient(MQTTClient):
    def __init__(self, id, broker_ip, broker_port=1883):
        super().__init__(id, broker_ip, broker_port)
        self.message_q = queue.Queue()

    @override
    def handle_message(self, client, topic, msg):
        print(f'{self.id} received: {topic} - {msg}')
        self.message_q.put(msg)


def main():
    fc = FollowerClient('Robot1', "10.183.37.93")
    fc.start() 
    planner = MovementPlanner()
    motor = MotorController(MotorConfig(speed=80))
    follower = PurePursuitFollower(motor, FollowerConfig())
    while True:
        try:
            msg = fc.message_q.get()
            # msg = {'message':'circle'}
            path = planner.plan_formation(msg['message'])
            follower.follow(path)
            motor.stop()
        finally:
            motor.stop()

main()