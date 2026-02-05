from MQTTClient import MQTTClient
from typing import override
import time

class FollowerClient(MQTTClient):
    @override
    def handle_message(self, client, topic, msg):
        print(f'{self.id} received: {topic} - {msg}')
        # self.publish_to_robot('leader', msg)

fc = FollowerClient('Robot1', "10.183.37.93")
fc.start()

while True:
    fc.publish_to_robot('Leader', {'heartbeat':time.time()})
    time.sleep(1)