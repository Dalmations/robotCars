from MQTTClient import MQTTClient
from typing import override

class LeaderClient(MQTTClient):
    @override
    def handle_message(self, client, topic, msg):
        print(f'{self.id} received: {topic} - {msg}')

lc = LeaderClient('Leader', "localhost")
lc.start()

while True:
    inp = input('Enter a message to broadcast\n')
    lc.publish_broadcast({'message':inp})
