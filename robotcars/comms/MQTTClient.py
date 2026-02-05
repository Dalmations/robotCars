import paho.mqtt.client as mqtt
import json

class MQTTClient:

    def __init__(self, id, broker_ip, broker_port=1883):
        self.id = id
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.client = mqtt.Client(
            client_id=id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def start(self):
        self.client.connect(self.broker_ip, self.broker_port)
        self.client.loop_start()
    
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        print(f'Robot {self.id} connected')
        client.subscribe(f'command/{self.id}')
        client.subscribe(f'command/broadcast')

    def on_message(self, client, userdata, msg):
        decoded_msg = json.loads(msg.payload.decode())
        self.handle_message(client, msg.topic, decoded_msg)
    
    def handle_message(self, client, topic, msg:dict):
        pass

    def on_disconnect(self, client, userdata, reason_code, properties):
        print(f'Robot {self.id} disconnected')

    def publish_broadcast(self, msg:dict):
        self.client.publish(
            'command/broadcast',
            json.dumps(msg)
        )
    
    def publish_to_robot(self, target_id, msg:dict):
        self.client.publish(
            f'command/{target_id}',
            json.dumps(msg)
        )