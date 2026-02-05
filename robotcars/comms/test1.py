import paho.mqtt.client as mqtt
import time
import os

BROKER = "localhost"
TOPIC = "stress/test"
rtts = []
def on_message(client, userdata, msg):
    rec_time = time.time_ns()
    sent_time = int(msg.payload.decode('ascii'))
    rtts.append(rec_time-sent_time)
    print((rec_time-sent_time)/1e6)


payload_size = 100_000  # 100 KB
payload = os.urandom(payload_size)

client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()
client.subscribe('leader')
count = 0
start = time.time()

for i in range(100):
    payload = str(time.time_ns()).encode('ascii')
    client.publish(TOPIC, payload)
    time.sleep(0.5)

print(f'average RTT: {(sum(rtts)/len(rtts))/1e6}')


#while time.time() - start < 30:  # run for 10 seconds
#    client.publish(TOPIC, payload, qos=0)
#    count += 1
#    time.sleep(1/3000)

#duration = time.time() - start
#print(f"Sent {count} messages")
#print(f'Messages per second: {count/duration}')
#print(f"Throughput ≈ {(count * payload_size) / duration / 1e6:.2f} MB/s")
