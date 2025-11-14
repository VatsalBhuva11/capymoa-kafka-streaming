from kafka import KafkaProducer
import json, time, random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

count = 0
drift_phase = 0

while True:
    # Phase 1: y = 2x + noise (initial concept)
    # Phase 2: y = -3x + noise (drifted concept)
    if count == 200:  # Trigger drift after 200 messages
        drift_phase = 1
        print("Concept drift introduced!")

    x = random.uniform(0, 1)
    noise = random.gauss(0, 0.1)
    if drift_phase == 0:
        y = 2 * x + random.gauss(0, 0.05)
    else:
        y = -5 * x + random.gauss(0, 0.05)

    msg = {'x': x, 'y': y}
    producer.send('stream-data', msg)
    print(f"Sent: {msg}")
    count += 1
    time.sleep(0.2)
