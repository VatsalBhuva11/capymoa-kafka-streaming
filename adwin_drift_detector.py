from kafka import KafkaConsumer
from sklearn.linear_model import SGDRegressor
from river.drift import ADWIN
import numpy as np, json, time

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'stream-data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Initialize model and drift detector
model = SGDRegressor(random_state=42, learning_rate='constant', eta0=0.0001)
adwin = ADWIN(delta=0.001)
trained = False

print("Listening to stream-data topic...")

for msg in consumer:
    data = msg.value
    X = np.array([[data['x']]])
    y = np.array([data['y']])

    if not trained:
        model.partial_fit(X, y)
        trained = True
        continue

    pred = model.predict(X)
    error = abs(pred - y)[0]

    # Update ADWIN
    adwin.update(error)

    # Check if drift was detected
    if adwin.drift_detected:
        print("Drift detected! Reinitializing model.\n")
        model = SGDRegressor(random_state=42, learning_rate='constant', eta0=0.0001)
        adwin = ADWIN(delta=0.002)  # reset detector window
        trained = False
        continue

    # Incremental training
    model.partial_fit(X, y)

    print(f"X={X[0][0]:.3f}, y={y[0]:.3f}, pred={pred[0]:.3f}, error={error:.3f}")

    time.sleep(0.1)
