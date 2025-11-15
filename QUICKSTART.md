# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start Kafka

```bash
./start_kafka.sh
# OR
docker-compose up -d
```

Wait ~10 seconds for Kafka to be ready.

## 3. Verify Setup

```bash
python verify_setup.py
```

## 4. Run a Simple Experiment

### Terminal 1: Start Producer
```bash
python producer.py --dataset electricity --stream-rate 0.1
```

### Terminal 2: Start Consumer
```bash
python consumer.py --topic ml-stream-electricity --task classification --model hoeffding_tree --log-file results/test.csv
```

## 5. Run Full Experiments

```bash
# Run all experiments (limited to 5000 instances each for speed)
python experiment_runner.py --all --max-instances 5000 --stream-rate 0.01

# Generate comparison report
python experiment_runner.py --compare-only
```

## Common Commands

### Stream Electricity Dataset
```bash
python producer.py --dataset electricity --stream-rate 0.1 --inject-drift
```

### Stream Bike Dataset
```bash
python producer.py --dataset bike --stream-rate 0.1 --inject-drift
```

### Classification Models
```bash
python consumer.py --topic ml-stream-electricity --task classification --model hoeffding_tree
python consumer.py --topic ml-stream-electricity --task classification --model arf
python consumer.py --topic ml-stream-electricity --task classification --model knn
```

### Regression Models
```bash
python consumer.py --topic ml-stream-bike --task regression --model fimtdd
python consumer.py --topic ml-stream-bike --task regression --model arf
python consumer.py --topic ml-stream-bike --task regression --model knn
```

## Stop Services

```bash
./stop_kafka.sh
# OR
docker-compose down
```

## Troubleshooting

1. **Kafka connection error**: Make sure Docker containers are running
   ```bash
   docker-compose ps
   ```

2. **Port already in use**: Stop other Kafka instances or change ports in `docker-compose.yml`

3. **Dataset download slow**: First run will download datasets (~50MB). Be patient.

4. **Consumer timeout**: Increase `consumer_timeout_ms` in `consumer.py` if producer is slow

