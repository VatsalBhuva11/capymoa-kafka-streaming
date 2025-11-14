# Quick Start Guide

## Prerequisites Setup

### 1. Install Docker

Install Docker and Docker Compose:
- **Linux**: Follow [Docker installation guide](https://docs.docker.com/engine/install/)
- **macOS**: Install [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)
- **Windows**: Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

Verify installation:
```bash
docker --version
docker compose version  # or docker-compose --version
```

### 2. Start Kafka and Zookeeper with Docker

**Option A: Using helper script (recommended)**
```bash
chmod +x start_kafka.sh
./start_kafka.sh
```

**Option B: Using Docker Compose directly**
```bash
docker compose up -d

# Wait a few seconds for services to start, then create topic
docker exec -it kafka bash -c "kafka-topics --create \
  --topic stream-data \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1"

```

**Stop services:**
```bash
./stop_kafka.sh
# or
docker compose down
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: CapyMOA requires Java. If you see JPype errors:
```bash
# Linux
sudo apt-get install default-jdk

# macOS  
brew install openjdk

# Then reinstall
pip install --upgrade --force-reinstall capymoa jpype1
```

### 4. Verify Setup

```bash
python verify_setup.py
```

## Running the Pipeline

### Basic Usage

**Single dataset with dashboard:**
```bash
python main.py --mode single --dataset electricity --model hoeffding_tree --dashboard
```

**All datasets:**
```bash
python main.py --mode all
```

**Run experiments:**
```bash
python main.py --mode experiments
```

### Examples

**Classification example:**
```bash
python main.py --mode single --dataset covtype --model naive_bayes --max-instances 2000
```

**Regression example:**
```bash
python main.py --mode single --dataset fried --model sgd --no-preprocess
```

**With drift detection:**
```bash
python main.py --mode single --dataset sensor --model hoeffding_tree --dashboard
```

## Using Individual Components

### Stream a Dataset

```python
from dataset_loader import DatasetStreamer

streamer = DatasetStreamer()
streamer.stream_dataset('electricity', delay=0.01, max_instances=1000)
```

### Run Pipeline

```python
from streaming_pipeline import StreamingPipeline

pipeline = StreamingPipeline(
    model_type='hoeffding_tree',
    preprocess=True,
    drift_detection=True
)
results = pipeline.run(max_instances=1000)
```

### Run Experiments

```python
from experiment_runner import ExperimentRunner

runner = ExperimentRunner()
runner.run_experiment(
    dataset_name='electricity',
    model_type='hoeffding_tree',
    preprocess=True,
    max_instances=1000
)
runner.print_summary()
```

## Troubleshooting

### Kafka Connection Error
- Ensure Docker containers are running: `docker compose ps`
- Check Kafka logs: `docker compose logs kafka`
- Verify Kafka is accessible: `docker compose exec kafka kafka-topics --list --bootstrap-server localhost:9092`
- Check if topic exists: `docker compose exec kafka kafka-topics --describe --topic stream-data --bootstrap-server localhost:9092`
- Restart services: `./stop_kafka.sh && ./start_kafka.sh`

### Import Errors
- Install missing packages: `pip install <package-name>`
- Check Python version: `python --version` (requires Python 3.8+)

### Dataset Loading Issues
- Verify CapyMOA installation: `python -c "import capymoa; print(capymoa.__version__)"`
- Check internet connection (datasets may download on first use)

### Dashboard Not Loading
- Check if port 8050 is available: `lsof -i :8050`
- Try different port: Modify `dashboard.py` port parameter

## Next Steps

1. **Explore different models**: Try different model types for each dataset
2. **Experiment with preprocessing**: Compare with/without preprocessing
3. **Analyze drift detection**: Observe how drift detection affects model performance
4. **Customize experiments**: Modify `experiment_runner.py` for your own experiments

## Performance Tips

- Use `--delay 0.001` for faster streaming (simulation)
- Limit instances with `--max-instances` for quick testing
- Disable preprocessing with `--no-preprocess` for faster processing
- Use simpler models (e.g., `sgd`) for faster training

