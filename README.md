# Streaming Machine Learning Pipeline with Kafka and CapyMOA

A complete streaming machine learning pipeline that processes datasets in real-time using Kafka, trains online models with CapyMOA, and handles concept drift detection.

## Features

- **Streaming Infrastructure**: Kafka + Zookeeper running in Docker
- **Online Learning**: CapyMOA models for classification and regression
- **Concept Drift Detection**: ADWIN drift detector with automatic model reinitialization
- **Real-time Evaluation**: Test-then-train (prequential) evaluation
- **Metrics Tracking**: Cumulative and rolling window metrics
- **Multiple Models**: Support for HoeffdingTree, AdaptiveRandomForest, KNN, FIMTDD
- **Datasets**: Electricity (classification) and Bike (regression)

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- pip

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /home/vb11x/Downloads/group14
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start Kafka and Zookeeper:
```bash
docker-compose up -d
```

Wait a few seconds for Kafka to be ready. You can check the status with:
```bash
docker-compose ps
```

## Usage

### 1. Start Kafka (if not already running)

```bash
docker-compose up -d
```

### 2. Stream Data to Kafka

#### Stream Electricity dataset (Classification):
```bash
python producer.py --dataset electricity --stream-rate 0.1
```

#### Stream Bike dataset (Regression):
```bash
python producer.py --dataset bike --stream-rate 0.1
```

#### Stream both datasets:
```bash
python producer.py --dataset both --stream-rate 0.1
```

#### With concept drift injection:
```bash
python producer.py --dataset electricity --stream-rate 0.1 --inject-drift
```

**Options:**
- `--dataset`: Dataset to stream (`electricity`, `bike`, or `both`)
- `--stream-rate`: Delay between messages in seconds (default: 0.1)
- `--inject-drift`: Inject artificial concept drift
- `--bootstrap-servers`: Kafka bootstrap servers (default: localhost:9092)

### 3. Consume and Train Models

#### Classification (Electricity dataset):
```bash
python consumer.py --topic ml-stream-electricity --task classification --model hoeffding_tree
```

#### Regression (Bike dataset):
```bash
python consumer.py --topic ml-stream-bike --task regression --model fimtdd
```

**Available Models:**
- **Classification**: `hoeffding_tree`, `arf`, `knn`
- **Regression**: `fimtdd`, `arf`, `knn`

**Options:**
- `--topic`: Kafka topic to consume from
- `--task`: Task type (`classification` or `regression`)
- `--model`: Model to use
- `--no-drift-detection`: Disable drift detection
- `--log-file`: File to log metrics (CSV format)
- `--max-instances`: Maximum number of instances to process

### 4. Run Complete Experiments

Run all predefined experiments:
```bash
python experiment_runner.py --all --stream-rate 0.01 --max-instances 5000
```

Run specific experiments:
```bash
python experiment_runner.py --experiments electricity:classification:hoeffding_tree bike:regression:fimtdd
```

Generate comparison report from existing results:
```bash
python experiment_runner.py --compare-only
```

## Project Structure

```
group14/
├── docker-compose.yml      # Kafka + Zookeeper setup
├── producer.py             # Kafka producer for streaming datasets
├── consumer.py             # Kafka consumer with online learning
├── dashboard.py            # Live visualization dashboard
├── experiment_runner.py    # Experiment runner and comparison tool
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── results/               # Experiment results (generated)
    ├── *_metrics.csv      # Metrics logs
    ├── *_results.json     # Experiment results
    └── comparison_report.html  # Comparison report
```

## Metrics Tracked

### Classification (Electricity):
- Cumulative Accuracy
- Rolling Window Accuracy (1000 instances)
- Drift Events

### Regression (Bike):
- Cumulative MAE (Mean Absolute Error)
- Cumulative MSE (Mean Squared Error)
- Rolling Window MAE (1000 instances)
- Drift Events

## Concept Drift

The system includes:
- **Drift Injection**: Artificial drift can be injected in the producer
- **Drift Detection**: ADWIN detector monitors prediction errors
- **Automatic Recovery**: Models are reinitialized when drift is detected

## Stopping Services

Stop Kafka and Zookeeper:
```bash
docker-compose down
```

## Troubleshooting

1. **Kafka connection errors**: Make sure Docker containers are running:
   ```bash
   docker-compose ps
   ```

2. **Port conflicts**: If ports 2181 or 9092 are in use, modify `docker-compose.yml`

3. **Dataset download issues**: CapyMOA will automatically download datasets on first use

4. **Memory issues**: Reduce `--max-instances` or `--stream-rate` for faster processing

## Example Workflow

1. Start Kafka:
   ```bash
   docker-compose up -d
   ```

2. In terminal 1, start producer:
   ```bash
   python producer.py --dataset electricity --stream-rate 0.1 --inject-drift
   ```

3. In terminal 2, start consumer:
   ```bash
   python consumer.py --topic ml-stream-electricity --task classification --model arf --log-file results/electricity_arf.csv
   ```

4. View results in `results/electricity_arf_results.json`

### 5. Live Visualization Dashboard

Start the interactive dashboard to monitor metrics in real-time:

```bash
python dashboard.py
```

The dashboard will be available at `http://127.0.0.1:8050`

**Dashboard Features:**
- **Real-time Updates**: Automatically refreshes every 2 seconds (configurable)
- **Multi-Model Comparison**: Compare multiple models side-by-side
- **Interactive Plots**: 
  - Cumulative metrics over time
  - Rolling window metrics (last 1000 instances)
  - Concept drift events visualization
- **Filters**: Filter by task type, dataset, and select specific models
- **Summary Cards**: Quick view of best performing models
- **Metrics Table**: Latest metrics for all running experiments

**Dashboard Options:**
- `--results-dir`: Directory containing CSV log files (default: results)
- `--port`: Port to run dashboard on (default: 8050)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--update-interval`: Seconds between auto-refreshes (default: 2)
- `--debug`: Run in debug mode

**Example:**
```bash
# Start dashboard with custom settings
python dashboard.py --port 8050 --update-interval 3

# In another terminal, run consumer with logging
python consumer.py --topic ml-stream-electricity --task classification \
    --model hoeffding_tree --log-file results/electricity_hoeffding_tree_metrics.csv
```

The dashboard will automatically detect and display metrics from all CSV files in the results directory.

## References

- [CapyMOA Documentation](https://capymoa.org/)
- [Kafka Python Client](https://kafka-python.readthedocs.io/)
- [Electricity Dataset](https://capymoa.org/api/modules/capymoa.datasets.Electricity.html)
- [Bike Dataset](https://capymoa.org/api/modules/capymoa.datasets.Bike.html)

