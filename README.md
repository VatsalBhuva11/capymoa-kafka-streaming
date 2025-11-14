# Streaming Machine Learning Pipeline

A comprehensive streaming ML pipeline for online learning with concept drift detection, real-time evaluation, and visualization.

## Features

- **Stream Setup**: Load and stream datasets through Kafka
- **Online Preprocessing**: Per-instance preprocessing (scaling, encoding, missing values)
- **Incremental Learning**: Test-then-train evaluation with incremental models
- **Metrics Tracking**: Rolling and cumulative performance metrics
- **Concept Drift**: Drift injection and detection (ADWIN, DDM)
- **Live Dashboard**: Real-time visualization of metrics and drift events
- **Experiments**: Automated experiment runner for model comparison

## Datasets

### Classification
- **Electricity**: Electricity market dataset
- **Covtype**: Forest cover type dataset
- **Sensor**: Sensor stream dataset

### Regression
- **Fried**: Fried dataset
- **Bike**: Bike sharing dataset

All datasets are loaded from [CapyMOA](https://capymoa.org/).

## Installation

### Prerequisites

1. **Docker**: Install Docker and Docker Compose
   - **Linux**: [Docker installation guide](https://docs.docker.com/engine/install/)
   - **macOS/Windows**: [Docker Desktop](https://docs.docker.com/desktop/)
   
   Verify installation:
   ```bash
   docker --version
   docker compose version
   ```

2. **Start Kafka and Zookeeper with Docker**:
   ```bash
   # Using helper script (recommended)
   chmod +x start_kafka.sh
   ./start_kafka.sh
   
   # Or using Docker Compose directly
   docker compose up -d
   docker compose exec kafka kafka-topics --create \
       --topic stream-data \
       --bootstrap-server localhost:9092 \
       --partitions 1 \
       --replication-factor 1
   ```

3. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: CapyMOA requires Java to be installed. If you encounter JPype errors:
   ```bash
   # Linux
   sudo apt-get install default-jdk
   
   # macOS
   brew install openjdk
   
   # Then reinstall CapyMOA
   pip install --upgrade --force-reinstall capymoa jpype1
   ```

## Usage

### Quick Start

1. **Start Kafka with Docker**:
   ```bash
   ./start_kafka.sh
   ```

2. **Run a single dataset**:
   ```bash
   python main.py --mode single --dataset electricity --model hoeffding_tree --dashboard
   ```

3. **Run all datasets**:
   ```bash
   python main.py --mode all
   ```

4. **Run experiments**:
   ```bash
   python main.py --mode experiments
   ```

5. **View dashboard only**:
   ```bash
   python main.py --mode dashboard
   ```

### Command Line Options

```
--mode: single, all, experiments, dashboard
--dataset: Dataset name (electricity, covtype, sensor, fried, bike)
--model: Model type (hoeffding_tree, naive_bayes, sgd, arf, linear)
--no-preprocess: Disable preprocessing
--no-drift: Disable drift detection
--max-instances: Maximum instances to process
--delay: Delay between instances (seconds)
--dashboard: Enable live dashboard
```

### Examples

**Classification with drift detection**:
```bash
python main.py --mode single --dataset electricity --model hoeffding_tree --dashboard
```

**Regression without preprocessing**:
```bash
python main.py --mode single --dataset fried --model sgd --no-preprocess
```

**Run experiments with limited instances**:
```bash
python main.py --mode experiments
```

## Architecture

### Components

1. **dataset_loader.py**: Loads CapyMOA datasets and streams through Kafka
2. **online_preprocessor.py**: Per-instance preprocessing for streams
3. **incremental_models.py**: Incremental learners (classification & regression)
4. **metrics_tracker.py**: Tracks rolling and cumulative metrics
5. **drift_handler.py**: Drift injection and detection
6. **streaming_pipeline.py**: Main pipeline orchestrator
7. **dashboard.py**: Live visualization dashboard
8. **experiment_runner.py**: Automated experiment execution

### Pipeline Flow

```
Dataset → Kafka Producer → Kafka Topic → Kafka Consumer → 
Preprocessing → Test (Predict) → Metrics Update → 
Drift Detection → Train (Update Model) → Repeat
```

## Models

### Classification Models
- **Hoeffding Tree**: Incremental decision tree
- **Naive Bayes**: Gaussian Naive Bayes
- **SGD**: Stochastic Gradient Descent (Logistic Regression)
- **ARF**: Adaptive Random Forest

### Regression Models
- **SGD**: Stochastic Gradient Descent (Linear Regression)
- **Linear**: Passive Aggressive Regressor
- **ARF**: Adaptive Random Forest Regressor

## Metrics

### Classification
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro-averaged F1 score

### Regression
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

All metrics are tracked both **cumulatively** (over all instances) and **rolling** (over a window of recent instances).

## Concept Drift

### Detection
- **ADWIN**: Adaptive Windowing
- **DDM**: Drift Detection Method

### Injection
Drift can be injected at specific instances:
- **Sudden drift**: Immediate change in data distribution
- **Gradual drift**: Progressive change over time

## Dashboard

The dashboard provides:
- **Current Metrics**: Real-time performance metrics
- **Performance Trends**: Historical metric evolution
- **Drift Alerts**: Notifications when drift is detected

Access the dashboard at `http://localhost:8050` when running with `--dashboard` flag.

## Experiments

The experiment runner compares:
- Different models
- With/without preprocessing
- Across all datasets

Results are saved to `experiment_results.json` and a summary table is printed.

## Evaluation Method

**Test-Then-Train (Prequential Evaluation)**:
1. For each instance:
   - Predict using current model
   - Update metrics
   - Check for drift
   - Update model with instance

This provides an unbiased estimate of model performance over time.

## File Structure

```
group14/
├── main.py                    # Main entry point
├── dataset_loader.py          # Dataset loading and streaming
├── online_preprocessor.py     # Online preprocessing
├── incremental_models.py      # Incremental learning models
├── metrics_tracker.py         # Metrics tracking
├── drift_handler.py          # Drift injection and detection
├── streaming_pipeline.py     # Main pipeline
├── dashboard.py              # Visualization dashboard
├── experiment_runner.py      # Experiment execution
├── verify_setup.py           # Setup verification
├── example_usage.py          # Usage examples
├── docker-compose.yml        # Docker Compose config for Kafka
├── start_kafka.sh           # Helper script to start Kafka
├── stop_kafka.sh            # Helper script to stop Kafka
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── QUICKSTART.md            # Quick start guide
├── PROJECT_SUMMARY.md       # Project summary
├── kafka_producer.py         # Simple producer (example)
└── adwin_drift_detector.py   # Simple detector (example)
```

## Troubleshooting

### Kafka Connection Issues
- Ensure Docker containers are running: `docker compose ps`
- Check Kafka logs: `docker compose logs kafka`
- Verify Kafka is accessible: `docker compose exec kafka kafka-topics --list --bootstrap-server localhost:9092`
- Check if topic exists: `docker compose exec kafka kafka-topics --describe --topic stream-data --bootstrap-server localhost:9092`
- Restart services: `./stop_kafka.sh && ./start_kafka.sh`

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Ensure CapyMOA is installed: `pip install capymoa`

### Dashboard Not Loading
- Check port 8050 is available
- Try accessing `http://0.0.0.0:8050` or `http://127.0.0.1:8050`

## Performance Tips

- Use `--delay 0.001` for faster streaming
- Limit instances with `--max-instances` for quick testing
- Disable preprocessing with `--no-preprocess` for faster processing
- Use simpler models (e.g., `sgd`) for faster training

## Citation

If you use this code, please cite:
- CapyMOA: https://capymoa.org/
- River: https://riverml.xyz/
- Kafka: https://kafka.apache.org/

## License

This project is provided as-is for educational and research purposes.

