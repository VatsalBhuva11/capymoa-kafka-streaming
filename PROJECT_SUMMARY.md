# Project Summary

## Overview

This project implements a comprehensive streaming machine learning pipeline with online learning, concept drift detection, and real-time evaluation capabilities.

## Components Implemented

### 1. Stream Setup ✓
- **dataset_loader.py**: Loads CapyMOA datasets (Electricity, Covtype, Sensor, Fried, Bike)
- Streams data instances one-by-one through Kafka in correct order
- Supports all required datasets for classification and regression

### 2. Online Preprocessing ✓
- **online_preprocessor.py**: Per-instance preprocessing suitable for streams
- Features:
  - Standardization/normalization
  - Missing value handling (mean, median, zero, drop)
  - Running statistics for online scaling
  - Categorical feature detection

### 3. Model Training (Online/Incremental) ✓
- **incremental_models.py**: Incremental learners for classification and regression
- **Classification models**: Hoeffding Tree, Naive Bayes, SGD, Adaptive Random Forest
- **Regression models**: SGD, Linear (PA), Adaptive Random Forest
- **Test-then-train (prequential) evaluation**: Implemented in streaming_pipeline.py

### 4. Metrics & Evaluation ✓
- **metrics_tracker.py**: Tracks rolling and cumulative performance metrics
- **Classification metrics**: Accuracy, F1 Score
- **Regression metrics**: MAE, MSE, RMSE
- Logs and stores metric evolution over time

### 5. Concept Drift Handling ✓
- **drift_handler.py**: Drift injection and detection
- **Detection**: ADWIN, DDM
- **Injection**: Sudden and gradual drift scenarios
- Records and logs detected drift events

### 6. Dashboard/Visualization ✓
- **dashboard.py**: Live dashboard using Dash
- Features:
  - Real-time metrics display
  - Performance trends visualization
  - Drift event alerts
  - Interactive plots with Plotly

### 7. Experiments & Comparison ✓
- **experiment_runner.py**: Automated experiment execution
- Compares models across datasets
- Tests different preprocessing settings
- Generates summary tables and saves results

### 8. Documentation ✓
- **README.md**: Comprehensive documentation
- **QUICKSTART.md**: Quick start guide
- **verify_setup.py**: Setup verification script
- **example_usage.py**: Usage examples

## Main Entry Points

### main.py
Primary entry point with command-line interface:
- `--mode single`: Run single dataset
- `--mode all`: Run all datasets
- `--mode experiments`: Run comparison experiments
- `--mode dashboard`: View dashboard only

### streaming_pipeline.py
Core pipeline orchestrator that:
- Consumes from Kafka
- Applies preprocessing
- Performs test-then-train evaluation
- Detects concept drift
- Tracks metrics

## File Structure

```
group14/
├── main.py                    # Main entry point
├── dataset_loader.py          # Dataset loading & streaming
├── online_preprocessor.py     # Online preprocessing
├── incremental_models.py      # Incremental models
├── metrics_tracker.py         # Metrics tracking
├── drift_handler.py          # Drift handling
├── streaming_pipeline.py     # Main pipeline
├── dashboard.py              # Visualization
├── experiment_runner.py      # Experiments
├── verify_setup.py           # Setup verification
├── example_usage.py          # Usage examples
├── docker-compose.yml        # Docker Compose config for Kafka
├── start_kafka.sh           # Helper script to start Kafka
├── stop_kafka.sh            # Helper script to stop Kafka
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── PROJECT_SUMMARY.md       # This file
└── .gitignore               # Git ignore rules
```

## Usage Examples

### Basic Usage
```bash
# Single dataset with dashboard
python main.py --mode single --dataset electricity --dashboard

# All datasets
python main.py --mode all

# Run experiments
python main.py --mode experiments
```

### Verify Setup
```bash
python verify_setup.py
```

## Key Features

1. **Streaming Architecture**: Kafka-based producer-consumer pattern
2. **Online Learning**: Incremental models that update with each instance
3. **Prequential Evaluation**: Test-then-train approach for unbiased evaluation
4. **Drift Adaptation**: Automatic drift detection and model reset
5. **Real-time Monitoring**: Live dashboard with metrics and alerts
6. **Experiment Automation**: Systematic comparison across configurations

## Dependencies

- **capymoa**: Dataset loading
- **kafka-python**: Kafka integration
- **river**: Incremental learning and metrics
- **scikit-learn**: Additional ML utilities
- **dash/plotly**: Dashboard visualization
- **pandas/numpy**: Data handling

## Testing

Run setup verification:
```bash
python verify_setup.py
```

Run example usage:
```bash
python example_usage.py single
```

## Next Steps

1. Install Docker: https://docs.docker.com/get-docker/
2. Run verification: `python verify_setup.py`
3. Start Kafka with Docker: `./start_kafka.sh`
4. Run a test: `python main.py --mode single --dataset electricity --max-instances 100`
5. Explore experiments: `python main.py --mode experiments`

## Notes

- All datasets are loaded from CapyMOA library
- Kafka and Zookeeper run in Docker containers (use `./start_kafka.sh`)
- Docker must be installed and running before starting Kafka
- Dashboard runs on port 8050 by default
- Results are saved to `experiment_results.json`

