# Drift Analysis and Regression Support

## New Features

### 1. SGDRegressor Support for Regression

The consumer now supports SGDRegressor (Stochastic Gradient Descent Regressor) for regression tasks. This extends the classification pipeline to support regression by using CapyMOA's built-in SGDRegressor.

**Usage:**
```bash
python consumer.py --topic ml-stream-bike --task regression --model sgd --log-file results/bike_regression_sgd_metrics.csv
```

**Available Regression Models:**
- `fimtdd` - Fast Incremental Model Tree for Drift Detection
- `arf` - Adaptive Random Forest Regressor
- `knn` - K-Nearest Neighbors Regressor
- `sgd` - Stochastic Gradient Descent Regressor (NEW)

### 2. Real-time MAE Tracking

The consumer already tracks:
- **Cumulative MAE**: Mean Absolute Error across all instances (resets after drift)
- **Window MAE**: Rolling window MAE (last 1000 instances)
- **Cumulative MSE**: Mean Squared Error
- **Window MSE**: Rolling window MSE

All metrics are logged to CSV files and displayed in real-time.

### 3. Comparative Regression Evaluation

The dashboard supports comparative evaluation of multiple regression models:
- Side-by-side comparison of MAE/MSE across models
- Real-time performance tracking
- Model-specific drift event visualization

## Drift Analysis Tool

### Overview

The `drift_analysis.py` script performs comprehensive analysis of drift patterns and their impact on model performance.

### Features

1. **Mean Shift Analysis**
   - Analyzes performance shifts around drift events
   - Calculates before/after performance metrics
   - Identifies magnitude and direction of shifts

2. **Class Imbalance Analysis** (Classification)
   - Analyzes accuracy variance changes
   - Detects class distribution shifts
   - Correlates imbalance with performance drops

3. **Drift-Performance Correlation**
   - Correlates drift events with immediate performance drops
   - Tracks recovery time and recovery amount
   - Compares recovery patterns across models

4. **Visualizations**
   - Drift events timeline
   - Performance trajectory with drift markers
   - Mean shift analysis charts
   - Class imbalance analysis (for classification)
   - Performance correlation plots

### Usage

```bash
# Basic usage (analyzes results/ directory)
python drift_analysis.py

# Custom results directory
python drift_analysis.py --results-dir results --output-dir results

# Custom report location
python drift_analysis.py --report results/my_drift_report.txt
```

### Generated Outputs

1. **Visualizations** (saved to `results/`):
   - `drift_events_timeline.png` - Timeline of all drift events
   - `drift_performance_correlation.png` - Correlation between drifts and performance
   - `mean_shift_analysis.png` - Mean shift patterns analysis
   - `performance_trajectory_with_drifts.png` - Performance over time with drift markers
   - `class_imbalance_analysis.png` - Class imbalance analysis (classification only)

2. **Text Report** (`results/drift_analysis_report.txt`):
   - Summary statistics
   - Mean shift analysis results
   - Performance correlation metrics
   - Class imbalance statistics

### Example Workflow

1. **Run experiments with multiple models:**
   ```bash
   # Run FIMTDD
   python consumer.py --topic ml-stream-bike --task regression --model fimtdd --log-file results/bike_fimtdd_metrics.csv
   
   # Run ARF
   python consumer.py --topic ml-stream-bike --task regression --model arf --log-file results/bike_arf_metrics.csv
   
   # Run SGD
   python consumer.py --topic ml-stream-bike --task regression --model sgd --log-file results/bike_sgd_metrics.csv
   ```

2. **Run drift analysis:**
   ```bash
   python drift_analysis.py
   ```

3. **View results:**
   - Check generated PNG files in `results/` directory
   - Read the text report for detailed statistics
   - Use the dashboard to view real-time comparisons

## Dashboard Integration

The dashboard automatically displays:
- All regression models (including SGD)
- Real-time MAE/MSE metrics
- Drift events visualization
- Comparative performance charts

Access the dashboard:
```bash
python dashboard.py
```

## Key Metrics Explained

### For Regression:
- **Cumulative MAE**: Average absolute error across all instances (resets after drift)
- **Window MAE**: Average absolute error in last 1000 instances
- **Cumulative MSE**: Average squared error across all instances
- **Window MSE**: Average squared error in last 1000 instances

### Drift Analysis Metrics:
- **Immediate Drop**: Performance change right after drift detection
- **Recovery Time**: Number of instances needed to recover performance
- **Recovery Amount**: How much performance improved after recovery period
- **Shift Magnitude**: Absolute value of performance shift

## Notes

- Cumulative metrics reset after drift detection to track current model segment performance
- A 200-instance warmup period is used after drift to exclude initial poor predictions
- Window metrics always show recent performance (last 1000 instances)
- All visualizations use dark mode theme matching the dashboard

