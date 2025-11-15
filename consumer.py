"""
Kafka Consumer for online learning with CapyMOA.
Handles preprocessing, online model training, evaluation, and concept drift detection.
"""
from kafka import KafkaConsumer
from capymoa.classifier import HoeffdingTree, AdaptiveRandomForestClassifier, KNN
from capymoa.regressor import FIMTDD, AdaptiveRandomForestRegressor, KNNRegressor
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator
from river.drift import ADWIN
import json
import numpy as np
import argparse
import sys
import time
from datetime import datetime
from collections import deque
import os


class OnlinePreprocessor:
    """Online preprocessing for streaming data."""
    
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.feature_min = None
        self.feature_max = None
        self.feature_mean = None
        self.feature_std = None
        self.n_samples = 0
        self.warmup_samples = 100  # Use first 100 samples for normalization stats
        
    def update_stats(self, features):
        """Update running statistics for normalization."""
        features = np.array(features)
        
        if self.n_samples == 0:
            self.feature_min = features.copy()
            self.feature_max = features.copy()
            self.feature_mean = features.copy()
            self.feature_std = np.zeros_like(features)
        else:
            # Update min/max
            self.feature_min = np.minimum(self.feature_min, features)
            self.feature_max = np.maximum(self.feature_max, features)
            
            # Update mean using Welford's algorithm
            old_mean = self.feature_mean.copy()
            self.feature_mean += (features - self.feature_mean) / (self.n_samples + 1)
            
            # Update std (simplified version)
            if self.n_samples > 1:
                self.feature_std += (features - old_mean) * (features - self.feature_mean)
        
        self.n_samples += 1
    
    def normalize(self, features):
        """Normalize features using min-max scaling."""
        features = np.array(features)
        
        if self.n_samples < self.warmup_samples:
            # Return original features during warmup
            return features
        
        # Min-max normalization
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1  # Avoid division by zero
        
        normalized = (features - self.feature_min) / feature_range
        return normalized
    
    def preprocess(self, features):
        """Preprocess a single instance."""
        self.update_stats(features)
        normalized_features = self.normalize(features)
        
        # Handle missing values (replace NaN with 0)
        normalized_features = np.nan_to_num(normalized_features, nan=0.0)
        
        return normalized_features


class MetricsTracker:
    """Track metrics for online learning."""
    
    def __init__(self, task_type='classification', window_size=1000):
        self.task_type = task_type
        self.window_size = window_size
        
        # Cumulative metrics
        self.total_instances = 0
        self.cumulative_correct = 0
        self.cumulative_errors = []
        
        # Rolling window metrics
        self.window_predictions = deque(maxlen=window_size)
        self.window_actuals = deque(maxlen=window_size)
        self.window_errors = deque(maxlen=window_size)
        
        # Drift events
        self.drift_events = []
        
    def update(self, prediction, actual, error=None):
        """Update metrics with a new prediction."""
        self.total_instances += 1
        
        if self.task_type == 'classification':
            is_correct = int(prediction == actual)
            self.cumulative_correct += is_correct
            if error is None:
                error = 1.0 - is_correct
        
        self.window_predictions.append(prediction)
        self.window_actuals.append(actual)
        
        if error is not None:
            self.cumulative_errors.append(error)
            self.window_errors.append(error)
    
    def record_drift(self, instance_id, detector_name):
        """Record a drift event."""
        self.drift_events.append({
            'instance_id': instance_id,
            'detector': detector_name,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_cumulative_accuracy(self):
        """Get cumulative accuracy for classification."""
        if self.task_type != 'classification' or self.total_instances == 0:
            return None
        return self.cumulative_correct / self.total_instances
    
    def get_window_accuracy(self):
        """Get window accuracy for classification."""
        if self.task_type != 'classification' or len(self.window_predictions) == 0:
            return None
        
        correct = sum(1 for p, a in zip(self.window_predictions, self.window_actuals) if p == a)
        return correct / len(self.window_predictions)
    
    def get_cumulative_mae(self):
        """Get cumulative MAE for regression."""
        if self.task_type != 'regression' or len(self.cumulative_errors) == 0:
            return None
        return np.mean(self.cumulative_errors)
    
    def get_cumulative_mse(self):
        """Get cumulative MSE for regression."""
        if self.task_type != 'regression' or len(self.cumulative_errors) == 0:
            return None
        return np.mean([e**2 for e in self.cumulative_errors])
    
    def get_window_mae(self):
        """Get window MAE for regression."""
        if self.task_type != 'regression' or len(self.window_errors) == 0:
            return None
        return np.mean(list(self.window_errors))
    
    def get_window_mse(self):
        """Get window MSE for regression."""
        if self.task_type != 'regression' or len(self.window_errors) == 0:
            return None
        return np.mean([e**2 for e in self.window_errors])
    
    def get_summary(self):
        """Get summary of all metrics."""
        summary = {
            'total_instances': self.total_instances,
            'drift_events': len(self.drift_events),
            'drift_details': self.drift_events
        }
        
        if self.task_type == 'classification':
            summary['cumulative_accuracy'] = self.get_cumulative_accuracy()
            summary['window_accuracy'] = self.get_window_accuracy()
        else:
            summary['cumulative_mae'] = self.get_cumulative_mae()
            summary['cumulative_mse'] = self.get_cumulative_mse()
            summary['window_mae'] = self.get_window_mae()
            summary['window_mse'] = self.get_window_mse()
        
        return summary


def create_classifier(model_name, schema):
    """Create a classification model."""
    if model_name == 'hoeffding_tree':
        return HoeffdingTree(schema=schema)
    elif model_name == 'arf':
        return AdaptiveRandomForestClassifier(schema=schema, ensemble_size=10)
    elif model_name == 'knn':
        return KNN(schema=schema, k=5)
    else:
        raise ValueError(f"Unknown classifier: {model_name}")


def create_regressor(model_name, schema):
    """Create a regression model."""
    if model_name == 'fimtdd':
        return FIMTDD(schema=schema)
    elif model_name == 'arf':
        return AdaptiveRandomForestRegressor(schema=schema, ensemble_size=10)
    elif model_name == 'knn':
        return KNNRegressor(schema=schema, k=5)
    else:
        raise ValueError(f"Unknown regressor: {model_name}")


def process_classification_stream(consumer, topic_name, model_name, use_drift_detection=True, 
                                  log_file=None, max_instances=None):
    """Process classification stream."""
    from capymoa.datasets import Electricity
    
    # Load dataset to get schema
    stream = Electricity()
    schema = stream.get_schema()
    
    # Create model and evaluator
    model = create_classifier(model_name, schema)
    evaluator = ClassificationEvaluator(schema=schema)
    
    # Create drift detector
    drift_detector = None
    if use_drift_detection:
        drift_detector = ADWIN(delta=0.002)
    
    # Initialize preprocessor and metrics
    preprocessor = OnlinePreprocessor(task_type='classification')
    metrics = MetricsTracker(task_type='classification', window_size=1000)
    
    print(f"Model: {model_name}")
    print(f"Schema: {schema}")
    print(f"Drift detection: {'Enabled' if use_drift_detection else 'Disabled'}")
    print("-" * 60)
    
    instance_count = 0
    last_log_time = time.time()
    
    try:
        for message in consumer:
            data = message.value
            instance_count += 1
            
            # Extract features and label
            features = np.array(data['features'])
            label = int(data['label'])
            
            # Debug: Check what we're receiving from Kafka
            if instance_count <= 10 or (instance_count % 1000 == 0 and instance_count <= 2000):
                print(f"CONSUMER RECEIVED Instance {instance_count}: label={label} from Kafka message, instance_id={data.get('instance_id', 'N/A')}")
            
            # Preprocess
            processed_features = preprocessor.preprocess(features)
            
            # Create instance for CapyMOA using from_array (recommended method)
            from capymoa.instance import LabeledInstance
            # from_array takes: schema, feature vector, and class index (label value)
            instance = LabeledInstance.from_array(schema, processed_features, label)
            
            # Test-then-train evaluation
            prediction = model.predict(instance)
            
            # Debug: Check prediction and label for first few instances and periodically
            if instance_count <= 10 or (instance_count % 1000 == 0 and instance_count <= 2000):
                print(f"DEBUG Instance {instance_count}: prediction={prediction} (type: {type(prediction).__name__}), label={label} (type: {type(label).__name__}), match={prediction == label}")
            
            # Use label variable (from Kafka message) instead of accessing instance.y
            evaluator.update(label, prediction)
            
            # Update metrics
            # Ensure both are same type for comparison
            pred_val = prediction
            label_val = label
            # Convert to int if both are numeric
            if isinstance(prediction, (int, float, np.integer, np.floating)) and isinstance(label, (int, float, np.integer, np.floating)):
                pred_val = int(prediction)
                label_val = int(label)
            is_correct = int(pred_val == label_val)
            metrics.update(pred_val, label_val, error=1.0 - is_correct)
            
            # Drift detection on prediction error
            if drift_detector is not None:
                error = 1.0 - is_correct
                drift_detector.update(error)
                
                if drift_detector.drift_detected:
                    print(f"\n[DRIFT DETECTED] Instance {instance_count} - Reinitializing model")
                    metrics.record_drift(instance_count, 'ADWIN')
                    model = create_classifier(model_name, schema)
                    drift_detector = ADWIN(delta=0.002)  # Reset detector
            
            # Train model
            model.train(instance)
            
            # Log progress
            if instance_count % 1000 == 0 or (time.time() - last_log_time) > 5:
                cum_acc = metrics.get_cumulative_accuracy()
                win_acc = metrics.get_window_accuracy()
                win_acc_str = f"{win_acc:.4f}" if win_acc is not None else "N/A"
                cum_acc_str = f"{cum_acc:.4f}" if cum_acc is not None else "N/A"
                
                # Debug: Check prediction diversity in window
                if len(metrics.window_predictions) > 0:
                    unique_preds = len(set(metrics.window_predictions))
                    unique_actuals = len(set(metrics.window_actuals))
                    print(f"Instance {instance_count:6d} | "
                          f"Cumulative Acc: {cum_acc_str} | "
                          f"Window Acc: {win_acc_str:>6} | "
                          f"Drifts: {len(metrics.drift_events)} | "
                          f"Unique preds: {unique_preds}/{len(metrics.window_predictions)}, "
                          f"Unique actuals: {unique_actuals}/{len(metrics.window_actuals)}")
                else:
                    print(f"Instance {instance_count:6d} | "
                          f"Cumulative Acc: {cum_acc_str} | "
                          f"Window Acc: {win_acc_str:>6} | "
                          f"Drifts: {len(metrics.drift_events)}")
                last_log_time = time.time()
                
                # Write to log file
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"{instance_count},{cum_acc if cum_acc is not None else 'N/A'},{win_acc if win_acc is not None else 'N/A'}\n")
            
            if max_instances and instance_count >= max_instances:
                break
                
    except KeyboardInterrupt:
        print(f"\nStopped processing. Total instances: {instance_count}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    summary = metrics.get_summary()
    for key, value in summary.items():
        if key != 'drift_details':
            print(f"{key}: {value}")
    
    return metrics.get_summary()


def process_regression_stream(consumer, topic_name, model_name, use_drift_detection=True,
                              log_file=None, max_instances=None):
    """Process regression stream."""
    from capymoa.datasets import Bike
    
    # Load dataset to get schema
    stream = Bike()
    schema = stream.get_schema()
    
    # Create model and evaluator
    model = create_regressor(model_name, schema)
    evaluator = RegressionEvaluator(schema=schema)
    
    # Create drift detector
    drift_detector = None
    if use_drift_detection:
        drift_detector = ADWIN(delta=0.002)
    
    # Initialize preprocessor and metrics
    preprocessor = OnlinePreprocessor(task_type='regression')
    metrics = MetricsTracker(task_type='regression', window_size=1000)
    
    print(f"Model: {model_name}")
    print(f"Schema: {schema}")
    print(f"Drift detection: {'Enabled' if use_drift_detection else 'Disabled'}")
    print("-" * 60)
    
    instance_count = 0
    last_log_time = time.time()
    
    try:
        for message in consumer:
            data = message.value
            instance_count += 1
            
            # Extract features and target
            features = np.array(data['features'])
            target = float(data['target'])
            
            # Preprocess
            processed_features = preprocessor.preprocess(features)
            
            # Create instance for CapyMOA using from_array (recommended method)
            from capymoa.instance import RegressionInstance
            # from_array takes: schema, feature vector, and target value
            instance = RegressionInstance.from_array(schema, processed_features, target)
            
            # Test-then-train evaluation
            prediction = model.predict(instance)
            # Use target variable instead of instance.y
            evaluator.update(target, prediction)
            
            # Calculate error
            error = abs(prediction - target)
            metrics.update(prediction, target, error=error)
            
            # Drift detection on prediction error
            if drift_detector is not None:
                drift_detector.update(error)
                
                if drift_detector.drift_detected:
                    print(f"\n[DRIFT DETECTED] Instance {instance_count} - Reinitializing model")
                    metrics.record_drift(instance_count, 'ADWIN')
                    model = create_regressor(model_name, schema)
                    drift_detector = ADWIN(delta=0.002)  # Reset detector
            
            # Train model
            model.train(instance)
            
            # Log progress
            if instance_count % 1000 == 0 or (time.time() - last_log_time) > 5:
                cum_mae = metrics.get_cumulative_mae()
                cum_mse = metrics.get_cumulative_mse()
                win_mae = metrics.get_window_mae()
                cum_mae_str = f"{cum_mae:.4f}" if cum_mae is not None else "N/A"
                win_mae_str = f"{win_mae:.4f}" if win_mae is not None else "N/A"
                print(f"Instance {instance_count:6d} | "
                      f"Cumulative MAE: {cum_mae_str} | "
                      f"Cumulative MSE: {cum_mse:.4f if cum_mse is not None else 'N/A'} | "
                      f"Window MAE: {win_mae_str:>6} | "
                      f"Drifts: {len(metrics.drift_events)}")
                last_log_time = time.time()
                
                # Write to log file
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"{instance_count},{cum_mae if cum_mae is not None else 'N/A'},{cum_mse if cum_mse is not None else 'N/A'},{win_mae if win_mae is not None else 'N/A'}\n")
            
            if max_instances and instance_count >= max_instances:
                break
                
    except KeyboardInterrupt:
        print(f"\nStopped processing. Total instances: {instance_count}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    summary = metrics.get_summary()
    for key, value in summary.items():
        if key != 'drift_details':
            print(f"{key}: {value}")
    
    return metrics.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Online learning consumer with CapyMOA')
    parser.add_argument('--topic', type=str, required=True,
                       help='Kafka topic to consume from')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'],
                       required=True, help='Task type')
    parser.add_argument('--model', type=str, 
                       choices=['hoeffding_tree', 'arf', 'knn', 'fimtdd'],
                       default='hoeffding_tree',
                       help='Model to use')
    parser.add_argument('--no-drift-detection', action='store_true',
                       help='Disable drift detection')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--log-file', type=str, default=None,
                       help='File to log metrics')
    parser.add_argument('--max-instances', type=int, default=None,
                       help='Maximum number of instances to process')
    
    args = parser.parse_args()
    
    # Validate model for task
    if args.task == 'classification' and args.model == 'fimtdd':
        print("Error: FIMTDD is a regressor, not a classifier")
        sys.exit(1)
    if args.task == 'regression' and args.model in ['hoeffding_tree', 'arf', 'knn']:
        print(f"Error: {args.model} is a classifier, not a regressor")
        sys.exit(1)
    
    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_servers,
            auto_offset_reset='latest',  # Start from latest messages, not old ones
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=30000,  # Wait 30 seconds for new messages before timing out
            enable_auto_commit=True,
            group_id='ml-consumer-group'  # Use consumer group to track offsets
        )
        print(f"Connected to Kafka at {args.bootstrap_servers}")
        print(f"Consuming from topic: {args.topic}")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        print("Make sure Kafka is running (docker-compose up)")
        sys.exit(1)
    
    # Initialize log file
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else '.', exist_ok=True)
        if args.task == 'classification':
            with open(args.log_file, 'w') as f:
                f.write("instance,cumulative_accuracy,window_accuracy\n")
        else:
            with open(args.log_file, 'w') as f:
                f.write("instance,cumulative_mae,cumulative_mse,window_mae\n")
    
    # Process stream
    if args.task == 'classification':
        results = process_classification_stream(
            consumer, args.topic, args.model,
            use_drift_detection=not args.no_drift_detection,
            log_file=args.log_file,
            max_instances=args.max_instances
        )
    else:
        results = process_regression_stream(
            consumer, args.topic, args.model,
            use_drift_detection=not args.no_drift_detection,
            log_file=args.log_file,
            max_instances=args.max_instances
        )
    
    consumer.close()
    
    # Save results
    if args.log_file:
        results_file = args.log_file.replace('.csv', '_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
