"""
Main Streaming Pipeline
Orchestrates the complete streaming ML pipeline with test-then-train evaluation.
"""

import json
import time
import numpy as np
from typing import Dict, Optional
from kafka import KafkaConsumer
from online_preprocessor import OnlinePreprocessor
from incremental_models import IncrementalClassifier, IncrementalRegressor
from metrics_tracker import MetricsTracker
from drift_handler import DriftDetector, DriftInjector


class StreamingPipeline:
    """Main pipeline for streaming ML with test-then-train evaluation."""
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'stream-data',
                 model_type: str = 'hoeffding_tree',
                 preprocess: bool = True,
                 drift_detection: bool = True,
                 drift_detector_type: str = 'adwin',
                 drift_injection: Optional[Dict] = None):
        """
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic to consume from
            model_type: Type of model to use
            preprocess: Whether to apply preprocessing
            drift_detection: Whether to detect drift
            drift_detector_type: Type of drift detector
            drift_injection: Dict with drift injection config
        """
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=10000  # 10 second timeout
        )
        self.topic = topic
        self.model_type = model_type
        self.preprocess = preprocess
        self.drift_detection = drift_detection
        self.drift_injection = drift_injection
        
        # Initialize components
        self.preprocessor = OnlinePreprocessor(standardize=preprocess) if preprocess else None
        self.model = None
        self.metrics = None
        self.drift_detector = DriftDetector(detector_type=drift_detector_type) if drift_detection else None
        
        # State
        self.current_dataset = None
        self.is_classification = None
        self.instance_count = 0
        self.results = {
            'metrics_history': [],
            'drift_events': [],
            'final_metrics': {}
        }
    
    def _initialize_model(self, is_classification: bool):
        """Initialize model based on task type."""
        if is_classification:
            self.model = IncrementalClassifier(model_type=self.model_type)
        else:
            self.model = IncrementalRegressor(model_type=self.model_type)
    
    def _initialize_metrics(self, is_classification: bool):
        """Initialize metrics tracker."""
        task_type = 'classification' if is_classification else 'regression'
        self.metrics = MetricsTracker(task_type=task_type)
    
    def run(self, max_instances: Optional[int] = None):
        """
        Run the streaming pipeline with test-then-train evaluation.
        
        Args:
            max_instances: Maximum number of instances to process
        """
        print(f"Starting streaming pipeline...")
        print(f"Model: {self.model_type}")
        print(f"Preprocessing: {self.preprocess}")
        print(f"Drift detection: {self.drift_detection}")
        print(f"{'='*60}\n")
        
        try:
            for msg in self.consumer:
                if max_instances and self.instance_count >= max_instances:
                    break
                
                data = msg.value
                
                # Check if new dataset
                if self.current_dataset != data.get('dataset'):
                    self.current_dataset = data.get('dataset')
                    self.is_classification = data.get('is_classification', True)
                    self._initialize_model(self.is_classification)
                    self._initialize_metrics(self.is_classification)
                    if self.drift_detector:
                        self.drift_detector.reset()
                    print(f"\n{'='*60}")
                    print(f"Processing dataset: {self.current_dataset}")
                    print(f"Task: {'Classification' if self.is_classification else 'Regression'}")
                    print(f"{'='*60}\n")
                
                # Extract data
                X = np.array(data['features'])
                y_true = data['target']
                instance_id = data.get('instance_id', self.instance_count)
                
                # Inject drift if specified
                if self.drift_injection and data.get('drift_injected', False):
                    drift_type = self.drift_injection.get('type', 'feature_shift')
                    magnitude = self.drift_injection.get('magnitude', 1.0)
                    X, y_true = DriftInjector.inject_sudden_drift(X, y_true, drift_type, magnitude)
                    print(f"*** Drift injected at instance {instance_id} ***")
                
                # Preprocess
                if self.preprocessor:
                    X = self.preprocessor.preprocess_instance(X, is_training=False)
                
                # Test-then-train: Predict first
                if self.is_classification:
                    y_pred = self.model.predict(X)
                    y_proba = self.model.predict_proba(X)
                else:
                    y_pred = self.model.predict(X)
                    y_proba = None
                
                # Calculate error for drift detection
                if self.is_classification:
                    error = 1.0 if y_true != y_pred else 0.0
                else:
                    error = abs(y_true - y_pred)
                
                # Update drift detector
                drift_detected = False
                if self.drift_detector:
                    drift_detected = self.drift_detector.update(error)
                    if drift_detected:
                        print(f"\n*** DRIFT DETECTED at instance {instance_id} ***")
                        print(f"Resetting model...\n")
                        self.model.reset()
                        if self.preprocessor:
                            self.preprocessor.reset()
                
                # Update metrics
                self.metrics.update(y_true, y_pred, y_proba)
                
                # Train model (test-then-train)
                # Preprocess for training
                if self.preprocessor:
                    X_train = self.preprocessor.preprocess_instance(X, is_training=True)
                else:
                    X_train = X
                
                if self.is_classification:
                    self.model.partial_fit(X_train, int(y_true))
                else:
                    self.model.partial_fit(X_train, float(y_true))
                
                self.instance_count += 1
                
                # Print progress
                if self.instance_count % 100 == 0:
                    current_metrics = self.metrics.get_current_metrics()
                    print(f"Instance {self.instance_count}: {current_metrics}")
                
                # Store metrics periodically
                if self.instance_count % 1000 == 0:
                    current_metrics = self.metrics.get_current_metrics()
                    self.results['metrics_history'].append({
                        'instance': self.instance_count,
                        'metrics': current_metrics
                    })
        
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Finalize results
            if self.metrics:
                self.results['final_metrics'] = self.metrics.get_current_metrics()
                self.results['metrics_history'] = self.metrics.get_history()
            
            if self.drift_detector:
                self.results['drift_events'] = self.drift_detector.get_drift_events()
            
            print(f"\n{'='*60}")
            print(f"Pipeline completed. Processed {self.instance_count} instances")
            print(f"{'='*60}\n")
        
        return self.results

