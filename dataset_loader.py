"""
Dataset Loader Module
Loads and streams capymoa datasets through Kafka producer.
"""

import json
import time
from typing import Iterator, Tuple, Optional
from kafka import KafkaProducer
import numpy as np
import pandas as pd
from capymoa.datasets import (
    Electricity,
    Covtype,
    Sensor,
    Fried,
    Bike
)


class DatasetStreamer:
    """Streams datasets instance-by-instance through Kafka."""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'stream-data'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
        self.dataset_map = {
            'electricity': Electricity,
            'covtype': Covtype,
            'sensor': Sensor,
            'fried': Fried,
            'bike': Bike
        }
    
    def load_dataset(self, dataset_name: str):
        """Load a capymoa dataset."""
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower not in self.dataset_map:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.dataset_map.keys())}")
        
        dataset_class = self.dataset_map[dataset_name_lower]
        return dataset_class()
    
    def stream_dataset(self, dataset_name: str, delay: float = 0.01, 
                      max_instances: Optional[int] = None,
                      drift_injection: Optional[dict] = None):
        """
        Stream dataset instances one-by-one through Kafka.
        
        Args:
            dataset_name: Name of the dataset to stream
            delay: Delay between instances (seconds)
            max_instances: Maximum number of instances to stream (None for all)
            drift_injection: Dict with 'at_instance' and 'type' for drift injection
        """
        dataset = self.load_dataset(dataset_name)
        
        # Determine if classification or regression
        is_classification = dataset_name.lower() in ['electricity', 'covtype', 'sensor']
        
        instance_count = 0
        drift_injected = False
        
        print(f"Starting to stream {dataset_name} dataset...")
        print(f"Task type: {'Classification' if is_classification else 'Regression'}")
        
        for instance in dataset:
            if max_instances and instance_count >= max_instances:
                break
            
            # Inject drift if specified
            if drift_injection and not drift_injected and instance_count >= drift_injection.get('at_instance', 0):
                drift_injected = True
                print(f"\n*** Concept drift injected at instance {instance_count} ***\n")
                # Drift injection logic will be handled in preprocessing
            
            # Extract features and target
            X = instance.x
            y = instance.y
            
            # Convert to dict for JSON serialization
            data = {
                'dataset': dataset_name,
                'instance_id': instance_count,
                'features': X.tolist() if hasattr(X, 'tolist') else list(X),
                'target': float(y),
                'is_classification': is_classification,
                'drift_injected': drift_injected and instance_count == drift_injection.get('at_instance', 0) if drift_injection else False
            }
            
            # Send to Kafka
            self.producer.send(self.topic, data)
            instance_count += 1
            
            if instance_count % 1000 == 0:
                print(f"Streamed {instance_count} instances...")
            
            time.sleep(delay)
        
        self.producer.flush()
        print(f"\nFinished streaming {instance_count} instances from {dataset_name}")
        return instance_count
    
    def stream_all_datasets(self, delay: float = 0.01, max_instances_per_dataset: Optional[int] = None):
        """Stream all datasets sequentially."""
        classification_datasets = ['electricity', 'covtype', 'sensor']
        regression_datasets = ['fried', 'bike']
        
        results = {}
        
        for dataset_name in classification_datasets + regression_datasets:
            print(f"\n{'='*60}")
            print(f"Streaming {dataset_name.upper()} dataset")
            print(f"{'='*60}\n")
            
            count = self.stream_dataset(dataset_name, delay, max_instances_per_dataset)
            results[dataset_name] = count
            time.sleep(2)  # Pause between datasets
        
        return results

