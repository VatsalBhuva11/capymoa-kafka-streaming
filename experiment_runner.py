"""
Experiment Runner Module
Runs experiments across datasets with different models and settings.
"""

import json
import time
from typing import Dict, List, Optional
import pandas as pd
from dataset_loader import DatasetStreamer
from streaming_pipeline import StreamingPipeline
from threading import Thread
import numpy as np


class ExperimentRunner:
    """Runs experiments and compares results."""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'stream-data'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.results = []
    
    def run_experiment(self,
                      dataset_name: str,
                      model_type: str,
                      preprocess: bool = True,
                      drift_detection: bool = True,
                      drift_detector_type: str = 'adwin',
                      max_instances: Optional[int] = None,
                      drift_injection: Optional[Dict] = None) -> Dict:
        """
        Run a single experiment.
        
        Returns:
            Experiment results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Experiment: {dataset_name} | Model: {model_type} | Preprocess: {preprocess}")
        print(f"{'='*80}\n")
        
        # Start producer in separate thread
        streamer = DatasetStreamer(self.bootstrap_servers, self.topic)
        producer_thread = Thread(
            target=streamer.stream_dataset,
            args=(dataset_name,),
            kwargs={'delay': 0.001, 'max_instances': max_instances, 'drift_injection': drift_injection}
        )
        producer_thread.daemon = True
        producer_thread.start()
        
        # Give producer time to start
        time.sleep(2)
        
        # Run pipeline
        pipeline = StreamingPipeline(
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic,
            model_type=model_type,
            preprocess=preprocess,
            drift_detection=drift_detection,
            drift_detector_type=drift_detector_type,
            drift_injection=drift_injection
        )
        
        results = pipeline.run(max_instances=max_instances)
        
        # Wait for producer to finish
        producer_thread.join(timeout=30)
        
        # Store experiment metadata
        experiment_result = {
            'dataset': dataset_name,
            'model': model_type,
            'preprocess': preprocess,
            'drift_detection': drift_detection,
            'drift_detector': drift_detector_type,
            'max_instances': max_instances or 'all',
            'results': results
        }
        
        self.results.append(experiment_result)
        return experiment_result
    
    def run_comparison_experiments(self, 
                                   datasets: List[str],
                                   models: List[str],
                                   max_instances_per_dataset: Optional[int] = 1000,
                                   preprocess_options: List[bool] = [True, False],
                                   drift_detection: bool = True):
        """
        Run comparison experiments across datasets and models.
        
        Args:
            datasets: List of dataset names
            models: List of model types
            max_instances_per_dataset: Max instances per dataset
            preprocess_options: List of preprocessing options to test
            drift_detection: Whether to use drift detection
        """
        print(f"\n{'='*80}")
        print(f"Starting Comparison Experiments")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Preprocessing options: {preprocess_options}")
        print(f"{'='*80}\n")
        
        for dataset in datasets:
            for model in models:
                for preprocess in preprocess_options:
                    try:
                        self.run_experiment(
                            dataset_name=dataset,
                            model_type=model,
                            preprocess=preprocess,
                            drift_detection=drift_detection,
                            max_instances=max_instances_per_dataset
                        )
                        time.sleep(5)  # Pause between experiments
                    except Exception as e:
                        print(f"Error in experiment {dataset}-{model}-preprocess{preprocess}: {e}")
                        continue
    
    def generate_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all experiments."""
        summary_data = []
        
        for exp in self.results:
            final_metrics = exp['results'].get('final_metrics', {})
            cumulative = final_metrics.get('cumulative', {})
            
            row = {
                'Dataset': exp['dataset'],
                'Model': exp['model'],
                'Preprocess': exp['preprocess'],
                'Drift Detection': exp['drift_detection'],
                'Instances': exp['results'].get('final_metrics', {}).get('instance_count', 0),
                'Drift Events': len(exp['results'].get('drift_events', []))
            }
            
            # Add task-specific metrics
            if 'accuracy' in cumulative:
                row['Accuracy'] = cumulative.get('accuracy', 0)
                row['F1'] = cumulative.get('f1', 0)
            else:
                row['MAE'] = cumulative.get('mae', 0)
                row['MSE'] = cumulative.get('mse', 0)
                row['RMSE'] = cumulative.get('rmse', 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filename: str = 'experiment_results.json'):
        """Save experiment results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print summary of experiments."""
        df = self.generate_summary()
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")


def run_standard_experiments():
    """Run standard set of experiments."""
    runner = ExperimentRunner()
    
    classification_datasets = ['electricity', 'covtype', 'sensor']
    regression_datasets = ['fried', 'bike']
    
    classification_models = ['hoeffding_tree', 'naive_bayes', 'sgd']
    regression_models = ['sgd', 'linear', 'arf']
    
    # Run classification experiments
    print("\nRunning classification experiments...")
    runner.run_comparison_experiments(
        datasets=classification_datasets,
        models=classification_models,
        max_instances_per_dataset=2000,
        preprocess_options=[True, False]
    )
    
    # Run regression experiments
    print("\nRunning regression experiments...")
    runner.run_comparison_experiments(
        datasets=regression_datasets,
        models=regression_models,
        max_instances_per_dataset=2000,
        preprocess_options=[True, False]
    )
    
    # Generate and print summary
    runner.print_summary()
    runner.save_results()
    
    return runner

