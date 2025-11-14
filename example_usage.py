"""
Example Usage Scripts
Demonstrates how to use different components of the pipeline.
"""

from dataset_loader import DatasetStreamer
from streaming_pipeline import StreamingPipeline
from experiment_runner import ExperimentRunner
import time
from threading import Thread


def example_single_dataset():
    """Example: Run pipeline for a single dataset."""
    print("Example: Single Dataset Pipeline")
    print("="*60)
    
    dataset_name = 'electricity'
    model_type = 'hoeffding_tree'
    
    # Start producer in background
    streamer = DatasetStreamer()
    producer_thread = Thread(
        target=streamer.stream_dataset,
        args=(dataset_name,),
        kwargs={'delay': 0.01, 'max_instances': 1000}
    )
    producer_thread.daemon = True
    producer_thread.start()
    
    time.sleep(2)  # Wait for producer to start
    
    # Run pipeline
    pipeline = StreamingPipeline(
        model_type=model_type,
        preprocess=True,
        drift_detection=True
    )
    
    results = pipeline.run(max_instances=1000)
    
    print("\nResults:")
    print(f"  Final metrics: {results['final_metrics']}")
    print(f"  Drift events: {len(results['drift_events'])}")
    
    return results


def example_with_drift_injection():
    """Example: Run pipeline with drift injection."""
    print("Example: Pipeline with Drift Injection")
    print("="*60)
    
    dataset_name = 'fried'
    
    # Configure drift injection
    drift_injection = {
        'at_instance': 500,
        'type': 'feature_shift',
        'magnitude': 2.0
    }
    
    # Start producer
    streamer = DatasetStreamer()
    producer_thread = Thread(
        target=streamer.stream_dataset,
        args=(dataset_name,),
        kwargs={
            'delay': 0.01,
            'max_instances': 1000,
            'drift_injection': drift_injection
        }
    )
    producer_thread.daemon = True
    producer_thread.start()
    
    time.sleep(2)
    
    # Run pipeline with drift injection
    pipeline = StreamingPipeline(
        model_type='sgd',
        preprocess=True,
        drift_detection=True,
        drift_injection=drift_injection
    )
    
    results = pipeline.run(max_instances=1000)
    
    print(f"\nDrift events detected: {len(results['drift_events'])}")
    for event in results['drift_events']:
        print(f"  Instance {event['instance']}: {event}")
    
    return results


def example_experiment_comparison():
    """Example: Run comparison experiments."""
    print("Example: Experiment Comparison")
    print("="*60)
    
    runner = ExperimentRunner()
    
    # Run experiments on a single dataset with different models
    datasets = ['electricity']
    models = ['hoeffding_tree', 'naive_bayes']
    
    for dataset in datasets:
        for model in models:
            print(f"\nRunning: {dataset} with {model}")
            runner.run_experiment(
                dataset_name=dataset,
                model_type=model,
                preprocess=True,
                max_instances=500
            )
            time.sleep(2)
    
    # Print summary
    runner.print_summary()
    
    return runner


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == 'single':
            example_single_dataset()
        elif example == 'drift':
            example_with_drift_injection()
        elif example == 'experiment':
            example_experiment_comparison()
        else:
            print(f"Unknown example: {example}")
            print("Available: single, drift, experiment")
    else:
        print("Usage: python example_usage.py [single|drift|experiment]")
        print("\nRunning single dataset example...")
        example_single_dataset()

