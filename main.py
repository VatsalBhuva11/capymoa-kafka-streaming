"""
Main Entry Point
Run the complete streaming ML pipeline.
"""

import argparse
import sys
from dataset_loader import DatasetStreamer
from streaming_pipeline import StreamingPipeline
from dashboard import StreamingDashboard
from experiment_runner import ExperimentRunner, run_standard_experiments
from threading import Thread
import time


def run_single_dataset(dataset_name: str, model_type: str = 'hoeffding_tree',
                      preprocess: bool = True, drift_detection: bool = True,
                      max_instances: int = None, delay: float = 0.01,
                      dashboard: bool = False):
    """Run pipeline for a single dataset."""
    
    # Start dashboard if requested
    dashboard_instance = None
    if dashboard:
        dashboard_instance = StreamingDashboard()
        dashboard_thread = Thread(target=dashboard_instance.run, args=(False,))
        dashboard_thread.daemon = True
        dashboard_thread.start()
        time.sleep(2)  # Give dashboard time to start
    
    # Start producer
    streamer = DatasetStreamer()
    producer_thread = Thread(
        target=streamer.stream_dataset,
        args=(dataset_name,),
        kwargs={'delay': delay, 'max_instances': max_instances}
    )
    producer_thread.daemon = True
    producer_thread.start()
    
    time.sleep(2)  # Give producer time to start
    
    # Run pipeline
    pipeline = StreamingPipeline(
        model_type=model_type,
        preprocess=preprocess,
        drift_detection=drift_detection
    )
    
    results = pipeline.run(max_instances=max_instances)
    
    # Update dashboard if running
    if dashboard_instance:
        task_type = 'classification' if dataset_name.lower() in ['electricity', 'covtype', 'sensor'] else 'regression'
        dashboard_instance.update_metrics(results['final_metrics'], task_type)
        dashboard_instance.update_drift_events(results['drift_events'])
    
    return results


def run_all_datasets():
    """Run pipeline for all datasets sequentially."""
    classification_datasets = ['electricity', 'covtype', 'sensor']
    regression_datasets = ['fried', 'bike']
    
    all_results = {}
    
    for dataset in classification_datasets + regression_datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*80}\n")
        
        model_type = 'hoeffding_tree' if dataset in classification_datasets else 'sgd'
        results = run_single_dataset(
            dataset_name=dataset,
            model_type=model_type,
            max_instances=5000,  # Limit for demo
            delay=0.001
        )
        all_results[dataset] = results
        time.sleep(5)  # Pause between datasets
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Streaming ML Pipeline')
    parser.add_argument('--mode', choices=['single', 'all', 'experiments', 'dashboard'],
                       default='single', help='Run mode')
    parser.add_argument('--dataset', type=str, help='Dataset name (for single mode)')
    parser.add_argument('--model', type=str, default='hoeffding_tree',
                       help='Model type')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Disable preprocessing')
    parser.add_argument('--no-drift', action='store_true',
                       help='Disable drift detection')
    parser.add_argument('--max-instances', type=int, help='Maximum instances to process')
    parser.add_argument('--delay', type=float, default=0.01,
                       help='Delay between instances (seconds)')
    parser.add_argument('--dashboard', action='store_true',
                       help='Enable live dashboard')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.dataset:
            print("Error: --dataset required for single mode")
            sys.exit(1)
        run_single_dataset(
            dataset_name=args.dataset,
            model_type=args.model,
            preprocess=not args.no_preprocess,
            drift_detection=not args.no_drift,
            max_instances=args.max_instances,
            delay=args.delay,
            dashboard=args.dashboard
        )
    
    elif args.mode == 'all':
        run_all_datasets()
    
    elif args.mode == 'experiments':
        run_standard_experiments()
    
    elif args.mode == 'dashboard':
        dashboard = StreamingDashboard()
        dashboard.run(debug=True)


if __name__ == '__main__':
    main()

