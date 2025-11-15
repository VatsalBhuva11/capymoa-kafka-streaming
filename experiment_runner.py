"""
Experiment runner for comparing different models across datasets.
Runs multiple experiments and generates comparison reports.
"""
import subprocess
import time
import json
import os
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_experiment(dataset, task, model, use_drift_detection=True, 
                   stream_rate=0.01, max_instances=None, results_dir='results'):
    """
    Run a single experiment.
    
    Returns:
        dict: Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset} - {task} - {model}")
    print(f"{'='*60}")
    
    topic_name = f"ml-stream-{dataset}"
    log_file = os.path.join(results_dir, f"{dataset}_{task}_{model}_metrics.csv")
    results_file = os.path.join(results_dir, f"{dataset}_{task}_{model}_results.json")
    
    # Start producer in background
    producer_cmd = [
        'python', 'producer.py',
        '--dataset', dataset,
        '--topic-prefix', 'ml-stream',
        '--stream-rate', str(stream_rate)
    ]
    
    if use_drift_detection:
        producer_cmd.append('--inject-drift')
    
    if max_instances:
        # Note: producer doesn't support max-instances directly, 
        # but we can limit via dataset size or let consumer handle it
        pass
    
    print("Starting producer...")
    producer_process = subprocess.Popen(
        producer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for producer to start and send some messages
    time.sleep(3)
    
    # Start consumer
    consumer_cmd = [
        'python', 'consumer.py',
        '--topic', topic_name,
        '--task', task,
        '--model', model,
        '--log-file', log_file
    ]
    
    if not use_drift_detection:
        consumer_cmd.append('--no-drift-detection')
    
    if max_instances:
        consumer_cmd.extend(['--max-instances', str(max_instances)])
    
    print("Starting consumer...")
    try:
        # Run consumer (it will process messages as they arrive)
        consumer_process = subprocess.run(
            consumer_cmd,
            timeout=3600,  # 1 hour timeout
            capture_output=True,
            text=True
        )
        
        # Give producer a moment to finish if it hasn't
        try:
            producer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Producer still running, terminate it
            producer_process.terminate()
            producer_process.wait()
        
        # Load results if available
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {
                'status': 'completed', 
                'output': consumer_process.stdout,
                'stderr': consumer_process.stderr
            }
            
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        producer_process.terminate()
        try:
            producer_process.wait(timeout=2)
        except:
            producer_process.kill()
        results = {'status': 'timeout'}
    except Exception as e:
        print(f"Error running experiment: {e}")
        producer_process.terminate()
        try:
            producer_process.wait(timeout=2)
        except:
            producer_process.kill()
        results = {'status': 'error', 'error': str(e)}
    
    return {
        'dataset': dataset,
        'task': task,
        'model': model,
        'drift_detection': use_drift_detection,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


def compare_results(results_dir='results', output_file='comparison_report.html'):
    """Compare results across all experiments and generate a report."""
    print("\nGenerating comparison report...")
    
    # Load all result files
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    # Extract metadata from filename
                    parts = filename.replace('_results.json', '').split('_')
                    if len(parts) >= 3:
                        result['dataset'] = parts[0]
                        result['task'] = parts[1]
                        result['model'] = '_'.join(parts[2:])
                    all_results.append(result)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not all_results:
        print("No results found to compare")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        if 'total_instances' in result:
            row = {
                'dataset': result.get('dataset', 'unknown'),
                'task': result.get('task', 'unknown'),
                'model': result.get('model', 'unknown'),
                'total_instances': result.get('total_instances', 0),
                'drift_events': result.get('drift_events', 0)
            }
            
            if result.get('task') == 'classification':
                row['cumulative_accuracy'] = result.get('cumulative_accuracy')
                row['window_accuracy'] = result.get('window_accuracy')
            else:
                row['cumulative_mae'] = result.get('cumulative_mae')
                row['cumulative_mse'] = result.get('cumulative_mse')
                row['window_mae'] = result.get('window_mae')
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
        </style>
    </head>
    <body>
        <h1>Streaming ML Experiment Comparison Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        {df.to_html(index=False, escape=False)}
        
        <h2>Best Models by Dataset</h2>
    """
    
    # Find best models
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        html_content += f"<h3>{dataset}</h3>"
        
        if dataset_df['task'].iloc[0] == 'classification':
            best = dataset_df.loc[dataset_df['cumulative_accuracy'].idxmax()]
            html_content += f"<p><strong>Best Model:</strong> {best['model']} "
            html_content += f"(Accuracy: {best['cumulative_accuracy']:.4f})</p>"
        else:
            best = dataset_df.loc[dataset_df['cumulative_mae'].idxmin()]
            html_content += f"<p><strong>Best Model:</strong> {best['model']} "
            html_content += f"(MAE: {best['cumulative_mae']:.4f})</p>"
    
    html_content += """
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Comparison report saved to {output_file}")
    
    # Also save CSV
    csv_file = output_file.replace('.html', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Comparison data saved to {csv_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Run streaming ML experiments')
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='Experiments to run (format: dataset:task:model)')
    parser.add_argument('--all', action='store_true',
                       help='Run all predefined experiments')
    parser.add_argument('--stream-rate', type=float, default=0.01,
                       help='Streaming rate in seconds')
    parser.add_argument('--max-instances', type=int, default=None,
                       help='Maximum instances per experiment')
    parser.add_argument('--no-drift', action='store_true',
                       help='Disable drift detection')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only generate comparison report from existing results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.compare_only:
        compare_results(args.results_dir)
        return
    
    # Define experiments
    if args.all:
        experiments = [
            ('electricity', 'classification', 'hoeffding_tree'),
            ('electricity', 'classification', 'arf'),
            ('electricity', 'classification', 'knn'),
            ('bike', 'regression', 'fimtdd'),
            ('bike', 'regression', 'arf'),
            ('bike', 'regression', 'knn'),
        ]
    elif args.experiments:
        experiments = []
        for exp in args.experiments:
            parts = exp.split(':')
            if len(parts) == 3:
                experiments.append(tuple(parts))
            else:
                print(f"Invalid experiment format: {exp} (expected dataset:task:model)")
                return
    else:
        print("Please specify --all or --experiments")
        return
    
    # Run experiments
    all_results = []
    for dataset, task, model in experiments:
        result = run_experiment(
            dataset, task, model,
            use_drift_detection=not args.no_drift,
            stream_rate=args.stream_rate,
            max_instances=args.max_instances,
            results_dir=args.results_dir
        )
        all_results.append(result)
        
        # Small delay between experiments
        time.sleep(2)
    
    # Save all results
    results_summary_file = os.path.join(args.results_dir, 'all_experiments.json')
    with open(results_summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiment results saved to {results_summary_file}")
    
    # Generate comparison
    compare_results(args.results_dir)


if __name__ == '__main__':
    main()

