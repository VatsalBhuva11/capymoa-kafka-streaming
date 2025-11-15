"""
Drift Analysis and Visualization Tool
Analyzes drift patterns (mean shift, class imbalance) and correlates drift events
with accuracy/performance drops across models.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
from datetime import datetime
import argparse


class DriftAnalyzer:
    """Analyze drift patterns and their impact on model performance."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.drift_data = []
        self.metrics_data = []
        
    def load_results(self):
        """Load all results files."""
        # Load JSON results files for drift events
        json_files = glob.glob(os.path.join(self.results_dir, '*_results.json'))
        for filepath in json_files:
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    filename = os.path.basename(filepath)
                    parts = filename.replace('_results.json', '').split('_')
                    if len(parts) >= 3:
                        dataset = parts[0]
                        task = parts[1]
                        model = '_'.join(parts[2:])
                        
                        if 'drift_details' in results and results['drift_details']:
                            for drift in results['drift_details']:
                                self.drift_data.append({
                                    'dataset': dataset,
                                    'task': task,
                                    'model': model,
                                    'instance_id': drift.get('instance_id', 0),
                                    'timestamp': drift.get('timestamp', ''),
                                    'file': filename
                                })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Load CSV metrics files
        csv_files = glob.glob(os.path.join(self.results_dir, '*_metrics.csv'))
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
                filename = os.path.basename(filepath)
                parts = filename.replace('_metrics.csv', '').split('_')
                if len(parts) >= 3:
                    dataset = parts[0]
                    task = parts[1]
                    model = '_'.join(parts[2:])
                    
                    df['dataset'] = dataset
                    df['task'] = task
                    df['model'] = model
                    df['file'] = filename
                    self.metrics_data.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        if not self.metrics_data:
            print("No metrics data found!")
            return
        
        self.metrics_df = pd.concat(self.metrics_data, ignore_index=True)
        self.drift_df = pd.DataFrame(self.drift_data) if self.drift_data else pd.DataFrame()
        
    def analyze_mean_shift(self, window_size=500):
        """Analyze mean shift patterns in target values around drift events."""
        if self.drift_df.empty:
            print("No drift events found for mean shift analysis")
            return None
        
        mean_shift_analysis = []
        
        for _, drift_row in self.drift_df.iterrows():
            dataset = drift_row['dataset']
            task = drift_row['task']
            model = drift_row['model']
            drift_instance = drift_row['instance_id']
            
            # Get metrics for this model
            model_metrics = self.metrics_df[
                (self.metrics_df['dataset'] == dataset) &
                (self.metrics_df['task'] == task) &
                (self.metrics_df['model'] == model)
            ].copy()
            
            if model_metrics.empty:
                continue
            
            # Get window before and after drift
            before_window = model_metrics[
                (model_metrics['instance'] >= drift_instance - window_size) &
                (model_metrics['instance'] < drift_instance)
            ]
            after_window = model_metrics[
                (model_metrics['instance'] > drift_instance) &
                (model_metrics['instance'] <= drift_instance + window_size)
            ]
            
            if task == 'classification':
                metric_col = 'cumulative_accuracy'
            else:
                metric_col = 'cumulative_mae'
            
            if metric_col in model_metrics.columns:
                before_mean = pd.to_numeric(
                    before_window[metric_col].replace('N/A', None), errors='coerce'
                ).mean() if not before_window.empty else None
                
                after_mean = pd.to_numeric(
                    after_window[metric_col].replace('N/A', None), errors='coerce'
                ).mean() if not after_window.empty else None
                
                if before_mean is not None and after_mean is not None:
                    if task == 'classification':
                        shift = after_mean - before_mean  # Accuracy drop
                    else:
                        shift = after_mean - before_mean  # MAE increase
                    
                    mean_shift_analysis.append({
                        'dataset': dataset,
                        'task': task,
                        'model': model,
                        'drift_instance': drift_instance,
                        'before_mean': before_mean,
                        'after_mean': after_mean,
                        'shift': shift,
                        'shift_magnitude': abs(shift),
                        'shift_type': 'decrease' if shift < 0 else 'increase'
                    })
        
        return pd.DataFrame(mean_shift_analysis)
    
    def analyze_class_imbalance(self, window_size=1000):
        """Analyze class imbalance patterns for classification tasks."""
        if self.drift_df.empty:
            print("No drift events found for class imbalance analysis")
            return None
        
        imbalance_analysis = []
        
        # This would require access to actual class distributions
        # For now, we'll analyze accuracy patterns which can indicate imbalance
        classification_drifts = self.drift_df[self.drift_df['task'] == 'classification']
        
        for _, drift_row in classification_drifts.iterrows():
            dataset = drift_row['dataset']
            model = drift_row['model']
            drift_instance = drift_row['instance_id']
            
            model_metrics = self.metrics_df[
                (self.metrics_df['dataset'] == dataset) &
                (self.metrics_df['task'] == 'classification') &
                (self.metrics_df['model'] == model)
            ].copy()
            
            if model_metrics.empty or 'cumulative_accuracy' not in model_metrics.columns:
                continue
            
            # Analyze accuracy variance before and after drift
            before_window = model_metrics[
                (model_metrics['instance'] >= drift_instance - window_size) &
                (model_metrics['instance'] < drift_instance)
            ]
            after_window = model_metrics[
                (model_metrics['instance'] > drift_instance) &
                (model_metrics['instance'] <= drift_instance + window_size)
            ]
            
            before_acc = pd.to_numeric(
                before_window['cumulative_accuracy'].replace('N/A', None), errors='coerce'
            )
            after_acc = pd.to_numeric(
                after_window['cumulative_accuracy'].replace('N/A', None), errors='coerce'
            )
            
            if not before_acc.empty and not after_acc.empty:
                before_std = before_acc.std()
                after_std = after_acc.std()
                before_mean = before_acc.mean()
                after_mean = after_acc.mean()
                
                imbalance_analysis.append({
                    'dataset': dataset,
                    'model': model,
                    'drift_instance': drift_instance,
                    'before_mean_acc': before_mean,
                    'after_mean_acc': after_mean,
                    'before_std': before_std,
                    'after_std': after_std,
                    'accuracy_drop': before_mean - after_mean,
                    'variance_change': after_std - before_std
                })
        
        return pd.DataFrame(imbalance_analysis)
    
    def correlate_drift_with_performance(self):
        """Correlate drift events with performance drops."""
        if self.drift_df.empty:
            print("No drift events found")
            return None
        
        correlations = []
        
        for _, drift_row in self.drift_df.iterrows():
            dataset = drift_row['dataset']
            task = drift_row['task']
            model = drift_row['model']
            drift_instance = drift_row['instance_id']
            
            model_metrics = self.metrics_df[
                (self.metrics_df['dataset'] == dataset) &
                (self.metrics_df['task'] == task) &
                (self.metrics_df['model'] == model)
            ].copy()
            
            if model_metrics.empty:
                continue
            
            # Get metrics around drift point
            if task == 'classification':
                metric_col = 'cumulative_accuracy'
                window_col = 'window_accuracy'
            else:
                metric_col = 'cumulative_mae'
                window_col = 'window_mae'
            
            if metric_col not in model_metrics.columns:
                continue
            
            # Convert to numeric
            model_metrics[metric_col] = pd.to_numeric(
                model_metrics[metric_col].replace('N/A', None), errors='coerce'
            )
            if window_col in model_metrics.columns:
                model_metrics[window_col] = pd.to_numeric(
                    model_metrics[window_col].replace('N/A', None), errors='coerce'
                )
            
            # Get performance before and after drift
            before_instances = model_metrics[model_metrics['instance'] < drift_instance]
            after_instances = model_metrics[model_metrics['instance'] > drift_instance]
            
            if not before_instances.empty and not after_instances.empty:
                before_perf = before_instances[metric_col].iloc[-1] if len(before_instances) > 0 else None
                after_perf = after_instances[metric_col].iloc[0] if len(after_instances) > 0 else None
                
                # Calculate recovery (performance after some instances)
                recovery_window = after_instances.head(500)
                if not recovery_window.empty:
                    recovery_perf = recovery_window[metric_col].iloc[-1]
                else:
                    recovery_perf = after_perf
                
                if before_perf is not None and after_perf is not None:
                    if task == 'classification':
                        # For classification, lower is worse
                        immediate_drop = before_perf - after_perf
                        recovery_amount = recovery_perf - after_perf
                    else:
                        # For regression MAE, higher is worse
                        immediate_drop = after_perf - before_perf
                        recovery_amount = after_perf - recovery_perf
                    
                    correlations.append({
                        'dataset': dataset,
                        'task': task,
                        'model': model,
                        'drift_instance': drift_instance,
                        'before_performance': before_perf,
                        'immediate_after_performance': after_perf,
                        'recovery_performance': recovery_perf,
                        'immediate_drop': immediate_drop,
                        'recovery_amount': recovery_amount,
                        'recovery_time': len(recovery_window) if not recovery_window.empty else 0
                    })
        
        return pd.DataFrame(correlations)
    
    def create_visualizations(self, output_dir='results'):
        """Create comprehensive drift analysis visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # 1. Drift events timeline
        if not self.drift_df.empty:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Timeline of drift events
            for idx, (dataset, task) in enumerate(self.drift_df[['dataset', 'task']].drop_duplicates().values):
                ax = axes[idx] if idx < 2 else axes[0]
                subset = self.drift_df[
                    (self.drift_df['dataset'] == dataset) &
                    (self.drift_df['task'] == task)
                ]
                
                for model in subset['model'].unique():
                    model_drifts = subset[subset['model'] == model]
                    ax.scatter(
                        model_drifts['instance_id'],
                        [model] * len(model_drifts),
                        label=f"{model}",
                        s=100,
                        alpha=0.7
                    )
                
                ax.set_xlabel('Instance Number', fontsize=12, color='white')
                ax.set_ylabel('Model', fontsize=12, color='white')
                ax.set_title(f'Drift Events Timeline - {dataset} ({task})', 
                           fontsize=14, fontweight='bold', color='white')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#1a1f2e')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'drift_events_timeline.png'), 
                       dpi=300, facecolor='#0f1419', bbox_inches='tight')
            plt.close()
        
        # 2. Performance drop correlation
        correlations = self.correlate_drift_with_performance()
        if correlations is not None and not correlations.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Drift Impact on Model Performance', 
                        fontsize=16, fontweight='bold', color='white')
            
            # Immediate drop by model
            ax1 = axes[0, 0]
            if 'immediate_drop' in correlations.columns:
                sns.boxplot(data=correlations, x='model', y='immediate_drop', ax=ax1)
                ax1.set_title('Immediate Performance Drop by Model', color='white')
                ax1.set_xlabel('Model', color='white')
                ax1.set_ylabel('Performance Drop', color='white')
                ax1.set_facecolor('#1a1f2e')
                ax1.tick_params(colors='white')
            
            # Recovery time by model
            ax2 = axes[0, 1]
            if 'recovery_time' in correlations.columns:
                sns.boxplot(data=correlations, x='model', y='recovery_time', ax=ax2)
                ax2.set_title('Recovery Time by Model', color='white')
                ax2.set_xlabel('Model', color='white')
                ax2.set_ylabel('Recovery Time (instances)', color='white')
                ax2.set_facecolor('#1a1f2e')
                ax2.tick_params(colors='white')
            
            # Performance trajectory around drift
            ax3 = axes[1, 0]
            for model in correlations['model'].unique()[:3]:  # Limit to 3 models for clarity
                model_corr = correlations[correlations['model'] == model]
                if not model_corr.empty:
                    ax3.scatter(
                        model_corr['drift_instance'],
                        model_corr['immediate_drop'],
                        label=model,
                        alpha=0.6,
                        s=100
                    )
            ax3.set_xlabel('Drift Instance', color='white')
            ax3.set_ylabel('Immediate Performance Drop', color='white')
            ax3.set_title('Performance Drop vs Drift Timing', color='white')
            ax3.legend()
            ax3.set_facecolor('#1a1f2e')
            ax3.tick_params(colors='white')
            ax3.grid(True, alpha=0.3)
            
            # Recovery amount
            ax4 = axes[1, 1]
            if 'recovery_amount' in correlations.columns:
                sns.violinplot(data=correlations, x='model', y='recovery_amount', ax=ax4)
                ax4.set_title('Recovery Amount by Model', color='white')
                ax4.set_xlabel('Model', color='white')
                ax4.set_ylabel('Recovery Amount', color='white')
                ax4.set_facecolor('#1a1f2e')
                ax4.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'drift_performance_correlation.png'), 
                       dpi=300, facecolor='#0f1419', bbox_inches='tight')
            plt.close()
        
        # 3. Mean shift analysis
        mean_shift = self.analyze_mean_shift()
        if mean_shift is not None and not mean_shift.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Mean Shift Analysis Around Drift Events', 
                        fontsize=16, fontweight='bold', color='white')
            
            # Shift magnitude by model
            ax1 = axes[0, 0]
            sns.boxplot(data=mean_shift, x='model', y='shift_magnitude', ax=ax1)
            ax1.set_title('Mean Shift Magnitude by Model', color='white')
            ax1.set_xlabel('Model', color='white')
            ax1.set_ylabel('Shift Magnitude', color='white')
            ax1.set_facecolor('#1a1f2e')
            ax1.tick_params(colors='white')
            
            # Before vs After comparison
            ax2 = axes[0, 1]
            for model in mean_shift['model'].unique():
                model_data = mean_shift[mean_shift['model'] == model]
                ax2.scatter(
                    model_data['before_mean'],
                    model_data['after_mean'],
                    label=model,
                    alpha=0.6,
                    s=100
                )
            # Add diagonal line
            min_val = min(mean_shift['before_mean'].min(), mean_shift['after_mean'].min())
            max_val = max(mean_shift['before_mean'].max(), mean_shift['after_mean'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No change')
            ax2.set_xlabel('Before Drift (Mean Performance)', color='white')
            ax2.set_ylabel('After Drift (Mean Performance)', color='white')
            ax2.set_title('Performance Before vs After Drift', color='white')
            ax2.legend()
            ax2.set_facecolor('#1a1f2e')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            
            # Shift distribution
            ax3 = axes[1, 0]
            sns.histplot(data=mean_shift, x='shift', hue='model', ax=ax3, kde=True)
            ax3.set_title('Distribution of Performance Shifts', color='white')
            ax3.set_xlabel('Performance Shift', color='white')
            ax3.set_ylabel('Frequency', color='white')
            ax3.set_facecolor('#1a1f2e')
            ax3.tick_params(colors='white')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Shift by drift instance
            ax4 = axes[1, 1]
            for model in mean_shift['model'].unique():
                model_data = mean_shift[mean_shift['model'] == model]
                ax4.scatter(
                    model_data['drift_instance'],
                    model_data['shift'],
                    label=model,
                    alpha=0.6,
                    s=100
                )
            ax4.set_xlabel('Drift Instance', color='white')
            ax4.set_ylabel('Performance Shift', color='white')
            ax4.set_title('Shift Magnitude vs Drift Timing', color='white')
            ax4.legend()
            ax4.set_facecolor('#1a1f2e')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mean_shift_analysis.png'), 
                       dpi=300, facecolor='#0f1419', bbox_inches='tight')
            plt.close()
        
        # 4. Detailed performance trajectory with drift markers
        if not self.metrics_df.empty and not self.drift_df.empty:
            fig, axes = plt.subplots(len(self.metrics_df['model'].unique()), 1, 
                                   figsize=(16, 6 * len(self.metrics_df['model'].unique())))
            if len(self.metrics_df['model'].unique()) == 1:
                axes = [axes]
            
            for idx, model in enumerate(self.metrics_df['model'].unique()):
                ax = axes[idx]
                model_metrics = self.metrics_df[self.metrics_df['model'] == model].copy()
                
                # Determine metric column
                if 'cumulative_accuracy' in model_metrics.columns:
                    metric_col = 'cumulative_accuracy'
                    ylabel = 'Cumulative Accuracy'
                elif 'cumulative_mae' in model_metrics.columns:
                    metric_col = 'cumulative_mae'
                    ylabel = 'Cumulative MAE'
                else:
                    continue
                
                # Convert to numeric
                model_metrics[metric_col] = pd.to_numeric(
                    model_metrics[metric_col].replace('N/A', None), errors='coerce'
                )
                
                # Plot performance
                for dataset in model_metrics['dataset'].unique():
                    dataset_metrics = model_metrics[model_metrics['dataset'] == dataset]
                    ax.plot(
                        dataset_metrics['instance'],
                        dataset_metrics[metric_col],
                        label=f"{dataset}",
                        linewidth=2,
                        alpha=0.8
                    )
                
                # Mark drift events
                model_drifts = self.drift_df[self.drift_df['model'] == model]
                for _, drift in model_drifts.iterrows():
                    ax.axvline(
                        x=drift['instance_id'],
                        color='red',
                        linestyle='--',
                        linewidth=2,
                        alpha=0.7,
                        label='Drift' if drift.name == model_drifts.index[0] else ''
                    )
                    # Add annotation
                    ax.annotate(
                        f"Drift\n@{drift['instance_id']}",
                        xy=(drift['instance_id'], ax.get_ylim()[1] * 0.95),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                    )
                
                ax.set_xlabel('Instance Number', fontsize=12, color='white')
                ax.set_ylabel(ylabel, fontsize=12, color='white')
                ax.set_title(f'Performance Trajectory: {model}', 
                           fontsize=14, fontweight='bold', color='white')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#1a1f2e')
                ax.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_trajectory_with_drifts.png'), 
                       dpi=300, facecolor='#0f1419', bbox_inches='tight')
            plt.close()
        
        # 5. Class imbalance analysis (for classification)
        imbalance = self.analyze_class_imbalance()
        if imbalance is not None and not imbalance.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Class Imbalance Analysis (Classification)', 
                        fontsize=16, fontweight='bold', color='white')
            
            # Accuracy drop distribution
            ax1 = axes[0, 0]
            sns.histplot(data=imbalance, x='accuracy_drop', hue='model', ax=ax1, kde=True)
            ax1.set_title('Accuracy Drop Distribution', color='white')
            ax1.set_xlabel('Accuracy Drop', color='white')
            ax1.set_ylabel('Frequency', color='white')
            ax1.set_facecolor('#1a1f2e')
            ax1.tick_params(colors='white')
            
            # Variance change
            ax2 = axes[0, 1]
            sns.boxplot(data=imbalance, x='model', y='variance_change', ax=ax2)
            ax2.set_title('Variance Change After Drift', color='white')
            ax2.set_xlabel('Model', color='white')
            ax2.set_ylabel('Variance Change', color='white')
            ax2.set_facecolor('#1a1f2e')
            ax2.tick_params(colors='white')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Before vs After accuracy
            ax3 = axes[1, 0]
            for model in imbalance['model'].unique():
                model_data = imbalance[imbalance['model'] == model]
                ax3.scatter(
                    model_data['before_mean_acc'],
                    model_data['after_mean_acc'],
                    label=model,
                    alpha=0.6,
                    s=100
                )
            min_acc = min(imbalance['before_mean_acc'].min(), imbalance['after_mean_acc'].min())
            max_acc = max(imbalance['before_mean_acc'].max(), imbalance['after_mean_acc'].max())
            ax3.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5)
            ax3.set_xlabel('Before Drift Accuracy', color='white')
            ax3.set_ylabel('After Drift Accuracy', color='white')
            ax3.set_title('Accuracy Before vs After Drift', color='white')
            ax3.legend()
            ax3.set_facecolor('#1a1f2e')
            ax3.tick_params(colors='white')
            ax3.grid(True, alpha=0.3)
            
            # Accuracy drop by drift instance
            ax4 = axes[1, 1]
            for model in imbalance['model'].unique():
                model_data = imbalance[imbalance['model'] == model]
                ax4.scatter(
                    model_data['drift_instance'],
                    model_data['accuracy_drop'],
                    label=model,
                    alpha=0.6,
                    s=100
                )
            ax4.set_xlabel('Drift Instance', color='white')
            ax4.set_ylabel('Accuracy Drop', color='white')
            ax4.set_title('Accuracy Drop vs Drift Timing', color='white')
            ax4.legend()
            ax4.set_facecolor('#1a1f2e')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_imbalance_analysis.png'), 
                       dpi=300, facecolor='#0f1419', bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to {output_dir}/")
        print("Generated files:")
        print("  - drift_events_timeline.png")
        print("  - drift_performance_correlation.png")
        print("  - mean_shift_analysis.png")
        print("  - performance_trajectory_with_drifts.png")
        if imbalance is not None and not imbalance.empty:
            print("  - class_imbalance_analysis.png")
    
    def generate_report(self, output_file='results/drift_analysis_report.txt'):
        """Generate a text report of drift analysis."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DRIFT ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total drift events: {len(self.drift_df)}\n")
            if not self.drift_df.empty:
                f.write(f"Datasets analyzed: {self.drift_df['dataset'].nunique()}\n")
                f.write(f"Models analyzed: {self.drift_df['model'].nunique()}\n")
                f.write(f"Tasks analyzed: {self.drift_df['task'].nunique()}\n\n")
            
            # Mean shift analysis
            mean_shift = self.analyze_mean_shift()
            if mean_shift is not None and not mean_shift.empty:
                f.write("MEAN SHIFT ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average shift magnitude: {mean_shift['shift_magnitude'].mean():.4f}\n")
                f.write(f"Max shift magnitude: {mean_shift['shift_magnitude'].max():.4f}\n")
                f.write(f"Min shift magnitude: {mean_shift['shift_magnitude'].min():.4f}\n\n")
                
                f.write("Shift by Model:\n")
                for model in mean_shift['model'].unique():
                    model_shifts = mean_shift[mean_shift['model'] == model]
                    f.write(f"  {model}: {model_shifts['shift_magnitude'].mean():.4f} "
                           f"(std: {model_shifts['shift_magnitude'].std():.4f})\n")
                f.write("\n")
            
            # Performance correlation
            correlations = self.correlate_drift_with_performance()
            if correlations is not None and not correlations.empty:
                f.write("PERFORMANCE CORRELATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average immediate drop: {correlations['immediate_drop'].mean():.4f}\n")
                f.write(f"Average recovery amount: {correlations['recovery_amount'].mean():.4f}\n")
                f.write(f"Average recovery time: {correlations['recovery_time'].mean():.1f} instances\n\n")
                
                f.write("Performance Impact by Model:\n")
                for model in correlations['model'].unique():
                    model_corr = correlations[correlations['model'] == model]
                    f.write(f"  {model}:\n")
                    f.write(f"    Avg immediate drop: {model_corr['immediate_drop'].mean():.4f}\n")
                    f.write(f"    Avg recovery time: {model_corr['recovery_time'].mean():.1f} instances\n")
                f.write("\n")
            
            # Class imbalance (if applicable)
            imbalance = self.analyze_class_imbalance()
            if imbalance is not None and not imbalance.empty:
                f.write("CLASS IMBALANCE ANALYSIS (Classification)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average accuracy drop: {imbalance['accuracy_drop'].mean():.4f}\n")
                f.write(f"Average variance change: {imbalance['variance_change'].mean():.4f}\n\n")
        
        print(f"\nReport saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze drift patterns and performance impacts')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing results files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save visualizations')
    parser.add_argument('--report', type=str, default='results/drift_analysis_report.txt',
                       help='Path to save text report')
    
    args = parser.parse_args()
    
    analyzer = DriftAnalyzer(results_dir=args.results_dir)
    print("Loading results...")
    analyzer.load_results()
    
    print("Analyzing drift patterns...")
    analyzer.create_visualizations(output_dir=args.output_dir)
    
    print("Generating report...")
    analyzer.generate_report(output_file=args.report)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

