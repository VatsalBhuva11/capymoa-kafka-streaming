"""
Metrics Tracking Module
Tracks rolling and cumulative performance metrics over time.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from river import metrics


class MetricsTracker:
    """Tracks performance metrics for streaming evaluation."""
    
    def __init__(self, task_type: str = 'classification', window_size: int = 1000):
        """
        Args:
            task_type: 'classification' or 'regression'
            window_size: Size of rolling window for metrics
        """
        self.task_type = task_type
        self.window_size = window_size
        
        # Initialize metrics based on task type
        if task_type == 'classification':
            self.cumulative_accuracy = metrics.Accuracy()
            self.cumulative_f1 = metrics.MacroF1()
            self.rolling_accuracy = deque(maxlen=window_size)
            self.rolling_f1 = deque(maxlen=window_size)
        else:  # regression
            self.cumulative_mae = metrics.MAE()
            self.cumulative_mse = metrics.MSE()
            self.rolling_mae = deque(maxlen=window_size)
            self.rolling_mse = deque(maxlen=window_size)
        
        # History for visualization
        self.history = {
            'instance_count': [],
            'cumulative_metrics': [],
            'rolling_metrics': []
        }
        
        self.instance_count = 0
    
    def update(self, y_true, y_pred, y_proba: Optional[Dict] = None):
        """
        Update metrics with a new prediction.
        
        Args:
            y_true: True label/value
            y_pred: Predicted label/value
            y_proba: Prediction probabilities (for classification F1)
        """
        self.instance_count += 1
        
        if self.task_type == 'classification':
            # Update cumulative metrics
            self.cumulative_accuracy.update(y_true, y_pred)
            # For F1, river expects predictions as dict or single value
            if y_proba:
                self.cumulative_f1.update(y_true, y_proba)
            else:
                # Convert single prediction to dict format
                pred_dict = {y_pred: 1.0} if isinstance(y_pred, (int, float)) else y_pred
                self.cumulative_f1.update(y_true, pred_dict)
            
            # Update rolling metrics
            is_correct = 1 if y_true == y_pred else 0
            self.rolling_accuracy.append(is_correct)
            
            # For rolling F1, we need to track recent predictions
            if len(self.rolling_f1) < self.window_size:
                self.rolling_f1.append(is_correct)
            else:
                self.rolling_f1.append(is_correct)
        
        else:  # regression
            error = abs(y_true - y_pred)
            squared_error = (y_true - y_pred) ** 2
            
            # Update cumulative metrics
            self.cumulative_mae.update(y_true, y_pred)
            self.cumulative_mse.update(y_true, y_pred)
            
            # Update rolling metrics
            self.rolling_mae.append(error)
            self.rolling_mse.append(squared_error)
        
        # Store in history periodically
        if self.instance_count % 100 == 0:
            self._update_history()
    
    def _update_history(self):
        """Update history for visualization."""
        self.history['instance_count'].append(self.instance_count)
        
        if self.task_type == 'classification':
            cum_metrics = {
                'accuracy': self.cumulative_accuracy.get(),
                'f1': self.cumulative_f1.get()
            }
            roll_metrics = {
                'accuracy': np.mean(self.rolling_accuracy) if self.rolling_accuracy else 0.0,
                'f1': np.mean(self.rolling_f1) if self.rolling_f1 else 0.0
            }
        else:
            cum_metrics = {
                'mae': self.cumulative_mae.get(),
                'mse': self.cumulative_mse.get(),
                'rmse': np.sqrt(self.cumulative_mse.get())
            }
            roll_metrics = {
                'mae': np.mean(self.rolling_mae) if self.rolling_mae else 0.0,
                'mse': np.mean(self.rolling_mse) if self.rolling_mse else 0.0,
                'rmse': np.sqrt(np.mean(self.rolling_mse)) if self.rolling_mse else 0.0
            }
        
        self.history['cumulative_metrics'].append(cum_metrics)
        self.history['rolling_metrics'].append(roll_metrics)
    
    def get_current_metrics(self) -> Dict:
        """Get current cumulative and rolling metrics."""
        if self.task_type == 'classification':
            return {
                'cumulative': {
                    'accuracy': self.cumulative_accuracy.get(),
                    'f1': self.cumulative_f1.get()
                },
                'rolling': {
                    'accuracy': np.mean(self.rolling_accuracy) if self.rolling_accuracy else 0.0,
                    'f1': np.mean(self.rolling_f1) if self.rolling_f1 else 0.0
                },
                'instance_count': self.instance_count
            }
        else:
            return {
                'cumulative': {
                    'mae': self.cumulative_mae.get(),
                    'mse': self.cumulative_mse.get(),
                    'rmse': np.sqrt(self.cumulative_mse.get())
                },
                'rolling': {
                    'mae': np.mean(self.rolling_mae) if self.rolling_mae else 0.0,
                    'mse': np.mean(self.rolling_mse) if self.rolling_mse else 0.0,
                    'rmse': np.sqrt(np.mean(self.rolling_mse)) if self.rolling_mse else 0.0
                },
                'instance_count': self.instance_count
            }
    
    def get_history(self) -> Dict:
        """Get full history for visualization."""
        return self.history
    
    def reset(self):
        """Reset all metrics."""
        if self.task_type == 'classification':
            self.cumulative_accuracy = metrics.Accuracy()
            self.cumulative_f1 = metrics.MacroF1()
            self.rolling_accuracy = deque(maxlen=self.window_size)
            self.rolling_f1 = deque(maxlen=self.window_size)
        else:
            self.cumulative_mae = metrics.MAE()
            self.cumulative_mse = metrics.MSE()
            self.rolling_mae = deque(maxlen=self.window_size)
            self.rolling_mse = deque(maxlen=self.window_size)
        
        self.history = {
            'instance_count': [],
            'cumulative_metrics': [],
            'rolling_metrics': []
        }
        self.instance_count = 0

