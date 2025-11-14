"""
Online Preprocessing Module
Handles per-instance preprocessing suitable for data streams.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from river import preprocessing


class OnlinePreprocessor:
    """Per-instance preprocessing for streaming data."""
    
    def __init__(self, standardize: bool = True, handle_missing: str = 'mean'):
        """
        Args:
            standardize: Whether to standardize features
            handle_missing: Strategy for missing values ('mean', 'median', 'zero', 'drop')
        """
        self.standardize = standardize
        self.handle_missing = handle_missing
        self.scaler = preprocessing.StandardScaler() if standardize else None
        self.feature_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'mean': 0.0})
        self.is_fitted = False
        self.n_features = None
        self.categorical_features = set()
        self.feature_encoders = {}
        
    def detect_categorical(self, features: np.ndarray, threshold: int = 10) -> set:
        """Detect categorical features based on unique value count."""
        categorical = set()
        for i in range(len(features)):
            unique_vals = len(np.unique(features))
            if unique_vals <= threshold:
                categorical.add(i)
        return categorical
    
    def preprocess_instance(self, features: np.ndarray, 
                           is_training: bool = True) -> np.ndarray:
        """
        Preprocess a single instance.
        
        Args:
            features: Feature vector
            is_training: Whether this is training data (for updating statistics)
        
        Returns:
            Preprocessed feature vector
        """
        features = np.array(features, dtype=float)
        
        # Handle missing values
        features = self._handle_missing_values(features, is_training)
        
        # Update feature statistics for standardization
        if is_training:
            self._update_stats(features)
        
        # Standardize if enabled
        if self.standardize and self.is_fitted:
            features = self._standardize(features)
        
        return features
    
    def _handle_missing_values(self, features: np.ndarray, is_training: bool) -> np.ndarray:
        """Handle missing values (NaN or None)."""
        if self.handle_missing == 'drop':
            # Remove features with missing values
            return features[~np.isnan(features)]
        elif self.handle_missing == 'zero':
            return np.nan_to_num(features, nan=0.0)
        elif self.handle_missing in ['mean', 'median']:
            # Use running mean/median
            for i in range(len(features)):
                if np.isnan(features[i]):
                    if self.is_fitted:
                        if self.handle_missing == 'mean':
                            features[i] = self.feature_stats[i]['mean']
                        else:  # median (approximate with mean for now)
                            features[i] = self.feature_stats[i]['mean']
                    else:
                        features[i] = 0.0
            return features
        else:
            return features
    
    def _update_stats(self, features: np.ndarray):
        """Update running statistics for standardization."""
        if self.n_features is None:
            self.n_features = len(features)
        
        for i in range(len(features)):
            if not np.isnan(features[i]):
                stats = self.feature_stats[i]
                stats['count'] += 1
                stats['sum'] += features[i]
                stats['mean'] = stats['sum'] / stats['count']
        
        # Mark as fitted after seeing some data
        if not self.is_fitted and any(s['count'] > 0 for s in self.feature_stats.values()):
            self.is_fitted = True
    
    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using running statistics."""
        standardized = np.zeros_like(features)
        for i in range(len(features)):
            stats = self.feature_stats[i]
            if stats['count'] > 1 and stats['mean'] != 0:
                # Simple standardization: (x - mean) / std
                # For simplicity, we use a running variance approximation
                mean = stats['mean']
                # Use a simple approximation for std (could be improved)
                std = max(1.0, abs(mean) * 0.5)  # Rough approximation
                standardized[i] = (features[i] - mean) / std
            else:
                standardized[i] = features[i]
        return standardized
    
    def reset(self):
        """Reset preprocessor state."""
        self.feature_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'mean': 0.0})
        self.is_fitted = False
        self.n_features = None

