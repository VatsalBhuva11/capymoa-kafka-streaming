"""
Incremental Model Training Module
Implements incremental learners for classification and regression.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from river import tree, naive_bayes, linear_model, ensemble
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from river.drift import ADWIN


class IncrementalClassifier:
    """Wrapper for incremental classification models."""
    
    def __init__(self, model_type: str = 'hoeffding_tree', **kwargs):
        """
        Args:
            model_type: Type of model ('hoeffding_tree', 'naive_bayes', 'sgd', 'arf')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_fitted = False
    
    def _create_model(self, model_type: str, **kwargs):
        """Create the appropriate model instance."""
        if model_type == 'hoeffding_tree':
            return tree.HoeffdingTreeClassifier(**kwargs)
        elif model_type == 'naive_bayes':
            return naive_bayes.GaussianNB(**kwargs)
        elif model_type == 'sgd':
            return linear_model.LogisticRegression(**kwargs)
        elif model_type == 'arf':
            return ensemble.AdaptiveRandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self, X: np.ndarray) -> int:
        """Make prediction for a single instance."""
        if not self.is_fitted:
            return 0  # Default prediction
        
        # Convert to dict format for river models
        X_dict = {i: float(x) for i, x in enumerate(X)}
        return self.model.predict_one(X_dict)
    
    def predict_proba(self, X: np.ndarray) -> Dict[int, float]:
        """Get prediction probabilities."""
        if not self.is_fitted:
            return {0: 1.0}
        
        X_dict = {i: float(x) for i, x in enumerate(X)}
        return self.model.predict_proba_one(X_dict)
    
    def partial_fit(self, X: np.ndarray, y: int):
        """Update model with a single instance (test-then-train)."""
        X_dict = {i: float(x) for i, x in enumerate(X)}
        self.model.learn_one(X_dict, y)
        self.is_fitted = True
    
    def reset(self):
        """Reset model to initial state."""
        self.model = self._create_model(self.model_type)
        self.is_fitted = False


class IncrementalRegressor:
    """Wrapper for incremental regression models."""
    
    def __init__(self, model_type: str = 'sgd', **kwargs):
        """
        Args:
            model_type: Type of model ('sgd', 'linear', 'arf')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_fitted = False
    
    def _create_model(self, model_type: str, **kwargs):
        """Create the appropriate model instance."""
        if model_type == 'sgd':
            return linear_model.LinearRegression(**kwargs)
        elif model_type == 'linear':
            return linear_model.PARegressor(**kwargs)
        elif model_type == 'arf':
            return ensemble.AdaptiveRandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self, X: np.ndarray) -> float:
        """Make prediction for a single instance."""
        if not self.is_fitted:
            return 0.0  # Default prediction
        
        # Convert to dict format for river models
        X_dict = {i: float(x) for i, x in enumerate(X)}
        return self.model.predict_one(X_dict)
    
    def partial_fit(self, X: np.ndarray, y: float):
        """Update model with a single instance (test-then-train)."""
        X_dict = {i: float(x) for i, x in enumerate(X)}
        self.model.learn_one(X_dict, y)
        self.is_fitted = True
    
    def reset(self):
        """Reset model to initial state."""
        self.model = self._create_model(self.model_type)
        self.is_fitted = False

