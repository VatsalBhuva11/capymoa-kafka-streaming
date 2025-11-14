"""
Concept Drift Handling Module
Injects and detects concept drift in streaming data.
"""

import numpy as np
from typing import Dict, List, Optional
from river.drift import ADWIN, DDM
import json


class DriftInjector:
    """Injects concept drift into data streams."""
    
    @staticmethod
    def inject_sudden_drift(features: np.ndarray, target: float, 
                           drift_type: str = 'feature_shift',
                           magnitude: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Inject sudden concept drift.
        
        Args:
            features: Original features
            target: Original target
            drift_type: Type of drift ('feature_shift', 'target_shift', 'noise_increase')
            magnitude: Magnitude of drift
        
        Returns:
            Modified (features, target)
        """
        features = np.array(features).copy()
        
        if drift_type == 'feature_shift':
            # Shift all features
            features = features + magnitude * np.random.randn(len(features))
        elif drift_type == 'target_shift':
            # Shift target value
            target = target + magnitude * np.random.randn()
        elif drift_type == 'noise_increase':
            # Increase noise in features
            features = features + magnitude * 0.5 * np.random.randn(len(features))
        
        return features, float(target)
    
    @staticmethod
    def inject_gradual_drift(features: np.ndarray, target: float,
                            progress: float, drift_type: str = 'feature_shift',
                            magnitude: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Inject gradual concept drift.
        
        Args:
            features: Original features
            target: Original target
            progress: Progress of drift (0.0 to 1.0)
            drift_type: Type of drift
            magnitude: Maximum magnitude of drift
        
        Returns:
            Modified (features, target)
        """
        # Scale magnitude by progress
        effective_magnitude = magnitude * progress
        return DriftInjector.inject_sudden_drift(features, target, drift_type, effective_magnitude)


class DriftDetector:
    """Detects concept drift in streaming data."""
    
    def __init__(self, detector_type: str = 'adwin', **kwargs):
        """
        Args:
            detector_type: 'adwin' or 'ddm'
        """
        self.detector_type = detector_type
        self.detector = self._create_detector(detector_type, **kwargs)
        self.drift_events = []
        self.instance_count = 0
    
    def _create_detector(self, detector_type: str, **kwargs):
        """Create drift detector instance."""
        if detector_type == 'adwin':
            return ADWIN(delta=kwargs.get('delta', 0.002))
        elif detector_type == 'ddm':
            return DDM(**kwargs)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def update(self, error: float):
        """
        Update detector with prediction error.
        
        Args:
            error: Prediction error (absolute difference)
        
        Returns:
            True if drift detected, False otherwise
        """
        self.instance_count += 1
        self.detector.update(error)
        
        if hasattr(self.detector, 'drift_detected') and self.detector.drift_detected:
            drift_event = {
                'instance': self.instance_count,
                'error': error,
                'detector_type': self.detector_type
            }
            self.drift_events.append(drift_event)
            return True
        
        return False
    
    def get_drift_events(self) -> List[Dict]:
        """Get list of detected drift events."""
        return self.drift_events
    
    def reset(self):
        """Reset detector state."""
        self.detector = self._create_detector(self.detector_type)
        self.drift_events = []
        self.instance_count = 0

