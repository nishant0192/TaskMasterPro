# ai-service/app/core/differential_privacy.py
"""
Differential privacy mechanisms for AI training
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DifferentialPrivacyManager:
    """
    Implements differential privacy for user data protection
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Probability of privacy violation
        
    def add_noise_to_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Add calibrated noise to model gradients for privacy
        """
        try:
            # Calculate sensitivity (max possible change in gradient)
            sensitivity = self._calculate_gradient_sensitivity(gradients)
            
            # Calculate noise scale based on differential privacy parameters
            noise_scale = sensitivity / self.epsilon
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, gradients.shape)
            private_gradients = gradients + noise
            
            logger.debug(f"Added DP noise with scale {noise_scale}")
            return private_gradients
            
        except Exception as e:
            logger.error(f"Failed to add differential privacy noise: {e}")
            return gradients
    
    def privatize_user_statistics(self, user_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Add noise to user statistics before aggregation
        """
        try:
            private_stats = {}
            
            for key, value in user_stats.items():
                # Add Laplace noise proportional to sensitivity
                sensitivity = self._get_statistic_sensitivity(key)
                noise_scale = sensitivity / self.epsilon
                
                noise = np.random.laplace(0, noise_scale)
                private_stats[key] = max(0, value + noise)  # Ensure non-negative
            
            return private_stats
            
        except Exception as e:
            logger.error(f"Failed to privatize statistics: {e}")
            return user_stats
    
    def _calculate_gradient_sensitivity(self, gradients: np.ndarray) -> float:
        """Calculate the sensitivity of gradients for DP noise calibration"""
        # For gradient descent, sensitivity is typically bounded by clipping
        return 1.0  # Assuming gradient clipping is applied
    
    def _get_statistic_sensitivity(self, stat_name: str) -> float:
        """Get sensitivity for different types of statistics"""
        sensitivities = {
            'completion_rate': 1.0,      # Adding/removing one task can change by at most 1
            'average_duration': 1440.0,  # Max task duration in minutes
            'priority_distribution': 1.0,
            'productivity_score': 1.0
        }
        return sensitivities.get(stat_name, 1.0)