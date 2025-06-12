# ai-service/app/core/federated_learning.py
"""
Federated learning implementation for privacy-preserving global model training
"""

from asyncio.log import logger
from typing import Any, Dict, List
from core.differential_privacy import DifferentialPrivacyManager

import numpy as np


class FederatedLearningManager:
    """
    Coordinates federated learning across user models
    """
    
    def __init__(self):
        self.dp_manager = DifferentialPrivacyManager()
        self.min_participants = 10  # Minimum users for global update
        
    async def aggregate_user_models(self, user_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate user model updates using federated averaging with privacy
        """
        try:
            if len(user_models) < self.min_participants:
                logger.warning(f"Insufficient participants for federated learning: {len(user_models)}")
                return None
            
            # 1. Extract model weights from each user
            all_weights = []
            user_contributions = []
            
            for user_model in user_models:
                if self._validate_user_consent(user_model['user_id']):
                    weights = user_model['model_weights']
                    
                    # Apply differential privacy to individual contributions
                    private_weights = self.dp_manager.add_noise_to_gradients(weights)
                    
                    all_weights.append(private_weights)
                    user_contributions.append(user_model['training_samples'])
            
            if len(all_weights) < self.min_participants:
                return None
            
            # 2. Weighted federated averaging
            total_samples = sum(user_contributions)
            aggregated_weights = np.zeros_like(all_weights[0])
            
            for weights, samples in zip(all_weights, user_contributions):
                weight_factor = samples / total_samples
                aggregated_weights += weight_factor * weights
            
            # 3. Additional privacy protection for aggregated model
            final_weights = self.dp_manager.add_noise_to_gradients(aggregated_weights)
            
            logger.info(f"Aggregated {len(all_weights)} user models with DP protection")
            
            return {
                'aggregated_weights': final_weights,
                'participants': len(all_weights),
                'total_samples': total_samples,
                'privacy_budget_used': self.dp_manager.epsilon
            }
            
        except Exception as e:
            logger.error(f"Federated aggregation failed: {e}")
            return None
    
    def _validate_user_consent(self, user_id: str) -> bool:
        """Validate that user has consented to federated learning"""
        # Check user's privacy settings
        return True  # Placeholder - would check actual consent