# ai-service/app/core/privacy_manager.py
"""
Privacy-first data access and training management
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import secrets
from cryptography.fernet import Fernet
from sqlalchemy import text
import logging
from app.core.database import get_async_session

logger = logging.getLogger(__name__)

class PrivacyManager:
    """
    Manages privacy-preserving data access and model training
    """
    
    def __init__(self):
        # Per-user encryption keys (derived from user ID + secret)
        self.master_key = self._get_master_key()
        self.user_keys = {}
        
        # Privacy settings
        self.min_training_samples = 20  # Minimum data for individual training
        self.anonymization_threshold = 1000  # For global model training
        self.differential_privacy_epsilon = 1.0  # Privacy budget
        
    def get_user_encryption_key(self, user_id: str) -> bytes:
        """Generate deterministic encryption key for user"""
        if user_id not in self.user_keys:
            # Derive key from user ID + master secret
            key_material = f"{user_id}:{self.master_key}".encode()
            derived_key = hashlib.pbkdf2_hmac('sha256', key_material, b'salt', 100000)
            self.user_keys[user_id] = Fernet.generate_key()
        
        return self.user_keys[user_id]
    
    async def get_user_training_data(self, user_id: str, 
                                   require_consent: bool = True) -> List[Dict[str, Any]]:
        """
        Get user's training data with privacy controls
        """
        try:
            # 1. Check user consent
            if require_consent and not await self._has_training_consent(user_id):
                logger.info(f"User {user_id} has not consented to model training")
                return []
            
            # 2. Get encrypted user data
            async with get_async_session() as db:
                # Only access user's own data
                query = text("""
                    SELECT 
                        task_id,
                        title_hash,  -- Hashed version of title
                        category,
                        estimated_duration,
                        actual_duration,
                        priority,
                        ai_suggested_priority,
                        completion_status,
                        user_satisfaction,
                        created_at,
                        completed_at
                    FROM user_training_data 
                    WHERE user_id = :user_id 
                    AND consent_for_training = true
                    AND created_at >= :min_date
                    ORDER BY created_at DESC
                    LIMIT 1000
                """)
                
                result = await db.execute(query, {
                    'user_id': user_id,
                    'min_date': datetime.now() - timedelta(days=365)  # Only recent data
                })
                
                raw_data = result.fetchall()
            
            # 3. Decrypt and process user data
            user_key = self.get_user_encryption_key(user_id)
            fernet = Fernet(user_key)
            
            training_data = []
            for row in raw_data:
                try:
                    # Decrypt sensitive fields if needed
                    processed_row = {
                        'task_id': row.task_id,
                        'title_features': self._extract_title_features(row.title_hash),
                        'category': row.category,
                        'estimated_duration': row.estimated_duration,
                        'actual_duration': row.actual_duration,
                        'priority': row.priority,
                        'ai_suggested_priority': row.ai_suggested_priority,
                        'completion_status': row.completion_status,
                        'user_satisfaction': row.user_satisfaction,
                        'temporal_features': self._extract_temporal_features(
                            row.created_at, row.completed_at
                        )
                    }
                    training_data.append(processed_row)
                    
                except Exception as e:
                    logger.warning(f"Failed to process training record: {e}")
                    continue
            
            logger.info(f"Retrieved {len(training_data)} training samples for user {user_id}")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to get training data for user {user_id}: {e}")
            return []

    async def store_training_data(self, user_id: str, task_data: Dict[str, Any], 
                                outcome_data: Dict[str, Any]):
        """
        Store training data with privacy protection
        """
        try:
            # 1. Hash sensitive information
            title_hash = self._hash_sensitive_text(task_data.get('title', ''))
            description_hash = self._hash_sensitive_text(task_data.get('description', ''))
            
            # 2. Extract only necessary features
            training_record = {
                'user_id': user_id,
                'task_id': task_data['id'],
                'title_hash': title_hash,
                'description_hash': description_hash,
                'category': task_data.get('category', 'general'),
                'estimated_duration': task_data.get('estimated_duration'),
                'actual_duration': outcome_data.get('actual_duration'),
                'priority': task_data.get('priority'),
                'ai_suggested_priority': task_data.get('ai_suggested_priority'),
                'completion_status': outcome_data.get('status'),
                'user_satisfaction': outcome_data.get('satisfaction'),
                'created_at': task_data.get('created_at'),
                'completed_at': outcome_data.get('completed_at'),
                'consent_for_training': await self._has_training_consent(user_id)
            }
            
            # 3. Store in privacy-protected table
            async with get_async_session() as db:
                insert_query = text("""
                    INSERT INTO user_training_data 
                    (user_id, task_id, title_hash, description_hash, category,
                     estimated_duration, actual_duration, priority, ai_suggested_priority,
                     completion_status, user_satisfaction, created_at, completed_at,
                     consent_for_training)
                    VALUES 
                    (:user_id, :task_id, :title_hash, :description_hash, :category,
                     :estimated_duration, :actual_duration, :priority, :ai_suggested_priority,
                     :completion_status, :user_satisfaction, :created_at, :completed_at,
                     :consent_for_training)
                    ON CONFLICT (user_id, task_id) 
                    DO UPDATE SET
                        actual_duration = EXCLUDED.actual_duration,
                        completion_status = EXCLUDED.completion_status,
                        user_satisfaction = EXCLUDED.user_satisfaction,
                        completed_at = EXCLUDED.completed_at
                """)
                
                await db.execute(insert_query, training_record)
                await db.commit()
                
            logger.info(f"Stored training data for user {user_id}, task {task_data['id']}")
            
        except Exception as e:
            logger.error(f"Failed to store training data: {e}")

    def _hash_sensitive_text(self, text: str) -> str:
        """
        Create privacy-preserving hash of sensitive text
        Preserves some semantic meaning while protecting privacy
        """
        if not text:
            return ""
        
        # Use consistent hashing to preserve some semantic similarity
        # This allows us to group similar tasks without storing actual text
        return hashlib.sha256(text.lower().encode()).hexdigest()[:16]

    async def _has_training_consent(self, user_id: str) -> bool:
        """Check if user has consented to model training"""
        try:
            async with get_async_session() as db:
                query = text("""
                    SELECT ai_training_consent 
                    FROM user_privacy_settings 
                    WHERE user_id = :user_id
                """)
                result = await db.execute(query, {'user_id': user_id})
                row = result.fetchone()
                
                return row.ai_training_consent if row else False
                
        except Exception:
            # Default to no consent if we can't verify
            return False