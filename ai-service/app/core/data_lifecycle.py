# ai-service/app/core/data_lifecycle.py
"""
Automated data lifecycle management for privacy compliance
"""

from asyncio.log import logger
from pydoc import text
from app.db.session import get_async_session


class DataLifecycleManager:
    """
    Manages data retention, anonymization, and deletion for privacy compliance
    """
    
    async def cleanup_expired_data(self):
        """Remove data that has exceeded retention periods"""
        try:
            async with get_async_session() as db:
                # Delete expired training data
                delete_query = text("""
                    DELETE FROM user_training_data 
                    WHERE retention_expires_at < CURRENT_TIMESTAMP
                """)
                
                result = await db.execute(delete_query)
                deleted_count = result.rowcount
                
                await db.commit()
                
                logger.info(f"Cleaned up {deleted_count} expired training records")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    async def anonymize_old_data(self):
        """Anonymize data older than specified threshold"""
        try:
            # Move old data to anonymized statistics
            async with get_async_session() as db:
                anonymization_query = text("""
                    INSERT INTO anonymized_task_statistics 
                    (task_category, duration_bucket, priority_level, completion_rate, 
                     time_period, sample_size, noise_applied)
                    SELECT 
                        category,
                        CASE 
                            WHEN actual_duration <= 30 THEN '0-30min'
                            WHEN actual_duration <= 60 THEN '30-60min'
                            WHEN actual_duration <= 120 THEN '60-120min'
                            ELSE '120min+'
                        END as duration_bucket,
                        priority,
                        AVG(CASE WHEN completion_status = 'completed' THEN 1.0 ELSE 0.0 END),
                        CASE 
                            WHEN EXTRACT(hour FROM created_at) BETWEEN 6 AND 12 THEN 'morning'
                            WHEN EXTRACT(hour FROM created_at) BETWEEN 12 AND 18 THEN 'afternoon'
                            ELSE 'evening'
                        END as time_period,
                        COUNT(*),
                        TRUE
                    FROM user_training_data 
                    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
                    AND consent_for_training = TRUE
                    GROUP BY category, duration_bucket, priority, time_period
                    HAVING COUNT(*) >= 5  -- Minimum group size for anonymization
                """)
                
                await db.execute(anonymization_query)
                
                # Delete the individual records after anonymization
                delete_old_query = text("""
                    DELETE FROM user_training_data 
                    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
                """)
                
                await db.execute(delete_old_query)
                await db.commit()
                
                logger.info("Completed data anonymization for old records")
                
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")