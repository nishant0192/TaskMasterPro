# ai-service/app/core/monitoring.py
"""
Production monitoring and observability for AI service
Handles metrics, logging, alerts, and performance tracking
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import psutil
import GPUtil

# Monitoring libraries
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Internal imports
from app.core.config import get_settings
from app.core.database import get_async_session
from app.models.ai_models import PredictionLog, AITrainingSession

logger = structlog.get_logger(__name__)
settings = get_settings()

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'ai_predictions_total',
    'Total number of AI predictions made',
    ['user_id', 'prediction_type', 'model_version']
)

PREDICTION_LATENCY = Histogram(
    'ai_prediction_duration_seconds',
    'Time spent on AI predictions',
    ['prediction_type']
)

MODEL_ACCURACY = Gauge(
    'ai_model_accuracy',
    'Current model accuracy score',
    ['model_type', 'user_id']
)

ACTIVE_USERS = Gauge(
    'ai_active_users',
    'Number of active users with personalized models'
)

SYSTEM_RESOURCES = Gauge(
    'ai_system_resource_usage',
    'System resource usage',
    ['resource_type']
)

TRAINING_DURATION = Histogram(
    'ai_training_duration_seconds',
    'Time spent on model training',
    ['training_type']
)

ERROR_COUNTER = Counter(
    'ai_errors_total',
    'Total number of AI service errors',
    ['error_type', 'component']
)

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    prediction_latency_p95: float
    prediction_accuracy: float
    active_users: int
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    error_rate: float
    throughput_per_minute: float

@dataclass
class AlertCondition:
    """Alert condition definition"""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_minutes: int
    severity: str  # 'critical', 'warning', 'info'
    description: str

class AIServiceMonitor:
    """
    Comprehensive monitoring system for AI service
    """
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_conditions = []
        self.active_alerts = {}
        self.performance_history = deque(maxlen=100)
        
        # Performance tracking
        self.prediction_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.user_activity = defaultdict(int)
        
        # Health check status
        self.component_health = {
            'ai_engine': 'unknown',
            'personalization_engine': 'unknown',
            'database': 'unknown',
            'models': 'unknown'
        }
        
        self._setup_alert_conditions()
        
    def _setup_alert_conditions(self):
        """Setup default alert conditions"""
        self.alert_conditions = [
            AlertCondition(
                name="high_prediction_latency",
                metric="prediction_latency_p95",
                threshold=5.0,  # 5 seconds
                comparison="gt",
                duration_minutes=5,
                severity="warning",
                description="AI prediction latency is above acceptable threshold"
            ),
            AlertCondition(
                name="low_model_accuracy",
                metric="prediction_accuracy",
                threshold=0.7,
                comparison="lt",
                duration_minutes=10,
                severity="critical",
                description="Model accuracy has dropped below acceptable level"
            ),
            AlertCondition(
                name="high_error_rate",
                metric="error_rate",
                threshold=0.05,  # 5%
                comparison="gt",
                duration_minutes=3,
                severity="critical",
                description="Error rate is above 5%"
            ),
            AlertCondition(
                name="high_memory_usage",
                metric="memory_usage_mb",
                threshold=8192,  # 8GB
                comparison="gt",
                duration_minutes=5,
                severity="warning",
                description="Memory usage is high"
            ),
            AlertCondition(
                name="low_throughput",
                metric="throughput_per_minute",
                threshold=10,
                comparison="lt",
                duration_minutes=10,
                severity="warning",
                description="Prediction throughput is below expected levels"
            )
        ]

    async def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("🔍 Starting AI service monitoring...")
        
        # Start background tasks
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._health_checker())
        asyncio.create_task(self._alert_processor())
        asyncio.create_task(self._performance_analyzer())
        
        logger.info("✅ AI service monitoring started")

    async def _metrics_collector(self):
        """Collect system and application metrics"""
        while True:
            try:
                # Collect system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # GPU metrics (if available)
                gpu_usage = 0.0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                except Exception:
                    pass  # GPU monitoring optional
                
                # Update Prometheus metrics
                SYSTEM_RESOURCES.labels(resource_type='memory_mb').set(
                    memory_info.used / 1024 / 1024
                )
                SYSTEM_RESOURCES.labels(resource_type='cpu_percent').set(cpu_percent)
                SYSTEM_RESOURCES.labels(resource_type='gpu_percent').set(gpu_usage)
                
                # Application metrics
                active_users_count = len(set(self.user_activity.keys()))
                ACTIVE_USERS.set(active_users_count)
                
                # Store performance snapshot
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    prediction_latency_p95=self._calculate_p95_latency(),
                    prediction_accuracy=self._calculate_average_accuracy(),
                    active_users=active_users_count,
                    memory_usage_mb=memory_info.used / 1024 / 1024,
                    cpu_usage_percent=cpu_percent,
                    gpu_usage_percent=gpu_usage,
                    error_rate=self._calculate_error_rate(),
                    throughput_per_minute=self._calculate_throughput()
                )
                
                self.performance_history.append(metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60)

    async def _health_checker(self):
        """Check health of AI service components"""
        while True:
            try:
                # Check database connectivity
                try:
                    async with get_async_session() as db:
                        await db.execute("SELECT 1")
                    self.component_health['database'] = 'healthy'
                except Exception as e:
                    self.component_health['database'] = 'unhealthy'
                    logger.error(f"Database health check failed: {e}")
                
                # Check AI engine components
                # This would check if models are loaded and responding
                self.component_health['ai_engine'] = 'healthy'  # Placeholder
                self.component_health['personalization_engine'] = 'healthy'  # Placeholder
                self.component_health['models'] = 'healthy'  # Placeholder
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)

    async def _alert_processor(self):
        """Process alert conditions and trigger alerts"""
        while True:
            try:
                if not self.performance_history:
                    await asyncio.sleep(30)
                    continue
                
                current_metrics = self.performance_history[-1]
                
                for condition in self.alert_conditions:
                    await self._check_alert_condition(condition, current_metrics)
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"Alert processing failed: {e}")
                await asyncio.sleep(60)

    async def _check_alert_condition(self, condition: AlertCondition, 
                                   metrics: PerformanceMetrics):
        """Check if an alert condition is triggered"""
        try:
            metric_value = getattr(metrics, condition.metric)
            
            # Check condition
            triggered = False
            if condition.comparison == 'gt':
                triggered = metric_value > condition.threshold
            elif condition.comparison == 'lt':
                triggered = metric_value < condition.threshold
            elif condition.comparison == 'eq':
                triggered = metric_value == condition.threshold
            
            alert_key = condition.name
            
            if triggered:
                if alert_key not in self.active_alerts:
                    # New alert
                    self.active_alerts[alert_key] = {
                        'condition': condition,
                        'first_triggered': datetime.now(),
                        'last_triggered': datetime.now(),
                        'trigger_count': 1,
                        'current_value': metric_value
                    }
                    
                    # Check if alert should fire (duration threshold met)
                    await asyncio.sleep(condition.duration_minutes * 60)
                    if alert_key in self.active_alerts:
                        await self._fire_alert(condition, metric_value)
                else:
                    # Update existing alert
                    self.active_alerts[alert_key]['last_triggered'] = datetime.now()
                    self.active_alerts[alert_key]['trigger_count'] += 1
                    self.active_alerts[alert_key]['current_value'] = metric_value
            else:
                # Clear alert if it exists
                if alert_key in self.active_alerts:
                    await self._clear_alert(condition)
                    del self.active_alerts[alert_key]
                    
        except Exception as e:
            logger.error(f"Alert condition check failed: {e}")

    async def _fire_alert(self, condition: AlertCondition, current_value: float):
        """Fire an alert"""
        alert_data = {
            'alert_name': condition.name,
            'severity': condition.severity,
            'description': condition.description,
            'current_value': current_value,
            'threshold': condition.threshold,
            'timestamp': datetime.now().isoformat(),
            'service': 'ai-service'
        }
        
        logger.error(
            f"🚨 ALERT FIRED: {condition.name}",
            alert_data=alert_data
        )
        
        # Here you would integrate with your alerting system
        # Examples: PagerDuty, Slack, email, etc.
        await self._send_alert_notification(alert_data)

    async def _clear_alert(self, condition: AlertCondition):
        """Clear an alert"""
        logger.info(f"✅ Alert cleared: {condition.name}")

    async def _send_alert_notification(self, alert_data: Dict[str, Any]):
        """Send alert notification (integrate with your alerting system)"""
        # Example integrations:
        # - Send to Slack webhook
        # - Send to PagerDuty
        # - Send email
        # - Write to alerting database
        pass

    async def _performance_analyzer(self):
        """Analyze performance trends and generate insights"""
        while True:
            try:
                if len(self.performance_history) < 10:
                    await asyncio.sleep(300)  # Wait for more data
                    continue
                
                # Analyze trends
                recent_metrics = list(self.performance_history)[-10:]
                
                # Calculate trends
                latency_trend = self._calculate_trend([m.prediction_latency_p95 for m in recent_metrics])
                accuracy_trend = self._calculate_trend([m.prediction_accuracy for m in recent_metrics])
                
                # Log performance insights
                if latency_trend > 0.1:  # Increasing latency
                    logger.warning("Performance degradation detected: latency increasing")
                
                if accuracy_trend < -0.05:  # Decreasing accuracy
                    logger.warning("Model performance degradation detected: accuracy decreasing")
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance analysis failed: {e}")
                await asyncio.sleep(300)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of a series of values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def _calculate_p95_latency(self) -> float:
        """Calculate 95th percentile latency"""
        all_times = []
        for times_list in self.prediction_times.values():
            all_times.extend(times_list)
        
        if not all_times:
            return 0.0
        
        sorted_times = sorted(all_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]

    def _calculate_average_accuracy(self) -> float:
        """Calculate average model accuracy across all users"""
        # This would query recent accuracy scores from database
        return 0.85  # Placeholder

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total_requests = sum(self.user_activity.values())
        total_errors = sum(self.error_counts.values())
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests

    def _calculate_throughput(self) -> float:
        """Calculate predictions per minute"""
        # Count predictions in last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_activity = sum(
            count for count in self.user_activity.values()
        )
        return recent_activity

    def record_prediction(self, user_id: str, prediction_type: str, 
                         duration: float, model_version: str = "1.0"):
        """Record a prediction for monitoring"""
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(
            user_id=user_id,
            prediction_type=prediction_type,
            model_version=model_version
        ).inc()
        
        PREDICTION_LATENCY.labels(
            prediction_type=prediction_type
        ).observe(duration)
        
        # Store for local analysis
        self.prediction_times[prediction_type].append(duration)
        self.user_activity[user_id] += 1
        
        # Keep only recent data
        if len(self.prediction_times[prediction_type]) > 100:
            self.prediction_times[prediction_type] = self.prediction_times[prediction_type][-100:]

    def record_error(self, error_type: str, component: str):
        """Record an error for monitoring"""
        ERROR_COUNTER.labels(
            error_type=error_type,
            component=component
        ).inc()
        
        self.error_counts[f"{component}_{error_type}"] += 1

    def record_model_accuracy(self, model_type: str, user_id: str, accuracy: float):
        """Record model accuracy"""
        MODEL_ACCURACY.labels(
            model_type=model_type,
            user_id=user_id
        ).set(accuracy)

    def record_training_duration(self, training_type: str, duration: float):
        """Record training duration"""
        TRAINING_DURATION.labels(
            training_type=training_type
        ).observe(duration)

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'status': 'healthy' if all(
                status == 'healthy' for status in self.component_health.values()
            ) else 'degraded',
            'components': self.component_health,
            'active_alerts': len(self.active_alerts),
            'last_check': datetime.now().isoformat()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {'status': 'insufficient_data'}
        
        latest = self.performance_history[-1]
        
        return {
            'current_metrics': asdict(latest),
            'trends': {
                'latency_trend': self._calculate_trend([
                    m.prediction_latency_p95 for m in self.performance_history
                ]),
                'accuracy_trend': self._calculate_trend([
                    m.prediction_accuracy for m in self.performance_history
                ])
            },
            'active_alerts': len(self.active_alerts),
            'system_health': self.get_health_status()
        }

    async def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("🧹 Cleaning up monitoring system...")
        self.metrics_buffer.clear()
        self.performance_history.clear()
        self.active_alerts.clear()
        logger.info("✅ Monitoring system cleanup complete")

# Global monitor instance
ai_monitor = AIServiceMonitor()

async def get_ai_monitor() -> AIServiceMonitor:
    """Get the global AI monitor instance"""
    return ai_monitor