# ai-service/app/core/monitoring.py
"""
AI Service Monitoring and Metrics Collection
Production-ready monitoring with performance tracking and alerting
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    value: float
    operation: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ErrorEvent:
    """Error event data"""
    timestamp: datetime
    error_type: str
    component: str
    message: str
    user_id: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class HealthStatus:
    """Component health status"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: float
    error_count: int
    details: Dict[str, Any] = None

class AIServiceMonitor:
    """
    Comprehensive monitoring for AI service components
    Features:
    - Performance metrics collection
    - Error tracking and alerting
    - Health status monitoring
    - Resource usage tracking
    - SLA monitoring
    """
    
    def __init__(self):
        # Metrics storage
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_events: deque = deque(maxlen=500)
        self.health_statuses: Dict[str, HealthStatus] = {}
        
        # Configuration
        self.metrics_retention_hours = 24
        self.health_check_interval = 300  # 5 minutes
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate threshold
            'response_time_p95': 2.0,  # 2 second p95 response time
            'availability': 0.99  # 99% availability
        }
        
        # Counters
        self.request_counter = 0
        self.error_counter = 0
        self.start_time = time.time()
        
        # Resource tracking
        self.resource_usage = {
            'memory_mb': 0,
            'cpu_percent': 0,
            'disk_usage_mb': 0,
            'active_connections': 0
        }
        
        # SLA tracking
        self.sla_metrics = {
            'availability': 1.0,
            'error_rate': 0.0,
            'avg_response_time': 0.0,
            'requests_per_minute': 0.0
        }
        
        logger.info("📊 AI Service Monitor initialized")
    
    def record_performance_metric(self, operation: str, value: float, user_id: Optional[str] = None, **metadata):
        """Record a performance metric"""
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                value=value,
                operation=operation,
                user_id=user_id,
                metadata=metadata
            )
            
            self.performance_metrics[operation].append(metric)
            
            # Update counters
            if operation == 'request_duration':
                self.request_counter += 1
            
            # Clean old metrics
            self._cleanup_old_metrics()
            
        except Exception as e:
            logger.error(f"❌ Failed to record performance metric: {e}")
    
    def record_error(self, error_type: str, component: str, message: str = "", 
                    user_id: Optional[str] = None, stack_trace: Optional[str] = None):
        """Record an error event"""
        try:
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                error_type=error_type,
                component=component,
                message=message,
                user_id=user_id,
                stack_trace=stack_trace
            )
            
            self.error_events.append(error_event)
            self.error_counter += 1
            
            # Log error
            logger.warning(
                f"🚨 Error recorded: {error_type} in {component}",
                extra={
                    'error_type': error_type,
                    'component': component,
                    'user_id': user_id
                }
            )
            
            # Check for alert conditions
            self._check_alert_conditions()
            
        except Exception as e:
            logger.error(f"❌ Failed to record error: {e}")
    
    def update_health_status(self, component: str, status: str, response_time: float = 0.0, **details):
        """Update component health status"""
        try:
            # Count recent errors for this component
            cutoff_time = datetime.now() - timedelta(minutes=5)
            recent_errors = sum(
                1 for error in self.error_events
                if error.component == component and error.timestamp > cutoff_time
            )
            
            self.health_statuses[component] = HealthStatus(
                component=component,
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=recent_errors,
                details=details
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update health status for {component}: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # Calculate overall metrics
            uptime = time.time() - self.start_time
            error_rate = self.error_counter / max(1, self.request_counter)
            
            # Calculate response time statistics
            response_times = [
                metric.value for metric in self.performance_metrics.get('request_duration', [])
                if metric.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0
            
            # Determine overall status
            overall_status = "healthy"
            if error_rate > self.alert_thresholds['error_rate']:
                overall_status = "degraded"
            if p95_response_time > self.alert_thresholds['response_time_p95']:
                overall_status = "degraded"
            
            # Check component statuses
            unhealthy_components = [
                comp for comp, status in self.health_statuses.items()
                if status.status == "unhealthy"
            ]
            if unhealthy_components:
                overall_status = "unhealthy"
            
            return {
                'status': overall_status,
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'metrics': {
                    'requests_total': self.request_counter,
                    'errors_total': self.error_counter,
                    'error_rate': error_rate,
                    'avg_response_time_ms': avg_response_time * 1000,
                    'p95_response_time_ms': p95_response_time * 1000,
                    'requests_per_minute': self._calculate_requests_per_minute()
                },
                'components': {
                    comp: {
                        'status': status.status,
                        'last_check': status.last_check.isoformat(),
                        'response_time_ms': status.response_time * 1000,
                        'recent_errors': status.error_count
                    }
                    for comp, status in self.health_statuses.items()
                },
                'resource_usage': self.resource_usage,
                'sla_compliance': self._calculate_sla_compliance()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get health status: {e}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_summary(self, operation: str, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specific operation"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            relevant_metrics = [
                metric for metric in self.performance_metrics.get(operation, [])
                if metric.timestamp > cutoff_time
            ]
            
            if not relevant_metrics:
                return {
                    'operation': operation,
                    'period_hours': hours,
                    'sample_count': 0,
                    'message': 'No data available for this period'
                }
            
            values = [metric.value for metric in relevant_metrics]
            
            return {
                'operation': operation,
                'period_hours': hours,
                'sample_count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'p50': self._calculate_percentile(values, 50),
                'p95': self._calculate_percentile(values, 95),
                'p99': self._calculate_percentile(values, 99)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary for {operation}: {e}")
            return {'error': str(e)}
    
    def get_error_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_errors = [
                error for error in self.error_events
                if error.timestamp > cutoff_time
            ]
            
            # Group by error type
            error_types = defaultdict(int)
            error_components = defaultdict(int)
            
            for error in recent_errors:
                error_types[error.error_type] += 1
                error_components[error.component] += 1
            
            return {
                'period_hours': hours,
                'total_errors': len(recent_errors),
                'error_rate': len(recent_errors) / max(1, self.request_counter),
                'by_type': dict(error_types),
                'by_component': dict(error_components),
                'recent_errors': [
                    {
                        'timestamp': error.timestamp.isoformat(),
                        'type': error.error_type,
                        'component': error.component,
                        'message': error.message
                    }
                    for error in recent_errors[-10:]  # Last 10 errors
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get error summary: {e}")
            return {'error': str(e)}
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute over last hour"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            recent_requests = [
                metric for metric in self.performance_metrics.get('request_duration', [])
                if metric.timestamp > cutoff_time
            ]
            
            if not recent_requests:
                return 0.0
            
            return len(recent_requests) / 60.0  # Per minute
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate requests per minute: {e}")
            return 0.0
    
    def _calculate_sla_compliance(self) -> Dict[str, float]:
        """Calculate SLA compliance metrics"""
        try:
            # Availability (uptime percentage)
            uptime = time.time() - self.start_time
            # Assume 99.9% availability target
            availability = min(1.0, uptime / (uptime + 60))  # Assume max 1 minute downtime
            
            # Error rate compliance
            error_rate = self.error_counter / max(1, self.request_counter)
            error_rate_compliance = 1.0 if error_rate <= self.alert_thresholds['error_rate'] else 0.0
            
            # Response time compliance
            recent_response_times = [
                metric.value for metric in self.performance_metrics.get('request_duration', [])
                if metric.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            if recent_response_times:
                p95_response_time = self._calculate_percentile(recent_response_times, 95)
                response_time_compliance = 1.0 if p95_response_time <= self.alert_thresholds['response_time_p95'] else 0.0
            else:
                response_time_compliance = 1.0
            
            return {
                'availability': availability,
                'error_rate_compliance': error_rate_compliance,
                'response_time_compliance': response_time_compliance,
                'overall_sla': (availability + error_rate_compliance + response_time_compliance) / 3
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate SLA compliance: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            
            for operation in self.performance_metrics:
                # Remove old metrics
                while (self.performance_metrics[operation] and 
                       self.performance_metrics[operation][0].timestamp < cutoff_time):
                    self.performance_metrics[operation].popleft()
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old metrics: {e}")
    
    def _check_alert_conditions(self):
        """Check if any alert conditions are met"""
        try:
            # Check error rate
            error_rate = self.error_counter / max(1, self.request_counter)
            if error_rate > self.alert_thresholds['error_rate']:
                logger.warning(
                    f"🚨 High error rate alert: {error_rate:.3f} > {self.alert_thresholds['error_rate']}"
                )
            
            # Check recent response times
            recent_response_times = [
                metric.value for metric in self.performance_metrics.get('request_duration', [])
                if metric.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            if recent_response_times:
                avg_response_time = sum(recent_response_times) / len(recent_response_times)
                if avg_response_time > self.alert_thresholds['response_time_p95']:
                    logger.warning(
                        f"🚨 High response time alert: {avg_response_time:.3f}s > {self.alert_thresholds['response_time_p95']}s"
                    )
                    
        except Exception as e:
            logger.error(f"❌ Failed to check alert conditions: {e}")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        try:
            if format == 'json':
                return json.dumps(self.get_health_status(), indent=2)
            elif format == 'prometheus':
                return self._export_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"❌ Failed to export metrics: {e}")
            return f"Error exporting metrics: {e}"
    
    def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            metrics = []
            
            # Basic counters
            metrics.append(f"ai_service_requests_total {self.request_counter}")
            metrics.append(f"ai_service_errors_total {self.error_counter}")
            
            # Error rate
            error_rate = self.error_counter / max(1, self.request_counter)
            metrics.append(f"ai_service_error_rate {error_rate}")
            
            # Uptime
            uptime = time.time() - self.start_time
            metrics.append(f"ai_service_uptime_seconds {uptime}")
            
            # Response time metrics
            recent_response_times = [
                metric.value for metric in self.performance_metrics.get('request_duration', [])
                if metric.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            if recent_response_times:
                avg_response_time = sum(recent_response_times) / len(recent_response_times)
                p95_response_time = self._calculate_percentile(recent_response_times, 95)
                
                metrics.append(f"ai_service_response_time_avg {avg_response_time}")
                metrics.append(f"ai_service_response_time_p95 {p95_response_time}")
            
            # Component health
            for component, status in self.health_statuses.items():
                status_value = 1 if status.status == 'healthy' else 0
                metrics.append(f'ai_service_component_healthy{{component="{component}"}} {status_value}')
            
            return '\n'.join(metrics)
            
        except Exception as e:
            logger.error(f"❌ Failed to export Prometheus metrics: {e}")
            return f"# Error exporting Prometheus metrics: {e}"
    
    async def start_background_monitoring(self):
        """Start background monitoring tasks"""
        try:
            logger.info("🔄 Starting background monitoring tasks")
            
            # Start periodic health checks
            asyncio.create_task(self._periodic_health_check())
            
            # Start resource monitoring
            asyncio.create_task(self._monitor_resources())
            
        except Exception as e:
            logger.error(f"❌ Failed to start background monitoring: {e}")
    
    async def _periodic_health_check(self):
        """Periodic health check for all components"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Update SLA metrics
                self.sla_metrics.update(self._calculate_sla_compliance())
                
                # Log health summary
                health = self.get_health_status()
                logger.info(
                    f"📊 Health Check - Status: {health['status']}, "
                    f"Error Rate: {health['metrics']['error_rate']:.3f}, "
                    f"Avg Response: {health['metrics']['avg_response_time_ms']:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"❌ Periodic health check failed: {e}")
    
    async def _monitor_resources(self):
        """Monitor system resources"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update resource usage (simplified - in production use psutil)
                import os
                import psutil
                
                process = psutil.Process(os.getpid())
                
                self.resource_usage.update({
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'active_connections': len(process.connections())
                })
                
            except ImportError:
                # psutil not available, use basic monitoring
                pass
            except Exception as e:
                logger.error(f"❌ Resource monitoring failed: {e}")

# Global monitor instance
ai_monitor = AIServiceMonitor()

# Convenience functions
def record_request_duration(duration: float, endpoint: str = "", user_id: str = None):
    """Record request duration"""
    ai_monitor.record_performance_metric('request_duration', duration, user_id, endpoint=endpoint)

def record_prediction_time(duration: float, model_type: str, user_id: str = None):
    """Record prediction time"""
    ai_monitor.record_performance_metric('prediction_time', duration, user_id, model_type=model_type)

def record_error(error_type: str, component: str, message: str = "", user_id: str = None):
    """Record an error"""
    ai_monitor.record_error(error_type, component, message, user_id)

def update_component_health(component: str, status: str, response_time: float = 0.0):
    """Update component health"""
    ai_monitor.update_health_status(component, status, response_time)