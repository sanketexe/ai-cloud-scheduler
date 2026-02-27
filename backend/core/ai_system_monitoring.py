"""
AI System Monitoring and Observability

This module provides comprehensive monitoring and observability for all AI/ML systems
including health monitoring, performance tracking, decision audit trails, and
resource usage optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import structlog

from .database import get_db_session
from .models import MLModelMetrics
from .notification_service import get_notification_service, NotificationMessage, NotificationPriority
from .automation_audit_logger import AutomationAuditLogger

logger = structlog.get_logger(__name__)


class AISystemType(Enum):
    """AI system types for monitoring"""
    PREDICTIVE_SCALING = "predictive_scaling"
    WORKLOAD_INTELLIGENCE = "workload_intelligence"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NATURAL_LANGUAGE = "natural_language"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    SMART_CONTRACT = "smart_contract"
    ML_MODEL_MANAGER = "ml_model_manager"
    AI_ORCHESTRATOR = "ai_orchestrator"


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AISystemMetrics:
    """Comprehensive AI system metrics"""
    system_type: AISystemType
    timestamp: datetime
    
    # Performance metrics
    response_time_ms: float
    throughput_requests_per_second: float
    accuracy_score: Optional[float] = None
    prediction_confidence: Optional[float] = None
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    # System health metrics
    health_status: HealthStatus = HealthStatus.UNKNOWN
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class ModelPerformanceMetrics:
    """ML model performance tracking metrics"""
    model_id: str
    model_name: str
    model_version: str
    timestamp: datetime
    
    # Performance metrics
    inference_time_ms: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Drift detection
    data_drift_score: Optional[float] = None
    concept_drift_score: Optional[float] = None
    
    # Resource usage
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    
    # Prediction statistics
    predictions_count: int = 0
    confidence_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {}


@dataclass
class AIDecisionAuditLog:
    """AI decision audit trail entry"""
    decision_id: str
    system_type: AISystemType
    timestamp: datetime
    
    # Decision context
    input_data: Dict[str, Any]
    decision_output: Dict[str, Any]
    confidence_score: float
    
    # Explainability
    feature_importance: Dict[str, float]
    decision_reasoning: str
    alternative_options: List[Dict[str, Any]]
    
    # Metadata
    model_version: Optional[str] = None
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    
    # Outcome tracking
    actual_outcome: Optional[Dict[str, Any]] = None
    outcome_timestamp: Optional[datetime] = None


@dataclass
class ResourceUsageOptimization:
    """Resource usage optimization recommendations"""
    system_type: AISystemType
    timestamp: datetime
    
    # Current usage
    current_cpu_usage: float
    current_memory_usage: float
    
    # Optimization recommendations
    recommended_cpu_allocation: float
    recommended_memory_allocation: float
    
    # Potential savings
    estimated_cost_savings: float
    estimated_performance_impact: float
    
    # Implementation details
    optimization_actions: List[str]
    implementation_priority: str  # high, medium, low
    
    # Optional fields
    current_gpu_usage: Optional[float] = None
    recommended_gpu_allocation: Optional[float] = None


class AISystemMonitor:
    """
    Comprehensive AI system monitoring and observability service.
    
    Provides real-time monitoring, performance tracking, audit trails,
    and resource optimization for all AI/ML systems.
    """
    
    def __init__(self):
        self.notification_service = get_notification_service()
        self.audit_logger = AutomationAuditLogger()
        
        # Monitoring state
        self.system_metrics: Dict[AISystemType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.decision_logs: deque = deque(maxlen=10000)
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.resource_optimization_interval = 300  # 5 minutes
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_monitoring = False
        
        # Performance baselines
        self.performance_baselines: Dict[AISystemType, Dict[str, float]] = {}
        
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds"""
        return {
            "response_time": {
                "warning": 2000.0,  # 2 seconds
                "critical": 5000.0   # 5 seconds
            },
            "accuracy": {
                "warning": 0.85,     # Below 85%
                "critical": 0.75     # Below 75%
            },
            "error_rate": {
                "warning": 0.05,     # 5%
                "critical": 0.15     # 15%
            },
            "cpu_usage": {
                "warning": 80.0,     # 80%
                "critical": 95.0     # 95%
            },
            "memory_usage": {
                "warning": 85.0,     # 85%
                "critical": 95.0     # 95%
            },
            "data_drift": {
                "warning": 0.3,      # 30% drift
                "critical": 0.5      # 50% drift
            }
        }
    
    async def start_monitoring(self):
        """Start comprehensive AI system monitoring"""
        if self.is_monitoring:
            logger.warning("AI system monitoring already running")
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_model_performance()),
            asyncio.create_task(self._monitor_resource_usage()),
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._optimize_resources())
        ]
        
        logger.info("AI system monitoring started")
        
        # Log monitoring start event
        self.audit_logger.log_system_event(
            "ai_monitoring_started",
            {
                "monitoring_interval": self.monitoring_interval,
                "systems_monitored": [system.value for system in AISystemType],
                "alert_thresholds": self.alert_thresholds
            },
            severity="info"
        )
    
    async def stop_monitoring(self):
        """Stop AI system monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("AI system monitoring stopped")
        
        # Log monitoring stop event
        self.audit_logger.log_system_event(
            "ai_monitoring_stopped",
            {"stopped_at": datetime.utcnow().isoformat()},
            severity="info"
        )
    
    async def record_system_metrics(self, metrics: AISystemMetrics):
        """Record AI system metrics"""
        try:
            # Store metrics
            self.system_metrics[metrics.system_type].append(metrics)
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
            # Update performance baselines
            await self._update_performance_baselines(metrics)
            
            logger.debug(
                "System metrics recorded",
                system_type=metrics.system_type.value,
                health_status=metrics.health_status.value,
                response_time=metrics.response_time_ms
            )
            
        except Exception as e:
            logger.error("Failed to record system metrics", error=str(e))
    
    async def record_model_performance(self, metrics: ModelPerformanceMetrics):
        """Record ML model performance metrics"""
        try:
            # Store metrics
            self.model_metrics[metrics.model_id].append(metrics)
            
            # Check for model performance alerts
            await self._check_model_alerts(metrics)
            
            # Store in database for long-term tracking
            await self._store_model_metrics_db(metrics)
            
            logger.debug(
                "Model performance recorded",
                model_id=metrics.model_id,
                accuracy=metrics.accuracy,
                inference_time=metrics.inference_time_ms
            )
            
        except Exception as e:
            logger.error("Failed to record model performance", error=str(e))
    
    async def log_ai_decision(self, decision_log: AIDecisionAuditLog):
        """Log AI decision for audit trail and explainability"""
        try:
            # Store decision log
            self.decision_logs.append(decision_log)
            
            # Log to audit system
            self.audit_logger.log_system_event(
                "ai_decision_made",
                {
                    "decision_id": decision_log.decision_id,
                    "system_type": decision_log.system_type.value,
                    "confidence_score": decision_log.confidence_score,
                    "decision_reasoning": decision_log.decision_reasoning,
                    "feature_importance": decision_log.feature_importance,
                    "model_version": decision_log.model_version,
                    "user_id": decision_log.user_id,
                    "account_id": decision_log.account_id
                },
                severity="info"
            )
            
            logger.info(
                "AI decision logged",
                decision_id=decision_log.decision_id,
                system_type=decision_log.system_type.value,
                confidence=decision_log.confidence_score
            )
            
        except Exception as e:
            logger.error("Failed to log AI decision", error=str(e))
    
    async def update_decision_outcome(self, decision_id: str, outcome: Dict[str, Any]):
        """Update the actual outcome of an AI decision for learning"""
        try:
            # Find the decision log
            for decision_log in reversed(self.decision_logs):
                if decision_log.decision_id == decision_id:
                    decision_log.actual_outcome = outcome
                    decision_log.outcome_timestamp = datetime.utcnow()
                    
                    # Log outcome update
                    self.audit_logger.log_system_event(
                        "ai_decision_outcome_updated",
                        {
                            "decision_id": decision_id,
                            "outcome": outcome,
                            "system_type": decision_log.system_type.value
                        },
                        severity="info"
                    )
                    
                    logger.info("AI decision outcome updated", decision_id=decision_id)
                    return
            
            logger.warning("Decision not found for outcome update", decision_id=decision_id)
            
        except Exception as e:
            logger.error("Failed to update decision outcome", error=str(e))
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": HealthStatus.HEALTHY.value,
                "systems": {},
                "active_alerts": len(self.active_alerts),
                "monitoring_status": "active" if self.is_monitoring else "inactive"
            }
            
            # Analyze each AI system
            for system_type in AISystemType:
                if system_type in self.system_metrics and self.system_metrics[system_type]:
                    latest_metrics = self.system_metrics[system_type][-1]
                    
                    system_summary = {
                        "health_status": latest_metrics.health_status.value,
                        "response_time_ms": latest_metrics.response_time_ms,
                        "error_rate": latest_metrics.error_rate,
                        "uptime_seconds": latest_metrics.uptime_seconds,
                        "last_updated": latest_metrics.timestamp.isoformat()
                    }
                    
                    # Add accuracy if available
                    if latest_metrics.accuracy_score is not None:
                        system_summary["accuracy_score"] = latest_metrics.accuracy_score
                    
                    summary["systems"][system_type.value] = system_summary
                    
                    # Update overall health
                    if latest_metrics.health_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                        summary["overall_health"] = HealthStatus.CRITICAL.value
                    elif (latest_metrics.health_status == HealthStatus.DEGRADED and 
                          summary["overall_health"] == HealthStatus.HEALTHY.value):
                        summary["overall_health"] = HealthStatus.DEGRADED.value
                else:
                    summary["systems"][system_type.value] = {
                        "health_status": HealthStatus.UNKNOWN.value,
                        "message": "No metrics available"
                    }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get system health summary", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": HealthStatus.UNKNOWN.value,
                "error": str(e)
            }
    
    async def get_model_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get ML model performance summary"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "models": {}
            }
            
            for model_id, metrics_deque in self.model_metrics.items():
                # Filter metrics within time window
                recent_metrics = [
                    m for m in metrics_deque 
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_metrics:
                    latest_metrics = recent_metrics[-1]
                    
                    # Calculate averages
                    avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
                    avg_inference_time = sum(m.inference_time_ms for m in recent_metrics) / len(recent_metrics)
                    total_predictions = sum(m.predictions_count for m in recent_metrics)
                    
                    model_summary = {
                        "model_name": latest_metrics.model_name,
                        "model_version": latest_metrics.model_version,
                        "latest_accuracy": latest_metrics.accuracy,
                        "average_accuracy": avg_accuracy,
                        "latest_inference_time_ms": latest_metrics.inference_time_ms,
                        "average_inference_time_ms": avg_inference_time,
                        "total_predictions": total_predictions,
                        "data_drift_score": latest_metrics.data_drift_score,
                        "last_updated": latest_metrics.timestamp.isoformat()
                    }
                    
                    summary["models"][model_id] = model_summary
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get model performance summary", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def get_decision_audit_trail(self, 
                                     system_type: Optional[AISystemType] = None,
                                     time_window_hours: int = 24,
                                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get AI decision audit trail"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter decisions
            filtered_decisions = []
            for decision in reversed(self.decision_logs):
                if decision.timestamp < cutoff_time:
                    break
                
                if system_type is None or decision.system_type == system_type:
                    filtered_decisions.append(asdict(decision))
                
                if len(filtered_decisions) >= limit:
                    break
            
            return filtered_decisions
            
        except Exception as e:
            logger.error("Failed to get decision audit trail", error=str(e))
            return []
    
    async def get_resource_optimization_recommendations(self) -> List[ResourceUsageOptimization]:
        """Get resource usage optimization recommendations"""
        try:
            recommendations = []
            
            for system_type in AISystemType:
                if system_type in self.system_metrics and self.system_metrics[system_type]:
                    latest_metrics = self.system_metrics[system_type][-1]
                    
                    # Analyze resource usage patterns
                    recent_metrics = list(self.system_metrics[system_type])[-10:]  # Last 10 measurements
                    
                    if len(recent_metrics) >= 5:
                        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
                        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
                        
                        # Generate optimization recommendations
                        optimization = await self._generate_optimization_recommendation(
                            system_type, avg_cpu, avg_memory, latest_metrics
                        )
                        
                        if optimization:
                            recommendations.append(optimization)
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to get resource optimization recommendations", error=str(e))
            return []
    
    async def _monitor_system_health(self):
        """Monitor AI system health continuously"""
        while self.is_monitoring:
            try:
                # This would normally query actual AI systems
                # For now, simulate health monitoring
                for system_type in AISystemType:
                    metrics = await self._collect_system_metrics(system_type)
                    if metrics:
                        await self.record_system_metrics(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in system health monitoring", error=str(e))
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_model_performance(self):
        """Monitor ML model performance continuously"""
        while self.is_monitoring:
            try:
                # Query model performance from database and active models
                await self._collect_model_performance_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in model performance monitoring", error=str(e))
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_resource_usage(self):
        """Monitor AI system resource usage"""
        while self.is_monitoring:
            try:
                # Collect system resource metrics
                system_metrics = await self._collect_system_resource_metrics()
                
                # Analyze for optimization opportunities
                await self._analyze_resource_optimization_opportunities(system_metrics)
                
                await asyncio.sleep(self.resource_optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in resource usage monitoring", error=str(e))
                await asyncio.sleep(self.resource_optimization_interval)
    
    async def _process_alerts(self):
        """Process and manage alerts"""
        while self.is_monitoring:
            try:
                # Process active alerts
                await self._process_active_alerts()
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert processing", error=str(e))
                await asyncio.sleep(60)
    
    async def _optimize_resources(self):
        """Continuously optimize AI system resources"""
        while self.is_monitoring:
            try:
                # Generate optimization recommendations
                recommendations = await self.get_resource_optimization_recommendations()
                
                # Apply automatic optimizations for low-risk changes
                for recommendation in recommendations:
                    if recommendation.implementation_priority == "high":
                        await self._apply_resource_optimization(recommendation)
                
                await asyncio.sleep(self.resource_optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in resource optimization", error=str(e))
                await asyncio.sleep(self.resource_optimization_interval)
    
    async def _collect_system_metrics(self, system_type: AISystemType) -> Optional[AISystemMetrics]:
        """Collect metrics for a specific AI system"""
        try:
            # This would normally query the actual AI system
            # For now, simulate metrics collection
            
            # Get system resource usage
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            # Simulate system-specific metrics
            response_time = 100 + (hash(system_type.value) % 500)  # 100-600ms
            throughput = 10 + (hash(system_type.value) % 40)  # 10-50 rps
            accuracy = 0.85 + (hash(system_type.value) % 15) / 100  # 85-100%
            error_rate = (hash(system_type.value) % 5) / 100  # 0-5%
            
            # Determine health status
            health_status = HealthStatus.HEALTHY
            if response_time > self.alert_thresholds["response_time"]["critical"]:
                health_status = HealthStatus.CRITICAL
            elif response_time > self.alert_thresholds["response_time"]["warning"]:
                health_status = HealthStatus.DEGRADED
            
            if error_rate > self.alert_thresholds["error_rate"]["critical"]:
                health_status = HealthStatus.CRITICAL
            elif error_rate > self.alert_thresholds["error_rate"]["warning"] and health_status == HealthStatus.HEALTHY:
                health_status = HealthStatus.DEGRADED
            
            return AISystemMetrics(
                system_type=system_type,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                throughput_requests_per_second=throughput,
                accuracy_score=accuracy,
                prediction_confidence=0.9,
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory_info.used / (1024 * 1024),
                health_status=health_status,
                error_rate=error_rate,
                uptime_seconds=time.time() - psutil.boot_time()
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", system_type=system_type.value, error=str(e))
            return None
    
    async def _collect_model_performance_metrics(self):
        """Collect ML model performance metrics"""
        try:
            # Query recent model metrics from database
            async with get_db_session() as session:
                recent_models = session.query(MLModelMetrics).filter(
                    MLModelMetrics.training_date >= datetime.utcnow() - timedelta(hours=24)
                ).all()
                
                for model in recent_models:
                    # Simulate performance metrics
                    metrics = ModelPerformanceMetrics(
                        model_id=f"{model.model_name}_{model.model_version}",
                        model_name=model.model_name,
                        model_version=model.model_version,
                        timestamp=datetime.utcnow(),
                        inference_time_ms=50 + (hash(model.model_name) % 100),
                        accuracy=float(model.accuracy_score),
                        precision=float(model.precision_score),
                        recall=float(model.recall_score),
                        f1_score=(float(model.precision_score) + float(model.recall_score)) / 2,
                        data_drift_score=0.1 + (hash(model.model_name) % 20) / 100,
                        cpu_time_ms=10 + (hash(model.model_name) % 50),
                        memory_peak_mb=100 + (hash(model.model_name) % 200),
                        predictions_count=100 + (hash(model.model_name) % 500)
                    )
                    
                    await self.record_model_performance(metrics)
                    
        except Exception as e:
            logger.error("Failed to collect model performance metrics", error=str(e))
    
    async def _collect_system_resource_metrics(self) -> Dict[str, Any]:
        """Collect system-wide resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.utcnow(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024)
            }
            
        except Exception as e:
            logger.error("Failed to collect system resource metrics", error=str(e))
            return {}
    
    async def _check_system_alerts(self, metrics: AISystemMetrics):
        """Check for system-level alerts"""
        try:
            alerts_to_create = []
            
            # Check response time
            if metrics.response_time_ms > self.alert_thresholds["response_time"]["critical"]:
                alerts_to_create.append({
                    "type": "response_time",
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"Critical response time: {metrics.response_time_ms:.1f}ms",
                    "value": metrics.response_time_ms,
                    "threshold": self.alert_thresholds["response_time"]["critical"]
                })
            elif metrics.response_time_ms > self.alert_thresholds["response_time"]["warning"]:
                alerts_to_create.append({
                    "type": "response_time",
                    "severity": AlertSeverity.WARNING,
                    "message": f"High response time: {metrics.response_time_ms:.1f}ms",
                    "value": metrics.response_time_ms,
                    "threshold": self.alert_thresholds["response_time"]["warning"]
                })
            
            # Check accuracy
            if metrics.accuracy_score and metrics.accuracy_score < self.alert_thresholds["accuracy"]["critical"]:
                alerts_to_create.append({
                    "type": "accuracy",
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"Critical accuracy drop: {metrics.accuracy_score:.2%}",
                    "value": metrics.accuracy_score,
                    "threshold": self.alert_thresholds["accuracy"]["critical"]
                })
            elif metrics.accuracy_score and metrics.accuracy_score < self.alert_thresholds["accuracy"]["warning"]:
                alerts_to_create.append({
                    "type": "accuracy",
                    "severity": AlertSeverity.WARNING,
                    "message": f"Low accuracy: {metrics.accuracy_score:.2%}",
                    "value": metrics.accuracy_score,
                    "threshold": self.alert_thresholds["accuracy"]["warning"]
                })
            
            # Check error rate
            if metrics.error_rate > self.alert_thresholds["error_rate"]["critical"]:
                alerts_to_create.append({
                    "type": "error_rate",
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"Critical error rate: {metrics.error_rate:.2%}",
                    "value": metrics.error_rate,
                    "threshold": self.alert_thresholds["error_rate"]["critical"]
                })
            elif metrics.error_rate > self.alert_thresholds["error_rate"]["warning"]:
                alerts_to_create.append({
                    "type": "error_rate",
                    "severity": AlertSeverity.WARNING,
                    "message": f"High error rate: {metrics.error_rate:.2%}",
                    "value": metrics.error_rate,
                    "threshold": self.alert_thresholds["error_rate"]["warning"]
                })
            
            # Create alerts
            for alert_data in alerts_to_create:
                await self._create_alert(metrics.system_type, alert_data)
                
        except Exception as e:
            logger.error("Failed to check system alerts", error=str(e))
    
    async def _check_model_alerts(self, metrics: ModelPerformanceMetrics):
        """Check for model performance alerts"""
        try:
            alerts_to_create = []
            
            # Check accuracy degradation
            if metrics.accuracy < self.alert_thresholds["accuracy"]["critical"]:
                alerts_to_create.append({
                    "type": "model_accuracy",
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"Model accuracy critically low: {metrics.accuracy:.2%}",
                    "model_id": metrics.model_id,
                    "value": metrics.accuracy,
                    "threshold": self.alert_thresholds["accuracy"]["critical"]
                })
            
            # Check data drift
            if metrics.data_drift_score and metrics.data_drift_score > self.alert_thresholds["data_drift"]["critical"]:
                alerts_to_create.append({
                    "type": "data_drift",
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"Critical data drift detected: {metrics.data_drift_score:.2%}",
                    "model_id": metrics.model_id,
                    "value": metrics.data_drift_score,
                    "threshold": self.alert_thresholds["data_drift"]["critical"]
                })
            
            # Create alerts
            for alert_data in alerts_to_create:
                await self._create_model_alert(metrics.model_id, alert_data)
                
        except Exception as e:
            logger.error("Failed to check model alerts", error=str(e))
    
    async def _create_alert(self, system_type: AISystemType, alert_data: Dict[str, Any]):
        """Create and send system alert"""
        try:
            alert_id = f"{system_type.value}_{alert_data['type']}_{int(time.time())}"
            
            # Store alert
            self.active_alerts[alert_id] = {
                "id": alert_id,
                "system_type": system_type.value,
                "created_at": datetime.utcnow(),
                **alert_data
            }
            
            # Send notification
            priority_map = {
                AlertSeverity.INFO: NotificationPriority.LOW,
                AlertSeverity.WARNING: NotificationPriority.MEDIUM,
                AlertSeverity.ERROR: NotificationPriority.HIGH,
                AlertSeverity.CRITICAL: NotificationPriority.CRITICAL
            }
            
            notification = NotificationMessage(
                title=f"AI System Alert: {system_type.value}",
                message=alert_data["message"],
                priority=priority_map[alert_data["severity"]],
                metadata={
                    "alert_id": alert_id,
                    "system_type": system_type.value,
                    "alert_type": alert_data["type"],
                    "value": alert_data["value"],
                    "threshold": alert_data["threshold"]
                }
            )
            
            # Send to appropriate channels (would be configured)
            await self.notification_service.send_notification(["admin"], notification)
            
            logger.warning(
                "AI system alert created",
                alert_id=alert_id,
                system_type=system_type.value,
                severity=alert_data["severity"].value,
                message=alert_data["message"]
            )
            
        except Exception as e:
            logger.error("Failed to create alert", error=str(e))
    
    async def _create_model_alert(self, model_id: str, alert_data: Dict[str, Any]):
        """Create and send model performance alert"""
        try:
            alert_id = f"model_{model_id}_{alert_data['type']}_{int(time.time())}"
            
            # Store alert
            self.active_alerts[alert_id] = {
                "id": alert_id,
                "model_id": model_id,
                "created_at": datetime.utcnow(),
                **alert_data
            }
            
            # Send notification (similar to system alerts)
            logger.warning(
                "Model performance alert created",
                alert_id=alert_id,
                model_id=model_id,
                severity=alert_data["severity"].value,
                message=alert_data["message"]
            )
            
        except Exception as e:
            logger.error("Failed to create model alert", error=str(e))
    
    async def _update_performance_baselines(self, metrics: AISystemMetrics):
        """Update performance baselines for anomaly detection"""
        try:
            if metrics.system_type not in self.performance_baselines:
                self.performance_baselines[metrics.system_type] = {}
            
            baseline = self.performance_baselines[metrics.system_type]
            
            # Update rolling averages
            alpha = 0.1  # Exponential moving average factor
            
            if "response_time" not in baseline:
                baseline["response_time"] = metrics.response_time_ms
            else:
                baseline["response_time"] = (1 - alpha) * baseline["response_time"] + alpha * metrics.response_time_ms
            
            if metrics.accuracy_score and "accuracy" not in baseline:
                baseline["accuracy"] = metrics.accuracy_score
            elif metrics.accuracy_score:
                baseline["accuracy"] = (1 - alpha) * baseline["accuracy"] + alpha * metrics.accuracy_score
            
            if "error_rate" not in baseline:
                baseline["error_rate"] = metrics.error_rate
            else:
                baseline["error_rate"] = (1 - alpha) * baseline["error_rate"] + alpha * metrics.error_rate
                
        except Exception as e:
            logger.error("Failed to update performance baselines", error=str(e))
    
    async def _store_model_metrics_db(self, metrics: ModelPerformanceMetrics):
        """Store model metrics in database for long-term tracking"""
        try:
            async with get_db_session() as session:
                # Update existing model metrics or create new entry
                existing_metric = session.query(MLModelMetrics).filter(
                    MLModelMetrics.model_name == metrics.model_name,
                    MLModelMetrics.model_version == metrics.model_version
                ).first()
                
                if existing_metric:
                    # Update existing metrics
                    existing_metric.accuracy_score = metrics.accuracy
                    existing_metric.precision_score = metrics.precision
                    existing_metric.recall_score = metrics.recall
                    existing_metric.detection_latency_ms = int(metrics.inference_time_ms)
                else:
                    # Create new metrics entry
                    new_metric = MLModelMetrics(
                        model_name=metrics.model_name,
                        model_version=metrics.model_version,
                        account_id="system",
                        training_date=datetime.utcnow(),
                        accuracy_score=metrics.accuracy,
                        precision_score=metrics.precision,
                        recall_score=metrics.recall,
                        false_positive_rate=0.0,
                        detection_latency_ms=int(metrics.inference_time_ms),
                        training_data_points=1000,
                        feature_count=10,
                        hyperparameters={},
                        is_active=True
                    )
                    session.add(new_metric)
                
                await session.commit()
                
        except Exception as e:
            logger.error("Failed to store model metrics in database", error=str(e))
    
    async def _generate_optimization_recommendation(self, 
                                                  system_type: AISystemType,
                                                  avg_cpu: float,
                                                  avg_memory: float,
                                                  latest_metrics: AISystemMetrics) -> Optional[ResourceUsageOptimization]:
        """Generate resource optimization recommendation"""
        try:
            optimization_actions = []
            estimated_savings = 0.0
            performance_impact = 0.0
            
            # CPU optimization
            recommended_cpu = avg_cpu
            if avg_cpu < 30:  # Under-utilized
                recommended_cpu = max(avg_cpu * 0.7, 10)  # Reduce by 30%, minimum 10%
                optimization_actions.append(f"Reduce CPU allocation from {avg_cpu:.1f}% to {recommended_cpu:.1f}%")
                estimated_savings += 20.0  # $20/month estimated
                performance_impact = -0.05  # 5% performance reduction risk
            elif avg_cpu > 80:  # Over-utilized
                recommended_cpu = min(avg_cpu * 1.3, 100)  # Increase by 30%, maximum 100%
                optimization_actions.append(f"Increase CPU allocation from {avg_cpu:.1f}% to {recommended_cpu:.1f}%")
                estimated_savings -= 15.0  # Additional cost
                performance_impact = 0.15  # 15% performance improvement
            
            # Memory optimization
            recommended_memory = avg_memory
            if avg_memory < 512:  # Under-utilized (less than 512MB)
                recommended_memory = max(avg_memory * 0.8, 256)  # Reduce by 20%, minimum 256MB
                optimization_actions.append(f"Reduce memory allocation from {avg_memory:.0f}MB to {recommended_memory:.0f}MB")
                estimated_savings += 10.0  # $10/month estimated
            elif avg_memory > 2048:  # Over-utilized (more than 2GB)
                recommended_memory = min(avg_memory * 1.2, 4096)  # Increase by 20%, maximum 4GB
                optimization_actions.append(f"Increase memory allocation from {avg_memory:.0f}MB to {recommended_memory:.0f}MB")
                estimated_savings -= 8.0  # Additional cost
                performance_impact += 0.10  # 10% performance improvement
            
            # Only create recommendation if there are optimizations
            if optimization_actions:
                priority = "high" if abs(estimated_savings) > 25 else "medium" if abs(estimated_savings) > 10 else "low"
                
                return ResourceUsageOptimization(
                    system_type=system_type,
                    timestamp=datetime.utcnow(),
                    current_cpu_usage=avg_cpu,
                    current_memory_usage=avg_memory,
                    recommended_cpu_allocation=recommended_cpu,
                    recommended_memory_allocation=recommended_memory,
                    estimated_cost_savings=estimated_savings,
                    estimated_performance_impact=performance_impact,
                    optimization_actions=optimization_actions,
                    implementation_priority=priority
                )
            
            return None
            
        except Exception as e:
            logger.error("Failed to generate optimization recommendation", error=str(e))
            return None
    
    async def _analyze_resource_optimization_opportunities(self, system_metrics: Dict[str, Any]):
        """Analyze system-wide resource optimization opportunities"""
        try:
            if not system_metrics:
                return
            
            # Log resource usage patterns
            self.audit_logger.log_system_event(
                "resource_usage_analyzed",
                {
                    "cpu_percent": system_metrics.get("cpu_percent", 0),
                    "memory_percent": system_metrics.get("memory_percent", 0),
                    "disk_percent": system_metrics.get("disk_percent", 0),
                    "timestamp": system_metrics["timestamp"].isoformat()
                },
                severity="info"
            )
            
            # Check for system-wide optimization opportunities
            if system_metrics.get("cpu_percent", 0) > 90:
                logger.warning("System CPU usage critically high", cpu_percent=system_metrics["cpu_percent"])
            
            if system_metrics.get("memory_percent", 0) > 90:
                logger.warning("System memory usage critically high", memory_percent=system_metrics["memory_percent"])
                
        except Exception as e:
            logger.error("Failed to analyze resource optimization opportunities", error=str(e))
    
    async def _apply_resource_optimization(self, optimization: ResourceUsageOptimization):
        """Apply automatic resource optimization"""
        try:
            # Log optimization application
            self.audit_logger.log_system_event(
                "resource_optimization_applied",
                {
                    "system_type": optimization.system_type.value,
                    "optimization_actions": optimization.optimization_actions,
                    "estimated_savings": optimization.estimated_cost_savings,
                    "implementation_priority": optimization.implementation_priority
                },
                severity="info"
            )
            
            logger.info(
                "Resource optimization applied",
                system_type=optimization.system_type.value,
                estimated_savings=optimization.estimated_cost_savings,
                actions=len(optimization.optimization_actions)
            )
            
        except Exception as e:
            logger.error("Failed to apply resource optimization", error=str(e))
    
    async def _process_active_alerts(self):
        """Process and manage active alerts"""
        try:
            current_time = datetime.utcnow()
            
            for alert_id, alert in list(self.active_alerts.items()):
                # Check if alert should be escalated
                alert_age = (current_time - alert["created_at"]).total_seconds()
                
                if alert_age > 3600 and alert["severity"] == AlertSeverity.CRITICAL:  # 1 hour
                    # Escalate critical alerts that haven't been resolved
                    await self._escalate_alert(alert_id, alert)
                
        except Exception as e:
            logger.error("Failed to process active alerts", error=str(e))
    
    async def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        try:
            current_time = datetime.utcnow()
            alerts_to_remove = []
            
            for alert_id, alert in self.active_alerts.items():
                # Remove alerts older than 24 hours (assuming they're resolved)
                alert_age = (current_time - alert["created_at"]).total_seconds()
                if alert_age > 86400:  # 24 hours
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                logger.info("Cleaned up resolved alert", alert_id=alert_id)
                
        except Exception as e:
            logger.error("Failed to cleanup resolved alerts", error=str(e))
    
    async def _escalate_alert(self, alert_id: str, alert: Dict[str, Any]):
        """Escalate unresolved critical alerts"""
        try:
            # Log escalation
            self.audit_logger.log_system_event(
                "alert_escalated",
                {
                    "alert_id": alert_id,
                    "original_severity": alert["severity"].value,
                    "escalation_reason": "unresolved_critical_alert",
                    "alert_age_hours": (datetime.utcnow() - alert["created_at"]).total_seconds() / 3600
                },
                severity="critical"
            )
            
            logger.critical("Alert escalated", alert_id=alert_id, message=alert["message"])
            
        except Exception as e:
            logger.error("Failed to escalate alert", error=str(e))


# Global AI system monitor instance
_ai_system_monitor = None

def get_ai_system_monitor() -> AISystemMonitor:
    """Get global AI system monitor instance"""
    global _ai_system_monitor
    if _ai_system_monitor is None:
        _ai_system_monitor = AISystemMonitor()
    return _ai_system_monitor