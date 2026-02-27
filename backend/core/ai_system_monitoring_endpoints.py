"""
AI System Monitoring and Observability Endpoints

This module provides REST API endpoints for AI system monitoring,
performance tracking, decision audit trails, and resource optimization.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from .ai_system_monitoring import (
    get_ai_system_monitor, AISystemMonitor, AISystemType, AISystemMetrics,
    ModelPerformanceMetrics, AIDecisionAuditLog, ResourceUsageOptimization,
    HealthStatus, AlertSeverity
)

router = APIRouter(prefix="/ai-monitoring", tags=["ai-monitoring"])


# Request/Response Models
class SystemMetricsRequest(BaseModel):
    """Request model for recording system metrics"""
    system_type: str
    response_time_ms: float
    throughput_requests_per_second: float
    accuracy_score: Optional[float] = None
    prediction_confidence: Optional[float] = None
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    error_rate: float = 0.0
    custom_metrics: Optional[Dict[str, float]] = None


class ModelMetricsRequest(BaseModel):
    """Request model for recording model performance metrics"""
    model_id: str
    model_name: str
    model_version: str
    inference_time_ms: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    data_drift_score: Optional[float] = None
    concept_drift_score: Optional[float] = None
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    predictions_count: int = 0
    confidence_distribution: Optional[Dict[str, int]] = None


class DecisionLogRequest(BaseModel):
    """Request model for logging AI decisions"""
    decision_id: str
    system_type: str
    input_data: Dict[str, Any]
    decision_output: Dict[str, Any]
    confidence_score: float
    feature_importance: Dict[str, float]
    decision_reasoning: str
    alternative_options: List[Dict[str, Any]]
    model_version: Optional[str] = None
    user_id: Optional[str] = None
    account_id: Optional[str] = None


class DecisionOutcomeRequest(BaseModel):
    """Request model for updating decision outcomes"""
    decision_id: str
    outcome: Dict[str, Any]


class AlertThresholdsRequest(BaseModel):
    """Request model for updating alert thresholds"""
    response_time_warning: Optional[float] = None
    response_time_critical: Optional[float] = None
    accuracy_warning: Optional[float] = None
    accuracy_critical: Optional[float] = None
    error_rate_warning: Optional[float] = None
    error_rate_critical: Optional[float] = None
    cpu_usage_warning: Optional[float] = None
    cpu_usage_critical: Optional[float] = None
    memory_usage_warning: Optional[float] = None
    memory_usage_critical: Optional[float] = None
    data_drift_warning: Optional[float] = None
    data_drift_critical: Optional[float] = None


def get_monitor() -> AISystemMonitor:
    """Dependency to get AI system monitor"""
    return get_ai_system_monitor()


@router.post("/start")
async def start_monitoring(monitor: AISystemMonitor = Depends(get_monitor)):
    """Start AI system monitoring"""
    try:
        await monitor.start_monitoring()
        return {
            "status": "success",
            "message": "AI system monitoring started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/stop")
async def stop_monitoring(monitor: AISystemMonitor = Depends(get_monitor)):
    """Stop AI system monitoring"""
    try:
        await monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "AI system monitoring stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/health")
async def get_monitoring_health(monitor: AISystemMonitor = Depends(get_monitor)):
    """Get AI system monitoring health status"""
    try:
        return {
            "status": "healthy",
            "monitoring_active": monitor.is_monitoring,
            "systems_monitored": [system.value for system in AISystemType],
            "active_alerts": len(monitor.active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring health: {str(e)}")


@router.get("/systems/health")
async def get_system_health_summary(monitor: AISystemMonitor = Depends(get_monitor)):
    """Get comprehensive AI system health summary"""
    try:
        summary = await monitor.get_system_health_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health summary: {str(e)}")


@router.get("/systems/{system_type}/health")
async def get_system_health(
    system_type: str,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Get health status for a specific AI system"""
    try:
        # Validate system type
        try:
            ai_system_type = AISystemType(system_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid system type: {system_type}")
        
        # Get recent metrics for the system
        if ai_system_type in monitor.system_metrics and monitor.system_metrics[ai_system_type]:
            latest_metrics = monitor.system_metrics[ai_system_type][-1]
            
            return {
                "system_type": system_type,
                "health_status": latest_metrics.health_status.value,
                "response_time_ms": latest_metrics.response_time_ms,
                "throughput_rps": latest_metrics.throughput_requests_per_second,
                "accuracy_score": latest_metrics.accuracy_score,
                "error_rate": latest_metrics.error_rate,
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "memory_usage_mb": latest_metrics.memory_usage_mb,
                "uptime_seconds": latest_metrics.uptime_seconds,
                "last_updated": latest_metrics.timestamp.isoformat()
            }
        else:
            return {
                "system_type": system_type,
                "health_status": HealthStatus.UNKNOWN.value,
                "message": "No metrics available for this system"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.post("/systems/metrics")
async def record_system_metrics(
    request: SystemMetricsRequest,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Record AI system metrics"""
    try:
        # Validate system type
        try:
            system_type = AISystemType(request.system_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid system type: {request.system_type}")
        
        # Determine health status based on metrics
        health_status = HealthStatus.HEALTHY
        if request.response_time_ms > 5000 or request.error_rate > 0.15:
            health_status = HealthStatus.CRITICAL
        elif request.response_time_ms > 2000 or request.error_rate > 0.05:
            health_status = HealthStatus.DEGRADED
        
        # Create metrics object
        metrics = AISystemMetrics(
            system_type=system_type,
            timestamp=datetime.utcnow(),
            response_time_ms=request.response_time_ms,
            throughput_requests_per_second=request.throughput_requests_per_second,
            accuracy_score=request.accuracy_score,
            prediction_confidence=request.prediction_confidence,
            cpu_usage_percent=request.cpu_usage_percent,
            memory_usage_mb=request.memory_usage_mb,
            gpu_usage_percent=request.gpu_usage_percent,
            gpu_memory_mb=request.gpu_memory_mb,
            health_status=health_status,
            error_rate=request.error_rate,
            uptime_seconds=0.0,  # Would be calculated from system start time
            custom_metrics=request.custom_metrics or {}
        )
        
        # Record metrics
        await monitor.record_system_metrics(metrics)
        
        return {
            "status": "success",
            "message": "System metrics recorded",
            "system_type": request.system_type,
            "health_status": health_status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record system metrics: {str(e)}")


@router.get("/models/performance")
async def get_model_performance_summary(
    time_window_hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Get ML model performance summary"""
    try:
        summary = await monitor.get_model_performance_summary(time_window_hours)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance summary: {str(e)}")


@router.get("/models/{model_id}/performance")
async def get_model_performance(
    model_id: str,
    time_window_hours: int = Query(24, ge=1, le=168),
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Get performance metrics for a specific model"""
    try:
        if model_id in monitor.model_metrics:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter metrics within time window
            recent_metrics = [
                m for m in monitor.model_metrics[model_id]
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                latest_metrics = recent_metrics[-1]
                
                # Calculate statistics
                avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
                avg_inference_time = sum(m.inference_time_ms for m in recent_metrics) / len(recent_metrics)
                total_predictions = sum(m.predictions_count for m in recent_metrics)
                
                return {
                    "model_id": model_id,
                    "model_name": latest_metrics.model_name,
                    "model_version": latest_metrics.model_version,
                    "time_window_hours": time_window_hours,
                    "metrics_count": len(recent_metrics),
                    "latest_metrics": {
                        "accuracy": latest_metrics.accuracy,
                        "precision": latest_metrics.precision,
                        "recall": latest_metrics.recall,
                        "f1_score": latest_metrics.f1_score,
                        "inference_time_ms": latest_metrics.inference_time_ms,
                        "data_drift_score": latest_metrics.data_drift_score,
                        "timestamp": latest_metrics.timestamp.isoformat()
                    },
                    "aggregated_metrics": {
                        "average_accuracy": avg_accuracy,
                        "average_inference_time_ms": avg_inference_time,
                        "total_predictions": total_predictions
                    }
                }
            else:
                return {
                    "model_id": model_id,
                    "message": f"No metrics available for the last {time_window_hours} hours"
                }
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.post("/models/metrics")
async def record_model_metrics(
    request: ModelMetricsRequest,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Record ML model performance metrics"""
    try:
        # Create metrics object
        metrics = ModelPerformanceMetrics(
            model_id=request.model_id,
            model_name=request.model_name,
            model_version=request.model_version,
            timestamp=datetime.utcnow(),
            inference_time_ms=request.inference_time_ms,
            accuracy=request.accuracy,
            precision=request.precision,
            recall=request.recall,
            f1_score=request.f1_score,
            data_drift_score=request.data_drift_score,
            concept_drift_score=request.concept_drift_score,
            cpu_time_ms=request.cpu_time_ms,
            memory_peak_mb=request.memory_peak_mb,
            predictions_count=request.predictions_count,
            confidence_distribution=request.confidence_distribution or {}
        )
        
        # Record metrics
        await monitor.record_model_performance(metrics)
        
        return {
            "status": "success",
            "message": "Model metrics recorded",
            "model_id": request.model_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record model metrics: {str(e)}")


@router.get("/decisions/audit-trail")
async def get_decision_audit_trail(
    system_type: Optional[str] = Query(None),
    time_window_hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000),
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Get AI decision audit trail"""
    try:
        # Validate system type if provided
        ai_system_type = None
        if system_type:
            try:
                ai_system_type = AISystemType(system_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid system type: {system_type}")
        
        # Get audit trail
        decisions = await monitor.get_decision_audit_trail(
            system_type=ai_system_type,
            time_window_hours=time_window_hours,
            limit=limit
        )
        
        return {
            "decisions": decisions,
            "count": len(decisions),
            "time_window_hours": time_window_hours,
            "system_type_filter": system_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get decision audit trail: {str(e)}")


@router.post("/decisions/log")
async def log_ai_decision(
    request: DecisionLogRequest,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Log AI decision for audit trail and explainability"""
    try:
        # Validate system type
        try:
            system_type = AISystemType(request.system_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid system type: {request.system_type}")
        
        # Create decision log
        decision_log = AIDecisionAuditLog(
            decision_id=request.decision_id,
            system_type=system_type,
            timestamp=datetime.utcnow(),
            input_data=request.input_data,
            decision_output=request.decision_output,
            confidence_score=request.confidence_score,
            feature_importance=request.feature_importance,
            decision_reasoning=request.decision_reasoning,
            alternative_options=request.alternative_options,
            model_version=request.model_version,
            user_id=request.user_id,
            account_id=request.account_id
        )
        
        # Log decision
        await monitor.log_ai_decision(decision_log)
        
        return {
            "status": "success",
            "message": "AI decision logged",
            "decision_id": request.decision_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log AI decision: {str(e)}")


@router.put("/decisions/{decision_id}/outcome")
async def update_decision_outcome(
    decision_id: str,
    request: DecisionOutcomeRequest,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Update the actual outcome of an AI decision"""
    try:
        await monitor.update_decision_outcome(decision_id, request.outcome)
        
        return {
            "status": "success",
            "message": "Decision outcome updated",
            "decision_id": decision_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update decision outcome: {str(e)}")


@router.get("/resources/optimization")
async def get_resource_optimization_recommendations(
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Get resource usage optimization recommendations"""
    try:
        recommendations = await monitor.get_resource_optimization_recommendations()
        
        return {
            "recommendations": [
                {
                    "system_type": rec.system_type.value,
                    "timestamp": rec.timestamp.isoformat(),
                    "current_usage": {
                        "cpu_percent": rec.current_cpu_usage,
                        "memory_mb": rec.current_memory_usage,
                        "gpu_percent": rec.current_gpu_usage
                    },
                    "recommended_allocation": {
                        "cpu_percent": rec.recommended_cpu_allocation,
                        "memory_mb": rec.recommended_memory_allocation,
                        "gpu_percent": rec.recommended_gpu_allocation
                    },
                    "estimated_cost_savings": rec.estimated_cost_savings,
                    "estimated_performance_impact": rec.estimated_performance_impact,
                    "optimization_actions": rec.optimization_actions,
                    "implementation_priority": rec.implementation_priority
                }
                for rec in recommendations
            ],
            "count": len(recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization recommendations: {str(e)}")


@router.get("/alerts")
async def get_active_alerts(monitor: AISystemMonitor = Depends(get_monitor)):
    """Get active alerts"""
    try:
        return {
            "alerts": list(monitor.active_alerts.values()),
            "count": len(monitor.active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active alerts: {str(e)}")


@router.get("/alerts/thresholds")
async def get_alert_thresholds(monitor: AISystemMonitor = Depends(get_monitor)):
    """Get current alert thresholds"""
    try:
        return {
            "thresholds": monitor.alert_thresholds,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert thresholds: {str(e)}")


@router.put("/alerts/thresholds")
async def update_alert_thresholds(
    request: AlertThresholdsRequest,
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Update alert thresholds"""
    try:
        # Update thresholds
        thresholds = monitor.alert_thresholds
        
        if request.response_time_warning is not None:
            thresholds["response_time"]["warning"] = request.response_time_warning
        if request.response_time_critical is not None:
            thresholds["response_time"]["critical"] = request.response_time_critical
        
        if request.accuracy_warning is not None:
            thresholds["accuracy"]["warning"] = request.accuracy_warning
        if request.accuracy_critical is not None:
            thresholds["accuracy"]["critical"] = request.accuracy_critical
        
        if request.error_rate_warning is not None:
            thresholds["error_rate"]["warning"] = request.error_rate_warning
        if request.error_rate_critical is not None:
            thresholds["error_rate"]["critical"] = request.error_rate_critical
        
        if request.cpu_usage_warning is not None:
            thresholds["cpu_usage"]["warning"] = request.cpu_usage_warning
        if request.cpu_usage_critical is not None:
            thresholds["cpu_usage"]["critical"] = request.cpu_usage_critical
        
        if request.memory_usage_warning is not None:
            thresholds["memory_usage"]["warning"] = request.memory_usage_warning
        if request.memory_usage_critical is not None:
            thresholds["memory_usage"]["critical"] = request.memory_usage_critical
        
        if request.data_drift_warning is not None:
            thresholds["data_drift"]["warning"] = request.data_drift_warning
        if request.data_drift_critical is not None:
            thresholds["data_drift"]["critical"] = request.data_drift_critical
        
        return {
            "status": "success",
            "message": "Alert thresholds updated",
            "thresholds": thresholds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert thresholds: {str(e)}")


@router.get("/metrics/export")
async def export_metrics(
    format: str = Query("prometheus", regex="^(prometheus|json)$"),
    monitor: AISystemMonitor = Depends(get_monitor)
):
    """Export metrics in various formats"""
    try:
        if format == "prometheus":
            # Generate Prometheus-style metrics
            metrics_lines = []
            
            # System health metrics
            for system_type in AISystemType:
                if system_type in monitor.system_metrics and monitor.system_metrics[system_type]:
                    latest_metrics = monitor.system_metrics[system_type][-1]
                    
                    # Health status (1 = healthy, 0.5 = degraded, 0 = unhealthy/critical)
                    health_value = 1.0 if latest_metrics.health_status == HealthStatus.HEALTHY else \
                                  0.5 if latest_metrics.health_status == HealthStatus.DEGRADED else 0.0
                    
                    metrics_lines.append(
                        f'ai_system_health{{system="{system_type.value}"}} {health_value}'
                    )
                    metrics_lines.append(
                        f'ai_system_response_time_ms{{system="{system_type.value}"}} {latest_metrics.response_time_ms}'
                    )
                    metrics_lines.append(
                        f'ai_system_error_rate{{system="{system_type.value}"}} {latest_metrics.error_rate}'
                    )
                    
                    if latest_metrics.accuracy_score is not None:
                        metrics_lines.append(
                            f'ai_system_accuracy{{system="{system_type.value}"}} {latest_metrics.accuracy_score}'
                        )
            
            # Model performance metrics
            for model_id, metrics_deque in monitor.model_metrics.items():
                if metrics_deque:
                    latest_metrics = metrics_deque[-1]
                    metrics_lines.append(
                        f'ml_model_accuracy{{model_id="{model_id}"}} {latest_metrics.accuracy}'
                    )
                    metrics_lines.append(
                        f'ml_model_inference_time_ms{{model_id="{model_id}"}} {latest_metrics.inference_time_ms}'
                    )
                    
                    if latest_metrics.data_drift_score is not None:
                        metrics_lines.append(
                            f'ml_model_data_drift{{model_id="{model_id}"}} {latest_metrics.data_drift_score}'
                        )
            
            # Active alerts count
            metrics_lines.append(f'ai_monitoring_active_alerts {len(monitor.active_alerts)}')
            
            return {
                "format": "prometheus",
                "metrics": "\n".join(metrics_lines),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        else:  # JSON format
            return {
                "format": "json",
                "system_health": await monitor.get_system_health_summary(),
                "model_performance": await monitor.get_model_performance_summary(),
                "active_alerts": list(monitor.active_alerts.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@router.get("/runbooks")
async def get_operational_runbooks():
    """Get operational runbooks for AI system management"""
    try:
        runbooks = {
            "ai_system_health_degradation": {
                "title": "AI System Health Degradation Response",
                "description": "Steps to diagnose and resolve AI system health issues",
                "severity": "medium",
                "steps": [
                    "1. Check system metrics dashboard for affected components",
                    "2. Review recent error logs and alert history",
                    "3. Verify resource utilization (CPU, memory, GPU)",
                    "4. Check model performance metrics for accuracy drops",
                    "5. Restart affected AI services if necessary",
                    "6. Scale resources if utilization is high",
                    "7. Contact on-call engineer if issues persist"
                ],
                "escalation_criteria": [
                    "Response time > 5 seconds for more than 10 minutes",
                    "Accuracy drop > 10% from baseline",
                    "Error rate > 15%",
                    "System unavailable for > 5 minutes"
                ]
            },
            "model_performance_degradation": {
                "title": "ML Model Performance Degradation",
                "description": "Response to model accuracy or performance issues",
                "severity": "high",
                "steps": [
                    "1. Check model performance dashboard",
                    "2. Analyze data drift scores and input distributions",
                    "3. Review recent training data quality",
                    "4. Check for concept drift in the problem domain",
                    "5. Rollback to previous model version if necessary",
                    "6. Trigger model retraining with recent data",
                    "7. Implement A/B testing for new model deployment"
                ],
                "escalation_criteria": [
                    "Accuracy drop > 15% from baseline",
                    "Data drift score > 50%",
                    "Model rollback fails",
                    "Business impact detected"
                ]
            },
            "resource_optimization": {
                "title": "AI System Resource Optimization",
                "description": "Optimize AI system resource usage and costs",
                "severity": "low",
                "steps": [
                    "1. Review resource optimization recommendations",
                    "2. Analyze current resource utilization patterns",
                    "3. Identify under-utilized or over-provisioned systems",
                    "4. Plan resource allocation changes during low-traffic periods",
                    "5. Implement gradual resource adjustments",
                    "6. Monitor performance impact of changes",
                    "7. Document optimization results and savings"
                ],
                "escalation_criteria": [
                    "Resource costs exceed budget by > 20%",
                    "Performance degradation after optimization",
                    "System instability after resource changes"
                ]
            },
            "alert_storm_management": {
                "title": "AI System Alert Storm Management",
                "description": "Handle multiple simultaneous alerts",
                "severity": "critical",
                "steps": [
                    "1. Identify root cause of alert storm",
                    "2. Prioritize critical system alerts first",
                    "3. Temporarily increase alert thresholds if needed",
                    "4. Focus on system-wide issues before individual components",
                    "5. Implement emergency scaling if resource-related",
                    "6. Coordinate with team to avoid duplicate efforts",
                    "7. Document incident and improve alerting rules"
                ],
                "escalation_criteria": [
                    "> 10 critical alerts in 5 minutes",
                    "Multiple AI systems affected simultaneously",
                    "Customer-facing services impacted"
                ]
            },
            "data_drift_response": {
                "title": "Data Drift Detection Response",
                "description": "Handle detected data drift in ML models",
                "severity": "medium",
                "steps": [
                    "1. Confirm data drift detection accuracy",
                    "2. Analyze drift patterns and affected features",
                    "3. Investigate root cause of data changes",
                    "4. Assess business impact of drift",
                    "5. Collect new representative training data",
                    "6. Retrain model with updated data distribution",
                    "7. Implement drift monitoring improvements"
                ],
                "escalation_criteria": [
                    "Drift score > 70%",
                    "Multiple models affected",
                    "Prediction accuracy drops significantly",
                    "Business metrics impacted"
                ]
            }
        }
        
        return {
            "runbooks": runbooks,
            "count": len(runbooks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get operational runbooks: {str(e)}")


@router.get("/runbooks/{runbook_id}")
async def get_runbook_details(runbook_id: str):
    """Get detailed runbook for specific scenario"""
    try:
        # This would normally fetch from a database or configuration
        # For now, return the runbook from the static list
        runbooks = (await get_operational_runbooks())["runbooks"]
        
        if runbook_id not in runbooks:
            raise HTTPException(status_code=404, detail=f"Runbook {runbook_id} not found")
        
        runbook = runbooks[runbook_id]
        
        # Add additional context
        runbook["id"] = runbook_id
        runbook["last_updated"] = datetime.utcnow().isoformat()
        runbook["usage_count"] = 0  # Would track actual usage
        
        return runbook
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get runbook details: {str(e)}")