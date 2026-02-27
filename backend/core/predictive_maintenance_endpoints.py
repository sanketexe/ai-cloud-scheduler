"""
Predictive Maintenance System API Endpoints

Provides REST API endpoints for the predictive maintenance system,
including health analysis, maintenance scheduling, and effectiveness tracking.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import uuid

from .predictive_maintenance_system import (
    PredictiveMaintenanceSystem,
    HealthMetric,
    HealthAssessment,
    MaintenanceRecommendation,
    MaintenanceOutcome,
    DegradationAlert,
    HealthStatus,
    MaintenanceType,
    MaintenancePriority,
    DegradationPattern
)
from .auth import get_current_user
from .models import User

logger = logging.getLogger(__name__)

# Initialize the predictive maintenance system
predictive_maintenance_system = PredictiveMaintenanceSystem()

router = APIRouter(prefix="/api/v1/predictive-maintenance", tags=["Predictive Maintenance"])

# Pydantic models for API requests/responses

class HealthMetricRequest(BaseModel):
    """Request model for health metric data"""
    resource_id: str
    metric_name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthAnalysisRequest(BaseModel):
    """Request model for health analysis"""
    resources: Dict[str, List[HealthMetricRequest]]

class HealthMetricResponse(BaseModel):
    """Response model for health metric"""
    resource_id: str
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthAssessmentResponse(BaseModel):
    """Response model for health assessment"""
    resource_id: str
    resource_type: str
    overall_status: str
    health_score: float
    metrics: List[HealthMetricResponse]
    issues_detected: List[str]
    recommendations: List[str]
    assessed_at: datetime
    next_assessment: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MaintenanceRecommendationResponse(BaseModel):
    """Response model for maintenance recommendation"""
    recommendation_id: str
    resource_id: str
    maintenance_type: str
    priority: str
    recommended_window_start: datetime
    recommended_window_end: datetime
    estimated_duration_hours: float
    description: str
    expected_benefits: List[str]
    risks_if_delayed: List[str]
    cost_estimate: Optional[float] = None
    confidence_score: float
    created_at: datetime

class DegradationAlertResponse(BaseModel):
    """Response model for degradation alert"""
    alert_id: str
    resource_id: str
    degradation_pattern: str
    severity: str
    detected_at: datetime
    predicted_failure_time: Optional[datetime] = None
    confidence: float
    description: str
    recommended_actions: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MaintenanceOutcomeRequest(BaseModel):
    """Request model for maintenance outcome"""
    recommendation_id: str
    resource_id: str
    executed_at: datetime
    duration_hours: float
    success: bool
    improvements_observed: List[str]
    issues_resolved: List[str]
    cost_actual: Optional[float] = None
    notes: str = ""

class EffectivenessReportResponse(BaseModel):
    """Response model for effectiveness report"""
    overall_effectiveness: float
    resource_type_effectiveness: Dict[str, float]
    prediction_accuracy: Dict[str, Dict[str, Any]]
    maintenance_statistics: Dict[str, Any]
    recommendations: List[str]

# Helper functions

def convert_health_metric_to_domain(metric_request: HealthMetricRequest) -> HealthMetric:
    """Convert API request model to domain model"""
    return HealthMetric(
        resource_id=metric_request.resource_id,
        metric_name=metric_request.metric_name,
        value=metric_request.value,
        timestamp=datetime.now(),
        unit=metric_request.unit,
        threshold_warning=metric_request.threshold_warning,
        threshold_critical=metric_request.threshold_critical,
        metadata=metric_request.metadata
    )

def convert_health_assessment_to_response(assessment: HealthAssessment) -> HealthAssessmentResponse:
    """Convert domain model to API response model"""
    return HealthAssessmentResponse(
        resource_id=assessment.resource_id,
        resource_type=assessment.resource_type,
        overall_status=assessment.overall_status.value,
        health_score=assessment.health_score,
        metrics=[
            HealthMetricResponse(
                resource_id=metric.resource_id,
                metric_name=metric.metric_name,
                value=metric.value,
                timestamp=metric.timestamp,
                unit=metric.unit,
                threshold_warning=metric.threshold_warning,
                threshold_critical=metric.threshold_critical,
                metadata=metric.metadata
            )
            for metric in assessment.metrics
        ],
        issues_detected=assessment.issues_detected,
        recommendations=assessment.recommendations,
        assessed_at=assessment.assessed_at,
        next_assessment=assessment.next_assessment,
        metadata=assessment.metadata
    )

def convert_maintenance_recommendation_to_response(recommendation: MaintenanceRecommendation) -> MaintenanceRecommendationResponse:
    """Convert domain model to API response model"""
    return MaintenanceRecommendationResponse(
        recommendation_id=recommendation.recommendation_id,
        resource_id=recommendation.resource_id,
        maintenance_type=recommendation.maintenance_type.value,
        priority=recommendation.priority.value,
        recommended_window_start=recommendation.recommended_window[0],
        recommended_window_end=recommendation.recommended_window[1],
        estimated_duration_hours=recommendation.estimated_duration.total_seconds() / 3600,
        description=recommendation.description,
        expected_benefits=recommendation.expected_benefits,
        risks_if_delayed=recommendation.risks_if_delayed,
        cost_estimate=recommendation.cost_estimate,
        confidence_score=recommendation.confidence_score,
        created_at=recommendation.created_at
    )

def convert_degradation_alert_to_response(alert: DegradationAlert) -> DegradationAlertResponse:
    """Convert domain model to API response model"""
    return DegradationAlertResponse(
        alert_id=alert.alert_id,
        resource_id=alert.resource_id,
        degradation_pattern=alert.degradation_pattern.value,
        severity=alert.severity.value,
        detected_at=alert.detected_at,
        predicted_failure_time=alert.predicted_failure_time,
        confidence=alert.confidence,
        description=alert.description,
        recommended_actions=alert.recommended_actions,
        metadata=alert.metadata
    )

# API Endpoints

@router.post("/analyze-health", response_model=Dict[str, HealthAssessmentResponse])
async def analyze_infrastructure_health(
    request: HealthAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze infrastructure health for multiple resources
    """
    try:
        logger.info(f"Health analysis requested by user {current_user.email} for {len(request.resources)} resources")
        
        # Convert request to domain models
        resources_metrics = {}
        for resource_id, metrics_requests in request.resources.items():
            resources_metrics[resource_id] = [
                convert_health_metric_to_domain(metric_request)
                for metric_request in metrics_requests
            ]
        
        # Perform health analysis
        assessments = await predictive_maintenance_system.analyze_infrastructure_health(
            resources_metrics
        )
        
        # Convert to response models
        response = {
            resource_id: convert_health_assessment_to_response(assessment)
            for resource_id, assessment in assessments.items()
        }
        
        logger.info(f"Health analysis completed for {len(response)} resources")
        return response
        
    except Exception as e:
        logger.error(f"Health analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health analysis failed: {str(e)}")

@router.get("/health-status/{resource_id}", response_model=HealthAssessmentResponse)
async def get_resource_health_status(
    resource_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get current health status for a specific resource
    """
    try:
        # Get current assessment from system
        if resource_id not in predictive_maintenance_system.active_assessments:
            raise HTTPException(status_code=404, detail=f"No health assessment found for resource {resource_id}")
        
        assessment = predictive_maintenance_system.active_assessments[resource_id]
        return convert_health_assessment_to_response(assessment)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health status for {resource_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")

@router.get("/maintenance-recommendations", response_model=List[MaintenanceRecommendationResponse])
async def get_maintenance_recommendations(
    resource_id: Optional[str] = None,
    priority: Optional[str] = None,
    maintenance_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get maintenance recommendations with optional filtering
    """
    try:
        recommendations = predictive_maintenance_system.maintenance_recommendations
        
        # Apply filters
        if resource_id:
            recommendations = [r for r in recommendations if r.resource_id == resource_id]
        
        if priority:
            try:
                priority_enum = MaintenancePriority(priority.lower())
                recommendations = [r for r in recommendations if r.priority == priority_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        if maintenance_type:
            try:
                type_enum = MaintenanceType(maintenance_type.lower())
                recommendations = [r for r in recommendations if r.maintenance_type == type_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid maintenance type: {maintenance_type}")
        
        # Convert to response models
        response = [
            convert_maintenance_recommendation_to_response(recommendation)
            for recommendation in recommendations
        ]
        
        logger.info(f"Retrieved {len(response)} maintenance recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get maintenance recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/degradation-alerts", response_model=List[DegradationAlertResponse])
async def get_degradation_alerts(
    resource_id: Optional[str] = None,
    severity: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get degradation alerts with optional filtering
    """
    try:
        alerts = predictive_maintenance_system.active_alerts
        
        # Apply filters
        if resource_id:
            alerts = [a for a in alerts if a.resource_id == resource_id]
        
        if severity:
            try:
                severity_enum = HealthStatus(severity.lower())
                alerts = [a for a in alerts if a.severity == severity_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        # Convert to response models
        response = [
            convert_degradation_alert_to_response(alert)
            for alert in alerts
        ]
        
        logger.info(f"Retrieved {len(response)} degradation alerts")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get degradation alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/maintenance-outcome")
async def record_maintenance_outcome(
    outcome_request: MaintenanceOutcomeRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Record the outcome of a maintenance action
    """
    try:
        logger.info(f"Recording maintenance outcome for resource {outcome_request.resource_id}")
        
        # Create domain model
        outcome = MaintenanceOutcome(
            outcome_id=str(uuid.uuid4()),
            recommendation_id=outcome_request.recommendation_id,
            resource_id=outcome_request.resource_id,
            executed_at=outcome_request.executed_at,
            duration=timedelta(hours=outcome_request.duration_hours),
            success=outcome_request.success,
            improvements_observed=outcome_request.improvements_observed,
            issues_resolved=outcome_request.issues_resolved,
            cost_actual=outcome_request.cost_actual,
            notes=outcome_request.notes
        )
        
        # Track effectiveness
        await predictive_maintenance_system.track_maintenance_effectiveness(outcome)
        
        logger.info(f"Maintenance outcome recorded successfully for {outcome_request.resource_id}")
        return {"message": "Maintenance outcome recorded successfully", "outcome_id": outcome.outcome_id}
        
    except Exception as e:
        logger.error(f"Failed to record maintenance outcome: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record outcome: {str(e)}")

@router.get("/effectiveness-report", response_model=EffectivenessReportResponse)
async def get_effectiveness_report(
    current_user: User = Depends(get_current_user)
):
    """
    Get system effectiveness report
    """
    try:
        logger.info("Generating effectiveness report")
        
        report = await predictive_maintenance_system.get_system_effectiveness_report()
        
        response = EffectivenessReportResponse(
            overall_effectiveness=report["overall_effectiveness"],
            resource_type_effectiveness=report["resource_type_effectiveness"],
            prediction_accuracy=report["prediction_accuracy"],
            maintenance_statistics=report["maintenance_statistics"],
            recommendations=report["recommendations"]
        )
        
        logger.info("Effectiveness report generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate effectiveness report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.post("/schedule-maintenance/{recommendation_id}")
async def schedule_maintenance(
    recommendation_id: str,
    usage_pattern: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Schedule maintenance for a specific recommendation
    """
    try:
        logger.info(f"Scheduling maintenance for recommendation {recommendation_id}")
        
        # Find the recommendation
        recommendation = next(
            (r for r in predictive_maintenance_system.maintenance_recommendations 
             if r.recommendation_id == recommendation_id), None
        )
        
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Recommendation {recommendation_id} not found")
        
        # Schedule maintenance
        scheduled_recommendation = await predictive_maintenance_system.maintenance_scheduler.schedule_maintenance(
            recommendation, usage_pattern
        )
        
        # Update the recommendation in the system
        for i, r in enumerate(predictive_maintenance_system.maintenance_recommendations):
            if r.recommendation_id == recommendation_id:
                predictive_maintenance_system.maintenance_recommendations[i] = scheduled_recommendation
                break
        
        response = convert_maintenance_recommendation_to_response(scheduled_recommendation)
        
        logger.info(f"Maintenance scheduled successfully for recommendation {recommendation_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule maintenance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule maintenance: {str(e)}")

@router.get("/dashboard")
async def get_predictive_maintenance_dashboard(
    current_user: User = Depends(get_current_user)
):
    """
    Get predictive maintenance dashboard data
    """
    try:
        logger.info("Generating predictive maintenance dashboard")
        
        # Get current system state
        active_assessments = len(predictive_maintenance_system.active_assessments)
        active_alerts = len(predictive_maintenance_system.active_alerts)
        pending_recommendations = len([
            r for r in predictive_maintenance_system.maintenance_recommendations
            if r.recommended_window[0] > datetime.now()
        ])
        
        # Calculate health distribution
        health_distribution = {"healthy": 0, "warning": 0, "degraded": 0, "critical": 0, "failing": 0}
        for assessment in predictive_maintenance_system.active_assessments.values():
            health_distribution[assessment.overall_status.value] += 1
        
        # Get recent alerts
        recent_alerts = sorted(
            predictive_maintenance_system.active_alerts,
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        # Get upcoming maintenance
        upcoming_maintenance = sorted([
            r for r in predictive_maintenance_system.maintenance_recommendations
            if r.recommended_window[0] > datetime.now()
        ], key=lambda x: x.recommended_window[0])[:10]
        
        dashboard_data = {
            "summary": {
                "active_assessments": active_assessments,
                "active_alerts": active_alerts,
                "pending_recommendations": pending_recommendations,
                "health_distribution": health_distribution
            },
            "recent_alerts": [
                convert_degradation_alert_to_response(alert)
                for alert in recent_alerts
            ],
            "upcoming_maintenance": [
                convert_maintenance_recommendation_to_response(recommendation)
                for recommendation in upcoming_maintenance
            ],
            "system_health_trend": {
                "timestamp": datetime.now(),
                "overall_health_score": sum(
                    assessment.health_score 
                    for assessment in predictive_maintenance_system.active_assessments.values()
                ) / max(len(predictive_maintenance_system.active_assessments), 1)
            }
        }
        
        logger.info("Predictive maintenance dashboard generated successfully")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

@router.delete("/alerts/{alert_id}")
async def dismiss_degradation_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Dismiss a degradation alert
    """
    try:
        logger.info(f"Dismissing degradation alert {alert_id}")
        
        # Find and remove the alert
        alert_found = False
        for i, alert in enumerate(predictive_maintenance_system.active_alerts):
            if alert.alert_id == alert_id:
                predictive_maintenance_system.active_alerts.pop(i)
                alert_found = True
                break
        
        if not alert_found:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        logger.info(f"Degradation alert {alert_id} dismissed successfully")
        return {"message": "Alert dismissed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dismiss alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dismiss alert: {str(e)}")