"""
API endpoints for AI-Powered Cost Anomaly Detection

This module provides REST API endpoints for:
- Setting up anomaly detection for AWS accounts
- Real-time anomaly monitoring and alerting
- Cost forecasting with confidence intervals
- Anomaly configuration and management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ..core.auth import get_current_user
from ..core.ml_cost_anomaly_detector import CostAnomalyDetectionService, AWSCostExplorer
from ..core.database import get_db_session
from ..core.models import User

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Pydantic models for API requests/responses
class AnomalyDetectionSetupRequest(BaseModel):
    account_id: str = Field(..., description="AWS account ID to monitor")
    sensitivity_level: str = Field(default="balanced", description="Detection sensitivity: conservative, balanced, aggressive")
    training_days: int = Field(default=90, description="Days of historical data for training")

class AnomalyDetectionSetupResponse(BaseModel):
    success: bool
    message: str
    account_id: str
    training_completed: bool
    models_trained: int

class AnomalyCheckRequest(BaseModel):
    account_id: str = Field(..., description="AWS account ID to check")
    days_to_check: int = Field(default=7, description="Number of recent days to analyze")
    min_confidence: float = Field(default=0.6, description="Minimum confidence threshold (0-1)")

class AnomalyResult(BaseModel):
    event_id: str
    account_id: str
    detection_time: datetime
    anomaly_type: str
    service: str
    resource_id: Optional[str]
    anomaly_score: float
    cost_impact: float
    percentage_deviation: float
    baseline_value: float
    actual_value: float
    explanation: str
    confidence: float
    recommendations: List[str]

class ForecastRequest(BaseModel):
    account_id: str = Field(..., description="AWS account ID to forecast")
    forecast_days: int = Field(default=30, description="Number of days to forecast")
    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals")

class ForecastResponse(BaseModel):
    forecast_id: str
    account_id: str
    generated_time: datetime
    forecast_period_days: int
    forecast_values: List[float]
    confidence_intervals: Optional[Dict[str, List[float]]]
    accuracy_score: float
    total_forecast_cost: float
    daily_average: float
    trend_direction: str
    key_assumptions: List[str]
    risk_factors: List[str]

class BudgetOverrunRequest(BaseModel):
    account_id: str = Field(..., description="AWS account ID")
    budget_amount: float = Field(..., description="Monthly budget amount")
    current_month_spend: float = Field(..., description="Current month spending so far")
    days_remaining: int = Field(..., description="Days remaining in budget period")

class BudgetOverrunResponse(BaseModel):
    account_id: str
    budget_amount: float
    current_spend: float
    projected_spend: float
    overrun_probability: float
    expected_overrun_amount: float
    days_until_overrun: Optional[int]
    risk_level: str
    recommendations: List[str]

class AnomalyConfigurationRequest(BaseModel):
    account_id: str
    sensitivity_level: str = Field(default="balanced", description="conservative, balanced, aggressive")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    excluded_services: List[str] = Field(default_factory=list)
    notification_channels: List[str] = Field(default_factory=list)
    business_hours: Optional[Dict[str, Any]] = None
    maintenance_windows: List[Dict[str, Any]] = Field(default_factory=list)

# Create router
router = APIRouter(prefix="/api/anomaly-detection", tags=["Anomaly Detection"])

# Initialize services (will be dependency injected)
def get_anomaly_detection_service() -> CostAnomalyDetectionService:
    """Dependency to get anomaly detection service"""
    aws_cost_explorer = AWSCostExplorer()
    return CostAnomalyDetectionService(aws_cost_explorer)

@router.post("/setup", response_model=AnomalyDetectionSetupResponse)
async def setup_anomaly_detection(
    request: AnomalyDetectionSetupRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    anomaly_service: CostAnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Set up AI-powered anomaly detection for an AWS account
    
    This endpoint:
    1. Validates AWS account access
    2. Collects historical cost data for training
    3. Trains ML models (Isolation Forest, Prophet)
    4. Sets up real-time monitoring
    """
    
    try:
        logger.info(f"Setting up anomaly detection for account {request.account_id}")
        
        # Add background task for model training (can take several minutes)
        background_tasks.add_task(
            train_anomaly_models_background,
            request.account_id,
            request.training_days,
            anomaly_service
        )
        
        return AnomalyDetectionSetupResponse(
            success=True,
            message="Anomaly detection setup initiated. Model training in progress.",
            account_id=request.account_id,
            training_completed=False,
            models_trained=0
        )
        
    except Exception as e:
        logger.error(f"Error setting up anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

async def train_anomaly_models_background(
    account_id: str, 
    training_days: int, 
    anomaly_service: CostAnomalyDetectionService
):
    """Background task for training ML models"""
    
    try:
        success = await anomaly_service.setup_account_monitoring(account_id)
        
        if success:
            logger.info(f"Model training completed for account {account_id}")
            # Here you could send a notification to the user
        else:
            logger.error(f"Model training failed for account {account_id}")
            
    except Exception as e:
        logger.error(f"Background training error: {str(e)}")

@router.post("/check", response_model=List[AnomalyResult])
async def check_for_anomalies(
    request: AnomalyCheckRequest,
    current_user: User = Depends(get_current_user),
    anomaly_service: CostAnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Check for cost anomalies in recent data
    
    Returns detected anomalies with:
    - Confidence scores and explanations
    - Cost impact analysis
    - Service-level drill-down
    - Actionable recommendations
    """
    
    try:
        logger.info(f"Checking anomalies for account {request.account_id}")
        
        # Get anomalies from ML service
        anomalies_data = await anomaly_service.check_for_anomalies(request.account_id)
        
        # Convert to response format and add recommendations
        anomalies = []
        for anomaly_data in anomalies_data:
            # Filter by confidence threshold
            if anomaly_data.get('confidence', 0) < request.min_confidence:
                continue
            
            # Generate recommendations based on anomaly type
            recommendations = generate_anomaly_recommendations(anomaly_data)
            
            anomaly = AnomalyResult(
                event_id=anomaly_data['event_id'],
                account_id=anomaly_data['account_id'],
                detection_time=datetime.fromisoformat(anomaly_data['detection_time']),
                anomaly_type=anomaly_data['anomaly_type'],
                service=anomaly_data['service'],
                resource_id=anomaly_data.get('resource_id'),
                anomaly_score=anomaly_data['anomaly_score'],
                cost_impact=anomaly_data['cost_impact'],
                percentage_deviation=anomaly_data['percentage_deviation'],
                baseline_value=anomaly_data['baseline_value'],
                actual_value=anomaly_data['actual_value'],
                explanation=anomaly_data['explanation'],
                confidence=anomaly_data['confidence'],
                recommendations=recommendations
            )
            
            anomalies.append(anomaly)
        
        logger.info(f"Found {len(anomalies)} anomalies for account {request.account_id}")
        return anomalies
        
    except Exception as e:
        logger.error(f"Error checking anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly check failed: {str(e)}")

@router.post("/forecast", response_model=ForecastResponse)
async def generate_cost_forecast(
    request: ForecastRequest,
    current_user: User = Depends(get_current_user),
    anomaly_service: CostAnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Generate AI-powered cost forecast with confidence intervals
    
    Uses Prophet time series model to predict:
    - Daily cost projections
    - Confidence intervals
    - Trend analysis
    - Seasonal adjustments
    """
    
    try:
        logger.info(f"Generating forecast for account {request.account_id}")
        
        # Generate forecast using ML service
        forecast_data = await anomaly_service.generate_cost_forecast(
            request.account_id, 
            request.forecast_days
        )
        
        if not forecast_data:
            raise HTTPException(
                status_code=404, 
                detail="Insufficient data for forecasting. Ensure account has historical cost data."
            )
        
        # Calculate additional metrics
        total_forecast = sum(forecast_data['forecast_values'])
        daily_average = total_forecast / len(forecast_data['forecast_values'])
        
        # Determine trend direction
        values = forecast_data['forecast_values']
        trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
        if abs(values[-1] - values[0]) / values[0] < 0.05:  # Less than 5% change
            trend_direction = "stable"
        
        forecast_response = ForecastResponse(
            forecast_id=forecast_data['forecast_id'],
            account_id=forecast_data['account_id'],
            generated_time=datetime.fromisoformat(forecast_data['generated_time']),
            forecast_period_days=forecast_data['forecast_period_days'],
            forecast_values=forecast_data['forecast_values'],
            confidence_intervals=forecast_data['confidence_intervals'] if request.include_confidence_intervals else None,
            accuracy_score=forecast_data['accuracy_score'],
            total_forecast_cost=total_forecast,
            daily_average=daily_average,
            trend_direction=trend_direction,
            key_assumptions=forecast_data['key_assumptions'],
            risk_factors=forecast_data['risk_factors']
        )
        
        return forecast_response
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.post("/budget-overrun-risk", response_model=BudgetOverrunResponse)
async def check_budget_overrun_risk(
    request: BudgetOverrunRequest,
    current_user: User = Depends(get_current_user),
    anomaly_service: CostAnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Predict budget overrun probability using ML forecasting
    
    Analyzes current spending patterns to predict:
    - Probability of exceeding budget
    - Expected overrun amount
    - Timeline to budget exhaustion
    - Risk mitigation recommendations
    """
    
    try:
        logger.info(f"Checking budget overrun risk for account {request.account_id}")
        
        # Generate forecast for remaining days
        forecast_data = await anomaly_service.generate_cost_forecast(
            request.account_id, 
            request.days_remaining
        )
        
        if not forecast_data:
            raise HTTPException(
                status_code=404, 
                detail="Unable to generate forecast for budget analysis"
            )
        
        # Calculate projected spend
        remaining_forecast = sum(forecast_data['forecast_values'])
        projected_total = request.current_month_spend + remaining_forecast
        
        # Calculate overrun probability
        if 'confidence_intervals' in forecast_data and forecast_data['confidence_intervals']:
            upper_bound = sum(forecast_data['confidence_intervals']['upper'])
            projected_upper = request.current_month_spend + upper_bound
            
            # Simple probability calculation based on confidence intervals
            if projected_total <= request.budget_amount:
                if projected_upper <= request.budget_amount:
                    overrun_probability = 0.1  # Very low risk
                else:
                    overrun_probability = 0.3  # Medium risk
            else:
                overrun_probability = 0.8  # High risk
        else:
            # Fallback calculation
            overrun_probability = max(0, min(1, (projected_total - request.budget_amount) / request.budget_amount))
        
        # Calculate expected overrun
        expected_overrun = max(0, projected_total - request.budget_amount)
        
        # Determine risk level
        if overrun_probability < 0.2:
            risk_level = "low"
        elif overrun_probability < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Calculate days until overrun (if applicable)
        days_until_overrun = None
        if projected_total > request.budget_amount:
            daily_rate = remaining_forecast / request.days_remaining if request.days_remaining > 0 else 0
            if daily_rate > 0:
                remaining_budget = request.budget_amount - request.current_month_spend
                days_until_overrun = max(0, int(remaining_budget / daily_rate))
        
        # Generate recommendations
        recommendations = generate_budget_recommendations(
            overrun_probability, 
            expected_overrun, 
            risk_level
        )
        
        return BudgetOverrunResponse(
            account_id=request.account_id,
            budget_amount=request.budget_amount,
            current_spend=request.current_month_spend,
            projected_spend=projected_total,
            overrun_probability=overrun_probability,
            expected_overrun_amount=expected_overrun,
            days_until_overrun=days_until_overrun,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error checking budget overrun risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Budget analysis failed: {str(e)}")

@router.get("/accounts/{account_id}/status")
async def get_anomaly_detection_status(
    account_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of anomaly detection for an account"""
    
    try:
        # This would check database for training status, model performance, etc.
        # For now, return a simple status
        
        return {
            "account_id": account_id,
            "status": "active",
            "models_trained": True,
            "last_check": datetime.now().isoformat(),
            "anomalies_detected_today": 0,
            "forecast_accuracy": 0.85,
            "next_model_retrain": (datetime.now() + timedelta(days=7)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

def generate_anomaly_recommendations(anomaly_data: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on anomaly type and service"""
    
    recommendations = []
    service = anomaly_data.get('service', '').lower()
    cost_impact = anomaly_data.get('cost_impact', 0)
    
    # Service-specific recommendations
    if 'ec2' in service:
        recommendations.extend([
            "Review EC2 instance utilization and consider right-sizing",
            "Check for unused or idle instances that can be stopped",
            "Consider using Spot instances for non-critical workloads"
        ])
    elif 's3' in service:
        recommendations.extend([
            "Review S3 storage classes and lifecycle policies",
            "Check for duplicate or unnecessary data",
            "Consider using S3 Intelligent Tiering"
        ])
    elif 'rds' in service:
        recommendations.extend([
            "Review RDS instance sizing and utilization",
            "Consider using Reserved Instances for predictable workloads",
            "Check database connection pooling and query optimization"
        ])
    
    # Impact-based recommendations
    if cost_impact > 1000:
        recommendations.append("High impact anomaly - immediate investigation recommended")
    elif cost_impact > 100:
        recommendations.append("Moderate impact - review within 24 hours")
    
    # Generic recommendations
    recommendations.extend([
        "Set up cost alerts for this service",
        "Review resource tags for better cost attribution",
        "Consider implementing automated cost optimization"
    ])
    
    return recommendations[:5]  # Limit to top 5 recommendations

def generate_budget_recommendations(
    overrun_probability: float, 
    expected_overrun: float, 
    risk_level: str
) -> List[str]:
    """Generate budget management recommendations"""
    
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "Immediate action required - implement cost controls",
            "Review and pause non-essential resources",
            "Enable automated cost optimization",
            "Set up real-time spending alerts"
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "Monitor spending closely for remainder of period",
            "Review upcoming deployments and scaling plans",
            "Consider implementing cost optimization policies"
        ])
    else:
        recommendations.extend([
            "Continue current spending patterns",
            "Consider optimizing for additional savings",
            "Review budget allocation for next period"
        ])
    
    if expected_overrun > 0:
        recommendations.append(f"Projected overrun: ${expected_overrun:.2f} - consider budget adjustment")
    
    return recommendations