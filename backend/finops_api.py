"""
FinOps REST API - Comprehensive API for all FinOps platform capabilities

This module provides REST API endpoints for:
- Cost attribution and chargeback reports
- Budget management CRUD operations  
- Waste detection and optimization recommendations
- RI optimization and commitment analysis
- Tagging compliance and policy management
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date, timedelta
from enum import Enum
import json
import logging
from decimal import Decimal

# Import existing FinOps engines
try:
    from cost_attribution_engine import CostAttributionEngine
    from budget_management_system import BudgetManager, AlertManager
    from waste_detection_engine import WasteDetectionEngine
    from ri_optimization_system import RIRecommendationEngine, RIPortfolioManager
    from tagging_compliance_system import TaggingComplianceSystem
    from tagging_policy_manager import TagPolicyManager
    from finops_engine import FinOpsEngine
except ImportError as e:
    logging.warning(f"Could not import FinOps engines: {e}")
    # Create mock classes for development
    class CostAttributionEngine:
        def get_cost_data(self, **kwargs): return {}
        def generate_chargeback_report(self, **kwargs): return {}
        def get_untagged_resources(self): return []
    
    class BudgetManager:
        def create_budget(self, **kwargs): return {}
        def get_budgets(self): return []
        def update_budget(self, **kwargs): return {}
        def delete_budget(self, **kwargs): return True
    
    class WasteDetectionEngine:
        def analyze_waste(self): return {}
        def get_optimization_recommendations(self): return []
    
    class RIRecommendationEngine:
        def get_ri_recommendations(self): return []
        def calculate_savings(self, **kwargs): return {}
    
    class TaggingComplianceSystem:
        def get_compliance_report(self): return {}
        def get_violations(self): return []

# Initialize security
security = HTTPBearer()

# Pydantic Models for API requests/responses

class TimeRange(BaseModel):
    """Time range for filtering data"""
    start_date: date
    end_date: date
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class CostDimension(str, Enum):
    """Cost attribution dimensions"""
    TEAM = "team"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    DEPARTMENT = "department"
    COST_CENTER = "cost_center"
    SERVICE = "service"

class AllocationMethod(str, Enum):
    """Cost allocation methods"""
    DIRECT = "direct"
    PROPORTIONAL = "proportional"
    USAGE_BASED = "usage_based"
    EQUAL_SPLIT = "equal_split"

# Cost Attribution Models
class CostDataRequest(BaseModel):
    """Request for cost data"""
    time_range: TimeRange
    dimensions: List[CostDimension] = Field(default=[CostDimension.TEAM])
    filters: Optional[Dict[str, Any]] = None
    granularity: str = Field(default="daily", regex="^(hourly|daily|weekly|monthly)$")

class CostDataResponse(BaseModel):
    """Cost data response"""
    total_cost: Decimal
    cost_breakdown: Dict[str, Decimal]
    time_range: TimeRange
    currency: str = "USD"
    last_updated: datetime

class ChargebackRequest(BaseModel):
    """Request for chargeback report"""
    cost_center: str
    time_range: TimeRange
    allocation_method: AllocationMethod = AllocationMethod.DIRECT
    include_shared_costs: bool = True

class ChargebackResponse(BaseModel):
    """Chargeback report response"""
    cost_center: str
    total_cost: Decimal
    direct_costs: Decimal
    allocated_shared_costs: Decimal
    cost_breakdown: Dict[str, Decimal]
    allocation_method: AllocationMethod
    time_range: TimeRange
    generated_at: datetime

class UntaggedResourceResponse(BaseModel):
    """Untagged resource information"""
    resource_id: str
    resource_type: str
    provider: str
    region: str
    estimated_monthly_cost: Decimal
    created_date: datetime
    suggested_tags: Dict[str, str]

# Budget Management Models
class BudgetType(str, Enum):
    """Budget types"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    PROJECT = "project"

class BudgetScope(BaseModel):
    """Budget scope definition"""
    dimension: CostDimension
    values: List[str]
    filters: Optional[Dict[str, Any]] = None

class BudgetRequest(BaseModel):
    """Budget creation/update request"""
    name: str = Field(..., min_length=1, max_length=100)
    amount: Decimal = Field(..., gt=0)
    budget_type: BudgetType
    scope: BudgetScope
    alert_thresholds: List[int] = Field(default=[50, 75, 90, 100])
    notification_emails: List[str] = []
    start_date: date
    end_date: Optional[date] = None

class BudgetResponse(BaseModel):
    """Budget response"""
    budget_id: str
    name: str
    amount: Decimal
    budget_type: BudgetType
    scope: BudgetScope
    alert_thresholds: List[int]
    current_spend: Decimal
    remaining_budget: Decimal
    utilization_percentage: float
    status: str
    created_at: datetime
    updated_at: datetime

class BudgetAlert(BaseModel):
    """Budget alert information"""
    alert_id: str
    budget_id: str
    threshold_percentage: int
    current_spend: Decimal
    budget_amount: Decimal
    alert_type: str
    triggered_at: datetime
    acknowledged: bool = False

# Waste Detection Models
class WasteType(str, Enum):
    """Types of waste detected"""
    UNUSED_RESOURCES = "unused_resources"
    UNDERUTILIZED_RESOURCES = "underutilized_resources"
    OVERSIZED_RESOURCES = "oversized_resources"
    ORPHANED_RESOURCES = "orphaned_resources"

class OptimizationRecommendation(BaseModel):
    """Resource optimization recommendation"""
    recommendation_id: str
    resource_id: str
    resource_type: str
    waste_type: WasteType
    current_cost: Decimal
    optimized_cost: Decimal
    potential_savings: Decimal
    confidence_score: float = Field(..., ge=0, le=1)
    risk_level: str
    recommendation_text: str
    implementation_effort: str
    estimated_downtime: Optional[str] = None

class WasteAnalysisRequest(BaseModel):
    """Request for waste analysis"""
    analysis_period_days: int = Field(default=30, ge=7, le=90)
    resource_types: Optional[List[str]] = None
    providers: Optional[List[str]] = None
    minimum_savings_threshold: Decimal = Field(default=Decimal("10.00"), ge=0)

class WasteAnalysisResponse(BaseModel):
    """Waste analysis results"""
    total_waste_cost: Decimal
    potential_savings: Decimal
    waste_breakdown: Dict[WasteType, Decimal]
    recommendations: List[OptimizationRecommendation]
    analysis_period: TimeRange
    generated_at: datetime

# RI Optimization Models
class CommitmentType(str, Enum):
    """Types of commitments"""
    RESERVED_INSTANCES = "reserved_instances"
    SAVINGS_PLANS = "savings_plans"
    COMMITTED_USE_DISCOUNTS = "committed_use_discounts"

class RIRecommendationRequest(BaseModel):
    """Request for RI recommendations"""
    analysis_period_days: int = Field(default=30, ge=30, le=365)
    commitment_term_months: int = Field(default=12, regex="^(12|36)$")
    payment_option: str = Field(default="no_upfront", regex="^(no_upfront|partial_upfront|all_upfront)$")
    minimum_savings_threshold: float = Field(default=0.1, ge=0, le=1)

class RIRecommendation(BaseModel):
    """RI recommendation"""
    recommendation_id: str
    resource_type: str
    instance_family: str
    region: str
    commitment_type: CommitmentType
    commitment_term_months: int
    payment_option: str
    recommended_quantity: int
    hourly_on_demand_cost: Decimal
    hourly_reserved_cost: Decimal
    estimated_monthly_savings: Decimal
    estimated_annual_savings: Decimal
    payback_period_months: float
    utilization_requirement: float
    confidence_score: float

class RIUtilizationResponse(BaseModel):
    """RI utilization tracking"""
    reservation_id: str
    instance_type: str
    region: str
    total_reserved_hours: int
    used_reserved_hours: int
    utilization_percentage: float
    wasted_cost: Decimal
    coverage_percentage: float
    recommendation: Optional[str] = None

# Tagging Compliance Models
class TagPolicy(BaseModel):
    """Tagging policy definition"""
    policy_name: str = Field(..., min_length=1, max_length=100)
    required_tags: List[str]
    optional_tags: List[str] = []
    tag_value_patterns: Optional[Dict[str, str]] = None
    resource_types: List[str] = []
    enforcement_level: str = Field(default="warning", regex="^(warning|blocking|audit_only)$")
    auto_remediation: bool = False

class TagViolation(BaseModel):
    """Tagging violation"""
    violation_id: str
    resource_id: str
    resource_type: str
    provider: str
    region: str
    violation_type: str
    missing_tags: List[str]
    invalid_tags: Dict[str, str]
    suggested_tags: Dict[str, str]
    detected_at: datetime
    severity: str

class ComplianceMetrics(BaseModel):
    """Tagging compliance metrics"""
    total_resources: int
    compliant_resources: int
    non_compliant_resources: int
    compliance_percentage: float
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]
    trend_data: Dict[str, float]
    last_updated: datetime

# Initialize FinOps engines (will be injected via dependency injection)
finops_engines = {}

def get_cost_attribution_engine():
    """Dependency injection for cost attribution engine"""
    if 'cost_attribution' not in finops_engines:
        finops_engines['cost_attribution'] = CostAttributionEngine()
    return finops_engines['cost_attribution']

def get_budget_manager():
    """Dependency injection for budget manager"""
    if 'budget_manager' not in finops_engines:
        finops_engines['budget_manager'] = BudgetManager()
    return finops_engines['budget_manager']

def get_waste_detection_engine():
    """Dependency injection for waste detection engine"""
    if 'waste_detection' not in finops_engines:
        finops_engines['waste_detection'] = WasteDetectionEngine()
    return finops_engines['waste_detection']

def get_ri_recommendation_engine():
    """Dependency injection for RI recommendation engine"""
    if 'ri_recommendation' not in finops_engines:
        finops_engines['ri_recommendation'] = RIRecommendationEngine()
    return finops_engines['ri_recommendation']

def get_tagging_compliance_system():
    """Dependency injection for tagging compliance system"""
    if 'tagging_compliance' not in finops_engines:
        finops_engines['tagging_compliance'] = TaggingComplianceSystem()
    return finops_engines['tagging_compliance']

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token - placeholder for actual authentication"""
    # TODO: Implement actual token verification
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

# Create FastAPI app
finops_app = FastAPI(
    title="FinOps Platform API",
    description="Comprehensive REST API for FinOps platform capabilities",
    version="1.0.0",
    docs_url="/finops/docs",
    redoc_url="/finops/redoc"
)

# Add CORS middleware
finops_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@finops_app.get("/finops/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# ===== COST ATTRIBUTION ENDPOINTS =====

@finops_app.post("/finops/cost-attribution/data", response_model=CostDataResponse)
async def get_cost_data(
    request: CostDataRequest,
    token: str = Depends(verify_token),
    engine: CostAttributionEngine = Depends(get_cost_attribution_engine)
):
    """
    Get detailed cost data with attribution by specified dimensions
    
    Supports filtering by team, project, environment, department, cost center, or service.
    Returns cost breakdown with configurable granularity (hourly, daily, weekly, monthly).
    """
    try:
        cost_data = engine.get_cost_data(
            start_date=request.time_range.start_date,
            end_date=request.time_range.end_date,
            dimensions=request.dimensions,
            filters=request.filters,
            granularity=request.granularity
        )
        
        return CostDataResponse(
            total_cost=Decimal(str(cost_data.get('total_cost', 0))),
            cost_breakdown=cost_data.get('breakdown', {}),
            time_range=request.time_range,
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cost data: {str(e)}")

@finops_app.post("/finops/cost-attribution/chargeback", response_model=ChargebackResponse)
async def generate_chargeback_report(
    request: ChargebackRequest,
    token: str = Depends(verify_token),
    engine: CostAttributionEngine = Depends(get_cost_attribution_engine)
):
    """
    Generate detailed chargeback report for a specific cost center
    
    Includes direct costs and allocated shared costs based on the specified allocation method.
    Supports multiple allocation methods: direct, proportional, usage-based, equal split.
    """
    try:
        chargeback_data = engine.generate_chargeback_report(
            cost_center=request.cost_center,
            start_date=request.time_range.start_date,
            end_date=request.time_range.end_date,
            allocation_method=request.allocation_method.value,
            include_shared_costs=request.include_shared_costs
        )
        
        return ChargebackResponse(
            cost_center=request.cost_center,
            total_cost=Decimal(str(chargeback_data.get('total_cost', 0))),
            direct_costs=Decimal(str(chargeback_data.get('direct_costs', 0))),
            allocated_shared_costs=Decimal(str(chargeback_data.get('shared_costs', 0))),
            cost_breakdown=chargeback_data.get('breakdown', {}),
            allocation_method=request.allocation_method,
            time_range=request.time_range,
            generated_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chargeback report: {str(e)}")

@finops_app.get("/finops/cost-attribution/untagged-resources", response_model=List[UntaggedResourceResponse])
async def get_untagged_resources(
    provider: Optional[str] = Query(None, description="Filter by cloud provider"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    min_cost: Optional[float] = Query(0, description="Minimum monthly cost threshold"),
    token: str = Depends(verify_token),
    engine: CostAttributionEngine = Depends(get_cost_attribution_engine)
):
    """
    Get list of untagged resources with cost impact and tag suggestions
    
    Identifies resources missing required tags and provides intelligent tag suggestions
    based on naming patterns and organizational context.
    """
    try:
        untagged_resources = engine.get_untagged_resources(
            provider=provider,
            resource_type=resource_type,
            min_cost=min_cost
        )
        
        return [
            UntaggedResourceResponse(
                resource_id=resource['resource_id'],
                resource_type=resource['resource_type'],
                provider=resource['provider'],
                region=resource['region'],
                estimated_monthly_cost=Decimal(str(resource['monthly_cost'])),
                created_date=resource['created_date'],
                suggested_tags=resource.get('suggested_tags', {})
            )
            for resource in untagged_resources
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving untagged resources: {str(e)}")

# ===== BUDGET MANAGEMENT ENDPOINTS =====

@finops_app.post("/finops/budgets", response_model=BudgetResponse)
async def create_budget(
    budget: BudgetRequest,
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Create a new budget with specified scope and alert thresholds
    
    Supports flexible budget creation by team, project, service, or custom dimensions.
    Configurable alert thresholds and notification channels.
    """
    try:
        created_budget = manager.create_budget(
            name=budget.name,
            amount=float(budget.amount),
            budget_type=budget.budget_type.value,
            scope=budget.scope.dict(),
            alert_thresholds=budget.alert_thresholds,
            notification_emails=budget.notification_emails,
            start_date=budget.start_date,
            end_date=budget.end_date
        )
        
        return BudgetResponse(
            budget_id=created_budget['budget_id'],
            name=created_budget['name'],
            amount=Decimal(str(created_budget['amount'])),
            budget_type=BudgetType(created_budget['budget_type']),
            scope=BudgetScope(**created_budget['scope']),
            alert_thresholds=created_budget['alert_thresholds'],
            current_spend=Decimal(str(created_budget.get('current_spend', 0))),
            remaining_budget=Decimal(str(created_budget.get('remaining_budget', created_budget['amount']))),
            utilization_percentage=created_budget.get('utilization_percentage', 0.0),
            status=created_budget.get('status', 'active'),
            created_at=created_budget['created_at'],
            updated_at=created_budget['updated_at']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating budget: {str(e)}")

@finops_app.get("/finops/budgets", response_model=List[BudgetResponse])
async def get_budgets(
    budget_type: Optional[BudgetType] = Query(None, description="Filter by budget type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Get list of all budgets with current spending and utilization
    
    Returns comprehensive budget information including current spend, remaining budget,
    and utilization percentage. Supports filtering by type and status.
    """
    try:
        budgets = manager.get_budgets(
            budget_type=budget_type.value if budget_type else None,
            status=status
        )
        
        return [
            BudgetResponse(
                budget_id=budget['budget_id'],
                name=budget['name'],
                amount=Decimal(str(budget['amount'])),
                budget_type=BudgetType(budget['budget_type']),
                scope=BudgetScope(**budget['scope']),
                alert_thresholds=budget['alert_thresholds'],
                current_spend=Decimal(str(budget.get('current_spend', 0))),
                remaining_budget=Decimal(str(budget.get('remaining_budget', budget['amount']))),
                utilization_percentage=budget.get('utilization_percentage', 0.0),
                status=budget.get('status', 'active'),
                created_at=budget['created_at'],
                updated_at=budget['updated_at']
            )
            for budget in budgets
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving budgets: {str(e)}")

@finops_app.get("/finops/budgets/{budget_id}", response_model=BudgetResponse)
async def get_budget(
    budget_id: str = Path(..., description="Budget ID"),
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Get detailed information for a specific budget
    
    Returns complete budget details including current spending, utilization,
    and recent alert history.
    """
    try:
        budget = manager.get_budget(budget_id)
        if not budget:
            raise HTTPException(status_code=404, detail="Budget not found")
        
        return BudgetResponse(
            budget_id=budget['budget_id'],
            name=budget['name'],
            amount=Decimal(str(budget['amount'])),
            budget_type=BudgetType(budget['budget_type']),
            scope=BudgetScope(**budget['scope']),
            alert_thresholds=budget['alert_thresholds'],
            current_spend=Decimal(str(budget.get('current_spend', 0))),
            remaining_budget=Decimal(str(budget.get('remaining_budget', budget['amount']))),
            utilization_percentage=budget.get('utilization_percentage', 0.0),
            status=budget.get('status', 'active'),
            created_at=budget['created_at'],
            updated_at=budget['updated_at']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving budget: {str(e)}")

@finops_app.put("/finops/budgets/{budget_id}", response_model=BudgetResponse)
async def update_budget(
    budget_id: str = Path(..., description="Budget ID"),
    budget_update: BudgetRequest = Body(...),
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Update an existing budget
    
    Allows modification of budget amount, scope, alert thresholds, and notification settings.
    Maintains budget history and tracks changes for audit purposes.
    """
    try:
        updated_budget = manager.update_budget(
            budget_id=budget_id,
            name=budget_update.name,
            amount=float(budget_update.amount),
            budget_type=budget_update.budget_type.value,
            scope=budget_update.scope.dict(),
            alert_thresholds=budget_update.alert_thresholds,
            notification_emails=budget_update.notification_emails,
            start_date=budget_update.start_date,
            end_date=budget_update.end_date
        )
        
        if not updated_budget:
            raise HTTPException(status_code=404, detail="Budget not found")
        
        return BudgetResponse(
            budget_id=updated_budget['budget_id'],
            name=updated_budget['name'],
            amount=Decimal(str(updated_budget['amount'])),
            budget_type=BudgetType(updated_budget['budget_type']),
            scope=BudgetScope(**updated_budget['scope']),
            alert_thresholds=updated_budget['alert_thresholds'],
            current_spend=Decimal(str(updated_budget.get('current_spend', 0))),
            remaining_budget=Decimal(str(updated_budget.get('remaining_budget', updated_budget['amount']))),
            utilization_percentage=updated_budget.get('utilization_percentage', 0.0),
            status=updated_budget.get('status', 'active'),
            created_at=updated_budget['created_at'],
            updated_at=updated_budget['updated_at']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating budget: {str(e)}")

@finops_app.delete("/finops/budgets/{budget_id}")
async def delete_budget(
    budget_id: str = Path(..., description="Budget ID"),
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Delete a budget
    
    Removes the budget and associated alerts. Historical spending data is preserved
    for audit and reporting purposes.
    """
    try:
        success = manager.delete_budget(budget_id)
        if not success:
            raise HTTPException(status_code=404, detail="Budget not found")
        
        return {"message": "Budget deleted successfully", "budget_id": budget_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting budget: {str(e)}")

@finops_app.get("/finops/budgets/{budget_id}/alerts", response_model=List[BudgetAlert])
async def get_budget_alerts(
    budget_id: str = Path(..., description="Budget ID"),
    days: int = Query(30, description="Number of days to look back for alerts"),
    token: str = Depends(verify_token),
    manager: BudgetManager = Depends(get_budget_manager)
):
    """
    Get recent alerts for a specific budget
    
    Returns alert history including threshold breaches, acknowledgments,
    and escalation status.
    """
    try:
        alerts = manager.get_budget_alerts(budget_id, days=days)
        
        return [
            BudgetAlert(
                alert_id=alert['alert_id'],
                budget_id=alert['budget_id'],
                threshold_percentage=alert['threshold_percentage'],
                current_spend=Decimal(str(alert['current_spend'])),
                budget_amount=Decimal(str(alert['budget_amount'])),
                alert_type=alert['alert_type'],
                triggered_at=alert['triggered_at'],
                acknowledged=alert.get('acknowledged', False)
            )
            for alert in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving budget alerts: {str(e)}")
# =====
 WASTE DETECTION ENDPOINTS =====

@finops_app.post("/finops/waste-detection/analyze", response_model=WasteAnalysisResponse)
async def analyze_waste(
    request: WasteAnalysisRequest,
    token: str = Depends(verify_token),
    engine: WasteDetectionEngine = Depends(get_waste_detection_engine)
):
    """
    Perform comprehensive waste analysis across cloud resources
    
    Identifies unused, underutilized, and oversized resources with detailed
    optimization recommendations and potential savings calculations.
    """
    try:
        analysis_results = engine.analyze_waste(
            analysis_period_days=request.analysis_period_days,
            resource_types=request.resource_types,
            providers=request.providers,
            minimum_savings_threshold=float(request.minimum_savings_threshold)
        )
        
        recommendations = [
            OptimizationRecommendation(
                recommendation_id=rec['recommendation_id'],
                resource_id=rec['resource_id'],
                resource_type=rec['resource_type'],
                waste_type=WasteType(rec['waste_type']),
                current_cost=Decimal(str(rec['current_cost'])),
                optimized_cost=Decimal(str(rec['optimized_cost'])),
                potential_savings=Decimal(str(rec['potential_savings'])),
                confidence_score=rec['confidence_score'],
                risk_level=rec['risk_level'],
                recommendation_text=rec['recommendation_text'],
                implementation_effort=rec['implementation_effort'],
                estimated_downtime=rec.get('estimated_downtime')
            )
            for rec in analysis_results.get('recommendations', [])
        ]
        
        return WasteAnalysisResponse(
            total_waste_cost=Decimal(str(analysis_results.get('total_waste_cost', 0))),
            potential_savings=Decimal(str(analysis_results.get('potential_savings', 0))),
            waste_breakdown=analysis_results.get('waste_breakdown', {}),
            recommendations=recommendations,
            analysis_period=TimeRange(
                start_date=datetime.now().date() - timedelta(days=request.analysis_period_days),
                end_date=datetime.now().date()
            ),
            generated_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing waste: {str(e)}")

@finops_app.get("/finops/waste-detection/recommendations", response_model=List[OptimizationRecommendation])
async def get_optimization_recommendations(
    waste_type: Optional[WasteType] = Query(None, description="Filter by waste type"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    min_savings: Optional[float] = Query(0, description="Minimum savings threshold"),
    token: str = Depends(verify_token),
    engine: WasteDetectionEngine = Depends(get_waste_detection_engine)
):
    """
    Get current optimization recommendations
    
    Returns actionable recommendations for reducing cloud waste with detailed
    cost impact analysis and implementation guidance.
    """
    try:
        recommendations = engine.get_optimization_recommendations(
            waste_type=waste_type.value if waste_type else None,
            risk_level=risk_level,
            min_savings=min_savings
        )
        
        return [
            OptimizationRecommendation(
                recommendation_id=rec['recommendation_id'],
                resource_id=rec['resource_id'],
                resource_type=rec['resource_type'],
                waste_type=WasteType(rec['waste_type']),
                current_cost=Decimal(str(rec['current_cost'])),
                optimized_cost=Decimal(str(rec['optimized_cost'])),
                potential_savings=Decimal(str(rec['potential_savings'])),
                confidence_score=rec['confidence_score'],
                risk_level=rec['risk_level'],
                recommendation_text=rec['recommendation_text'],
                implementation_effort=rec['implementation_effort'],
                estimated_downtime=rec.get('estimated_downtime')
            )
            for rec in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recommendations: {str(e)}")

@finops_app.post("/finops/waste-detection/recommendations/{recommendation_id}/implement")
async def implement_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID"),
    confirm: bool = Body(..., description="Confirmation to proceed with implementation"),
    token: str = Depends(verify_token),
    engine: WasteDetectionEngine = Depends(get_waste_detection_engine)
):
    """
    Implement an optimization recommendation
    
    Executes the recommended optimization action with safety checks and rollback capability.
    Requires explicit confirmation due to potential impact on resources.
    """
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Implementation requires explicit confirmation")
        
        result = engine.implement_recommendation(recommendation_id)
        
        return {
            "recommendation_id": recommendation_id,
            "status": result.get('status', 'completed'),
            "message": result.get('message', 'Recommendation implemented successfully'),
            "implemented_at": datetime.now().isoformat(),
            "rollback_available": result.get('rollback_available', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error implementing recommendation: {str(e)}")

# ===== RI OPTIMIZATION ENDPOINTS =====

@finops_app.post("/finops/ri-optimization/recommendations", response_model=List[RIRecommendation])
async def get_ri_recommendations(
    request: RIRecommendationRequest,
    token: str = Depends(verify_token),
    engine: RIRecommendationEngine = Depends(get_ri_recommendation_engine)
):
    """
    Get intelligent Reserved Instance recommendations
    
    Analyzes usage patterns to identify optimal RI purchase opportunities
    with detailed savings calculations and ROI analysis.
    """
    try:
        recommendations = engine.get_ri_recommendations(
            analysis_period_days=request.analysis_period_days,
            commitment_term_months=request.commitment_term_months,
            payment_option=request.payment_option,
            minimum_savings_threshold=request.minimum_savings_threshold
        )
        
        return [
            RIRecommendation(
                recommendation_id=rec['recommendation_id'],
                resource_type=rec['resource_type'],
                instance_family=rec['instance_family'],
                region=rec['region'],
                commitment_type=CommitmentType(rec['commitment_type']),
                commitment_term_months=rec['commitment_term_months'],
                payment_option=rec['payment_option'],
                recommended_quantity=rec['recommended_quantity'],
                hourly_on_demand_cost=Decimal(str(rec['hourly_on_demand_cost'])),
                hourly_reserved_cost=Decimal(str(rec['hourly_reserved_cost'])),
                estimated_monthly_savings=Decimal(str(rec['estimated_monthly_savings'])),
                estimated_annual_savings=Decimal(str(rec['estimated_annual_savings'])),
                payback_period_months=rec['payback_period_months'],
                utilization_requirement=rec['utilization_requirement'],
                confidence_score=rec['confidence_score']
            )
            for rec in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating RI recommendations: {str(e)}")

@finops_app.get("/finops/ri-optimization/utilization", response_model=List[RIUtilizationResponse])
async def get_ri_utilization(
    region: Optional[str] = Query(None, description="Filter by region"),
    instance_type: Optional[str] = Query(None, description="Filter by instance type"),
    token: str = Depends(verify_token),
    engine: RIRecommendationEngine = Depends(get_ri_recommendation_engine)
):
    """
    Get current RI utilization and coverage metrics
    
    Tracks utilization of existing reserved instances and identifies
    underutilized commitments or coverage gaps.
    """
    try:
        utilization_data = engine.get_ri_utilization(
            region=region,
            instance_type=instance_type
        )
        
        return [
            RIUtilizationResponse(
                reservation_id=util['reservation_id'],
                instance_type=util['instance_type'],
                region=util['region'],
                total_reserved_hours=util['total_reserved_hours'],
                used_reserved_hours=util['used_reserved_hours'],
                utilization_percentage=util['utilization_percentage'],
                wasted_cost=Decimal(str(util['wasted_cost'])),
                coverage_percentage=util['coverage_percentage'],
                recommendation=util.get('recommendation')
            )
            for util in utilization_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving RI utilization: {str(e)}")

@finops_app.post("/finops/ri-optimization/savings-calculator")
async def calculate_ri_savings(
    instance_type: str = Body(..., description="Instance type"),
    region: str = Body(..., description="AWS region"),
    quantity: int = Body(..., description="Number of instances"),
    commitment_term_months: int = Body(..., description="Commitment term (12 or 36 months)"),
    payment_option: str = Body(..., description="Payment option"),
    token: str = Depends(verify_token),
    engine: RIRecommendationEngine = Depends(get_ri_recommendation_engine)
):
    """
    Calculate potential savings for specific RI configuration
    
    Provides detailed financial analysis for a specific RI purchase scenario
    including break-even analysis and total cost of ownership.
    """
    try:
        savings_analysis = engine.calculate_ri_savings(
            instance_type=instance_type,
            region=region,
            quantity=quantity,
            commitment_term_months=commitment_term_months,
            payment_option=payment_option
        )
        
        return {
            "instance_type": instance_type,
            "region": region,
            "quantity": quantity,
            "commitment_term_months": commitment_term_months,
            "payment_option": payment_option,
            "on_demand_cost_monthly": Decimal(str(savings_analysis['on_demand_cost_monthly'])),
            "reserved_cost_monthly": Decimal(str(savings_analysis['reserved_cost_monthly'])),
            "monthly_savings": Decimal(str(savings_analysis['monthly_savings'])),
            "annual_savings": Decimal(str(savings_analysis['annual_savings'])),
            "total_savings_over_term": Decimal(str(savings_analysis['total_savings_over_term'])),
            "payback_period_months": savings_analysis['payback_period_months'],
            "savings_percentage": savings_analysis['savings_percentage'],
            "upfront_cost": Decimal(str(savings_analysis.get('upfront_cost', 0))),
            "break_even_utilization": savings_analysis['break_even_utilization']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating RI savings: {str(e)}")

# ===== TAGGING COMPLIANCE ENDPOINTS =====

@finops_app.post("/finops/tagging/policies")
async def create_tag_policy(
    policy: TagPolicy,
    token: str = Depends(verify_token),
    system: TaggingComplianceSystem = Depends(get_tagging_compliance_system)
):
    """
    Create a new tagging policy
    
    Defines required and optional tags for resources with validation patterns
    and enforcement levels. Supports automated remediation actions.
    """
    try:
        created_policy = system.create_tag_policy(
            policy_name=policy.policy_name,
            required_tags=policy.required_tags,
            optional_tags=policy.optional_tags,
            tag_value_patterns=policy.tag_value_patterns,
            resource_types=policy.resource_types,
            enforcement_level=policy.enforcement_level,
            auto_remediation=policy.auto_remediation
        )
        
        return {
            "policy_id": created_policy['policy_id'],
            "policy_name": created_policy['policy_name'],
            "status": "created",
            "created_at": created_policy['created_at']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating tag policy: {str(e)}")

@finops_app.get("/finops/tagging/policies")
async def get_tag_policies(
    enforcement_level: Optional[str] = Query(None, description="Filter by enforcement level"),
    token: str = Depends(verify_token),
    system: TaggingComplianceSystem = Depends(get_tagging_compliance_system)
):
    """
    Get list of all tagging policies
    
    Returns comprehensive policy information including enforcement settings,
    compliance metrics, and recent violation counts.
    """
    try:
        policies = system.get_tag_policies(enforcement_level=enforcement_level)
        
        return [
            {
                "policy_id": policy['policy_id'],
                "policy_name": policy['policy_name'],
                "required_tags": policy['required_tags'],
                "optional_tags": policy['optional_tags'],
                "tag_value_patterns": policy.get('tag_value_patterns', {}),
                "resource_types": policy['resource_types'],
                "enforcement_level": policy['enforcement_level'],
                "auto_remediation": policy['auto_remediation'],
                "compliance_percentage": policy.get('compliance_percentage', 0),
                "violation_count": policy.get('violation_count', 0),
                "created_at": policy['created_at'],
                "updated_at": policy['updated_at']
            }
            for policy in policies
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tag policies: {str(e)}")

@finops_app.get("/finops/tagging/compliance", response_model=ComplianceMetrics)
async def get_compliance_metrics(
    time_range_days: int = Query(30, description="Time range for trend analysis"),
    token: str = Depends(verify_token),
    system: TaggingComplianceSystem = Depends(get_tagging_compliance_system)
):
    """
    Get overall tagging compliance metrics
    
    Provides comprehensive compliance dashboard with trend analysis,
    violation breakdown, and improvement recommendations.
    """
    try:
        metrics = system.get_compliance_metrics(time_range_days=time_range_days)
        
        return ComplianceMetrics(
            total_resources=metrics['total_resources'],
            compliant_resources=metrics['compliant_resources'],
            non_compliant_resources=metrics['non_compliant_resources'],
            compliance_percentage=metrics['compliance_percentage'],
            violations_by_type=metrics['violations_by_type'],
            violations_by_severity=metrics['violations_by_severity'],
            trend_data=metrics['trend_data'],
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving compliance metrics: {str(e)}")

@finops_app.get("/finops/tagging/violations", response_model=List[TagViolation])
async def get_tag_violations(
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    limit: int = Query(100, description="Maximum number of violations to return"),
    token: str = Depends(verify_token),
    system: TaggingComplianceSystem = Depends(get_tagging_compliance_system)
):
    """
    Get current tagging violations
    
    Returns detailed violation information with suggested remediation actions
    and automated fix options where available.
    """
    try:
        violations = system.get_violations(
            severity=severity,
            resource_type=resource_type,
            limit=limit
        )
        
        return [
            TagViolation(
                violation_id=violation['violation_id'],
                resource_id=violation['resource_id'],
                resource_type=violation['resource_type'],
                provider=violation['provider'],
                region=violation['region'],
                violation_type=violation['violation_type'],
                missing_tags=violation['missing_tags'],
                invalid_tags=violation['invalid_tags'],
                suggested_tags=violation['suggested_tags'],
                detected_at=violation['detected_at'],
                severity=violation['severity']
            )
            for violation in violations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tag violations: {str(e)}")

@finops_app.post("/finops/tagging/violations/{violation_id}/remediate")
async def remediate_violation(
    violation_id: str = Path(..., description="Violation ID"),
    suggested_tags: Dict[str, str] = Body(..., description="Tags to apply for remediation"),
    token: str = Depends(verify_token),
    system: TaggingComplianceSystem = Depends(get_tagging_compliance_system)
):
    """
    Remediate a specific tagging violation
    
    Applies suggested tags to resolve compliance violations with audit trail
    and rollback capability for safety.
    """
    try:
        result = system.remediate_violation(
            violation_id=violation_id,
            suggested_tags=suggested_tags
        )
        
        return {
            "violation_id": violation_id,
            "status": result.get('status', 'completed'),
            "applied_tags": result.get('applied_tags', suggested_tags),
            "remediated_at": datetime.now().isoformat(),
            "rollback_available": result.get('rollback_available', True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error remediating violation: {str(e)}")

# ===== GENERAL FINOPS ENDPOINTS =====

@finops_app.get("/finops/dashboard/summary")
async def get_dashboard_summary(
    time_range_days: int = Query(30, description="Time range for summary data"),
    token: str = Depends(verify_token)
):
    """
    Get comprehensive FinOps dashboard summary
    
    Provides high-level metrics across all FinOps capabilities including
    cost trends, budget status, optimization opportunities, and compliance metrics.
    """
    try:
        # Aggregate data from all engines
        cost_engine = get_cost_attribution_engine()
        budget_manager = get_budget_manager()
        waste_engine = get_waste_detection_engine()
        ri_engine = get_ri_recommendation_engine()
        compliance_system = get_tagging_compliance_system()
        
        # Get summary data from each component
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=time_range_days)
        
        cost_summary = cost_engine.get_cost_summary(start_date, end_date)
        budget_summary = budget_manager.get_budget_summary()
        waste_summary = waste_engine.get_waste_summary()
        ri_summary = ri_engine.get_ri_summary()
        compliance_summary = compliance_system.get_compliance_summary()
        
        return {
            "summary_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": time_range_days
            },
            "cost_metrics": {
                "total_spend": cost_summary.get('total_spend', 0),
                "spend_trend": cost_summary.get('spend_trend', 0),
                "top_cost_centers": cost_summary.get('top_cost_centers', []),
                "untagged_cost_percentage": cost_summary.get('untagged_cost_percentage', 0)
            },
            "budget_metrics": {
                "total_budgets": budget_summary.get('total_budgets', 0),
                "budgets_at_risk": budget_summary.get('budgets_at_risk', 0),
                "budget_utilization_avg": budget_summary.get('budget_utilization_avg', 0),
                "active_alerts": budget_summary.get('active_alerts', 0)
            },
            "optimization_metrics": {
                "total_waste_identified": waste_summary.get('total_waste_identified', 0),
                "potential_monthly_savings": waste_summary.get('potential_monthly_savings', 0),
                "optimization_opportunities": waste_summary.get('optimization_opportunities', 0),
                "implemented_savings": waste_summary.get('implemented_savings', 0)
            },
            "ri_metrics": {
                "ri_coverage_percentage": ri_summary.get('ri_coverage_percentage', 0),
                "ri_utilization_avg": ri_summary.get('ri_utilization_avg', 0),
                "potential_ri_savings": ri_summary.get('potential_ri_savings', 0),
                "underutilized_ris": ri_summary.get('underutilized_ris', 0)
            },
            "compliance_metrics": {
                "overall_compliance_percentage": compliance_summary.get('overall_compliance_percentage', 0),
                "total_violations": compliance_summary.get('total_violations', 0),
                "critical_violations": compliance_summary.get('critical_violations', 0),
                "compliance_trend": compliance_summary.get('compliance_trend', 0)
            },
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard summary: {str(e)}")

@finops_app.get("/finops/reports/executive")
async def generate_executive_report(
    report_period: str = Query("monthly", regex="^(weekly|monthly|quarterly)$"),
    format: str = Query("json", regex="^(json|pdf)$"),
    token: str = Depends(verify_token)
):
    """
    Generate executive FinOps report
    
    Comprehensive executive summary with key metrics, trends, and recommendations
    for senior leadership. Available in JSON and PDF formats.
    """
    try:
        # Calculate date range based on report period
        end_date = datetime.now().date()
        if report_period == "weekly":
            start_date = end_date - timedelta(days=7)
        elif report_period == "monthly":
            start_date = end_date - timedelta(days=30)
        elif report_period == "quarterly":
            start_date = end_date - timedelta(days=90)
        
        # Generate comprehensive report data
        report_data = {
            "report_metadata": {
                "report_type": "executive_summary",
                "period": report_period,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.now().isoformat(),
                "format": format
            },
            "executive_summary": {
                "total_cloud_spend": 0,  # Will be populated by actual engines
                "spend_variance": 0,
                "cost_optimization_achieved": 0,
                "budget_performance": "on_track",
                "key_recommendations": []
            },
            "cost_analysis": {
                "spend_by_service": {},
                "spend_by_team": {},
                "cost_trends": [],
                "budget_vs_actual": {}
            },
            "optimization_summary": {
                "waste_eliminated": 0,
                "ri_savings_realized": 0,
                "optimization_pipeline": [],
                "roi_metrics": {}
            },
            "governance_status": {
                "compliance_score": 0,
                "policy_violations": 0,
                "remediation_progress": 0
            },
            "recommendations": {
                "immediate_actions": [],
                "strategic_initiatives": [],
                "risk_mitigation": []
            }
        }
        
        if format == "pdf":
            # TODO: Implement PDF generation
            return {"message": "PDF generation not yet implemented", "data": report_data}
        
        return report_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating executive report: {str(e)}")

# Error handlers
@finops_app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@finops_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception in FinOps API: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

# Export the app for integration with main API
__all__ = ['finops_app']