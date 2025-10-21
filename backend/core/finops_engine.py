# finops_engine.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations without sklearn
    SKLEARN_AVAILABLE = False
    import math


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    OTHER = "other"


class CostCategory(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    OTHER = "other"


class BudgetPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class OptimizationType(Enum):
    RIGHT_SIZING = "right_sizing"
    RESERVED_INSTANCES = "reserved_instances"
    SPOT_INSTANCES = "spot_instances"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    SCHEDULING_OPTIMIZATION = "scheduling_optimization"


class EffortLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostData:
    """Represents cost data from cloud providers"""
    provider: CloudProvider
    service: str
    resource_id: str
    cost_amount: float
    currency: str
    billing_period_start: datetime
    billing_period_end: datetime
    cost_category: CostCategory
    tags: Dict[str, str] = field(default_factory=dict)
    region: Optional[str] = None
    project_id: Optional[str] = None
    team: Optional[str] = None
    
    def __post_init__(self):
        if self.cost_amount < 0:
            raise ValueError("Cost amount cannot be negative")
        if self.billing_period_start >= self.billing_period_end:
            raise ValueError("Billing period start must be before end")


@dataclass
class NormalizedCostData:
    """Standardized cost data across providers"""
    resource_id: str
    resource_type: str
    cost_amount_usd: float
    usage_quantity: float
    usage_unit: str
    timestamp: datetime
    provider: CloudProvider
    region: str
    project_id: str
    team: str
    tags: Dict[str, str] = field(default_factory=dict)
    cost_category: CostCategory = CostCategory.OTHER


@dataclass
class Budget:
    """Budget definition and tracking"""
    budget_id: str
    name: str
    amount: float
    currency: str
    period: BudgetPeriod
    start_date: datetime
    end_date: Optional[datetime] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [50.0, 80.0, 100.0])
    scope_filters: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"teams": ["team1"], "projects": ["proj1"]}
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Budget amount must be positive")
        if not self.alert_thresholds:
            self.alert_thresholds = [50.0, 80.0, 100.0]


@dataclass
class BudgetStatus:
    """Current budget utilization status"""
    budget: Budget
    current_spend: float
    utilization_percentage: float
    projected_spend: float
    days_remaining: int
    is_on_track: bool
    triggered_alerts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    optimization_id: str
    optimization_type: OptimizationType
    resource_id: str
    current_cost: float
    optimized_cost: float
    potential_savings: float
    confidence_score: float  # 0-1
    implementation_effort: EffortLevel
    recommendation: str
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if self.potential_savings < 0:
            raise ValueError("Potential savings cannot be negative")


@dataclass
class CostForecast:
    """Cost forecasting results"""
    forecast_period_start: datetime
    forecast_period_end: datetime
    predicted_costs: List[float]
    confidence_intervals: List[tuple]  # (lower, upper) bounds
    model_accuracy: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SpendingAnalysis:
    """Comprehensive spending pattern analysis"""
    analysis_period_start: datetime
    analysis_period_end: datetime
    total_spend: float
    spend_by_provider: Dict[CloudProvider, float]
    spend_by_category: Dict[CostCategory, float]
    spend_by_team: Dict[str, float]
    spend_by_project: Dict[str, float]
    top_cost_drivers: List[Dict[str, Any]]
    growth_rate: float  # Percentage
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    optimization_opportunities: List[CostOptimization] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class CloudBillingAPI(ABC):
    """Abstract base class for cloud provider billing APIs"""
    
    @abstractmethod
    async def get_cost_data(self, start_date: datetime, end_date: datetime, 
                           filters: Optional[Dict[str, Any]] = None) -> List[CostData]:
        """Retrieve cost data from cloud provider"""
        pass
    
    @abstractmethod
    async def get_usage_data(self, start_date: datetime, end_date: datetime,
                            resource_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve usage data from cloud provider"""
        pass
    
    @abstractmethod
    def normalize_cost_data(self, raw_data: List[Dict[str, Any]]) -> List[NormalizedCostData]:
        """Normalize provider-specific cost data to standard format"""
        pass


class AWSBillingAPI(CloudBillingAPI):
    """AWS Cost Explorer API integration"""
    
    def __init__(self, access_key: str, secret_key: str, region: str = "us-east-1"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.logger = logging.getLogger(f"{__name__}.AWSBillingAPI")
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime,
                           filters: Optional[Dict[str, Any]] = None) -> List[CostData]:
        """Retrieve AWS cost data"""
        # Simulate AWS Cost Explorer API call
        self.logger.info(f"Fetching AWS cost data from {start_date} to {end_date}")
        
        # Mock data for demonstration
        mock_data = [
            {
                "service": "Amazon EC2-Instance",
                "cost": 150.75,
                "usage_quantity": 744,  # hours
                "resource_id": "i-1234567890abcdef0",
                "region": "us-east-1",
                "tags": {"Team": "backend", "Project": "web-app"}
            },
            {
                "service": "Amazon S3",
                "cost": 25.30,
                "usage_quantity": 100,  # GB
                "resource_id": "bucket-web-assets",
                "region": "us-east-1",
                "tags": {"Team": "frontend", "Project": "web-app"}
            }
        ]
        
        cost_data = []
        for item in mock_data:
            cost_data.append(CostData(
                provider=CloudProvider.AWS,
                service=item["service"],
                resource_id=item["resource_id"],
                cost_amount=item["cost"],
                currency="USD",
                billing_period_start=start_date,
                billing_period_end=end_date,
                cost_category=self._categorize_aws_service(item["service"]),
                tags=item["tags"],
                region=item["region"]
            ))
        
        return cost_data
    
    async def get_usage_data(self, start_date: datetime, end_date: datetime,
                            resource_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve AWS usage data"""
        # Mock usage data
        return [
            {
                "resource_id": "i-1234567890abcdef0",
                "resource_type": "EC2Instance",
                "usage_hours": 744,
                "cpu_utilization": 65.5,
                "memory_utilization": 70.2
            }
        ]
    
    def normalize_cost_data(self, raw_data: List[Dict[str, Any]]) -> List[NormalizedCostData]:
        """Normalize AWS cost data"""
        normalized = []
        for item in raw_data:
            normalized.append(NormalizedCostData(
                resource_id=item["resource_id"],
                resource_type=item.get("resource_type", "unknown"),
                cost_amount_usd=item["cost"],
                usage_quantity=item["usage_quantity"],
                usage_unit=item.get("usage_unit", "hours"),
                timestamp=datetime.now(),
                provider=CloudProvider.AWS,
                region=item.get("region", "unknown"),
                project_id=item.get("tags", {}).get("Project", "unknown"),
                team=item.get("tags", {}).get("Team", "unknown"),
                tags=item.get("tags", {}),
                cost_category=self._categorize_aws_service(item.get("service", ""))
            ))
        return normalized
    
    def _categorize_aws_service(self, service_name: str) -> CostCategory:
        """Categorize AWS service into cost category"""
        service_lower = service_name.lower()
        if "ec2" in service_lower or "compute" in service_lower:
            return CostCategory.COMPUTE
        elif "s3" in service_lower or "storage" in service_lower:
            return CostCategory.STORAGE
        elif "rds" in service_lower or "database" in service_lower:
            return CostCategory.DATABASE
        elif "vpc" in service_lower or "network" in service_lower:
            return CostCategory.NETWORK
        else:
            return CostCategory.OTHER


class GCPBillingAPI(CloudBillingAPI):
    """Google Cloud Billing API integration"""
    
    def __init__(self, project_id: str, credentials_path: str):
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.logger = logging.getLogger(f"{__name__}.GCPBillingAPI")
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime,
                           filters: Optional[Dict[str, Any]] = None) -> List[CostData]:
        """Retrieve GCP cost data"""
        self.logger.info(f"Fetching GCP cost data from {start_date} to {end_date}")
        
        # Mock GCP data
        mock_data = [
            {
                "service": "Compute Engine",
                "cost": 120.45,
                "resource_id": "instance-1",
                "region": "us-central1",
                "tags": {"team": "ml", "project": "analytics"}
            }
        ]
        
        cost_data = []
        for item in mock_data:
            cost_data.append(CostData(
                provider=CloudProvider.GCP,
                service=item["service"],
                resource_id=item["resource_id"],
                cost_amount=item["cost"],
                currency="USD",
                billing_period_start=start_date,
                billing_period_end=end_date,
                cost_category=self._categorize_gcp_service(item["service"]),
                tags=item["tags"],
                region=item["region"]
            ))
        
        return cost_data
    
    async def get_usage_data(self, start_date: datetime, end_date: datetime,
                            resource_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve GCP usage data"""
        return []
    
    def normalize_cost_data(self, raw_data: List[Dict[str, Any]]) -> List[NormalizedCostData]:
        """Normalize GCP cost data"""
        # Similar to AWS normalization
        return []
    
    def _categorize_gcp_service(self, service_name: str) -> CostCategory:
        """Categorize GCP service into cost category"""
        service_lower = service_name.lower()
        if "compute" in service_lower:
            return CostCategory.COMPUTE
        elif "storage" in service_lower:
            return CostCategory.STORAGE
        elif "sql" in service_lower:
            return CostCategory.DATABASE
        else:
            return CostCategory.OTHER


class AzureBillingAPI(CloudBillingAPI):
    """Azure Cost Management API integration"""
    
    def __init__(self, subscription_id: str, tenant_id: str, client_id: str, client_secret: str):
        self.subscription_id = subscription_id
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.logger = logging.getLogger(f"{__name__}.AzureBillingAPI")
    
    async def get_cost_data(self, start_date: datetime, end_date: datetime,
                           filters: Optional[Dict[str, Any]] = None) -> List[CostData]:
        """Retrieve Azure cost data"""
        self.logger.info(f"Fetching Azure cost data from {start_date} to {end_date}")
        return []  # Mock implementation
    
    async def get_usage_data(self, start_date: datetime, end_date: datetime,
                            resource_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve Azure usage data"""
        return []
    
    def normalize_cost_data(self, raw_data: List[Dict[str, Any]]) -> List[NormalizedCostData]:
        """Normalize Azure cost data"""
        return []


class CostCollector:
    """Centralized cost data collection from multiple cloud providers"""
    
    def __init__(self):
        self.billing_apis: Dict[CloudProvider, CloudBillingAPI] = {}
        self.logger = logging.getLogger(f"{__name__}.CostCollector")
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def register_billing_api(self, provider: CloudProvider, api: CloudBillingAPI):
        """Register a billing API for a cloud provider"""
        self.billing_apis[provider] = api
        self.logger.info(f"Registered billing API for {provider.value}")
    
    async def collect_cost_data(self, start_date: datetime, end_date: datetime,
                               providers: Optional[List[CloudProvider]] = None) -> List[CostData]:
        """Collect cost data from all or specified providers"""
        if providers is None:
            providers = list(self.billing_apis.keys())
        
        all_cost_data = []
        tasks = []
        
        for provider in providers:
            if provider in self.billing_apis:
                api = self.billing_apis[provider]
                task = api.get_cost_data(start_date, end_date)
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting cost data: {result}")
                else:
                    all_cost_data.extend(result)
        
        self.logger.info(f"Collected {len(all_cost_data)} cost data points")
        return all_cost_data
    
    def normalize_cost_data(self, cost_data: List[CostData]) -> List[NormalizedCostData]:
        """Normalize cost data across providers"""
        normalized_data = []
        
        for data in cost_data:
            # Convert to USD if needed (simplified)
            cost_usd = data.cost_amount
            if data.currency != "USD":
                cost_usd = self._convert_to_usd(data.cost_amount, data.currency)
            
            normalized = NormalizedCostData(
                resource_id=data.resource_id,
                resource_type=self._extract_resource_type(data.service),
                cost_amount_usd=cost_usd,
                usage_quantity=1.0,  # Simplified
                usage_unit="unit",
                timestamp=data.billing_period_start,
                provider=data.provider,
                region=data.region or "unknown",
                project_id=data.project_id or data.tags.get("Project", "unknown"),
                team=data.team or data.tags.get("Team", "unknown"),
                tags=data.tags,
                cost_category=data.cost_category
            )
            normalized_data.append(normalized)
        
        return normalized_data
    
    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert currency to USD (simplified implementation)"""
        # In a real implementation, this would use current exchange rates
        conversion_rates = {
            "EUR": 1.1,
            "GBP": 1.3,
            "JPY": 0.009,
            "CAD": 0.8
        }
        return amount * conversion_rates.get(currency, 1.0)
    
    def _extract_resource_type(self, service_name: str) -> str:
        """Extract resource type from service name"""
        service_lower = service_name.lower()
        if "ec2" in service_lower or "compute" in service_lower:
            return "compute_instance"
        elif "s3" in service_lower or "storage" in service_lower:
            return "storage_bucket"
        elif "rds" in service_lower or "sql" in service_lower:
            return "database"
        else:
            return "other"
    
    def attribute_costs(self, normalized_data: List[NormalizedCostData],
                       attribution_rules: Dict[str, Any]) -> List[NormalizedCostData]:
        """Apply cost attribution rules to categorize costs by project, team, etc."""
        for data in normalized_data:
            # Apply attribution rules based on tags and resource properties
            if "team_mapping" in attribution_rules:
                team_mapping = attribution_rules["team_mapping"]
                if data.resource_id in team_mapping:
                    data.team = team_mapping[data.resource_id]
            
            if "project_mapping" in attribution_rules:
                project_mapping = attribution_rules["project_mapping"]
                if data.resource_id in project_mapping:
                    data.project_id = project_mapping[data.resource_id]
        
        return normalized_data

class CostAnalyzer:
    """Analyzes spending patterns and identifies optimization opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostAnalyzer")
    
    def analyze_spending_patterns(self, cost_data: List[NormalizedCostData],
                                 analysis_period_days: int = 30) -> SpendingAnalysis:
        """Comprehensive analysis of spending patterns"""
        if not cost_data:
            raise ValueError("No cost data provided for analysis")
        
        # Filter data to analysis period
        end_date = max(data.timestamp for data in cost_data)
        start_date = end_date - timedelta(days=analysis_period_days)
        filtered_data = [d for d in cost_data if start_date <= d.timestamp <= end_date]
        
        if not filtered_data:
            raise ValueError("No cost data in specified analysis period")
        
        # Calculate total spend
        total_spend = sum(data.cost_amount_usd for data in filtered_data)
        
        # Analyze spend by provider
        spend_by_provider = {}
        for provider in CloudProvider:
            provider_spend = sum(
                data.cost_amount_usd for data in filtered_data 
                if data.provider == provider
            )
            if provider_spend > 0:
                spend_by_provider[provider] = provider_spend
        
        # Analyze spend by category
        spend_by_category = {}
        for category in CostCategory:
            category_spend = sum(
                data.cost_amount_usd for data in filtered_data 
                if data.cost_category == category
            )
            if category_spend > 0:
                spend_by_category[category] = category_spend
        
        # Analyze spend by team
        spend_by_team = {}
        for data in filtered_data:
            team = data.team
            if team not in spend_by_team:
                spend_by_team[team] = 0
            spend_by_team[team] += data.cost_amount_usd
        
        # Analyze spend by project
        spend_by_project = {}
        for data in filtered_data:
            project = data.project_id
            if project not in spend_by_project:
                spend_by_project[project] = 0
            spend_by_project[project] += data.cost_amount_usd
        
        # Identify top cost drivers
        resource_costs = {}
        for data in filtered_data:
            if data.resource_id not in resource_costs:
                resource_costs[data.resource_id] = {
                    'cost': 0,
                    'resource_type': data.resource_type,
                    'provider': data.provider,
                    'team': data.team,
                    'project': data.project_id
                }
            resource_costs[data.resource_id]['cost'] += data.cost_amount_usd
        
        top_cost_drivers = sorted(
            [{'resource_id': k, **v} for k, v in resource_costs.items()],
            key=lambda x: x['cost'],
            reverse=True
        )[:10]
        
        # Calculate growth rate (simplified)
        growth_rate = self._calculate_growth_rate(filtered_data)
        
        # Detect anomalies
        anomalies = self._detect_cost_anomalies(filtered_data)
        
        # Generate optimization opportunities
        optimizations = self._identify_optimization_opportunities(filtered_data)
        
        return SpendingAnalysis(
            analysis_period_start=start_date,
            analysis_period_end=end_date,
            total_spend=total_spend,
            spend_by_provider=spend_by_provider,
            spend_by_category=spend_by_category,
            spend_by_team=spend_by_team,
            spend_by_project=spend_by_project,
            top_cost_drivers=top_cost_drivers,
            growth_rate=growth_rate,
            anomalies_detected=anomalies,
            optimization_opportunities=optimizations
        )
    
    def _calculate_growth_rate(self, cost_data: List[NormalizedCostData]) -> float:
        """Calculate spending growth rate"""
        if len(cost_data) < 2:
            return 0.0
        
        # Group by day and calculate daily totals
        daily_costs = {}
        for data in cost_data:
            day = data.timestamp.date()
            if day not in daily_costs:
                daily_costs[day] = 0
            daily_costs[day] += data.cost_amount_usd
        
        if len(daily_costs) < 2:
            return 0.0
        
        # Calculate simple linear growth rate
        days = sorted(daily_costs.keys())
        costs = [daily_costs[day] for day in days]
        
        if len(costs) >= 2:
            # Simple growth rate calculation
            first_half = costs[:len(costs)//2]
            second_half = costs[len(costs)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_first > 0:
                return ((avg_second - avg_first) / avg_first) * 100
        
        return 0.0
    
    def _detect_cost_anomalies(self, cost_data: List[NormalizedCostData]) -> List[Dict[str, Any]]:
        """Detect cost anomalies using statistical methods"""
        anomalies = []
        
        # Group by resource and detect anomalies
        resource_costs = {}
        for data in cost_data:
            if data.resource_id not in resource_costs:
                resource_costs[data.resource_id] = []
            resource_costs[data.resource_id].append(data.cost_amount_usd)
        
        for resource_id, costs in resource_costs.items():
            if len(costs) >= 5:  # Need sufficient data points
                mean_cost = statistics.mean(costs)
                std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
                
                # Detect outliers (costs > 2 standard deviations from mean)
                for cost in costs:
                    if std_cost > 0 and abs(cost - mean_cost) > 2 * std_cost:
                        anomalies.append({
                            'resource_id': resource_id,
                            'anomalous_cost': cost,
                            'expected_cost': mean_cost,
                            'deviation': abs(cost - mean_cost),
                            'severity': 'high' if abs(cost - mean_cost) > 3 * std_cost else 'medium'
                        })
        
        return anomalies
    
    def _identify_optimization_opportunities(self, cost_data: List[NormalizedCostData]) -> List[CostOptimization]:
        """Identify cost optimization opportunities"""
        optimizations = []
        
        # Group by resource type for analysis
        resource_type_costs = {}
        for data in cost_data:
            if data.resource_type not in resource_type_costs:
                resource_type_costs[data.resource_type] = []
            resource_type_costs[data.resource_type].append(data)
        
        # Analyze compute instances for right-sizing opportunities
        if 'compute_instance' in resource_type_costs:
            compute_data = resource_type_costs['compute_instance']
            compute_optimizations = self._analyze_compute_optimization(compute_data)
            optimizations.extend(compute_optimizations)
        
        # Analyze storage for optimization opportunities
        if 'storage_bucket' in resource_type_costs:
            storage_data = resource_type_costs['storage_bucket']
            storage_optimizations = self._analyze_storage_optimization(storage_data)
            optimizations.extend(storage_optimizations)
        
        return optimizations
    
    def _analyze_compute_optimization(self, compute_data: List[NormalizedCostData]) -> List[CostOptimization]:
        """Analyze compute instances for optimization opportunities"""
        optimizations = []
        
        # Group by resource for analysis
        resource_costs = {}
        for data in compute_data:
            if data.resource_id not in resource_costs:
                resource_costs[data.resource_id] = []
            resource_costs[data.resource_id].append(data.cost_amount_usd)
        
        for resource_id, costs in resource_costs.items():
            if len(costs) >= 7:  # At least a week of data
                avg_cost = statistics.mean(costs)
                
                # Simple heuristic: if consistently high cost, suggest reserved instances
                if avg_cost > 50:  # Threshold for reserved instance recommendation
                    potential_savings = avg_cost * 0.3  # Assume 30% savings with reserved instances
                    
                    optimizations.append(CostOptimization(
                        optimization_id=f"ri_{resource_id}_{datetime.now().timestamp()}",
                        optimization_type=OptimizationType.RESERVED_INSTANCES,
                        resource_id=resource_id,
                        current_cost=avg_cost * 24 * 30,  # Monthly cost
                        optimized_cost=avg_cost * 24 * 30 * 0.7,  # 30% savings
                        potential_savings=potential_savings * 24 * 30,  # Monthly savings
                        confidence_score=0.8,
                        implementation_effort=EffortLevel.LOW,
                        recommendation=f"Consider purchasing reserved instances for {resource_id} to save approximately 30% on compute costs"
                    ))
        
        return optimizations
    
    def _analyze_storage_optimization(self, storage_data: List[NormalizedCostData]) -> List[CostOptimization]:
        """Analyze storage for optimization opportunities"""
        optimizations = []
        
        # Simple storage optimization: suggest lifecycle policies for high-cost storage
        resource_costs = {}
        for data in storage_data:
            if data.resource_id not in resource_costs:
                resource_costs[data.resource_id] = []
            resource_costs[data.resource_id].append(data.cost_amount_usd)
        
        for resource_id, costs in resource_costs.items():
            if costs:
                avg_cost = statistics.mean(costs)
                
                if avg_cost > 20:  # Threshold for storage optimization
                    potential_savings = avg_cost * 0.4  # Assume 40% savings with lifecycle policies
                    
                    optimizations.append(CostOptimization(
                        optimization_id=f"storage_{resource_id}_{datetime.now().timestamp()}",
                        optimization_type=OptimizationType.STORAGE_OPTIMIZATION,
                        resource_id=resource_id,
                        current_cost=avg_cost * 30,  # Monthly cost
                        optimized_cost=avg_cost * 30 * 0.6,  # 40% savings
                        potential_savings=potential_savings * 30,  # Monthly savings
                        confidence_score=0.7,
                        implementation_effort=EffortLevel.MEDIUM,
                        recommendation=f"Implement lifecycle policies for {resource_id} to automatically transition data to cheaper storage tiers"
                    ))
        
        return optimizations


class BudgetManager:
    """Manages budgets, alerts, and spending controls"""
    
    def __init__(self):
        self.budgets: Dict[str, Budget] = {}
        self.logger = logging.getLogger(f"{__name__}.BudgetManager")
        self.alert_callbacks: List[callable] = []
    
    def create_budget(self, budget: Budget) -> str:
        """Create a new budget"""
        if budget.budget_id in self.budgets:
            raise ValueError(f"Budget with ID {budget.budget_id} already exists")
        
        self.budgets[budget.budget_id] = budget
        self.logger.info(f"Created budget: {budget.name} (${budget.amount})")
        return budget.budget_id
    
    def update_budget(self, budget_id: str, updates: Dict[str, Any]) -> Budget:
        """Update an existing budget"""
        if budget_id not in self.budgets:
            raise ValueError(f"Budget {budget_id} not found")
        
        budget = self.budgets[budget_id]
        for key, value in updates.items():
            if hasattr(budget, key):
                setattr(budget, key, value)
        
        self.logger.info(f"Updated budget: {budget_id}")
        return budget
    
    def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget"""
        if budget_id in self.budgets:
            del self.budgets[budget_id]
            self.logger.info(f"Deleted budget: {budget_id}")
            return True
        return False
    
    def get_budget_status(self, budget_id: str, cost_data: List[NormalizedCostData]) -> BudgetStatus:
        """Get current status of a budget"""
        if budget_id not in self.budgets:
            raise ValueError(f"Budget {budget_id} not found")
        
        budget = self.budgets[budget_id]
        
        # Filter cost data based on budget scope and period
        filtered_costs = self._filter_costs_for_budget(budget, cost_data)
        current_spend = sum(data.cost_amount_usd for data in filtered_costs)
        
        # Calculate utilization percentage
        utilization_percentage = (current_spend / budget.amount) * 100 if budget.amount > 0 else 0
        
        # Calculate projected spend
        projected_spend = self._calculate_projected_spend(budget, filtered_costs)
        
        # Calculate days remaining in budget period
        days_remaining = self._calculate_days_remaining(budget)
        
        # Check if on track
        expected_spend_rate = budget.amount / self._get_budget_period_days(budget)
        current_daily_rate = current_spend / max(1, (datetime.now() - budget.start_date).days)
        is_on_track = current_daily_rate <= expected_spend_rate * 1.1  # 10% tolerance
        
        # Check for triggered alerts
        triggered_alerts = []
        for threshold in budget.alert_thresholds:
            if utilization_percentage >= threshold:
                alert_msg = f"Budget {budget.name} has reached {threshold}% utilization"
                triggered_alerts.append(alert_msg)
                self._trigger_alert(budget, threshold, current_spend, utilization_percentage)
        
        return BudgetStatus(
            budget=budget,
            current_spend=current_spend,
            utilization_percentage=utilization_percentage,
            projected_spend=projected_spend,
            days_remaining=days_remaining,
            is_on_track=is_on_track,
            triggered_alerts=triggered_alerts
        )
    
    def _filter_costs_for_budget(self, budget: Budget, cost_data: List[NormalizedCostData]) -> List[NormalizedCostData]:
        """Filter cost data based on budget scope and time period"""
        filtered = []
        
        for data in cost_data:
            # Check time period
            if data.timestamp < budget.start_date:
                continue
            if budget.end_date and data.timestamp > budget.end_date:
                continue
            
            # Check scope filters
            matches_scope = True
            for filter_type, filter_values in budget.scope_filters.items():
                if filter_type == "teams" and data.team not in filter_values:
                    matches_scope = False
                    break
                elif filter_type == "projects" and data.project_id not in filter_values:
                    matches_scope = False
                    break
                elif filter_type == "providers" and data.provider.value not in filter_values:
                    matches_scope = False
                    break
            
            if matches_scope:
                filtered.append(data)
        
        return filtered
    
    def _calculate_projected_spend(self, budget: Budget, filtered_costs: List[NormalizedCostData]) -> float:
        """Calculate projected spend for the budget period"""
        if not filtered_costs:
            return 0.0
        
        # Simple projection based on current daily rate
        days_elapsed = max(1, (datetime.now() - budget.start_date).days)
        total_spend = sum(data.cost_amount_usd for data in filtered_costs)
        daily_rate = total_spend / days_elapsed
        
        total_days = self._get_budget_period_days(budget)
        return daily_rate * total_days
    
    def _calculate_days_remaining(self, budget: Budget) -> int:
        """Calculate days remaining in budget period"""
        if budget.end_date:
            return max(0, (budget.end_date - datetime.now()).days)
        else:
            # Calculate based on budget period
            period_days = self._get_budget_period_days(budget)
            elapsed_days = (datetime.now() - budget.start_date).days
            return max(0, period_days - elapsed_days)
    
    def _get_budget_period_days(self, budget: Budget) -> int:
        """Get the number of days in the budget period"""
        if budget.end_date:
            return (budget.end_date - budget.start_date).days
        
        # Default period lengths
        period_days = {
            BudgetPeriod.DAILY: 1,
            BudgetPeriod.WEEKLY: 7,
            BudgetPeriod.MONTHLY: 30,
            BudgetPeriod.QUARTERLY: 90,
            BudgetPeriod.YEARLY: 365
        }
        return period_days.get(budget.period, 30)
    
    def _trigger_alert(self, budget: Budget, threshold: float, current_spend: float, utilization: float):
        """Trigger budget alert"""
        alert_data = {
            'budget_id': budget.budget_id,
            'budget_name': budget.name,
            'threshold': threshold,
            'current_spend': current_spend,
            'utilization_percentage': utilization,
            'timestamp': datetime.now()
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: callable):
        """Register a callback function for budget alerts"""
        self.alert_callbacks.append(callback)
    
    def get_all_budget_statuses(self, cost_data: List[NormalizedCostData]) -> Dict[str, BudgetStatus]:
        """Get status for all active budgets"""
        statuses = {}
        for budget_id, budget in self.budgets.items():
            if budget.is_active:
                statuses[budget_id] = self.get_budget_status(budget_id, cost_data)
        return statuses


class CostPredictor:
    """ML-based cost forecasting and prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostPredictor")
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
    
    def train_cost_prediction_model(self, cost_data: List[NormalizedCostData], 
                                   resource_id: Optional[str] = None) -> Dict[str, float]:
        """Train cost prediction model on historical data"""
        if not cost_data:
            raise ValueError("No cost data provided for training")
        
        if not SKLEARN_AVAILABLE:
            return self._train_simple_model(cost_data, resource_id)
        
        # Prepare training data
        df = self._prepare_training_data(cost_data, resource_id)
        
        if len(df) < 10:  # Need minimum data points
            raise ValueError("Insufficient data for training (minimum 10 points required)")
        
        # Create features and target
        X, y = self._create_features_and_target(df)
        
        # Scale features
        model_key = resource_id or "global"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_key] = scaler
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        self.models[model_key] = model
        
        # Calculate model metrics
        predictions = model.predict(X_scaled)
        mse = np.mean((y - predictions) ** 2)
        r2_score = model.score(X_scaled, y)
        
        self.logger.info(f"Trained cost prediction model for {model_key}: R² = {r2_score:.3f}")
        
        return {
            'mse': mse,
            'r2_score': r2_score,
            'training_samples': len(df)
        }
    
    def _train_simple_model(self, cost_data: List[NormalizedCostData], 
                           resource_id: Optional[str] = None) -> Dict[str, float]:
        """Simple model training without sklearn"""
        if resource_id:
            filtered_data = [d for d in cost_data if d.resource_id == resource_id]
        else:
            filtered_data = cost_data
        
        if len(filtered_data) < 10:
            raise ValueError("Insufficient data for training (minimum 10 points required)")
        
        # Simple moving average model
        costs = [d.cost_amount_usd for d in filtered_data]
        model_key = resource_id or "global"
        
        # Store simple statistics as "model"
        self.models[model_key] = {
            'type': 'simple_average',
            'mean': sum(costs) / len(costs),
            'trend': (costs[-1] - costs[0]) / len(costs) if len(costs) > 1 else 0,
            'data_points': len(costs)
        }
        
        # Calculate simple metrics
        mean_cost = sum(costs) / len(costs)
        mse = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
        
        return {
            'mse': mse,
            'r2_score': 0.5,  # Placeholder
            'training_samples': len(filtered_data)
        }
    
    def predict_costs(self, forecast_days: int, resource_id: Optional[str] = None,
                     scenario_factors: Optional[Dict[str, float]] = None) -> CostForecast:
        """Predict future costs using trained models"""
        model_key = resource_id or "global"
        
        if model_key not in self.models:
            raise ValueError(f"No trained model found for {model_key}")
        
        model = self.models[model_key]
        
        # Generate future dates
        start_date = datetime.now()
        end_date = start_date + timedelta(days=forecast_days)
        
        if not SKLEARN_AVAILABLE or isinstance(model, dict):
            return self._predict_simple(model, forecast_days, start_date, end_date, scenario_factors)
        
        scaler = self.scalers[model_key]
        
        # Create feature matrix for prediction
        future_features = self._create_future_features(forecast_days, scenario_factors)
        future_features_scaled = scaler.transform(future_features)
        
        # Make predictions
        predictions = model.predict(future_features_scaled)
        
        # Calculate confidence intervals (simplified)
        prediction_std = np.std(predictions)
        confidence_intervals = [
            (pred - 1.96 * prediction_std, pred + 1.96 * prediction_std)
            for pred in predictions
        ]
        
        # Determine trend direction
        if len(predictions) >= 2:
            trend_slope = (predictions[-1] - predictions[0]) / len(predictions)
            if trend_slope > 0.1:
                trend_direction = "increasing"
            elif trend_slope < -0.1:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        # Calculate model accuracy (simplified)
        model_accuracy = 0.8
        
        return CostForecast(
            forecast_period_start=start_date,
            forecast_period_end=end_date,
            predicted_costs=predictions.tolist(),
            confidence_intervals=confidence_intervals,
            model_accuracy=model_accuracy,
            trend_direction=trend_direction,
            seasonal_patterns=self._detect_seasonal_patterns(predictions.tolist())
        )
    
    def _predict_simple(self, model: Dict[str, Any], forecast_days: int, 
                       start_date: datetime, end_date: datetime,
                       scenario_factors: Optional[Dict[str, float]] = None) -> CostForecast:
        """Simple prediction without sklearn"""
        base_cost = model['mean']
        trend = model['trend']
        
        # Generate predictions
        predictions = []
        for i in range(forecast_days):
            predicted_cost = base_cost + (trend * i)
            
            # Apply scenario factors
            if scenario_factors and 'usage_multiplier' in scenario_factors:
                predicted_cost *= scenario_factors['usage_multiplier']
            
            predictions.append(max(0, predicted_cost))  # Ensure non-negative
        
        # Simple confidence intervals (±20%)
        confidence_intervals = [
            (pred * 0.8, pred * 1.2) for pred in predictions
        ]
        
        # Determine trend
        if trend > 0.1:
            trend_direction = "increasing"
        elif trend < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        return CostForecast(
            forecast_period_start=start_date,
            forecast_period_end=end_date,
            predicted_costs=predictions,
            confidence_intervals=confidence_intervals,
            model_accuracy=0.7,  # Lower accuracy for simple model
            trend_direction=trend_direction,
            seasonal_patterns=self._detect_seasonal_patterns(predictions)
        )
    
    def _prepare_training_data(self, cost_data: List[NormalizedCostData], 
                              resource_id: Optional[str] = None):
        """Prepare cost data for training"""
        # Filter by resource if specified
        if resource_id:
            filtered_data = [d for d in cost_data if d.resource_id == resource_id]
        else:
            filtered_data = cost_data
        
        if not SKLEARN_AVAILABLE:
            return filtered_data
        
        # Convert to DataFrame
        data_dicts = []
        for data in filtered_data:
            data_dicts.append({
                'timestamp': data.timestamp,
                'cost': data.cost_amount_usd,
                'day_of_week': data.timestamp.weekday(),
                'day_of_month': data.timestamp.day,
                'month': data.timestamp.month,
                'hour': data.timestamp.hour,
                'provider': data.provider.value,
                'resource_type': data.resource_type
            })
        
        df = pd.DataFrame(data_dicts)
        df = df.sort_values('timestamp')
        return df
    
    def _create_features_and_target(self, df):
        """Create feature matrix and target vector"""
        if not SKLEARN_AVAILABLE:
            return None, None
        
        # Time-based features
        features = []
        for _, row in df.iterrows():
            feature_vector = [
                row['day_of_week'],
                row['day_of_month'],
                row['month'],
                row['hour'],
                # Add trend feature (days since start)
                (row['timestamp'] - df['timestamp'].min()).days
            ]
            features.append(feature_vector)
        
        X = np.array(features)
        y = df['cost'].values
        
        return X, y
    
    def _create_future_features(self, forecast_days: int, 
                               scenario_factors: Optional[Dict[str, float]] = None):
        """Create feature matrix for future predictions"""
        if not SKLEARN_AVAILABLE:
            return None
        
        features = []
        base_date = datetime.now()
        
        for i in range(forecast_days):
            future_date = base_date + timedelta(days=i)
            feature_vector = [
                future_date.weekday(),
                future_date.day,
                future_date.month,
                future_date.hour,
                i  # Days from now
            ]
            
            # Apply scenario factors if provided
            if scenario_factors:
                for factor_name, factor_value in scenario_factors.items():
                    if factor_name == "usage_multiplier":
                        # Multiply all features by usage factor
                        feature_vector = [f * factor_value for f in feature_vector]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _detect_seasonal_patterns(self, predictions: List[float]) -> Dict[str, float]:
        """Detect seasonal patterns in predictions"""
        if len(predictions) < 7:
            return {}
        
        # Simple seasonal pattern detection
        patterns = {}
        
        # Weekly pattern (if we have at least a week of data)
        if len(predictions) >= 7:
            weekly_avg = np.mean(predictions[:7])
            patterns['weekly_average'] = weekly_avg
        
        # Monthly pattern (if we have at least a month of data)
        if len(predictions) >= 30:
            monthly_avg = np.mean(predictions[:30])
            patterns['monthly_average'] = monthly_avg
        
        return patterns


class FinOpsEngine:
    """Main FinOps intelligence engine coordinating all components"""
    
    def __init__(self):
        self.cost_collector = CostCollector()
        self.cost_analyzer = CostAnalyzer()
        self.budget_manager = BudgetManager()
        self.cost_predictor = CostPredictor()
        self.logger = logging.getLogger(f"{__name__}.FinOpsEngine")
        
        # Register default alert handler
        self.budget_manager.register_alert_callback(self._default_alert_handler)
    
    async def initialize_providers(self, provider_configs: Dict[CloudProvider, Dict[str, Any]]):
        """Initialize cloud provider billing APIs"""
        for provider, config in provider_configs.items():
            if provider == CloudProvider.AWS:
                api = AWSBillingAPI(
                    access_key=config['access_key'],
                    secret_key=config['secret_key'],
                    region=config.get('region', 'us-east-1')
                )
            elif provider == CloudProvider.GCP:
                api = GCPBillingAPI(
                    project_id=config['project_id'],
                    credentials_path=config['credentials_path']
                )
            elif provider == CloudProvider.AZURE:
                api = AzureBillingAPI(
                    subscription_id=config['subscription_id'],
                    tenant_id=config['tenant_id'],
                    client_id=config['client_id'],
                    client_secret=config['client_secret']
                )
            else:
                continue
            
            self.cost_collector.register_billing_api(provider, api)
    
    async def run_comprehensive_cost_analysis(self, analysis_days: int = 30) -> Dict[str, Any]:
        """Run comprehensive cost analysis across all providers"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_days)
        
        # Collect cost data
        self.logger.info("Collecting cost data from all providers...")
        raw_cost_data = await self.cost_collector.collect_cost_data(start_date, end_date)
        
        # Normalize cost data
        self.logger.info("Normalizing cost data...")
        normalized_data = self.cost_collector.normalize_cost_data(raw_cost_data)
        
        # Analyze spending patterns
        self.logger.info("Analyzing spending patterns...")
        spending_analysis = self.cost_analyzer.analyze_spending_patterns(normalized_data, analysis_days)
        
        # Get budget statuses
        self.logger.info("Checking budget statuses...")
        budget_statuses = self.budget_manager.get_all_budget_statuses(normalized_data)
        
        # Generate cost forecasts
        self.logger.info("Generating cost forecasts...")
        try:
            # Train prediction model if we have enough data
            if len(normalized_data) >= 10:
                training_metrics = self.cost_predictor.train_cost_prediction_model(normalized_data)
                cost_forecast = self.cost_predictor.predict_costs(30)  # 30-day forecast
            else:
                training_metrics = {}
                cost_forecast = None
        except Exception as e:
            self.logger.warning(f"Could not generate cost forecast: {e}")
            training_metrics = {}
            cost_forecast = None
        
        return {
            'spending_analysis': spending_analysis,
            'budget_statuses': budget_statuses,
            'cost_forecast': cost_forecast,
            'training_metrics': training_metrics,
            'data_points_analyzed': len(normalized_data),
            'analysis_period': {
                'start': start_date,
                'end': end_date,
                'days': analysis_days
            }
        }
    
    def _default_alert_handler(self, alert_data: Dict[str, Any]):
        """Default handler for budget alerts"""
        self.logger.warning(
            f"BUDGET ALERT: {alert_data['budget_name']} has reached "
            f"{alert_data['threshold']}% utilization "
            f"(${alert_data['current_spend']:.2f})"
        )
    
    def create_budget(self, name: str, amount: float, period: BudgetPeriod,
                     scope_filters: Optional[Dict[str, List[str]]] = None,
                     alert_thresholds: Optional[List[float]] = None) -> str:
        """Create a new budget with the specified parameters"""
        import time
        import random
        
        # Generate unique budget ID
        budget_id = f"budget_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        budget = Budget(
            budget_id=budget_id,
            name=name,
            amount=amount,
            currency="USD",
            period=period,
            start_date=datetime.now(),
            scope_filters=scope_filters or {},
            alert_thresholds=alert_thresholds or [50.0, 80.0, 100.0]
        )
        
        return self.budget_manager.create_budget(budget)
    
    def get_optimization_recommendations(self, cost_data: List[NormalizedCostData]) -> List[CostOptimization]:
        """Get cost optimization recommendations"""
        spending_analysis = self.cost_analyzer.analyze_spending_patterns(cost_data)
        return spending_analysis.optimization_opportunities