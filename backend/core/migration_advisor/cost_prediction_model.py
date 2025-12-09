"""
Cost Prediction Model for Cloud Migration Advisor

This module implements the CostPredictionModel that estimates workload costs
across different cloud providers and scores them based on budget constraints.

Requirements: 3.4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from decimal import Decimal
from enum import Enum

from .provider_catalog import (
    CloudProvider, CloudProviderName, ServiceCategory,
    PricingModel, ProviderCatalog
)
from .service_catalog_data import get_provider_catalog


class CostCategory(Enum):
    """Cost categories for estimation"""
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORKING = "networking"
    DATA_TRANSFER = "data_transfer"
    SUPPORT = "support"
    OTHER = "other"


@dataclass
class WorkloadCostEstimate:
    """Cost estimate for a specific workload"""
    workload_name: str
    provider: CloudProviderName
    monthly_cost: Decimal
    annual_cost: Decimal
    cost_breakdown: Dict[CostCategory, Decimal] = field(default_factory=dict)
    pricing_model_used: PricingModel = PricingModel.ON_DEMAND
    assumptions: List[str] = field(default_factory=list)
    confidence_level: float = 0.8  # 0.0 to 1.0
    
    def __repr__(self):
        return f"<WorkloadCostEstimate(workload='{self.workload_name}', monthly=${self.monthly_cost})>"


@dataclass
class ProviderCostComparison:
    """Cost comparison across providers"""
    provider: CloudProviderName
    total_monthly_cost: Decimal
    total_annual_cost: Decimal
    workload_costs: List[WorkloadCostEstimate] = field(default_factory=list)
    cost_breakdown: Dict[CostCategory, Decimal] = field(default_factory=dict)
    potential_savings: Optional[Decimal] = None
    cost_optimization_opportunities: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<ProviderCostComparison(provider='{self.provider.value}', monthly=${self.total_monthly_cost})>"


@dataclass
class CostScore:
    """Cost scoring result"""
    provider: CloudProviderName
    cost_score: float  # 0.0 to 1.0 (higher is better)
    estimated_monthly_cost: Decimal
    budget_fit_score: float  # How well it fits budget
    value_score: float  # Cost vs features value
    cost_efficiency_rating: str  # excellent, good, fair, poor
    cost_comparison: str  # Description of cost position
    
    def __repr__(self):
        return f"<CostScore(provider='{self.provider.value}', score={self.cost_score}, rating='{self.cost_efficiency_rating}')>"


class CostPredictionModel:
    """
    ML-based cost prediction model that estimates workload costs across
    cloud providers and scores them based on budget constraints.
    """
    
    def __init__(self, provider_catalog: Optional[ProviderCatalog] = None):
        """
        Initialize the cost prediction model
        
        Args:
            provider_catalog: Optional provider catalog
        """
        self.catalog = provider_catalog or get_provider_catalog()
        
        # Base pricing estimates (simplified for MVP)
        self.base_pricing = self._initialize_base_pricing()
        
        # Regional cost multipliers
        self.regional_multipliers = {
            "us-east": 1.0,
            "us-west": 1.05,
            "eu-west": 1.1,
            "eu-central": 1.08,
            "ap-southeast": 1.15,
            "ap-northeast": 1.12,
        }
    
    def _initialize_base_pricing(self) -> Dict[CloudProviderName, Dict[CostCategory, Decimal]]:
        """
        Initialize base pricing estimates per provider
        
        Returns:
            Dictionary of base pricing by provider and category
        """
        return {
            CloudProviderName.AWS: {
                CostCategory.COMPUTE: Decimal("0.10"),  # per vCPU hour
                CostCategory.STORAGE: Decimal("0.023"),  # per GB/month
                CostCategory.DATABASE: Decimal("0.15"),  # per vCPU hour
                CostCategory.NETWORKING: Decimal("0.09"),  # per GB transfer
                CostCategory.DATA_TRANSFER: Decimal("0.09"),  # per GB
                CostCategory.SUPPORT: Decimal("100.00"),  # base monthly
            },
            CloudProviderName.GCP: {
                CostCategory.COMPUTE: Decimal("0.095"),  # per vCPU hour
                CostCategory.STORAGE: Decimal("0.020"),  # per GB/month
                CostCategory.DATABASE: Decimal("0.14"),  # per vCPU hour
                CostCategory.NETWORKING: Decimal("0.08"),  # per GB transfer
                CostCategory.DATA_TRANSFER: Decimal("0.08"),  # per GB
                CostCategory.SUPPORT: Decimal("150.00"),  # base monthly
            },
            CloudProviderName.AZURE: {
                CostCategory.COMPUTE: Decimal("0.105"),  # per vCPU hour
                CostCategory.STORAGE: Decimal("0.024"),  # per GB/month
                CostCategory.DATABASE: Decimal("0.16"),  # per vCPU hour
                CostCategory.NETWORKING: Decimal("0.087"),  # per GB transfer
                CostCategory.DATA_TRANSFER: Decimal("0.087"),  # per GB
                CostCategory.SUPPORT: Decimal("100.00"),  # base monthly
            }
        }
    
    def estimate_workload_cost(
        self,
        provider_name: CloudProviderName,
        compute_cores: int,
        memory_gb: int,
        storage_tb: float,
        data_transfer_tb: float = 1.0,
        database_cores: int = 0,
        region: str = "us-east",
        workload_name: str = "default"
    ) -> WorkloadCostEstimate:
        """
        Estimate cost for a specific workload on a provider
        
        Args:
            provider_name: Cloud provider
            compute_cores: Number of compute vCPUs
            memory_gb: Memory in GB
            storage_tb: Storage in TB
            data_transfer_tb: Data transfer in TB per month
            database_cores: Database vCPUs
            region: Target region
            workload_name: Name of the workload
            
        Returns:
            WorkloadCostEstimate with detailed cost breakdown
        """
        base_prices = self.base_pricing.get(provider_name, {})
        regional_multiplier = self.regional_multipliers.get(region, 1.0)
        
        # Calculate costs by category
        cost_breakdown = {}
        
        # Compute costs (730 hours per month)
        compute_cost = (
            Decimal(compute_cores) * 
            base_prices.get(CostCategory.COMPUTE, Decimal("0.10")) * 
            Decimal("730") * 
            Decimal(str(regional_multiplier))
        )
        cost_breakdown[CostCategory.COMPUTE] = compute_cost
        
        # Storage costs
        storage_cost = (
            Decimal(str(storage_tb * 1024)) *  # Convert TB to GB
            base_prices.get(CostCategory.STORAGE, Decimal("0.023")) *
            Decimal(str(regional_multiplier))
        )
        cost_breakdown[CostCategory.STORAGE] = storage_cost
        
        # Database costs
        if database_cores > 0:
            database_cost = (
                Decimal(database_cores) *
                base_prices.get(CostCategory.DATABASE, Decimal("0.15")) *
                Decimal("730") *
                Decimal(str(regional_multiplier))
            )
            cost_breakdown[CostCategory.DATABASE] = database_cost
        
        # Data transfer costs
        data_transfer_cost = (
            Decimal(str(data_transfer_tb * 1024)) *  # Convert TB to GB
            base_prices.get(CostCategory.DATA_TRANSFER, Decimal("0.09"))
        )
        cost_breakdown[CostCategory.DATA_TRANSFER] = data_transfer_cost
        
        # Support costs
        support_cost = base_prices.get(CostCategory.SUPPORT, Decimal("100.00"))
        cost_breakdown[CostCategory.SUPPORT] = support_cost
        
        # Calculate totals
        monthly_cost = sum(cost_breakdown.values())
        annual_cost = monthly_cost * Decimal("12")
        
        # Add assumptions
        assumptions = [
            f"Using on-demand pricing in {region} region",
            f"730 hours per month compute utilization",
            "Standard support tier included",
            "Estimates based on typical workload patterns"
        ]
        
        return WorkloadCostEstimate(
            workload_name=workload_name,
            provider=provider_name,
            monthly_cost=monthly_cost,
            annual_cost=annual_cost,
            cost_breakdown=cost_breakdown,
            pricing_model_used=PricingModel.ON_DEMAND,
            assumptions=assumptions,
            confidence_level=0.75
        )
    
    def compare_provider_costs(
        self,
        workload_specs: List[Dict],
        providers: Optional[List[CloudProviderName]] = None
    ) -> Dict[CloudProviderName, ProviderCostComparison]:
        """
        Compare costs across multiple providers for given workloads
        
        Args:
            workload_specs: List of workload specifications
            providers: Optional list of providers to compare
            
        Returns:
            Dictionary mapping providers to cost comparisons
        """
        if providers is None:
            providers = [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]
        
        comparisons = {}
        
        for provider_name in providers:
            workload_costs = []
            total_monthly = Decimal("0")
            category_totals = {}
            
            for spec in workload_specs:
                estimate = self.estimate_workload_cost(
                    provider_name=provider_name,
                    compute_cores=spec.get("compute_cores", 0),
                    memory_gb=spec.get("memory_gb", 0),
                    storage_tb=spec.get("storage_tb", 0),
                    data_transfer_tb=spec.get("data_transfer_tb", 1.0),
                    database_cores=spec.get("database_cores", 0),
                    region=spec.get("region", "us-east"),
                    workload_name=spec.get("name", "workload")
                )
                
                workload_costs.append(estimate)
                total_monthly += estimate.monthly_cost
                
                # Aggregate by category
                for category, cost in estimate.cost_breakdown.items():
                    category_totals[category] = category_totals.get(category, Decimal("0")) + cost
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_cost_optimizations(
                provider_name, workload_costs, category_totals
            )
            
            comparisons[provider_name] = ProviderCostComparison(
                provider=provider_name,
                total_monthly_cost=total_monthly,
                total_annual_cost=total_monthly * Decimal("12"),
                workload_costs=workload_costs,
                cost_breakdown=category_totals,
                cost_optimization_opportunities=optimization_opportunities
            )
        
        # Calculate potential savings
        if comparisons:
            min_cost = min(c.total_monthly_cost for c in comparisons.values())
            for comparison in comparisons.values():
                comparison.potential_savings = comparison.total_monthly_cost - min_cost
        
        return comparisons
    
    def _identify_cost_optimizations(
        self,
        provider_name: CloudProviderName,
        workload_costs: List[WorkloadCostEstimate],
        category_totals: Dict[CostCategory, Decimal]
    ) -> List[str]:
        """
        Identify cost optimization opportunities
        
        Args:
            provider_name: Cloud provider
            workload_costs: List of workload cost estimates
            category_totals: Total costs by category
            
        Returns:
            List of optimization recommendations
        """
        opportunities = []
        
        # Check for reserved instance opportunities
        total_compute = category_totals.get(CostCategory.COMPUTE, Decimal("0"))
        if total_compute > Decimal("1000"):
            savings = total_compute * Decimal("0.3")  # 30% typical RI savings
            opportunities.append(
                f"Reserved instances could save ~${savings:.2f}/month on compute"
            )
        
        # Check for storage optimization
        total_storage = category_totals.get(CostCategory.STORAGE, Decimal("0"))
        if total_storage > Decimal("500"):
            opportunities.append(
                "Consider tiered storage for infrequently accessed data"
            )
        
        # Check for data transfer optimization
        total_transfer = category_totals.get(CostCategory.DATA_TRANSFER, Decimal("0"))
        if total_transfer > Decimal("200"):
            opportunities.append(
                "Optimize data transfer with CDN or regional caching"
            )
        
        # Provider-specific optimizations
        if provider_name == CloudProviderName.AWS:
            opportunities.append("Consider AWS Savings Plans for flexible discounts")
        elif provider_name == CloudProviderName.GCP:
            opportunities.append("GCP offers sustained use discounts automatically")
        elif provider_name == CloudProviderName.AZURE:
            opportunities.append("Azure Hybrid Benefit available for existing licenses")
        
        return opportunities
    
    def calculate_cost_score(
        self,
        provider_name: CloudProviderName,
        estimated_monthly_cost: Decimal,
        target_monthly_budget: Decimal,
        current_monthly_cost: Optional[Decimal] = None,
        service_quality_score: float = 0.8
    ) -> CostScore:
        """
        Calculate cost scoring based on budget constraints
        
        Args:
            provider_name: Cloud provider
            estimated_monthly_cost: Estimated monthly cost
            target_monthly_budget: Target budget
            current_monthly_cost: Current infrastructure cost
            service_quality_score: Quality score of services (0-1)
            
        Returns:
            CostScore with detailed scoring
        """
        # Calculate budget fit score (how well it fits the budget)
        if target_monthly_budget > 0:
            cost_ratio = float(estimated_monthly_cost / target_monthly_budget)
            if cost_ratio <= 0.8:
                budget_fit_score = 1.0
            elif cost_ratio <= 1.0:
                budget_fit_score = 0.9
            elif cost_ratio <= 1.2:
                budget_fit_score = 0.7
            elif cost_ratio <= 1.5:
                budget_fit_score = 0.5
            else:
                budget_fit_score = 0.3
        else:
            budget_fit_score = 0.5
        
        # Calculate value score (cost vs service quality)
        # Lower cost with high quality = better value
        normalized_cost = min(float(estimated_monthly_cost / Decimal("10000")), 1.0)
        value_score = (service_quality_score * 0.6) + ((1.0 - normalized_cost) * 0.4)
        
        # Calculate overall cost score
        cost_score = (budget_fit_score * 0.6) + (value_score * 0.4)
        
        # Determine cost efficiency rating
        if cost_score >= 0.85:
            efficiency_rating = "excellent"
        elif cost_score >= 0.7:
            efficiency_rating = "good"
        elif cost_score >= 0.5:
            efficiency_rating = "fair"
        else:
            efficiency_rating = "poor"
        
        # Generate cost comparison description
        if current_monthly_cost:
            savings = current_monthly_cost - estimated_monthly_cost
            if savings > 0:
                comparison = f"${savings:.2f}/month savings vs current infrastructure"
            else:
                comparison = f"${abs(savings):.2f}/month increase vs current infrastructure"
        else:
            comparison = f"Estimated at ${estimated_monthly_cost:.2f}/month"
        
        return CostScore(
            provider=provider_name,
            cost_score=cost_score,
            estimated_monthly_cost=estimated_monthly_cost,
            budget_fit_score=budget_fit_score,
            value_score=value_score,
            cost_efficiency_rating=efficiency_rating,
            cost_comparison=comparison
        )
    
    def score_providers_by_cost(
        self,
        cost_comparisons: Dict[CloudProviderName, ProviderCostComparison],
        target_monthly_budget: Decimal,
        current_monthly_cost: Optional[Decimal] = None,
        service_quality_scores: Optional[Dict[CloudProviderName, float]] = None
    ) -> Dict[CloudProviderName, CostScore]:
        """
        Score all providers based on cost efficiency
        
        Args:
            cost_comparisons: Cost comparisons for each provider
            target_monthly_budget: Target monthly budget
            current_monthly_cost: Current infrastructure cost
            service_quality_scores: Optional service quality scores per provider
            
        Returns:
            Dictionary mapping providers to cost scores
        """
        cost_scores = {}
        
        for provider_name, comparison in cost_comparisons.items():
            service_quality = 0.8  # Default
            if service_quality_scores and provider_name in service_quality_scores:
                service_quality = service_quality_scores[provider_name]
            
            score = self.calculate_cost_score(
                provider_name=provider_name,
                estimated_monthly_cost=comparison.total_monthly_cost,
                target_monthly_budget=target_monthly_budget,
                current_monthly_cost=current_monthly_cost,
                service_quality_score=service_quality
            )
            
            cost_scores[provider_name] = score
        
        return cost_scores
