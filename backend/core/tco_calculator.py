"""
Total Cost of Ownership (TCO) Calculator

Calculates comprehensive TCO including infrastructure, operational, support,
compliance, and hidden costs across multiple cloud providers and time horizons.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .multi_cloud_models import (
    WorkloadSpec, TCOAnalysis, TCOBreakdown, CloudProvider,
    CostRecommendation
)
from .cost_factors import CostFactors
from .pricing.base_pricing_client import BasePricingClient
from .pricing import AWSPricingClient, GCPPricingClient, AzurePricingClient

logger = logging.getLogger(__name__)


class TCOCalculator:
    """
    Comprehensive Total Cost of Ownership calculator for multi-cloud workloads.
    
    Calculates TCO including:
    - Base infrastructure costs
    - Data transfer and egress costs
    - Support and professional services
    - Operational overhead
    - Compliance and security costs
    - Hidden costs and fees
    - Multi-year projections with growth
    """
    
    def __init__(self):
        """Initialize TCO calculator with pricing clients."""
        self.pricing_clients = {
            "aws": AWSPricingClient(),
            "gcp": GCPPricingClient(),
            "azure": AzurePricingClient()
        }
        self.cost_factors = CostFactors()
    
    async def calculate_comprehensive_tco(
        self,
        workload: WorkloadSpec,
        time_horizon_years: int = 3,
        providers: Optional[List[str]] = None,
        regions: Optional[Dict[str, str]] = None,
        support_levels: Optional[Dict[str, str]] = None,
        growth_assumptions: Optional[Dict[str, Decimal]] = None,
        include_reserved_instances: bool = True
    ) -> TCOAnalysis:
        """
        Calculate comprehensive TCO analysis for a workload across providers.
        
        Args:
            workload: Workload specification
            time_horizon_years: Analysis time horizon (1-10 years)
            providers: List of providers to analyze
            regions: Preferred regions by provider
            support_levels: Support levels by provider
            growth_assumptions: Custom growth rate assumptions
            include_reserved_instances: Whether to include RI discounts
            
        Returns:
            TCOAnalysis with detailed breakdown by provider and year
        """
        logger.info(f"Calculating TCO for workload: {workload.name}")
        
        if providers is None:
            providers = ["aws", "gcp", "azure"]
        
        if regions is None:
            regions = {
                "aws": "us-east-1",
                "gcp": "us-central1",
                "azure": "eastus"
            }
        
        if support_levels is None:
            support_levels = {
                "aws": "business",
                "gcp": "standard", 
                "azure": "standard"
            }
        
        if growth_assumptions is None:
            growth_assumptions = CostFactors.DEFAULT_GROWTH_ASSUMPTIONS
        
        # Calculate TCO for each provider
        provider_tco = {}
        total_tco_by_provider = {}
        
        for provider in providers:
            region = regions.get(provider, "us-east-1")
            support_level = support_levels.get(provider, "standard")
            
            logger.info(f"Calculating TCO for {provider} in {region}")
            
            yearly_breakdown = await self._calculate_provider_tco(
                workload=workload,
                provider=provider,
                region=region,
                support_level=support_level,
                time_horizon_years=time_horizon_years,
                growth_assumptions=growth_assumptions,
                include_reserved_instances=include_reserved_instances
            )
            
            provider_tco[CloudProvider(provider)] = yearly_breakdown
            total_tco_by_provider[CloudProvider(provider)] = sum(
                breakdown.total_cost for breakdown in yearly_breakdown
            )
        
        # Determine lowest TCO provider
        lowest_tco_provider = min(total_tco_by_provider.keys(), 
                                 key=lambda p: total_tco_by_provider[p])
        
        # Calculate savings percentage
        lowest_tco = total_tco_by_provider[lowest_tco_provider]
        highest_tco = max(total_tco_by_provider.values())
        tco_savings_percentage = float((highest_tco - lowest_tco) / highest_tco * 100) if highest_tco > 0 else 0.0
        
        # Calculate hidden cost factors
        hidden_cost_factors = await self._calculate_hidden_cost_factors(
            workload, providers, regions, time_horizon_years
        )
        
        # Generate risk factors
        risk_factors = self._generate_risk_factors(workload, providers, time_horizon_years)
        
        return TCOAnalysis(
            workload_name=workload.name,
            analysis_id=f"tco_{workload.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            time_horizon_years=time_horizon_years,
            provider_tco=provider_tco,
            total_tco_by_provider=total_tco_by_provider,
            lowest_tco_provider=lowest_tco_provider,
            tco_savings_percentage=tco_savings_percentage,
            hidden_cost_factors=hidden_cost_factors,
            risk_factors=risk_factors
        )
    
    async def _calculate_provider_tco(
        self,
        workload: WorkloadSpec,
        provider: str,
        region: str,
        support_level: str,
        time_horizon_years: int,
        growth_assumptions: Dict[str, Decimal],
        include_reserved_instances: bool
    ) -> List[TCOBreakdown]:
        """Calculate year-by-year TCO breakdown for a specific provider."""
        
        yearly_breakdowns = []
        
        # Calculate base year costs
        base_infrastructure_cost = await self.calculate_base_infrastructure_costs(
            workload, provider, region
        )
        
        # Apply reserved instance discounts if enabled
        if include_reserved_instances and time_horizon_years >= 1:
            ri_term = "3_year" if time_horizon_years >= 3 else "1_year"
            ri_discount = CostFactors.get_reserved_instance_discount(provider, ri_term)
            base_infrastructure_cost = base_infrastructure_cost * (Decimal("1") - ri_discount)
        
        for year in range(1, time_horizon_years + 1):
            # Apply growth to infrastructure costs
            infrastructure_cost = CostFactors.apply_growth_rate(
                base_infrastructure_cost, 
                year - 1, 
                growth_assumptions.get("compute_growth", Decimal("0.15"))
            )
            
            # Calculate other cost components
            data_transfer_cost = await self.calculate_data_transfer_costs(
                workload, provider, region, year, growth_assumptions
            )
            
            support_cost = self.calculate_support_costs(
                infrastructure_cost, provider, support_level
            )
            
            operational_cost = self.calculate_operational_overhead(
                workload, provider, infrastructure_cost
            )
            
            compliance_cost = self.calculate_compliance_costs(
                workload.compliance_requirements, provider
            )
            
            # Training costs (higher in first year, then reduced)
            training_cost = self._calculate_training_costs(provider, year, infrastructure_cost)
            
            # Migration costs (only in first year)
            migration_cost = Decimal("0")
            if year == 1:
                migration_cost = self._calculate_migration_costs(workload, provider)
            
            # Calculate total cost for the year
            total_cost = (
                infrastructure_cost + 
                data_transfer_cost + 
                support_cost + 
                operational_cost + 
                compliance_cost + 
                training_cost + 
                migration_cost
            )
            
            breakdown = TCOBreakdown(
                year=year,
                infrastructure_cost=infrastructure_cost,
                operational_cost=operational_cost,
                support_cost=support_cost,
                training_cost=training_cost,
                migration_cost=migration_cost,
                total_cost=total_cost
            )
            
            yearly_breakdowns.append(breakdown)
        
        return yearly_breakdowns
    
    async def calculate_base_infrastructure_costs(
        self, 
        workload: WorkloadSpec, 
        provider: str,
        region: str = None
    ) -> Decimal:
        """
        Calculate base infrastructure costs for compute, storage, and network.
        
        Args:
            workload: Workload specification
            provider: Cloud provider name
            region: Target region
            
        Returns:
            Monthly base infrastructure cost
        """
        logger.info(f"Calculating base infrastructure costs for {provider}")
        
        if region is None:
            region = "us-east-1" if provider == "aws" else "us-central1" if provider == "gcp" else "eastus"
        
        total_cost = Decimal("0")
        
        try:
            pricing_client = self.pricing_clients[provider]
            
            # Calculate compute costs
            if workload.compute:
                compute_pricing = await pricing_client.get_compute_pricing(
                    region=region,
                    instance_type=None,  # Will get all available types
                    operating_system=workload.compute.operating_system
                )
                
                if compute_pricing:
                    # Find best matching instance type
                    best_match = self._find_best_compute_match(workload.compute, compute_pricing)
                    if best_match:
                        total_cost += best_match.price_per_month
                        logger.debug(f"Compute cost: ${best_match.price_per_month}/month")
            
            # Calculate storage costs
            if workload.storage:
                for storage_spec in workload.storage:
                    storage_pricing = await pricing_client.get_storage_pricing(
                        region=region,
                        storage_type=storage_spec.storage_type.value if hasattr(storage_spec.storage_type, 'value') else str(storage_spec.storage_type)
                    )
                    
                    if storage_pricing:
                        # Calculate monthly storage cost
                        storage_cost = self._calculate_storage_monthly_cost(storage_spec, storage_pricing[0])
                        total_cost += storage_cost
                        logger.debug(f"Storage cost: ${storage_cost}/month")
            
            # Calculate network costs
            if workload.network and workload.network.components:
                network_pricing = await pricing_client.get_network_pricing(region=region)
                
                if network_pricing:
                    network_cost = self._calculate_network_monthly_cost(workload.network, network_pricing)
                    total_cost += network_cost
                    logger.debug(f"Network cost: ${network_cost}/month")
            
            # Apply regional cost multiplier
            regional_multiplier = CostFactors.get_regional_multiplier(provider, region)
            total_cost = total_cost * regional_multiplier
            
            logger.info(f"Base infrastructure cost for {provider}: ${total_cost}/month")
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating base infrastructure costs for {provider}: {e}")
            # Return estimated cost based on workload specs
            return self._estimate_infrastructure_cost(workload, provider)
    
    async def calculate_data_transfer_costs(
        self, 
        workload: WorkloadSpec, 
        provider: str,
        region: str = None,
        year: int = 1,
        growth_assumptions: Optional[Dict[str, Decimal]] = None
    ) -> Decimal:
        """
        Calculate data transfer and egress costs.
        
        Args:
            workload: Workload specification
            provider: Cloud provider name
            region: Target region
            year: Year number for growth calculations
            growth_assumptions: Growth rate assumptions
            
        Returns:
            Monthly data transfer cost
        """
        logger.info(f"Calculating data transfer costs for {provider}")
        
        if growth_assumptions is None:
            growth_assumptions = CostFactors.DEFAULT_GROWTH_ASSUMPTIONS
        
        # Estimate monthly data transfer based on workload
        base_monthly_transfer_gb = self._estimate_monthly_data_transfer(workload)
        
        # Apply growth for the year
        monthly_transfer_gb = CostFactors.apply_growth_rate(
            base_monthly_transfer_gb,
            year - 1,
            growth_assumptions.get("data_growth", Decimal("0.40"))
        )
        
        # Calculate costs for different transfer types
        outbound_rate = CostFactors.get_data_transfer_rate(provider, "outbound_internet")
        cross_region_rate = CostFactors.get_data_transfer_rate(provider, "cross_region")
        
        # Assume 80% outbound internet, 20% cross-region
        outbound_cost = monthly_transfer_gb * Decimal("0.8") * outbound_rate
        cross_region_cost = monthly_transfer_gb * Decimal("0.2") * cross_region_rate
        
        total_transfer_cost = outbound_cost + cross_region_cost
        
        logger.debug(f"Data transfer cost for {provider}: ${total_transfer_cost}/month")
        return total_transfer_cost
    
    def calculate_support_costs(
        self, 
        base_cost: Decimal, 
        provider: str,
        support_level: str = "standard"
    ) -> Decimal:
        """
        Calculate support fees and professional services costs.
        
        Args:
            base_cost: Base infrastructure cost
            provider: Cloud provider name
            support_level: Support tier level
            
        Returns:
            Monthly support cost
        """
        logger.info(f"Calculating support costs for {provider} ({support_level})")
        
        # Get support cost percentage
        support_percentage = CostFactors.get_support_cost_percentage(provider, support_level)
        
        # Calculate percentage-based cost
        percentage_cost = base_cost * support_percentage
        
        # Get minimum support cost
        minimum_cost = CostFactors.get_minimum_support_cost(provider, support_level)
        
        # Use the higher of percentage or minimum
        support_cost = max(percentage_cost, minimum_cost)
        
        logger.debug(f"Support cost for {provider}: ${support_cost}/month")
        return support_cost
    
    def calculate_operational_overhead(
        self, 
        workload: WorkloadSpec, 
        provider: str,
        base_cost: Decimal
    ) -> Decimal:
        """
        Calculate operational overhead based on workload complexity.
        
        Args:
            workload: Workload specification
            provider: Cloud provider name
            base_cost: Base infrastructure cost
            
        Returns:
            Monthly operational overhead cost
        """
        logger.info(f"Calculating operational overhead for {provider}")
        
        # Determine workload complexity
        complexity = self._determine_workload_complexity(workload)
        
        # Get operational overhead percentage
        overhead_percentage = CostFactors.get_operational_overhead_total(complexity)
        
        # Calculate overhead cost
        overhead_cost = base_cost * overhead_percentage
        
        # Add provider-specific hidden costs
        hidden_cost_multipliers = [
            CostFactors.get_hidden_cost_multiplier(provider, "api_calls"),
            CostFactors.get_hidden_cost_multiplier(provider, "storage_operations")
        ]
        
        for multiplier in hidden_cost_multipliers:
            overhead_cost += base_cost * multiplier
        
        logger.debug(f"Operational overhead for {provider}: ${overhead_cost}/month")
        return overhead_cost
    
    def calculate_compliance_costs(
        self, 
        requirements: List[str], 
        provider: str
    ) -> Decimal:
        """
        Calculate compliance and security costs.
        
        Args:
            requirements: List of compliance requirements
            provider: Cloud provider name
            
        Returns:
            Monthly compliance cost
        """
        logger.info(f"Calculating compliance costs for {provider}")
        
        if not requirements:
            return Decimal("0")
        
        # Get annual compliance costs
        annual_compliance_cost = CostFactors.get_compliance_cost_total(requirements)
        
        # Convert to monthly cost
        monthly_compliance_cost = annual_compliance_cost / Decimal("12")
        
        logger.debug(f"Compliance cost for {provider}: ${monthly_compliance_cost}/month")
        return monthly_compliance_cost
    
    def generate_multi_year_projections(
        self, 
        tco: TCOAnalysis, 
        years: int
    ) -> Dict[int, TCOAnalysis]:
        """
        Generate multi-year TCO projections with growth assumptions.
        
        Args:
            tco: Base TCO analysis
            years: Number of years to project
            
        Returns:
            Dictionary mapping year to TCO analysis
        """
        logger.info(f"Generating {years}-year TCO projections")
        
        projections = {}
        
        for year in range(1, years + 1):
            # Create projection based on base TCO
            year_tco = TCOAnalysis(
                workload_name=tco.workload_name,
                analysis_id=f"{tco.analysis_id}_year_{year}",
                time_horizon_years=1,
                provider_tco={},
                total_tco_by_provider={},
                lowest_tco_provider=tco.lowest_tco_provider,
                tco_savings_percentage=tco.tco_savings_percentage,
                hidden_cost_factors=tco.hidden_cost_factors,
                risk_factors=tco.risk_factors
            )
            
            # Apply growth to each provider's costs
            for provider, yearly_breakdowns in tco.provider_tco.items():
                if yearly_breakdowns and len(yearly_breakdowns) >= year:
                    year_breakdown = yearly_breakdowns[year - 1]
                    year_tco.provider_tco[provider] = [year_breakdown]
                    year_tco.total_tco_by_provider[provider] = year_breakdown.total_cost
            
            projections[year] = year_tco
        
        return projections
    
    # Helper methods
    
    def _find_best_compute_match(self, compute_spec, pricing_options):
        """Find the best matching compute instance from pricing options."""
        
        required_vcpus = compute_spec.vcpus
        required_memory = compute_spec.memory_gb
        
        # Filter options that meet minimum requirements
        suitable_options = [
            option for option in pricing_options
            if option.vcpus >= required_vcpus and option.memory_gb >= required_memory
        ]
        
        if not suitable_options:
            # If no exact match, find the closest larger option
            suitable_options = pricing_options
        
        # Sort by cost efficiency (cost per vCPU + cost per GB memory)
        def cost_efficiency(option):
            cpu_cost = option.price_per_month / option.vcpus if option.vcpus > 0 else float('inf')
            memory_cost = option.price_per_month / option.memory_gb if option.memory_gb > 0 else float('inf')
            return cpu_cost + memory_cost
        
        return min(suitable_options, key=cost_efficiency) if suitable_options else None
    
    def _calculate_storage_monthly_cost(self, storage_spec, storage_pricing):
        """Calculate monthly storage cost based on specification and pricing."""
        
        capacity_gb = storage_spec.capacity_gb
        price_per_gb_month = storage_pricing.price_per_gb_month
        
        base_cost = Decimal(str(capacity_gb)) * price_per_gb_month
        
        # Add IOPS costs if applicable
        if hasattr(storage_spec, 'iops_requirement') and storage_spec.iops_requirement:
            if storage_pricing.iops_price:
                iops_cost = Decimal(str(storage_spec.iops_requirement)) * storage_pricing.iops_price
                base_cost += iops_cost
        
        return base_cost
    
    def _calculate_network_monthly_cost(self, network_spec, network_pricing):
        """Calculate monthly network service costs."""
        
        total_cost = Decimal("0")
        
        for component in network_spec.components:
            # Find matching network service pricing
            for pricing in network_pricing:
                if hasattr(component, 'service_type'):
                    service_type = component.service_type.value if hasattr(component.service_type, 'value') else str(component.service_type)
                else:
                    service_type = component.get('service_type', 'unknown')
                
                if pricing.service_type == service_type or service_type in pricing.service_type:
                    if pricing.price_per_hour:
                        # Convert hourly to monthly (24 * 30 = 720 hours)
                        monthly_cost = pricing.price_per_hour * Decimal("720")
                        total_cost += monthly_cost
                    break
        
        return total_cost
    
    def _estimate_infrastructure_cost(self, workload: WorkloadSpec, provider: str) -> Decimal:
        """Estimate infrastructure cost when pricing API is unavailable."""
        
        # Basic cost estimation based on workload specs
        base_cost = Decimal("0")
        
        # Compute cost estimation
        if workload.compute:
            vcpu_cost_per_hour = Decimal("0.05")  # Rough estimate
            memory_cost_per_gb_hour = Decimal("0.01")  # Rough estimate
            
            hourly_compute = (
                Decimal(str(workload.compute.vcpus)) * vcpu_cost_per_hour +
                Decimal(str(workload.compute.memory_gb)) * memory_cost_per_gb_hour
            )
            monthly_compute = hourly_compute * Decimal("720")  # 24 * 30 hours
            base_cost += monthly_compute
        
        # Storage cost estimation
        if workload.storage:
            for storage in workload.storage:
                storage_cost_per_gb = Decimal("0.023")  # Rough estimate for object storage
                monthly_storage = Decimal(str(storage.capacity_gb)) * storage_cost_per_gb
                base_cost += monthly_storage
        
        # Network cost estimation (rough estimate)
        if workload.network and workload.network.components:
            network_cost = Decimal("50") * len(workload.network.components)  # $50 per component
            base_cost += network_cost
        
        return base_cost
    
    def _estimate_monthly_data_transfer(self, workload: WorkloadSpec) -> Decimal:
        """Estimate monthly data transfer based on workload characteristics."""
        
        # Base estimation logic
        base_transfer = Decimal("100")  # 100 GB base
        
        # Add based on compute resources
        if workload.compute:
            vcpu_factor = Decimal(str(workload.compute.vcpus)) * Decimal("50")  # 50 GB per vCPU
            base_transfer += vcpu_factor
        
        # Add based on storage
        if workload.storage:
            for storage in workload.storage:
                if hasattr(storage, 'storage_type') and storage.storage_type == "object":
                    # Object storage typically has more data transfer
                    storage_factor = Decimal(str(storage.capacity_gb)) * Decimal("0.1")  # 10% of storage per month
                    base_transfer += storage_factor
        
        return base_transfer
    
    def _determine_workload_complexity(self, workload: WorkloadSpec) -> str:
        """Determine workload complexity based on specifications."""
        
        complexity_score = 0
        
        # Compute complexity
        if workload.compute:
            if workload.compute.vcpus > 16:
                complexity_score += 2
            elif workload.compute.vcpus > 4:
                complexity_score += 1
            
            if workload.compute.memory_gb > 64:
                complexity_score += 2
            elif workload.compute.memory_gb > 16:
                complexity_score += 1
        
        # Storage complexity
        if workload.storage:
            complexity_score += len(workload.storage)
            for storage in workload.storage:
                if storage.capacity_gb > 1000:
                    complexity_score += 1
        
        # Network complexity
        if workload.network and workload.network.components:
            complexity_score += len(workload.network.components)
        
        # Database complexity
        if workload.database:
            complexity_score += 2
        
        # Compliance complexity
        if workload.compliance_requirements:
            complexity_score += len(workload.compliance_requirements)
        
        # Determine complexity level
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 7:
            return "moderate"
        elif complexity_score <= 12:
            return "complex"
        else:
            return "enterprise"
    
    def _calculate_training_costs(self, provider: str, year: int, infrastructure_cost: Decimal) -> Decimal:
        """Calculate training and certification costs."""
        
        if year == 1:
            # Higher training costs in first year
            return infrastructure_cost * Decimal("0.05")  # 5% of infrastructure cost
        elif year == 2:
            # Reduced training costs in second year
            return infrastructure_cost * Decimal("0.02")  # 2% of infrastructure cost
        else:
            # Minimal ongoing training costs
            return infrastructure_cost * Decimal("0.01")  # 1% of infrastructure cost
    
    def _calculate_migration_costs(self, workload: WorkloadSpec, provider: str) -> Decimal:
        """Calculate one-time migration costs."""
        
        # Base migration cost
        base_migration = Decimal("10000")  # $10,000 base migration cost
        
        # Add based on workload complexity
        complexity = self._determine_workload_complexity(workload)
        complexity_multipliers = {
            "simple": Decimal("1.0"),
            "moderate": Decimal("2.0"),
            "complex": Decimal("4.0"),
            "enterprise": Decimal("8.0")
        }
        
        migration_cost = base_migration * complexity_multipliers.get(complexity, Decimal("2.0"))
        
        return migration_cost
    
    async def _calculate_hidden_cost_factors(
        self, 
        workload: WorkloadSpec, 
        providers: List[str], 
        regions: Dict[str, str], 
        years: int
    ) -> Dict[str, Dict[CloudProvider, Decimal]]:
        """Calculate hidden cost factors by provider."""
        
        hidden_factors = {}
        
        for cost_type in ["data_egress", "api_calls", "storage_operations", "cross_region_traffic"]:
            hidden_factors[cost_type] = {}
            
            for provider in providers:
                multiplier = CostFactors.get_hidden_cost_multiplier(provider, cost_type)
                # Estimate annual hidden cost based on infrastructure
                base_cost = await self.calculate_base_infrastructure_costs(workload, provider, regions.get(provider))
                annual_hidden_cost = base_cost * multiplier * Decimal("12") * years
                
                hidden_factors[cost_type][CloudProvider(provider)] = annual_hidden_cost
        
        return hidden_factors
    
    def _generate_risk_factors(
        self, 
        workload: WorkloadSpec, 
        providers: List[str], 
        years: int
    ) -> List[str]:
        """Generate list of risk factors for TCO analysis."""
        
        risk_factors = []
        
        # Time horizon risks
        if years > 5:
            risk_factors.append("Long-term pricing volatility risk")
        
        # Workload complexity risks
        complexity = self._determine_workload_complexity(workload)
        if complexity in ["complex", "enterprise"]:
            risk_factors.append("High complexity may increase operational costs")
        
        # Compliance risks
        if workload.compliance_requirements:
            risk_factors.append("Compliance requirements may increase costs over time")
        
        # Multi-provider risks
        if len(providers) > 1:
            risk_factors.append("Provider pricing changes may affect cost comparisons")
        
        # Technology evolution risks
        risk_factors.append("New services and pricing models may emerge")
        risk_factors.append("Reserved instance pricing may change")
        
        return risk_factors
    
    async def close(self):
        """Close pricing client connections."""
        for client in self.pricing_clients.values():
            await client.close()