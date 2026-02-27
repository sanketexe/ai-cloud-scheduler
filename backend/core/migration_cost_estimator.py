"""
Migration Cost Estimator

Calculates comprehensive migration costs and ROI analysis for moving workloads
between cloud providers, including data transfer, downtime, retraining, and
break-even analysis.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .multi_cloud_models import (
    WorkloadSpec, MigrationAnalysis, MigrationCostBreakdown, RiskAssessment,
    CloudProvider, CostRecommendation
)
from .cost_factors import CostFactors
from .tco_calculator import TCOCalculator

logger = logging.getLogger(__name__)


class MigrationComplexity(str, Enum):
    """Migration complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class MigrationApproach(str, Enum):
    """Migration approach strategies"""
    LIFT_AND_SHIFT = "lift_and_shift"
    RE_PLATFORM = "re_platform"
    RE_ARCHITECT = "re_architect"
    HYBRID = "hybrid"


class MigrationCostEstimator:
    """
    Comprehensive migration cost estimator for multi-cloud workload migrations.
    
    Calculates:
    - Data transfer costs
    - Downtime impact costs
    - Re-architecture and refactoring costs
    - Training and certification expenses
    - Break-even analysis and payback periods
    - Risk assessment and mitigation strategies
    """
    
    def __init__(self):
        """Initialize migration cost estimator."""
        self.tco_calculator = TCOCalculator()
        self.cost_factors = CostFactors()
    async def analyze_migration_costs(
        self,
        workload: WorkloadSpec,
        source_provider: str,
        target_provider: str,
        current_monthly_cost: Optional[Decimal] = None,
        migration_approach: str = "lift_and_shift",
        timeline_preference: Optional[str] = None,
        risk_tolerance: str = "medium",
        team_size: int = 5,
        business_criticality: str = "medium"
    ) -> MigrationAnalysis:
        """
        Analyze comprehensive migration costs and timeline.
        
        Args:
            workload: Workload specification
            source_provider: Source cloud provider
            target_provider: Target cloud provider
            current_monthly_cost: Current monthly cost (if known)
            migration_approach: Migration strategy approach
            timeline_preference: Preferred timeline (aggressive/moderate/conservative)
            risk_tolerance: Risk tolerance level (low/medium/high)
            team_size: Size of migration team
            business_criticality: Business criticality level
            
        Returns:
            MigrationAnalysis with detailed cost breakdown and recommendations
        """
        logger.info(f"Analyzing migration from {source_provider} to {target_provider}")
        
        # Determine migration complexity
        complexity = self._determine_migration_complexity(
            workload, source_provider, target_provider, migration_approach
        )
        
        # Calculate current cost if not provided
        if current_monthly_cost is None:
            current_monthly_cost = await self.tco_calculator.calculate_base_infrastructure_costs(
                workload, source_provider
            )
        
        # Calculate target provider cost
        target_monthly_cost = await self.tco_calculator.calculate_base_infrastructure_costs(
            workload, target_provider
        )
        
        # Calculate migration cost breakdown
        migration_costs = await self._calculate_migration_cost_breakdown(
            workload=workload,
            source_provider=source_provider,
            target_provider=target_provider,
            complexity=complexity,
            migration_approach=migration_approach,
            team_size=team_size,
            business_criticality=business_criticality
        )
        
        # Estimate migration timeline
        timeline_days = self.estimate_migration_timeline(
            workload, source_provider, target_provider, complexity, timeline_preference
        )
        
        # Calculate monthly savings
        monthly_savings = current_monthly_cost - target_monthly_cost
        
        # Calculate break-even period
        break_even_months = self.calculate_break_even_period(
            migration_costs.total_migration_cost, monthly_savings
        ) if monthly_savings > 0 else None
        
        # Assess migration risks
        risk_assessment = self.assess_migration_risks(
            workload, source_provider, target_provider, complexity, risk_tolerance
        )
        
        # Generate recommendations
        recommended_approach = self._recommend_migration_approach(
            complexity, risk_assessment, timeline_days, monthly_savings
        )
        
        key_considerations = self._generate_key_considerations(
            workload, source_provider, target_provider, complexity, migration_costs
        )
        
        return MigrationAnalysis(
            analysis_id=f"migration_{workload.name}_{source_provider}_to_{target_provider}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            workload_name=workload.name,
            source_provider=CloudProvider(source_provider),
            target_provider=CloudProvider(target_provider),
            migration_costs=migration_costs,
            estimated_timeline_days=timeline_days,
            break_even_months=break_even_months,
            monthly_savings=monthly_savings,
            risk_assessment=risk_assessment,
            recommended_approach=recommended_approach,
            key_considerations=key_considerations
        )
    async def calculate_data_transfer_costs(
        self, 
        data_volume_gb: int, 
        source_provider: str, 
        target_provider: str,
        transfer_method: str = "internet"
    ) -> Decimal:
        """
        Calculate data transfer costs for migration between providers.
        
        Args:
            data_volume_gb: Total data volume to transfer in GB
            source_provider: Source cloud provider
            target_provider: Target cloud provider
            transfer_method: Transfer method (internet, direct_connect, physical)
            
        Returns:
            Total data transfer cost
        """
        logger.info(f"Calculating data transfer costs: {data_volume_gb}GB from {source_provider} to {target_provider}")
        
        if transfer_method == "internet":
            # Use standard egress rates
            egress_rate = CostFactors.get_data_transfer_rate(source_provider, "outbound_internet")
            transfer_cost = Decimal(str(data_volume_gb)) * egress_rate
            
        elif transfer_method == "direct_connect":
            # Direct connect typically has lower per-GB costs but setup fees
            setup_cost = Decimal("1000")  # $1,000 setup cost
            per_gb_rate = Decimal("0.02")  # $0.02 per GB
            transfer_cost = setup_cost + (Decimal(str(data_volume_gb)) * per_gb_rate)
            
        elif transfer_method == "physical":
            # Physical transfer (AWS Snowball, etc.)
            if data_volume_gb <= 80000:  # 80TB
                transfer_cost = Decimal("200")  # Per device cost
            else:
                # Multiple devices needed
                devices_needed = (data_volume_gb // 80000) + 1
                transfer_cost = Decimal("200") * devices_needed
        else:
            # Default to internet transfer
            egress_rate = CostFactors.get_data_transfer_rate(source_provider, "outbound_internet")
            transfer_cost = Decimal(str(data_volume_gb)) * egress_rate
        
        logger.debug(f"Data transfer cost: ${transfer_cost}")
        return transfer_cost
    
    def estimate_downtime_impact(
        self, 
        workload: WorkloadSpec, 
        migration_hours: int,
        business_criticality: str = "medium",
        revenue_per_hour: Optional[Decimal] = None
    ) -> Decimal:
        """
        Estimate downtime impact and associated business costs.
        
        Args:
            workload: Workload specification
            migration_hours: Expected downtime in hours
            business_criticality: Business criticality level
            revenue_per_hour: Revenue impact per hour (if known)
            
        Returns:
            Total downtime impact cost
        """
        logger.info(f"Estimating downtime impact for {migration_hours} hours")
        
        if revenue_per_hour is not None:
            # Use provided revenue impact
            downtime_cost = revenue_per_hour * Decimal(str(migration_hours))
        else:
            # Estimate based on workload characteristics and criticality
            base_hourly_impact = self._estimate_hourly_business_impact(workload, business_criticality)
            downtime_cost = base_hourly_impact * Decimal(str(migration_hours))
        
        logger.debug(f"Downtime impact cost: ${downtime_cost}")
        return downtime_cost
    def calculate_retraining_costs(
        self, 
        team_size: int, 
        source_provider: str, 
        target_provider: str,
        skill_gap_level: str = "medium"
    ) -> Decimal:
        """
        Calculate training and certification expenses for team.
        
        Args:
            team_size: Number of team members to train
            source_provider: Source cloud provider
            target_provider: Target cloud provider
            skill_gap_level: Skill gap level (low/medium/high)
            
        Returns:
            Total retraining cost
        """
        logger.info(f"Calculating retraining costs for {team_size} team members")
        
        # Base training costs per person
        base_training_costs = {
            "low": Decimal("2000"),     # $2,000 per person for basic training
            "medium": Decimal("5000"),  # $5,000 per person for comprehensive training
            "high": Decimal("10000")    # $10,000 per person for extensive training
        }
        
        per_person_cost = base_training_costs.get(skill_gap_level, Decimal("5000"))
        
        # Certification costs
        certification_cost_per_person = Decimal("500")  # Average certification cost
        
        # Total training cost
        total_training_cost = (per_person_cost + certification_cost_per_person) * Decimal(str(team_size))
        
        # Add productivity loss during training (20% of salary for training period)
        average_monthly_salary = Decimal("8000")  # $8,000 average monthly salary
        training_months = Decimal("2")  # 2 months training period
        productivity_loss = average_monthly_salary * training_months * Decimal("0.2") * Decimal(str(team_size))
        
        total_cost = total_training_cost + productivity_loss
        
        logger.debug(f"Retraining cost: ${total_cost}")
        return total_cost
    
    def estimate_migration_timeline(
        self, 
        workload: WorkloadSpec, 
        source_provider: str, 
        target_provider: str,
        complexity: MigrationComplexity,
        timeline_preference: Optional[str] = None
    ) -> int:
        """
        Estimate migration timeline in days.
        
        Args:
            workload: Workload specification
            source_provider: Source cloud provider
            target_provider: Target cloud provider
            complexity: Migration complexity level
            timeline_preference: Timeline preference (aggressive/moderate/conservative)
            
        Returns:
            Estimated timeline in days
        """
        logger.info(f"Estimating migration timeline for {complexity} migration")
        
        # Base timeline by complexity
        base_timelines = {
            MigrationComplexity.SIMPLE: 30,      # 30 days
            MigrationComplexity.MODERATE: 90,    # 90 days
            MigrationComplexity.COMPLEX: 180,    # 180 days
            MigrationComplexity.ENTERPRISE: 365  # 365 days
        }
        
        base_days = base_timelines.get(complexity, 90)
        
        # Adjust based on timeline preference
        timeline_multipliers = {
            "aggressive": Decimal("0.7"),    # 30% faster
            "moderate": Decimal("1.0"),      # Standard timeline
            "conservative": Decimal("1.5")   # 50% longer
        }
        
        multiplier = timeline_multipliers.get(timeline_preference, Decimal("1.0"))
        estimated_days = int(Decimal(str(base_days)) * multiplier)
        
        # Add buffer for workload-specific factors
        if workload.database:
            estimated_days += 14  # Additional 2 weeks for database migration
        
        if workload.compliance_requirements:
            estimated_days += 21  # Additional 3 weeks for compliance validation
        
        if len(workload.storage) > 3:
            estimated_days += 7   # Additional week for complex storage migration
        
        logger.debug(f"Estimated migration timeline: {estimated_days} days")
        return estimated_days
    def calculate_break_even_period(
        self, 
        migration_cost: Decimal, 
        monthly_savings: Decimal
    ) -> Optional[int]:
        """
        Calculate break-even period in months.
        
        Args:
            migration_cost: Total migration cost
            monthly_savings: Monthly cost savings after migration
            
        Returns:
            Break-even period in months, or None if no savings
        """
        logger.info(f"Calculating break-even period")
        
        if monthly_savings <= 0:
            logger.warning("No monthly savings - migration may not be cost-effective")
            return None
        
        break_even_months = int((migration_cost / monthly_savings).to_integral_value())
        
        logger.debug(f"Break-even period: {break_even_months} months")
        return break_even_months
    
    def assess_migration_risks(
        self, 
        workload: WorkloadSpec, 
        source_provider: str, 
        target_provider: str,
        complexity: MigrationComplexity,
        risk_tolerance: str = "medium"
    ) -> RiskAssessment:
        """
        Assess migration risks and mitigation strategies.
        
        Args:
            workload: Workload specification
            source_provider: Source cloud provider
            target_provider: Target cloud provider
            complexity: Migration complexity level
            risk_tolerance: Risk tolerance level
            
        Returns:
            RiskAssessment with identified risks and mitigation strategies
        """
        logger.info(f"Assessing migration risks for {complexity} migration")
        
        risk_factors = []
        mitigation_strategies = []
        
        # Complexity-based risks
        if complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.ENTERPRISE]:
            risk_factors.append({
                "category": "Technical Complexity",
                "description": "High complexity migration may face technical challenges",
                "impact": "high",
                "probability": "medium"
            })
            mitigation_strategies.append("Conduct thorough proof-of-concept testing")
            mitigation_strategies.append("Implement phased migration approach")
        
        # Data migration risks
        if workload.storage and sum(s.capacity_gb for s in workload.storage) > 10000:
            risk_factors.append({
                "category": "Data Migration",
                "description": "Large data volumes increase migration complexity and risk",
                "impact": "high",
                "probability": "medium"
            })
            mitigation_strategies.append("Use dedicated data transfer services")
            mitigation_strategies.append("Implement data validation and integrity checks")
        
        # Database migration risks
        if workload.database:
            risk_factors.append({
                "category": "Database Migration",
                "description": "Database migration may cause data consistency issues",
                "impact": "high",
                "probability": "low"
            })
            mitigation_strategies.append("Use database migration tools and services")
            mitigation_strategies.append("Implement comprehensive backup and rollback procedures")
        
        # Compliance risks
        if workload.compliance_requirements:
            risk_factors.append({
                "category": "Compliance",
                "description": "Compliance requirements may not be met in target environment",
                "impact": "high",
                "probability": "low"
            })
            mitigation_strategies.append("Validate compliance controls in target environment")
            mitigation_strategies.append("Engage compliance team early in migration planning")
        
        # Performance risks
        risk_factors.append({
            "category": "Performance",
            "description": "Application performance may be impacted during migration",
            "impact": "medium",
            "probability": "medium"
        })
        mitigation_strategies.append("Conduct performance testing in target environment")
        mitigation_strategies.append("Plan for performance optimization post-migration")
        
        # Team readiness risks
        risk_factors.append({
            "category": "Team Readiness",
            "description": "Team may lack expertise in target cloud platform",
            "impact": "medium",
            "probability": "high"
        })
        mitigation_strategies.append("Provide comprehensive training before migration")
        mitigation_strategies.append("Engage cloud consulting services for guidance")
        
        # Calculate overall risk level
        high_impact_risks = len([r for r in risk_factors if r["impact"] == "high"])
        if high_impact_risks >= 3:
            overall_risk_level = "high"
        elif high_impact_risks >= 1:
            overall_risk_level = "medium"
        else:
            overall_risk_level = "low"
        
        # Calculate success probability
        risk_score = sum(
            3 if r["impact"] == "high" else 2 if r["impact"] == "medium" else 1
            for r in risk_factors
        )
        
        if risk_score <= 5:
            success_probability = 0.9
        elif risk_score <= 10:
            success_probability = 0.75
        elif risk_score <= 15:
            success_probability = 0.6
        else:
            success_probability = 0.4
        
        return RiskAssessment(
            overall_risk_level=overall_risk_level,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            success_probability=success_probability
        )
    # Helper methods
    
    async def _calculate_migration_cost_breakdown(
        self,
        workload: WorkloadSpec,
        source_provider: str,
        target_provider: str,
        complexity: MigrationComplexity,
        migration_approach: str,
        team_size: int,
        business_criticality: str
    ) -> MigrationCostBreakdown:
        """Calculate detailed migration cost breakdown."""
        
        # Estimate data volume for transfer
        total_data_gb = sum(s.capacity_gb for s in workload.storage) if workload.storage else 1000
        
        # Calculate data transfer costs
        data_transfer_cost = await self.calculate_data_transfer_costs(
            total_data_gb, source_provider, target_provider
        )
        
        # Calculate downtime costs
        downtime_hours = self._estimate_downtime_hours(complexity, migration_approach)
        downtime_cost = self.estimate_downtime_impact(
            workload, downtime_hours, business_criticality
        )
        
        # Calculate retraining costs
        skill_gap = self._assess_skill_gap(source_provider, target_provider)
        retraining_cost = self.calculate_retraining_costs(
            team_size, source_provider, target_provider, skill_gap
        )
        
        # Calculate consulting costs
        consulting_cost = self._calculate_consulting_costs(complexity, migration_approach)
        
        # Calculate tool licensing costs
        tool_licensing_cost = self._calculate_tool_licensing_costs(complexity, total_data_gb)
        
        # Calculate testing costs
        testing_cost = self._calculate_testing_costs(workload, complexity)
        
        # Calculate total migration cost
        total_migration_cost = (
            data_transfer_cost +
            downtime_cost +
            retraining_cost +
            consulting_cost +
            tool_licensing_cost +
            testing_cost
        )
        
        return MigrationCostBreakdown(
            data_transfer_cost=data_transfer_cost,
            downtime_cost=downtime_cost,
            retraining_cost=retraining_cost,
            consulting_cost=consulting_cost,
            tool_licensing_cost=tool_licensing_cost,
            testing_cost=testing_cost,
            total_migration_cost=total_migration_cost
        )
    
    def _determine_migration_complexity(
        self,
        workload: WorkloadSpec,
        source_provider: str,
        target_provider: str,
        migration_approach: str
    ) -> MigrationComplexity:
        """Determine migration complexity based on workload and approach."""
        
        complexity_score = 0
        
        # Workload complexity factors
        if workload.compute and workload.compute.vcpus > 16:
            complexity_score += 2
        
        if workload.storage and len(workload.storage) > 3:
            complexity_score += 2
        
        if workload.database:
            complexity_score += 3
        
        if workload.network and len(workload.network.components) > 2:
            complexity_score += 2
        
        if workload.compliance_requirements:
            complexity_score += len(workload.compliance_requirements)
        
        # Migration approach complexity
        approach_complexity = {
            "lift_and_shift": 0,
            "re_platform": 2,
            "re_architect": 4,
            "hybrid": 3
        }
        complexity_score += approach_complexity.get(migration_approach, 2)
        
        # Provider compatibility
        if source_provider == target_provider:
            complexity_score -= 2  # Same provider is easier
        
        # Determine complexity level
        if complexity_score <= 3:
            return MigrationComplexity.SIMPLE
        elif complexity_score <= 7:
            return MigrationComplexity.MODERATE
        elif complexity_score <= 12:
            return MigrationComplexity.COMPLEX
        else:
            return MigrationComplexity.ENTERPRISE
    def _estimate_hourly_business_impact(
        self, 
        workload: WorkloadSpec, 
        business_criticality: str
    ) -> Decimal:
        """Estimate hourly business impact based on workload and criticality."""
        
        # Base impact by criticality
        base_impacts = {
            "low": Decimal("500"),      # $500/hour
            "medium": Decimal("2000"),  # $2,000/hour
            "high": Decimal("10000"),   # $10,000/hour
            "critical": Decimal("50000") # $50,000/hour
        }
        
        base_impact = base_impacts.get(business_criticality, Decimal("2000"))
        
        # Adjust based on workload characteristics
        if workload.compute and workload.compute.vcpus > 32:
            base_impact *= Decimal("1.5")  # Large workloads have higher impact
        
        if workload.database:
            base_impact *= Decimal("1.3")  # Database workloads are more critical
        
        return base_impact
    
    def _estimate_downtime_hours(
        self, 
        complexity: MigrationComplexity, 
        migration_approach: str
    ) -> int:
        """Estimate downtime hours based on complexity and approach."""
        
        base_downtime = {
            MigrationComplexity.SIMPLE: 4,      # 4 hours
            MigrationComplexity.MODERATE: 12,   # 12 hours
            MigrationComplexity.COMPLEX: 24,    # 24 hours
            MigrationComplexity.ENTERPRISE: 48  # 48 hours
        }
        
        downtime = base_downtime.get(complexity, 12)
        
        # Adjust based on migration approach
        approach_multipliers = {
            "lift_and_shift": 1.0,
            "re_platform": 1.5,
            "re_architect": 2.0,
            "hybrid": 1.3
        }
        
        multiplier = approach_multipliers.get(migration_approach, 1.0)
        return int(downtime * multiplier)
    
    def _assess_skill_gap(self, source_provider: str, target_provider: str) -> str:
        """Assess skill gap between source and target providers."""
        
        # Provider similarity matrix
        similarity_matrix = {
            ("aws", "azure"): "medium",
            ("aws", "gcp"): "medium",
            ("azure", "gcp"): "high",
            ("gcp", "azure"): "high",
            ("azure", "aws"): "medium",
            ("gcp", "aws"): "medium"
        }
        
        if source_provider == target_provider:
            return "low"
        
        return similarity_matrix.get((source_provider, target_provider), "high")
    
    def _calculate_consulting_costs(
        self, 
        complexity: MigrationComplexity, 
        migration_approach: str
    ) -> Decimal:
        """Calculate consulting and professional services costs."""
        
        base_consulting_costs = {
            MigrationComplexity.SIMPLE: Decimal("10000"),     # $10,000
            MigrationComplexity.MODERATE: Decimal("25000"),   # $25,000
            MigrationComplexity.COMPLEX: Decimal("75000"),    # $75,000
            MigrationComplexity.ENTERPRISE: Decimal("200000") # $200,000
        }
        
        base_cost = base_consulting_costs.get(complexity, Decimal("25000"))
        
        # Adjust based on migration approach
        if migration_approach == "re_architect":
            base_cost *= Decimal("1.5")
        elif migration_approach == "re_platform":
            base_cost *= Decimal("1.2")
        
        return base_cost
    
    def _calculate_tool_licensing_costs(
        self, 
        complexity: MigrationComplexity, 
        data_volume_gb: int
    ) -> Decimal:
        """Calculate migration tool licensing costs."""
        
        base_tool_costs = {
            MigrationComplexity.SIMPLE: Decimal("1000"),     # $1,000
            MigrationComplexity.MODERATE: Decimal("5000"),   # $5,000
            MigrationComplexity.COMPLEX: Decimal("15000"),   # $15,000
            MigrationComplexity.ENTERPRISE: Decimal("50000") # $50,000
        }
        
        base_cost = base_tool_costs.get(complexity, Decimal("5000"))
        
        # Add data transfer tool costs for large volumes
        if data_volume_gb > 10000:  # > 10TB
            base_cost += Decimal("5000")  # Additional tools for large data migration
        
        return base_cost
    def _calculate_testing_costs(
        self, 
        workload: WorkloadSpec, 
        complexity: MigrationComplexity
    ) -> Decimal:
        """Calculate testing and validation costs."""
        
        base_testing_costs = {
            MigrationComplexity.SIMPLE: Decimal("5000"),     # $5,000
            MigrationComplexity.MODERATE: Decimal("15000"),  # $15,000
            MigrationComplexity.COMPLEX: Decimal("40000"),   # $40,000
            MigrationComplexity.ENTERPRISE: Decimal("100000") # $100,000
        }
        
        base_cost = base_testing_costs.get(complexity, Decimal("15000"))
        
        # Add costs for specific testing requirements
        if workload.database:
            base_cost += Decimal("10000")  # Database testing
        
        if workload.compliance_requirements:
            base_cost += Decimal("5000") * len(workload.compliance_requirements)
        
        return base_cost
    
    def _recommend_migration_approach(
        self,
        complexity: MigrationComplexity,
        risk_assessment: RiskAssessment,
        timeline_days: int,
        monthly_savings: Decimal
    ) -> str:
        """Recommend optimal migration approach based on analysis."""
        
        if complexity == MigrationComplexity.SIMPLE and risk_assessment.overall_risk_level == "low":
            return "Proceed with lift-and-shift migration using automated tools"
        elif complexity == MigrationComplexity.MODERATE and monthly_savings > Decimal("5000"):
            return "Implement phased migration with re-platforming for key components"
        elif risk_assessment.overall_risk_level == "high":
            return "Consider hybrid approach with extensive testing and gradual cutover"
        elif timeline_days > 180:
            return "Break migration into smaller phases to reduce risk and complexity"
        else:
            return "Proceed with standard migration approach with proper planning and testing"
    
    def _generate_key_considerations(
        self,
        workload: WorkloadSpec,
        source_provider: str,
        target_provider: str,
        complexity: MigrationComplexity,
        migration_costs: MigrationCostBreakdown
    ) -> List[str]:
        """Generate key considerations for the migration."""
        
        considerations = []
        
        # Cost considerations
        if migration_costs.total_migration_cost > Decimal("100000"):
            considerations.append("High migration cost requires careful ROI analysis and executive approval")
        
        # Timeline considerations
        if complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.ENTERPRISE]:
            considerations.append("Complex migration requires dedicated project management and resources")
        
        # Technical considerations
        if workload.database:
            considerations.append("Database migration requires careful planning for data consistency and minimal downtime")
        
        if workload.compliance_requirements:
            considerations.append("Compliance requirements must be validated in target environment before migration")
        
        # Team considerations
        considerations.append("Team training should begin well before migration to ensure readiness")
        
        # Risk considerations
        considerations.append("Comprehensive backup and rollback procedures are essential")
        considerations.append("Pilot migration with non-critical workloads is recommended")
        
        return considerations
    
    async def close(self):
        """Close any open connections."""
        await self.tco_calculator.close()