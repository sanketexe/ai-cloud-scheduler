"""
Migration Complexity Calculator for Cloud Migration Advisor

This module implements the migration complexity scoring algorithm that assesses
the difficulty and risk of migrating to different cloud providers.

Requirements: 3.1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from decimal import Decimal

from .provider_catalog import CloudProvider, CloudProviderName, ProviderCatalog
from .service_catalog_data import get_provider_catalog


class MigrationComplexityLevel(Enum):
    """Migration complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskLevel(Enum):
    """Risk levels for migration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplexityFactor:
    """Individual complexity factor"""
    factor_name: str
    factor_category: str  # infrastructure, data, application, integration
    complexity_score: float  # 0.0 to 1.0
    impact_level: str  # low, medium, high
    description: str
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class MigrationRisk:
    """Individual migration risk"""
    risk_name: str
    risk_category: str  # technical, operational, security, compliance
    risk_level: RiskLevel
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    description: str
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class MigrationComplexityAssessment:
    """Complete migration complexity assessment"""
    provider: CloudProvider
    source_infrastructure: str  # on_premises, aws, gcp, azure, hybrid
    overall_complexity_score: float  # 0.0 to 1.0
    complexity_level: MigrationComplexityLevel
    overall_risk_level: RiskLevel
    estimated_duration_weeks: int
    estimated_effort_hours: int
    complexity_factors: List[ComplexityFactor] = field(default_factory=list)
    migration_risks: List[MigrationRisk] = field(default_factory=list)
    key_challenges: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MigrationComplexityCalculator:
    """
    Calculator that assesses migration complexity and risk based on
    current infrastructure and target cloud provider.
    """
    
    def __init__(self, provider_catalog: Optional[ProviderCatalog] = None):
        self.catalog = provider_catalog or get_provider_catalog()
        
        # Complexity weights by category
        self.category_weights = {
            "infrastructure": 1.0,
            "data": 1.2,
            "application": 1.0,
            "integration": 0.9,
            "security": 1.1,
            "compliance": 1.0,
        }
        
        # Base complexity by source infrastructure
        self.base_complexity = {
            "on_premises": 0.7,
            "aws": 0.3,
            "gcp": 0.3,
            "azure": 0.3,
            "hybrid": 0.6,
            "multi_cloud": 0.8,
        }
    
    def calculate_complexity(
        self,
        provider_name: CloudProviderName,
        source_infrastructure: str,
        workload_count: int = 1,
        data_volume_tb: float = 1.0,
        has_databases: bool = False,
        has_legacy_apps: bool = False,
        integration_count: int = 0,
        compliance_requirements: Optional[List[str]] = None
    ) -> MigrationComplexityAssessment:
        """
        Calculate migration complexity for a provider
        
        Args:
            provider_name: Target cloud provider
            source_infrastructure: Current infrastructure type
            workload_count: Number of workloads to migrate
            data_volume_tb: Total data volume in TB
            has_databases: Whether databases need migration
            has_legacy_apps: Whether legacy applications exist
            integration_count: Number of integrations
            compliance_requirements: Optional compliance requirements
            
        Returns:
            MigrationComplexityAssessment with detailed analysis
        """
        provider = self.catalog.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name.value} not found")
        
        # Calculate complexity factors
        complexity_factors = []
        
        # Infrastructure complexity
        infra_factor = self._calculate_infrastructure_complexity(
            source_infrastructure, workload_count
        )
        complexity_factors.append(infra_factor)
        
        # Data migration complexity
        data_factor = self._calculate_data_complexity(
            data_volume_tb, has_databases
        )
        complexity_factors.append(data_factor)
        
        # Application complexity
        app_factor = self._calculate_application_complexity(
            has_legacy_apps, workload_count
        )
        complexity_factors.append(app_factor)
        
        # Integration complexity
        if integration_count > 0:
            integration_factor = self._calculate_integration_complexity(
                integration_count
            )
            complexity_factors.append(integration_factor)
        
        # Compliance complexity
        if compliance_requirements:
            compliance_factor = self._calculate_compliance_complexity(
                len(compliance_requirements)
            )
            complexity_factors.append(compliance_factor)
        
        # Calculate overall complexity score
        overall_score = self._calculate_overall_complexity_score(
            complexity_factors, source_infrastructure
        )
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(overall_score)
        
        # Identify migration risks
        migration_risks = self._identify_migration_risks(
            source_infrastructure, complexity_factors, provider
        )
        
        # Calculate overall risk level
        overall_risk = self._calculate_overall_risk_level(migration_risks)
        
        # Estimate duration and effort
        duration_weeks = self._estimate_duration(
            overall_score, workload_count, data_volume_tb
        )
        effort_hours = self._estimate_effort(
            overall_score, workload_count, data_volume_tb
        )
        
        # Identify key challenges
        key_challenges = self._identify_key_challenges(
            complexity_factors, migration_risks
        )
        
        # Identify success factors
        success_factors = self._identify_success_factors(
            provider, source_infrastructure
        )
        
        # Generate recommendations
        recommendations = self._generate_complexity_recommendations(
            complexity_level, migration_risks, source_infrastructure
        )
        
        return MigrationComplexityAssessment(
            provider=provider,
            source_infrastructure=source_infrastructure,
            overall_complexity_score=overall_score,
            complexity_level=complexity_level,
            overall_risk_level=overall_risk,
            estimated_duration_weeks=duration_weeks,
            estimated_effort_hours=effort_hours,
            complexity_factors=complexity_factors,
            migration_risks=migration_risks,
            key_challenges=key_challenges,
            success_factors=success_factors,
            recommendations=recommendations
        )
    
    def _calculate_infrastructure_complexity(
        self,
        source_infrastructure: str,
        workload_count: int
    ) -> ComplexityFactor:
        """Calculate infrastructure migration complexity"""
        base_score = self.base_complexity.get(source_infrastructure, 0.5)
        
        # Adjust for workload count
        if workload_count > 50:
            base_score += 0.2
        elif workload_count > 20:
            base_score += 0.1
        
        base_score = min(base_score, 1.0)
        
        impact = "high" if base_score > 0.7 else "medium" if base_score > 0.4 else "low"
        
        return ComplexityFactor(
            factor_name="Infrastructure Migration",
            factor_category="infrastructure",
            complexity_score=base_score,
            impact_level=impact,
            description=f"Migrating {workload_count} workloads from {source_infrastructure}",
            mitigation_strategies=[
                "Use phased migration approach",
                "Implement automated migration tools",
                "Conduct pilot migration first"
            ]
        )
    
    def _calculate_data_complexity(
        self,
        data_volume_tb: float,
        has_databases: bool
    ) -> ComplexityFactor:
        """Calculate data migration complexity"""
        score = 0.3
        
        # Adjust for data volume
        if data_volume_tb > 100:
            score += 0.4
        elif data_volume_tb > 10:
            score += 0.2
        elif data_volume_tb > 1:
            score += 0.1
        
        # Adjust for databases
        if has_databases:
            score += 0.2
        
        score = min(score, 1.0)
        
        impact = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
        
        strategies = [
            "Use data transfer appliances for large volumes",
            "Implement incremental data sync",
            "Plan for minimal downtime window"
        ]
        
        if has_databases:
            strategies.append("Use database migration services")
            strategies.append("Test database replication thoroughly")
        
        return ComplexityFactor(
            factor_name="Data Migration",
            factor_category="data",
            complexity_score=score,
            impact_level=impact,
            description=f"Migrating {data_volume_tb} TB of data" + (" including databases" if has_databases else ""),
            mitigation_strategies=strategies
        )
    
    def _calculate_application_complexity(
        self,
        has_legacy_apps: bool,
        workload_count: int
    ) -> ComplexityFactor:
        """Calculate application migration complexity"""
        score = 0.3
        
        if has_legacy_apps:
            score += 0.3
        
        if workload_count > 30:
            score += 0.2
        elif workload_count > 10:
            score += 0.1
        
        score = min(score, 1.0)
        
        impact = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
        
        strategies = [
            "Assess application dependencies",
            "Modernize applications where possible",
            "Use containerization for portability"
        ]
        
        if has_legacy_apps:
            strategies.append("Consider refactoring legacy applications")
            strategies.append("Evaluate lift-and-shift vs replatforming")
        
        return ComplexityFactor(
            factor_name="Application Migration",
            factor_category="application",
            complexity_score=score,
            impact_level=impact,
            description="Migrating applications" + (" including legacy systems" if has_legacy_apps else ""),
            mitigation_strategies=strategies
        )
    
    def _calculate_integration_complexity(
        self,
        integration_count: int
    ) -> ComplexityFactor:
        """Calculate integration complexity"""
        score = 0.2 + (integration_count * 0.05)
        score = min(score, 1.0)
        
        impact = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
        
        return ComplexityFactor(
            factor_name="Integration Migration",
            factor_category="integration",
            complexity_score=score,
            impact_level=impact,
            description=f"Migrating {integration_count} integrations",
            mitigation_strategies=[
                "Map all integration points",
                "Test integrations in staging environment",
                "Implement API gateways for flexibility"
            ]
        )
    
    def _calculate_compliance_complexity(
        self,
        compliance_count: int
    ) -> ComplexityFactor:
        """Calculate compliance complexity"""
        score = 0.3 + (compliance_count * 0.1)
        score = min(score, 1.0)
        
        impact = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
        
        return ComplexityFactor(
            factor_name="Compliance Requirements",
            factor_category="compliance",
            complexity_score=score,
            impact_level=impact,
            description=f"Meeting {compliance_count} compliance requirements",
            mitigation_strategies=[
                "Verify provider compliance certifications",
                "Implement compliance monitoring",
                "Document compliance controls"
            ]
        )
    
    def _calculate_overall_complexity_score(
        self,
        complexity_factors: List[ComplexityFactor],
        source_infrastructure: str
    ) -> float:
        """Calculate overall complexity score"""
        if not complexity_factors:
            return 0.5
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for factor in complexity_factors:
            weight = self.category_weights.get(factor.factor_category, 1.0)
            weighted_score += factor.complexity_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _determine_complexity_level(self, score: float) -> MigrationComplexityLevel:
        """Determine complexity level from score"""
        if score >= 0.8:
            return MigrationComplexityLevel.VERY_HIGH
        elif score >= 0.6:
            return MigrationComplexityLevel.HIGH
        elif score >= 0.4:
            return MigrationComplexityLevel.MEDIUM
        else:
            return MigrationComplexityLevel.LOW
    
    def _identify_migration_risks(
        self,
        source_infrastructure: str,
        complexity_factors: List[ComplexityFactor],
        provider: CloudProvider
    ) -> List[MigrationRisk]:
        """Identify migration risks"""
        risks = []
        
        # Data loss risk
        data_factors = [f for f in complexity_factors if f.factor_category == "data"]
        if data_factors and data_factors[0].complexity_score > 0.5:
            risks.append(MigrationRisk(
                risk_name="Data Loss or Corruption",
                risk_category="technical",
                risk_level=RiskLevel.HIGH,
                probability=0.3,
                impact=0.9,
                description="Risk of data loss during migration",
                mitigation_actions=[
                    "Implement comprehensive backup strategy",
                    "Use checksums to verify data integrity",
                    "Test data migration in non-production environment"
                ]
            ))
        
        # Downtime risk
        risks.append(MigrationRisk(
            risk_name="Extended Downtime",
            risk_category="operational",
            risk_level=RiskLevel.MEDIUM,
            probability=0.4,
            impact=0.7,
            description="Risk of extended service downtime",
            mitigation_actions=[
                "Plan migration during maintenance windows",
                "Implement blue-green deployment",
                "Prepare rollback procedures"
            ]
        ))
        
        # Compatibility risk
        if source_infrastructure in ["on_premises", "hybrid"]:
            risks.append(MigrationRisk(
                risk_name="Application Compatibility",
                risk_category="technical",
                risk_level=RiskLevel.MEDIUM,
                probability=0.5,
                impact=0.6,
                description="Applications may not be compatible with cloud environment",
                mitigation_actions=[
                    "Conduct compatibility assessment",
                    "Refactor incompatible applications",
                    "Use containers for better portability"
                ]
            ))
        
        return risks
    
    def _calculate_overall_risk_level(self, risks: List[MigrationRisk]) -> RiskLevel:
        """Calculate overall risk level"""
        if not risks:
            return RiskLevel.LOW
        
        # Define risk level ordering
        risk_order = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        # Find the highest risk level
        max_risk_value = max(risk_order[risk.risk_level] for risk in risks)
        
        # Return the corresponding risk level
        for risk_level, value in risk_order.items():
            if value == max_risk_value:
                return risk_level
        
        return RiskLevel.LOW
    
    def _estimate_duration(
        self,
        complexity_score: float,
        workload_count: int,
        data_volume_tb: float
    ) -> int:
        """Estimate migration duration in weeks"""
        base_weeks = 4
        
        # Adjust for complexity
        base_weeks += int(complexity_score * 12)
        
        # Adjust for workload count
        base_weeks += (workload_count // 10) * 2
        
        # Adjust for data volume
        if data_volume_tb > 50:
            base_weeks += 4
        elif data_volume_tb > 10:
            base_weeks += 2
        
        return max(base_weeks, 2)
    
    def _estimate_effort(
        self,
        complexity_score: float,
        workload_count: int,
        data_volume_tb: float
    ) -> int:
        """Estimate migration effort in hours"""
        base_hours = 160  # 1 person-month
        
        # Adjust for complexity
        base_hours += int(complexity_score * 480)
        
        # Adjust for workload count
        base_hours += workload_count * 40
        
        # Adjust for data volume
        base_hours += int(data_volume_tb * 10)
        
        return max(base_hours, 80)
    
    def _identify_key_challenges(
        self,
        complexity_factors: List[ComplexityFactor],
        migration_risks: List[MigrationRisk]
    ) -> List[str]:
        """Identify key migration challenges"""
        challenges = []
        
        # High complexity factors
        high_factors = [f for f in complexity_factors if f.complexity_score > 0.6]
        for factor in high_factors:
            challenges.append(f"{factor.factor_name}: {factor.description}")
        
        # High risks
        high_risks = [r for r in migration_risks if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        for risk in high_risks:
            challenges.append(f"Risk: {risk.risk_name}")
        
        return challenges[:5]  # Top 5 challenges
    
    def _identify_success_factors(
        self,
        provider: CloudProvider,
        source_infrastructure: str
    ) -> List[str]:
        """Identify success factors"""
        factors = []
        
        factors.append("Comprehensive migration planning")
        factors.append("Experienced migration team")
        factors.append("Adequate testing and validation")
        
        if source_infrastructure == provider.provider_name.value:
            factors.append("Same-provider migration reduces complexity")
        
        return factors
    
    def _generate_complexity_recommendations(
        self,
        complexity_level: MigrationComplexityLevel,
        migration_risks: List[MigrationRisk],
        source_infrastructure: str
    ) -> List[str]:
        """Generate recommendations based on complexity"""
        recommendations = []
        
        if complexity_level in [MigrationComplexityLevel.HIGH, MigrationComplexityLevel.VERY_HIGH]:
            recommendations.append("Consider engaging migration specialists")
            recommendations.append("Implement phased migration approach")
            recommendations.append("Allocate additional time and resources")
        else:
            recommendations.append("Standard migration approach suitable")
        
        high_risks = [r for r in migration_risks if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risks:
            recommendations.append("Address high-risk items before migration")
        
        return recommendations
    
    def compare_providers_complexity(
        self,
        source_infrastructure: str,
        workload_count: int = 1,
        data_volume_tb: float = 1.0,
        providers: Optional[List[CloudProviderName]] = None
    ) -> Dict[CloudProviderName, MigrationComplexityAssessment]:
        """Compare migration complexity across providers"""
        if providers is None:
            providers = [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]
        
        assessments = {}
        for provider_name in providers:
            assessment = self.calculate_complexity(
                provider_name=provider_name,
                source_infrastructure=source_infrastructure,
                workload_count=workload_count,
                data_volume_tb=data_volume_tb
            )
            assessments[provider_name] = assessment
        
        return assessments
