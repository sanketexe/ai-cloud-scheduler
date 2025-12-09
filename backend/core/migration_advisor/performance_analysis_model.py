"""
Performance Analysis Model for Cloud Migration Advisor

This module implements the PerformanceAnalysisModel that evaluates cloud provider
performance capabilities against requirements.

Requirements: 3.1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from .provider_catalog import (
    CloudProvider, CloudProviderName, PerformanceCapability,
    ProviderCatalog
)
from .service_catalog_data import get_provider_catalog


class PerformanceMetric(Enum):
    """Performance metrics to evaluate"""
    COMPUTE_CAPACITY = "compute_capacity"
    STORAGE_CAPACITY = "storage_capacity"
    NETWORK_BANDWIDTH = "network_bandwidth"
    GPU_AVAILABILITY = "gpu_availability"
    AUTO_SCALING = "auto_scaling"
    LOAD_BALANCING = "load_balancing"
    AVAILABILITY_SLA = "availability_sla"
    LATENCY = "latency"


@dataclass
class PerformanceRequirement:
    """Individual performance requirement"""
    metric: PerformanceMetric
    required_value: float
    unit: str
    priority: str  # critical, high, medium, low
    description: str


@dataclass
class PerformanceMatch:
    """Match between requirement and provider capability"""
    requirement: PerformanceRequirement
    provider_capability: Optional[float]
    meets_requirement: bool
    performance_score: float  # 0.0 to 1.0
    match_details: str
    headroom_percentage: Optional[float] = None  # How much extra capacity
    limitations: List[str] = field(default_factory=list)


@dataclass
class PerformanceGap:
    """Gap in performance capability"""
    requirement: PerformanceRequirement
    gap_severity: str  # critical, high, medium, low
    impact_description: str
    workaround_options: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class ProviderPerformanceEvaluation:
    """Complete performance evaluation for a provider"""
    provider: CloudProvider
    performance_matches: List[PerformanceMatch]
    performance_gaps: List[PerformanceGap]
    overall_performance_score: float  # 0.0 to 1.0
    critical_requirements_met: bool
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceAnalysisModel:
    """
    ML-based performance analysis model that evaluates cloud provider
    performance capabilities against requirements.
    """
    
    def __init__(self, provider_catalog: Optional[ProviderCatalog] = None):
        self.catalog = provider_catalog or get_provider_catalog()
        
        # Metric importance weights
        self.metric_weights = {
            PerformanceMetric.COMPUTE_CAPACITY: 1.0,
            PerformanceMetric.STORAGE_CAPACITY: 0.9,
            PerformanceMetric.NETWORK_BANDWIDTH: 0.9,
            PerformanceMetric.GPU_AVAILABILITY: 0.8,
            PerformanceMetric.AUTO_SCALING: 0.8,
            PerformanceMetric.LOAD_BALANCING: 0.7,
            PerformanceMetric.AVAILABILITY_SLA: 1.0,
            PerformanceMetric.LATENCY: 0.9,
        }
    
    def evaluate_provider(
        self,
        provider_name: CloudProviderName,
        performance_requirements: List[PerformanceRequirement],
        availability_target: float = 99.9,
        latency_requirements: Optional[Dict[str, int]] = None
    ) -> ProviderPerformanceEvaluation:
        """
        Evaluate how well a provider meets performance requirements
        
        Args:
            provider_name: Cloud provider to evaluate
            performance_requirements: List of performance requirements
            availability_target: Target availability percentage
            latency_requirements: Optional latency requirements by region
            
        Returns:
            ProviderPerformanceEvaluation with detailed results
        """
        provider = self.catalog.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name.value} not found")
        
        # Match performance requirements
        performance_matches = []
        performance_gaps = []
        
        for requirement in performance_requirements:
            match = self._match_performance_requirement(provider, requirement)
            performance_matches.append(match)
            
            if not match.meets_requirement and requirement.priority in ["critical", "high"]:
                gap = self._create_performance_gap(requirement, match)
                performance_gaps.append(gap)
        
        # Calculate overall score
        overall_score = self._calculate_overall_performance_score(performance_matches)
        
        # Check if critical requirements are met
        critical_met = self._check_critical_requirements(performance_matches)
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_performance_strengths_weaknesses(
            provider, performance_matches, performance_gaps
        )
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            provider, performance_gaps
        )
        
        return ProviderPerformanceEvaluation(
            provider=provider,
            performance_matches=performance_matches,
            performance_gaps=performance_gaps,
            overall_performance_score=overall_score,
            critical_requirements_met=critical_met,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _match_performance_requirement(
        self,
        provider: CloudProvider,
        requirement: PerformanceRequirement
    ) -> PerformanceMatch:
        """
        Match a performance requirement to provider capability
        
        Args:
            provider: Cloud provider
            requirement: Performance requirement
            
        Returns:
            PerformanceMatch object
        """
        if not provider.performance_capabilities:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=None,
                meets_requirement=False,
                performance_score=0.0,
                match_details="Performance capabilities not available",
                limitations=["Provider performance data not available"]
            )
        
        perf_cap = provider.performance_capabilities
        
        # Match based on metric type
        if requirement.metric == PerformanceMetric.COMPUTE_CAPACITY:
            return self._match_compute_capacity(requirement, perf_cap)
        elif requirement.metric == PerformanceMetric.STORAGE_CAPACITY:
            return self._match_storage_capacity(requirement, perf_cap)
        elif requirement.metric == PerformanceMetric.NETWORK_BANDWIDTH:
            return self._match_network_bandwidth(requirement, perf_cap)
        elif requirement.metric == PerformanceMetric.GPU_AVAILABILITY:
            return self._match_gpu_availability(requirement, perf_cap)
        elif requirement.metric == PerformanceMetric.AUTO_SCALING:
            return self._match_auto_scaling(requirement, perf_cap)
        elif requirement.metric == PerformanceMetric.LOAD_BALANCING:
            return self._match_load_balancing(requirement, perf_cap)
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=None,
                meets_requirement=False,
                performance_score=0.5,
                match_details=f"Metric {requirement.metric.value} not evaluated"
            )
    
    def _match_compute_capacity(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match compute capacity requirement"""
        if perf_cap.max_compute_instances is None:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=None,
                meets_requirement=True,
                performance_score=0.8,
                match_details="Compute capacity effectively unlimited"
            )
        
        provider_capacity = float(perf_cap.max_compute_instances)
        required_capacity = requirement.required_value
        
        if provider_capacity >= required_capacity:
            headroom = ((provider_capacity - required_capacity) / required_capacity) * 100
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_capacity,
                meets_requirement=True,
                performance_score=1.0,
                match_details=f"Provider supports up to {provider_capacity} instances",
                headroom_percentage=headroom
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_capacity,
                meets_requirement=False,
                performance_score=provider_capacity / required_capacity,
                match_details=f"Provider capacity ({provider_capacity}) below requirement ({required_capacity})",
                limitations=["May need to request quota increase"]
            )
    
    def _match_storage_capacity(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match storage capacity requirement"""
        if perf_cap.max_storage_capacity_tb is None:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=None,
                meets_requirement=True,
                performance_score=0.9,
                match_details="Storage capacity effectively unlimited"
            )
        
        provider_capacity = float(perf_cap.max_storage_capacity_tb)
        required_capacity = requirement.required_value
        
        if provider_capacity >= required_capacity:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_capacity,
                meets_requirement=True,
                performance_score=1.0,
                match_details=f"Provider supports up to {provider_capacity} TB storage"
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_capacity,
                meets_requirement=False,
                performance_score=provider_capacity / required_capacity,
                match_details=f"Storage capacity may be insufficient"
            )
    
    def _match_network_bandwidth(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match network bandwidth requirement"""
        if perf_cap.max_network_bandwidth_gbps is None:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=None,
                meets_requirement=True,
                performance_score=0.8,
                match_details="Network bandwidth scales with instance type"
            )
        
        provider_bandwidth = float(perf_cap.max_network_bandwidth_gbps)
        required_bandwidth = requirement.required_value
        
        if provider_bandwidth >= required_bandwidth:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_bandwidth,
                meets_requirement=True,
                performance_score=1.0,
                match_details=f"Provider supports up to {provider_bandwidth} Gbps"
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=provider_bandwidth,
                meets_requirement=False,
                performance_score=provider_bandwidth / required_bandwidth,
                match_details=f"Network bandwidth may be insufficient"
            )
    
    def _match_gpu_availability(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match GPU availability requirement"""
        if perf_cap.gpu_availability:
            gpu_types = len(perf_cap.gpu_types) if perf_cap.gpu_types else 1
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=float(gpu_types),
                meets_requirement=True,
                performance_score=1.0,
                match_details=f"GPU available: {', '.join(perf_cap.gpu_types) if perf_cap.gpu_types else 'Yes'}"
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=0.0,
                meets_requirement=False,
                performance_score=0.0,
                match_details="GPU not available",
                limitations=["No GPU support"]
            )
    
    def _match_auto_scaling(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match auto-scaling requirement"""
        if perf_cap.auto_scaling_capabilities:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=1.0,
                meets_requirement=True,
                performance_score=1.0,
                match_details="Auto-scaling supported"
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=0.0,
                meets_requirement=False,
                performance_score=0.0,
                match_details="Auto-scaling not available"
            )
    
    def _match_load_balancing(
        self,
        requirement: PerformanceRequirement,
        perf_cap: PerformanceCapability
    ) -> PerformanceMatch:
        """Match load balancing requirement"""
        if perf_cap.load_balancing_options:
            options_count = len(perf_cap.load_balancing_options)
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=float(options_count),
                meets_requirement=True,
                performance_score=1.0,
                match_details=f"Load balancing available: {', '.join(perf_cap.load_balancing_options)}"
            )
        else:
            return PerformanceMatch(
                requirement=requirement,
                provider_capability=0.0,
                meets_requirement=False,
                performance_score=0.5,
                match_details="Load balancing options not specified"
            )
    
    def _calculate_overall_performance_score(
        self,
        performance_matches: List[PerformanceMatch]
    ) -> float:
        """Calculate overall performance score"""
        if not performance_matches:
            return 0.0
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for match in performance_matches:
            weight = self.metric_weights.get(match.requirement.metric, 1.0)
            
            # Adjust weight by priority
            if match.requirement.priority == "critical":
                weight *= 1.5
            elif match.requirement.priority == "high":
                weight *= 1.2
            elif match.requirement.priority == "low":
                weight *= 0.8
            
            weighted_score += match.performance_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _check_critical_requirements(
        self,
        performance_matches: List[PerformanceMatch]
    ) -> bool:
        """Check if all critical requirements are met"""
        for match in performance_matches:
            if match.requirement.priority == "critical" and not match.meets_requirement:
                return False
        return True
    
    def _create_performance_gap(
        self,
        requirement: PerformanceRequirement,
        match: PerformanceMatch
    ) -> PerformanceGap:
        """Create a performance gap"""
        return PerformanceGap(
            requirement=requirement,
            gap_severity=requirement.priority,
            impact_description=f"Performance requirement not met: {requirement.description}",
            workaround_options=[
                "Request quota increase from provider",
                "Optimize workload to reduce requirements",
                "Consider distributed architecture"
            ],
            alternative_approaches=[
                "Use multiple regions for capacity",
                "Implement caching to reduce load"
            ]
        )
    
    def _analyze_performance_strengths_weaknesses(
        self,
        provider: CloudProvider,
        performance_matches: List[PerformanceMatch],
        performance_gaps: List[PerformanceGap]
    ) -> tuple[List[str], List[str]]:
        """Analyze performance strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze matches
        met_requirements = [m for m in performance_matches if m.meets_requirement]
        if len(met_requirements) == len(performance_matches):
            strengths.append("All performance requirements met")
        elif len(met_requirements) >= len(performance_matches) * 0.8:
            strengths.append(f"{len(met_requirements)} of {len(performance_matches)} requirements met")
        
        # Analyze gaps
        critical_gaps = [g for g in performance_gaps if g.gap_severity == "critical"]
        if critical_gaps:
            weaknesses.append(f"{len(critical_gaps)} critical performance gaps")
        
        # Provider-specific strengths
        if provider.performance_capabilities:
            perf_cap = provider.performance_capabilities
            if perf_cap.gpu_availability:
                strengths.append("GPU compute available")
            if perf_cap.auto_scaling_capabilities:
                strengths.append("Auto-scaling supported")
        
        return strengths, weaknesses
    
    def _generate_performance_recommendations(
        self,
        provider: CloudProvider,
        performance_gaps: List[PerformanceGap]
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not performance_gaps:
            recommendations.append("Provider meets all performance requirements")
        else:
            critical_gaps = [g for g in performance_gaps if g.gap_severity == "critical"]
            if critical_gaps:
                recommendations.append("Address critical performance gaps before migration")
            else:
                recommendations.append("Monitor performance during migration")
        
        return recommendations
    
    def compare_providers_performance(
        self,
        performance_requirements: List[PerformanceRequirement],
        providers: Optional[List[CloudProviderName]] = None
    ) -> Dict[CloudProviderName, ProviderPerformanceEvaluation]:
        """Compare performance across multiple providers"""
        if providers is None:
            providers = [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]
        
        evaluations = {}
        for provider_name in providers:
            evaluation = self.evaluate_provider(
                provider_name=provider_name,
                performance_requirements=performance_requirements
            )
            evaluations[provider_name] = evaluation
        
        return evaluations
