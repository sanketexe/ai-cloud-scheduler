"""
ML-Based Recommendation Engine for Cloud Migration Advisor

This module implements the complete recommendation engine that aggregates
scores from multiple models to generate provider recommendations.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from decimal import Decimal
from enum import Enum

from .provider_catalog import CloudProvider, CloudProviderName, ProviderCatalog
from .service_catalog_data import get_provider_catalog
from .service_matching_engine import ServiceMatchingModel, ProviderServiceEvaluation
from .cost_prediction_model import CostPredictionModel, CostScore
from .compliance_evaluation_model import ComplianceEvaluationModel, ProviderComplianceEvaluation
from .performance_analysis_model import PerformanceAnalysisModel, ProviderPerformanceEvaluation
from .migration_complexity_calculator import MigrationComplexityCalculator, MigrationComplexityAssessment


@dataclass
class ScoringWeights:
    """Configurable weights for recommendation scoring"""
    service_availability_weight: float = 0.25
    pricing_weight: float = 0.20
    compliance_weight: float = 0.20
    technical_fit_weight: float = 0.15
    migration_complexity_weight: float = 0.20
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (
            self.service_availability_weight +
            self.pricing_weight +
            self.compliance_weight +
            self.technical_fit_weight +
            self.migration_complexity_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "service_availability": self.service_availability_weight,
            "pricing": self.pricing_weight,
            "compliance": self.compliance_weight,
            "technical_fit": self.technical_fit_weight,
            "migration_complexity": self.migration_complexity_weight
        }


@dataclass
class ProviderScore:
    """Aggregated score for a provider"""
    provider: CloudProvider
    overall_score: float  # 0.0 to 1.0
    service_availability_score: float
    pricing_score: float
    compliance_score: float
    technical_fit_score: float
    migration_complexity_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self):
        return f"<ProviderScore(provider='{self.provider.provider_name.value}', score={self.overall_score:.3f})>"


@dataclass
class ProviderRecommendation:
    """Complete recommendation for a provider"""
    provider: CloudProvider
    rank: int
    overall_score: float
    confidence_score: float  # 0.0 to 1.0
    justification: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    key_differentiators: List[str] = field(default_factory=list)
    estimated_monthly_cost: Optional[Decimal] = None
    migration_duration_weeks: Optional[int] = None
    
    def __repr__(self):
        return f"<ProviderRecommendation(rank={self.rank}, provider='{self.provider.provider_name.value}', score={self.overall_score:.3f})>"


@dataclass
class ComparisonMatrix:
    """Side-by-side provider comparison"""
    providers: List[CloudProvider]
    service_comparison: Dict[CloudProviderName, float]
    cost_comparison: Dict[CloudProviderName, Decimal]
    compliance_comparison: Dict[CloudProviderName, float]
    performance_comparison: Dict[CloudProviderName, float]
    complexity_comparison: Dict[CloudProviderName, float]
    key_differences: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<ComparisonMatrix(providers={len(self.providers)})>"


@dataclass
class RecommendationReport:
    """Complete recommendation report"""
    primary_recommendation: ProviderRecommendation
    alternative_recommendations: List[ProviderRecommendation]
    comparison_matrix: ComparisonMatrix
    scoring_weights: ScoringWeights
    overall_confidence: float
    key_findings: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"<RecommendationReport(primary='{self.primary_recommendation.provider.provider_name.value}', confidence={self.overall_confidence:.3f})>"


class RecommendationEngine:
    """
    ML-based recommendation engine that aggregates multiple scoring models
    to generate comprehensive cloud provider recommendations.
    """
    
    def __init__(self, provider_catalog: Optional[ProviderCatalog] = None):
        """Initialize the recommendation engine"""
        self.catalog = provider_catalog or get_provider_catalog()
        
        # Initialize component models
        self.service_matcher = ServiceMatchingModel(self.catalog)
        self.cost_predictor = CostPredictionModel(self.catalog)
        self.compliance_evaluator = ComplianceEvaluationModel(self.catalog)
        self.performance_analyzer = PerformanceAnalysisModel(self.catalog)
        self.complexity_calculator = MigrationComplexityCalculator(self.catalog)
        
        # Default scoring weights
        self.default_weights = ScoringWeights()
    
    def generate_recommendations(
        self,
        required_services: List[str],
        target_monthly_budget: Decimal,
        compliance_requirements: List[str],
        source_infrastructure: str = "on_premises",
        workload_specs: Optional[List[Dict]] = None,
        performance_requirements: Optional[List] = None,
        data_residency_requirements: Optional[List[str]] = None,
        scoring_weights: Optional[ScoringWeights] = None,
        providers: Optional[List[CloudProviderName]] = None
    ) -> RecommendationReport:
        """
        Generate comprehensive provider recommendations
        
        Args:
            required_services: List of required cloud services
            target_monthly_budget: Target monthly budget
            compliance_requirements: List of compliance frameworks
            source_infrastructure: Current infrastructure type
            workload_specs: Optional workload specifications for cost estimation
            performance_requirements: Optional performance requirements
            data_residency_requirements: Optional data residency requirements
            scoring_weights: Optional custom scoring weights
            providers: Optional list of providers to evaluate
            
        Returns:
            RecommendationReport with ranked recommendations
        """
        if providers is None:
            providers = [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]
        
        if scoring_weights is None:
            scoring_weights = self.default_weights
        
        # Evaluate all providers
        provider_scores = {}
        service_evaluations = {}
        cost_scores = {}
        compliance_evaluations = {}
        performance_evaluations = {}
        complexity_assessments = {}
        
        for provider_name in providers:
            # Service matching
            service_eval = self.service_matcher.evaluate_provider(
                provider_name, required_services
            )
            service_evaluations[provider_name] = service_eval
            
            # Cost prediction
            if workload_specs:
                cost_comparisons = self.cost_predictor.compare_provider_costs(
                    workload_specs, [provider_name]
                )
                cost_comparison = cost_comparisons[provider_name]
                cost_score = self.cost_predictor.calculate_cost_score(
                    provider_name=provider_name,
                    estimated_monthly_cost=cost_comparison.total_monthly_cost,
                    target_monthly_budget=target_monthly_budget,
                    service_quality_score=service_eval.overall_match_score
                )
                cost_scores[provider_name] = cost_score
            else:
                # Default cost score
                cost_scores[provider_name] = None
            
            # Compliance evaluation
            compliance_eval = self.compliance_evaluator.evaluate_provider(
                provider_name=provider_name,
                required_frameworks=compliance_requirements,
                data_residency_requirements=data_residency_requirements
            )
            compliance_evaluations[provider_name] = compliance_eval
            
            # Performance analysis
            if performance_requirements:
                perf_eval = self.performance_analyzer.evaluate_provider(
                    provider_name=provider_name,
                    performance_requirements=performance_requirements
                )
                performance_evaluations[provider_name] = perf_eval
            else:
                performance_evaluations[provider_name] = None
            
            # Migration complexity
            complexity_assessment = self.complexity_calculator.calculate_complexity(
                provider_name=provider_name,
                source_infrastructure=source_infrastructure,
                workload_count=len(workload_specs) if workload_specs else 1
            )
            complexity_assessments[provider_name] = complexity_assessment
            
            # Calculate aggregated score
            provider_score = self._calculate_provider_score(
                provider_name=provider_name,
                service_eval=service_eval,
                cost_score=cost_scores[provider_name],
                compliance_eval=compliance_eval,
                perf_eval=performance_evaluations[provider_name],
                complexity_assessment=complexity_assessment,
                weights=scoring_weights
            )
            provider_scores[provider_name] = provider_score
        
        # Rank providers
        ranked_providers = sorted(
            provider_scores.values(),
            key=lambda x: x.overall_score,
            reverse=True
        )
        
        # Generate recommendations
        recommendations = []
        for rank, provider_score in enumerate(ranked_providers, 1):
            recommendation = self._generate_provider_recommendation(
                rank=rank,
                provider_score=provider_score,
                service_eval=service_evaluations[provider_score.provider.provider_name],
                cost_score=cost_scores[provider_score.provider.provider_name],
                compliance_eval=compliance_evaluations[provider_score.provider.provider_name],
                complexity_assessment=complexity_assessments[provider_score.provider.provider_name]
            )
            recommendations.append(recommendation)
        
        # Generate comparison matrix
        comparison_matrix = self._generate_comparison_matrix(
            provider_scores=provider_scores,
            cost_scores=cost_scores,
            compliance_evaluations=compliance_evaluations,
            complexity_assessments=complexity_assessments
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            recommendations[0], provider_scores
        )
        
        # Generate key findings
        key_findings = self._generate_key_findings(
            recommendations, comparison_matrix
        )
        
        return RecommendationReport(
            primary_recommendation=recommendations[0],
            alternative_recommendations=recommendations[1:],
            comparison_matrix=comparison_matrix,
            scoring_weights=scoring_weights,
            overall_confidence=overall_confidence,
            key_findings=key_findings
        )
    
    def _calculate_provider_score(
        self,
        provider_name: CloudProviderName,
        service_eval: ProviderServiceEvaluation,
        cost_score: Optional[CostScore],
        compliance_eval: ProviderComplianceEvaluation,
        perf_eval: Optional[ProviderPerformanceEvaluation],
        complexity_assessment: MigrationComplexityAssessment,
        weights: ScoringWeights
    ) -> ProviderScore:
        """Calculate aggregated provider score"""
        provider = self.catalog.get_provider(provider_name)
        
        # Component scores
        service_score = service_eval.overall_match_score
        pricing_score = cost_score.cost_score if cost_score else 0.7  # Default
        compliance_score = compliance_eval.overall_compliance_score
        technical_score = perf_eval.overall_performance_score if perf_eval else 0.8  # Default
        
        # Migration complexity score (inverse - lower complexity = higher score)
        complexity_score = 1.0 - complexity_assessment.overall_complexity_score
        
        # Calculate weighted overall score
        overall_score = (
            service_score * weights.service_availability_weight +
            pricing_score * weights.pricing_weight +
            compliance_score * weights.compliance_weight +
            technical_score * weights.technical_fit_weight +
            complexity_score * weights.migration_complexity_weight
        )
        
        return ProviderScore(
            provider=provider,
            overall_score=overall_score,
            service_availability_score=service_score,
            pricing_score=pricing_score,
            compliance_score=compliance_score,
            technical_fit_score=technical_score,
            migration_complexity_score=complexity_score,
            component_scores={
                "service_availability": service_score,
                "pricing": pricing_score,
                "compliance": compliance_score,
                "technical_fit": technical_score,
                "migration_complexity": complexity_score
            }
        )
    
    def _generate_provider_recommendation(
        self,
        rank: int,
        provider_score: ProviderScore,
        service_eval: ProviderServiceEvaluation,
        cost_score: Optional[CostScore],
        compliance_eval: ProviderComplianceEvaluation,
        complexity_assessment: MigrationComplexityAssessment
    ) -> ProviderRecommendation:
        """Generate recommendation for a provider"""
        # Aggregate strengths
        strengths = []
        strengths.extend(service_eval.strengths[:2])
        strengths.extend(compliance_eval.strengths[:2])
        if cost_score and cost_score.cost_efficiency_rating in ["excellent", "good"]:
            strengths.append(f"Cost-effective: {cost_score.cost_efficiency_rating}")
        
        # Aggregate weaknesses
        weaknesses = []
        weaknesses.extend(service_eval.weaknesses[:2])
        weaknesses.extend(compliance_eval.weaknesses[:2])
        if complexity_assessment.complexity_level.value in ["high", "very_high"]:
            weaknesses.append(f"Migration complexity: {complexity_assessment.complexity_level.value}")
        
        # Key differentiators
        differentiators = []
        if provider_score.service_availability_score >= 0.9:
            differentiators.append("Excellent service coverage")
        if provider_score.compliance_score >= 0.9:
            differentiators.append("Strong compliance support")
        if provider_score.pricing_score >= 0.85:
            differentiators.append("Cost-effective pricing")
        
        # Generate justification
        justification = self._generate_justification(
            rank, provider_score, service_eval, cost_score, compliance_eval
        )
        
        # Calculate confidence
        confidence = self._calculate_recommendation_confidence(provider_score)
        
        return ProviderRecommendation(
            provider=provider_score.provider,
            rank=rank,
            overall_score=provider_score.overall_score,
            confidence_score=confidence,
            justification=justification,
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
            key_differentiators=differentiators,
            estimated_monthly_cost=cost_score.estimated_monthly_cost if cost_score else None,
            migration_duration_weeks=complexity_assessment.estimated_duration_weeks
        )
    
    def _generate_justification(
        self,
        rank: int,
        provider_score: ProviderScore,
        service_eval: ProviderServiceEvaluation,
        cost_score: Optional[CostScore],
        compliance_eval: ProviderComplianceEvaluation
    ) -> str:
        """Generate justification text"""
        provider_name = provider_score.provider.display_name
        
        if rank == 1:
            justification = f"{provider_name} is the top recommendation with an overall score of {provider_score.overall_score:.2f}. "
        else:
            justification = f"{provider_name} ranks #{rank} with an overall score of {provider_score.overall_score:.2f}. "
        
        # Add key reasons
        reasons = []
        if provider_score.service_availability_score >= 0.85:
            reasons.append(f"excellent service coverage ({provider_score.service_availability_score:.0%})")
        if provider_score.compliance_score >= 0.85:
            reasons.append(f"strong compliance support ({provider_score.compliance_score:.0%})")
        if cost_score and provider_score.pricing_score >= 0.80:
            reasons.append(f"cost-effective pricing")
        
        if reasons:
            justification += "Key strengths include " + ", ".join(reasons) + ". "
        
        # Add service gaps if any
        if service_eval.service_gaps:
            justification += f"Note: {len(service_eval.service_gaps)} service gaps identified. "
        
        return justification
    
    def _calculate_recommendation_confidence(self, provider_score: ProviderScore) -> float:
        """Calculate confidence score for recommendation"""
        # Base confidence on score consistency
        scores = [
            provider_score.service_availability_score,
            provider_score.pricing_score,
            provider_score.compliance_score,
            provider_score.technical_fit_score,
            provider_score.migration_complexity_score
        ]
        
        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher confidence
        confidence = 1.0 - min(variance * 2, 0.5)
        
        # Adjust for overall score
        confidence = (confidence * 0.7) + (provider_score.overall_score * 0.3)
        
        return confidence
    
    def _generate_comparison_matrix(
        self,
        provider_scores: Dict[CloudProviderName, ProviderScore],
        cost_scores: Dict[CloudProviderName, Optional[CostScore]],
        compliance_evaluations: Dict[CloudProviderName, ProviderComplianceEvaluation],
        complexity_assessments: Dict[CloudProviderName, MigrationComplexityAssessment]
    ) -> ComparisonMatrix:
        """Generate side-by-side comparison matrix"""
        providers = [score.provider for score in provider_scores.values()]
        
        service_comparison = {
            name: score.service_availability_score
            for name, score in provider_scores.items()
        }
        
        cost_comparison = {
            name: cost_scores[name].estimated_monthly_cost if cost_scores[name] else Decimal("0")
            for name in provider_scores.keys()
        }
        
        compliance_comparison = {
            name: score.compliance_score
            for name, score in provider_scores.items()
        }
        
        performance_comparison = {
            name: score.technical_fit_score
            for name, score in provider_scores.items()
        }
        
        complexity_comparison = {
            name: complexity_assessments[name].overall_complexity_score
            for name in provider_scores.keys()
        }
        
        # Identify key differences
        key_differences = self._identify_key_differences(
            provider_scores, cost_scores, compliance_evaluations
        )
        
        return ComparisonMatrix(
            providers=providers,
            service_comparison=service_comparison,
            cost_comparison=cost_comparison,
            compliance_comparison=compliance_comparison,
            performance_comparison=performance_comparison,
            complexity_comparison=complexity_comparison,
            key_differences=key_differences
        )
    
    def _identify_key_differences(
        self,
        provider_scores: Dict[CloudProviderName, ProviderScore],
        cost_scores: Dict[CloudProviderName, Optional[CostScore]],
        compliance_evaluations: Dict[CloudProviderName, ProviderComplianceEvaluation]
    ) -> List[str]:
        """Identify key differences between providers"""
        differences = []
        
        # Cost differences
        costs = {name: cs.estimated_monthly_cost for name, cs in cost_scores.items() if cs}
        if costs:
            min_cost_provider = min(costs, key=costs.get)
            max_cost_provider = max(costs, key=costs.get)
            cost_diff = costs[max_cost_provider] - costs[min_cost_provider]
            if cost_diff > Decimal("100"):
                differences.append(
                    f"{min_cost_provider.value.upper()} is ${cost_diff:.2f}/month cheaper than {max_cost_provider.value.upper()}"
                )
        
        # Service coverage differences
        service_scores = {name: score.service_availability_score for name, score in provider_scores.items()}
        best_service = max(service_scores, key=service_scores.get)
        if service_scores[best_service] >= 0.9:
            differences.append(f"{best_service.value.upper()} has the best service coverage")
        
        # Compliance differences
        compliance_scores = {name: eval.overall_compliance_score for name, eval in compliance_evaluations.items()}
        best_compliance = max(compliance_scores, key=compliance_scores.get)
        if compliance_scores[best_compliance] >= 0.9:
            differences.append(f"{best_compliance.value.upper()} has the strongest compliance support")
        
        return differences[:5]
    
    def _calculate_overall_confidence(
        self,
        primary_recommendation: ProviderRecommendation,
        provider_scores: Dict[CloudProviderName, ProviderScore]
    ) -> float:
        """Calculate overall confidence in recommendations"""
        # Base confidence on primary recommendation
        base_confidence = primary_recommendation.confidence_score
        
        # Adjust for score separation
        scores = [score.overall_score for score in provider_scores.values()]
        scores.sort(reverse=True)
        
        if len(scores) > 1:
            score_gap = scores[0] - scores[1]
            # Larger gap = higher confidence
            confidence_boost = min(score_gap * 0.5, 0.2)
            base_confidence += confidence_boost
        
        return min(base_confidence, 1.0)
    
    def _generate_key_findings(
        self,
        recommendations: List[ProviderRecommendation],
        comparison_matrix: ComparisonMatrix
    ) -> List[str]:
        """Generate key findings"""
        findings = []
        
        primary = recommendations[0]
        findings.append(
            f"{primary.provider.display_name} is the top recommendation with {primary.overall_score:.0%} overall score"
        )
        
        if primary.estimated_monthly_cost:
            findings.append(
                f"Estimated monthly cost: ${primary.estimated_monthly_cost:.2f}"
            )
        
        if primary.migration_duration_weeks:
            findings.append(
                f"Estimated migration duration: {primary.migration_duration_weeks} weeks"
            )
        
        findings.extend(comparison_matrix.key_differences[:2])
        
        return findings[:5]
    
    def adjust_weights_and_regenerate(
        self,
        new_weights: ScoringWeights,
        previous_report: RecommendationReport,
        **kwargs
    ) -> RecommendationReport:
        """
        Adjust scoring weights and regenerate recommendations
        
        Args:
            new_weights: New scoring weights
            previous_report: Previous recommendation report
            **kwargs: Original parameters for generate_recommendations
            
        Returns:
            New RecommendationReport with adjusted weights
        """
        # Validate new weights
        if not isinstance(new_weights, ScoringWeights):
            raise ValueError("new_weights must be a ScoringWeights instance")
        
        # Regenerate with new weights
        kwargs['scoring_weights'] = new_weights
        return self.generate_recommendations(**kwargs)
