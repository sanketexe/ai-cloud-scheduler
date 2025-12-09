"""
Compliance Evaluation Model for Cloud Migration Advisor

This module implements the ComplianceEvaluationModel that evaluates how well
cloud providers meet compliance and regulatory requirements.

Requirements: 3.1, 3.4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from .provider_catalog import (
    CloudProvider, CloudProviderName, ComplianceFramework,
    ComplianceCertification, ProviderCatalog
)
from .service_catalog_data import get_provider_catalog
from .compliance_catalog import get_compliance_matcher


class ComplianceMatchLevel(Enum):
    """Level of compliance match"""
    FULL = "full"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    NOT_AVAILABLE = "not_available"


@dataclass
class ComplianceMatch:
    """Match between required and available compliance"""
    required_framework: str
    framework_enum: Optional[ComplianceFramework]
    provider_certification: Optional[ComplianceCertification]
    match_level: ComplianceMatchLevel
    compliance_score: float
    match_details: str
    regions_covered: List[str] = field(default_factory=list)
    services_covered: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class ComplianceGap:
    """Gap in compliance coverage"""
    required_framework: str
    gap_type: str
    severity: str
    impact_description: str
    mitigation_options: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class DataResidencyEvaluation:
    """Evaluation of data residency requirements"""
    required_regions: List[str]
    provider: CloudProviderName
    available_regions: List[str]
    compliant_regions: List[str]
    non_compliant_regions: List[str]
    residency_score: float
    residency_status: str


@dataclass
class ProviderComplianceEvaluation:
    """Complete compliance evaluation for a provider"""
    provider: CloudProvider
    compliance_matches: List[ComplianceMatch]
    compliance_gaps: List[ComplianceGap]
    data_residency_evaluation: Optional[DataResidencyEvaluation]
    overall_compliance_score: float
    compliance_coverage_percentage: float
    critical_gaps_count: int
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ComplianceEvaluationModel:
    """
    ML-based compliance evaluation model that assesses cloud providers
    against regulatory and compliance requirements.
    """
    
    def __init__(self, provider_catalog: Optional[ProviderCatalog] = None):
        self.catalog = provider_catalog or get_provider_catalog()
        self.compliance_catalog = get_compliance_matcher()
        
        self.framework_weights = {
            ComplianceFramework.GDPR: 1.0,
            ComplianceFramework.HIPAA: 1.0,
            ComplianceFramework.SOC2: 0.9,
            ComplianceFramework.ISO27001: 0.9,
            ComplianceFramework.PCI_DSS: 1.0,
            ComplianceFramework.FedRAMP: 0.95,
            ComplianceFramework.HITRUST: 0.9,
            ComplianceFramework.FISMA: 0.85,
            ComplianceFramework.CCPA: 0.9,
            ComplianceFramework.NIST: 0.85,
        }
        
        self.framework_mappings = self._initialize_framework_mappings()
    
    def _initialize_framework_mappings(self) -> Dict[str, ComplianceFramework]:
        mappings = {}
        for framework in ComplianceFramework:
            mappings[framework.value] = framework
            mappings[framework.value.upper()] = framework
        
        mappings["gdpr"] = ComplianceFramework.GDPR
        mappings["hipaa"] = ComplianceFramework.HIPAA
        mappings["soc2"] = ComplianceFramework.SOC2
        mappings["soc 2"] = ComplianceFramework.SOC2
        mappings["iso27001"] = ComplianceFramework.ISO27001
        mappings["iso 27001"] = ComplianceFramework.ISO27001
        mappings["pci-dss"] = ComplianceFramework.PCI_DSS
        mappings["pci dss"] = ComplianceFramework.PCI_DSS
        
        return mappings
    
    def evaluate_provider(
        self,
        provider_name: CloudProviderName,
        required_frameworks: List[str],
        data_residency_requirements: Optional[List[str]] = None,
        required_certifications: Optional[List[str]] = None
    ) -> ProviderComplianceEvaluation:
        provider = self.catalog.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name.value} not found")
        
        compliance_matches = []
        compliance_gaps = []
        
        for framework_name in required_frameworks:
            match = self._match_compliance_framework(provider, framework_name)
            compliance_matches.append(match)
            
            if match.match_level in [ComplianceMatchLevel.INSUFFICIENT, ComplianceMatchLevel.NOT_AVAILABLE]:
                gap = self._create_compliance_gap(framework_name, match)
                compliance_gaps.append(gap)
        
        data_residency_eval = None
        if data_residency_requirements:
            data_residency_eval = self._evaluate_data_residency(provider, data_residency_requirements)
        
        overall_score = self._calculate_overall_compliance_score(compliance_matches, data_residency_eval)
        coverage_percentage = self._calculate_coverage_percentage(compliance_matches)
        critical_gaps = len([g for g in compliance_gaps if g.severity in ["critical", "high"]])
        
        strengths, weaknesses = self._analyze_compliance_strengths_weaknesses(
            provider, compliance_matches, compliance_gaps, data_residency_eval
        )
        
        recommendations = self._generate_compliance_recommendations(
            provider, compliance_gaps, data_residency_eval
        )
        
        return ProviderComplianceEvaluation(
            provider=provider,
            compliance_matches=compliance_matches,
            compliance_gaps=compliance_gaps,
            data_residency_evaluation=data_residency_eval,
            overall_compliance_score=overall_score,
            compliance_coverage_percentage=coverage_percentage,
            critical_gaps_count=critical_gaps,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _match_compliance_framework(self, provider: CloudProvider, framework_name: str) -> ComplianceMatch:
        framework_enum = self.framework_mappings.get(framework_name.lower())
        
        if not framework_enum:
            return ComplianceMatch(
                required_framework=framework_name,
                framework_enum=None,
                provider_certification=None,
                match_level=ComplianceMatchLevel.NOT_AVAILABLE,
                compliance_score=0.0,
                match_details=f"Framework '{framework_name}' not recognized"
            )
        
        if framework_enum in provider.compliance_frameworks:
            cert = None
            for certification in provider.compliance_certifications:
                if certification.framework == framework_enum:
                    cert = certification
                    break
            
            if cert:
                return ComplianceMatch(
                    required_framework=framework_name,
                    framework_enum=framework_enum,
                    provider_certification=cert,
                    match_level=ComplianceMatchLevel.FULL,
                    compliance_score=1.0,
                    match_details=f"Provider has {cert.certification_name} certification",
                    regions_covered=cert.regions_covered,
                    services_covered=cert.services_covered
                )
        
        related_score = self._check_related_compliance(provider, framework_enum)
        if related_score > 0.5:
            return ComplianceMatch(
                required_framework=framework_name,
                framework_enum=framework_enum,
                provider_certification=None,
                match_level=ComplianceMatchLevel.PARTIAL,
                compliance_score=related_score,
                match_details=f"Provider has related compliance certifications",
                gaps=["Direct certification not available"]
            )
        
        return ComplianceMatch(
            required_framework=framework_name,
            framework_enum=framework_enum,
            provider_certification=None,
            match_level=ComplianceMatchLevel.NOT_AVAILABLE,
            compliance_score=0.0,
            match_details=f"Provider does not have {framework_name} certification",
            gaps=["Certification not available"]
        )
    
    def _check_related_compliance(self, provider: CloudProvider, framework: ComplianceFramework) -> float:
        related_frameworks = {
            ComplianceFramework.HIPAA: [ComplianceFramework.HITRUST, ComplianceFramework.SOC2],
            ComplianceFramework.GDPR: [ComplianceFramework.ISO27001, ComplianceFramework.SOC2],
            ComplianceFramework.PCI_DSS: [ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
            ComplianceFramework.FedRAMP: [ComplianceFramework.FISMA, ComplianceFramework.NIST],
        }
        
        related = related_frameworks.get(framework, [])
        matches = sum(1 for f in related if f in provider.compliance_frameworks)
        
        if matches > 0:
            return 0.5 + (matches * 0.1)
        
        return 0.0
    
    def _evaluate_data_residency(self, provider: CloudProvider, required_regions: List[str]) -> DataResidencyEvaluation:
        available_regions = [r.region_id for r in provider.regions]
        compliant_regions = []
        non_compliant_regions = []
        
        for required_region in required_regions:
            matched = False
            for available_region in available_regions:
                if required_region.lower() in available_region.lower():
                    compliant_regions.append(required_region)
                    matched = True
                    break
            
            if not matched:
                non_compliant_regions.append(required_region)
        
        residency_score = len(compliant_regions) / len(required_regions) if required_regions else 1.0
        
        if residency_score >= 1.0:
            status = "fully_compliant"
        elif residency_score >= 0.7:
            status = "partially_compliant"
        else:
            status = "non_compliant"
        
        return DataResidencyEvaluation(
            required_regions=required_regions,
            provider=provider.provider_name,
            available_regions=available_regions,
            compliant_regions=compliant_regions,
            non_compliant_regions=non_compliant_regions,
            residency_score=residency_score,
            residency_status=status
        )
    
    def _calculate_overall_compliance_score(
        self, compliance_matches: List[ComplianceMatch], data_residency_eval: Optional[DataResidencyEvaluation]
    ) -> float:
        if not compliance_matches:
            return 0.0
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for match in compliance_matches:
            weight = 1.0
            if match.framework_enum:
                weight = self.framework_weights.get(match.framework_enum, 1.0)
            
            weighted_score += match.compliance_score * weight
            total_weight += weight
        
        compliance_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        if data_residency_eval:
            overall_score = (compliance_score * 0.8) + (data_residency_eval.residency_score * 0.2)
        else:
            overall_score = compliance_score
        
        return overall_score
    
    def _calculate_coverage_percentage(self, compliance_matches: List[ComplianceMatch]) -> float:
        if not compliance_matches:
            return 0.0
        
        full_matches = sum(1 for m in compliance_matches if m.match_level == ComplianceMatchLevel.FULL)
        return (full_matches / len(compliance_matches)) * 100.0
    
    def _create_compliance_gap(self, framework_name: str, match: ComplianceMatch) -> ComplianceGap:
        if match.framework_enum in [ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS]:
            severity = "critical"
        elif match.framework_enum in [ComplianceFramework.SOC2, ComplianceFramework.ISO27001]:
            severity = "high"
        else:
            severity = "medium"
        
        gap_type = "missing_certification" if match.match_level == ComplianceMatchLevel.NOT_AVAILABLE else "limited_coverage"
        
        return ComplianceGap(
            required_framework=framework_name,
            gap_type=gap_type,
            severity=severity,
            impact_description=f"Required {framework_name} compliance not fully met",
            mitigation_options=[
                "Implement additional security controls",
                "Use third-party compliance tools"
            ]
        )
    
    def _analyze_compliance_strengths_weaknesses(
        self, provider: CloudProvider, compliance_matches: List[ComplianceMatch],
        compliance_gaps: List[ComplianceGap], data_residency_eval: Optional[DataResidencyEvaluation]
    ) -> Tuple[List[str], List[str]]:
        strengths = []
        weaknesses = []
        
        full_matches = [m for m in compliance_matches if m.match_level == ComplianceMatchLevel.FULL]
        if len(full_matches) == len(compliance_matches):
            strengths.append("All required compliance certifications available")
        
        critical_gaps = [g for g in compliance_gaps if g.severity == "critical"]
        if critical_gaps:
            weaknesses.append(f"{len(critical_gaps)} critical compliance gaps identified")
        
        if data_residency_eval:
            if data_residency_eval.residency_status == "fully_compliant":
                strengths.append("All data residency requirements met")
            elif data_residency_eval.residency_status == "non_compliant":
                weaknesses.append("Data residency requirements not adequately met")
        
        return strengths, weaknesses
    
    def _generate_compliance_recommendations(
        self, provider: CloudProvider, compliance_gaps: List[ComplianceGap],
        data_residency_eval: Optional[DataResidencyEvaluation]
    ) -> List[str]:
        recommendations = []
        
        if not compliance_gaps:
            recommendations.append("Provider meets all compliance requirements")
        else:
            critical_gaps = [g for g in compliance_gaps if g.severity == "critical"]
            if critical_gaps:
                recommendations.append("Address critical compliance gaps before migration")
        
        return recommendations
    
    def compare_providers_compliance(
        self, required_frameworks: List[str], data_residency_requirements: Optional[List[str]] = None,
        providers: Optional[List[CloudProviderName]] = None
    ) -> Dict[CloudProviderName, ProviderComplianceEvaluation]:
        if providers is None:
            providers = [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]
        
        evaluations = {}
        for provider_name in providers:
            evaluation = self.evaluate_provider(
                provider_name=provider_name,
                required_frameworks=required_frameworks,
                data_residency_requirements=data_residency_requirements
            )
            evaluations[provider_name] = evaluation
        
        return evaluations
