"""
Compliance Certification Catalog and Mapping

This module provides comprehensive compliance certification data for cloud providers
and utilities for matching requirements to provider certifications.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from .provider_catalog import (
    ComplianceCertification, ComplianceFramework, CloudProviderName
)


def create_aws_compliance_certifications() -> List[ComplianceCertification]:
    """Create AWS compliance certifications"""
    certifications = []
    
    # GDPR
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.GDPR,
        certification_name="GDPR Compliance",
        description="General Data Protection Regulation compliance for EU data protection",
        regions_covered=["eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1"],
        services_covered=["ec2", "s3", "rds", "lambda", "dynamodb", "aurora"],
        certification_date="2018-05-25",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/gdpr-center/"
    ))
    
    # HIPAA
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.HIPAA,
        certification_name="HIPAA Eligible Services",
        description="Health Insurance Portability and Accountability Act compliance",
        regions_covered=["us-east-1", "us-east-2", "us-west-1", "us-west-2"],
        services_covered=["ec2", "s3", "rds", "dynamodb", "lambda", "ecs", "eks"],
        certification_date="2009-01-01",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/hipaa-compliance/"
    ))
    
    # SOC 2
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.SOC2,
        certification_name="SOC 2 Type II",
        description="Service Organization Control 2 Type II certification",
        regions_covered=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        services_covered=["ec2", "s3", "rds", "lambda", "dynamodb", "ecs", "eks", "sagemaker"],
        certification_date="2010-01-01",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/soc-faqs/"
    ))
    
    # ISO 27001
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.ISO27001,
        certification_name="ISO 27001:2013",
        description="Information security management system certification",
        regions_covered=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        services_covered=["ec2", "s3", "rds", "lambda", "dynamodb", "ecs", "eks"],
        certification_date="2010-01-01",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/iso-27001-faqs/"
    ))
    
    # PCI DSS
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.PCI_DSS,
        certification_name="PCI DSS Level 1",
        description="Payment Card Industry Data Security Standard",
        regions_covered=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        services_covered=["ec2", "s3", "rds", "lambda", "dynamodb"],
        certification_date="2010-01-01",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/pci-dss-level-1-faqs/"
    ))
    
    # FedRAMP
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.FedRAMP,
        certification_name="FedRAMP High",
        description="Federal Risk and Authorization Management Program",
        regions_covered=["us-east-1", "us-west-2"],
        services_covered=["ec2", "s3", "rds", "lambda"],
        certification_date="2013-01-01",
        audit_report_available=True,
        documentation_url="https://aws.amazon.com/compliance/fedramp/"
    ))
    
    return certifications


def create_gcp_compliance_certifications() -> List[ComplianceCertification]:
    """Create GCP compliance certifications"""
    certifications = []
    
    # GDPR
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.GDPR,
        certification_name="GDPR Compliance",
        description="General Data Protection Regulation compliance for EU data protection",
        regions_covered=["europe-west1", "europe-west2", "europe-west3", "europe-west4"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql", "cloud_functions", "gke"],
        certification_date="2018-05-25",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/privacy/gdpr"
    ))
    
    # HIPAA
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.HIPAA,
        certification_name="HIPAA Compliance",
        description="Health Insurance Portability and Accountability Act compliance",
        regions_covered=["us-central1", "us-east1", "us-west1"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql", "gke", "cloud_functions"],
        certification_date="2015-01-01",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/security/compliance/hipaa"
    ))
    
    # SOC 2
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.SOC2,
        certification_name="SOC 2 Type II",
        description="Service Organization Control 2 Type II certification",
        regions_covered=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql", "gke", "bigquery"],
        certification_date="2011-01-01",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/security/compliance/soc-2"
    ))
    
    # ISO 27001
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.ISO27001,
        certification_name="ISO 27001:2013",
        description="Information security management system certification",
        regions_covered=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql", "gke"],
        certification_date="2013-01-01",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/security/compliance/iso-27001"
    ))
    
    # PCI DSS
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.PCI_DSS,
        certification_name="PCI DSS v3.2.1",
        description="Payment Card Industry Data Security Standard",
        regions_covered=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql"],
        certification_date="2015-01-01",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/security/compliance/pci-dss"
    ))
    
    # FedRAMP
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.FedRAMP,
        certification_name="FedRAMP Moderate",
        description="Federal Risk and Authorization Management Program",
        regions_covered=["us-central1", "us-east1"],
        services_covered=["compute_engine", "cloud_storage", "cloud_sql"],
        certification_date="2017-01-01",
        audit_report_available=True,
        documentation_url="https://cloud.google.com/security/compliance/fedramp"
    ))
    
    return certifications


def create_azure_compliance_certifications() -> List[ComplianceCertification]:
    """Create Azure compliance certifications"""
    certifications = []
    
    # GDPR
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.GDPR,
        certification_name="GDPR Compliance",
        description="General Data Protection Regulation compliance for EU data protection",
        regions_covered=["westeurope", "northeurope", "francecentral", "germanywestcentral"],
        services_covered=["virtual_machines", "blob_storage", "sql_database", "azure_functions", "aks"],
        certification_date="2018-05-25",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/gdpr"
    ))
    
    # HIPAA
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.HIPAA,
        certification_name="HIPAA/HITECH Compliance",
        description="Health Insurance Portability and Accountability Act compliance",
        regions_covered=["eastus", "eastus2", "westus", "westus2"],
        services_covered=["virtual_machines", "blob_storage", "sql_database", "aks", "azure_functions"],
        certification_date="2013-01-01",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/offering-hipaa-hitech"
    ))
    
    # SOC 2
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.SOC2,
        certification_name="SOC 2 Type II",
        description="Service Organization Control 2 Type II certification",
        regions_covered=["eastus", "westus", "westeurope", "southeastasia"],
        services_covered=["virtual_machines", "blob_storage", "sql_database", "aks", "machine_learning"],
        certification_date="2012-01-01",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/offering-soc"
    ))
    
    # ISO 27001
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.ISO27001,
        certification_name="ISO 27001:2013",
        description="Information security management system certification",
        regions_covered=["eastus", "westus", "westeurope", "southeastasia"],
        services_covered=["virtual_machines", "blob_storage", "sql_database", "aks"],
        certification_date="2012-01-01",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/offering-iso-27001"
    ))
    
    # PCI DSS
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.PCI_DSS,
        certification_name="PCI DSS Level 1",
        description="Payment Card Industry Data Security Standard",
        regions_covered=["eastus", "westus", "westeurope", "southeastasia"],
        services_covered=["virtual_machines", "blob_storage", "sql_database"],
        certification_date="2013-01-01",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/offering-pci-dss"
    ))
    
    # FedRAMP
    certifications.append(ComplianceCertification(
        framework=ComplianceFramework.FedRAMP,
        certification_name="FedRAMP High",
        description="Federal Risk and Authorization Management Program",
        regions_covered=["usgovvirginia", "usgovtexas"],
        services_covered=["virtual_machines", "blob_storage", "sql_database"],
        certification_date="2014-01-01",
        audit_report_available=True,
        documentation_url="https://docs.microsoft.com/en-us/compliance/regulatory/offering-fedramp"
    ))
    
    return certifications


@dataclass
class ComplianceMatchResult:
    """Result of compliance matching"""
    provider: CloudProviderName
    framework: ComplianceFramework
    is_compliant: bool
    certification: Optional[ComplianceCertification] = None
    coverage_score: float = 0.0  # 0.0 to 1.0
    gaps: List[str] = field(default_factory=list)
    notes: str = ""


class ComplianceMatcher:
    """Utility for matching compliance requirements to provider certifications"""
    
    def __init__(self):
        self.aws_certifications = create_aws_compliance_certifications()
        self.gcp_certifications = create_gcp_compliance_certifications()
        self.azure_certifications = create_azure_compliance_certifications()
        
        # Build lookup maps
        self._build_lookup_maps()
    
    def _build_lookup_maps(self):
        """Build lookup maps for efficient searching"""
        self.provider_certifications = {
            CloudProviderName.AWS: self.aws_certifications,
            CloudProviderName.GCP: self.gcp_certifications,
            CloudProviderName.AZURE: self.azure_certifications
        }
        
        # Framework to certifications map
        self.framework_map = {
            CloudProviderName.AWS: {},
            CloudProviderName.GCP: {},
            CloudProviderName.AZURE: {}
        }
        
        for provider, certs in self.provider_certifications.items():
            for cert in certs:
                self.framework_map[provider][cert.framework] = cert
    
    def check_compliance(
        self,
        provider: CloudProviderName,
        framework: ComplianceFramework,
        required_regions: Optional[List[str]] = None,
        required_services: Optional[List[str]] = None
    ) -> ComplianceMatchResult:
        """
        Check if a provider meets compliance requirements
        
        Args:
            provider: Cloud provider to check
            framework: Compliance framework required
            required_regions: List of regions that need compliance
            required_services: List of services that need compliance
            
        Returns:
            ComplianceMatchResult with compliance status and details
        """
        cert = self.framework_map.get(provider, {}).get(framework)
        
        if not cert:
            return ComplianceMatchResult(
                provider=provider,
                framework=framework,
                is_compliant=False,
                coverage_score=0.0,
                gaps=["Provider does not have this certification"],
                notes=f"{provider.value} does not support {framework.value}"
            )
        
        gaps = []
        coverage_score = 1.0
        
        # Check region coverage
        if required_regions:
            missing_regions = set(required_regions) - set(cert.regions_covered)
            if missing_regions:
                gaps.append(f"Certification not available in regions: {', '.join(missing_regions)}")
                coverage_score *= (len(cert.regions_covered) / len(required_regions))
        
        # Check service coverage
        if required_services:
            missing_services = set(required_services) - set(cert.services_covered)
            if missing_services:
                gaps.append(f"Certification not available for services: {', '.join(missing_services)}")
                coverage_score *= (len(cert.services_covered) / len(required_services))
        
        is_compliant = len(gaps) == 0
        
        return ComplianceMatchResult(
            provider=provider,
            framework=framework,
            is_compliant=is_compliant,
            certification=cert,
            coverage_score=coverage_score,
            gaps=gaps,
            notes=cert.description if is_compliant else "Partial compliance"
        )
    
    def compare_compliance_across_providers(
        self,
        frameworks: List[ComplianceFramework],
        required_regions: Optional[List[str]] = None,
        required_services: Optional[List[str]] = None
    ) -> Dict[CloudProviderName, List[ComplianceMatchResult]]:
        """
        Compare compliance support across all providers
        
        Args:
            frameworks: List of compliance frameworks to check
            required_regions: List of regions that need compliance
            required_services: List of services that need compliance
            
        Returns:
            Dictionary mapping providers to their compliance results
        """
        results = {}
        
        for provider in [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]:
            provider_results = []
            for framework in frameworks:
                result = self.check_compliance(
                    provider=provider,
                    framework=framework,
                    required_regions=required_regions,
                    required_services=required_services
                )
                provider_results.append(result)
            results[provider] = provider_results
        
        return results
    
    def get_compliance_score(
        self,
        provider: CloudProviderName,
        frameworks: List[ComplianceFramework],
        required_regions: Optional[List[str]] = None,
        required_services: Optional[List[str]] = None
    ) -> float:
        """
        Calculate overall compliance score for a provider
        
        Args:
            provider: Cloud provider to score
            frameworks: List of required compliance frameworks
            required_regions: List of regions that need compliance
            required_services: List of services that need compliance
            
        Returns:
            Compliance score from 0.0 to 1.0
        """
        if not frameworks:
            return 1.0
        
        total_score = 0.0
        for framework in frameworks:
            result = self.check_compliance(
                provider=provider,
                framework=framework,
                required_regions=required_regions,
                required_services=required_services
            )
            total_score += result.coverage_score
        
        return total_score / len(frameworks)
    
    def get_supported_frameworks(self, provider: CloudProviderName) -> List[ComplianceFramework]:
        """Get list of compliance frameworks supported by a provider"""
        return list(self.framework_map[provider].keys())
    
    def get_certification_details(
        self,
        provider: CloudProviderName,
        framework: ComplianceFramework
    ) -> Optional[ComplianceCertification]:
        """Get detailed certification information"""
        return self.framework_map.get(provider, {}).get(framework)


# Global compliance matcher instance
_compliance_matcher_instance = None


def get_compliance_matcher() -> ComplianceMatcher:
    """Get or create the global compliance matcher instance"""
    global _compliance_matcher_instance
    if _compliance_matcher_instance is None:
        _compliance_matcher_instance = ComplianceMatcher()
    return _compliance_matcher_instance
