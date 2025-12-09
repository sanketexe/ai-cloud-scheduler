"""
Verification script for Cloud Provider Catalog and Data Layer

This script tests the provider catalog, service catalog, compliance mapping,
and pricing data layer implementations.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.migration_advisor.provider_catalog import CloudProviderName, ServiceCategory, ComplianceFramework
from core.migration_advisor.service_catalog_data import get_provider_catalog
from core.migration_advisor.compliance_catalog import get_compliance_matcher
from core.migration_advisor.pricing_catalog import get_cost_estimator


def verify_service_catalog():
    """Verify service catalog implementation"""
    print("=" * 80)
    print("VERIFYING SERVICE CATALOG")
    print("=" * 80)
    
    catalog = get_provider_catalog()
    
    # Check providers
    print(f"\n✓ Providers loaded: {len(catalog.providers)}")
    for provider_name, provider in catalog.providers.items():
        print(f"  - {provider.display_name}: {len(provider.services)} services")
    
    # Check AWS services
    aws = catalog.get_provider(CloudProviderName.AWS)
    print(f"\n✓ AWS Services by Category:")
    for category, service_ids in aws.service_categories.items():
        print(f"  - {category.value}: {len(service_ids)} services")
    
    # Check GCP services
    gcp = catalog.get_provider(CloudProviderName.GCP)
    print(f"\n✓ GCP Services by Category:")
    for category, service_ids in gcp.service_categories.items():
        print(f"  - {category.value}: {len(service_ids)} services")
    
    # Check Azure services
    azure = catalog.get_provider(CloudProviderName.AZURE)
    print(f"\n✓ Azure Services by Category:")
    for category, service_ids in azure.service_categories.items():
        print(f"  - {category.value}: {len(service_ids)} services")
    
    # Check service comparisons
    print(f"\n✓ Service Comparisons: {len(catalog.service_comparisons)}")
    for comparison in catalog.service_comparisons:
        print(f"  - {comparison.service_purpose}")
        print(f"    AWS: {comparison.aws_service.service_name if comparison.aws_service else 'N/A'}")
        print(f"    GCP: {comparison.gcp_service.service_name if comparison.gcp_service else 'N/A'}")
        print(f"    Azure: {comparison.azure_service.service_name if comparison.azure_service else 'N/A'}")
    
    # Check regions
    print(f"\n✓ AWS Regions: {len(aws.regions)}")
    for region in aws.regions:
        print(f"  - {region.region_name} ({region.region_id}): {region.availability_zones} AZs")
    
    print("\n✅ Service Catalog Verification Complete\n")


def verify_compliance_mapping():
    """Verify compliance certification mapping"""
    print("=" * 80)
    print("VERIFYING COMPLIANCE MAPPING")
    print("=" * 80)
    
    matcher = get_compliance_matcher()
    
    # Check supported frameworks
    print("\n✓ Supported Compliance Frameworks:")
    for provider in [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]:
        frameworks = matcher.get_supported_frameworks(provider)
        print(f"  - {provider.value}: {len(frameworks)} frameworks")
        for framework in frameworks:
            print(f"    • {framework.value}")
    
    # Test compliance checking
    print("\n✓ Testing Compliance Checks:")
    test_frameworks = [ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOC2]
    
    for framework in test_frameworks:
        print(f"\n  {framework.value}:")
        for provider in [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]:
            result = matcher.check_compliance(provider, framework)
            status = "✓" if result.is_compliant else "✗"
            print(f"    {status} {provider.value}: {result.coverage_score:.2%} coverage")
    
    # Test compliance scoring
    print("\n✓ Compliance Scores:")
    for provider in [CloudProviderName.AWS, CloudProviderName.GCP, CloudProviderName.AZURE]:
        score = matcher.get_compliance_score(provider, test_frameworks)
        print(f"  - {provider.value}: {score:.2%}")
    
    print("\n✅ Compliance Mapping Verification Complete\n")


def verify_pricing_data():
    """Verify pricing data layer"""
    print("=" * 80)
    print("VERIFYING PRICING DATA LAYER")
    print("=" * 80)
    
    estimator = get_cost_estimator()
    
    # Test compute cost estimation
    print("\n✓ Compute Cost Estimates (medium instance, 730 hours/month):")
    compute_costs = estimator.compare_costs_across_providers(
        workload_type="compute",
        instance_type="medium",
        hours_per_month=730
    )
    
    for provider, estimate in compute_costs.items():
        print(f"  - {provider.value}: ${estimate.monthly_cost:.2f}/month (${estimate.annual_cost:.2f}/year)")
    
    # Test storage cost estimation
    print("\n✓ Storage Cost Estimates (1000 GB, standard class):")
    storage_costs = estimator.compare_costs_across_providers(
        workload_type="storage",
        storage_gb=1000,
        storage_class="standard"
    )
    
    for provider, estimate in storage_costs.items():
        print(f"  - {provider.value}: ${estimate.monthly_cost:.2f}/month (${estimate.annual_cost:.2f}/year)")
    
    # Test database cost estimation
    print("\n✓ Database Cost Estimates (small instance, 100 GB storage):")
    db_costs = estimator.compare_costs_across_providers(
        workload_type="database",
        instance_type="small",
        storage_gb=100,
        hours_per_month=730
    )
    
    for provider, estimate in db_costs.items():
        print(f"  - {provider.value}: ${estimate.monthly_cost:.2f}/month (${estimate.annual_cost:.2f}/year)")
    
    # Find cheapest provider
    print("\n✓ Cheapest Provider Analysis:")
    
    cheapest_compute = estimator.get_cheapest_provider(
        workload_type="compute",
        instance_type="medium",
        hours_per_month=730
    )
    print(f"  - Compute: {cheapest_compute[0].value} (${cheapest_compute[1].monthly_cost:.2f}/month)")
    
    cheapest_storage = estimator.get_cheapest_provider(
        workload_type="storage",
        storage_gb=1000,
        storage_class="standard"
    )
    print(f"  - Storage: {cheapest_storage[0].value} (${cheapest_storage[1].monthly_cost:.2f}/month)")
    
    cheapest_db = estimator.get_cheapest_provider(
        workload_type="database",
        instance_type="small",
        storage_gb=100,
        hours_per_month=730
    )
    print(f"  - Database: {cheapest_db[0].value} (${cheapest_db[1].monthly_cost:.2f}/month)")
    
    print("\n✅ Pricing Data Layer Verification Complete\n")


def main():
    """Run all verification tests"""
    print("\n" + "=" * 80)
    print("CLOUD PROVIDER CATALOG AND DATA LAYER VERIFICATION")
    print("=" * 80 + "\n")
    
    try:
        verify_service_catalog()
        verify_compliance_mapping()
        verify_pricing_data()
        
        print("=" * 80)
        print("✅ ALL VERIFICATIONS PASSED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
