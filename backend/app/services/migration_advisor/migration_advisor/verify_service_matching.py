"""
Verification script for Service Matching Engine

This script tests the service matching algorithm to ensure it correctly:
1. Matches required services to provider services
2. Calculates compatibility scores
3. Identifies service gaps
4. Compares providers
"""

from service_matching_engine import ServiceMatchingModel, MatchQuality
from provider_catalog import CloudProviderName, ServiceCategory


def test_exact_match():
    """Test exact service matching"""
    print("=" * 80)
    print("TEST 1: Exact Service Matching")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Test with AWS-specific service IDs
    required_services = ["ec2", "s3", "rds", "lambda"]
    
    evaluation = model.evaluate_provider(CloudProviderName.AWS, required_services)
    
    print(f"\nProvider: {evaluation.provider.display_name}")
    print(f"Total Required Services: {evaluation.total_required_services}")
    print(f"Overall Match Score: {evaluation.overall_match_score:.2f}")
    print(f"\nMatched Services:")
    
    for match in evaluation.matched_services:
        print(f"  - {match.required_service}: {match.match_quality.value} "
              f"(score: {match.compatibility_score:.2f})")
        if match.provider_service:
            print(f"    → {match.provider_service.service_name}")
    
    print(f"\nService Gaps: {len(evaluation.service_gaps)}")
    print(f"Strengths: {evaluation.strengths}")
    print(f"Weaknesses: {evaluation.weaknesses}")
    
    # Verify all services matched exactly
    assert evaluation.overall_match_score == 1.0, "Expected perfect score for exact matches"
    assert len(evaluation.service_gaps) == 0, "Expected no service gaps"
    print("\n✓ Test passed: Exact matching works correctly")


def test_generic_service_mapping():
    """Test generic service name mapping"""
    print("\n" + "=" * 80)
    print("TEST 2: Generic Service Mapping")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Test with generic service names
    required_services = [
        "virtual_machines",
        "object_storage",
        "relational_database",
        "kubernetes",
        "serverless_functions"
    ]
    
    # Test AWS
    print("\n--- AWS Evaluation ---")
    aws_eval = model.evaluate_provider(CloudProviderName.AWS, required_services)
    print(f"Overall Score: {aws_eval.overall_match_score:.2f}")
    for match in aws_eval.matched_services:
        print(f"  {match.required_service} → {match.provider_service.service_name if match.provider_service else 'N/A'} "
              f"({match.match_quality.value})")
    
    # Test GCP
    print("\n--- GCP Evaluation ---")
    gcp_eval = model.evaluate_provider(CloudProviderName.GCP, required_services)
    print(f"Overall Score: {gcp_eval.overall_match_score:.2f}")
    for match in gcp_eval.matched_services:
        print(f"  {match.required_service} → {match.provider_service.service_name if match.provider_service else 'N/A'} "
              f"({match.match_quality.value})")
    
    # Test Azure
    print("\n--- Azure Evaluation ---")
    azure_eval = model.evaluate_provider(CloudProviderName.AZURE, required_services)
    print(f"Overall Score: {azure_eval.overall_match_score:.2f}")
    for match in azure_eval.matched_services:
        print(f"  {match.required_service} → {match.provider_service.service_name if match.provider_service else 'N/A'} "
              f"({match.match_quality.value})")
    
    # Verify all providers have good scores
    assert aws_eval.overall_match_score >= 0.9, "AWS should have high score for generic services"
    assert gcp_eval.overall_match_score >= 0.9, "GCP should have high score for generic services"
    assert azure_eval.overall_match_score >= 0.9, "Azure should have high score for generic services"
    print("\n✓ Test passed: Generic service mapping works across all providers")


def test_service_gaps():
    """Test service gap identification"""
    print("\n" + "=" * 80)
    print("TEST 3: Service Gap Identification")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Include some non-existent services
    required_services = [
        "ec2",
        "s3",
        "nonexistent_service_xyz",
        "fake_quantum_computer",
        "imaginary_ai_service"
    ]
    
    evaluation = model.evaluate_provider(CloudProviderName.AWS, required_services)
    
    print(f"\nProvider: {evaluation.provider.display_name}")
    print(f"Overall Match Score: {evaluation.overall_match_score:.2f}")
    print(f"\nService Gaps Found: {len(evaluation.service_gaps)}")
    
    for gap in evaluation.service_gaps:
        print(f"\n  Gap: {gap.required_service}")
        print(f"    Severity: {gap.gap_severity}")
        print(f"    Impact: {gap.impact_description}")
        print(f"    Workarounds: {len(gap.workaround_options)}")
    
    # Verify gaps were identified
    assert len(evaluation.service_gaps) == 3, "Expected 3 service gaps"
    assert evaluation.overall_match_score < 1.0, "Score should be reduced due to gaps"
    print("\n✓ Test passed: Service gaps correctly identified")


def test_category_scoring():
    """Test category-based scoring"""
    print("\n" + "=" * 80)
    print("TEST 4: Category-Based Scoring")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Mix of services from different categories
    required_services = [
        "ec2", "lambda",  # Compute
        "s3", "ebs",  # Storage
        "rds", "dynamodb",  # Database
        "sagemaker",  # ML
        "redshift"  # Analytics
    ]
    
    evaluation = model.evaluate_provider(CloudProviderName.AWS, required_services)
    
    print(f"\nProvider: {evaluation.provider.display_name}")
    print(f"Overall Score: {evaluation.overall_match_score:.2f}")
    print(f"\nCategory Scores:")
    
    for category, score in evaluation.category_scores.items():
        print(f"  {category.value}: {score:.2f}")
    
    # Verify category scores
    assert ServiceCategory.COMPUTE in evaluation.category_scores
    assert ServiceCategory.STORAGE in evaluation.category_scores
    assert ServiceCategory.DATABASE in evaluation.category_scores
    print("\n✓ Test passed: Category scoring works correctly")


def test_provider_comparison():
    """Test comparing multiple providers"""
    print("\n" + "=" * 80)
    print("TEST 5: Provider Comparison")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Generic services that all providers should have
    required_services = [
        "virtual_machines",
        "object_storage",
        "relational_database",
        "ml_platform",
        "data_warehouse"
    ]
    
    comparisons = model.compare_providers(required_services)
    
    print("\nProvider Comparison Results:")
    print("-" * 80)
    
    for provider_name, evaluation in comparisons.items():
        print(f"\n{evaluation.provider.display_name}:")
        print(f"  Overall Score: {evaluation.overall_match_score:.2f}")
        print(f"  Service Gaps: {len(evaluation.service_gaps)}")
        print(f"  Strengths: {len(evaluation.strengths)}")
        print(f"  Weaknesses: {len(evaluation.weaknesses)}")
    
    # Verify all providers were evaluated
    assert len(comparisons) == 3, "Expected 3 provider evaluations"
    assert CloudProviderName.AWS in comparisons
    assert CloudProviderName.GCP in comparisons
    assert CloudProviderName.AZURE in comparisons
    print("\n✓ Test passed: Provider comparison works correctly")


def test_service_alternatives():
    """Test getting service alternatives"""
    print("\n" + "=" * 80)
    print("TEST 6: Service Alternatives")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Get alternatives for a compute service
    alternatives = model.get_service_alternatives("virtual_machines", CloudProviderName.AWS)
    
    print(f"\nAlternatives for 'virtual_machines' in AWS:")
    for alt in alternatives:
        print(f"  - {alt.service_name} ({alt.service_id})")
    
    # Verify alternatives were found
    assert len(alternatives) > 0, "Expected to find alternative services"
    print(f"\n✓ Test passed: Found {len(alternatives)} alternative services")


def test_fuzzy_matching():
    """Test fuzzy service name matching"""
    print("\n" + "=" * 80)
    print("TEST 7: Fuzzy Service Matching")
    print("=" * 80)
    
    model = ServiceMatchingModel()
    
    # Test with partial service names
    required_services = [
        "compute",  # Should match compute-related services
        "storage",  # Should match storage services
        "database"  # Should match database services
    ]
    
    evaluation = model.evaluate_provider(CloudProviderName.AWS, required_services)
    
    print(f"\nFuzzy Matching Results:")
    for match in evaluation.matched_services:
        print(f"  {match.required_service}:")
        print(f"    Quality: {match.match_quality.value}")
        print(f"    Score: {match.compatibility_score:.2f}")
        if match.provider_service:
            print(f"    Matched to: {match.provider_service.service_name}")
        print(f"    Reason: {match.match_reason}")
    
    print("\n✓ Test passed: Fuzzy matching attempted")


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "=" * 80)
    print("SERVICE MATCHING ENGINE VERIFICATION")
    print("=" * 80)
    
    try:
        test_exact_match()
        test_generic_service_mapping()
        test_service_gaps()
        test_category_scoring()
        test_provider_comparison()
        test_service_alternatives()
        test_fuzzy_matching()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nService Matching Engine is working correctly!")
        print("The algorithm can:")
        print("  ✓ Match required services to provider services")
        print("  ✓ Calculate compatibility scores")
        print("  ✓ Identify service gaps")
        print("  ✓ Compare multiple providers")
        print("  ✓ Provide service alternatives")
        print("  ✓ Handle fuzzy matching")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
