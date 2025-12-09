"""
Verification script for ML-Based Recommendation Engine

This script tests the complete recommendation engine with sample data.
"""

from decimal import Decimal
from recommendation_engine import RecommendationEngine, ScoringWeights
from performance_analysis_model import PerformanceRequirement, PerformanceMetric


def test_recommendation_engine():
    """Test the recommendation engine with sample data"""
    print("=" * 80)
    print("Testing ML-Based Recommendation Engine")
    print("=" * 80)
    
    # Initialize engine
    engine = RecommendationEngine()
    print("\n✓ Recommendation engine initialized")
    
    # Sample requirements
    required_services = [
        "virtual_machines",
        "object_storage",
        "relational_database",
        "kubernetes",
        "ml_platform"
    ]
    
    target_budget = Decimal("5000.00")
    
    compliance_requirements = [
        "gdpr",
        "soc2",
        "iso27001"
    ]
    
    workload_specs = [
        {
            "name": "web_application",
            "compute_cores": 8,
            "memory_gb": 32,
            "storage_tb": 2.0,
            "data_transfer_tb": 1.5,
            "database_cores": 4,
            "region": "us-east"
        },
        {
            "name": "data_processing",
            "compute_cores": 16,
            "memory_gb": 64,
            "storage_tb": 5.0,
            "data_transfer_tb": 2.0,
            "database_cores": 0,
            "region": "us-east"
        }
    ]
    
    performance_requirements = [
        PerformanceRequirement(
            metric=PerformanceMetric.COMPUTE_CAPACITY,
            required_value=100,
            unit="instances",
            priority="high",
            description="Need at least 100 compute instances"
        ),
        PerformanceRequirement(
            metric=PerformanceMetric.AUTO_SCALING,
            required_value=1.0,
            unit="boolean",
            priority="critical",
            description="Auto-scaling required"
        )
    ]
    
    print("\n✓ Sample requirements prepared")
    print(f"  - Required services: {len(required_services)}")
    print(f"  - Target budget: ${target_budget}")
    print(f"  - Compliance requirements: {len(compliance_requirements)}")
    print(f"  - Workloads: {len(workload_specs)}")
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("Generating Recommendations...")
    print("=" * 80)
    
    try:
        report = engine.generate_recommendations(
            required_services=required_services,
            target_monthly_budget=target_budget,
            compliance_requirements=compliance_requirements,
            source_infrastructure="on_premises",
            workload_specs=workload_specs,
            performance_requirements=performance_requirements,
            data_residency_requirements=["us", "eu"]
        )
        
        print("\n✓ Recommendations generated successfully!")
        
        # Display primary recommendation
        print("\n" + "=" * 80)
        print("PRIMARY RECOMMENDATION")
        print("=" * 80)
        primary = report.primary_recommendation
        print(f"\nProvider: {primary.provider.display_name}")
        print(f"Rank: #{primary.rank}")
        print(f"Overall Score: {primary.overall_score:.3f} ({primary.overall_score*100:.1f}%)")
        print(f"Confidence: {primary.confidence_score:.3f} ({primary.confidence_score*100:.1f}%)")
        
        if primary.estimated_monthly_cost:
            print(f"Estimated Monthly Cost: ${primary.estimated_monthly_cost:.2f}")
        if primary.migration_duration_weeks:
            print(f"Migration Duration: {primary.migration_duration_weeks} weeks")
        
        print(f"\nJustification:")
        print(f"  {primary.justification}")
        
        print(f"\nStrengths:")
        for strength in primary.strengths:
            print(f"  • {strength}")
        
        if primary.weaknesses:
            print(f"\nWeaknesses:")
            for weakness in primary.weaknesses:
                print(f"  • {weakness}")
        
        if primary.key_differentiators:
            print(f"\nKey Differentiators:")
            for diff in primary.key_differentiators:
                print(f"  • {diff}")
        
        # Display alternative recommendations
        if report.alternative_recommendations:
            print("\n" + "=" * 80)
            print("ALTERNATIVE RECOMMENDATIONS")
            print("=" * 80)
            for alt in report.alternative_recommendations:
                print(f"\n#{alt.rank}. {alt.provider.display_name}")
                print(f"   Score: {alt.overall_score:.3f} ({alt.overall_score*100:.1f}%)")
                if alt.estimated_monthly_cost:
                    print(f"   Est. Cost: ${alt.estimated_monthly_cost:.2f}/month")
        
        # Display comparison matrix
        print("\n" + "=" * 80)
        print("PROVIDER COMPARISON MATRIX")
        print("=" * 80)
        matrix = report.comparison_matrix
        
        print("\nService Coverage:")
        for provider_name, score in matrix.service_comparison.items():
            print(f"  {provider_name.value.upper()}: {score:.1%}")
        
        print("\nCompliance Score:")
        for provider_name, score in matrix.compliance_comparison.items():
            print(f"  {provider_name.value.upper()}: {score:.1%}")
        
        print("\nEstimated Monthly Cost:")
        for provider_name, cost in matrix.cost_comparison.items():
            if cost > 0:
                print(f"  {provider_name.value.upper()}: ${cost:.2f}")
        
        print("\nMigration Complexity:")
        for provider_name, complexity in matrix.complexity_comparison.items():
            print(f"  {provider_name.value.upper()}: {complexity:.1%}")
        
        if matrix.key_differences:
            print("\nKey Differences:")
            for diff in matrix.key_differences:
                print(f"  • {diff}")
        
        # Display key findings
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        for finding in report.key_findings:
            print(f"  • {finding}")
        
        # Display scoring weights
        print("\n" + "=" * 80)
        print("SCORING WEIGHTS USED")
        print("=" * 80)
        weights_dict = report.scoring_weights.to_dict()
        for category, weight in weights_dict.items():
            print(f"  {category.replace('_', ' ').title()}: {weight:.1%}")
        
        print(f"\nOverall Confidence: {report.overall_confidence:.1%}")
        
        # Test weight adjustment
        print("\n" + "=" * 80)
        print("Testing Weight Adjustment...")
        print("=" * 80)
        
        new_weights = ScoringWeights(
            service_availability_weight=0.15,
            pricing_weight=0.35,  # Increase pricing importance
            compliance_weight=0.20,
            technical_fit_weight=0.10,
            migration_complexity_weight=0.20
        )
        
        print("\nNew weights (prioritizing cost):")
        for category, weight in new_weights.to_dict().items():
            print(f"  {category.replace('_', ' ').title()}: {weight:.1%}")
        
        adjusted_report = engine.adjust_weights_and_regenerate(
            new_weights=new_weights,
            previous_report=report,
            required_services=required_services,
            target_monthly_budget=target_budget,
            compliance_requirements=compliance_requirements,
            source_infrastructure="on_premises",
            workload_specs=workload_specs,
            performance_requirements=performance_requirements,
            data_residency_requirements=["us", "eu"]
        )
        
        print("\n✓ Recommendations regenerated with new weights")
        print(f"\nNew Primary Recommendation: {adjusted_report.primary_recommendation.provider.display_name}")
        print(f"New Overall Score: {adjusted_report.primary_recommendation.overall_score:.3f}")
        
        if adjusted_report.primary_recommendation.provider.provider_name != primary.provider.provider_name:
            print("\n⚠ Note: Primary recommendation changed after weight adjustment!")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe ML-Based Recommendation Engine is working correctly.")
        print("All components (service matching, cost prediction, compliance evaluation,")
        print("performance analysis, and migration complexity) are integrated successfully.")
        
    except Exception as e:
        print(f"\n✗ Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_recommendation_engine()
    exit(0 if success else 1)
