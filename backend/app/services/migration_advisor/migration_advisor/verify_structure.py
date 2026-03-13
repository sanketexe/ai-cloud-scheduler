"""
Simple verification script to check migration advisor structure

This script verifies the module structure without requiring database connection.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def verify_imports():
    """Verify that all models can be imported"""
    print("Verifying migration advisor module structure...\n")
    
    try:
        # Test enum imports
        from backend.core.migration_advisor.models import (
            MigrationStatus,
            CompanySize,
            InfrastructureType,
            ExperienceLevel,
            PhaseStatus,
            OwnershipStatus,
            MigrationRiskLevel,
        )
        print("✓ Enums imported successfully")
        
        # Test model imports
        from backend.core.migration_advisor.models import (
            MigrationProject,
            OrganizationProfile,
            WorkloadProfile,
            PerformanceRequirements,
            ComplianceRequirements,
            BudgetConstraints,
            TechnicalRequirements,
            ProviderEvaluation,
            RecommendationReport,
            MigrationPlan,
            MigrationPhase,
            OrganizationalStructure,
            CategorizedResource,
        )
        print("✓ All models imported successfully")
        
        # Test module-level imports
        from backend.core.migration_advisor import (
            MigrationProject as MP,
            OrganizationProfile as OP,
        )
        print("✓ Module-level imports working")
        
        # Verify enum values
        assert MigrationStatus.ASSESSMENT.value == "assessment"
        assert CompanySize.MEDIUM.value == "medium"
        assert InfrastructureType.ON_PREMISES.value == "on_premises"
        print("✓ Enum values verified")
        
        print("\n✅ All verification checks passed!")
        print("\nMigration Advisor module structure is correctly set up.")
        print("\nNext steps:")
        print("1. Run database migrations: alembic upgrade head")
        print("2. Implement assessment engines")
        print("3. Implement recommendation engine")
        print("4. Create API endpoints")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
