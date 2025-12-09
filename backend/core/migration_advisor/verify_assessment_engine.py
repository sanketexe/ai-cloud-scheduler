"""
Verification script for Migration Assessment Engine

This script verifies that the assessment engine components are properly structured
and can be imported without running full tests.
"""

import sys
from typing import List, Tuple

def verify_imports() -> List[Tuple[str, bool, str]]:
    """Verify that all components can be imported"""
    results = []
    
    # Test model imports
    try:
        from .models import (
            MigrationProject, OrganizationProfile, MigrationStatus,
            CompanySize, InfrastructureType, ExperienceLevel
        )
        results.append(("Models", True, "All model classes imported successfully"))
    except Exception as e:
        results.append(("Models", False, f"Failed to import models: {str(e)}"))
    
    # Test assessment engine imports
    try:
        from .assessment_engine import (
            MigrationProjectManager,
            OrganizationProfiler,
            AssessmentTimelineEstimator,
            MigrationAssessmentEngine
        )
        results.append(("Assessment Engine", True, "All engine classes imported successfully"))
    except Exception as e:
        results.append(("Assessment Engine", False, f"Failed to import engine: {str(e)}"))
    
    # Test endpoint imports
    try:
        from .assessment_endpoints import router
        results.append(("API Endpoints", True, "API router imported successfully"))
    except Exception as e:
        results.append(("API Endpoints", False, f"Failed to import endpoints: {str(e)}"))
    
    return results


def verify_class_structure() -> List[Tuple[str, bool, str]]:
    """Verify that classes have expected methods"""
    results = []
    
    try:
        from .assessment_engine import MigrationProjectManager
        
        expected_methods = [
            'create_migration_project',
            'get_project',
            'update_project_status',
            'list_projects',
            'delete_project'
        ]
        
        for method in expected_methods:
            if hasattr(MigrationProjectManager, method):
                results.append((f"MigrationProjectManager.{method}", True, "Method exists"))
            else:
                results.append((f"MigrationProjectManager.{method}", False, "Method missing"))
                
    except Exception as e:
        results.append(("MigrationProjectManager", False, f"Failed to verify: {str(e)}"))
    
    try:
        from .assessment_engine import OrganizationProfiler
        
        expected_methods = [
            'create_organization_profile',
            'get_profile',
            'update_profile',
            'analyze_infrastructure_type'
        ]
        
        for method in expected_methods:
            if hasattr(OrganizationProfiler, method):
                results.append((f"OrganizationProfiler.{method}", True, "Method exists"))
            else:
                results.append((f"OrganizationProfiler.{method}", False, "Method missing"))
                
    except Exception as e:
        results.append(("OrganizationProfiler", False, f"Failed to verify: {str(e)}"))
    
    try:
        from .assessment_engine import AssessmentTimelineEstimator
        
        expected_methods = [
            'estimate_assessment_duration'
        ]
        
        for method in expected_methods:
            if hasattr(AssessmentTimelineEstimator, method):
                results.append((f"AssessmentTimelineEstimator.{method}", True, "Method exists"))
            else:
                results.append((f"AssessmentTimelineEstimator.{method}", False, "Method missing"))
                
    except Exception as e:
        results.append(("AssessmentTimelineEstimator", False, f"Failed to verify: {str(e)}"))
    
    return results


def verify_api_endpoints() -> List[Tuple[str, bool, str]]:
    """Verify that API endpoints are defined"""
    results = []
    
    try:
        from .assessment_endpoints import router
        
        # Check that router has routes
        if hasattr(router, 'routes') and len(router.routes) > 0:
            results.append(("API Routes", True, f"Found {len(router.routes)} routes"))
            
            # List the routes
            for route in router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    methods = ', '.join(route.methods) if route.methods else 'N/A'
                    results.append((f"  {route.path}", True, f"Methods: {methods}"))
        else:
            results.append(("API Routes", False, "No routes found"))
            
    except Exception as e:
        results.append(("API Endpoints", False, f"Failed to verify: {str(e)}"))
    
    return results


def print_results(results: List[Tuple[str, bool, str]]):
    """Print verification results"""
    print("\n" + "="*80)
    print("Migration Assessment Engine Verification Results")
    print("="*80 + "\n")
    
    passed = 0
    failed = 0
    
    for component, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {component:40} | {message}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print(f"Total: {passed + failed} checks | Passed: {passed} | Failed: {failed}")
    print("="*80 + "\n")
    
    return failed == 0


def main():
    """Run all verifications"""
    all_results = []
    
    print("\n1. Verifying imports...")
    all_results.extend(verify_imports())
    
    print("\n2. Verifying class structure...")
    all_results.extend(verify_class_structure())
    
    print("\n3. Verifying API endpoints...")
    all_results.extend(verify_api_endpoints())
    
    success = print_results(all_results)
    
    if success:
        print("✓ All verifications passed!")
        return 0
    else:
        print("✗ Some verifications failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
