import sys
import os

# Add the current directory to sys.path so we can import 'app'
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from app.services.migration_advisor.migration_advisor.assessment_endpoints import router as assessment_router
    print("Success: assessment_endpoints")
    from app.services.migration_advisor.migration_advisor.requirements_endpoints import router as requirements_router
    print("Success: requirements_endpoints")
    from app.services.migration_advisor.migration_advisor.recommendation_endpoints import router as recommendation_router
    print("Success: recommendation_endpoints")
    from app.services.migration_advisor.migration_advisor.migration_planning_endpoints import router as planning_router
    print("Success: migration_planning_endpoints")
    from app.services.migration_advisor.migration_advisor.resource_organization_endpoints import router as resource_org_router
    print("Success: resource_organization_endpoints")
    from app.services.migration_advisor.migration_advisor.dimensional_management_endpoints import router as dimensional_router
    print("Success: dimensional_management_endpoints")
    from app.services.migration_advisor.migration_advisor.integration_endpoints import router as integration_router
    print("Success: integration_endpoints")
    from app.services.migration_advisor.migration_advisor.report_endpoints import router as report_router
    print("Success: report_endpoints")
    
    print("All imports successful!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
