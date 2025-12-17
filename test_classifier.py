#!/usr/bin/env python3
"""
Test script for the file classifier to verify it correctly identifies
files according to the requirements.
"""

from cleanup_classifier import FileClassifier, FileCategory, FileAction


def test_classifier():
    """Test the classifier with specific files mentioned in requirements"""
    classifier = FileClassifier()
    
    # Test cases based on requirements
    test_cases = [
        # Core files that should be kept (Requirement 5.1-5.5)
        ("backend/core/models.py", FileCategory.CORE, FileAction.KEEP),
        ("backend/main.py", FileCategory.CORE, FileAction.KEEP),
        ("frontend/src/App.tsx", FileCategory.CORE, FileAction.KEEP),
        ("docker-compose.yml", FileCategory.CORE, FileAction.KEEP),
        ("backend/alembic/versions/001_initial_migration.py", FileCategory.CORE, FileAction.KEEP),
        
        # Demo files that should be removed (Requirement 2.1-2.5)
        ("SHOWCASE.md", FileCategory.DEMO, FileAction.REMOVE),
        ("FEATURES_AND_CAPABILITIES.md", FileCategory.DEMO, FileAction.REMOVE),
        ("docs/images/demo/api-documentation.png", FileCategory.DEMO, FileAction.REMOVE),
        ("docs/images/screenshots/Screenshot 2025-12-10 074004.png", FileCategory.DEMO, FileAction.REMOVE),
        
        # Redundant documentation (Requirement 1.1, 1.5)
        ("DATA_EXPLANATION.md", FileCategory.DOCS, FileAction.REMOVE),
        ("MIGRATION_FLOW_EXPLANATION.md", FileCategory.DOCS, FileAction.REMOVE),
        ("MIGRATION_LOADING_FIX.md", FileCategory.DOCS, FileAction.REMOVE),
        ("TECHNICAL_INTERVIEW_GUIDE.md", FileCategory.DOCS, FileAction.REMOVE),
        ("START_PLATFORM.md", FileCategory.DOCS, FileAction.REMOVE),
        
        # Build artifacts and cache (Requirement 4.1-4.5)
        ("backend/__pycache__", FileCategory.CACHE, FileAction.REMOVE),
        ("backend/.pytest_cache", FileCategory.CACHE, FileAction.REMOVE),
        ("frontend/node_modules", FileCategory.CACHE, FileAction.REMOVE),
        (".venv", FileCategory.CACHE, FileAction.REMOVE),
        
        # Essential documentation that should be kept (Requirement 1.2-1.4)
        ("README.md", FileCategory.CORE, FileAction.KEEP),
        ("CONTRIBUTING.md", FileCategory.DOCS, FileAction.KEEP),
        ("SETUP_GUIDE.md", FileCategory.DOCS, FileAction.KEEP),
        ("DEPLOYMENT_GUIDE.md", FileCategory.DOCS, FileAction.KEEP),
        
        # Configuration files (Requirement 3.2-3.4)
        (".env", FileCategory.CONFIG, FileAction.KEEP),
        ("docker-compose.prod.yml", FileCategory.CONFIG, FileAction.KEEP),
        ("k8s/postgres-deployment.yaml", FileCategory.CONFIG, FileAction.KEEP),
        
        # Scripts that should be preserved
        ("scripts/setup_finops.py", FileCategory.CONFIG, FileAction.KEEP),
        ("scripts/docker-health-check.sh", FileCategory.CONFIG, FileAction.KEEP),
        
        # Redundant scripts that should be removed
        ("start_dev_tasks.py", FileCategory.DEMO, FileAction.REMOVE),
        ("scripts/add-demo-images.ps1", FileCategory.DEMO, FileAction.REMOVE),
    ]
    
    print("Testing File Classification:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for file_path, expected_category, expected_action in test_cases:
        classification = classifier.classify_file(file_path)
        
        category_match = classification.category == expected_category
        action_match = classification.action == expected_action
        
        if category_match and action_match:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"{status} {file_path}")
        print(f"    Expected: {expected_category.value} -> {expected_action.value}")
        print(f"    Got:      {classification.category.value} -> {classification.action.value}")
        print(f"    Reason:   {classification.reason}")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nSome tests failed. The classifier may need adjustments.")
        return False
    else:
        print("\nAll tests passed! The classifier is working correctly.")
        return True


if __name__ == "__main__":
    test_classifier()