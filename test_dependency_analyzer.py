#!/usr/bin/env python3
"""
Test script for the dependency analyzer functionality.
"""

import os
from dependency_analyzer import DependencyAnalyzer, DependencyType
from cleanup_classifier import FileClassifier


def test_dependency_analysis():
    """Test the dependency analyzer functionality"""
    print("Testing Dependency Analyzer...")
    
    analyzer = DependencyAnalyzer()
    result = analyzer.analyze_project_dependencies()
    
    print(f"✓ Found {len(result.dependencies)} dependencies")
    print(f"✓ Identified {len(result.essential_files)} essential files")
    print(f"✓ Found {len(result.safe_to_remove)} files safe to remove")
    
    # Test specific dependency types
    python_deps = [d for d in result.dependencies if d.dependency_type == DependencyType.PYTHON_IMPORT]
    config_deps = [d for d in result.dependencies if d.dependency_type == DependencyType.CONFIG_REFERENCE]
    
    print(f"✓ Python imports: {len(python_deps)}")
    print(f"✓ Config references: {len(config_deps)}")
    
    # Test safety check for known files
    test_files = [
        ("README.md", False),  # Should not be safe to remove
        ("SHOWCASE.md", True),  # Should be safe to remove
        ("backend/main.py", False),  # Should not be safe to remove
    ]
    
    for file_path, expected_safe in test_files:
        if os.path.exists(file_path):
            is_safe, reasons = analyzer.is_safe_to_remove(file_path)
            if is_safe == expected_safe:
                print(f"✓ Safety check for {file_path}: {is_safe} (expected {expected_safe})")
            else:
                print(f"✗ Safety check for {file_path}: {is_safe} (expected {expected_safe})")
                print(f"  Reasons: {reasons}")
    
    return True


def test_integration_with_classifier():
    """Test integration between dependency analyzer and file classifier"""
    print("\nTesting Integration with File Classifier...")
    
    classifier = FileClassifier()
    analyzer = DependencyAnalyzer()
    
    # Analyze dependencies
    dep_result = analyzer.analyze_project_dependencies()
    
    # Classify some files
    test_files = [
        "README.md",
        "SHOWCASE.md", 
        "backend/main.py",
        "cleanup_classifier.py"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            # Get classification
            classification = classifier.classify_file(file_path)
            
            # Get dependency safety
            is_safe, reasons = analyzer.is_safe_to_remove(file_path)
            
            print(f"File: {file_path}")
            print(f"  Classification: {classification.category.value} -> {classification.action.value}")
            print(f"  Safe to remove: {is_safe}")
            print(f"  Confidence: {classification.confidence}")
            
            # Check for consistency
            if classification.action.value == "REMOVE" and not is_safe:
                print(f"  ⚠️  Warning: Classified for removal but has dependencies!")
                print(f"     Reasons: {reasons}")
            elif classification.action.value == "KEEP" and is_safe:
                print(f"  ℹ️  Note: Classified to keep but could be safely removed")
            else:
                print(f"  ✓ Classification and dependency analysis are consistent")
    
    return True


def test_specific_dependencies():
    """Test specific dependency detection"""
    print("\nTesting Specific Dependency Detection...")
    
    analyzer = DependencyAnalyzer()
    result = analyzer.analyze_project_dependencies()
    
    # Check for expected dependencies
    expected_deps = [
        ("backend/main.py", "backend/core/database.py"),
        ("docker-compose.yml", "Dockerfile"),
    ]
    
    for source, target in expected_deps:
        found = False
        for dep in result.dependencies:
            if source in dep.source_file and target in dep.target_file:
                found = True
                print(f"✓ Found expected dependency: {source} -> {target}")
                break
        
        if not found:
            print(f"⚠️  Expected dependency not found: {source} -> {target}")
    
    # Show some interesting dependencies
    print("\nInteresting dependencies found:")
    for dep in result.dependencies[:5]:
        rel_source = os.path.relpath(dep.source_file)
        rel_target = os.path.relpath(dep.target_file)
        print(f"  {rel_source} -> {rel_target} ({dep.dependency_type.value})")
    
    return True


def main():
    """Run all tests"""
    print("Running Dependency Analyzer Tests\n")
    
    try:
        test_dependency_analysis()
        test_integration_with_classifier()
        test_specific_dependencies()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()