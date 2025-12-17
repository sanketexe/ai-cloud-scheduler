#!/usr/bin/env python3
"""
Demo script showing the integrated dependency analysis system.
"""

import os
from cleanup_classifier import FileClassifier
from dependency_analyzer import DependencyAnalyzer


def demo_dependency_analysis():
    """Demonstrate the dependency analysis functionality"""
    print("🔍 Project Cleanup with Dependency Analysis Demo")
    print("=" * 50)
    
    # Initialize the integrated classifier
    classifier = FileClassifier()
    
    print("\n1. Analyzing project dependencies...")
    
    # Classify some sample files with dependency analysis
    test_files = [
        "README.md",
        "SHOWCASE.md",
        "FEATURES_AND_CAPABILITIES.md",
        "backend/main.py",
        "cleanup_classifier.py",
        "dependency_analyzer.py",
        "demo_classifier.py",
        "test_classifier.py",
    ]
    
    print("\n2. File Classification Results:")
    print("-" * 80)
    print(f"{'File':<35} {'Category':<12} {'Action':<8} {'Safe':<6} {'Reason'}")
    print("-" * 80)
    
    for file_path in test_files:
        if os.path.exists(file_path):
            # Classify with dependency analysis
            classification = classifier.classify_file(file_path, use_dependency_analysis=True)
            
            # Check safety
            is_safe, _ = classifier.dependency_analyzer.is_safe_to_remove(file_path)
            
            # Format output
            file_short = file_path[:32] + "..." if len(file_path) > 35 else file_path
            category = classification.category.value
            action = classification.action.value
            safe_str = "✓" if is_safe else "✗"
            reason = classification.reason[:40] + "..." if len(classification.reason) > 43 else classification.reason
            
            print(f"{file_short:<35} {category:<12} {action:<8} {safe_str:<6} {reason}")
    
    print("\n3. Dependency Graph Analysis:")
    print("-" * 50)
    
    # Show some interesting dependencies
    analyzer = DependencyAnalyzer()
    result = analyzer.analyze_project_dependencies()
    
    print(f"Total dependencies found: {len(result.dependencies)}")
    print(f"Essential files identified: {len(result.essential_files)}")
    print(f"Files safe to remove: {len(result.safe_to_remove)}")
    
    print("\nKey dependencies:")
    for dep in result.dependencies[:8]:
        source = os.path.relpath(dep.source_file)
        target = os.path.relpath(dep.target_file)
        print(f"  {source} → {target} ({dep.dependency_type.value})")
    
    print("\n4. Safety Analysis Examples:")
    print("-" * 40)
    
    safety_tests = [
        ("README.md", "Core documentation"),
        ("SHOWCASE.md", "Demo file"),
        ("backend/main.py", "Main application"),
        ("cleanup_classifier.py", "Utility script"),
    ]
    
    for file_path, description in safety_tests:
        if os.path.exists(file_path):
            is_safe, reasons = analyzer.is_safe_to_remove(file_path)
            status = "SAFE TO REMOVE" if is_safe else "KEEP (HAS DEPENDENCIES)"
            print(f"\n{description} ({file_path}):")
            print(f"  Status: {status}")
            if reasons:
                for reason in reasons[:2]:  # Show first 2 reasons
                    print(f"  Reason: {reason}")
    
    print("\n5. Cleanup Recommendations:")
    print("-" * 35)
    
    # Get files that are both classified for removal and safe to remove
    removable_files = []
    for file_path in test_files:
        if os.path.exists(file_path):
            classification = classifier.classify_file(file_path, use_dependency_analysis=True)
            is_safe, _ = analyzer.is_safe_to_remove(file_path)
            
            if classification.action.value == "REMOVE" and is_safe:
                removable_files.append((file_path, classification.reason))
    
    if removable_files:
        print("Files recommended for removal:")
        for file_path, reason in removable_files:
            print(f"  ✓ {file_path} - {reason}")
    else:
        print("No files in the test set are recommended for removal.")
    
    print("\n6. Integration Benefits:")
    print("-" * 25)
    print("✓ Prevents accidental removal of files with dependencies")
    print("✓ Identifies transitive dependencies automatically")
    print("✓ Provides detailed reasoning for each decision")
    print("✓ Ensures core functionality is preserved")
    print("✓ Safe cleanup with rollback capability")
    
    print(f"\n🎉 Dependency analysis complete!")
    print(f"The system analyzed {len(result.dependencies)} dependencies")
    print(f"and identified {len(result.essential_files)} essential files.")


def demo_specific_scenarios():
    """Demo specific dependency scenarios"""
    print("\n" + "=" * 50)
    print("🔬 Specific Dependency Scenarios")
    print("=" * 50)
    
    analyzer = DependencyAnalyzer()
    result = analyzer.analyze_project_dependencies()
    
    # Scenario 1: Python import dependencies
    print("\n1. Python Import Dependencies:")
    python_deps = [d for d in result.dependencies if d.dependency_type.value == "PYTHON_IMPORT"]
    for dep in python_deps[:5]:
        source = os.path.basename(dep.source_file)
        target = os.path.basename(dep.target_file)
        print(f"   {source} imports {target}")
    
    # Scenario 2: Configuration dependencies
    print("\n2. Configuration Dependencies:")
    config_deps = [d for d in result.dependencies if d.dependency_type.value == "CONFIG_REFERENCE"]
    if config_deps:
        for dep in config_deps[:3]:
            source = os.path.basename(dep.source_file)
            target = os.path.basename(dep.target_file)
            print(f"   {source} references {target}")
    else:
        print("   No configuration dependencies found in current analysis")
    
    # Scenario 3: Essential file protection
    print("\n3. Essential File Protection:")
    essential_examples = [
        "backend/main.py",
        "frontend/src/App.tsx", 
        "docker-compose.yml",
        "README.md"
    ]
    
    for file_path in essential_examples:
        if os.path.exists(file_path):
            is_safe, reasons = analyzer.is_safe_to_remove(file_path)
            status = "Protected" if not is_safe else "Not Protected"
            print(f"   {file_path}: {status}")


if __name__ == "__main__":
    demo_dependency_analysis()
    demo_specific_scenarios()