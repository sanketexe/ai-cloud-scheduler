#!/usr/bin/env python3
"""
Demonstration of the file classifier on the actual project.
Shows classification results for key files mentioned in requirements.
"""

from cleanup_classifier import FileClassifier, FileCategory, FileAction
import os


def demo_classification():
    """Demonstrate the classifier on actual project files"""
    classifier = FileClassifier()
    
    # Key files from the project that we want to verify
    key_files = [
        # Files that should be removed (from requirements)
        "DATA_EXPLANATION.md",
        "MIGRATION_FLOW_EXPLANATION.md", 
        "MIGRATION_LOADING_FIX.md",
        "MIGRATION_WIZARD_COMPLETE.md",
        "MIGRATION_WIZARD_FIXES.md",
        "FEATURES_AND_CAPABILITIES.md",
        "TECHNICAL_INTERVIEW_GUIDE.md",
        "START_PLATFORM.md",
        "start_dev_tasks.py",
        
        # Files that should be kept
        "README.md",
        "CONTRIBUTING.md",
        "SETUP_GUIDE.md",
        "DEPLOYMENT_GUIDE.md",
        "docker-compose.yml",
        "backend/main.py",
        "frontend/src/App.tsx",
        
        # Cache/build artifacts
        "backend/__pycache__",
        ".pytest_cache",
        ".venv",
    ]
    
    print("File Classification Demo")
    print("=" * 80)
    print()
    
    files_to_remove = []
    files_to_keep = []
    
    for file_path in key_files:
        if os.path.exists(file_path):
            classification = classifier.classify_file(file_path)
            
            action_symbol = "🗑️" if classification.action == FileAction.REMOVE else "✅"
            category_color = {
                FileCategory.CORE: "🔵",
                FileCategory.DOCS: "📄", 
                FileCategory.CONFIG: "⚙️",
                FileCategory.DEMO: "🎭",
                FileCategory.CACHE: "💾",
                FileCategory.BUILD_ARTIFACT: "🔨"
            }
            
            print(f"{action_symbol} {category_color.get(classification.category, '❓')} {file_path}")
            print(f"   Category: {classification.category.value}")
            print(f"   Action: {classification.action.value}")
            print(f"   Reason: {classification.reason}")
            print(f"   Confidence: {classification.confidence:.1%}")
            print()
            
            if classification.action == FileAction.REMOVE:
                files_to_remove.append(file_path)
            else:
                files_to_keep.append(file_path)
        else:
            print(f"❌ {file_path} - File not found")
            print()
    
    print("=" * 80)
    print("SUMMARY:")
    print(f"Files to remove: {len(files_to_remove)}")
    print(f"Files to keep: {len(files_to_keep)}")
    print()
    
    if files_to_remove:
        print("Files marked for removal:")
        for file_path in files_to_remove:
            print(f"  - {file_path}")
        print()
    
    if files_to_keep:
        print("Files to keep:")
        for file_path in files_to_keep:
            print(f"  - {file_path}")


if __name__ == "__main__":
    demo_classification()