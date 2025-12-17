#!/usr/bin/env python3
"""
File Classification System for Project Cleanup

This module provides functionality to classify files in the project based on their
type, importance, and role in the system. It categorizes files into different
types (CORE, CONFIG, DOCS, DEMO, BUILD_ARTIFACT, CACHE) and determines
appropriate actions (KEEP, REMOVE, CONSOLIDATE).
"""

import os
import re
from enum import Enum
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from pathlib import Path
from dependency_analyzer import DependencyAnalyzer


class FileCategory(Enum):
    """Categories for file classification"""
    CORE = "CORE"                    # Essential application files
    CONFIG = "CONFIG"                # Configuration files
    DOCS = "DOCS"                    # Documentation files
    DEMO = "DEMO"                    # Demo and showcase files
    BUILD_ARTIFACT = "BUILD_ARTIFACT"  # Generated build files
    CACHE = "CACHE"                  # Cache directories and files


class FileAction(Enum):
    """Actions to take on classified files"""
    KEEP = "KEEP"                    # Keep the file
    REMOVE = "REMOVE"                # Remove the file
    CONSOLIDATE = "CONSOLIDATE"      # Consolidate with other files


@dataclass
class FileClassification:
    """Classification result for a file"""
    path: str
    category: FileCategory
    action: FileAction
    reason: str
    dependencies: List[str]
    confidence: float  # 0.0 to 1.0


class FileClassifier:
    """
    Classifies files based on patterns, content, and project structure.
    
    This classifier implements the rules defined in the requirements:
    - Identifies core functionality files that must be preserved
    - Detects demo and showcase files for removal
    - Finds redundant documentation and configuration files
    - Locates build artifacts and cache directories
    """
    
    def __init__(self):
        self._init_classification_rules()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def _init_classification_rules(self):
        """Initialize classification patterns and rules"""
        
        # Core application files - must always be kept
        self.core_patterns = {
            # Backend core files
            r'backend/core/.*\.py$': 'Backend core business logic',
            r'backend/.*models\.py$': 'Database models',
            r'backend/.*repositories\.py$': 'Data access layer',
            r'backend/.*endpoints\.py$': 'API endpoints',
            r'backend/main\.py$': 'Main application entry point',
            r'backend/finops_api\.py$': 'Main API module',
            
            # Frontend core files
            r'frontend/src/.*\.(tsx?|jsx?)$': 'Frontend components and logic',
            r'frontend/src/App\.tsx$': 'Main React application',
            r'frontend/src/index\.tsx$': 'Frontend entry point',
            r'frontend/package\.json$': 'Frontend dependencies',
            
            # Database files
            r'backend/alembic/.*\.py$': 'Database migrations',
            r'backend/init-db\.sql$': 'Database initialization',
            
            # Essential configuration
            r'docker-compose\.yml$': 'Main Docker configuration',
            r'Dockerfile$': 'Docker build configuration',
            r'backend/requirements\.txt$': 'Python dependencies',
        }
        
        # Configuration files
        self.config_patterns = {
            r'.*\.env.*$': 'Environment configuration',
            r'.*config.*\.(json|yml|yaml|ini)$': 'Configuration files',
            r'docker-compose\..*\.yml$': 'Docker compose variants',
            r'k8s/.*\.(yaml|yml)$': 'Kubernetes manifests',
            r'monitoring/.*\.(yml|yaml|conf)$': 'Monitoring configuration',
            r'\.gitignore$': 'Git ignore rules',
            r'alembic\.ini$': 'Alembic configuration',
        }
        
        # Documentation files
        self.docs_patterns = {
            r'README\.md$': 'Main project documentation',
            r'CONTRIBUTING\.md$': 'Contribution guidelines',
            r'SETUP_GUIDE\.md$': 'Setup instructions',
            r'DEPLOYMENT_GUIDE\.md$': 'Deployment instructions',
            r'QUICK_START\.md$': 'Quick start guide',
            r'PROJECT_STRUCTURE\.md$': 'Project structure documentation',
            r'LICENSE$': 'License file',
            r'docs/.*\.md$': 'Documentation files',
        }
        
        # Demo and showcase files - candidates for removal
        self.demo_patterns = {
            r'SHOWCASE\.md$': 'Showcase documentation',
            r'FEATURES_AND_CAPABILITIES\.md$': 'Feature showcase',
            r'TECHNICAL_INTERVIEW_GUIDE\.md$': 'Interview guide',
            r'DATA_EXPLANATION\.md$': 'Data explanation (redundant)',
            r'MIGRATION_.*\.md$': 'Migration-specific documentation',
            r'START_PLATFORM\.md$': 'Platform start guide (redundant)',
            r'start_dev_tasks\.py$': 'Development task script (redundant)',
            r'docs/images/demo/.*': 'Demo images',
            r'docs/images/screenshots/.*': 'Screenshot files',
            r'scripts/add-demo-images\.ps1$': 'Demo image script',
        }
        
        # Build artifacts and cache - should be removed
        self.build_artifact_patterns = {
            r'.*/__pycache__/.*': 'Python cache files',
            r'.*\.pytest_cache/.*': 'Pytest cache files',
            r'frontend/node_modules/.*': 'Node.js dependencies',
            r'.*\.pyc$': 'Python compiled files',
            r'.*\.pyo$': 'Python optimized files',
            r'.*\.egg-info/.*': 'Python package info',
            r'\.venv/.*': 'Virtual environment',
        }
        
        # Cache directories
        self.cache_patterns = {
            r'.*/__pycache__$': 'Python cache directory',
            r'.*\.pytest_cache$': 'Pytest cache directory',
            r'frontend/node_modules$': 'Node modules directory',
            r'\.venv$': 'Virtual environment directory',
        }
        
        # Essential files that should never be removed
        self.essential_core_files = {
            'README.md',
            'LICENSE',
            'docker-compose.yml',
            'Dockerfile',
            'backend/main.py',
            'backend/finops_api.py',
            'frontend/src/App.tsx',
            'frontend/src/index.tsx',
            'frontend/package.json',
            '.gitignore',
        }
        
        # Essential documentation files
        self.essential_docs = {
            'CONTRIBUTING.md',
            'SETUP_GUIDE.md',
            'DEPLOYMENT_GUIDE.md',
        }
        
        # Redundant documentation files (specific files identified for removal)
        self.redundant_docs = {
            'DATA_EXPLANATION.md': 'Redundant with README',
            'MIGRATION_FLOW_EXPLANATION.md': 'Specific implementation details',
            'MIGRATION_LOADING_FIX.md': 'Temporary fix documentation',
            'MIGRATION_WIZARD_COMPLETE.md': 'Completion status file',
            'MIGRATION_WIZARD_FIXES.md': 'Fix documentation',
            'FEATURES_AND_CAPABILITIES.md': 'Duplicates README content',
            'TECHNICAL_INTERVIEW_GUIDE.md': 'Not needed for production',
            'START_PLATFORM.md': 'Duplicates QUICK_START.md',
        }
    
    def classify_file(self, file_path: str, use_dependency_analysis: bool = True) -> FileClassification:
        """
        Classify a single file based on its path and characteristics.
        
        Args:
            file_path: Path to the file to classify
            use_dependency_analysis: Whether to use dependency analysis for safety checks
            
        Returns:
            FileClassification object with category, action, and reasoning
        """
        normalized_path = file_path.replace('\\', '/')
        
        # Get dependency information if requested
        dependencies = []
        if use_dependency_analysis:
            is_safe, dep_reasons = self.dependency_analyzer.is_safe_to_remove(file_path)
            if not is_safe:
                dependencies = dep_reasons
        
        # Check if it's an essential core file that must be kept
        if any(essential in normalized_path for essential in self.essential_core_files):
            return FileClassification(
                path=file_path,
                category=FileCategory.CORE,
                action=FileAction.KEEP,
                reason="Essential file for core functionality",
                dependencies=dependencies,
                confidence=1.0
            )
        
        # Check if it's an essential documentation file
        if any(essential in normalized_path for essential in self.essential_docs):
            return FileClassification(
                path=file_path,
                category=FileCategory.DOCS,
                action=FileAction.KEEP,
                reason="Essential documentation file",
                dependencies=dependencies,
                confidence=1.0
            )
        
        # Check for cache directories and files
        if self._matches_patterns(normalized_path, self.cache_patterns):
            # Override action if file has dependencies
            action = FileAction.KEEP if dependencies else FileAction.REMOVE
            return FileClassification(
                path=file_path,
                category=FileCategory.CACHE,
                action=action,
                reason="Cache directory that can be regenerated" if not dependencies else "Cache file with dependencies",
                dependencies=dependencies,
                confidence=0.95
            )
        
        # Check for build artifacts
        if self._matches_patterns(normalized_path, self.build_artifact_patterns):
            # Override action if file has dependencies
            action = FileAction.KEEP if dependencies else FileAction.REMOVE
            return FileClassification(
                path=file_path,
                category=FileCategory.BUILD_ARTIFACT,
                action=action,
                reason="Build artifact that can be regenerated" if not dependencies else "Build artifact with dependencies",
                dependencies=dependencies,
                confidence=0.9
            )
        
        # Check for redundant documentation first (before demo patterns)
        filename = os.path.basename(file_path)
        if filename in self.redundant_docs:
            # Override action if file has dependencies
            action = FileAction.KEEP if dependencies else FileAction.REMOVE
            return FileClassification(
                path=file_path,
                category=FileCategory.DOCS,
                action=action,
                reason=self.redundant_docs[filename] if not dependencies else f"{self.redundant_docs[filename]} (but has dependencies)",
                dependencies=dependencies,
                confidence=0.9
            )
        
        # Check for demo files
        if self._matches_patterns(normalized_path, self.demo_patterns):
            # Override action if file has dependencies
            action = FileAction.KEEP if dependencies else FileAction.REMOVE
            return FileClassification(
                path=file_path,
                category=FileCategory.DEMO,
                action=action,
                reason="Demo or showcase file not needed for production" if not dependencies else "Demo file with dependencies",
                dependencies=dependencies,
                confidence=0.85
            )
        
        # Check for core application files
        if self._matches_patterns(normalized_path, self.core_patterns):
            return FileClassification(
                path=file_path,
                category=FileCategory.CORE,
                action=FileAction.KEEP,
                reason="Core application functionality",
                dependencies=dependencies,
                confidence=0.95
            )
        
        # Check for configuration files
        if self._matches_patterns(normalized_path, self.config_patterns):
            return FileClassification(
                path=file_path,
                category=FileCategory.CONFIG,
                action=FileAction.KEEP,
                reason="Configuration file",
                dependencies=dependencies,
                confidence=0.8
            )
        
        # Check for documentation files
        if self._matches_patterns(normalized_path, self.docs_patterns):
            return FileClassification(
                path=file_path,
                category=FileCategory.DOCS,
                action=FileAction.KEEP,
                reason="Essential documentation",
                dependencies=dependencies,
                confidence=0.8
            )
        
        # Default classification for unmatched files
        return FileClassification(
            path=file_path,
            category=FileCategory.CONFIG,  # Conservative default
            action=FileAction.KEEP,
            reason="Unclassified file - keeping for safety",
            dependencies=dependencies,
            confidence=0.3
        )
    
    def _matches_patterns(self, file_path: str, patterns: Dict[str, str]) -> bool:
        """Check if file path matches any of the given patterns"""
        for pattern in patterns.keys():
            if re.search(pattern, file_path):
                return True
        return False
    
    def classify_directory(self, directory_path: str, use_dependency_analysis: bool = True) -> List[FileClassification]:
        """
        Classify all files in a directory recursively.
        
        Args:
            directory_path: Path to the directory to classify
            use_dependency_analysis: Whether to use dependency analysis for safety checks
            
        Returns:
            List of FileClassification objects for all files
        """
        classifications = []
        
        # Analyze dependencies once for the entire project if requested
        if use_dependency_analysis:
            print("Analyzing project dependencies...")
            self.dependency_analyzer.analyze_project_dependencies()
        
        for root, dirs, files in os.walk(directory_path):
            # Classify directories themselves (for cache directories)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                classification = self.classify_file(dir_path, use_dependency_analysis)
                classifications.append(classification)
            
            # Classify files
            for file_name in files:
                file_path = os.path.join(root, file_name)
                classification = self.classify_file(file_path, use_dependency_analysis)
                classifications.append(classification)
        
        return classifications
    
    def classify_with_dependency_analysis(self, directory_path: str = ".") -> List[FileClassification]:
        """
        Classify all files with full dependency analysis to ensure safe cleanup.
        
        Args:
            directory_path: Path to the directory to classify
            
        Returns:
            List of FileClassification objects with dependency information
        """
        return self.classify_directory(directory_path, use_dependency_analysis=True)
    
    def get_cleanup_summary(self, classifications: List[FileClassification]) -> Dict:
        """
        Generate a summary of the cleanup plan.
        
        Args:
            classifications: List of file classifications
            
        Returns:
            Dictionary with cleanup statistics and file lists
        """
        summary = {
            'total_files': len(classifications),
            'files_to_keep': [],
            'files_to_remove': [],
            'files_to_consolidate': [],
            'categories': {category.value: 0 for category in FileCategory},
            'actions': {action.value: 0 for action in FileAction}
        }
        
        for classification in classifications:
            # Count by category
            summary['categories'][classification.category.value] += 1
            
            # Count by action
            summary['actions'][classification.action.value] += 1
            
            # Group by action
            if classification.action == FileAction.KEEP:
                summary['files_to_keep'].append(classification.path)
            elif classification.action == FileAction.REMOVE:
                summary['files_to_remove'].append(classification.path)
            elif classification.action == FileAction.CONSOLIDATE:
                summary['files_to_consolidate'].append(classification.path)
        
        return summary


def main():
    """Example usage of the file classifier"""
    classifier = FileClassifier()
    
    # Classify current directory
    current_dir = "."
    classifications = classifier.classify_directory(current_dir)
    
    # Generate summary
    summary = classifier.get_cleanup_summary(classifications)
    
    print("File Classification Summary:")
    print(f"Total files analyzed: {summary['total_files']}")
    print(f"Files to keep: {summary['actions']['KEEP']}")
    print(f"Files to remove: {summary['actions']['REMOVE']}")
    print(f"Files to consolidate: {summary['actions']['CONSOLIDATE']}")
    
    print("\nFiles marked for removal:")
    for file_path in summary['files_to_remove'][:10]:  # Show first 10
        classification = next(c for c in classifications if c.path == file_path)
        print(f"  {file_path} - {classification.reason}")
    
    if len(summary['files_to_remove']) > 10:
        print(f"  ... and {len(summary['files_to_remove']) - 10} more files")


if __name__ == "__main__":
    main()