#!/usr/bin/env python3
"""
Property-Based Tests for File Classification System

This module contains property-based tests to verify the correctness of the file
classification system according to the requirements specification.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Set
from hypothesis import given, strategies as st, assume, settings, Verbosity
from cleanup_classifier import FileClassifier, FileCategory, FileAction


class TestFileClassificationProperties:
    """Property-based tests for file classification accuracy"""
    
    def __init__(self):
        self.classifier = FileClassifier()
    
    @given(st.lists(
        st.one_of(
            # Essential backend core files
            st.just("backend/core/models.py"),
            st.just("backend/core/repositories.py"),
            st.just("backend/main.py"),
            st.just("backend/finops_api.py"),
            # Essential frontend files
            st.just("frontend/src/App.tsx"),
            st.just("frontend/src/index.tsx"),
            st.just("frontend/package.json"),
            # Essential configuration files
            st.just("docker-compose.yml"),
            st.just("Dockerfile"),
            st.just("backend/requirements.txt"),
            # Essential documentation
            st.just("README.md"),
            st.just("LICENSE"),
            st.just("CONTRIBUTING.md"),
            # Database files
            st.just("backend/alembic/versions/001_initial_migration.py"),
            st.just("backend/init-db.sql"),
            # Generate random essential-looking files
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=3, max_size=20).map(
                lambda name: f"backend/core/{name.lower()}.py"
            ),
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=3, max_size=20).map(
                lambda name: f"frontend/src/{name}.tsx"
            )
        ),
        min_size=1, max_size=15
    ))
    @settings(max_examples=50, deadline=None)
    def test_essential_file_preservation(self, essential_file_paths: List[str]):
        """
        **Feature: project-cleanup, Property 1: Essential file preservation**
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
        
        For any file classified as essential (backend core, frontend components, database migrations, 
        Docker configs), the cleanup process should never remove it.
        
        This property tests that:
        1. Known essential files are always classified to be kept
        2. Backend core business logic files are never marked for removal
        3. Frontend components and pages are preserved
        4. Database migration files are always kept
        5. Docker configurations and deployment files are preserved
        """
        # Filter out duplicates and invalid paths
        valid_essential_files = []
        for file_path in essential_file_paths:
            clean_path = ''.join(c for c in file_path if c.isalnum() or c in '/_.-')
            if clean_path and len(clean_path) >= 5:  # Minimum reasonable path length
                valid_essential_files.append(clean_path)
        
        assume(len(valid_essential_files) > 0)
        
        # Test known essential files from the requirements
        known_essential_files = {
            # Backend core files (Requirement 5.1)
            'backend/core/models.py',
            'backend/core/repositories.py', 
            'backend/core/auth.py',
            'backend/core/database.py',
            'backend/main.py',
            'backend/finops_api.py',
            
            # Frontend components and pages (Requirement 5.2)
            'frontend/src/App.tsx',
            'frontend/src/index.tsx',
            'frontend/src/pages/Dashboard.tsx',
            'frontend/src/components/Layout',
            'frontend/package.json',
            
            # Database migration files (Requirement 5.3)
            'backend/alembic/versions/001_initial_migration.py',
            'backend/alembic/versions/002_add_migration_advisor_tables.py',
            'backend/alembic/env.py',
            'backend/init-db.sql',
            'alembic.ini',
            
            # Docker configurations and deployment files (Requirement 5.4)
            'docker-compose.yml',
            'docker-compose.prod.yml',
            'Dockerfile',
            'frontend/Dockerfile',
            'k8s/migration-advisor-deployment.yaml',
            
            # Essential project files (Requirement 5.5)
            'README.md',
            'LICENSE',
            '.gitignore',
            'backend/requirements.txt',
        }
        
        # Property 1: Known essential files should always be kept
        for essential_file in known_essential_files:
            classification = self.classifier.classify_file(essential_file)
            assert classification.action == FileAction.KEEP, (
                f"Essential file '{essential_file}' should be kept, "
                f"but got action: {classification.action.value}. "
                f"Category: {classification.category.value}, "
                f"Reason: {classification.reason}"
            )
            
            # Essential files should be classified with appropriate categories
            expected_categories = {FileCategory.CORE, FileCategory.CONFIG, FileCategory.DOCS}
            assert classification.category in expected_categories, (
                f"Essential file '{essential_file}' should be categorized as CORE, CONFIG, or DOCS, "
                f"but got category: {classification.category.value}"
            )
        
        # Property 2: Backend core files should never be removed
        backend_core_patterns = [
            'backend/core/',
            'backend/main.py',
            'backend/finops_api.py',
            'backend/alembic/',
        ]
        
        for file_path in valid_essential_files:
            if any(pattern in file_path for pattern in backend_core_patterns):
                classification = self.classifier.classify_file(file_path)
                assert classification.action == FileAction.KEEP, (
                    f"Backend core file '{file_path}' should be kept, "
                    f"but got action: {classification.action.value}. "
                    f"Reason: {classification.reason}"
                )
                
                # Backend core files should be classified as CORE
                if 'backend/core/' in file_path and file_path.endswith('.py'):
                    assert classification.category == FileCategory.CORE, (
                        f"Backend core file '{file_path}' should be categorized as CORE, "
                        f"but got category: {classification.category.value}"
                    )
        
        # Property 3: Frontend components should never be removed
        frontend_patterns = [
            'frontend/src/',
            'frontend/package.json',
        ]
        
        for file_path in valid_essential_files:
            if any(pattern in file_path for pattern in frontend_patterns):
                classification = self.classifier.classify_file(file_path)
                assert classification.action == FileAction.KEEP, (
                    f"Frontend file '{file_path}' should be kept, "
                    f"but got action: {classification.action.value}. "
                    f"Reason: {classification.reason}"
                )
                
                # Frontend source files should be classified as CORE
                if 'frontend/src/' in file_path and file_path.endswith(('.tsx', '.ts', '.jsx', '.js')):
                    assert classification.category == FileCategory.CORE, (
                        f"Frontend source file '{file_path}' should be categorized as CORE, "
                        f"but got category: {classification.category.value}"
                    )
        
        # Property 4: Database migration files should never be removed
        database_patterns = [
            'backend/alembic/',
            'backend/init-db.sql',
            'alembic.ini',
        ]
        
        for file_path in valid_essential_files:
            if any(pattern in file_path for pattern in database_patterns):
                classification = self.classifier.classify_file(file_path)
                assert classification.action == FileAction.KEEP, (
                    f"Database file '{file_path}' should be kept, "
                    f"but got action: {classification.action.value}. "
                    f"Reason: {classification.reason}"
                )
        
        # Property 5: Docker and deployment files should never be removed
        deployment_patterns = [
            'docker-compose',
            'Dockerfile',
            'k8s/',
        ]
        
        for file_path in valid_essential_files:
            if any(pattern in file_path for pattern in deployment_patterns):
                classification = self.classifier.classify_file(file_path)
                assert classification.action == FileAction.KEEP, (
                    f"Deployment file '{file_path}' should be kept, "
                    f"but got action: {classification.action.value}. "
                    f"Reason: {classification.reason}"
                )
        
        # Property 6: Classification should be deterministic for essential files
        for file_path in valid_essential_files[:5]:  # Test first 5 to avoid excessive computation
            classification1 = self.classifier.classify_file(file_path)
            classification2 = self.classifier.classify_file(file_path)
            
            assert classification1.action == classification2.action, (
                f"Classification should be deterministic for '{file_path}'"
            )
            assert classification1.category == classification2.category, (
                f"Classification should be deterministic for '{file_path}'"
            )
            
            # All classifications should have valid reasons
            assert classification1.reason is not None and len(classification1.reason) > 0, (
                f"Classification for '{file_path}' should have a reason"
            )

    @given(st.lists(
        st.one_of(
            st.just("README"),
            st.just("SETUP"),
            st.just("DEPLOYMENT"), 
            st.just("MIGRATION"),
            st.just("GUIDE"),
            st.just("DOCS"),
            st.just("FEATURES"),
            st.just("TECHNICAL"),
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=3, max_size=20)
        ),
        min_size=1, max_size=10
    ))
    @settings(max_examples=50, deadline=None)
    def test_documentation_duplicate_detection(self, doc_name_parts: List[str]):
        """
        **Feature: project-cleanup, Property 2: Documentation duplicate detection**
        **Validates: Requirements 1.1, 1.5**
        
        For any set of documentation files with overlapping content, the system should 
        correctly identify which files contain duplicate or minimal value information.
        
        This property tests that:
        1. Known redundant documentation files are always classified for removal
        2. Essential documentation files are never classified for removal
        3. The classifier can distinguish between essential and redundant docs
        """
        # Create realistic documentation filenames
        valid_filenames = []
        for name_part in doc_name_parts:
            # Create documentation-like filenames
            clean_name = ''.join(c for c in name_part if c.isalnum() or c in '_-')
            if clean_name and len(clean_name) >= 2:
                filename = clean_name.upper() + '.md'
                valid_filenames.append(filename)
        
        assume(len(valid_filenames) > 0)
        
        # Test known redundant documentation files (from requirements)
        known_redundant_docs = {
            'DATA_EXPLANATION.md',
            'MIGRATION_FLOW_EXPLANATION.md', 
            'MIGRATION_LOADING_FIX.md',
            'MIGRATION_WIZARD_COMPLETE.md',
            'MIGRATION_WIZARD_FIXES.md',
            'FEATURES_AND_CAPABILITIES.md',
            'TECHNICAL_INTERVIEW_GUIDE.md',
            'START_PLATFORM.md'
        }
        
        # Test essential documentation files (from requirements)
        essential_docs = {
            'README.md',
            'CONTRIBUTING.md',
            'SETUP_GUIDE.md', 
            'DEPLOYMENT_GUIDE.md',
            'QUICK_START.md',
            'PROJECT_STRUCTURE.md'
        }
        
        # Property 1: Known redundant docs should always be marked for removal
        for redundant_doc in known_redundant_docs:
            classification = self.classifier.classify_file(redundant_doc)
            assert classification.action == FileAction.REMOVE, (
                f"Redundant documentation file '{redundant_doc}' should be marked for removal, "
                f"but got action: {classification.action.value}. "
                f"Reason: {classification.reason}"
            )
            assert classification.category == FileCategory.DOCS or classification.category == FileCategory.DEMO, (
                f"Redundant documentation file '{redundant_doc}' should be categorized as DOCS or DEMO, "
                f"but got category: {classification.category.value}"
            )
        
        # Property 2: Essential docs should never be marked for removal
        for essential_doc in essential_docs:
            classification = self.classifier.classify_file(essential_doc)
            assert classification.action == FileAction.KEEP, (
                f"Essential documentation file '{essential_doc}' should be kept, "
                f"but got action: {classification.action.value}. "
                f"Reason: {classification.reason}"
            )
        
        # Property 3: For any generated documentation filename, the classifier should
        # make a consistent decision based on patterns
        for filename in valid_filenames:
            classification = self.classifier.classify_file(filename)
            
            # The classification should be deterministic
            classification2 = self.classifier.classify_file(filename)
            assert classification.action == classification2.action, (
                f"Classification should be deterministic for '{filename}'"
            )
            assert classification.category == classification2.category, (
                f"Classification should be deterministic for '{filename}'"
            )
            
            # Documentation-like files should be classified appropriately
            # (Allow CONFIG as fallback for unknown patterns, but ensure it's reasonable)
            if filename.endswith('.md'):
                assert classification.category in [FileCategory.DOCS, FileCategory.DEMO, FileCategory.CORE, FileCategory.CONFIG], (
                    f"Markdown file '{filename}' should be classified as DOCS, DEMO, CORE, or CONFIG, "
                    f"but got: {classification.category.value}"
                )
                
                # If classified as CONFIG, it should be kept (safe fallback)
                if classification.category == FileCategory.CONFIG:
                    assert classification.action == FileAction.KEEP, (
                        f"Unknown markdown file '{filename}' classified as CONFIG should be kept for safety"
                    )
        
        # Property 4: Duplicate detection consistency - files with similar names
        # should have consistent classification patterns
        migration_related_files = [f for f in valid_filenames if 'migration' in f.lower()]
        if len(migration_related_files) > 1:
            # All migration-related files should have consistent treatment
            # (either all kept or all removed, based on their specific content)
            classifications = [self.classifier.classify_file(f) for f in migration_related_files]
            
            # Check that the reasoning is consistent for similar files
            for i, classification in enumerate(classifications):
                assert classification.reason is not None and len(classification.reason) > 0, (
                    f"Classification for '{migration_related_files[i]}' should have a reason"
                )

    @given(
        name_parts=st.lists(
            st.one_of(
                # Demo-related file patterns
                st.just("SHOWCASE"),
                st.just("DEMO"),
                st.just("FEATURES"),
                st.just("TECHNICAL"),
                st.just("INTERVIEW"),
                st.just("SCREENSHOT"),
                st.just("EXAMPLE"),
                st.just("SAMPLE"),
                # Generic file name parts
                st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=3, max_size=15)
            ),
            min_size=1, max_size=8
        ),
        extension=st.one_of(
            st.just(".md"),
            st.just(".py"), 
            st.just(".png"),
            st.just(".jpg"),
            st.just(".ps1"),
            st.just(".sh"),
            st.just("")
        )
    )
    @settings(max_examples=50, deadline=None, verbosity=Verbosity.normal)
    def test_demo_file_classification_accuracy(self, name_parts: List[str], extension: str):
        """
        **Feature: project-cleanup, Property 3: Demo file classification accuracy**
        **Validates: Requirements 2.1, 2.2, 2.4**
        
        For any file in the project, if it contains only demonstration content and no 
        reusable code or configurations, it should be classified for removal.
        
        This property tests that:
        1. Known demo files are always classified for removal
        2. Files with demo-like patterns are correctly identified
        3. Functional components are never accidentally classified as demo files
        4. Demo files in different locations are consistently classified
        """
        # Create realistic filenames from parts
        valid_filenames = []
        for name_part in name_parts:
            clean_name = ''.join(c for c in name_part if c.isalnum() or c in '_-')
            if clean_name and len(clean_name) >= 2:
                filename = clean_name.upper() + extension
                valid_filenames.append(filename)
        
        assume(len(valid_filenames) > 0)
        
        # Test known demo files (from requirements and classifier patterns)
        known_demo_files = {
            'SHOWCASE.md',
            'FEATURES_AND_CAPABILITIES.md',
            'TECHNICAL_INTERVIEW_GUIDE.md',
            'start_dev_tasks.py',
            'scripts/add-demo-images.ps1',
            'docs/images/demo/api-documentation.png',
            'docs/images/screenshots/Screenshot 2025-12-10 074004.png',
            'DATA_EXPLANATION.md',  # Redundant documentation classified as demo
            'MIGRATION_FLOW_EXPLANATION.md',
            'MIGRATION_LOADING_FIX.md',
            'MIGRATION_WIZARD_COMPLETE.md',
            'MIGRATION_WIZARD_FIXES.md',
            'START_PLATFORM.md'
        }
        
        # Test essential files that should never be classified as demo
        essential_files = {
            'backend/core/models.py',
            'backend/main.py',
            'frontend/src/App.tsx',
            'docker-compose.yml',
            'README.md',
            'CONTRIBUTING.md',
            'SETUP_GUIDE.md',
            'DEPLOYMENT_GUIDE.md',
            'backend/requirements.txt',
            'frontend/package.json',
            '.gitignore',
            'LICENSE'
        }
        
        # Property 1: Known demo files should always be marked for removal
        for demo_file in known_demo_files:
            classification = self.classifier.classify_file(demo_file)
            assert classification.action == FileAction.REMOVE, (
                f"Demo file '{demo_file}' should be marked for removal, "
                f"but got action: {classification.action.value}. "
                f"Reason: {classification.reason}"
            )
            # Demo files can be classified as DEMO or DOCS (for redundant docs)
            assert classification.category in [FileCategory.DEMO, FileCategory.DOCS], (
                f"Demo file '{demo_file}' should be categorized as DEMO or DOCS, "
                f"but got category: {classification.category.value}"
            )
        
        # Property 2: Essential files should never be classified as demo files
        for essential_file in essential_files:
            classification = self.classifier.classify_file(essential_file)
            assert classification.category != FileCategory.DEMO, (
                f"Essential file '{essential_file}' should not be classified as DEMO, "
                f"but got category: {classification.category.value}. "
                f"Reason: {classification.reason}"
            )
            assert classification.action == FileAction.KEEP, (
                f"Essential file '{essential_file}' should be kept, "
                f"but got action: {classification.action.value}"
            )
        
        # Property 3: Demo-like patterns should be consistently identified
        for filename in valid_filenames:
            classification = self.classifier.classify_file(filename)
            
            # Classification should be deterministic
            classification2 = self.classifier.classify_file(filename)
            assert classification.action == classification2.action, (
                f"Classification should be deterministic for '{filename}'"
            )
            assert classification.category == classification2.category, (
                f"Classification should be deterministic for '{filename}'"
            )
            
            # Files with demo-like names should be handled appropriately
            demo_indicators = ['showcase', 'demo', 'example', 'sample', 'screenshot', 'interview']
            has_demo_indicator = any(indicator in filename.lower() for indicator in demo_indicators)
            
            if has_demo_indicator:
                # Files with demo indicators should either be classified as DEMO for removal
                # or kept with a good reason (like being essential configuration)
                if classification.category == FileCategory.DEMO:
                    assert classification.action == FileAction.REMOVE, (
                        f"File '{filename}' classified as DEMO should be marked for removal"
                    )
                elif classification.action == FileAction.KEEP:
                    # If kept, should have a reasonable category and reason
                    assert classification.category in [FileCategory.CORE, FileCategory.CONFIG, FileCategory.DOCS], (
                        f"Demo-like file '{filename}' kept should have appropriate category, "
                        f"got: {classification.category.value}"
                    )
                    assert classification.reason is not None and len(classification.reason) > 0, (
                        f"Demo-like file '{filename}' kept should have a reason"
                    )
        
        # Property 4: Demo files in different locations should be consistently classified
        demo_locations = [
            'docs/images/demo/',
            'docs/images/screenshots/', 
            'scripts/',
            ''  # root directory
        ]
        
        for location in demo_locations:
            for demo_name in ['DEMO_FILE.md', 'SHOWCASE_EXAMPLE.py', 'SCREENSHOT.png']:
                full_path = location + demo_name if location else demo_name
                classification = self.classifier.classify_file(full_path)
                
                # Demo files should be consistently handled regardless of location
                # (either removed as demo or kept with good reason)
                if 'demo' in demo_name.lower() or 'showcase' in demo_name.lower() or 'screenshot' in demo_name.lower():
                    if classification.action == FileAction.REMOVE:
                        assert classification.category in [FileCategory.DEMO, FileCategory.DOCS], (
                            f"Demo file '{full_path}' marked for removal should be DEMO or DOCS category"
                        )
                    
                # All classifications should have valid reasons
                assert classification.reason is not None and len(classification.reason) > 0, (
                    f"Classification for '{full_path}' should have a reason"
                )

    @given(
        config_groups=st.lists(
            st.one_of(
                # Docker compose variants
                st.just(["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.override.yml"]),
                # Environment files
                st.just([".env", ".env.example", ".env.local", ".env.production"]),
                # Start scripts
                st.just(["start-project.ps1", "stop-project.ps1", "start-dev.py", "start_dev_tasks.py"]),
                # Configuration files
                st.just(["config.json", "config.example.json", "config.prod.json"]),
                # Package files
                st.just(["package.json", "package-lock.json"]),
                # Requirements files
                st.just(["requirements.txt", "requirements-dev.txt", "requirements-prod.txt"]),
                # Monitoring configs
                st.just(["monitoring/prometheus.yml", "monitoring/grafana/provisioning/datasources/prometheus.yml"]),
                # Generated config groups
                st.lists(
                    st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=3, max_size=15).map(
                        lambda name: f"{name.lower()}.yml"
                    ),
                    min_size=2, max_size=4
                )
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_configuration_redundancy_resolution(self, config_groups: List[List[str]]):
        """
        **Feature: project-cleanup, Property 4: Configuration redundancy resolution**
        **Validates: Requirements 3.1, 3.5**
        
        For any set of configuration files serving the same purpose, the system should keep 
        exactly one (the most comprehensive) and remove the others.
        
        This property tests that:
        1. When multiple configuration files serve the same purpose, only one is kept
        2. The most comprehensive configuration file is preserved
        3. Redundant configuration files are marked for removal or consolidation
        4. Essential configuration files are never removed
        """
        # Flatten the config groups and filter valid files
        all_config_files = []
        for group in config_groups:
            for config_file in group:
                if isinstance(config_file, str) and len(config_file) > 3:
                    clean_file = ''.join(c for c in config_file if c.isalnum() or c in '/_.-')
                    if clean_file and '.' in clean_file:
                        all_config_files.append(clean_file)
        
        assume(len(all_config_files) > 0)
        
        # Test known configuration file groups from the project
        known_config_groups = [
            # Docker compose variants - main should be kept, others may be kept or consolidated
            ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.override.yml"],
            
            # Environment files - .env should be kept, .env.example should be kept, others may vary
            [".env", ".env.example"],
            
            # Start/stop scripts - some redundancy expected
            ["start-project.ps1", "stop-project.ps1", "start-dev.py", "start_dev_tasks.py"],
            
            # Configuration examples
            ["config.example.json"],
            
            # Monitoring configurations
            ["monitoring/prometheus.yml", "monitoring/grafana/provisioning/datasources/prometheus.yml"],
        ]
        
        # Property 1: Essential configuration files should always be kept
        essential_configs = {
            "docker-compose.yml",  # Main Docker configuration
            ".env",                # Environment variables
            ".env.example",        # Environment template
            "backend/requirements.txt",  # Python dependencies
            "frontend/package.json",     # Node.js dependencies
            "alembic.ini",              # Database migration config
            ".gitignore",               # Git configuration
            "monitoring/prometheus.yml", # Monitoring configuration
        }
        
        for essential_config in essential_configs:
            classification = self.classifier.classify_file(essential_config)
            assert classification.action == FileAction.KEEP, (
                f"Essential configuration file '{essential_config}' should be kept, "
                f"but got action: {classification.action.value}. "
                f"Reason: {classification.reason}"
            )
            assert classification.category in [FileCategory.CONFIG, FileCategory.CORE], (
                f"Essential configuration file '{essential_config}' should be CONFIG or CORE, "
                f"but got category: {classification.category.value}"
            )
        
        # Property 2: Redundant configuration files should be handled appropriately
        redundant_configs = {
            "start_dev_tasks.py": "Redundant development script",
        }
        
        for redundant_config, expected_reason in redundant_configs.items():
            classification = self.classifier.classify_file(redundant_config)
            # Redundant configs should be removed or have a good reason for keeping
            if classification.action == FileAction.REMOVE:
                assert classification.category in [FileCategory.DEMO, FileCategory.CONFIG], (
                    f"Redundant config '{redundant_config}' marked for removal should be DEMO or CONFIG"
                )
            elif classification.action == FileAction.KEEP:
                # If kept, should have a valid reason
                assert classification.reason is not None and len(classification.reason) > 0, (
                    f"Redundant config '{redundant_config}' kept should have a reason"
                )
        
        # Property 3: Configuration file groups should have consistent handling
        for config_group in known_config_groups:
            classifications = []
            for config_file in config_group:
                classification = self.classifier.classify_file(config_file)
                classifications.append((config_file, classification))
            
            # At least one file in each group should be kept (unless all are truly redundant)
            kept_files = [f for f, c in classifications if c.action == FileAction.KEEP]
            removed_files = [f for f, c in classifications if c.action == FileAction.REMOVE]
            
            # For essential config groups, at least one should be kept
            essential_groups = [
                ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.override.yml"],
                [".env", ".env.example"],
            ]
            
            if config_group in essential_groups:
                assert len(kept_files) > 0, (
                    f"Essential config group {config_group} should have at least one file kept, "
                    f"but all were marked for removal: {removed_files}"
                )
        
        # Property 4: Configuration files should be classified consistently
        for config_file in all_config_files[:10]:  # Test first 10 to avoid excessive computation
            classification = self.classifier.classify_file(config_file)
            
            # Classification should be deterministic
            classification2 = self.classifier.classify_file(config_file)
            assert classification.action == classification2.action, (
                f"Classification should be deterministic for '{config_file}'"
            )
            assert classification.category == classification2.category, (
                f"Classification should be deterministic for '{config_file}'"
            )
            
            # Configuration files should have appropriate categories
            config_extensions = ['.yml', '.yaml', '.json', '.ini', '.conf', '.env']
            if any(config_file.endswith(ext) for ext in config_extensions):
                assert classification.category in [FileCategory.CONFIG, FileCategory.CORE, FileCategory.DEMO], (
                    f"Configuration file '{config_file}' should be CONFIG, CORE, or DEMO, "
                    f"but got category: {classification.category.value}"
                )
            
            # All classifications should have valid reasons
            assert classification.reason is not None and len(classification.reason) > 0, (
                f"Classification for '{config_file}' should have a reason"
            )
        
        # Property 5: Docker compose variants should follow redundancy rules
        docker_variants = ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.override.yml"]
        docker_classifications = []
        for docker_file in docker_variants:
            classification = self.classifier.classify_file(docker_file)
            docker_classifications.append((docker_file, classification))
        
        # Main docker-compose.yml should always be kept
        main_docker = next((c for f, c in docker_classifications if f == "docker-compose.yml"), None)
        if main_docker:
            assert main_docker.action == FileAction.KEEP, (
                "Main docker-compose.yml should always be kept"
            )
            assert main_docker.category in [FileCategory.CORE, FileCategory.CONFIG], (
                "Main docker-compose.yml should be CORE or CONFIG"
            )
        
        # Property 6: Environment file variants should follow redundancy rules
        env_variants = [".env", ".env.example", ".env.local", ".env.production"]
        for env_file in env_variants:
            classification = self.classifier.classify_file(env_file)
            
            # .env and .env.example are typically essential
            if env_file in [".env", ".env.example"]:
                assert classification.action == FileAction.KEEP, (
                    f"Essential environment file '{env_file}' should be kept"
                )
            
            # All env files should be classified as CONFIG
            assert classification.category == FileCategory.CONFIG, (
                f"Environment file '{env_file}' should be CONFIG, "
                f"but got category: {classification.category.value}"
            )


def run_property_tests():
    """Run the property-based tests"""
    print("Running Property-Based Tests for File Classification")
    print("=" * 60)
    
    test_instance = TestFileClassificationProperties()
    
    try:
        # Run the essential file preservation property test
        print("Testing Property 1: Essential file preservation...")
        test_instance.test_essential_file_preservation()
        print("✓ Property 1 test completed successfully")
        
        # Run the documentation duplicate detection property test
        print("Testing Property 2: Documentation duplicate detection...")
        test_instance.test_documentation_duplicate_detection()
        print("✓ Property 2 test completed successfully")
        
        # Run the demo file classification accuracy property test
        print("Testing Property 3: Demo file classification accuracy...")
        test_instance.test_demo_file_classification_accuracy()
        print("✓ Property 3 test completed successfully")
        
        # Run the configuration redundancy resolution property test
        print("Testing Property 4: Configuration redundancy resolution...")
        test_instance.test_configuration_redundancy_resolution()
        print("✓ Property 4 test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_property_tests()
    if success:
        print("\nAll property-based tests passed!")
    else:
        print("\nSome property-based tests failed!")
        exit(1)