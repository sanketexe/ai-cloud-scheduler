#!/usr/bin/env python3
"""
Cleanup Execution Engine for Project Cleanup

This module provides safe file removal functionality with rollback capability,
validation checks, and backup mechanisms. It implements the requirements for
safe cleanup operations while preserving core functionality.
"""

import os
import shutil
import json
import hashlib
import tempfile
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import subprocess

from cleanup_classifier import FileClassifier, FileClassification, FileAction
from dependency_analyzer import DependencyAnalyzer


class CleanupStatus(Enum):
    """Status of cleanup operations"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class ValidationResult(Enum):
    """Result of validation checks"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class BackupEntry:
    """Entry in the backup manifest"""
    original_path: str
    backup_path: str
    file_hash: str
    size: int
    timestamp: str
    is_directory: bool


@dataclass
class CleanupOperation:
    """Represents a single cleanup operation"""
    file_path: str
    action: FileAction
    classification: FileClassification
    status: CleanupStatus
    backup_entry: Optional[BackupEntry] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class CleanupPlan:
    """Complete cleanup execution plan"""
    operations: List[CleanupOperation]
    backup_directory: str
    validation_results: List[Dict]
    estimated_space_saved: int
    created_at: str
    project_root: str


@dataclass
class ValidationCheck:
    """Validation check result"""
    check_name: str
    result: ValidationResult
    message: str
    details: Optional[Dict] = None


class CleanupExecutionEngine:
    """
    Safe cleanup execution engine with rollback capability.
    
    This engine implements the requirements for safe file removal:
    - Validates files before deletion using dependency analysis
    - Creates backups for recovery
    - Provides rollback mechanism
    - Ensures core functionality is preserved
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.classifier = FileClassifier()
        self.dependency_analyzer = DependencyAnalyzer(str(self.project_root))
        self.backup_root = None
        self.current_plan = None
        
    def create_cleanup_plan(self, use_dependency_analysis: bool = True) -> CleanupPlan:
        """
        Create a comprehensive cleanup plan with validation.
        
        Args:
            use_dependency_analysis: Whether to use dependency analysis for safety
            
        Returns:
            CleanupPlan with all operations and validation results
        """
        print("Creating cleanup plan...")
        
        # Classify all files
        classifications = self.classifier.classify_directory(
            str(self.project_root), 
            use_dependency_analysis=use_dependency_analysis
        )
        
        # Create operations for files marked for removal
        operations = []
        for classification in classifications:
            if classification.action in [FileAction.REMOVE, FileAction.CONSOLIDATE]:
                operation = CleanupOperation(
                    file_path=classification.path,
                    action=classification.action,
                    classification=classification,
                    status=CleanupStatus.PENDING,
                    timestamp=datetime.now().isoformat()
                )
                operations.append(operation)
        
        # Run validation checks
        validation_results = self._run_validation_checks(operations)
        
        # Calculate estimated space savings
        estimated_space = self._calculate_space_savings(operations)
        
        # Create backup directory
        backup_dir = self._create_backup_directory()
        
        plan = CleanupPlan(
            operations=operations,
            backup_directory=backup_dir,
            validation_results=[asdict(v) for v in validation_results],
            estimated_space_saved=estimated_space,
            created_at=datetime.now().isoformat(),
            project_root=str(self.project_root)
        )
        
        self.current_plan = plan
        return plan
    
    def _run_validation_checks(self, operations: List[CleanupOperation]) -> List[ValidationCheck]:
        """Run comprehensive validation checks before cleanup"""
        checks = []
        
        # Check 1: Verify no essential files are marked for removal
        essential_check = self._validate_essential_files(operations)
        checks.append(essential_check)
        
        # Check 2: Verify dependency safety
        dependency_check = self._validate_dependencies(operations)
        checks.append(dependency_check)
        
        # Check 3: Verify Docker functionality will be preserved
        docker_check = self._validate_docker_functionality(operations)
        checks.append(docker_check)
        
        # Check 4: Verify build process will work
        build_check = self._validate_build_process(operations)
        checks.append(build_check)
        
        # Check 5: Check for potential data loss
        data_loss_check = self._validate_data_preservation(operations)
        checks.append(data_loss_check)
        
        return checks
    
    def _validate_essential_files(self, operations: List[CleanupOperation]) -> ValidationCheck:
        """Validate that no essential files are marked for removal"""
        essential_files = {
            'README.md',
            'LICENSE',
            'docker-compose.yml',
            'Dockerfile',
            'backend/main.py',
            'backend/finops_api.py',
            'frontend/src/App.tsx',
            'frontend/src/index.tsx',
            'frontend/package.json',
            'backend/requirements.txt',
            '.gitignore',
        }
        
        violations = []
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                rel_path = os.path.relpath(operation.file_path, self.project_root)
                if rel_path in essential_files:
                    violations.append(rel_path)
                
                # Check if it's in backend/core (all core files are essential)
                if 'backend/core/' in rel_path and rel_path.endswith('.py'):
                    violations.append(rel_path)
                
                # Check if it's a frontend src file
                if 'frontend/src/' in rel_path and rel_path.endswith(('.tsx', '.ts', '.jsx', '.js')):
                    violations.append(rel_path)
        
        if violations:
            return ValidationCheck(
                check_name="Essential Files Protection",
                result=ValidationResult.FAILED,
                message=f"Essential files marked for removal: {', '.join(violations)}",
                details={"violations": violations}
            )
        else:
            return ValidationCheck(
                check_name="Essential Files Protection",
                result=ValidationResult.PASSED,
                message="No essential files marked for removal"
            )
    
    def _validate_dependencies(self, operations: List[CleanupOperation]) -> ValidationCheck:
        """Validate that files with dependencies are not removed"""
        violations = []
        warnings = []
        
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                if operation.classification.dependencies:
                    # File has dependencies - this is a violation
                    violations.append({
                        'file': os.path.relpath(operation.file_path, self.project_root),
                        'dependencies': operation.classification.dependencies
                    })
                
                # Additional check: verify file is safe to remove
                is_safe, reasons = self.dependency_analyzer.is_safe_to_remove(operation.file_path)
                if not is_safe:
                    warnings.append({
                        'file': os.path.relpath(operation.file_path, self.project_root),
                        'reasons': reasons
                    })
        
        if violations:
            return ValidationCheck(
                check_name="Dependency Safety",
                result=ValidationResult.FAILED,
                message=f"Files with dependencies marked for removal: {len(violations)} files",
                details={"violations": violations, "warnings": warnings}
            )
        elif warnings:
            return ValidationCheck(
                check_name="Dependency Safety",
                result=ValidationResult.WARNING,
                message=f"Potential dependency issues: {len(warnings)} files",
                details={"warnings": warnings}
            )
        else:
            return ValidationCheck(
                check_name="Dependency Safety",
                result=ValidationResult.PASSED,
                message="All files are safe to remove based on dependency analysis"
            )
    
    def _validate_docker_functionality(self, operations: List[CleanupOperation]) -> ValidationCheck:
        """Validate that Docker functionality will be preserved"""
        docker_files = ['Dockerfile', 'docker-compose.yml', 'docker-compose.prod.yml']
        removed_docker_files = []
        
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                rel_path = os.path.relpath(operation.file_path, self.project_root)
                if rel_path in docker_files:
                    removed_docker_files.append(rel_path)
        
        if removed_docker_files:
            return ValidationCheck(
                check_name="Docker Functionality",
                result=ValidationResult.FAILED,
                message=f"Critical Docker files marked for removal: {', '.join(removed_docker_files)}",
                details={"removed_files": removed_docker_files}
            )
        else:
            return ValidationCheck(
                check_name="Docker Functionality",
                result=ValidationResult.PASSED,
                message="Docker configuration files will be preserved"
            )
    
    def _validate_build_process(self, operations: List[CleanupOperation]) -> ValidationCheck:
        """Validate that build process will continue to work"""
        build_files = [
            'frontend/package.json',
            'backend/requirements.txt',
            'requirements.txt',
            'frontend/tsconfig.json'
        ]
        
        removed_build_files = []
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                rel_path = os.path.relpath(operation.file_path, self.project_root)
                if rel_path in build_files:
                    removed_build_files.append(rel_path)
        
        if removed_build_files:
            return ValidationCheck(
                check_name="Build Process",
                result=ValidationResult.FAILED,
                message=f"Build configuration files marked for removal: {', '.join(removed_build_files)}",
                details={"removed_files": removed_build_files}
            )
        else:
            return ValidationCheck(
                check_name="Build Process",
                result=ValidationResult.PASSED,
                message="Build configuration files will be preserved"
            )
    
    def _validate_data_preservation(self, operations: List[CleanupOperation]) -> ValidationCheck:
        """Validate that important data files are not lost"""
        data_patterns = [
            r'.*\.sql$',
            r'.*migrations.*\.py$',
            r'.*\.env$',
            r'.*config.*\.(json|yml|yaml)$'
        ]
        
        import re
        potential_data_loss = []
        
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                rel_path = os.path.relpath(operation.file_path, self.project_root)
                for pattern in data_patterns:
                    if re.match(pattern, rel_path):
                        potential_data_loss.append(rel_path)
                        break
        
        if potential_data_loss:
            return ValidationCheck(
                check_name="Data Preservation",
                result=ValidationResult.WARNING,
                message=f"Potential data files marked for removal: {len(potential_data_loss)} files",
                details={"files": potential_data_loss}
            )
        else:
            return ValidationCheck(
                check_name="Data Preservation",
                result=ValidationResult.PASSED,
                message="No data files marked for removal"
            )
    
    def _calculate_space_savings(self, operations: List[CleanupOperation]) -> int:
        """Calculate estimated space savings from cleanup operations"""
        total_size = 0
        
        for operation in operations:
            if operation.action == FileAction.REMOVE:
                try:
                    if os.path.exists(operation.file_path):
                        if os.path.isfile(operation.file_path):
                            total_size += os.path.getsize(operation.file_path)
                        elif os.path.isdir(operation.file_path):
                            for root, dirs, files in os.walk(operation.file_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    if os.path.exists(file_path):
                                        total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue
        
        return total_size
    
    def _create_backup_directory(self) -> str:
        """Create a backup directory for the cleanup operation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"cleanup_backup_{timestamp}"
        backup_dir = os.path.join(tempfile.gettempdir(), backup_name)
        
        os.makedirs(backup_dir, exist_ok=True)
        self.backup_root = backup_dir
        return backup_dir
    
    def execute_cleanup_plan(self, plan: CleanupPlan, dry_run: bool = False) -> Dict:
        """
        Execute the cleanup plan with safety checks and backup.
        
        Args:
            plan: CleanupPlan to execute
            dry_run: If True, only simulate the cleanup without making changes
            
        Returns:
            Dictionary with execution results
        """
        if dry_run:
            return self._simulate_cleanup(plan)
        
        print(f"Executing cleanup plan with {len(plan.operations)} operations...")
        
        # Validate plan before execution
        validation_passed = self._validate_plan_before_execution(plan)
        if not validation_passed:
            return {
                'success': False,
                'error': 'Plan validation failed',
                'operations_completed': 0,
                'operations_failed': 0
            }
        
        # Create backup manifest
        backup_manifest = []
        operations_completed = 0
        operations_failed = 0
        
        try:
            # Execute operations in safe order
            for operation in plan.operations:
                try:
                    operation.status = CleanupStatus.IN_PROGRESS
                    
                    # Create backup before removal
                    if operation.action == FileAction.REMOVE:
                        backup_entry = self._create_backup(operation.file_path, plan.backup_directory)
                        operation.backup_entry = backup_entry
                        backup_manifest.append(backup_entry)
                        
                        # Remove the file/directory
                        self._safe_remove(operation.file_path)
                    
                    operation.status = CleanupStatus.COMPLETED
                    operations_completed += 1
                    
                except Exception as e:
                    operation.status = CleanupStatus.FAILED
                    operation.error_message = str(e)
                    operations_failed += 1
                    print(f"Failed to process {operation.file_path}: {e}")
            
            # Save backup manifest
            self._save_backup_manifest(backup_manifest, plan.backup_directory)
            
            # Update .gitignore for removed cache directories
            self._update_gitignore_for_removed_caches(plan)
            
            return {
                'success': True,
                'operations_completed': operations_completed,
                'operations_failed': operations_failed,
                'backup_directory': plan.backup_directory,
                'space_saved': plan.estimated_space_saved
            }
            
        except Exception as e:
            print(f"Critical error during cleanup execution: {e}")
            return {
                'success': False,
                'error': str(e),
                'operations_completed': operations_completed,
                'operations_failed': operations_failed,
                'backup_directory': plan.backup_directory
            }
    
    def _simulate_cleanup(self, plan: CleanupPlan) -> Dict:
        """Simulate cleanup execution without making changes"""
        print("Simulating cleanup execution (dry run)...")
        
        simulated_operations = 0
        for operation in plan.operations:
            if os.path.exists(operation.file_path):
                simulated_operations += 1
        
        return {
            'success': True,
            'dry_run': True,
            'operations_simulated': simulated_operations,
            'estimated_space_saved': plan.estimated_space_saved,
            'backup_directory': plan.backup_directory
        }
    
    def _validate_plan_before_execution(self, plan: CleanupPlan) -> bool:
        """Final validation before executing the plan"""
        # Check that all validation results passed or only have warnings
        for validation in plan.validation_results:
            if validation['result'] == ValidationResult.FAILED.value:
                print(f"Validation failed: {validation['check_name']} - {validation['message']}")
                return False
        
        # Check that backup directory exists and is writable
        if not os.path.exists(plan.backup_directory):
            print(f"Backup directory does not exist: {plan.backup_directory}")
            return False
        
        if not os.access(plan.backup_directory, os.W_OK):
            print(f"Backup directory is not writable: {plan.backup_directory}")
            return False
        
        return True
    
    def _create_backup(self, file_path: str, backup_dir: str) -> BackupEntry:
        """Create a backup of a file or directory"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate relative path from project root
        rel_path = os.path.relpath(file_path, self.project_root)
        backup_path = os.path.join(backup_dir, rel_path)
        
        # Ensure backup directory structure exists
        backup_parent = os.path.dirname(backup_path)
        os.makedirs(backup_parent, exist_ok=True)
        
        # Calculate file hash and size
        if os.path.isfile(file_path):
            file_hash = self._calculate_file_hash(file_path)
            size = os.path.getsize(file_path)
            shutil.copy2(file_path, backup_path)
            is_directory = False
        else:
            # For directories, create a hash of the directory structure
            file_hash = self._calculate_directory_hash(file_path)
            size = self._get_directory_size(file_path)
            shutil.copytree(file_path, backup_path)
            is_directory = True
        
        return BackupEntry(
            original_path=file_path,
            backup_path=backup_path,
            file_hash=file_hash,
            size=size,
            timestamp=datetime.now().isoformat(),
            is_directory=is_directory
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except (IOError, OSError):
            return "unknown"
    
    def _calculate_directory_hash(self, dir_path: str) -> str:
        """Calculate hash of directory structure and contents"""
        hash_sha256 = hashlib.sha256()
        
        try:
            for root, dirs, files in os.walk(dir_path):
                # Sort to ensure consistent hashing
                dirs.sort()
                files.sort()
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dir_path)
                    hash_sha256.update(rel_path.encode())
                    
                    try:
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_sha256.update(chunk)
                    except (IOError, OSError):
                        continue
            
            return hash_sha256.hexdigest()
        except (IOError, OSError):
            return "unknown"
    
    def _get_directory_size(self, dir_path: str) -> int:
        """Calculate total size of directory"""
        total_size = 0
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        continue
        except (OSError, IOError):
            pass
        return total_size
    
    def _safe_remove(self, file_path: str):
        """Safely remove a file or directory"""
        if not os.path.exists(file_path):
            return
        
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except (OSError, IOError) as e:
            raise Exception(f"Failed to remove {file_path}: {e}")
    
    def _save_backup_manifest(self, backup_manifest: List[BackupEntry], backup_dir: str):
        """Save backup manifest for rollback purposes"""
        manifest_path = os.path.join(backup_dir, "backup_manifest.json")
        
        manifest_data = {
            'created_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'entries': [asdict(entry) for entry in backup_manifest]
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    def _update_gitignore_for_removed_caches(self, plan: CleanupPlan):
        """Update .gitignore to include removed cache directories"""
        gitignore_path = os.path.join(self.project_root, '.gitignore')
        
        # Find cache directories that were removed
        cache_dirs_removed = []
        for operation in plan.operations:
            if (operation.action == FileAction.REMOVE and 
                operation.status == CleanupStatus.COMPLETED and
                operation.classification.category.value == 'CACHE'):
                
                rel_path = os.path.relpath(operation.file_path, self.project_root)
                if os.path.basename(rel_path) in ['__pycache__', '.pytest_cache', 'node_modules']:
                    cache_dirs_removed.append(os.path.basename(rel_path))
        
        if not cache_dirs_removed:
            return
        
        # Read existing .gitignore
        existing_entries = set()
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                existing_entries = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
        
        # Add new entries
        new_entries = []
        for cache_dir in set(cache_dirs_removed):
            if cache_dir not in existing_entries and f"*/{cache_dir}/" not in existing_entries:
                new_entries.append(f"*/{cache_dir}/")
        
        if new_entries:
            with open(gitignore_path, 'a') as f:
                f.write('\n# Added by cleanup process\n')
                for entry in new_entries:
                    f.write(f"{entry}\n")
    
    def rollback_cleanup(self, backup_directory: str) -> Dict:
        """
        Rollback a cleanup operation using the backup.
        
        Args:
            backup_directory: Path to the backup directory
            
        Returns:
            Dictionary with rollback results
        """
        print(f"Rolling back cleanup from backup: {backup_directory}")
        
        # Load backup manifest
        manifest_path = os.path.join(backup_directory, "backup_manifest.json")
        if not os.path.exists(manifest_path):
            return {
                'success': False,
                'error': 'Backup manifest not found'
            }
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            restored_files = 0
            failed_restorations = 0
            
            # Restore files from backup
            for entry_data in manifest_data['entries']:
                entry = BackupEntry(**entry_data)
                
                try:
                    # Ensure target directory exists
                    target_dir = os.path.dirname(entry.original_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Restore file or directory
                    if entry.is_directory:
                        if os.path.exists(entry.original_path):
                            shutil.rmtree(entry.original_path)
                        shutil.copytree(entry.backup_path, entry.original_path)
                    else:
                        shutil.copy2(entry.backup_path, entry.original_path)
                    
                    restored_files += 1
                    
                except Exception as e:
                    print(f"Failed to restore {entry.original_path}: {e}")
                    failed_restorations += 1
            
            return {
                'success': True,
                'restored_files': restored_files,
                'failed_restorations': failed_restorations,
                'backup_directory': backup_directory
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Rollback failed: {e}"
            }
    
    def get_cleanup_summary(self, plan: CleanupPlan) -> Dict:
        """Generate a summary of the cleanup plan"""
        summary = {
            'total_operations': len(plan.operations),
            'operations_by_status': {},
            'operations_by_action': {},
            'operations_by_category': {},
            'estimated_space_saved_mb': round(plan.estimated_space_saved / (1024 * 1024), 2),
            'validation_results': plan.validation_results,
            'backup_directory': plan.backup_directory,
            'created_at': plan.created_at
        }
        
        # Count by status
        for status in CleanupStatus:
            count = sum(1 for op in plan.operations if op.status == status)
            summary['operations_by_status'][status.value] = count
        
        # Count by action
        for action in FileAction:
            count = sum(1 for op in plan.operations if op.action == action)
            summary['operations_by_action'][action.value] = count
        
        # Count by category
        categories = set(op.classification.category for op in plan.operations)
        for category in categories:
            count = sum(1 for op in plan.operations if op.classification.category == category)
            summary['operations_by_category'][category.value] = count
        
        return summary


def main():
    """Example usage of the cleanup execution engine"""
    engine = CleanupExecutionEngine()
    
    # Create cleanup plan
    print("Creating cleanup plan...")
    plan = engine.create_cleanup_plan()
    
    # Show summary
    summary = engine.get_cleanup_summary(plan)
    print(f"\nCleanup Plan Summary:")
    print(f"Total operations: {summary['total_operations']}")
    print(f"Estimated space saved: {summary['estimated_space_saved_mb']} MB")
    print(f"Backup directory: {summary['backup_directory']}")
    
    print(f"\nOperations by action:")
    for action, count in summary['operations_by_action'].items():
        if count > 0:
            print(f"  {action}: {count}")
    
    print(f"\nValidation results:")
    for validation in summary['validation_results']:
        result_icon = "✓" if validation['result'] == 'PASSED' else "⚠" if validation['result'] == 'WARNING' else "✗"
        print(f"  {result_icon} {validation['check_name']}: {validation['message']}")
    
    # Ask user for confirmation
    print(f"\nReady to execute cleanup plan.")
    print("This will remove files and create backups for rollback.")
    
    # For demo purposes, run in dry-run mode
    print("\nRunning in dry-run mode...")
    result = engine.execute_cleanup_plan(plan, dry_run=True)
    
    if result['success']:
        print(f"Dry run completed successfully!")
        print(f"Would process {result['operations_simulated']} operations")
        print(f"Would save approximately {summary['estimated_space_saved_mb']} MB")
    else:
        print(f"Dry run failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()