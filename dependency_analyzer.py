#!/usr/bin/env python3
"""
Dependency Analysis System for Project Cleanup

This module provides functionality to analyze file dependencies in the project
to ensure that no essential files are removed during cleanup. It parses import
statements, configuration references, and builds a dependency graph to prevent
breaking changes.
"""

import os
import re
import ast
import json
import yaml
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies between files"""
    PYTHON_IMPORT = "PYTHON_IMPORT"          # Python import statements
    CONFIG_REFERENCE = "CONFIG_REFERENCE"    # Configuration file references
    DOCKER_REFERENCE = "DOCKER_REFERENCE"   # Docker file references
    SCRIPT_REFERENCE = "SCRIPT_REFERENCE"   # Script file references
    STATIC_REFERENCE = "STATIC_REFERENCE"   # Static file references (images, etc.)
    PACKAGE_DEPENDENCY = "PACKAGE_DEPENDENCY"  # Package.json dependencies


@dataclass
class Dependency:
    """Represents a dependency relationship between files"""
    source_file: str
    target_file: str
    dependency_type: DependencyType
    line_number: Optional[int] = None
    context: Optional[str] = None


@dataclass
class DependencyAnalysisResult:
    """Result of dependency analysis"""
    dependencies: List[Dependency]
    essential_files: Set[str]
    safe_to_remove: Set[str]
    potential_issues: List[str]


class DependencyAnalyzer:
    """
    Analyzes file dependencies to ensure safe cleanup operations.
    
    This analyzer implements the requirements for preserving core functionality:
    - Identifies Python import dependencies
    - Parses configuration file references
    - Analyzes Docker and script dependencies
    - Builds dependency graph to prevent breaking changes
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.dependencies: List[Dependency] = []
        self.file_cache: Dict[str, str] = {}
        
    def analyze_project_dependencies(self) -> DependencyAnalysisResult:
        """
        Analyze all dependencies in the project.
        
        Returns:
            DependencyAnalysisResult with complete dependency information
        """
        self.dependencies = []
        
        # Analyze different types of files
        self._analyze_python_files()
        self._analyze_config_files()
        self._analyze_docker_files()
        self._analyze_package_files()
        
        # Build dependency graph and identify essential files
        essential_files = self._identify_essential_files()
        safe_to_remove = self._identify_safe_files()
        potential_issues = self._identify_potential_issues()
        
        return DependencyAnalysisResult(
            dependencies=self.dependencies,
            essential_files=essential_files,
            safe_to_remove=safe_to_remove,
            potential_issues=potential_issues
        )
    
    def _analyze_python_files(self):
        """Analyze Python files for import dependencies"""
        for root, dirs, files in os.walk(self.project_root):
            # Skip cache and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._parse_python_imports(file_path)
    
    def _parse_python_imports(self, file_path: str):
        """Parse Python file for import statements"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_cache[file_path] = content
            
            # Parse AST to find imports
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._add_python_dependency(file_path, alias.name, node.lineno)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._add_python_dependency(file_path, node.module, node.lineno)
            except SyntaxError:
                # If AST parsing fails, use regex fallback
                self._parse_python_imports_regex(file_path, content)
                
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    def _parse_python_imports_regex(self, file_path: str, content: str):
        """Fallback regex-based import parsing"""
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module_name = match.group(1)
                    self._add_python_dependency(file_path, module_name, line_num)
    
    def _add_python_dependency(self, source_file: str, module_name: str, line_number: int):
        """Add a Python import dependency"""
        # Convert module name to potential file path
        target_file = self._resolve_python_module(source_file, module_name)
        if target_file:
            dependency = Dependency(
                source_file=source_file,
                target_file=target_file,
                dependency_type=DependencyType.PYTHON_IMPORT,
                line_number=line_number,
                context=f"import {module_name}"
            )
            self.dependencies.append(dependency)
    
    def _resolve_python_module(self, source_file: str, module_name: str) -> Optional[str]:
        """Resolve Python module name to actual file path"""
        # Handle relative imports within the project
        if module_name.startswith('.'):
            # Relative import - resolve relative to source file directory
            source_dir = os.path.dirname(source_file)
            # This is a simplified resolution - could be enhanced
            return None
        
        # Check if it's a local module within the project
        possible_paths = [
            os.path.join(self.project_root, module_name.replace('.', os.sep) + '.py'),
            os.path.join(self.project_root, module_name.replace('.', os.sep), '__init__.py'),
            os.path.join(self.project_root, 'backend', module_name.replace('.', os.sep) + '.py'),
            os.path.join(self.project_root, 'backend', module_name.replace('.', os.sep), '__init__.py'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return str(Path(path).resolve())
        
        return None
    
    def _analyze_config_files(self):
        """Analyze configuration files for references"""
        config_files = [
            'docker-compose.yml',
            'docker-compose.prod.yml',
            'docker-compose.override.yml',
            'alembic.ini',
            '.env',
            '.env.example',
        ]
        
        for config_file in config_files:
            file_path = os.path.join(self.project_root, config_file)
            if os.path.exists(file_path):
                self._parse_config_file(file_path)
    
    def _parse_config_file(self, file_path: str):
        """Parse configuration file for references"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_cache[file_path] = content
            
            # Parse YAML files
            if file_path.endswith(('.yml', '.yaml')):
                self._parse_yaml_references(file_path, content)
            # Parse INI files
            elif file_path.endswith('.ini'):
                self._parse_ini_references(file_path, content)
            # Parse env files
            elif '.env' in os.path.basename(file_path):
                self._parse_env_references(file_path, content)
                
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    def _parse_yaml_references(self, file_path: str, content: str):
        """Parse YAML file for file references"""
        try:
            data = yaml.safe_load(content)
            self._extract_yaml_file_references(file_path, data)
        except yaml.YAMLError:
            # Fallback to regex parsing
            self._parse_yaml_references_regex(file_path, content)
    
    def _extract_yaml_file_references(self, file_path: str, data, path_prefix=""):
        """Recursively extract file references from YAML data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    # Look for file references
                    if any(keyword in key.lower() for keyword in ['file', 'path', 'script', 'dockerfile']):
                        target_file = self._resolve_config_reference(file_path, value)
                        if target_file:
                            dependency = Dependency(
                                source_file=file_path,
                                target_file=target_file,
                                dependency_type=DependencyType.CONFIG_REFERENCE,
                                context=f"{path_prefix}.{key}" if path_prefix else key
                            )
                            self.dependencies.append(dependency)
                else:
                    self._extract_yaml_file_references(file_path, value, f"{path_prefix}.{key}" if path_prefix else key)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._extract_yaml_file_references(file_path, item, f"{path_prefix}[{i}]")
    
    def _parse_yaml_references_regex(self, file_path: str, content: str):
        """Fallback regex parsing for YAML files"""
        # Look for common file reference patterns
        patterns = [
            r'dockerfile:\s*([^\s\n]+)',
            r'build:\s*([^\s\n]+)',
            r'volumes:\s*-\s*([^:\s\n]+):',
            r'file:\s*([^\s\n]+)',
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    target_file = self._resolve_config_reference(file_path, match.group(1))
                    if target_file:
                        dependency = Dependency(
                            source_file=file_path,
                            target_file=target_file,
                            dependency_type=DependencyType.CONFIG_REFERENCE,
                            line_number=line_num,
                            context=line.strip()
                        )
                        self.dependencies.append(dependency)
    
    def _parse_ini_references(self, file_path: str, content: str):
        """Parse INI file for references"""
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                value = value.strip()
                
                # Look for file references
                if any(keyword in key.lower() for keyword in ['file', 'path', 'script']):
                    target_file = self._resolve_config_reference(file_path, value)
                    if target_file:
                        dependency = Dependency(
                            source_file=file_path,
                            target_file=target_file,
                            dependency_type=DependencyType.CONFIG_REFERENCE,
                            line_number=line_num,
                            context=line
                        )
                        self.dependencies.append(dependency)
    
    def _parse_env_references(self, file_path: str, content: str):
        """Parse environment file for references"""
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                value = value.strip().strip('"\'')
                
                # Look for file references in environment variables
                if any(keyword in key.lower() for keyword in ['file', 'path', 'cert', 'key']):
                    target_file = self._resolve_config_reference(file_path, value)
                    if target_file:
                        dependency = Dependency(
                            source_file=file_path,
                            target_file=target_file,
                            dependency_type=DependencyType.CONFIG_REFERENCE,
                            line_number=line_num,
                            context=line
                        )
                        self.dependencies.append(dependency)
    
    def _resolve_config_reference(self, source_file: str, reference: str) -> Optional[str]:
        """Resolve configuration file reference to actual file path"""
        # Skip URLs and absolute paths outside project
        if reference.startswith(('http://', 'https://', '/')):
            return None
        
        # Resolve relative to project root or source file directory
        possible_paths = [
            os.path.join(self.project_root, reference),
            os.path.join(os.path.dirname(source_file), reference),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return str(Path(path).resolve())
        
        return None
    
    def _analyze_docker_files(self):
        """Analyze Docker files for references"""
        docker_files = ['Dockerfile', 'frontend/Dockerfile']
        
        for docker_file in docker_files:
            file_path = os.path.join(self.project_root, docker_file)
            if os.path.exists(file_path):
                self._parse_dockerfile(file_path)
    
    def _parse_dockerfile(self, file_path: str):
        """Parse Dockerfile for file references"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_cache[file_path] = content
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Look for COPY and ADD instructions
                if line.startswith(('COPY ', 'ADD ')):
                    parts = line.split()
                    if len(parts) >= 3:
                        source_path = parts[1]
                        target_file = self._resolve_config_reference(file_path, source_path)
                        if target_file:
                            dependency = Dependency(
                                source_file=file_path,
                                target_file=target_file,
                                dependency_type=DependencyType.DOCKER_REFERENCE,
                                line_number=line_num,
                                context=line
                            )
                            self.dependencies.append(dependency)
                            
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    def _analyze_package_files(self):
        """Analyze package.json and requirements.txt files"""
        package_files = [
            'frontend/package.json',
            'backend/requirements.txt',
            'requirements.txt',
        ]
        
        for package_file in package_files:
            file_path = os.path.join(self.project_root, package_file)
            if os.path.exists(file_path):
                if package_file.endswith('package.json'):
                    self._parse_package_json(file_path)
                elif package_file.endswith('requirements.txt'):
                    self._parse_requirements_txt(file_path)
    
    def _parse_package_json(self, file_path: str):
        """Parse package.json for script references"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.file_cache[file_path] = json.dumps(data, indent=2)
            
            # Check scripts section for file references
            if 'scripts' in data:
                for script_name, script_command in data['scripts'].items():
                    # Look for file references in script commands
                    file_refs = re.findall(r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)', script_command)
                    for file_ref in file_refs:
                        target_file = self._resolve_config_reference(file_path, file_ref)
                        if target_file:
                            dependency = Dependency(
                                source_file=file_path,
                                target_file=target_file,
                                dependency_type=DependencyType.SCRIPT_REFERENCE,
                                context=f"script: {script_name}"
                            )
                            self.dependencies.append(dependency)
                            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    
    def _parse_requirements_txt(self, file_path: str):
        """Parse requirements.txt for local package references"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_cache[file_path] = content
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Look for local file references (e.g., -e ./local_package)
                    if line.startswith('-e ') and not line.startswith('-e git+'):
                        local_path = line[3:].strip()
                        target_file = self._resolve_config_reference(file_path, local_path)
                        if target_file:
                            dependency = Dependency(
                                source_file=file_path,
                                target_file=target_file,
                                dependency_type=DependencyType.PACKAGE_DEPENDENCY,
                                line_number=line_num,
                                context=line
                            )
                            self.dependencies.append(dependency)
                            
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    def _identify_essential_files(self) -> Set[str]:
        """Identify files that are essential based on dependency analysis"""
        essential_files = set()
        
        # Start with known essential files (use absolute paths)
        known_essential = {
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
        
        for file in known_essential:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                essential_files.add(str(Path(full_path).resolve()))
        
        # Add all backend core files as essential
        backend_core_path = os.path.join(self.project_root, 'backend', 'core')
        if os.path.exists(backend_core_path):
            for root, dirs, files in os.walk(backend_core_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        essential_files.add(str(Path(file_path).resolve()))
        
        # Add all frontend src files as essential
        frontend_src_path = os.path.join(self.project_root, 'frontend', 'src')
        if os.path.exists(frontend_src_path):
            for root, dirs, files in os.walk(frontend_src_path):
                for file in files:
                    if file.endswith(('.tsx', '.ts', '.jsx', '.js', '.css')):
                        file_path = os.path.join(root, file)
                        essential_files.add(str(Path(file_path).resolve()))
        
        # Add database migration files as essential
        alembic_path = os.path.join(self.project_root, 'backend', 'alembic')
        if os.path.exists(alembic_path):
            for root, dirs, files in os.walk(alembic_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        essential_files.add(str(Path(file_path).resolve()))
        
        # Add files that are dependencies of essential files
        self._add_transitive_dependencies(essential_files)
        
        return essential_files
    
    def _add_transitive_dependencies(self, essential_files: Set[str]):
        """Add transitive dependencies to essential files set"""
        changed = True
        while changed:
            changed = False
            new_essential = set(essential_files)
            
            for dependency in self.dependencies:
                if dependency.source_file in essential_files:
                    if dependency.target_file not in essential_files:
                        new_essential.add(dependency.target_file)
                        changed = True
            
            essential_files.update(new_essential)
    
    def _identify_safe_files(self) -> Set[str]:
        """Identify files that are safe to remove"""
        safe_files = set()
        essential_files = self._identify_essential_files()
        
        # Files that are not essential and not dependencies of essential files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                file_path = str(Path(os.path.join(root, file)).resolve())
                if file_path not in essential_files:
                    # Check if it's a dependency of any essential file
                    is_dependency = any(
                        dep.target_file == file_path and dep.source_file in essential_files
                        for dep in self.dependencies
                    )
                    if not is_dependency:
                        safe_files.add(file_path)
        
        return safe_files
    
    def _identify_potential_issues(self) -> List[str]:
        """Identify potential issues with the dependency analysis"""
        issues = []
        
        # Check for broken dependencies
        for dependency in self.dependencies:
            if not os.path.exists(dependency.target_file):
                issues.append(f"Broken dependency: {dependency.source_file} references non-existent {dependency.target_file}")
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies()
        for cycle in circular_deps:
            issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        return issues
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the dependency graph"""
        # Build adjacency list
        graph = {}
        for dependency in self.dependencies:
            if dependency.source_file not in graph:
                graph[dependency.source_file] = []
            graph[dependency.source_file].append(dependency.target_file)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph as an adjacency list"""
        graph = {}
        for dependency in self.dependencies:
            if dependency.source_file not in graph:
                graph[dependency.source_file] = []
            graph[dependency.source_file].append(dependency.target_file)
        return graph
    
    def is_safe_to_remove(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Check if a file is safe to remove.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (is_safe, list_of_reasons)
        """
        reasons = []
        file_path_resolved = str(Path(file_path).resolve())
        
        # Check if file is essential
        essential_files = self._identify_essential_files()
        if file_path_resolved in essential_files:
            reasons.append("File is marked as essential")
            return False, reasons
        
        # Check if file is a dependency of any other file
        dependents = [
            dep.source_file for dep in self.dependencies 
            if dep.target_file == file_path_resolved
        ]
        
        if dependents:
            reasons.append(f"File is referenced by: {', '.join([os.path.relpath(d) for d in dependents])}")
            return False, reasons
        
        reasons.append("File has no dependencies and is not essential")
        return True, reasons


def main():
    """Example usage of the dependency analyzer"""
    analyzer = DependencyAnalyzer()
    
    print("Analyzing project dependencies...")
    result = analyzer.analyze_project_dependencies()
    
    print(f"\nDependency Analysis Results:")
    print(f"Total dependencies found: {len(result.dependencies)}")
    print(f"Essential files: {len(result.essential_files)}")
    print(f"Safe to remove: {len(result.safe_to_remove)}")
    print(f"Potential issues: {len(result.potential_issues)}")
    
    if result.potential_issues:
        print("\nPotential Issues:")
        for issue in result.potential_issues[:5]:  # Show first 5
            print(f"  - {issue}")
    
    print("\nSample dependencies:")
    for dep in result.dependencies[:10]:  # Show first 10
        rel_source = os.path.relpath(dep.source_file)
        rel_target = os.path.relpath(dep.target_file)
        print(f"  {rel_source} -> {rel_target} ({dep.dependency_type.value})")
    
    # Test safety check for a specific file
    test_file = "SHOWCASE.md"
    if os.path.exists(test_file):
        is_safe, reasons = analyzer.is_safe_to_remove(test_file)
        print(f"\nSafety check for {test_file}:")
        print(f"  Safe to remove: {is_safe}")
        for reason in reasons:
            print(f"  Reason: {reason}")


if __name__ == "__main__":
    main()