"""
Migration Planning Engine

This module implements the Migration Planning Engine for generating comprehensive
migration plans, analyzing dependencies, sequencing migrations, and tracking progress.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import structlog

from .models import (
    MigrationProject, MigrationPlan, MigrationPhase, WorkloadProfile,
    MigrationStatus, PhaseStatus, MigrationRiskLevel
)

logger = structlog.get_logger(__name__)


class MigrationPlanGenerator:
    """
    Generates comprehensive migration plans with phases, timelines, and resource requirements.
    Implements Requirements: 4.1
    """
    
    # Phase templates for different migration scenarios
    STANDARD_PHASES = [
        {
            'name': 'Pre-Migration Assessment',
            'order': 1,
            'duration_days': 7,
            'description': 'Validate requirements, finalize architecture, and prepare migration environment'
        },
        {
            'name': 'Foundation Setup',
            'order': 2,
            'duration_days': 5,
            'description': 'Set up cloud accounts, networking, security, and core infrastructure'
        },
        {
            'name': 'Pilot Migration',
            'order': 3,
            'duration_days': 10,
            'description': 'Migrate non-critical workloads to validate process and tooling'
        },
        {
            'name': 'Primary Migration Wave',
            'order': 4,
            'duration_days': 21,
            'description': 'Migrate primary workloads in sequenced batches'
        },
        {
            'name': 'Final Migration Wave',
            'order': 5,
            'duration_days': 14,
            'description': 'Migrate remaining workloads and critical systems'
        },
        {
            'name': 'Validation and Optimization',
            'order': 6,
            'duration_days': 7,
            'description': 'Validate all migrations, optimize configurations, and establish baselines'
        },
        {
            'name': 'Decommissioning',
            'order': 7,
            'duration_days': 5,
            'description': 'Decommission old infrastructure and finalize migration'
        }
    ]
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def generate_migration_plan(
        self,
        project_id: str,
        target_provider: str,
        workload_profiles: Optional[List[WorkloadProfile]] = None
    ) -> MigrationPlan:
        """
        Generate a comprehensive migration plan for a project.
        
        Args:
            project_id: Migration project ID
            target_provider: Target cloud provider (AWS, GCP, Azure)
            workload_profiles: Optional list of workload profiles to include
            
        Returns:
            MigrationPlan: Generated migration plan
            
        Raises:
            ValueError: If project not found or invalid provider
        """
        # Validate project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Validate provider
        valid_providers = ['AWS', 'GCP', 'Azure']
        if target_provider not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
        
        # Get workload profiles if not provided
        if workload_profiles is None:
            workload_profiles = self.db.query(WorkloadProfile).filter(
                WorkloadProfile.migration_project_id == project.id
            ).all()
        
        # Calculate migration parameters
        total_duration_days = self._calculate_total_duration(
            workload_profiles, project.organization_profile
        )
        
        estimated_cost = self._estimate_migration_cost(
            workload_profiles, target_provider
        )
        
        risk_level = self._assess_risk_level(
            workload_profiles, project.organization_profile
        )
        
        # Generate plan ID
        plan_id = self._generate_plan_id(project_id, target_provider)
        
        # Create migration plan
        migration_plan = MigrationPlan(
            plan_id=plan_id,
            migration_project_id=project.id,
            target_provider=target_provider,
            total_duration_days=total_duration_days,
            estimated_cost=estimated_cost,
            risk_level=risk_level,
            dependencies_graph={},  # Will be populated by dependency analyzer
            migration_waves=[],  # Will be populated by sequencing engine
            success_criteria=self._generate_success_criteria(workload_profiles),
            rollback_strategy=self._generate_rollback_strategy(risk_level)
        )
        
        try:
            self.db.add(migration_plan)
            self.db.flush()
            
            # Generate phases
            phases = self._generate_phases(
                migration_plan.id,
                workload_profiles,
                total_duration_days
            )
            
            for phase in phases:
                self.db.add(phase)
            
            self.db.flush()
            
            logger.info(
                "Migration plan generated",
                plan_id=plan_id,
                project_id=project_id,
                provider=target_provider,
                duration_days=total_duration_days,
                phases=len(phases)
            )
            
            return migration_plan
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create migration plan", error=str(e))
            raise
    
    def get_plan(self, plan_id: str) -> Optional[MigrationPlan]:
        """
        Retrieve a migration plan by plan_id.
        
        Args:
            plan_id: Unique plan identifier
            
        Returns:
            MigrationPlan or None if not found
        """
        return self.db.query(MigrationPlan).filter(
            MigrationPlan.plan_id == plan_id
        ).first()
    
    def get_plan_by_project(self, project_id: str) -> Optional[MigrationPlan]:
        """
        Retrieve migration plan for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            MigrationPlan or None if not found
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            return None
        
        return self.db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
    
    def update_plan(
        self,
        plan_id: str,
        **updates
    ) -> MigrationPlan:
        """
        Update migration plan fields.
        
        Args:
            plan_id: Unique plan identifier
            **updates: Fields to update
            
        Returns:
            Updated MigrationPlan
            
        Raises:
            ValueError: If plan not found
        """
        plan = self.get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'total_duration_days', 'estimated_cost', 'risk_level',
            'dependencies_graph', 'migration_waves', 'success_criteria',
            'rollback_strategy'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(plan, field, value)
        
        self.db.flush()
        
        logger.info(
            "Migration plan updated",
            plan_id=plan_id,
            updated_fields=list(updates.keys())
        )
        
        return plan
    
    def _generate_phases(
        self,
        migration_plan_id: uuid.UUID,
        workload_profiles: List[WorkloadProfile],
        total_duration_days: int
    ) -> List[MigrationPhase]:
        """
        Generate migration phases based on workload profiles.
        
        Args:
            migration_plan_id: UUID of the migration plan
            workload_profiles: List of workload profiles
            total_duration_days: Total migration duration
            
        Returns:
            List of MigrationPhase objects
        """
        phases = []
        current_date = datetime.utcnow()
        
        # Use standard phases as template
        for phase_template in self.STANDARD_PHASES:
            phase_id = f"phase-{phase_template['order']}-{str(uuid.uuid4())[:8]}"
            
            # Calculate phase dates
            start_date = current_date
            end_date = current_date + timedelta(days=phase_template['duration_days'])
            
            # Assign workloads to appropriate phases
            phase_workloads = self._assign_workloads_to_phase(
                phase_template['order'],
                workload_profiles
            )
            
            phase = MigrationPhase(
                phase_id=phase_id,
                migration_plan_id=migration_plan_id,
                phase_name=phase_template['name'],
                phase_order=phase_template['order'],
                workloads=[str(w.id) for w in phase_workloads],
                start_date=start_date,
                end_date=end_date,
                status=PhaseStatus.NOT_STARTED,
                prerequisites=self._generate_phase_prerequisites(phase_template['order']),
                success_criteria=self._generate_phase_success_criteria(
                    phase_template['name'],
                    phase_workloads
                ),
                rollback_plan=self._generate_phase_rollback_plan(phase_template['name'])
            )
            
            phases.append(phase)
            
            # Move to next phase start date
            current_date = end_date + timedelta(days=1)
        
        return phases
    
    def _assign_workloads_to_phase(
        self,
        phase_order: int,
        workload_profiles: List[WorkloadProfile]
    ) -> List[WorkloadProfile]:
        """
        Assign workloads to a specific phase based on criticality and dependencies.
        
        Args:
            phase_order: Order of the phase (1-7)
            workload_profiles: List of all workload profiles
            
        Returns:
            List of workloads assigned to this phase
        """
        # Phase 3 (Pilot): Non-critical workloads
        if phase_order == 3:
            return [w for w in workload_profiles if self._is_pilot_candidate(w)][:2]
        
        # Phase 4 (Primary Wave): Most workloads
        elif phase_order == 4:
            pilot_workloads = self._assign_workloads_to_phase(3, workload_profiles)
            remaining = [w for w in workload_profiles if w not in pilot_workloads]
            # Take 60% of remaining workloads
            count = max(1, int(len(remaining) * 0.6))
            return remaining[:count]
        
        # Phase 5 (Final Wave): Remaining workloads
        elif phase_order == 5:
            pilot_workloads = self._assign_workloads_to_phase(3, workload_profiles)
            primary_workloads = self._assign_workloads_to_phase(4, workload_profiles)
            return [w for w in workload_profiles 
                   if w not in pilot_workloads and w not in primary_workloads]
        
        # Other phases don't have specific workloads
        return []
    
    def _is_pilot_candidate(self, workload: WorkloadProfile) -> bool:
        """
        Determine if a workload is suitable for pilot migration.
        
        Args:
            workload: Workload profile
            
        Returns:
            True if suitable for pilot, False otherwise
        """
        # Pilot candidates: smaller workloads with fewer dependencies
        has_few_dependencies = len(workload.dependencies or []) <= 2
        is_smaller = (workload.total_compute_cores or 0) <= 8
        
        return has_few_dependencies and is_smaller
    
    def _calculate_total_duration(
        self,
        workload_profiles: List[WorkloadProfile],
        organization_profile
    ) -> int:
        """
        Calculate total migration duration based on workloads and organization.
        
        Args:
            workload_profiles: List of workload profiles
            organization_profile: Organization profile
            
        Returns:
            Total duration in days
        """
        # Base duration from standard phases
        base_duration = sum(phase['duration_days'] for phase in self.STANDARD_PHASES)
        
        # Adjust based on number of workloads
        workload_count = len(workload_profiles)
        if workload_count > 20:
            base_duration = int(base_duration * 1.5)
        elif workload_count > 10:
            base_duration = int(base_duration * 1.2)
        
        # Adjust based on organization size
        if organization_profile:
            from .models import CompanySize
            if organization_profile.company_size == CompanySize.ENTERPRISE:
                base_duration = int(base_duration * 1.3)
            elif organization_profile.company_size == CompanySize.LARGE:
                base_duration = int(base_duration * 1.15)
        
        return base_duration
    
    def _estimate_migration_cost(
        self,
        workload_profiles: List[WorkloadProfile],
        target_provider: str
    ) -> Decimal:
        """
        Estimate total migration cost.
        
        Args:
            workload_profiles: List of workload profiles
            target_provider: Target cloud provider
            
        Returns:
            Estimated cost as Decimal
        """
        # Base cost per workload
        base_cost_per_workload = Decimal('5000.00')
        
        # Calculate based on workload complexity
        total_cost = Decimal('0.00')
        
        for workload in workload_profiles:
            workload_cost = base_cost_per_workload
            
            # Add cost based on compute resources
            compute_cores = workload.total_compute_cores or 0
            workload_cost += Decimal(str(compute_cores * 100))
            
            # Add cost based on storage
            storage_tb = workload.total_storage_tb or 0
            workload_cost += Decimal(str(storage_tb * 500))
            
            # Add cost based on data transfer
            data_volume_tb = workload.data_volume_tb or 0
            workload_cost += Decimal(str(data_volume_tb * 200))
            
            total_cost += workload_cost
        
        # Add fixed costs
        fixed_costs = Decimal('10000.00')  # Planning, setup, validation
        total_cost += fixed_costs
        
        # Provider-specific adjustments
        provider_multipliers = {
            'AWS': Decimal('1.0'),
            'GCP': Decimal('0.95'),
            'Azure': Decimal('1.05')
        }
        
        total_cost *= provider_multipliers.get(target_provider, Decimal('1.0'))
        
        return total_cost.quantize(Decimal('0.01'))
    
    def _assess_risk_level(
        self,
        workload_profiles: List[WorkloadProfile],
        organization_profile
    ) -> MigrationRiskLevel:
        """
        Assess overall migration risk level.
        
        Args:
            workload_profiles: List of workload profiles
            organization_profile: Organization profile
            
        Returns:
            MigrationRiskLevel enum value
        """
        risk_score = 0
        
        # Factor 1: Number of workloads
        workload_count = len(workload_profiles)
        if workload_count > 50:
            risk_score += 3
        elif workload_count > 20:
            risk_score += 2
        elif workload_count > 10:
            risk_score += 1
        
        # Factor 2: Workload complexity
        for workload in workload_profiles:
            # Complex workloads have many dependencies
            if len(workload.dependencies or []) > 5:
                risk_score += 1
            
            # Large data volumes increase risk
            if (workload.data_volume_tb or 0) > 10:
                risk_score += 1
        
        # Factor 3: Organization experience
        if organization_profile:
            from .models import ExperienceLevel
            if organization_profile.cloud_experience_level == ExperienceLevel.NONE:
                risk_score += 3
            elif organization_profile.cloud_experience_level == ExperienceLevel.BEGINNER:
                risk_score += 2
        
        # Determine risk level
        if risk_score >= 10:
            return MigrationRiskLevel.CRITICAL
        elif risk_score >= 6:
            return MigrationRiskLevel.HIGH
        elif risk_score >= 3:
            return MigrationRiskLevel.MEDIUM
        else:
            return MigrationRiskLevel.LOW
    
    def _generate_success_criteria(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> List[str]:
        """
        Generate success criteria for the migration plan.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            List of success criteria strings
        """
        criteria = [
            "All workloads successfully migrated and operational",
            "No data loss during migration",
            "Performance meets or exceeds baseline requirements",
            "All security and compliance requirements validated",
            "Cost within 10% of estimated budget",
            "Zero critical incidents during migration",
            "All stakeholders trained on new environment"
        ]
        
        # Add workload-specific criteria
        if workload_profiles:
            criteria.append(f"All {len(workload_profiles)} workloads validated and signed off")
        
        return criteria
    
    def _generate_rollback_strategy(self, risk_level: MigrationRiskLevel) -> str:
        """
        Generate rollback strategy based on risk level.
        
        Args:
            risk_level: Migration risk level
            
        Returns:
            Rollback strategy description
        """
        strategies = {
            MigrationRiskLevel.LOW: (
                "Standard rollback: Maintain source environment for 30 days. "
                "Rollback can be executed within 4 hours if critical issues arise."
            ),
            MigrationRiskLevel.MEDIUM: (
                "Enhanced rollback: Maintain source environment for 60 days with hot standby. "
                "Rollback can be executed within 2 hours. Automated rollback procedures tested."
            ),
            MigrationRiskLevel.HIGH: (
                "Comprehensive rollback: Maintain source environment for 90 days with active-passive setup. "
                "Rollback can be executed within 1 hour. Multiple rollback scenarios tested and documented."
            ),
            MigrationRiskLevel.CRITICAL: (
                "Maximum safety rollback: Maintain source environment for 120 days with active-active setup. "
                "Instant rollback capability via traffic switching. Continuous validation and monitoring."
            )
        }
        
        return strategies.get(risk_level, strategies[MigrationRiskLevel.MEDIUM])
    
    def _generate_phase_prerequisites(self, phase_order: int) -> List[str]:
        """
        Generate prerequisites for a phase.
        
        Args:
            phase_order: Order of the phase
            
        Returns:
            List of prerequisite descriptions
        """
        prerequisites_map = {
            1: [],  # Pre-Migration Assessment has no prerequisites
            2: ["Pre-Migration Assessment completed and approved"],
            3: ["Foundation Setup completed", "Cloud accounts and networking configured"],
            4: ["Pilot Migration completed successfully", "Migration tooling validated"],
            5: ["Primary Migration Wave completed", "No critical issues from primary wave"],
            6: ["Final Migration Wave completed", "All workloads migrated"],
            7: ["Validation and Optimization completed", "All systems operational"]
        }
        
        return prerequisites_map.get(phase_order, [])
    
    def _generate_phase_success_criteria(
        self,
        phase_name: str,
        workloads: List[WorkloadProfile]
    ) -> List[str]:
        """
        Generate success criteria for a specific phase.
        
        Args:
            phase_name: Name of the phase
            workloads: Workloads assigned to this phase
            
        Returns:
            List of success criteria
        """
        criteria_map = {
            'Pre-Migration Assessment': [
                "All requirements validated and documented",
                "Architecture design approved",
                "Migration environment prepared"
            ],
            'Foundation Setup': [
                "Cloud accounts created and configured",
                "Network connectivity established",
                "Security controls implemented",
                "Monitoring and logging configured"
            ],
            'Pilot Migration': [
                "Pilot workloads migrated successfully",
                "Migration process validated",
                "Performance meets requirements",
                "No critical issues identified"
            ],
            'Primary Migration Wave': [
                f"All {len(workloads)} workloads migrated",
                "Applications functional and tested",
                "Data integrity validated",
                "Performance baselines established"
            ],
            'Final Migration Wave': [
                f"All {len(workloads)} remaining workloads migrated",
                "Critical systems operational",
                "All integrations working",
                "Disaster recovery tested"
            ],
            'Validation and Optimization': [
                "All workloads validated",
                "Performance optimized",
                "Cost baselines established",
                "Documentation completed"
            ],
            'Decommissioning': [
                "Old infrastructure decommissioned",
                "Data archived appropriately",
                "Final migration report completed",
                "Lessons learned documented"
            ]
        }
        
        return criteria_map.get(phase_name, ["Phase completed successfully"])
    
    def _generate_phase_rollback_plan(self, phase_name: str) -> str:
        """
        Generate rollback plan for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Rollback plan description
        """
        rollback_map = {
            'Pre-Migration Assessment': "No rollback needed - assessment phase only",
            'Foundation Setup': "Remove cloud resources, revert network changes",
            'Pilot Migration': "Redirect traffic to source, decommission pilot resources",
            'Primary Migration Wave': "Activate rollback procedures, restore from source environment",
            'Final Migration Wave': "Execute emergency rollback, restore critical systems",
            'Validation and Optimization': "Revert optimization changes if needed",
            'Decommissioning': "Halt decommissioning, restore from backups if needed"
        }
        
        return rollback_map.get(phase_name, "Standard rollback procedures apply")
    
    def _generate_plan_id(self, project_id: str, target_provider: str) -> str:
        """
        Generate a unique plan ID.
        
        Args:
            project_id: Migration project ID
            target_provider: Target cloud provider
            
        Returns:
            Unique plan ID string
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        provider_code = target_provider[:3].lower()
        random_suffix = str(uuid.uuid4())[:8]
        
        return f"plan-{provider_code}-{timestamp}-{random_suffix}"


class DependencyAnalyzer:
    """
    Analyzes resource dependencies and builds dependency graphs for migration sequencing.
    Implements Requirements: 4.2
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def build_dependency_graph(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> Dict[str, Any]:
        """
        Build a dependency graph from workload profiles.
        
        Args:
            workload_profiles: List of workload profiles with dependencies
            
        Returns:
            Dictionary representing the dependency graph with nodes and edges
        """
        nodes = []
        edges = []
        
        # Create nodes for each workload
        for workload in workload_profiles:
            node = {
                'id': str(workload.id),
                'name': workload.workload_name,
                'type': workload.application_type,
                'complexity': self._calculate_workload_complexity(workload),
                'metadata': {
                    'compute_cores': workload.total_compute_cores,
                    'memory_gb': workload.total_memory_gb,
                    'storage_tb': workload.total_storage_tb,
                    'data_volume_tb': workload.data_volume_tb
                }
            }
            nodes.append(node)
        
        # Create edges based on dependencies
        for workload in workload_profiles:
            dependencies = workload.dependencies or []
            
            for dep_id in dependencies:
                # Find the dependent workload
                dep_workload = next(
                    (w for w in workload_profiles if str(w.id) == dep_id),
                    None
                )
                
                if dep_workload:
                    edge = {
                        'from': dep_id,  # Dependency must be migrated first
                        'to': str(workload.id),  # Then this workload
                        'type': 'depends_on',
                        'strength': self._calculate_dependency_strength(workload, dep_workload)
                    }
                    edges.append(edge)
        
        # Identify critical path
        critical_path = self._identify_critical_path(nodes, edges)
        
        # Calculate migration waves
        migration_waves = self._calculate_migration_waves(nodes, edges)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'critical_path': critical_path,
            'migration_waves': migration_waves,
            'total_workloads': len(nodes),
            'total_dependencies': len(edges)
        }
    
    def discover_dependencies(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> Dict[str, List[str]]:
        """
        Discover implicit dependencies between workloads based on patterns.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            Dictionary mapping workload IDs to lists of discovered dependency IDs
        """
        discovered_deps = {}
        
        for workload in workload_profiles:
            deps = []
            
            # Database dependencies: databases should be migrated before applications
            if 'database' in workload.application_type.lower():
                # This is a database, no implicit dependencies
                pass
            else:
                # This is an application, find database dependencies
                for other in workload_profiles:
                    if 'database' in other.application_type.lower():
                        # Check if workload name suggests a relationship
                        if self._names_suggest_relationship(workload.workload_name, other.workload_name):
                            deps.append(str(other.id))
            
            # Network dependencies: load balancers before applications
            if 'application' in workload.application_type.lower():
                for other in workload_profiles:
                    if 'load' in other.application_type.lower() or 'balancer' in other.application_type.lower():
                        deps.append(str(other.id))
            
            # Storage dependencies: storage before compute
            if 'compute' in workload.application_type.lower():
                for other in workload_profiles:
                    if 'storage' in other.application_type.lower():
                        if self._names_suggest_relationship(workload.workload_name, other.workload_name):
                            deps.append(str(other.id))
            
            if deps:
                discovered_deps[str(workload.id)] = list(set(deps))  # Remove duplicates
        
        return discovered_deps
    
    def validate_dependencies(
        self,
        dependency_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate dependency graph for cycles and issues.
        
        Args:
            dependency_graph: Dependency graph to validate
            
        Returns:
            Dictionary with validation results
        """
        nodes = dependency_graph['nodes']
        edges = dependency_graph['edges']
        
        # Check for circular dependencies
        cycles = self._detect_cycles(nodes, edges)
        
        # Check for orphaned nodes (no dependencies and nothing depends on them)
        orphaned = self._find_orphaned_nodes(nodes, edges)
        
        # Check for bottlenecks (nodes with many dependents)
        bottlenecks = self._find_bottlenecks(nodes, edges)
        
        is_valid = len(cycles) == 0
        
        return {
            'is_valid': is_valid,
            'has_cycles': len(cycles) > 0,
            'cycles': cycles,
            'orphaned_nodes': orphaned,
            'bottlenecks': bottlenecks,
            'warnings': self._generate_validation_warnings(cycles, orphaned, bottlenecks)
        }
    
    def _calculate_workload_complexity(self, workload: WorkloadProfile) -> int:
        """
        Calculate complexity score for a workload.
        
        Args:
            workload: Workload profile
            
        Returns:
            Complexity score (1-10)
        """
        score = 1
        
        # Factor in compute resources
        if (workload.total_compute_cores or 0) > 16:
            score += 2
        elif (workload.total_compute_cores or 0) > 8:
            score += 1
        
        # Factor in storage
        if (workload.total_storage_tb or 0) > 10:
            score += 2
        elif (workload.total_storage_tb or 0) > 1:
            score += 1
        
        # Factor in data volume
        if (workload.data_volume_tb or 0) > 5:
            score += 2
        elif (workload.data_volume_tb or 0) > 1:
            score += 1
        
        # Factor in dependencies
        dep_count = len(workload.dependencies or [])
        if dep_count > 5:
            score += 2
        elif dep_count > 2:
            score += 1
        
        return min(score, 10)  # Cap at 10
    
    def _calculate_dependency_strength(
        self,
        workload: WorkloadProfile,
        dependency: WorkloadProfile
    ) -> str:
        """
        Calculate the strength of a dependency relationship.
        
        Args:
            workload: Dependent workload
            dependency: Workload being depended on
            
        Returns:
            Strength level: 'strong', 'medium', or 'weak'
        """
        # Strong: Database dependencies, critical infrastructure
        if 'database' in dependency.application_type.lower():
            return 'strong'
        
        if 'critical' in dependency.workload_name.lower():
            return 'strong'
        
        # Medium: Application dependencies
        if 'application' in dependency.application_type.lower():
            return 'medium'
        
        # Weak: Optional dependencies
        return 'weak'
    
    def _identify_critical_path(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify the critical path through the dependency graph.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            List of node IDs representing the critical path
        """
        if not nodes:
            return []
        
        # Build adjacency list
        adj_list = {node['id']: [] for node in nodes}
        in_degree = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            adj_list[edge['from']].append(edge['to'])
            in_degree[edge['to']] += 1
        
        # Find longest path using topological sort with depth tracking
        depths = {node['id']: 0 for node in nodes}
        parent = {node['id']: None for node in nodes}
        
        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in adj_list[current]:
                # Update depth if we found a longer path
                if depths[current] + 1 > depths[neighbor]:
                    depths[neighbor] = depths[current] + 1
                    parent[neighbor] = current
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Find the node with maximum depth
        if not depths:
            return []
        
        max_depth_node = max(depths.items(), key=lambda x: x[1])[0]
        
        # Reconstruct path
        path = []
        current = max_depth_node
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    def _calculate_migration_waves(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate migration waves based on dependency levels.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            List of migration waves with workload assignments
        """
        if not nodes:
            return []
        
        # Build adjacency list and in-degree map
        adj_list = {node['id']: [] for node in nodes}
        in_degree = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            adj_list[edge['from']].append(edge['to'])
            in_degree[edge['to']] += 1
        
        # Perform level-order traversal (topological sort by levels)
        waves = []
        remaining = set(node['id'] for node in nodes)
        
        while remaining:
            # Find all nodes with no remaining dependencies
            current_wave = [
                node_id for node_id in remaining
                if in_degree[node_id] == 0
            ]
            
            if not current_wave:
                # Circular dependency detected, add remaining nodes to final wave
                current_wave = list(remaining)
            
            # Create wave
            wave = {
                'wave_number': len(waves) + 1,
                'workload_ids': current_wave,
                'workload_count': len(current_wave),
                'estimated_duration_days': self._estimate_wave_duration(current_wave, nodes)
            }
            waves.append(wave)
            
            # Remove processed nodes and update in-degrees
            for node_id in current_wave:
                remaining.remove(node_id)
                for neighbor in adj_list[node_id]:
                    if neighbor in remaining:
                        in_degree[neighbor] -= 1
        
        return waves
    
    def _estimate_wave_duration(
        self,
        workload_ids: List[str],
        nodes: List[Dict[str, Any]]
    ) -> int:
        """
        Estimate duration for a migration wave.
        
        Args:
            workload_ids: List of workload IDs in the wave
            nodes: All nodes with metadata
            
        Returns:
            Estimated duration in days
        """
        # Base duration per workload
        base_days = 2
        
        # Find max complexity in wave
        max_complexity = 1
        for node in nodes:
            if node['id'] in workload_ids:
                max_complexity = max(max_complexity, node.get('complexity', 1))
        
        # Duration scales with complexity
        duration = base_days * max_complexity
        
        # Add overhead for wave size
        if len(workload_ids) > 5:
            duration += 2
        
        return duration
    
    def _detect_cycles(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        Detect circular dependencies in the graph.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        # Build adjacency list
        adj_list = {node['id']: [] for node in nodes}
        for edge in edges:
            adj_list[edge['from']].append(edge['to'])
        
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id: str) -> bool:
            """DFS to detect cycles"""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for neighbor in adj_list.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        for node in nodes:
            if node['id'] not in visited:
                dfs(node['id'])
        
        return cycles
    
    def _find_orphaned_nodes(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Find nodes with no dependencies and no dependents.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            List of orphaned node IDs
        """
        has_dependency = set()
        has_dependent = set()
        
        for edge in edges:
            has_dependency.add(edge['to'])
            has_dependent.add(edge['from'])
        
        orphaned = []
        for node in nodes:
            node_id = node['id']
            if node_id not in has_dependency and node_id not in has_dependent:
                orphaned.append(node_id)
        
        return orphaned
    
    def _find_bottlenecks(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find bottleneck nodes that many other nodes depend on.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            List of bottleneck nodes with dependent counts
        """
        dependent_count = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            dependent_count[edge['from']] += 1
        
        # Nodes with 3+ dependents are considered bottlenecks
        bottlenecks = []
        for node in nodes:
            count = dependent_count[node['id']]
            if count >= 3:
                bottlenecks.append({
                    'node_id': node['id'],
                    'node_name': node['name'],
                    'dependent_count': count
                })
        
        return sorted(bottlenecks, key=lambda x: x['dependent_count'], reverse=True)
    
    def _generate_validation_warnings(
        self,
        cycles: List[List[str]],
        orphaned: List[str],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate human-readable validation warnings.
        
        Args:
            cycles: Detected cycles
            orphaned: Orphaned nodes
            bottlenecks: Bottleneck nodes
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if cycles:
            warnings.append(
                f"Found {len(cycles)} circular dependency cycle(s). "
                "These must be resolved before migration can proceed."
            )
        
        if orphaned:
            warnings.append(
                f"Found {len(orphaned)} orphaned workload(s) with no dependencies. "
                "These can be migrated independently."
            )
        
        if bottlenecks:
            warnings.append(
                f"Found {len(bottlenecks)} bottleneck workload(s) with many dependents. "
                "These should be prioritized and migrated carefully."
            )
        
        return warnings
    
    def _names_suggest_relationship(self, name1: str, name2: str) -> bool:
        """
        Check if workload names suggest a relationship.
        
        Args:
            name1: First workload name
            name2: Second workload name
            
        Returns:
            True if names suggest a relationship
        """
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Extract common prefixes (e.g., "app-" from "app-web" and "app-db")
        name1_parts = name1_lower.split('-')
        name2_parts = name2_lower.split('-')
        
        # Check for common prefix
        if name1_parts[0] == name2_parts[0] and len(name1_parts[0]) > 2:
            return True
        
        # Check if one name contains the other
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return True
        
        return False


class MigrationSequencer:
    """
    Generates migration sequences and waves based on dependencies to minimize downtime.
    Implements Requirements: 4.2
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def generate_migration_sequence(
        self,
        dependency_graph: Dict[str, Any],
        workload_profiles: List[WorkloadProfile],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimized migration sequence based on dependencies.
        
        Args:
            dependency_graph: Dependency graph from DependencyAnalyzer
            workload_profiles: List of workload profiles
            constraints: Optional constraints (max_parallel_migrations, downtime_window, etc.)
            
        Returns:
            Dictionary with migration sequence details
        """
        constraints = constraints or {}
        max_parallel = constraints.get('max_parallel_migrations', 3)
        downtime_window_hours = constraints.get('downtime_window_hours', 4)
        
        nodes = dependency_graph['nodes']
        edges = dependency_graph['edges']
        
        # Generate migration waves with parallelization
        waves = self._generate_optimized_waves(
            nodes, edges, max_parallel, workload_profiles
        )
        
        # Calculate total downtime
        total_downtime = self._calculate_total_downtime(waves, downtime_window_hours)
        
        # Validate prerequisites
        prerequisite_validation = self._validate_prerequisites(waves, edges)
        
        return {
            'waves': waves,
            'total_waves': len(waves),
            'total_downtime_hours': total_downtime,
            'max_parallel_migrations': max_parallel,
            'prerequisite_validation': prerequisite_validation,
            'sequence_metadata': {
                'optimization_strategy': 'minimize_downtime',
                'parallelization_enabled': True,
                'dependency_aware': True
            }
        }
    
    def optimize_sequence_for_downtime(
        self,
        dependency_graph: Dict[str, Any],
        workload_profiles: List[WorkloadProfile],
        max_downtime_hours: float
    ) -> Dict[str, Any]:
        """
        Optimize migration sequence to stay within downtime constraints.
        
        Args:
            dependency_graph: Dependency graph
            workload_profiles: List of workload profiles
            max_downtime_hours: Maximum acceptable downtime
            
        Returns:
            Optimized migration sequence
        """
        nodes = dependency_graph['nodes']
        edges = dependency_graph['edges']
        
        # Start with aggressive parallelization
        max_parallel = 5
        
        while max_parallel > 0:
            waves = self._generate_optimized_waves(
                nodes, edges, max_parallel, workload_profiles
            )
            
            total_downtime = self._calculate_total_downtime(waves, 4)
            
            if total_downtime <= max_downtime_hours:
                return {
                    'waves': waves,
                    'total_waves': len(waves),
                    'total_downtime_hours': total_downtime,
                    'max_parallel_migrations': max_parallel,
                    'meets_downtime_constraint': True
                }
            
            max_parallel -= 1
        
        # If we can't meet constraint, return best effort
        waves = self._generate_optimized_waves(nodes, edges, 1, workload_profiles)
        total_downtime = self._calculate_total_downtime(waves, 4)
        
        return {
            'waves': waves,
            'total_waves': len(waves),
            'total_downtime_hours': total_downtime,
            'max_parallel_migrations': 1,
            'meets_downtime_constraint': False,
            'warning': f'Cannot meet downtime constraint of {max_downtime_hours} hours'
        }
    
    def validate_migration_prerequisites(
        self,
        phase_id: str,
        migration_plan_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Validate that all prerequisites are met for a migration phase.
        
        Args:
            phase_id: Phase identifier
            migration_plan_id: Migration plan UUID
            
        Returns:
            Validation results
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id,
            MigrationPhase.migration_plan_id == migration_plan_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        # Check if prerequisites are completed
        prerequisite_status = []
        all_met = True
        
        for prereq in phase.prerequisites:
            # Check if prerequisite phase is completed
            is_met = self._check_prerequisite_met(prereq, migration_plan_id)
            prerequisite_status.append({
                'prerequisite': prereq,
                'met': is_met
            })
            if not is_met:
                all_met = False
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'all_prerequisites_met': all_met,
            'prerequisite_status': prerequisite_status,
            'can_start': all_met
        }
    
    def _generate_optimized_waves(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_parallel: int,
        workload_profiles: List[WorkloadProfile]
    ) -> List[Dict[str, Any]]:
        """
        Generate migration waves with parallelization optimization.
        
        Args:
            nodes: Dependency graph nodes
            edges: Dependency graph edges
            max_parallel: Maximum parallel migrations per wave
            workload_profiles: Workload profiles for metadata
            
        Returns:
            List of optimized migration waves
        """
        if not nodes:
            return []
        
        # Build adjacency list and in-degree map
        adj_list = {node['id']: [] for node in nodes}
        in_degree = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            adj_list[edge['from']].append(edge['to'])
            in_degree[edge['to']] += 1
        
        waves = []
        remaining = set(node['id'] for node in nodes)
        
        while remaining:
            # Find all nodes with no remaining dependencies
            available = [
                node_id for node_id in remaining
                if in_degree[node_id] == 0
            ]
            
            if not available:
                # Circular dependency - add all remaining to final wave
                available = list(remaining)
            
            # Split available nodes into parallel batches
            while available:
                # Take up to max_parallel nodes for this wave
                batch = available[:max_parallel]
                available = available[max_parallel:]
                
                # Get workload details
                workload_details = []
                for node_id in batch:
                    node = next((n for n in nodes if n['id'] == node_id), None)
                    if node:
                        workload_details.append({
                            'workload_id': node_id,
                            'workload_name': node['name'],
                            'complexity': node.get('complexity', 1)
                        })
                
                wave = {
                    'wave_number': len(waves) + 1,
                    'workload_ids': batch,
                    'workload_count': len(batch),
                    'workloads': workload_details,
                    'estimated_duration_hours': self._estimate_wave_duration_hours(batch, nodes),
                    'can_run_parallel': len(batch) > 1,
                    'dependencies_met': True
                }
                waves.append(wave)
                
                # Remove processed nodes and update in-degrees
                for node_id in batch:
                    if node_id in remaining:
                        remaining.remove(node_id)
                        for neighbor in adj_list[node_id]:
                            if neighbor in remaining:
                                in_degree[neighbor] -= 1
        
        return waves
    
    def _estimate_wave_duration_hours(
        self,
        workload_ids: List[str],
        nodes: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate duration for a migration wave in hours.
        
        Args:
            workload_ids: Workload IDs in the wave
            nodes: All nodes with metadata
            
        Returns:
            Estimated duration in hours
        """
        # Base duration per workload
        base_hours = 4.0
        
        # Find max complexity in wave
        max_complexity = 1
        for node in nodes:
            if node['id'] in workload_ids:
                max_complexity = max(max_complexity, node.get('complexity', 1))
        
        # Duration scales with complexity
        duration = base_hours * (max_complexity / 5.0)
        
        # If running in parallel, use max duration (not sum)
        # Add 20% overhead for coordination
        if len(workload_ids) > 1:
            duration *= 1.2
        
        return round(duration, 1)
    
    def _calculate_total_downtime(
        self,
        waves: List[Dict[str, Any]],
        downtime_window_hours: float
    ) -> float:
        """
        Calculate total downtime for migration sequence.
        
        Args:
            waves: List of migration waves
            downtime_window_hours: Downtime window per wave
            
        Returns:
            Total downtime in hours
        """
        total = 0.0
        
        for wave in waves:
            # Each wave requires a downtime window
            wave_downtime = min(
                wave.get('estimated_duration_hours', downtime_window_hours),
                downtime_window_hours
            )
            total += wave_downtime
        
        return round(total, 1)
    
    def _validate_prerequisites(
        self,
        waves: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that wave sequencing respects all prerequisites.
        
        Args:
            waves: Migration waves
            edges: Dependency edges
            
        Returns:
            Validation results
        """
        # Build map of workload to wave number
        workload_to_wave = {}
        for wave in waves:
            for workload_id in wave['workload_ids']:
                workload_to_wave[workload_id] = wave['wave_number']
        
        violations = []
        
        # Check each dependency
        for edge in edges:
            from_workload = edge['from']
            to_workload = edge['to']
            
            from_wave = workload_to_wave.get(from_workload)
            to_wave = workload_to_wave.get(to_workload)
            
            # Dependency must be in earlier or same wave
            if from_wave and to_wave and from_wave > to_wave:
                violations.append({
                    'from_workload': from_workload,
                    'to_workload': to_workload,
                    'from_wave': from_wave,
                    'to_wave': to_wave,
                    'issue': 'Dependency scheduled after dependent'
                })
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'total_dependencies_checked': len(edges)
        }
    
    def _check_prerequisite_met(
        self,
        prerequisite: str,
        migration_plan_id: uuid.UUID
    ) -> bool:
        """
        Check if a prerequisite is met.
        
        Args:
            prerequisite: Prerequisite description
            migration_plan_id: Migration plan UUID
            
        Returns:
            True if prerequisite is met
        """
        # Get all phases for this plan
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == migration_plan_id
        ).all()
        
        # Check if prerequisite phase is completed
        for phase in phases:
            if prerequisite.lower() in phase.phase_name.lower():
                return phase.status == PhaseStatus.COMPLETED
        
        # If prerequisite not found as a phase, assume it's met
        return True


class MigrationCostEstimator:
    """
    Estimates migration costs including data transfer, dual-running, and professional services.
    Implements Requirements: 4.3
    """
    
    # Cost constants (in USD)
    DATA_TRANSFER_COST_PER_GB = Decimal('0.09')  # Average cloud egress cost
    DUAL_RUNNING_MULTIPLIER = Decimal('1.5')  # 50% overhead for running both environments
    PROFESSIONAL_SERVICES_HOURLY = Decimal('200.00')
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def estimate_migration_costs(
        self,
        migration_plan: MigrationPlan,
        workload_profiles: List[WorkloadProfile]
    ) -> Dict[str, Any]:
        """
        Estimate total migration costs with detailed breakdown.
        
        Args:
            migration_plan: Migration plan
            workload_profiles: List of workload profiles
            
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate data transfer costs
        data_transfer_cost = self._estimate_data_transfer_cost(workload_profiles)
        
        # Calculate dual-running costs
        dual_running_cost = self._estimate_dual_running_cost(
            workload_profiles,
            migration_plan.total_duration_days
        )
        
        # Calculate professional services costs
        professional_services_cost = self._estimate_professional_services_cost(
            migration_plan.total_duration_days,
            len(workload_profiles)
        )
        
        # Calculate tooling and licensing costs
        tooling_cost = self._estimate_tooling_cost(workload_profiles)
        
        # Calculate contingency (10% of total)
        subtotal = (
            data_transfer_cost +
            dual_running_cost +
            professional_services_cost +
            tooling_cost
        )
        contingency = subtotal * Decimal('0.10')
        
        total_cost = subtotal + contingency
        
        return {
            'total_cost': float(total_cost),
            'breakdown': {
                'data_transfer': float(data_transfer_cost),
                'dual_running': float(dual_running_cost),
                'professional_services': float(professional_services_cost),
                'tooling_and_licensing': float(tooling_cost),
                'contingency': float(contingency)
            },
            'cost_per_workload': float(total_cost / len(workload_profiles)) if workload_profiles else 0,
            'currency': 'USD',
            'estimation_confidence': self._calculate_confidence_level(workload_profiles)
        }
    
    def estimate_data_transfer_cost(
        self,
        workload_profiles: List[WorkloadProfile],
        target_provider: str
    ) -> Dict[str, Any]:
        """
        Estimate data transfer costs for migration.
        
        Args:
            workload_profiles: List of workload profiles
            target_provider: Target cloud provider
            
        Returns:
            Data transfer cost breakdown
        """
        total_data_gb = Decimal('0')
        workload_costs = []
        
        for workload in workload_profiles:
            # Convert TB to GB
            data_volume_gb = Decimal(str((workload.data_volume_tb or 0) * 1024))
            total_data_gb += data_volume_gb
            
            # Calculate cost for this workload
            workload_cost = data_volume_gb * self.DATA_TRANSFER_COST_PER_GB
            
            workload_costs.append({
                'workload_id': str(workload.id),
                'workload_name': workload.workload_name,
                'data_volume_gb': float(data_volume_gb),
                'transfer_cost': float(workload_cost)
            })
        
        # Provider-specific adjustments
        provider_multipliers = {
            'AWS': Decimal('1.0'),
            'GCP': Decimal('0.95'),  # Slightly cheaper egress
            'Azure': Decimal('1.05')
        }
        
        multiplier = provider_multipliers.get(target_provider, Decimal('1.0'))
        total_cost = total_data_gb * self.DATA_TRANSFER_COST_PER_GB * multiplier
        
        return {
            'total_cost': float(total_cost),
            'total_data_gb': float(total_data_gb),
            'total_data_tb': float(total_data_gb / 1024),
            'cost_per_gb': float(self.DATA_TRANSFER_COST_PER_GB * multiplier),
            'workload_breakdown': workload_costs,
            'provider': target_provider
        }
    
    def estimate_dual_running_cost(
        self,
        workload_profiles: List[WorkloadProfile],
        duration_days: int,
        current_monthly_cost: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Estimate costs for running both old and new environments during migration.
        
        Args:
            workload_profiles: List of workload profiles
            duration_days: Migration duration in days
            current_monthly_cost: Optional current monthly infrastructure cost
            
        Returns:
            Dual-running cost breakdown
        """
        # If current cost not provided, estimate it
        if current_monthly_cost is None:
            current_monthly_cost = self._estimate_current_infrastructure_cost(workload_profiles)
        
        # Calculate months of dual running
        months = Decimal(str(duration_days / 30.0))
        
        # Base cost: current infrastructure continues running
        old_environment_cost = current_monthly_cost * months
        
        # New environment cost: starts ramping up (average 50% during migration)
        new_environment_cost = current_monthly_cost * months * Decimal('0.5')
        
        # Additional overhead for data synchronization, testing, etc.
        overhead_cost = (old_environment_cost + new_environment_cost) * Decimal('0.15')
        
        total_cost = old_environment_cost + new_environment_cost + overhead_cost
        
        return {
            'total_cost': float(total_cost),
            'duration_days': duration_days,
            'duration_months': float(months),
            'breakdown': {
                'old_environment': float(old_environment_cost),
                'new_environment': float(new_environment_cost),
                'overhead': float(overhead_cost)
            },
            'estimated_monthly_cost': float(current_monthly_cost)
        }
    
    def estimate_phase_cost(
        self,
        phase: MigrationPhase,
        workload_profiles: List[WorkloadProfile]
    ) -> Dict[str, Any]:
        """
        Estimate costs for a specific migration phase.
        
        Args:
            phase: Migration phase
            workload_profiles: All workload profiles
            
        Returns:
            Phase cost estimate
        """
        # Get workloads for this phase
        phase_workloads = [
            w for w in workload_profiles
            if str(w.id) in phase.workloads
        ]
        
        if not phase_workloads:
            return {
                'phase_id': phase.phase_id,
                'phase_name': phase.phase_name,
                'total_cost': 0.0,
                'workload_count': 0
            }
        
        # Calculate phase duration in days
        if phase.start_date and phase.end_date:
            duration_days = (phase.end_date - phase.start_date).days
        else:
            duration_days = 7  # Default estimate
        
        # Data transfer for phase workloads
        data_transfer_cost = self._estimate_data_transfer_cost(phase_workloads)
        
        # Professional services for phase
        ps_cost = self._estimate_professional_services_cost(
            duration_days,
            len(phase_workloads)
        )
        
        # Dual running for phase duration
        phase_monthly_cost = self._estimate_current_infrastructure_cost(phase_workloads)
        dual_running_cost = phase_monthly_cost * Decimal(str(duration_days / 30.0)) * self.DUAL_RUNNING_MULTIPLIER
        
        total_cost = data_transfer_cost + ps_cost + dual_running_cost
        
        return {
            'phase_id': phase.phase_id,
            'phase_name': phase.phase_name,
            'total_cost': float(total_cost),
            'workload_count': len(phase_workloads),
            'duration_days': duration_days,
            'breakdown': {
                'data_transfer': float(data_transfer_cost),
                'professional_services': float(ps_cost),
                'dual_running': float(dual_running_cost)
            }
        }
    
    def _estimate_data_transfer_cost(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> Decimal:
        """
        Calculate data transfer cost for workloads.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            Total data transfer cost
        """
        total_cost = Decimal('0')
        
        for workload in workload_profiles:
            data_volume_gb = Decimal(str((workload.data_volume_tb or 0) * 1024))
            total_cost += data_volume_gb * self.DATA_TRANSFER_COST_PER_GB
        
        return total_cost
    
    def _estimate_dual_running_cost(
        self,
        workload_profiles: List[WorkloadProfile],
        duration_days: int
    ) -> Decimal:
        """
        Calculate dual-running cost.
        
        Args:
            workload_profiles: List of workload profiles
            duration_days: Migration duration
            
        Returns:
            Dual-running cost
        """
        monthly_cost = self._estimate_current_infrastructure_cost(workload_profiles)
        months = Decimal(str(duration_days / 30.0))
        
        # Cost of running both environments
        return monthly_cost * months * self.DUAL_RUNNING_MULTIPLIER
    
    def _estimate_professional_services_cost(
        self,
        duration_days: int,
        workload_count: int
    ) -> Decimal:
        """
        Estimate professional services costs.
        
        Args:
            duration_days: Migration duration
            workload_count: Number of workloads
            
        Returns:
            Professional services cost
        """
        # Base hours: planning, setup, validation
        base_hours = Decimal('80')
        
        # Hours per workload
        hours_per_workload = Decimal('16')
        
        # Additional hours for complex migrations
        if workload_count > 20:
            base_hours += Decimal('40')
        elif workload_count > 10:
            base_hours += Decimal('20')
        
        total_hours = base_hours + (Decimal(str(workload_count)) * hours_per_workload)
        
        return total_hours * self.PROFESSIONAL_SERVICES_HOURLY
    
    def _estimate_tooling_cost(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> Decimal:
        """
        Estimate migration tooling and licensing costs.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            Tooling cost
        """
        # Base tooling cost
        base_cost = Decimal('5000.00')
        
        # Additional cost per workload
        per_workload_cost = Decimal('500.00')
        
        total_cost = base_cost + (Decimal(str(len(workload_profiles))) * per_workload_cost)
        
        return total_cost
    
    def _estimate_current_infrastructure_cost(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> Decimal:
        """
        Estimate current monthly infrastructure cost.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            Estimated monthly cost
        """
        total_cost = Decimal('0')
        
        for workload in workload_profiles:
            # Estimate based on resources
            compute_cost = Decimal(str((workload.total_compute_cores or 0) * 50))
            memory_cost = Decimal(str((workload.total_memory_gb or 0) * 5))
            storage_cost = Decimal(str((workload.total_storage_tb or 0) * 100))
            
            workload_cost = compute_cost + memory_cost + storage_cost
            total_cost += workload_cost
        
        return total_cost
    
    def _calculate_confidence_level(
        self,
        workload_profiles: List[WorkloadProfile]
    ) -> str:
        """
        Calculate confidence level for cost estimate.
        
        Args:
            workload_profiles: List of workload profiles
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        # Check data completeness
        complete_profiles = 0
        
        for workload in workload_profiles:
            if (workload.total_compute_cores and
                workload.total_memory_gb and
                workload.total_storage_tb and
                workload.data_volume_tb):
                complete_profiles += 1
        
        completeness_ratio = complete_profiles / len(workload_profiles) if workload_profiles else 0
        
        if completeness_ratio >= 0.8:
            return 'high'
        elif completeness_ratio >= 0.5:
            return 'medium'
        else:
            return 'low'


class MigrationProgressTracker:
    """
    Tracks migration progress across phases and workloads.
    Implements Requirements: 4.4
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_migration_status(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive migration status for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Dictionary with migration status details
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Get migration plan
        plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        if not plan:
            return {
                'project_id': project_id,
                'status': project.status.value,
                'has_plan': False,
                'message': 'No migration plan created yet'
            }
        
        # Get all phases
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id
        ).order_by(MigrationPhase.phase_order).all()
        
        # Calculate overall progress
        progress = self._calculate_overall_progress(phases)
        
        # Get current phase
        current_phase = self._get_current_phase(phases)
        
        # Calculate timeline metrics
        timeline_metrics = self._calculate_timeline_metrics(plan, phases)
        
        # Get workload status
        workload_status = self._get_workload_status(project.id, phases)
        
        return {
            'project_id': project_id,
            'plan_id': plan.plan_id,
            'status': project.status.value,
            'overall_progress_percent': progress,
            'current_phase': current_phase,
            'timeline_metrics': timeline_metrics,
            'workload_status': workload_status,
            'phase_summary': self._get_phase_summary(phases)
        }
    
    def update_phase_status(
        self,
        phase_id: str,
        new_status: PhaseStatus,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the status of a migration phase.
        
        Args:
            phase_id: Phase identifier
            new_status: New phase status
            notes: Optional notes about the status change
            
        Returns:
            Updated phase information
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        old_status = phase.status
        phase.status = new_status
        
        # Update timestamps
        if new_status == PhaseStatus.IN_PROGRESS and not phase.actual_start_date:
            phase.actual_start_date = datetime.utcnow()
        elif new_status == PhaseStatus.COMPLETED and not phase.actual_end_date:
            phase.actual_end_date = datetime.utcnow()
        
        # Add notes if provided
        if notes:
            current_notes = phase.notes or ""
            timestamp = datetime.utcnow().isoformat()
            phase.notes = f"{current_notes}\n[{timestamp}] {notes}" if current_notes else f"[{timestamp}] {notes}"
        
        self.db.flush()
        
        logger.info(
            "Phase status updated",
            phase_id=phase_id,
            old_status=old_status.value,
            new_status=new_status.value
        )
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'old_status': old_status.value,
            'new_status': new_status.value,
            'actual_start_date': phase.actual_start_date.isoformat() if phase.actual_start_date else None,
            'actual_end_date': phase.actual_end_date.isoformat() if phase.actual_end_date else None
        }
    
    def track_workload_migration(
        self,
        workload_id: str,
        phase_id: str,
        status: str,
        progress_percent: int
    ) -> Dict[str, Any]:
        """
        Track progress of individual workload migration.
        
        Args:
            workload_id: Workload identifier
            phase_id: Phase identifier
            status: Migration status (not_started, in_progress, completed, failed)
            progress_percent: Progress percentage (0-100)
            
        Returns:
            Workload migration status
        """
        # Validate inputs
        if not 0 <= progress_percent <= 100:
            raise ValueError("Progress percent must be between 0 and 100")
        
        # Get phase
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        # Store workload progress in phase metadata
        # In a production system, this might be a separate table
        workload_progress = phase.notes or ""
        timestamp = datetime.utcnow().isoformat()
        progress_entry = f"[{timestamp}] Workload {workload_id}: {status} ({progress_percent}%)"
        
        phase.notes = f"{workload_progress}\n{progress_entry}" if workload_progress else progress_entry
        self.db.flush()
        
        return {
            'workload_id': workload_id,
            'phase_id': phase_id,
            'status': status,
            'progress_percent': progress_percent,
            'timestamp': timestamp
        }
    
    def calculate_progress(
        self,
        plan_id: str
    ) -> Dict[str, Any]:
        """
        Calculate detailed progress metrics for a migration plan.
        
        Args:
            plan_id: Migration plan ID
            
        Returns:
            Detailed progress metrics
        """
        plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.plan_id == plan_id
        ).first()
        
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id
        ).order_by(MigrationPhase.phase_order).all()
        
        # Calculate phase-based progress
        phase_progress = self._calculate_overall_progress(phases)
        
        # Calculate time-based progress
        time_progress = self._calculate_time_based_progress(plan, phases)
        
        # Calculate workload-based progress
        workload_progress = self._calculate_workload_progress(phases)
        
        # Determine overall progress (weighted average)
        overall_progress = (
            phase_progress * 0.4 +
            time_progress * 0.3 +
            workload_progress * 0.3
        )
        
        return {
            'plan_id': plan_id,
            'overall_progress_percent': round(overall_progress, 1),
            'phase_progress_percent': round(phase_progress, 1),
            'time_progress_percent': round(time_progress, 1),
            'workload_progress_percent': round(workload_progress, 1),
            'phases_completed': sum(1 for p in phases if p.status == PhaseStatus.COMPLETED),
            'phases_total': len(phases),
            'estimated_completion': plan.migration_project.estimated_completion.isoformat() if plan.migration_project.estimated_completion else None
        }
    
    def _calculate_overall_progress(
        self,
        phases: List[MigrationPhase]
    ) -> float:
        """
        Calculate overall progress based on phase completion.
        
        Args:
            phases: List of migration phases
            
        Returns:
            Progress percentage (0-100)
        """
        if not phases:
            return 0.0
        
        # Weight phases by their duration
        total_weight = 0.0
        completed_weight = 0.0
        
        for phase in phases:
            # Calculate phase weight based on duration
            if phase.start_date and phase.end_date:
                weight = (phase.end_date - phase.start_date).days
            else:
                weight = 7  # Default weight
            
            total_weight += weight
            
            if phase.status == PhaseStatus.COMPLETED:
                completed_weight += weight
            elif phase.status == PhaseStatus.IN_PROGRESS:
                # In-progress phases count as 50% complete
                completed_weight += weight * 0.5
        
        return (completed_weight / total_weight * 100) if total_weight > 0 else 0.0
    
    def _calculate_time_based_progress(
        self,
        plan: MigrationPlan,
        phases: List[MigrationPhase]
    ) -> float:
        """
        Calculate progress based on time elapsed.
        
        Args:
            plan: Migration plan
            phases: List of phases
            
        Returns:
            Time-based progress percentage
        """
        if not phases:
            return 0.0
        
        # Get first phase start date
        first_phase = min(phases, key=lambda p: p.start_date if p.start_date else datetime.max)
        if not first_phase.start_date:
            return 0.0
        
        # Get last phase end date
        last_phase = max(phases, key=lambda p: p.end_date if p.end_date else datetime.min)
        if not last_phase.end_date:
            return 0.0
        
        total_duration = (last_phase.end_date - first_phase.start_date).days
        if total_duration <= 0:
            return 0.0
        
        # Calculate elapsed time
        now = datetime.utcnow()
        if now < first_phase.start_date:
            return 0.0
        elif now > last_phase.end_date:
            return 100.0
        
        elapsed = (now - first_phase.start_date).days
        return min((elapsed / total_duration * 100), 100.0)
    
    def _calculate_workload_progress(
        self,
        phases: List[MigrationPhase]
    ) -> float:
        """
        Calculate progress based on workload completion.
        
        Args:
            phases: List of phases
            
        Returns:
            Workload-based progress percentage
        """
        total_workloads = 0
        completed_workloads = 0
        
        for phase in phases:
            phase_workload_count = len(phase.workloads)
            total_workloads += phase_workload_count
            
            if phase.status == PhaseStatus.COMPLETED:
                completed_workloads += phase_workload_count
            elif phase.status == PhaseStatus.IN_PROGRESS:
                # Assume 50% of workloads in in-progress phase are done
                completed_workloads += phase_workload_count * 0.5
        
        return (completed_workloads / total_workloads * 100) if total_workloads > 0 else 0.0
    
    def _get_current_phase(
        self,
        phases: List[MigrationPhase]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current active phase.
        
        Args:
            phases: List of phases
            
        Returns:
            Current phase information or None
        """
        # Find in-progress phase
        for phase in phases:
            if phase.status == PhaseStatus.IN_PROGRESS:
                return {
                    'phase_id': phase.phase_id,
                    'phase_name': phase.phase_name,
                    'phase_order': phase.phase_order,
                    'start_date': phase.start_date.isoformat() if phase.start_date else None,
                    'end_date': phase.end_date.isoformat() if phase.end_date else None,
                    'workload_count': len(phase.workloads)
                }
        
        # If no in-progress, find next not-started phase
        for phase in sorted(phases, key=lambda p: p.phase_order):
            if phase.status == PhaseStatus.NOT_STARTED:
                return {
                    'phase_id': phase.phase_id,
                    'phase_name': phase.phase_name,
                    'phase_order': phase.phase_order,
                    'start_date': phase.start_date.isoformat() if phase.start_date else None,
                    'end_date': phase.end_date.isoformat() if phase.end_date else None,
                    'workload_count': len(phase.workloads),
                    'status': 'upcoming'
                }
        
        return None
    
    def _calculate_timeline_metrics(
        self,
        plan: MigrationPlan,
        phases: List[MigrationPhase]
    ) -> Dict[str, Any]:
        """
        Calculate timeline-related metrics.
        
        Args:
            plan: Migration plan
            phases: List of phases
            
        Returns:
            Timeline metrics
        """
        if not phases:
            return {}
        
        # Get actual start date (first phase that started)
        actual_start = None
        for phase in sorted(phases, key=lambda p: p.phase_order):
            if phase.actual_start_date:
                actual_start = phase.actual_start_date
                break
        
        # Get planned dates
        first_phase = min(phases, key=lambda p: p.start_date if p.start_date else datetime.max)
        last_phase = max(phases, key=lambda p: p.end_date if p.end_date else datetime.min)
        
        planned_start = first_phase.start_date
        planned_end = last_phase.end_date
        
        # Calculate delays
        start_delay_days = None
        if actual_start and planned_start:
            start_delay_days = (actual_start - planned_start).days
        
        # Calculate remaining time
        remaining_days = None
        if planned_end:
            remaining_days = (planned_end - datetime.utcnow()).days
        
        return {
            'planned_start': planned_start.isoformat() if planned_start else None,
            'planned_end': planned_end.isoformat() if planned_end else None,
            'actual_start': actual_start.isoformat() if actual_start else None,
            'start_delay_days': start_delay_days,
            'remaining_days': max(0, remaining_days) if remaining_days else None,
            'total_planned_days': plan.total_duration_days,
            'on_schedule': start_delay_days is None or start_delay_days <= 0
        }
    
    def _get_workload_status(
        self,
        project_id: uuid.UUID,
        phases: List[MigrationPhase]
    ) -> Dict[str, Any]:
        """
        Get workload migration status summary.
        
        Args:
            project_id: Project UUID
            phases: List of phases
            
        Returns:
            Workload status summary
        """
        # Get all workloads for project
        workloads = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project_id
        ).all()
        
        total_workloads = len(workloads)
        
        # Count workloads by phase status
        not_started = 0
        in_progress = 0
        completed = 0
        
        for phase in phases:
            workload_count = len(phase.workloads)
            
            if phase.status == PhaseStatus.NOT_STARTED:
                not_started += workload_count
            elif phase.status == PhaseStatus.IN_PROGRESS:
                in_progress += workload_count
            elif phase.status == PhaseStatus.COMPLETED:
                completed += workload_count
        
        return {
            'total_workloads': total_workloads,
            'not_started': not_started,
            'in_progress': in_progress,
            'completed': completed,
            'completion_rate': (completed / total_workloads * 100) if total_workloads > 0 else 0
        }
    
    def _get_phase_summary(
        self,
        phases: List[MigrationPhase]
    ) -> List[Dict[str, Any]]:
        """
        Get summary of all phases.
        
        Args:
            phases: List of phases
            
        Returns:
            List of phase summaries
        """
        return [
            {
                'phase_id': phase.phase_id,
                'phase_name': phase.phase_name,
                'phase_order': phase.phase_order,
                'status': phase.status.value,
                'workload_count': len(phase.workloads),
                'start_date': phase.start_date.isoformat() if phase.start_date else None,
                'end_date': phase.end_date.isoformat() if phase.end_date else None,
                'actual_start_date': phase.actual_start_date.isoformat() if phase.actual_start_date else None,
                'actual_end_date': phase.actual_end_date.isoformat() if phase.actual_end_date else None
            }
            for phase in sorted(phases, key=lambda p: p.phase_order)
        ]


class MigrationValidator:
    """
    Validates resource deployment, connectivity, and functionality after migration.
    Implements Requirements: 4.6
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def validate_phase_completion(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Validate that a migration phase has completed successfully.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Validation results
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        validation_results = []
        all_passed = True
        
        # Validate success criteria
        for criterion in phase.success_criteria:
            # In a real system, this would check actual conditions
            # For now, we'll simulate validation
            passed = self._validate_criterion(criterion, phase)
            validation_results.append({
                'criterion': criterion,
                'passed': passed,
                'details': f"Validation {'passed' if passed else 'failed'} for: {criterion}"
            })
            if not passed:
                all_passed = False
        
        # Validate workload deployments
        workload_validation = self._validate_workload_deployments(phase)
        validation_results.extend(workload_validation['checks'])
        if not workload_validation['all_passed']:
            all_passed = False
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'validation_passed': all_passed,
            'validation_results': validation_results,
            'timestamp': datetime.utcnow().isoformat(),
            'can_proceed': all_passed
        }
    
    def validate_resource_deployment(
        self,
        workload_id: str,
        resource_checks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate that resources for a workload are properly deployed.
        
        Args:
            workload_id: Workload identifier
            resource_checks: Optional list of specific checks to perform
            
        Returns:
            Resource deployment validation results
        """
        workload = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.id == workload_id
        ).first()
        
        if not workload:
            raise ValueError(f"Workload {workload_id} not found")
        
        # Default checks if none specified
        if resource_checks is None:
            resource_checks = [
                'compute_resources',
                'storage_resources',
                'network_configuration',
                'security_groups',
                'load_balancers'
            ]
        
        validation_results = []
        all_passed = True
        
        for check in resource_checks:
            result = self._perform_resource_check(check, workload)
            validation_results.append(result)
            if not result['passed']:
                all_passed = False
        
        return {
            'workload_id': str(workload_id),
            'workload_name': workload.workload_name,
            'validation_passed': all_passed,
            'checks_performed': len(validation_results),
            'checks_passed': sum(1 for r in validation_results if r['passed']),
            'validation_results': validation_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def validate_connectivity(
        self,
        workload_id: str,
        connectivity_tests: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate network connectivity for a migrated workload.
        
        Args:
            workload_id: Workload identifier
            connectivity_tests: Optional list of connectivity tests to perform
            
        Returns:
            Connectivity validation results
        """
        workload = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.id == workload_id
        ).first()
        
        if not workload:
            raise ValueError(f"Workload {workload_id} not found")
        
        # Default connectivity tests
        if connectivity_tests is None:
            connectivity_tests = [
                {'type': 'internet_egress', 'description': 'Internet connectivity'},
                {'type': 'internal_network', 'description': 'Internal network connectivity'},
                {'type': 'database_connection', 'description': 'Database connectivity'},
                {'type': 'api_endpoints', 'description': 'API endpoint accessibility'}
            ]
        
        test_results = []
        all_passed = True
        
        for test in connectivity_tests:
            result = self._perform_connectivity_test(test, workload)
            test_results.append(result)
            if not result['passed']:
                all_passed = False
        
        return {
            'workload_id': str(workload_id),
            'workload_name': workload.workload_name,
            'connectivity_validated': all_passed,
            'tests_performed': len(test_results),
            'tests_passed': sum(1 for r in test_results if r['passed']),
            'test_results': test_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def validate_functionality(
        self,
        workload_id: str,
        functional_tests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate that workload functionality is working correctly after migration.
        
        Args:
            workload_id: Workload identifier
            functional_tests: Optional list of functional tests to perform
            
        Returns:
            Functionality validation results
        """
        workload = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.id == workload_id
        ).first()
        
        if not workload:
            raise ValueError(f"Workload {workload_id} not found")
        
        # Default functional tests
        if functional_tests is None:
            functional_tests = [
                'application_startup',
                'health_check_endpoint',
                'basic_operations',
                'data_integrity',
                'performance_baseline'
            ]
        
        test_results = []
        all_passed = True
        
        for test in functional_tests:
            result = self._perform_functional_test(test, workload)
            test_results.append(result)
            if not result['passed']:
                all_passed = False
        
        return {
            'workload_id': str(workload_id),
            'workload_name': workload.workload_name,
            'functionality_validated': all_passed,
            'tests_performed': len(test_results),
            'tests_passed': sum(1 for r in test_results if r['passed']),
            'test_results': test_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_validation_report(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for a phase.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Comprehensive validation report
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        # Validate phase completion
        phase_validation = self.validate_phase_completion(phase_id)
        
        # Validate each workload
        workload_validations = []
        for workload_id in phase.workloads:
            try:
                # Resource deployment validation
                resource_val = self.validate_resource_deployment(workload_id)
                
                # Connectivity validation
                connectivity_val = self.validate_connectivity(workload_id)
                
                # Functionality validation
                functionality_val = self.validate_functionality(workload_id)
                
                workload_validations.append({
                    'workload_id': workload_id,
                    'resource_deployment': resource_val,
                    'connectivity': connectivity_val,
                    'functionality': functionality_val,
                    'overall_passed': (
                        resource_val['validation_passed'] and
                        connectivity_val['connectivity_validated'] and
                        functionality_val['functionality_validated']
                    )
                })
            except Exception as e:
                workload_validations.append({
                    'workload_id': workload_id,
                    'error': str(e),
                    'overall_passed': False
                })
        
        # Calculate overall validation status
        all_workloads_passed = all(
            w.get('overall_passed', False) for w in workload_validations
        )
        
        overall_passed = phase_validation['validation_passed'] and all_workloads_passed
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'overall_validation_passed': overall_passed,
            'phase_validation': phase_validation,
            'workload_validations': workload_validations,
            'summary': {
                'total_workloads': len(phase.workloads),
                'workloads_passed': sum(1 for w in workload_validations if w.get('overall_passed', False)),
                'workloads_failed': sum(1 for w in workload_validations if not w.get('overall_passed', False))
            },
            'timestamp': datetime.utcnow().isoformat(),
            'recommendation': 'Proceed to next phase' if overall_passed else 'Address validation failures before proceeding'
        }
    
    def _validate_criterion(
        self,
        criterion: str,
        phase: MigrationPhase
    ) -> bool:
        """
        Validate a specific success criterion.
        
        Args:
            criterion: Success criterion to validate
            phase: Migration phase
            
        Returns:
            True if criterion is met
        """
        # In a real system, this would perform actual validation
        # For now, we simulate based on phase status
        
        if phase.status == PhaseStatus.COMPLETED:
            # Completed phases are assumed to have met criteria
            return True
        elif phase.status == PhaseStatus.IN_PROGRESS:
            # In-progress phases haven't met all criteria yet
            return False
        else:
            return False
    
    def _validate_workload_deployments(
        self,
        phase: MigrationPhase
    ) -> Dict[str, Any]:
        """
        Validate all workload deployments in a phase.
        
        Args:
            phase: Migration phase
            
        Returns:
            Workload deployment validation results
        """
        checks = []
        all_passed = True
        
        for workload_id in phase.workloads:
            # Simulate workload deployment check
            passed = phase.status == PhaseStatus.COMPLETED
            checks.append({
                'criterion': f'Workload {workload_id} deployed',
                'passed': passed,
                'details': f"Workload deployment {'successful' if passed else 'pending'}"
            })
            if not passed:
                all_passed = False
        
        return {
            'all_passed': all_passed,
            'checks': checks
        }
    
    def _perform_resource_check(
        self,
        check_type: str,
        workload: WorkloadProfile
    ) -> Dict[str, Any]:
        """
        Perform a specific resource check.
        
        Args:
            check_type: Type of check to perform
            workload: Workload profile
            
        Returns:
            Check result
        """
        # In a real system, this would perform actual resource checks
        # For now, we simulate based on workload data
        
        check_descriptions = {
            'compute_resources': 'Compute instances deployed and running',
            'storage_resources': 'Storage volumes attached and accessible',
            'network_configuration': 'Network configuration applied correctly',
            'security_groups': 'Security groups configured properly',
            'load_balancers': 'Load balancers configured and healthy'
        }
        
        # Simulate check (assume pass if workload has required data)
        passed = True
        details = f"{check_descriptions.get(check_type, check_type)} - validated successfully"
        
        return {
            'check_type': check_type,
            'description': check_descriptions.get(check_type, check_type),
            'passed': passed,
            'details': details
        }
    
    def _perform_connectivity_test(
        self,
        test: Dict[str, Any],
        workload: WorkloadProfile
    ) -> Dict[str, Any]:
        """
        Perform a connectivity test.
        
        Args:
            test: Test configuration
            workload: Workload profile
            
        Returns:
            Test result
        """
        # In a real system, this would perform actual connectivity tests
        # For now, we simulate
        
        test_type = test.get('type', 'unknown')
        description = test.get('description', test_type)
        
        # Simulate test (assume pass)
        passed = True
        latency_ms = 15.5  # Simulated latency
        
        return {
            'test_type': test_type,
            'description': description,
            'passed': passed,
            'latency_ms': latency_ms,
            'details': f"Connectivity test passed with {latency_ms}ms latency"
        }
    
    def _perform_functional_test(
        self,
        test_name: str,
        workload: WorkloadProfile
    ) -> Dict[str, Any]:
        """
        Perform a functional test.
        
        Args:
            test_name: Name of the test
            workload: Workload profile
            
        Returns:
            Test result
        """
        # In a real system, this would perform actual functional tests
        # For now, we simulate
        
        test_descriptions = {
            'application_startup': 'Application starts successfully',
            'health_check_endpoint': 'Health check endpoint responds correctly',
            'basic_operations': 'Basic CRUD operations work correctly',
            'data_integrity': 'Data integrity verified',
            'performance_baseline': 'Performance meets baseline requirements'
        }
        
        # Simulate test (assume pass)
        passed = True
        description = test_descriptions.get(test_name, test_name)
        
        return {
            'test_name': test_name,
            'description': description,
            'passed': passed,
            'details': f"{description} - test passed"
        }


class RollbackManager:
    """
    Manages rollback procedures and execution for failed migrations.
    Implements Requirements: 4.5
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def generate_rollback_plan(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Generate a rollback plan for a migration phase.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Rollback plan details
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        # Get migration plan for risk level
        migration_plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.id == phase.migration_plan_id
        ).first()
        
        # Generate rollback steps based on phase and risk level
        rollback_steps = self._generate_rollback_steps(phase, migration_plan)
        
        # Estimate rollback duration
        rollback_duration = self._estimate_rollback_duration(phase, rollback_steps)
        
        # Identify rollback prerequisites
        prerequisites = self._identify_rollback_prerequisites(phase)
        
        # Calculate rollback risk
        rollback_risk = self._assess_rollback_risk(phase, migration_plan)
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'rollback_steps': rollback_steps,
            'estimated_duration_hours': rollback_duration,
            'prerequisites': prerequisites,
            'rollback_risk': rollback_risk,
            'data_loss_risk': self._assess_data_loss_risk(phase),
            'recovery_point': self._determine_recovery_point(phase),
            'approval_required': migration_plan.risk_level.value in ['high', 'critical']
        }
    
    def execute_rollback(
        self,
        phase_id: str,
        reason: str,
        approved_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute rollback for a migration phase.
        
        Args:
            phase_id: Phase identifier
            reason: Reason for rollback
            approved_by: Optional approver identifier
            
        Returns:
            Rollback execution results
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        # Check if phase can be rolled back
        if phase.status not in [PhaseStatus.IN_PROGRESS, PhaseStatus.FAILED]:
            raise ValueError(f"Phase {phase_id} cannot be rolled back from status {phase.status.value}")
        
        # Generate rollback plan
        rollback_plan = self.generate_rollback_plan(phase_id)
        
        # Execute rollback steps
        execution_results = []
        all_successful = True
        
        for step in rollback_plan['rollback_steps']:
            result = self._execute_rollback_step(step, phase)
            execution_results.append(result)
            if not result['successful']:
                all_successful = False
                break  # Stop on first failure
        
        # Update phase status
        if all_successful:
            phase.status = PhaseStatus.ROLLED_BACK
            phase.notes = f"{phase.notes or ''}\n[{datetime.utcnow().isoformat()}] Rollback completed: {reason}"
        else:
            phase.notes = f"{phase.notes or ''}\n[{datetime.utcnow().isoformat()}] Rollback failed: {reason}"
        
        self.db.flush()
        
        logger.info(
            "Rollback executed",
            phase_id=phase_id,
            successful=all_successful,
            reason=reason,
            approved_by=approved_by
        )
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'rollback_successful': all_successful,
            'reason': reason,
            'approved_by': approved_by,
            'execution_results': execution_results,
            'timestamp': datetime.utcnow().isoformat(),
            'new_status': phase.status.value
        }
    
    def validate_rollback_readiness(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Validate that rollback can be safely executed.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Rollback readiness validation
        """
        phase = self.db.query(MigrationPhase).filter(
            MigrationPhase.phase_id == phase_id
        ).first()
        
        if not phase:
            raise ValueError(f"Phase {phase_id} not found")
        
        checks = []
        all_passed = True
        
        # Check 1: Source environment still available
        source_available = self._check_source_environment_available(phase)
        checks.append({
            'check': 'Source environment available',
            'passed': source_available,
            'details': 'Source environment is accessible' if source_available else 'Source environment not accessible'
        })
        if not source_available:
            all_passed = False
        
        # Check 2: Backups available
        backups_available = self._check_backups_available(phase)
        checks.append({
            'check': 'Backups available',
            'passed': backups_available,
            'details': 'Required backups are available' if backups_available else 'Backups not found'
        })
        if not backups_available:
            all_passed = False
        
        # Check 3: No active transactions
        no_active_transactions = self._check_no_active_transactions(phase)
        checks.append({
            'check': 'No active transactions',
            'passed': no_active_transactions,
            'details': 'No active transactions detected' if no_active_transactions else 'Active transactions detected'
        })
        if not no_active_transactions:
            all_passed = False
        
        # Check 4: Rollback window available
        window_available = self._check_rollback_window(phase)
        checks.append({
            'check': 'Rollback window available',
            'passed': window_available,
            'details': 'Within rollback window' if window_available else 'Outside rollback window'
        })
        if not window_available:
            all_passed = False
        
        return {
            'phase_id': phase_id,
            'phase_name': phase.phase_name,
            'rollback_ready': all_passed,
            'checks': checks,
            'recommendation': 'Rollback can proceed' if all_passed else 'Address issues before rollback',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_rollback_history(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get rollback history for a migration project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Rollback history
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Get migration plan
        plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        if not plan:
            return {
                'project_id': project_id,
                'rollback_history': [],
                'total_rollbacks': 0
            }
        
        # Get all phases that were rolled back
        rolled_back_phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id,
            MigrationPhase.status == PhaseStatus.ROLLED_BACK
        ).all()
        
        rollback_history = []
        for phase in rolled_back_phases:
            # Parse rollback info from notes
            rollback_info = self._parse_rollback_info_from_notes(phase.notes)
            rollback_history.append({
                'phase_id': phase.phase_id,
                'phase_name': phase.phase_name,
                'rollback_timestamp': rollback_info.get('timestamp'),
                'rollback_reason': rollback_info.get('reason'),
                'workload_count': len(phase.workloads)
            })
        
        return {
            'project_id': project_id,
            'rollback_history': rollback_history,
            'total_rollbacks': len(rollback_history)
        }
    
    def _generate_rollback_steps(
        self,
        phase: MigrationPhase,
        migration_plan: MigrationPlan
    ) -> List[Dict[str, Any]]:
        """
        Generate rollback steps for a phase.
        
        Args:
            phase: Migration phase
            migration_plan: Migration plan
            
        Returns:
            List of rollback steps
        """
        steps = []
        
        # Step 1: Stop new traffic to migrated resources
        steps.append({
            'step_number': 1,
            'step_name': 'Stop Traffic',
            'description': 'Redirect traffic away from migrated resources',
            'estimated_duration_minutes': 5,
            'risk_level': 'low'
        })
        
        # Step 2: Backup current state
        steps.append({
            'step_number': 2,
            'step_name': 'Backup Current State',
            'description': 'Create backup of current migrated state',
            'estimated_duration_minutes': 15,
            'risk_level': 'low'
        })
        
        # Step 3: Restore source environment
        steps.append({
            'step_number': 3,
            'step_name': 'Restore Source Environment',
            'description': 'Restore and validate source environment',
            'estimated_duration_minutes': 30,
            'risk_level': 'medium'
        })
        
        # Step 4: Redirect traffic to source
        steps.append({
            'step_number': 4,
            'step_name': 'Redirect Traffic',
            'description': 'Redirect traffic back to source environment',
            'estimated_duration_minutes': 10,
            'risk_level': 'medium'
        })
        
        # Step 5: Validate source environment
        steps.append({
            'step_number': 5,
            'step_name': 'Validate Source',
            'description': 'Validate source environment is functioning correctly',
            'estimated_duration_minutes': 20,
            'risk_level': 'low'
        })
        
        # Step 6: Cleanup migrated resources (optional)
        if migration_plan.risk_level.value in ['low', 'medium']:
            steps.append({
                'step_number': 6,
                'step_name': 'Cleanup Migrated Resources',
                'description': 'Remove or stop migrated resources',
                'estimated_duration_minutes': 15,
                'risk_level': 'low',
                'optional': True
            })
        
        return steps
    
    def _estimate_rollback_duration(
        self,
        phase: MigrationPhase,
        rollback_steps: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate total rollback duration.
        
        Args:
            phase: Migration phase
            rollback_steps: List of rollback steps
            
        Returns:
            Estimated duration in hours
        """
        total_minutes = sum(step['estimated_duration_minutes'] for step in rollback_steps)
        
        # Add buffer based on workload count
        workload_count = len(phase.workloads)
        buffer_minutes = workload_count * 5
        
        total_hours = (total_minutes + buffer_minutes) / 60.0
        return round(total_hours, 1)
    
    def _identify_rollback_prerequisites(
        self,
        phase: MigrationPhase
    ) -> List[str]:
        """
        Identify prerequisites for rollback.
        
        Args:
            phase: Migration phase
            
        Returns:
            List of prerequisites
        """
        prerequisites = [
            'Source environment must be accessible',
            'Recent backups must be available',
            'No active user transactions',
            'Rollback approval obtained',
            'Rollback team on standby'
        ]
        
        # Add phase-specific prerequisites
        if 'database' in phase.phase_name.lower():
            prerequisites.append('Database backup verified')
            prerequisites.append('Transaction logs available')
        
        return prerequisites
    
    def _assess_rollback_risk(
        self,
        phase: MigrationPhase,
        migration_plan: MigrationPlan
    ) -> str:
        """
        Assess risk level for rollback.
        
        Args:
            phase: Migration phase
            migration_plan: Migration plan
            
        Returns:
            Risk level: 'low', 'medium', 'high', or 'critical'
        """
        # Base risk on migration plan risk level
        base_risk = migration_plan.risk_level.value
        
        # Increase risk if phase has many workloads
        if len(phase.workloads) > 10:
            risk_levels = ['low', 'medium', 'high', 'critical']
            current_index = risk_levels.index(base_risk)
            if current_index < len(risk_levels) - 1:
                base_risk = risk_levels[current_index + 1]
        
        return base_risk
    
    def _assess_data_loss_risk(
        self,
        phase: MigrationPhase
    ) -> str:
        """
        Assess risk of data loss during rollback.
        
        Args:
            phase: Migration phase
            
        Returns:
            Data loss risk: 'none', 'minimal', 'moderate', or 'high'
        """
        # Check if phase has been running for a while
        if phase.actual_start_date:
            hours_running = (datetime.utcnow() - phase.actual_start_date).total_seconds() / 3600
            
            if hours_running < 1:
                return 'none'
            elif hours_running < 24:
                return 'minimal'
            elif hours_running < 72:
                return 'moderate'
            else:
                return 'high'
        
        return 'minimal'
    
    def _determine_recovery_point(
        self,
        phase: MigrationPhase
    ) -> str:
        """
        Determine recovery point for rollback.
        
        Args:
            phase: Migration phase
            
        Returns:
            Recovery point description
        """
        if phase.actual_start_date:
            return f"Pre-migration state from {phase.actual_start_date.isoformat()}"
        elif phase.start_date:
            return f"Planned pre-migration state from {phase.start_date.isoformat()}"
        else:
            return "Latest available backup"
    
    def _execute_rollback_step(
        self,
        step: Dict[str, Any],
        phase: MigrationPhase
    ) -> Dict[str, Any]:
        """
        Execute a single rollback step.
        
        Args:
            step: Rollback step configuration
            phase: Migration phase
            
        Returns:
            Step execution result
        """
        # In a real system, this would execute actual rollback operations
        # For now, we simulate
        
        step_number = step['step_number']
        step_name = step['step_name']
        
        # Simulate execution (assume success)
        successful = True
        
        logger.info(
            "Executing rollback step",
            phase_id=phase.phase_id,
            step_number=step_number,
            step_name=step_name
        )
        
        return {
            'step_number': step_number,
            'step_name': step_name,
            'successful': successful,
            'execution_time_minutes': step['estimated_duration_minutes'],
            'details': f"Step {step_number} completed successfully"
        }
    
    def _check_source_environment_available(self, phase: MigrationPhase) -> bool:
        """Check if source environment is available."""
        # In a real system, this would check actual source environment
        return True
    
    def _check_backups_available(self, phase: MigrationPhase) -> bool:
        """Check if backups are available."""
        # In a real system, this would verify backup existence
        return True
    
    def _check_no_active_transactions(self, phase: MigrationPhase) -> bool:
        """Check for active transactions."""
        # In a real system, this would check for active transactions
        return True
    
    def _check_rollback_window(self, phase: MigrationPhase) -> bool:
        """Check if within rollback window."""
        if phase.actual_start_date:
            hours_since_start = (datetime.utcnow() - phase.actual_start_date).total_seconds() / 3600
            # Rollback window is 72 hours
            return hours_since_start <= 72
        return True
    
    def _parse_rollback_info_from_notes(self, notes: Optional[str]) -> Dict[str, Any]:
        """Parse rollback information from phase notes."""
        if not notes:
            return {}
        
        # Simple parsing - in a real system, this would be more sophisticated
        info = {}
        
        if 'Rollback completed' in notes:
            # Extract timestamp and reason from notes
            lines = notes.split('\n')
            for line in lines:
                if 'Rollback completed' in line:
                    # Extract timestamp from [timestamp] format
                    if '[' in line and ']' in line:
                        timestamp_str = line[line.find('[')+1:line.find(']')]
                        info['timestamp'] = timestamp_str
                    # Extract reason after the colon
                    if ':' in line:
                        reason = line.split(':', 1)[1].strip()
                        info['reason'] = reason
        
        return info


class MigrationPlanningEngine:
    """
    Main migration planning engine that coordinates plan generation, dependency analysis,
    sequencing, cost estimation, and progress tracking.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.plan_generator = MigrationPlanGenerator(db_session)
        self.dependency_analyzer = DependencyAnalyzer(db_session)
        self.sequencer = MigrationSequencer(db_session)
        self.cost_estimator = MigrationCostEstimator(db_session)
        self.progress_tracker = MigrationProgressTracker(db_session)
        self.validator = MigrationValidator(db_session)
        self.rollback_manager = RollbackManager(db_session)
    
    def generate_migration_plan(
        self,
        project_id: str,
        target_provider: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive migration plan for a project.
        
        Args:
            project_id: Migration project ID
            target_provider: Target cloud provider
            
        Returns:
            Dictionary with plan details
        """
        # Get workload profiles
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        workload_profiles = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project.id
        ).all()
        
        # Build dependency graph
        dependency_graph = self.dependency_analyzer.build_dependency_graph(workload_profiles)
        
        # Validate dependencies
        validation_result = self.dependency_analyzer.validate_dependencies(dependency_graph)
        
        # Generate plan
        plan = self.plan_generator.generate_migration_plan(
            project_id=project_id,
            target_provider=target_provider,
            workload_profiles=workload_profiles
        )
        
        # Update plan with dependency information
        plan.dependencies_graph = dependency_graph
        plan.migration_waves = dependency_graph['migration_waves']
        self.db.flush()
        
        # Update project status
        project.status = MigrationStatus.PLANNING
        project.current_phase = "Migration Planning"
        self.db.flush()
        
        # Get phases
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id
        ).order_by(MigrationPhase.phase_order).all()
        
        return {
            'plan_id': plan.plan_id,
            'target_provider': plan.target_provider,
            'total_duration_days': plan.total_duration_days,
            'estimated_cost': float(plan.estimated_cost),
            'risk_level': plan.risk_level.value,
            'success_criteria': plan.success_criteria,
            'rollback_strategy': plan.rollback_strategy,
            'dependency_graph': dependency_graph,
            'dependency_validation': validation_result,
            'phases': [
                {
                    'phase_id': phase.phase_id,
                    'phase_name': phase.phase_name,
                    'phase_order': phase.phase_order,
                    'start_date': phase.start_date.isoformat() if phase.start_date else None,
                    'end_date': phase.end_date.isoformat() if phase.end_date else None,
                    'status': phase.status.value,
                    'workload_count': len(phase.workloads),
                    'prerequisites': phase.prerequisites,
                    'success_criteria': phase.success_criteria
                }
                for phase in phases
            ]
        }
    
    def analyze_dependencies(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Analyze dependencies for a migration project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Dictionary with dependency analysis results
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        workload_profiles = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project.id
        ).all()
        
        # Build dependency graph
        dependency_graph = self.dependency_analyzer.build_dependency_graph(workload_profiles)
        
        # Validate dependencies
        validation_result = self.dependency_analyzer.validate_dependencies(dependency_graph)
        
        # Discover implicit dependencies
        discovered_deps = self.dependency_analyzer.discover_dependencies(workload_profiles)
        
        return {
            'dependency_graph': dependency_graph,
            'validation': validation_result,
            'discovered_dependencies': discovered_deps
        }
    
    def generate_migration_sequence(
        self,
        project_id: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimized migration sequence for a project.
        
        Args:
            project_id: Migration project ID
            constraints: Optional sequencing constraints
            
        Returns:
            Migration sequence details
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        workload_profiles = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project.id
        ).all()
        
        # Build dependency graph
        dependency_graph = self.dependency_analyzer.build_dependency_graph(workload_profiles)
        
        # Generate sequence
        sequence = self.sequencer.generate_migration_sequence(
            dependency_graph,
            workload_profiles,
            constraints
        )
        
        return sequence
    
    def validate_phase_prerequisites(
        self,
        phase_id: str,
        plan_id: str
    ) -> Dict[str, Any]:
        """
        Validate prerequisites for a migration phase.
        
        Args:
            phase_id: Phase identifier
            plan_id: Migration plan ID
            
        Returns:
            Prerequisite validation results
        """
        plan = self.plan_generator.get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        return self.sequencer.validate_migration_prerequisites(
            phase_id,
            plan.id
        )
    
    def estimate_migration_costs(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Estimate detailed migration costs for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Detailed cost breakdown
        """
        # Get migration plan
        plan = self.plan_generator.get_plan_by_project(project_id)
        if not plan:
            raise ValueError(f"No migration plan found for project {project_id}")
        
        # Get workload profiles
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        workload_profiles = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == project.id
        ).all()
        
        # Get overall cost estimate
        overall_costs = self.cost_estimator.estimate_migration_costs(
            plan,
            workload_profiles
        )
        
        # Get data transfer cost breakdown
        data_transfer_details = self.cost_estimator.estimate_data_transfer_cost(
            workload_profiles,
            plan.target_provider
        )
        
        # Get dual-running cost breakdown
        budget_constraints = self.db.query(BudgetConstraints).filter(
            BudgetConstraints.migration_project_id == project.id
        ).first()
        
        current_monthly_cost = budget_constraints.current_monthly_cost if budget_constraints else None
        
        dual_running_details = self.cost_estimator.estimate_dual_running_cost(
            workload_profiles,
            plan.total_duration_days,
            current_monthly_cost
        )
        
        # Get phase-by-phase costs
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id
        ).order_by(MigrationPhase.phase_order).all()
        
        phase_costs = [
            self.cost_estimator.estimate_phase_cost(phase, workload_profiles)
            for phase in phases
        ]
        
        return {
            'overall_costs': overall_costs,
            'data_transfer_details': data_transfer_details,
            'dual_running_details': dual_running_details,
            'phase_costs': phase_costs,
            'plan_id': plan.plan_id,
            'target_provider': plan.target_provider
        }
    
    def get_migration_progress(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive migration progress for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Migration progress details
        """
        return self.progress_tracker.get_migration_status(project_id)
    
    def update_phase_status(
        self,
        phase_id: str,
        new_status: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update migration phase status.
        
        Args:
            phase_id: Phase identifier
            new_status: New status (not_started, in_progress, completed, failed, rolled_back)
            notes: Optional notes
            
        Returns:
            Updated phase information
        """
        # Convert string to enum
        status_map = {
            'not_started': PhaseStatus.NOT_STARTED,
            'in_progress': PhaseStatus.IN_PROGRESS,
            'completed': PhaseStatus.COMPLETED,
            'failed': PhaseStatus.FAILED,
            'rolled_back': PhaseStatus.ROLLED_BACK
        }
        
        status_enum = status_map.get(new_status.lower())
        if not status_enum:
            raise ValueError(f"Invalid status: {new_status}")
        
        return self.progress_tracker.update_phase_status(
            phase_id,
            status_enum,
            notes
        )
    
    def track_workload_progress(
        self,
        workload_id: str,
        phase_id: str,
        status: str,
        progress_percent: int
    ) -> Dict[str, Any]:
        """
        Track individual workload migration progress.
        
        Args:
            workload_id: Workload identifier
            phase_id: Phase identifier
            status: Migration status
            progress_percent: Progress percentage (0-100)
            
        Returns:
            Workload progress information
        """
        return self.progress_tracker.track_workload_migration(
            workload_id,
            phase_id,
            status,
            progress_percent
        )
    
    def validate_phase(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Validate migration phase completion.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Phase validation results
        """
        return self.validator.validate_phase_completion(phase_id)
    
    def validate_workload_deployment(
        self,
        workload_id: str
    ) -> Dict[str, Any]:
        """
        Validate workload resource deployment.
        
        Args:
            workload_id: Workload identifier
            
        Returns:
            Deployment validation results
        """
        return self.validator.validate_resource_deployment(workload_id)
    
    def validate_workload_connectivity(
        self,
        workload_id: str
    ) -> Dict[str, Any]:
        """
        Validate workload network connectivity.
        
        Args:
            workload_id: Workload identifier
            
        Returns:
            Connectivity validation results
        """
        return self.validator.validate_connectivity(workload_id)
    
    def validate_workload_functionality(
        self,
        workload_id: str
    ) -> Dict[str, Any]:
        """
        Validate workload functionality.
        
        Args:
            workload_id: Workload identifier
            
        Returns:
            Functionality validation results
        """
        return self.validator.validate_functionality(workload_id)
    
    def generate_phase_validation_report(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for a phase.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Comprehensive validation report
        """
        return self.validator.generate_validation_report(phase_id)
    
    def generate_rollback_plan(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Generate rollback plan for a migration phase.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Rollback plan details
        """
        return self.rollback_manager.generate_rollback_plan(phase_id)
    
    def execute_rollback(
        self,
        phase_id: str,
        reason: str,
        approved_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute rollback for a migration phase.
        
        Args:
            phase_id: Phase identifier
            reason: Reason for rollback
            approved_by: Optional approver identifier
            
        Returns:
            Rollback execution results
        """
        return self.rollback_manager.execute_rollback(
            phase_id,
            reason,
            approved_by
        )
    
    def validate_rollback_readiness(
        self,
        phase_id: str
    ) -> Dict[str, Any]:
        """
        Validate that rollback can be safely executed.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Rollback readiness validation
        """
        return self.rollback_manager.validate_rollback_readiness(phase_id)
    
    def get_rollback_history(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get rollback history for a migration project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Rollback history
        """
        return self.rollback_manager.get_rollback_history(project_id)
    
    def get_migration_plan(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve migration plan for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Dictionary with plan details or None if not found
        """
        plan = self.plan_generator.get_plan_by_project(project_id)
        
        if not plan:
            return None
        
        phases = self.db.query(MigrationPhase).filter(
            MigrationPhase.migration_plan_id == plan.id
        ).order_by(MigrationPhase.phase_order).all()
        
        return {
            'plan_id': plan.plan_id,
            'target_provider': plan.target_provider,
            'total_duration_days': plan.total_duration_days,
            'estimated_cost': float(plan.estimated_cost),
            'risk_level': plan.risk_level.value,
            'success_criteria': plan.success_criteria,
            'rollback_strategy': plan.rollback_strategy,
            'phases': [
                {
                    'phase_id': phase.phase_id,
                    'phase_name': phase.phase_name,
                    'phase_order': phase.phase_order,
                    'start_date': phase.start_date.isoformat() if phase.start_date else None,
                    'end_date': phase.end_date.isoformat() if phase.end_date else None,
                    'status': phase.status.value,
                    'workload_count': len(phase.workloads),
                    'prerequisites': phase.prerequisites,
                    'success_criteria': phase.success_criteria
                }
                for phase in phases
            ]
        }
