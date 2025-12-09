"""
Workload and Requirements Analysis Engine

This module implements the analysis engine for workload profiling, performance requirements,
compliance assessment, budget analysis, and technical requirements mapping.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import structlog

from .models import (
    MigrationProject, WorkloadProfile, PerformanceRequirements,
    ComplianceRequirements, BudgetConstraints, TechnicalRequirements
)

logger = structlog.get_logger(__name__)


class WorkloadProfiler:
    """
    Manages workload profiling including data collection and validation.
    Implements Requirement: 2.1
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_workload_profile(
        self,
        migration_project_id: uuid.UUID,
        workload_name: str,
        application_type: str,
        total_compute_cores: Optional[int] = None,
        total_memory_gb: Optional[int] = None,
        total_storage_tb: Optional[float] = None,
        database_types: Optional[List[str]] = None,
        data_volume_tb: Optional[float] = None,
        peak_transaction_rate: Optional[int] = None,
        workload_patterns: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> WorkloadProfile:
        """
        Create a workload profile for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            workload_name: Name of the workload
            application_type: Type of application (web, database, analytics, etc.)
            total_compute_cores: Total CPU cores required
            total_memory_gb: Total memory in GB
            total_storage_tb: Total storage in TB
            database_types: List of database types used
            data_volume_tb: Data volume in TB
            peak_transaction_rate: Peak transactions per second
            workload_patterns: Usage patterns and characteristics
            dependencies: List of dependent workload names
            
        Returns:
            WorkloadProfile: Created workload profile
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Validate workload data
        self._validate_workload_data(
            workload_name, application_type, total_compute_cores,
            total_memory_gb, total_storage_tb
        )
        
        # Create workload profile
        workload = WorkloadProfile(
            migration_project_id=migration_project_id,
            workload_name=workload_name.strip(),
            application_type=application_type.strip(),
            total_compute_cores=total_compute_cores,
            total_memory_gb=total_memory_gb,
            total_storage_tb=total_storage_tb,
            database_types=database_types or [],
            data_volume_tb=data_volume_tb,
            peak_transaction_rate=peak_transaction_rate,
            workload_patterns=workload_patterns or {},
            dependencies=dependencies or []
        )
        
        try:
            self.db.add(workload)
            self.db.flush()
            
            logger.info(
                "Workload profile created",
                project_id=str(migration_project_id),
                workload_name=workload_name,
                application_type=application_type
            )
            
            return workload
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create workload profile", error=str(e))
            raise
    
    def get_workload_profiles(
        self,
        migration_project_id: uuid.UUID
    ) -> List[WorkloadProfile]:
        """
        Retrieve all workload profiles for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            List of WorkloadProfile objects
        """
        return self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == migration_project_id
        ).all()
    
    def get_workload_by_id(self, workload_id: uuid.UUID) -> Optional[WorkloadProfile]:
        """
        Retrieve a specific workload profile by ID.
        
        Args:
            workload_id: UUID of the workload
            
        Returns:
            WorkloadProfile or None if not found
        """
        return self.db.query(WorkloadProfile).filter(
            WorkloadProfile.id == workload_id
        ).first()
    
    def update_workload_profile(
        self,
        workload_id: uuid.UUID,
        **updates
    ) -> WorkloadProfile:
        """
        Update workload profile fields.
        
        Args:
            workload_id: UUID of the workload
            **updates: Fields to update
            
        Returns:
            Updated WorkloadProfile
            
        Raises:
            ValueError: If workload not found
        """
        workload = self.get_workload_by_id(workload_id)
        if not workload:
            raise ValueError(f"Workload {workload_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'workload_name', 'application_type', 'total_compute_cores',
            'total_memory_gb', 'total_storage_tb', 'database_types',
            'data_volume_tb', 'peak_transaction_rate', 'workload_patterns',
            'dependencies'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(workload, field, value)
        
        self.db.flush()
        
        logger.info(
            "Workload profile updated",
            workload_id=str(workload_id),
            updated_fields=list(updates.keys())
        )
        
        return workload
    
    def analyze_workload_patterns(
        self,
        workload_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze workload patterns and provide insights.
        
        Args:
            workload_data: Dictionary with workload characteristics
            
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = {
            'workload_type': self._classify_workload_type(workload_data),
            'resource_intensity': self._assess_resource_intensity(workload_data),
            'scalability_requirements': self._assess_scalability_needs(workload_data),
            'storage_characteristics': self._analyze_storage_needs(workload_data)
        }
        
        return patterns
    
    def _validate_workload_data(
        self,
        workload_name: str,
        application_type: str,
        total_compute_cores: Optional[int],
        total_memory_gb: Optional[int],
        total_storage_tb: Optional[float]
    ) -> None:
        """Validate workload data."""
        if not workload_name or len(workload_name.strip()) == 0:
            raise ValueError("Workload name cannot be empty")
        
        if not application_type or len(application_type.strip()) == 0:
            raise ValueError("Application type cannot be empty")
        
        if total_compute_cores is not None and total_compute_cores < 0:
            raise ValueError("Total compute cores must be non-negative")
        
        if total_memory_gb is not None and total_memory_gb < 0:
            raise ValueError("Total memory must be non-negative")
        
        if total_storage_tb is not None and total_storage_tb < 0:
            raise ValueError("Total storage must be non-negative")
    
    def _classify_workload_type(self, workload_data: Dict[str, Any]) -> str:
        """Classify workload based on characteristics."""
        app_type = workload_data.get('application_type', '').lower()
        
        if 'database' in app_type or 'db' in app_type:
            return 'database-intensive'
        elif 'web' in app_type or 'api' in app_type:
            return 'web-application'
        elif 'analytics' in app_type or 'data' in app_type:
            return 'analytics-intensive'
        elif 'ml' in app_type or 'ai' in app_type:
            return 'compute-intensive'
        else:
            return 'general-purpose'
    
    def _assess_resource_intensity(self, workload_data: Dict[str, Any]) -> str:
        """Assess resource intensity level."""
        cores = workload_data.get('total_compute_cores', 0)
        memory = workload_data.get('total_memory_gb', 0)
        
        if cores > 64 or memory > 256:
            return 'high'
        elif cores > 16 or memory > 64:
            return 'medium'
        else:
            return 'low'
    
    def _assess_scalability_needs(self, workload_data: Dict[str, Any]) -> str:
        """Assess scalability requirements."""
        patterns = workload_data.get('workload_patterns', {})
        peak_rate = workload_data.get('peak_transaction_rate', 0)
        
        if peak_rate > 10000 or patterns.get('highly_variable', False):
            return 'high-elasticity'
        elif peak_rate > 1000:
            return 'moderate-elasticity'
        else:
            return 'stable'
    
    def _analyze_storage_needs(self, workload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze storage characteristics."""
        storage_tb = workload_data.get('total_storage_tb', 0)
        data_volume_tb = workload_data.get('data_volume_tb', 0)
        
        return {
            'total_storage_tb': storage_tb,
            'data_volume_tb': data_volume_tb,
            'storage_type': 'high-capacity' if storage_tb > 10 else 'standard'
        }


class PerformanceAnalyzer:
    """
    Manages performance requirements analysis and validation.
    Implements Requirement: 2.2
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_performance_requirements(
        self,
        migration_project_id: uuid.UUID,
        availability_target: float,
        latency_requirements: Optional[Dict[str, Any]] = None,
        disaster_recovery_rto: Optional[int] = None,
        disaster_recovery_rpo: Optional[int] = None,
        geographic_distribution: Optional[List[str]] = None,
        peak_load_multiplier: Optional[float] = None,
        additional_requirements: Optional[Dict[str, Any]] = None
    ) -> PerformanceRequirements:
        """
        Create performance requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            availability_target: Target availability percentage (e.g., 99.99)
            latency_requirements: Latency profiles by region/service
            disaster_recovery_rto: Recovery Time Objective in minutes
            disaster_recovery_rpo: Recovery Point Objective in minutes
            geographic_distribution: Required regions for deployment
            peak_load_multiplier: Peak load multiplier for capacity planning
            additional_requirements: Additional performance requirements
            
        Returns:
            PerformanceRequirements: Created performance requirements
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Validate performance data
        self._validate_performance_data(
            availability_target, disaster_recovery_rto, disaster_recovery_rpo
        )
        
        # Create performance requirements
        perf_req = PerformanceRequirements(
            migration_project_id=migration_project_id,
            availability_target=availability_target,
            latency_requirements=latency_requirements or {},
            disaster_recovery_rto=disaster_recovery_rto,
            disaster_recovery_rpo=disaster_recovery_rpo,
            geographic_distribution=geographic_distribution or [],
            peak_load_multiplier=peak_load_multiplier,
            additional_requirements=additional_requirements or {}
        )
        
        try:
            self.db.add(perf_req)
            self.db.flush()
            
            logger.info(
                "Performance requirements created",
                project_id=str(migration_project_id),
                availability_target=availability_target
            )
            
            return perf_req
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create performance requirements", error=str(e))
            raise ValueError("Performance requirements already exist for this project")
    
    def get_performance_requirements(
        self,
        migration_project_id: uuid.UUID
    ) -> Optional[PerformanceRequirements]:
        """
        Retrieve performance requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            PerformanceRequirements or None if not found
        """
        return self.db.query(PerformanceRequirements).filter(
            PerformanceRequirements.migration_project_id == migration_project_id
        ).first()
    
    def update_performance_requirements(
        self,
        migration_project_id: uuid.UUID,
        **updates
    ) -> PerformanceRequirements:
        """
        Update performance requirements fields.
        
        Args:
            migration_project_id: UUID of the migration project
            **updates: Fields to update
            
        Returns:
            Updated PerformanceRequirements
            
        Raises:
            ValueError: If requirements not found
        """
        perf_req = self.get_performance_requirements(migration_project_id)
        if not perf_req:
            raise ValueError(f"Performance requirements for project {migration_project_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'availability_target', 'latency_requirements', 'disaster_recovery_rto',
            'disaster_recovery_rpo', 'geographic_distribution', 'peak_load_multiplier',
            'additional_requirements'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(perf_req, field, value)
        
        self.db.flush()
        
        logger.info(
            "Performance requirements updated",
            project_id=str(migration_project_id),
            updated_fields=list(updates.keys())
        )
        
        return perf_req
    
    def validate_performance_profile(
        self,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate performance requirements and provide recommendations.
        
        Args:
            performance_data: Dictionary with performance requirements
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        availability = performance_data.get('availability_target', 0)
        rto = performance_data.get('disaster_recovery_rto')
        rpo = performance_data.get('disaster_recovery_rpo')
        
        # Check availability target
        if availability >= 99.99:
            validation['recommendations'].append(
                'High availability target requires multi-AZ deployment'
            )
        
        # Check DR requirements
        if rto and rto < 60:
            validation['recommendations'].append(
                'Low RTO requires active-active or hot standby configuration'
            )
        
        if rpo and rpo < 15:
            validation['recommendations'].append(
                'Low RPO requires synchronous replication'
            )
        
        # Check geographic distribution
        geo_dist = performance_data.get('geographic_distribution', [])
        if len(geo_dist) > 3:
            validation['recommendations'].append(
                'Multi-region deployment increases complexity and cost'
            )
        
        return validation
    
    def _validate_performance_data(
        self,
        availability_target: float,
        disaster_recovery_rto: Optional[int],
        disaster_recovery_rpo: Optional[int]
    ) -> None:
        """Validate performance requirements data."""
        if not 0 <= availability_target <= 100:
            raise ValueError("Availability target must be between 0 and 100")
        
        if disaster_recovery_rto is not None and disaster_recovery_rto < 0:
            raise ValueError("RTO must be non-negative")
        
        if disaster_recovery_rpo is not None and disaster_recovery_rpo < 0:
            raise ValueError("RPO must be non-negative")


class ComplianceAssessor:
    """
    Manages compliance requirements assessment and validation.
    Implements Requirement: 2.3
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_compliance_requirements(
        self,
        migration_project_id: uuid.UUID,
        regulatory_frameworks: Optional[List[str]] = None,
        data_residency_requirements: Optional[List[str]] = None,
        industry_certifications: Optional[List[str]] = None,
        security_standards: Optional[List[str]] = None,
        audit_requirements: Optional[Dict[str, Any]] = None,
        additional_compliance: Optional[Dict[str, Any]] = None
    ) -> ComplianceRequirements:
        """
        Create compliance requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            regulatory_frameworks: List of regulatory frameworks (GDPR, HIPAA, etc.)
            data_residency_requirements: Countries/regions for data residency
            industry_certifications: Required industry certifications
            security_standards: Required security standards
            audit_requirements: Audit and logging requirements
            additional_compliance: Additional compliance requirements
            
        Returns:
            ComplianceRequirements: Created compliance requirements
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Create compliance requirements
        compliance_req = ComplianceRequirements(
            migration_project_id=migration_project_id,
            regulatory_frameworks=regulatory_frameworks or [],
            data_residency_requirements=data_residency_requirements or [],
            industry_certifications=industry_certifications or [],
            security_standards=security_standards or [],
            audit_requirements=audit_requirements or {},
            additional_compliance=additional_compliance or {}
        )
        
        try:
            self.db.add(compliance_req)
            self.db.flush()
            
            logger.info(
                "Compliance requirements created",
                project_id=str(migration_project_id),
                frameworks=len(regulatory_frameworks or [])
            )
            
            return compliance_req
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create compliance requirements", error=str(e))
            raise ValueError("Compliance requirements already exist for this project")
    
    def get_compliance_requirements(
        self,
        migration_project_id: uuid.UUID
    ) -> Optional[ComplianceRequirements]:
        """
        Retrieve compliance requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            ComplianceRequirements or None if not found
        """
        return self.db.query(ComplianceRequirements).filter(
            ComplianceRequirements.migration_project_id == migration_project_id
        ).first()
    
    def update_compliance_requirements(
        self,
        migration_project_id: uuid.UUID,
        **updates
    ) -> ComplianceRequirements:
        """
        Update compliance requirements fields.
        
        Args:
            migration_project_id: UUID of the migration project
            **updates: Fields to update
            
        Returns:
            Updated ComplianceRequirements
            
        Raises:
            ValueError: If requirements not found
        """
        compliance_req = self.get_compliance_requirements(migration_project_id)
        if not compliance_req:
            raise ValueError(f"Compliance requirements for project {migration_project_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'regulatory_frameworks', 'data_residency_requirements',
            'industry_certifications', 'security_standards',
            'audit_requirements', 'additional_compliance'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(compliance_req, field, value)
        
        self.db.flush()
        
        logger.info(
            "Compliance requirements updated",
            project_id=str(migration_project_id),
            updated_fields=list(updates.keys())
        )
        
        return compliance_req
    
    def validate_compliance_profile(
        self,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate compliance requirements and identify potential issues.
        
        Args:
            compliance_data: Dictionary with compliance requirements
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        frameworks = compliance_data.get('regulatory_frameworks', [])
        data_residency = compliance_data.get('data_residency_requirements', [])
        
        # Check for conflicting requirements
        if 'GDPR' in frameworks and not data_residency:
            validation['warnings'].append(
                'GDPR requires data residency specification'
            )
        
        # Check for high-compliance frameworks
        high_compliance_frameworks = ['HIPAA', 'PCI-DSS', 'FedRAMP']
        if any(fw in frameworks for fw in high_compliance_frameworks):
            validation['recommendations'].append(
                'High-compliance frameworks require dedicated compliance features'
            )
        
        # Check data residency complexity
        if len(data_residency) > 5:
            validation['recommendations'].append(
                'Multiple data residency requirements increase deployment complexity'
            )
        
        return validation


class BudgetAnalyzer:
    """
    Manages budget constraints analysis and validation.
    Implements Requirement: 2.4
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_budget_constraints(
        self,
        migration_project_id: uuid.UUID,
        migration_budget: float,
        current_monthly_cost: Optional[float] = None,
        target_monthly_cost: Optional[float] = None,
        cost_optimization_priority: str = 'medium',
        acceptable_cost_variance: Optional[float] = None,
        currency: str = 'USD',
        additional_constraints: Optional[Dict[str, Any]] = None
    ) -> BudgetConstraints:
        """
        Create budget constraints for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            migration_budget: Total budget for migration
            current_monthly_cost: Current monthly infrastructure cost
            target_monthly_cost: Target monthly cost after migration
            cost_optimization_priority: Priority level (low, medium, high)
            acceptable_cost_variance: Acceptable cost variance percentage
            currency: Currency code (default: USD)
            additional_constraints: Additional budget constraints
            
        Returns:
            BudgetConstraints: Created budget constraints
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Validate budget data
        self._validate_budget_data(
            migration_budget, current_monthly_cost, target_monthly_cost,
            cost_optimization_priority
        )
        
        # Create budget constraints
        budget = BudgetConstraints(
            migration_project_id=migration_project_id,
            migration_budget=migration_budget,
            current_monthly_cost=current_monthly_cost,
            target_monthly_cost=target_monthly_cost,
            cost_optimization_priority=cost_optimization_priority,
            acceptable_cost_variance=acceptable_cost_variance,
            currency=currency,
            additional_constraints=additional_constraints or {}
        )
        
        try:
            self.db.add(budget)
            self.db.flush()
            
            logger.info(
                "Budget constraints created",
                project_id=str(migration_project_id),
                migration_budget=float(migration_budget)
            )
            
            return budget
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create budget constraints", error=str(e))
            raise ValueError("Budget constraints already exist for this project")
    
    def get_budget_constraints(
        self,
        migration_project_id: uuid.UUID
    ) -> Optional[BudgetConstraints]:
        """
        Retrieve budget constraints for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            BudgetConstraints or None if not found
        """
        return self.db.query(BudgetConstraints).filter(
            BudgetConstraints.migration_project_id == migration_project_id
        ).first()
    
    def update_budget_constraints(
        self,
        migration_project_id: uuid.UUID,
        **updates
    ) -> BudgetConstraints:
        """
        Update budget constraints fields.
        
        Args:
            migration_project_id: UUID of the migration project
            **updates: Fields to update
            
        Returns:
            Updated BudgetConstraints
            
        Raises:
            ValueError: If constraints not found
        """
        budget = self.get_budget_constraints(migration_project_id)
        if not budget:
            raise ValueError(f"Budget constraints for project {migration_project_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'migration_budget', 'current_monthly_cost', 'target_monthly_cost',
            'cost_optimization_priority', 'acceptable_cost_variance',
            'currency', 'additional_constraints'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(budget, field, value)
        
        self.db.flush()
        
        logger.info(
            "Budget constraints updated",
            project_id=str(migration_project_id),
            updated_fields=list(updates.keys())
        )
        
        return budget
    
    def analyze_cost_optimization_priority(
        self,
        budget_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze budget constraints and provide cost optimization recommendations.
        
        Args:
            budget_data: Dictionary with budget information
            
        Returns:
            Dictionary with cost analysis and recommendations
        """
        analysis = {
            'priority_level': budget_data.get('cost_optimization_priority', 'medium'),
            'recommendations': [],
            'cost_saving_opportunities': []
        }
        
        current_cost = budget_data.get('current_monthly_cost', 0)
        target_cost = budget_data.get('target_monthly_cost', 0)
        migration_budget = budget_data.get('migration_budget', 0)
        
        # Calculate cost reduction target
        if current_cost and target_cost:
            reduction_pct = ((current_cost - target_cost) / current_cost) * 100
            analysis['target_cost_reduction_pct'] = reduction_pct
            
            if reduction_pct > 30:
                analysis['recommendations'].append(
                    'Aggressive cost reduction target - consider reserved instances and spot instances'
                )
            elif reduction_pct > 15:
                analysis['recommendations'].append(
                    'Moderate cost reduction - focus on right-sizing and reserved instances'
                )
        
        # Analyze migration budget adequacy
        if current_cost and migration_budget:
            budget_ratio = migration_budget / (current_cost * 12)
            analysis['migration_budget_ratio'] = budget_ratio
            
            if budget_ratio < 0.1:
                analysis['recommendations'].append(
                    'Migration budget may be insufficient for comprehensive migration'
                )
        
        return analysis
    
    def _validate_budget_data(
        self,
        migration_budget: float,
        current_monthly_cost: Optional[float],
        target_monthly_cost: Optional[float],
        cost_optimization_priority: str
    ) -> None:
        """Validate budget constraints data."""
        if migration_budget <= 0:
            raise ValueError("Migration budget must be positive")
        
        if current_monthly_cost is not None and current_monthly_cost < 0:
            raise ValueError("Current monthly cost must be non-negative")
        
        if target_monthly_cost is not None and target_monthly_cost < 0:
            raise ValueError("Target monthly cost must be non-negative")
        
        valid_priorities = ['low', 'medium', 'high']
        if cost_optimization_priority not in valid_priorities:
            raise ValueError(f"Cost optimization priority must be one of {valid_priorities}")


class TechnicalRequirementsMapper:
    """
    Manages technical requirements mapping and validation.
    Implements Requirement: 2.5
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_technical_requirements(
        self,
        migration_project_id: uuid.UUID,
        required_services: Optional[List[str]] = None,
        ml_ai_requirements: Optional[Dict[str, Any]] = None,
        analytics_requirements: Optional[Dict[str, Any]] = None,
        container_orchestration: bool = False,
        serverless_requirements: bool = False,
        specialized_compute: Optional[List[str]] = None,
        integration_requirements: Optional[Dict[str, Any]] = None,
        additional_technical: Optional[Dict[str, Any]] = None
    ) -> TechnicalRequirements:
        """
        Create technical requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            required_services: List of required cloud services
            ml_ai_requirements: Machine learning and AI requirements
            analytics_requirements: Analytics and data processing requirements
            container_orchestration: Whether container orchestration is needed
            serverless_requirements: Whether serverless capabilities are needed
            specialized_compute: List of specialized compute types (GPU, HPC, etc.)
            integration_requirements: Integration requirements with other systems
            additional_technical: Additional technical requirements
            
        Returns:
            TechnicalRequirements: Created technical requirements
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Create technical requirements
        tech_req = TechnicalRequirements(
            migration_project_id=migration_project_id,
            required_services=required_services or [],
            ml_ai_requirements=ml_ai_requirements or {},
            analytics_requirements=analytics_requirements or {},
            container_orchestration=container_orchestration,
            serverless_requirements=serverless_requirements,
            specialized_compute=specialized_compute or [],
            integration_requirements=integration_requirements or {},
            additional_technical=additional_technical or {}
        )
        
        try:
            self.db.add(tech_req)
            self.db.flush()
            
            logger.info(
                "Technical requirements created",
                project_id=str(migration_project_id),
                services=len(required_services or [])
            )
            
            return tech_req
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create technical requirements", error=str(e))
            raise ValueError("Technical requirements already exist for this project")
    
    def get_technical_requirements(
        self,
        migration_project_id: uuid.UUID
    ) -> Optional[TechnicalRequirements]:
        """
        Retrieve technical requirements for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            TechnicalRequirements or None if not found
        """
        return self.db.query(TechnicalRequirements).filter(
            TechnicalRequirements.migration_project_id == migration_project_id
        ).first()
    
    def update_technical_requirements(
        self,
        migration_project_id: uuid.UUID,
        **updates
    ) -> TechnicalRequirements:
        """
        Update technical requirements fields.
        
        Args:
            migration_project_id: UUID of the migration project
            **updates: Fields to update
            
        Returns:
            Updated TechnicalRequirements
            
        Raises:
            ValueError: If requirements not found
        """
        tech_req = self.get_technical_requirements(migration_project_id)
        if not tech_req:
            raise ValueError(f"Technical requirements for project {migration_project_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'required_services', 'ml_ai_requirements', 'analytics_requirements',
            'container_orchestration', 'serverless_requirements', 'specialized_compute',
            'integration_requirements', 'additional_technical'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(tech_req, field, value)
        
        self.db.flush()
        
        logger.info(
            "Technical requirements updated",
            project_id=str(migration_project_id),
            updated_fields=list(updates.keys())
        )
        
        return tech_req
    
    def map_service_requirements(
        self,
        technical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map technical requirements to cloud service categories.
        
        Args:
            technical_data: Dictionary with technical requirements
            
        Returns:
            Dictionary with service mapping results
        """
        service_mapping = {
            'compute': [],
            'storage': [],
            'database': [],
            'networking': [],
            'ml_ai': [],
            'analytics': [],
            'containers': [],
            'serverless': []
        }
        
        # Map required services to categories
        required_services = technical_data.get('required_services', [])
        for service in required_services:
            service_lower = service.lower()
            if any(kw in service_lower for kw in ['compute', 'vm', 'ec2', 'instance']):
                service_mapping['compute'].append(service)
            elif any(kw in service_lower for kw in ['storage', 's3', 'blob', 'disk']):
                service_mapping['storage'].append(service)
            elif any(kw in service_lower for kw in ['database', 'db', 'sql', 'nosql']):
                service_mapping['database'].append(service)
            elif any(kw in service_lower for kw in ['network', 'vpc', 'cdn', 'dns']):
                service_mapping['networking'].append(service)
        
        # Add ML/AI services
        if technical_data.get('ml_ai_requirements'):
            service_mapping['ml_ai'].append('ML/AI Platform')
        
        # Add analytics services
        if technical_data.get('analytics_requirements'):
            service_mapping['analytics'].append('Analytics Platform')
        
        # Add container services
        if technical_data.get('container_orchestration'):
            service_mapping['containers'].append('Container Orchestration')
        
        # Add serverless services
        if technical_data.get('serverless_requirements'):
            service_mapping['serverless'].append('Serverless Platform')
        
        return service_mapping
    
    def validate_service_mapping(
        self,
        service_mapping: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Validate service mapping and identify potential issues.
        
        Args:
            service_mapping: Dictionary with mapped services
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check for missing essential services
        if not service_mapping.get('compute'):
            validation['warnings'].append('No compute services specified')
        
        if not service_mapping.get('storage'):
            validation['warnings'].append('No storage services specified')
        
        # Check for complex requirements
        total_services = sum(len(services) for services in service_mapping.values())
        if total_services > 15:
            validation['recommendations'].append(
                'Large number of services may increase migration complexity'
            )
        
        # Check for specialized requirements
        if service_mapping.get('ml_ai'):
            validation['recommendations'].append(
                'ML/AI requirements need specialized cloud services'
            )
        
        if service_mapping.get('containers'):
            validation['recommendations'].append(
                'Container orchestration requires Kubernetes or equivalent'
            )
        
        return validation


class RequirementsCompletenessValidator:
    """
    Validates completeness and consistency of all requirements.
    Implements Requirement: 2.6
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def validate_requirements_completeness(
        self,
        migration_project_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Validate that all required data has been collected for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            Dictionary with validation results
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        validation = {
            'is_complete': True,
            'missing_items': [],
            'warnings': [],
            'completeness_score': 0.0
        }
        
        # Check each requirement category
        categories_checked = 0
        categories_complete = 0
        
        # Check workload profiles
        workloads = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == migration_project_id
        ).all()
        categories_checked += 1
        if workloads:
            categories_complete += 1
        else:
            validation['missing_items'].append('workload_profiles')
            validation['is_complete'] = False
        
        # Check performance requirements
        perf_req = self.db.query(PerformanceRequirements).filter(
            PerformanceRequirements.migration_project_id == migration_project_id
        ).first()
        categories_checked += 1
        if perf_req:
            categories_complete += 1
        else:
            validation['missing_items'].append('performance_requirements')
            validation['is_complete'] = False
        
        # Check compliance requirements
        compliance_req = self.db.query(ComplianceRequirements).filter(
            ComplianceRequirements.migration_project_id == migration_project_id
        ).first()
        categories_checked += 1
        if compliance_req:
            categories_complete += 1
        else:
            validation['missing_items'].append('compliance_requirements')
            validation['is_complete'] = False
        
        # Check budget constraints
        budget = self.db.query(BudgetConstraints).filter(
            BudgetConstraints.migration_project_id == migration_project_id
        ).first()
        categories_checked += 1
        if budget:
            categories_complete += 1
        else:
            validation['missing_items'].append('budget_constraints')
            validation['is_complete'] = False
        
        # Check technical requirements
        tech_req = self.db.query(TechnicalRequirements).filter(
            TechnicalRequirements.migration_project_id == migration_project_id
        ).first()
        categories_checked += 1
        if tech_req:
            categories_complete += 1
        else:
            validation['missing_items'].append('technical_requirements')
            validation['is_complete'] = False
        
        # Calculate completeness score
        validation['completeness_score'] = (categories_complete / categories_checked) * 100
        
        logger.info(
            "Requirements completeness validated",
            project_id=str(migration_project_id),
            completeness_score=validation['completeness_score'],
            is_complete=validation['is_complete']
        )
        
        return validation
    
    def check_consistency(
        self,
        migration_project_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Check consistency across requirement categories.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            Dictionary with consistency check results
        """
        consistency = {
            'is_consistent': True,
            'issues': [],
            'warnings': []
        }
        
        # Get all requirements
        workloads = self.db.query(WorkloadProfile).filter(
            WorkloadProfile.migration_project_id == migration_project_id
        ).all()
        
        perf_req = self.db.query(PerformanceRequirements).filter(
            PerformanceRequirements.migration_project_id == migration_project_id
        ).first()
        
        budget = self.db.query(BudgetConstraints).filter(
            BudgetConstraints.migration_project_id == migration_project_id
        ).first()
        
        # Check workload vs performance consistency
        if workloads and perf_req:
            total_cores = sum(w.total_compute_cores or 0 for w in workloads)
            if total_cores > 100 and perf_req.availability_target < 99.9:
                consistency['warnings'].append(
                    'Large workload with low availability target may indicate inconsistency'
                )
        
        # Check budget vs workload consistency
        if workloads and budget:
            total_storage = sum(w.total_storage_tb or 0 for w in workloads)
            if total_storage > 100 and budget.cost_optimization_priority == 'high':
                consistency['warnings'].append(
                    'Large storage requirements with high cost optimization may be challenging'
                )
        
        return consistency


class WorkloadAnalysisEngine:
    """
    Main analysis engine that coordinates all requirement analysis components.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.workload_profiler = WorkloadProfiler(db_session)
        self.performance_analyzer = PerformanceAnalyzer(db_session)
        self.compliance_assessor = ComplianceAssessor(db_session)
        self.budget_analyzer = BudgetAnalyzer(db_session)
        self.technical_mapper = TechnicalRequirementsMapper(db_session)
        self.completeness_validator = RequirementsCompletenessValidator(db_session)
    
    def analyze_workloads(
        self,
        project_id: str,
        workload_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze workloads and create workload profile.
        
        Args:
            project_id: Migration project ID
            workload_data: Workload information
            
        Returns:
            Dictionary with workload analysis results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create workload profile
        workload = self.workload_profiler.create_workload_profile(
            migration_project_id=project.id,
            **workload_data
        )
        
        # Analyze patterns
        patterns = self.workload_profiler.analyze_workload_patterns(workload_data)
        
        return {
            'workload_id': str(workload.id),
            'workload_name': workload.workload_name,
            'application_type': workload.application_type,
            'patterns': patterns
        }
    
    def assess_performance_requirements(
        self,
        project_id: str,
        perf_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess performance requirements and create performance profile.
        
        Args:
            project_id: Migration project ID
            perf_requirements: Performance requirements data
            
        Returns:
            Dictionary with performance assessment results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create performance requirements
        perf_req = self.performance_analyzer.create_performance_requirements(
            migration_project_id=project.id,
            **perf_requirements
        )
        
        # Validate profile
        validation = self.performance_analyzer.validate_performance_profile(perf_requirements)
        
        return {
            'performance_requirements_id': str(perf_req.id),
            'availability_target': float(perf_req.availability_target),
            'validation': validation
        }
    
    def evaluate_compliance_needs(
        self,
        project_id: str,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate compliance needs and create compliance profile.
        
        Args:
            project_id: Migration project ID
            compliance_data: Compliance requirements data
            
        Returns:
            Dictionary with compliance evaluation results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create compliance requirements
        compliance_req = self.compliance_assessor.create_compliance_requirements(
            migration_project_id=project.id,
            **compliance_data
        )
        
        # Validate profile
        validation = self.compliance_assessor.validate_compliance_profile(compliance_data)
        
        return {
            'compliance_requirements_id': str(compliance_req.id),
            'regulatory_frameworks': compliance_req.regulatory_frameworks,
            'validation': validation
        }
    
    def analyze_budget_constraints(
        self,
        project_id: str,
        budget_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze budget constraints and create budget profile.
        
        Args:
            project_id: Migration project ID
            budget_data: Budget constraints data
            
        Returns:
            Dictionary with budget analysis results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create budget constraints
        budget = self.budget_analyzer.create_budget_constraints(
            migration_project_id=project.id,
            **budget_data
        )
        
        # Analyze cost optimization
        analysis = self.budget_analyzer.analyze_cost_optimization_priority(budget_data)
        
        return {
            'budget_constraints_id': str(budget.id),
            'migration_budget': float(budget.migration_budget),
            'analysis': analysis
        }
    
    def map_technical_requirements(
        self,
        project_id: str,
        tech_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map technical requirements to cloud services.
        
        Args:
            project_id: Migration project ID
            tech_requirements: Technical requirements data
            
        Returns:
            Dictionary with service mapping results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create technical requirements
        tech_req = self.technical_mapper.create_technical_requirements(
            migration_project_id=project.id,
            **tech_requirements
        )
        
        # Map services
        service_mapping = self.technical_mapper.map_service_requirements(tech_requirements)
        
        # Validate mapping
        validation = self.technical_mapper.validate_service_mapping(service_mapping)
        
        return {
            'technical_requirements_id': str(tech_req.id),
            'service_mapping': service_mapping,
            'validation': validation
        }
    
    def validate_requirements_completeness(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Validate completeness of all requirements for a project.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Dictionary with validation results
        """
        # Get project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Validate completeness
        completeness = self.completeness_validator.validate_requirements_completeness(project.id)
        
        # Check consistency
        consistency = self.completeness_validator.check_consistency(project.id)
        
        return {
            'completeness': completeness,
            'consistency': consistency
        }
