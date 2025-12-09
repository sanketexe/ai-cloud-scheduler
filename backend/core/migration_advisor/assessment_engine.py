"""
Migration Assessment Engine

This module implements the Migration Assessment Engine for capturing user intent,
conducting initial assessments, and managing migration project lifecycle.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import structlog

from .models import (
    MigrationProject, OrganizationProfile, MigrationStatus,
    CompanySize, InfrastructureType, ExperienceLevel
)
from ..models import User

logger = structlog.get_logger(__name__)


class MigrationProjectManager:
    """
    Manages migration project lifecycle including creation, updates, and status tracking.
    Implements Requirements: 1.4, 1.5
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_migration_project(
        self,
        organization_name: str,
        created_by_user_id: uuid.UUID,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> MigrationProject:
        """
        Create a new migration project with initial assessment status.
        
        Args:
            organization_name: Name of the organization migrating
            created_by_user_id: UUID of the user creating the project
            initial_context: Optional initial context data
            
        Returns:
            MigrationProject: Created migration project
            
        Raises:
            ValueError: If user doesn't exist or organization name is invalid
            IntegrityError: If project_id already exists
        """
        # Validate user exists
        user = self.db.query(User).filter(User.id == created_by_user_id).first()
        if not user:
            raise ValueError(f"User with id {created_by_user_id} not found")
        
        # Validate organization name
        if not organization_name or len(organization_name.strip()) == 0:
            raise ValueError("Organization name cannot be empty")
        
        # Generate unique project ID
        project_id = self._generate_project_id(organization_name)
        
        # Create migration project
        migration_project = MigrationProject(
            project_id=project_id,
            organization_name=organization_name.strip(),
            status=MigrationStatus.ASSESSMENT,
            current_phase="Initial Assessment",
            created_by=created_by_user_id
        )
        
        try:
            self.db.add(migration_project)
            self.db.flush()  # Get the ID without committing
            
            logger.info(
                "Migration project created",
                project_id=project_id,
                organization=organization_name,
                created_by=str(created_by_user_id)
            )
            
            return migration_project
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create migration project", error=str(e))
            raise
    
    def get_project(self, project_id: str) -> Optional[MigrationProject]:
        """
        Retrieve a migration project by project_id.
        
        Args:
            project_id: Unique project identifier
            
        Returns:
            MigrationProject or None if not found
        """
        return self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
    
    def get_project_by_uuid(self, project_uuid: uuid.UUID) -> Optional[MigrationProject]:
        """
        Retrieve a migration project by UUID.
        
        Args:
            project_uuid: Project UUID
            
        Returns:
            MigrationProject or None if not found
        """
        return self.db.query(MigrationProject).filter(
            MigrationProject.id == project_uuid
        ).first()
    
    def update_project_status(
        self,
        project_id: str,
        new_status: MigrationStatus,
        current_phase: Optional[str] = None
    ) -> MigrationProject:
        """
        Update migration project status and optionally the current phase.
        
        Args:
            project_id: Unique project identifier
            new_status: New migration status
            current_phase: Optional new phase description
            
        Returns:
            Updated MigrationProject
            
        Raises:
            ValueError: If project not found or invalid status transition
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Validate status transition
        if not self._is_valid_status_transition(project.status, new_status):
            raise ValueError(
                f"Invalid status transition from {project.status.value} to {new_status.value}"
            )
        
        project.status = new_status
        if current_phase:
            project.current_phase = current_phase
        
        # Set completion date if moving to COMPLETE status
        if new_status == MigrationStatus.COMPLETE and not project.actual_completion:
            project.actual_completion = datetime.utcnow()
        
        self.db.flush()
        
        logger.info(
            "Migration project status updated",
            project_id=project_id,
            old_status=project.status.value,
            new_status=new_status.value,
            phase=current_phase
        )
        
        return project
    
    def update_estimated_completion(
        self,
        project_id: str,
        estimated_completion: datetime
    ) -> MigrationProject:
        """
        Update the estimated completion date for a migration project.
        
        Args:
            project_id: Unique project identifier
            estimated_completion: New estimated completion datetime
            
        Returns:
            Updated MigrationProject
            
        Raises:
            ValueError: If project not found
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        project.estimated_completion = estimated_completion
        self.db.flush()
        
        logger.info(
            "Migration project estimated completion updated",
            project_id=project_id,
            estimated_completion=estimated_completion.isoformat()
        )
        
        return project
    
    def list_projects(
        self,
        status: Optional[MigrationStatus] = None,
        created_by: Optional[uuid.UUID] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MigrationProject]:
        """
        List migration projects with optional filtering.
        
        Args:
            status: Optional status filter
            created_by: Optional user filter
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of MigrationProject objects
        """
        query = self.db.query(MigrationProject)
        
        if status:
            query = query.filter(MigrationProject.status == status)
        
        if created_by:
            query = query.filter(MigrationProject.created_by == created_by)
        
        query = query.order_by(MigrationProject.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        return query.all()
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a migration project (soft delete by setting status to CANCELLED).
        
        Args:
            project_id: Unique project identifier
            
        Returns:
            True if deleted, False if not found
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        project.status = MigrationStatus.CANCELLED
        self.db.flush()
        
        logger.info("Migration project cancelled", project_id=project_id)
        return True
    
    def _generate_project_id(self, organization_name: str) -> str:
        """
        Generate a unique project ID based on organization name and timestamp.
        
        Args:
            organization_name: Organization name
            
        Returns:
            Unique project ID string
        """
        # Create a slug from organization name
        slug = organization_name.lower().replace(" ", "-")[:20]
        # Add timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # Add random suffix
        random_suffix = str(uuid.uuid4())[:8]
        
        return f"mig-{slug}-{timestamp}-{random_suffix}"
    
    def _is_valid_status_transition(
        self,
        current_status: MigrationStatus,
        new_status: MigrationStatus
    ) -> bool:
        """
        Validate if a status transition is allowed.
        
        Args:
            current_status: Current migration status
            new_status: Desired new status
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            MigrationStatus.ASSESSMENT: [
                MigrationStatus.ANALYSIS,
                MigrationStatus.CANCELLED
            ],
            MigrationStatus.ANALYSIS: [
                MigrationStatus.RECOMMENDATION,
                MigrationStatus.ASSESSMENT,  # Allow going back
                MigrationStatus.CANCELLED
            ],
            MigrationStatus.RECOMMENDATION: [
                MigrationStatus.PLANNING,
                MigrationStatus.ANALYSIS,  # Allow going back
                MigrationStatus.CANCELLED
            ],
            MigrationStatus.PLANNING: [
                MigrationStatus.EXECUTION,
                MigrationStatus.RECOMMENDATION,  # Allow going back
                MigrationStatus.CANCELLED
            ],
            MigrationStatus.EXECUTION: [
                MigrationStatus.COMPLETE,
                MigrationStatus.PLANNING,  # Allow going back
                MigrationStatus.CANCELLED
            ],
            MigrationStatus.COMPLETE: [],  # No transitions from complete
            MigrationStatus.CANCELLED: []  # No transitions from cancelled
        }
        
        # Allow staying in same status
        if current_status == new_status:
            return True
        
        return new_status in valid_transitions.get(current_status, [])


class OrganizationProfiler:
    """
    Manages organization profile data collection and validation.
    Implements Requirements: 1.2, 1.3
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_organization_profile(
        self,
        migration_project_id: uuid.UUID,
        company_size: CompanySize,
        industry: str,
        current_infrastructure: InfrastructureType,
        it_team_size: int,
        cloud_experience_level: ExperienceLevel,
        geographic_presence: Optional[List[str]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> OrganizationProfile:
        """
        Create an organization profile for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            company_size: Size category of the company
            industry: Industry sector
            current_infrastructure: Current infrastructure type
            it_team_size: Number of IT team members
            cloud_experience_level: Team's cloud experience level
            geographic_presence: List of regions/countries
            additional_context: Additional context data
            
        Returns:
            OrganizationProfile: Created profile
            
        Raises:
            ValueError: If validation fails or project not found
        """
        # Validate migration project exists
        project = self.db.query(MigrationProject).filter(
            MigrationProject.id == migration_project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {migration_project_id} not found")
        
        # Validate inputs
        self._validate_profile_data(
            company_size, industry, it_team_size, cloud_experience_level
        )
        
        # Create organization profile
        profile = OrganizationProfile(
            migration_project_id=migration_project_id,
            company_size=company_size,
            industry=industry.strip(),
            current_infrastructure=current_infrastructure,
            it_team_size=it_team_size,
            cloud_experience_level=cloud_experience_level,
            geographic_presence=geographic_presence or [],
            additional_context=additional_context or {}
        )
        
        try:
            self.db.add(profile)
            self.db.flush()
            
            logger.info(
                "Organization profile created",
                project_id=str(migration_project_id),
                company_size=company_size.value,
                industry=industry
            )
            
            return profile
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error("Failed to create organization profile", error=str(e))
            raise ValueError("Organization profile already exists for this project")
    
    def get_profile(self, migration_project_id: uuid.UUID) -> Optional[OrganizationProfile]:
        """
        Retrieve organization profile for a migration project.
        
        Args:
            migration_project_id: UUID of the migration project
            
        Returns:
            OrganizationProfile or None if not found
        """
        return self.db.query(OrganizationProfile).filter(
            OrganizationProfile.migration_project_id == migration_project_id
        ).first()
    
    def update_profile(
        self,
        migration_project_id: uuid.UUID,
        **updates
    ) -> OrganizationProfile:
        """
        Update organization profile fields.
        
        Args:
            migration_project_id: UUID of the migration project
            **updates: Fields to update
            
        Returns:
            Updated OrganizationProfile
            
        Raises:
            ValueError: If profile not found or validation fails
        """
        profile = self.get_profile(migration_project_id)
        if not profile:
            raise ValueError(f"Profile for project {migration_project_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'company_size', 'industry', 'current_infrastructure',
            'geographic_presence', 'it_team_size', 'cloud_experience_level',
            'additional_context'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(profile, field, value)
        
        self.db.flush()
        
        logger.info(
            "Organization profile updated",
            project_id=str(migration_project_id),
            updated_fields=list(updates.keys())
        )
        
        return profile
    
    def analyze_infrastructure_type(
        self,
        infrastructure_details: Dict[str, Any]
    ) -> InfrastructureType:
        """
        Analyze infrastructure details and determine infrastructure type.
        
        Args:
            infrastructure_details: Dictionary with infrastructure information
                Expected keys: 'on_premises', 'cloud_providers', 'hybrid'
                
        Returns:
            InfrastructureType: Detected infrastructure type
        """
        has_on_premises = infrastructure_details.get('on_premises', False)
        cloud_providers = infrastructure_details.get('cloud_providers', [])
        has_cloud = len(cloud_providers) > 0
        
        if has_on_premises and has_cloud:
            return InfrastructureType.HYBRID
        elif has_cloud and len(cloud_providers) > 1:
            return InfrastructureType.MULTI_CLOUD
        elif has_cloud:
            return InfrastructureType.CLOUD
        else:
            return InfrastructureType.ON_PREMISES
    
    def _validate_profile_data(
        self,
        company_size: CompanySize,
        industry: str,
        it_team_size: int,
        cloud_experience_level: ExperienceLevel
    ) -> None:
        """
        Validate organization profile data.
        
        Args:
            company_size: Company size enum
            industry: Industry string
            it_team_size: IT team size
            cloud_experience_level: Experience level enum
            
        Raises:
            ValueError: If validation fails
        """
        if not industry or len(industry.strip()) == 0:
            raise ValueError("Industry cannot be empty")
        
        if it_team_size < 0:
            raise ValueError("IT team size must be non-negative")
        
        # Validate company size matches IT team size reasonably
        size_team_ranges = {
            CompanySize.SMALL: (0, 10),
            CompanySize.MEDIUM: (1, 50),
            CompanySize.LARGE: (5, 500),
            CompanySize.ENTERPRISE: (10, 10000)
        }
        
        min_team, max_team = size_team_ranges[company_size]
        if not (min_team <= it_team_size <= max_team):
            logger.warning(
                "IT team size unusual for company size",
                company_size=company_size.value,
                it_team_size=it_team_size,
                expected_range=f"{min_team}-{max_team}"
            )


class AssessmentTimelineEstimator:
    """
    Estimates assessment timeline based on organization characteristics.
    Implements Requirements: 1.5
    """
    
    # Base assessment durations in days
    BASE_DURATIONS = {
        CompanySize.SMALL: 7,
        CompanySize.MEDIUM: 14,
        CompanySize.LARGE: 21,
        CompanySize.ENTERPRISE: 30
    }
    
    # Infrastructure complexity multipliers
    INFRASTRUCTURE_MULTIPLIERS = {
        InfrastructureType.ON_PREMISES: 1.0,
        InfrastructureType.CLOUD: 0.8,
        InfrastructureType.HYBRID: 1.3,
        InfrastructureType.MULTI_CLOUD: 1.5
    }
    
    # Experience level adjustments (in days)
    EXPERIENCE_ADJUSTMENTS = {
        ExperienceLevel.NONE: 5,
        ExperienceLevel.BEGINNER: 3,
        ExperienceLevel.INTERMEDIATE: 0,
        ExperienceLevel.ADVANCED: -3
    }
    
    def estimate_assessment_duration(
        self,
        company_size: CompanySize,
        current_infrastructure: InfrastructureType,
        cloud_experience_level: ExperienceLevel,
        it_team_size: int
    ) -> Dict[str, Any]:
        """
        Estimate the duration for completing the migration assessment.
        
        Args:
            company_size: Size of the company
            current_infrastructure: Current infrastructure type
            cloud_experience_level: Team's cloud experience
            it_team_size: Size of IT team
            
        Returns:
            Dictionary with estimation details:
                - estimated_days: Total estimated days
                - estimated_completion_date: Projected completion date
                - breakdown: Detailed breakdown of estimation factors
        """
        # Start with base duration
        base_days = self.BASE_DURATIONS[company_size]
        
        # Apply infrastructure complexity multiplier
        infrastructure_multiplier = self.INFRASTRUCTURE_MULTIPLIERS[current_infrastructure]
        adjusted_days = base_days * infrastructure_multiplier
        
        # Apply experience level adjustment
        experience_adjustment = self.EXPERIENCE_ADJUSTMENTS[cloud_experience_level]
        adjusted_days += experience_adjustment
        
        # Apply team size factor (larger teams can work faster)
        team_factor = self._calculate_team_factor(it_team_size, company_size)
        final_days = adjusted_days * team_factor
        
        # Round to nearest day, minimum 3 days
        final_days = max(3, round(final_days))
        
        # Calculate estimated completion date (business days)
        estimated_completion = self._add_business_days(datetime.utcnow(), final_days)
        
        breakdown = {
            'base_days': base_days,
            'infrastructure_multiplier': infrastructure_multiplier,
            'experience_adjustment_days': experience_adjustment,
            'team_size_factor': team_factor,
            'factors': {
                'company_size': company_size.value,
                'infrastructure_type': current_infrastructure.value,
                'experience_level': cloud_experience_level.value,
                'it_team_size': it_team_size
            }
        }
        
        logger.info(
            "Assessment timeline estimated",
            estimated_days=final_days,
            company_size=company_size.value,
            infrastructure=current_infrastructure.value
        )
        
        return {
            'estimated_days': final_days,
            'estimated_completion_date': estimated_completion,
            'breakdown': breakdown
        }
    
    def _calculate_team_factor(self, it_team_size: int, company_size: CompanySize) -> float:
        """
        Calculate team size factor for timeline adjustment.
        
        Larger teams relative to company size can complete assessments faster.
        
        Args:
            it_team_size: Size of IT team
            company_size: Company size category
            
        Returns:
            Multiplier factor (0.7 to 1.2)
        """
        # Expected team sizes for company sizes
        expected_team_sizes = {
            CompanySize.SMALL: 3,
            CompanySize.MEDIUM: 10,
            CompanySize.LARGE: 50,
            CompanySize.ENTERPRISE: 200
        }
        
        expected_size = expected_team_sizes[company_size]
        ratio = it_team_size / expected_size if expected_size > 0 else 1.0
        
        # Convert ratio to factor (more team = faster, but with diminishing returns)
        if ratio >= 1.5:
            return 0.7  # Large team, faster completion
        elif ratio >= 1.0:
            return 0.85
        elif ratio >= 0.5:
            return 1.0  # Normal team size
        else:
            return 1.2  # Small team, slower completion
    
    def _add_business_days(self, start_date: datetime, days: int) -> datetime:
        """
        Add business days to a date (excluding weekends).
        
        Args:
            start_date: Starting date
            days: Number of business days to add
            
        Returns:
            Resulting datetime
        """
        current_date = start_date
        days_added = 0
        
        while days_added < days:
            current_date += timedelta(days=1)
            # Skip weekends (5 = Saturday, 6 = Sunday)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date


class MigrationAssessmentEngine:
    """
    Main assessment engine that coordinates project management, profiling, and timeline estimation.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.project_manager = MigrationProjectManager(db_session)
        self.profiler = OrganizationProfiler(db_session)
        self.timeline_estimator = AssessmentTimelineEstimator()
    
    def initiate_migration_assessment(
        self,
        organization_name: str,
        created_by_user_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Initiate a new migration assessment by creating a project.
        
        Args:
            organization_name: Name of the organization
            created_by_user_id: UUID of the user creating the assessment
            
        Returns:
            Dictionary with project details
        """
        project = self.project_manager.create_migration_project(
            organization_name=organization_name,
            created_by_user_id=created_by_user_id
        )
        
        return {
            'project_id': project.project_id,
            'project_uuid': str(project.id),
            'organization_name': project.organization_name,
            'status': project.status.value,
            'current_phase': project.current_phase,
            'created_at': project.created_at.isoformat()
        }
    
    def collect_organization_profile(
        self,
        project_id: str,
        profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect and store organization profile data, then estimate timeline.
        
        Args:
            project_id: Migration project ID
            profile_data: Organization profile information
            
        Returns:
            Dictionary with profile and timeline estimation
        """
        # Get project
        project = self.project_manager.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Parse profile data
        company_size = CompanySize(profile_data['company_size'])
        industry = profile_data['industry']
        current_infrastructure = InfrastructureType(profile_data['current_infrastructure'])
        it_team_size = profile_data['it_team_size']
        cloud_experience_level = ExperienceLevel(profile_data['cloud_experience_level'])
        geographic_presence = profile_data.get('geographic_presence', [])
        additional_context = profile_data.get('additional_context', {})
        
        # Create organization profile
        profile = self.profiler.create_organization_profile(
            migration_project_id=project.id,
            company_size=company_size,
            industry=industry,
            current_infrastructure=current_infrastructure,
            it_team_size=it_team_size,
            cloud_experience_level=cloud_experience_level,
            geographic_presence=geographic_presence,
            additional_context=additional_context
        )
        
        # Estimate assessment timeline
        timeline = self.timeline_estimator.estimate_assessment_duration(
            company_size=company_size,
            current_infrastructure=current_infrastructure,
            cloud_experience_level=cloud_experience_level,
            it_team_size=it_team_size
        )
        
        # Update project with estimated completion
        self.project_manager.update_estimated_completion(
            project_id=project_id,
            estimated_completion=timeline['estimated_completion_date']
        )
        
        return {
            'profile': {
                'company_size': profile.company_size.value,
                'industry': profile.industry,
                'current_infrastructure': profile.current_infrastructure.value,
                'it_team_size': profile.it_team_size,
                'cloud_experience_level': profile.cloud_experience_level.value,
                'geographic_presence': profile.geographic_presence
            },
            'timeline_estimation': timeline
        }
    
    def validate_assessment_completeness(self, project_id: str) -> Dict[str, Any]:
        """
        Validate that all required assessment data has been collected.
        
        Args:
            project_id: Migration project ID
            
        Returns:
            Dictionary with validation results
        """
        project = self.project_manager.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        validation_results = {
            'is_complete': True,
            'missing_items': [],
            'warnings': []
        }
        
        # Check organization profile
        if not project.organization_profile:
            validation_results['is_complete'] = False
            validation_results['missing_items'].append('organization_profile')
        
        return validation_results
