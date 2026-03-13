from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime

from app.services.startup_migration.models import (
    StartupMigrationProject,
    StartupDatabaseAssessment,
    ProjectStatus
)

class AssessmentService:
    """
    Service for managing startup migration assessments.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_project(self, company_name: str, email: str, phone: Optional[str] = None) -> StartupMigrationProject:
        """Create a new migration project"""
        project = StartupMigrationProject(
            company_name=company_name,
            email=email,
            phone=phone,
            status=ProjectStatus.ASSESSMENT
        )
        self.db.add(project)
        await self.db.commit()
        await self.db.refresh(project)
        return project

    async def create_assessment(self, project_id: UUID, assessment_data: Dict[str, Any]) -> StartupDatabaseAssessment:
        """Create a database assessment for a project"""
        assessment = StartupDatabaseAssessment(
            project_id=project_id,
            **assessment_data
        )
        self.db.add(assessment)
        await self.db.commit()
        await self.db.refresh(assessment)
        
        # Update project status if needed
        # project = await self.get_project(project_id)
        # project.status = ProjectStatus.COMPARISON
        # await self.db.commit()
        
        return assessment

    async def get_project(self, project_id: UUID) -> Optional[StartupMigrationProject]:
        """Get project by ID"""
        result = await self.db.execute(select(StartupMigrationProject).where(StartupMigrationProject.id == project_id))
        return result.scalars().first()

    async def get_assessment(self, project_id: UUID) -> Optional[StartupDatabaseAssessment]:
        """Get assessment by project ID"""
        result = await self.db.execute(select(StartupDatabaseAssessment).where(StartupDatabaseAssessment.project_id == project_id))
        return result.scalars().first()
