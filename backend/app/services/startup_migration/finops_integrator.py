from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.services.startup_migration.models import StartupFinOpsIntegration

class FinOpsIntegrator:
    """
    Handles integration with external FinOps tools or internal tracking.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def setup_integration(self, project_id: UUID, config: Dict[str, Any]) -> StartupFinOpsIntegration:
        """Configure FinOps integration"""
        
        integration = StartupFinOpsIntegration(
            project_id=project_id,
            finops_organization_id=None, # Mock for now
            organization_name=config.get('organization_name', 'My Startup'),
            monthly_budget=config.get('budget', 1000.00),
            notification_email=config.get('email', 'admin@example.com'),
            notification_preferences={'email': True, 'slack': False},
            integration_status="active"
        )
        
        self.db.add(integration)
        await self.db.commit()
        await self.db.refresh(integration)
        
        return integration

    async def get_integration(self, project_id: UUID) -> Optional[StartupFinOpsIntegration]:
        """Get integration status"""
        result = await self.db.execute(select(StartupFinOpsIntegration).where(StartupFinOpsIntegration.project_id == project_id))
        return result.scalars().first()
