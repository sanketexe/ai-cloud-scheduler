"""
Script to create tables directly without Alembic (for initial setup/testing)
"""
import asyncio
import sys
import os

# Add backend to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.database import engine, Base
from app.models.models import *

async def init_db():
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("Tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
