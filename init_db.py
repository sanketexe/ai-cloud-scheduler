"""Initialize SQLite database with all tables"""
from dotenv import load_dotenv
load_dotenv()

from backend.core.database import db_config, Base
from backend.core.models import *

# Create all tables
engine = db_config.create_sync_engine()
Base.metadata.create_all(engine)
print("✓ Database tables created successfully!")
print(f"✓ Database location: {db_config.database_url}")
