"""
Database configuration and connection management for FinOps Platform
"""

import os
from typing import AsyncGenerator, Optional, Generator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import structlog

logger = structlog.get_logger(__name__)

# Base class for all models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self):
        # Database URLs
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://finops:finops@localhost:5432/finops_db"
        )
        self.async_database_url = os.getenv(
            "ASYNC_DATABASE_URL",
            "postgresql+asyncpg://finops:finops@localhost:5432/finops_db"
        )
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "30"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Connection retry settings
        self.max_retries = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("DB_RETRY_DELAY", "1"))
        
        # Initialize engines
        self._sync_engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
    
    def create_sync_engine(self):
        """Create synchronous database engine"""
        if self._sync_engine is None:
            self._sync_engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Validate connections before use
                echo=os.getenv("DB_ECHO", "false").lower() == "true"
            )
            
            # Add connection event listeners
            @event.listens_for(self._sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                """Set database-specific settings on connection"""
                if "postgresql" in self.database_url:
                    # Set timezone to UTC for PostgreSQL
                    with dbapi_connection.cursor() as cursor:
                        cursor.execute("SET timezone TO 'UTC'")
            
            logger.info("Synchronous database engine created", 
                       url=self.database_url.split('@')[0] + '@***')
        
        return self._sync_engine
    
    def create_async_engine(self):
        """Create asynchronous database engine"""
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                self.async_database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,
                echo=os.getenv("DB_ECHO", "false").lower() == "true"
            )
            
            logger.info("Asynchronous database engine created",
                       url=self.async_database_url.split('@')[0] + '@***')
        
        return self._async_engine
    
    def get_session_factory(self) -> sessionmaker:
        """Get synchronous session factory"""
        if self._session_factory is None:
            engine = self.create_sync_engine()
            self._session_factory = sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    def get_async_session_factory(self) -> async_sessionmaker:
        """Get asynchronous session factory"""
        if self._async_session_factory is None:
            engine = self.create_async_engine()
            self._async_session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    def create_tables(self):
        """Create all database tables"""
        try:
            engine = self.create_sync_engine()
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            engine = self.create_sync_engine()
            Base.metadata.drop_all(bind=engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise
    
    async def check_connection(self) -> bool:
        """Check if database connection is healthy"""
        try:
            from sqlalchemy import text
            async_engine = self.create_async_engine()
            async with async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection check successful")
            return True
        except Exception as e:
            logger.error("Database connection check failed", error=str(e))
            return False
    
    def close_connections(self):
        """Close all database connections"""
        if self._sync_engine:
            self._sync_engine.dispose()
            logger.info("Synchronous database connections closed")
        
        if self._async_engine:
            self._async_engine.dispose()
            logger.info("Asynchronous database connections closed")

# Global database configuration instance
db_config = DatabaseConfig()

# Dependency for FastAPI
def get_db_session() -> Generator[Session, None, None]:
    """Dependency to get database session for FastAPI endpoints"""
    session_factory = db_config.get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session error", error=str(e))
        raise
    finally:
        session.close()

async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session for FastAPI endpoints"""
    async_session_factory = db_config.get_async_session_factory()
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Async database session error", error=str(e))
            raise

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions"""
    async_session_factory = db_config.get_async_session_factory()
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database context error", error=str(e))
            raise

# Health check function
async def database_health_check() -> dict:
    """Health check for database connectivity"""
    try:
        is_healthy = await db_config.check_connection()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database": "postgresql",
            "pool_size": db_config.pool_size,
            "max_overflow": db_config.max_overflow
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": "postgresql"
        }