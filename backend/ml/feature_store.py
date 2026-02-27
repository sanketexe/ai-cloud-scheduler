"""
Centralized Feature Store for ML Feature Management

Provides centralized storage, versioning, and serving of ML features
for advanced AI/ML systems. Supports feature discovery, lineage tracking,
and real-time feature serving for production ML models.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
import json
import sqlite3
import hashlib
import threading
from pathlib import Path
from collections import defaultdict

logger = structlog.get_logger(__name__)


class FeatureType(Enum):
    """Types of features in the feature store"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    EMBEDDING = "embedding"
    TEXT = "text"


class FeatureStatus(Enum):
    """Status of features in the store"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"


@dataclass
class FeatureDefinition:
    """Definition of a feature in the store"""
    feature_name: str
    feature_type: FeatureType
    description: str
    source_table: str
    computation_logic: str
    
    # Metadata
    owner: str
    tags: List[str] = field(default_factory=list)
    status: FeatureStatus = FeatureStatus.ACTIVE
    version: str = "1.0.0"
    
    # Data characteristics
    expected_data_type: str = "float64"
    expected_range: Optional[Tuple[float, float]] = None
    null_allowed: bool = True
    
    # Lineage and dependencies
    upstream_features: List[str] = field(default_factory=list)
    downstream_models: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    usage_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class FeatureGroup:
    """Group of related features"""
    group_name: str
    description: str
    features: List[str]
    owner: str
    
    # Configuration
    refresh_interval_minutes: int = 60
    retention_days: int = 90
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureValue:
    """Individual feature value with metadata"""
    feature_name: str
    entity_id: str
    value: Any
    timestamp: datetime
    version: str = "1.0.0"
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Complete feature vector for an entity"""
    entity_id: str
    timestamp: datetime
    features: Dict[str, Any]
    feature_names: List[str]
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """
    Centralized Feature Store for ML Feature Management.
    
    Provides feature definition, storage, versioning, serving,
    and lineage tracking for production ML systems.
    """
    
    def __init__(self, storage_path: str = "feature_store_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database connections
        self.metadata_db_path = self.storage_path / "feature_metadata.db"
        self.features_db_path = self.storage_path / "feature_values.db"
        
        self.metadata_conn = None
        self.features_conn = None
        
        # In-memory caches
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_cache: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        
        # Configuration
        self.cache_size_limit = 10000
        self.cache_ttl_minutes = 30
        self.batch_size = 1000
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'features_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'feature_updates': 0,
            'last_cleanup': datetime.utcnow()
        }
        
        logger.info("Feature Store initialized", storage_path=str(self.storage_path))
    
    async def initialize(self):
        """Initialize feature store databases and load metadata"""
        logger.info("Initializing Feature Store")
        
        try:
            # Initialize databases
            await self._initialize_databases()
            
            # Load existing feature definitions
            await self._load_feature_definitions()
            
            # Load feature groups
            await self._load_feature_groups()
            
            logger.info("Feature Store initialization completed",
                       features=len(self.feature_definitions),
                       groups=len(self.feature_groups))
            
        except Exception as e:
            logger.error("Feature Store initialization failed", error=str(e))
            raise
    
    async def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """
        Register a new feature definition.
        
        Args:
            feature_def: Feature definition to register
            
        Returns:
            True if registered successfully
        """
        logger.info("Registering feature", feature_name=feature_def.feature_name)
        
        try:
            # Validate feature definition
            if not self._validate_feature_definition(feature_def):
                logger.error("Feature definition validation failed", 
                           feature_name=feature_def.feature_name)
                return False
            
            # Check for existing feature
            if feature_def.feature_name in self.feature_definitions:
                existing_def = self.feature_definitions[feature_def.feature_name]
                if existing_def.version == feature_def.version:
                    logger.warning("Feature version already exists", 
                                 feature_name=feature_def.feature_name,
                                 version=feature_def.version)
                    return False
            
            # Store in database
            await self._store_feature_definition(feature_def)
            
            # Update in-memory cache
            with self.lock:
                self.feature_definitions[feature_def.feature_name] = feature_def
            
            logger.info("Feature registered successfully", 
                       feature_name=feature_def.feature_name,
                       version=feature_def.version)
            
            return True
            
        except Exception as e:
            logger.error("Feature registration failed", 
                        error=str(e), 
                        feature_name=feature_def.feature_name)
            return False
    
    async def create_feature_group(self, group: FeatureGroup) -> bool:
        """
        Create a new feature group.
        
        Args:
            group: Feature group to create
            
        Returns:
            True if created successfully
        """
        logger.info("Creating feature group", group_name=group.group_name)
        
        try:
            # Validate feature group
            if not self._validate_feature_group(group):
                logger.error("Feature group validation failed", 
                           group_name=group.group_name)
                return False
            
            # Store in database
            await self._store_feature_group(group)
            
            # Update in-memory cache
            with self.lock:
                self.feature_groups[group.group_name] = group
            
            logger.info("Feature group created successfully", 
                       group_name=group.group_name,
                       features=len(group.features))
            
            return True
            
        except Exception as e:
            logger.error("Feature group creation failed", 
                        error=str(e), 
                        group_name=group.group_name)
            return False
    
    async def store_feature_values(self, feature_values: List[FeatureValue]) -> int:
        """
        Store feature values in the feature store.
        
        Args:
            feature_values: List of feature values to store
            
        Returns:
            Number of values stored successfully
        """
        logger.info("Storing feature values", count=len(feature_values))
        
        try:
            stored_count = 0
            
            # Group by feature name for efficient storage
            feature_groups = defaultdict(list)
            for fv in feature_values:
                feature_groups[fv.feature_name].append(fv)
            
            # Store each feature group
            for feature_name, values in feature_groups.items():
                count = await self._store_feature_value_batch(feature_name, values)
                stored_count += count
                
                # Update cache
                await self._update_feature_cache(feature_name, values)
            
            # Update statistics
            self.stats['feature_updates'] += stored_count
            
            logger.info("Feature values stored successfully", 
                       stored=stored_count, 
                       total=len(feature_values))
            
            return stored_count
            
        except Exception as e:
            logger.error("Feature value storage failed", error=str(e))
            return 0
    
    async def get_feature_vector(self, 
                               entity_id: str,
                               feature_names: List[str],
                               timestamp: Optional[datetime] = None) -> Optional[FeatureVector]:
        """
        Get feature vector for an entity.
        
        Args:
            entity_id: Entity identifier
            feature_names: List of feature names to retrieve
            timestamp: Point-in-time for feature values (default: latest)
            
        Returns:
            Feature vector or None if not found
        """
        logger.debug("Getting feature vector", 
                    entity_id=entity_id, 
                    features=len(feature_names))
        
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            features = {}
            retrieved_features = []
            
            # Get each feature value
            for feature_name in feature_names:
                value = await self._get_feature_value(entity_id, feature_name, timestamp)
                if value is not None:
                    features[feature_name] = value.value
                    retrieved_features.append(feature_name)
            
            if not features:
                logger.debug("No features found for entity", entity_id=entity_id)
                return None
            
            # Update usage statistics
            await self._update_feature_usage_stats(retrieved_features)
            
            # Update statistics
            self.stats['features_served'] += len(features)
            
            feature_vector = FeatureVector(
                entity_id=entity_id,
                timestamp=timestamp,
                features=features,
                feature_names=retrieved_features,
                metadata={
                    'requested_features': feature_names,
                    'retrieved_features': retrieved_features,
                    'retrieval_timestamp': datetime.utcnow()
                }
            )
            
            logger.debug("Feature vector retrieved", 
                        entity_id=entity_id, 
                        features_retrieved=len(features))
            
            return feature_vector
            
        except Exception as e:
            logger.error("Feature vector retrieval failed", 
                        error=str(e), 
                        entity_id=entity_id)
            return None
    
    async def get_feature_group_vector(self, 
                                     entity_id: str,
                                     group_name: str,
                                     timestamp: Optional[datetime] = None) -> Optional[FeatureVector]:
        """
        Get feature vector for a feature group.
        
        Args:
            entity_id: Entity identifier
            group_name: Feature group name
            timestamp: Point-in-time for feature values
            
        Returns:
            Feature vector or None if not found
        """
        logger.debug("Getting feature group vector", 
                    entity_id=entity_id, 
                    group_name=group_name)
        
        try:
            if group_name not in self.feature_groups:
                logger.error("Feature group not found", group_name=group_name)
                return None
            
            group = self.feature_groups[group_name]
            return await self.get_feature_vector(entity_id, group.features, timestamp)
            
        except Exception as e:
            logger.error("Feature group vector retrieval failed", 
                        error=str(e), 
                        entity_id=entity_id,
                        group_name=group_name)
            return None
    
    async def get_batch_feature_vectors(self, 
                                      entity_ids: List[str],
                                      feature_names: List[str],
                                      timestamp: Optional[datetime] = None) -> List[FeatureVector]:
        """
        Get feature vectors for multiple entities.
        
        Args:
            entity_ids: List of entity identifiers
            feature_names: List of feature names to retrieve
            timestamp: Point-in-time for feature values
            
        Returns:
            List of feature vectors
        """
        logger.info("Getting batch feature vectors", 
                   entities=len(entity_ids), 
                   features=len(feature_names))
        
        try:
            vectors = []
            
            # Process in batches for efficiency
            for i in range(0, len(entity_ids), self.batch_size):
                batch_entities = entity_ids[i:i + self.batch_size]
                
                # Get vectors for batch
                batch_vectors = await asyncio.gather(*[
                    self.get_feature_vector(entity_id, feature_names, timestamp)
                    for entity_id in batch_entities
                ], return_exceptions=True)
                
                # Filter out None values and exceptions
                for vector in batch_vectors:
                    if isinstance(vector, FeatureVector):
                        vectors.append(vector)
            
            logger.info("Batch feature vectors retrieved", 
                       requested=len(entity_ids), 
                       retrieved=len(vectors))
            
            return vectors
            
        except Exception as e:
            logger.error("Batch feature vector retrieval failed", error=str(e))
            return []
    
    async def search_features(self, 
                            query: str,
                            feature_type: Optional[FeatureType] = None,
                            tags: Optional[List[str]] = None,
                            owner: Optional[str] = None) -> List[FeatureDefinition]:
        """
        Search for features based on criteria.
        
        Args:
            query: Search query (matches name and description)
            feature_type: Optional feature type filter
            tags: Optional tag filters
            owner: Optional owner filter
            
        Returns:
            List of matching feature definitions
        """
        logger.info("Searching features", 
                   query=query, 
                   feature_type=feature_type,
                   tags=tags,
                   owner=owner)
        
        try:
            matching_features = []
            
            for feature_def in self.feature_definitions.values():
                # Check query match
                if query.lower() not in feature_def.feature_name.lower() and \
                   query.lower() not in feature_def.description.lower():
                    continue
                
                # Check type filter
                if feature_type and feature_def.feature_type != feature_type:
                    continue
                
                # Check tag filters
                if tags and not any(tag in feature_def.tags for tag in tags):
                    continue
                
                # Check owner filter
                if owner and feature_def.owner != owner:
                    continue
                
                matching_features.append(feature_def)
            
            logger.info("Feature search completed", 
                       query=query, 
                       matches=len(matching_features))
            
            return matching_features
            
        except Exception as e:
            logger.error("Feature search failed", error=str(e))
            return []
    
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """
        Get feature lineage information.
        
        Args:
            feature_name: Feature name to get lineage for
            
        Returns:
            Dictionary with upstream and downstream dependencies
        """
        logger.info("Getting feature lineage", feature_name=feature_name)
        
        try:
            if feature_name not in self.feature_definitions:
                logger.error("Feature not found", feature_name=feature_name)
                return {}
            
            feature_def = self.feature_definitions[feature_name]
            
            # Build lineage graph
            lineage = {
                'feature_name': feature_name,
                'upstream_features': feature_def.upstream_features,
                'downstream_models': feature_def.downstream_models,
                'feature_groups': [],
                'dependencies': {}
            }
            
            # Find feature groups containing this feature
            for group_name, group in self.feature_groups.items():
                if feature_name in group.features:
                    lineage['feature_groups'].append(group_name)
            
            # Get recursive dependencies
            lineage['dependencies'] = await self._get_recursive_dependencies(feature_name)
            
            logger.info("Feature lineage retrieved", 
                       feature_name=feature_name,
                       upstream=len(feature_def.upstream_features),
                       downstream=len(feature_def.downstream_models))
            
            return lineage
            
        except Exception as e:
            logger.error("Feature lineage retrieval failed", 
                        error=str(e), 
                        feature_name=feature_name)
            return {}
    
    async def get_feature_statistics(self, 
                                   feature_name: str,
                                   days: int = 30) -> Dict[str, Any]:
        """
        Get feature usage and quality statistics.
        
        Args:
            feature_name: Feature name
            days: Number of days to analyze
            
        Returns:
            Dictionary with feature statistics
        """
        logger.info("Getting feature statistics", 
                   feature_name=feature_name, 
                   days=days)
        
        try:
            if feature_name not in self.feature_definitions:
                logger.error("Feature not found", feature_name=feature_name)
                return {}
            
            feature_def = self.feature_definitions[feature_name]
            
            # Get recent feature values for analysis
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            recent_values = await self._get_recent_feature_values(feature_name, cutoff_date)
            
            statistics = {
                'feature_name': feature_name,
                'feature_type': feature_def.feature_type.value,
                'status': feature_def.status.value,
                'version': feature_def.version,
                'usage_count': feature_def.usage_count,
                'last_accessed': feature_def.last_accessed.isoformat() if feature_def.last_accessed else None,
                'analysis_period_days': days,
                'data_points': len(recent_values),
                'value_statistics': {},
                'quality_statistics': {}
            }
            
            if recent_values:
                # Value statistics
                values = [fv.value for fv in recent_values if isinstance(fv.value, (int, float))]
                if values:
                    statistics['value_statistics'] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
                
                # Quality statistics
                quality_scores = [fv.quality_score for fv in recent_values]
                statistics['quality_statistics'] = {
                    'avg_quality_score': np.mean(quality_scores),
                    'min_quality_score': np.min(quality_scores),
                    'quality_violations': len([q for q in quality_scores if q < 0.8])
                }
                
                # Temporal statistics
                timestamps = [fv.timestamp for fv in recent_values]
                statistics['temporal_statistics'] = {
                    'first_timestamp': min(timestamps).isoformat(),
                    'last_timestamp': max(timestamps).isoformat(),
                    'update_frequency_hours': days * 24 / len(recent_values) if recent_values else 0
                }
            
            logger.info("Feature statistics retrieved", 
                       feature_name=feature_name,
                       data_points=len(recent_values))
            
            return statistics
            
        except Exception as e:
            logger.error("Feature statistics retrieval failed", 
                        error=str(e), 
                        feature_name=feature_name)
            return {}
    
    def get_store_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics and statistics"""
        with self.lock:
            return {
                'feature_definitions': len(self.feature_definitions),
                'feature_groups': len(self.feature_groups),
                'cache_size': sum(len(entity_cache) for entity_cache in self.feature_cache.values()),
                'statistics': self.stats.copy(),
                'storage_info': {
                    'metadata_db_size': self._get_file_size(self.metadata_db_path),
                    'features_db_size': self._get_file_size(self.features_db_path),
                    'storage_path': str(self.storage_path)
                }
            }
    
    async def cleanup_old_features(self, retention_days: int = 90) -> Dict[str, int]:
        """
        Clean up old feature values beyond retention period.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Cleanup statistics
        """
        logger.info("Cleaning up old features", retention_days=retention_days)
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Clean up feature values
            deleted_values = await self._cleanup_old_feature_values(cutoff_date)
            
            # Clean up cache
            cache_cleaned = await self._cleanup_feature_cache(cutoff_date)
            
            # Update statistics
            self.stats['last_cleanup'] = datetime.utcnow()
            
            cleanup_stats = {
                'deleted_feature_values': deleted_values,
                'cache_entries_cleaned': cache_cleaned,
                'retention_days': retention_days,
                'cleanup_date': datetime.utcnow().isoformat()
            }
            
            logger.info("Feature cleanup completed", stats=cleanup_stats)
            return cleanup_stats
            
        except Exception as e:
            logger.error("Feature cleanup failed", error=str(e))
            return {}
    
    async def _initialize_databases(self):
        """Initialize SQLite databases for metadata and feature values"""
        # Metadata database
        self.metadata_conn = sqlite3.connect(str(self.metadata_db_path), check_same_thread=False)
        self.metadata_conn.execute("PRAGMA journal_mode=WAL")
        
        # Feature definitions table
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_definitions (
                feature_name TEXT PRIMARY KEY,
                feature_type TEXT NOT NULL,
                description TEXT,
                source_table TEXT,
                computation_logic TEXT,
                owner TEXT,
                tags TEXT,
                status TEXT,
                version TEXT,
                expected_data_type TEXT,
                expected_range TEXT,
                null_allowed BOOLEAN,
                upstream_features TEXT,
                downstream_models TEXT,
                created_at TEXT,
                updated_at TEXT,
                usage_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        
        # Feature groups table
        self.metadata_conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_groups (
                group_name TEXT PRIMARY KEY,
                description TEXT,
                features TEXT,
                owner TEXT,
                refresh_interval_minutes INTEGER,
                retention_days INTEGER,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        self.metadata_conn.commit()
        
        # Feature values database
        self.features_conn = sqlite3.connect(str(self.features_db_path), check_same_thread=False)
        self.features_conn.execute("PRAGMA journal_mode=WAL")
        
        # Feature values table
        self.features_conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                version TEXT,
                quality_score REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(feature_name, entity_id, timestamp)
            )
        """)
        
        # Create indexes for performance
        self.features_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_entity_time 
            ON feature_values(feature_name, entity_id, timestamp DESC)
        """)
        
        self.features_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON feature_values(timestamp DESC)
        """)
        
        self.features_conn.commit()
    
    async def _load_feature_definitions(self):
        """Load feature definitions from database"""
        cursor = self.metadata_conn.cursor()
        cursor.execute("SELECT * FROM feature_definitions")
        
        for row in cursor.fetchall():
            feature_def = self._row_to_feature_definition(row)
            self.feature_definitions[feature_def.feature_name] = feature_def
    
    async def _load_feature_groups(self):
        """Load feature groups from database"""
        cursor = self.metadata_conn.cursor()
        cursor.execute("SELECT * FROM feature_groups")
        
        for row in cursor.fetchall():
            group = self._row_to_feature_group(row)
            self.feature_groups[group.group_name] = group
    
    def _validate_feature_definition(self, feature_def: FeatureDefinition) -> bool:
        """Validate feature definition"""
        if not feature_def.feature_name or not feature_def.description:
            return False
        
        if not feature_def.owner:
            return False
        
        return True
    
    def _validate_feature_group(self, group: FeatureGroup) -> bool:
        """Validate feature group"""
        if not group.group_name or not group.description:
            return False
        
        if not group.features:
            return False
        
        if not group.owner:
            return False
        
        return True
    
    async def _store_feature_definition(self, feature_def: FeatureDefinition):
        """Store feature definition in database"""
        cursor = self.metadata_conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feature_definitions 
            (feature_name, feature_type, description, source_table, computation_logic,
             owner, tags, status, version, expected_data_type, expected_range, null_allowed,
             upstream_features, downstream_models, created_at, updated_at, usage_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feature_def.feature_name,
            feature_def.feature_type.value,
            feature_def.description,
            feature_def.source_table,
            feature_def.computation_logic,
            feature_def.owner,
            json.dumps(feature_def.tags),
            feature_def.status.value,
            feature_def.version,
            feature_def.expected_data_type,
            json.dumps(feature_def.expected_range) if feature_def.expected_range else None,
            feature_def.null_allowed,
            json.dumps(feature_def.upstream_features),
            json.dumps(feature_def.downstream_models),
            feature_def.created_at.isoformat(),
            feature_def.updated_at.isoformat(),
            feature_def.usage_count,
            feature_def.last_accessed.isoformat() if feature_def.last_accessed else None
        ))
        
        self.metadata_conn.commit()
    
    async def _store_feature_group(self, group: FeatureGroup):
        """Store feature group in database"""
        cursor = self.metadata_conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feature_groups 
            (group_name, description, features, owner, refresh_interval_minutes, 
             retention_days, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            group.group_name,
            group.description,
            json.dumps(group.features),
            group.owner,
            group.refresh_interval_minutes,
            group.retention_days,
            json.dumps(group.tags),
            group.created_at.isoformat(),
            group.updated_at.isoformat()
        ))
        
        self.metadata_conn.commit()
    
    async def _store_feature_value_batch(self, feature_name: str, values: List[FeatureValue]) -> int:
        """Store batch of feature values"""
        cursor = self.features_conn.cursor()
        
        records = []
        for fv in values:
            records.append((
                fv.feature_name,
                fv.entity_id,
                json.dumps(fv.value),
                fv.timestamp.isoformat(),
                fv.version,
                fv.quality_score,
                json.dumps(fv.metadata)
            ))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO feature_values 
            (feature_name, entity_id, value, timestamp, version, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self.features_conn.commit()
        return len(records)
    
    async def _get_feature_value(self, 
                                entity_id: str, 
                                feature_name: str, 
                                timestamp: datetime) -> Optional[FeatureValue]:
        """Get feature value for entity at specific timestamp"""
        # Check cache first
        cache_key = f"{entity_id}_{feature_name}"
        if cache_key in self.feature_cache.get(entity_id, {}):
            cached_value = self.feature_cache[entity_id][cache_key]
            if (timestamp - cached_value.timestamp).total_seconds() < self.cache_ttl_minutes * 60:
                self.stats['cache_hits'] += 1
                return cached_value
        
        self.stats['cache_misses'] += 1
        
        # Query database
        cursor = self.features_conn.cursor()
        cursor.execute("""
            SELECT feature_name, entity_id, value, timestamp, version, quality_score, metadata
            FROM feature_values
            WHERE feature_name = ? AND entity_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (feature_name, entity_id, timestamp.isoformat()))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        feature_value = FeatureValue(
            feature_name=row[0],
            entity_id=row[1],
            value=json.loads(row[2]),
            timestamp=datetime.fromisoformat(row[3]),
            version=row[4],
            quality_score=row[5],
            metadata=json.loads(row[6]) if row[6] else {}
        )
        
        return feature_value
    
    async def _update_feature_cache(self, feature_name: str, values: List[FeatureValue]):
        """Update feature cache with new values"""
        with self.lock:
            for fv in values:
                if fv.entity_id not in self.feature_cache:
                    self.feature_cache[fv.entity_id] = {}
                
                cache_key = f"{fv.entity_id}_{feature_name}"
                self.feature_cache[fv.entity_id][cache_key] = fv
                
                # Limit cache size
                if len(self.feature_cache) > self.cache_size_limit:
                    # Remove oldest entries
                    oldest_entity = min(self.feature_cache.keys())
                    del self.feature_cache[oldest_entity]
    
    async def _update_feature_usage_stats(self, feature_names: List[str]):
        """Update feature usage statistics"""
        current_time = datetime.utcnow()
        
        for feature_name in feature_names:
            if feature_name in self.feature_definitions:
                feature_def = self.feature_definitions[feature_name]
                feature_def.usage_count += 1
                feature_def.last_accessed = current_time
                
                # Update in database
                cursor = self.metadata_conn.cursor()
                cursor.execute("""
                    UPDATE feature_definitions 
                    SET usage_count = ?, last_accessed = ?
                    WHERE feature_name = ?
                """, (feature_def.usage_count, current_time.isoformat(), feature_name))
                
                self.metadata_conn.commit()
    
    async def _get_recursive_dependencies(self, feature_name: str, visited: Optional[set] = None) -> Dict[str, Any]:
        """Get recursive feature dependencies"""
        if visited is None:
            visited = set()
        
        if feature_name in visited:
            return {}
        
        visited.add(feature_name)
        
        if feature_name not in self.feature_definitions:
            return {}
        
        feature_def = self.feature_definitions[feature_name]
        dependencies = {
            'direct_upstream': feature_def.upstream_features,
            'indirect_upstream': {}
        }
        
        # Get indirect dependencies
        for upstream_feature in feature_def.upstream_features:
            indirect_deps = await self._get_recursive_dependencies(upstream_feature, visited.copy())
            if indirect_deps:
                dependencies['indirect_upstream'][upstream_feature] = indirect_deps
        
        return dependencies
    
    async def _get_recent_feature_values(self, feature_name: str, cutoff_date: datetime) -> List[FeatureValue]:
        """Get recent feature values for analysis"""
        cursor = self.features_conn.cursor()
        cursor.execute("""
            SELECT feature_name, entity_id, value, timestamp, version, quality_score, metadata
            FROM feature_values
            WHERE feature_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (feature_name, cutoff_date.isoformat()))
        
        values = []
        for row in cursor.fetchall():
            fv = FeatureValue(
                feature_name=row[0],
                entity_id=row[1],
                value=json.loads(row[2]),
                timestamp=datetime.fromisoformat(row[3]),
                version=row[4],
                quality_score=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            )
            values.append(fv)
        
        return values
    
    async def _cleanup_old_feature_values(self, cutoff_date: datetime) -> int:
        """Clean up old feature values"""
        cursor = self.features_conn.cursor()
        cursor.execute("""
            DELETE FROM feature_values 
            WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        deleted_count = cursor.rowcount
        self.features_conn.commit()
        
        return deleted_count
    
    async def _cleanup_feature_cache(self, cutoff_date: datetime) -> int:
        """Clean up old cache entries"""
        cleaned_count = 0
        
        with self.lock:
            entities_to_remove = []
            
            for entity_id, entity_cache in self.feature_cache.items():
                features_to_remove = []
                
                for cache_key, feature_value in entity_cache.items():
                    if feature_value.timestamp < cutoff_date:
                        features_to_remove.append(cache_key)
                
                for cache_key in features_to_remove:
                    del entity_cache[cache_key]
                    cleaned_count += 1
                
                if not entity_cache:
                    entities_to_remove.append(entity_id)
            
            for entity_id in entities_to_remove:
                del self.feature_cache[entity_id]
        
        return cleaned_count
    
    def _row_to_feature_definition(self, row) -> FeatureDefinition:
        """Convert database row to FeatureDefinition"""
        return FeatureDefinition(
            feature_name=row[0],
            feature_type=FeatureType(row[1]),
            description=row[2],
            source_table=row[3],
            computation_logic=row[4],
            owner=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            status=FeatureStatus(row[7]),
            version=row[8],
            expected_data_type=row[9],
            expected_range=json.loads(row[10]) if row[10] else None,
            null_allowed=bool(row[11]),
            upstream_features=json.loads(row[12]) if row[12] else [],
            downstream_models=json.loads(row[13]) if row[13] else [],
            created_at=datetime.fromisoformat(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
            usage_count=row[16],
            last_accessed=datetime.fromisoformat(row[17]) if row[17] else None
        )
    
    def _row_to_feature_group(self, row) -> FeatureGroup:
        """Convert database row to FeatureGroup"""
        return FeatureGroup(
            group_name=row[0],
            description=row[1],
            features=json.loads(row[2]),
            owner=row[3],
            refresh_interval_minutes=row[4],
            retention_days=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8])
        )
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            return 0
    
    async def close(self):
        """Close database connections"""
        if self.metadata_conn:
            self.metadata_conn.close()
        
        if self.features_conn:
            self.features_conn.close()
        
        logger.info("Feature Store connections closed")