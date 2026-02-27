"""
ML Model Registry for Model Versioning and Deployment

Provides centralized model versioning, artifact management, deployment tracking,
and lifecycle management for advanced AI/ML systems. Supports model lineage,
A/B testing, canary deployments, and automated rollbacks.
"""

import asyncio
import pickle
import json
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import structlog
import sqlite3
import threading
from collections import defaultdict
import uuid
import tarfile
import tempfile

logger = structlog.get_logger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    REGISTERED = "registered"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStage(Enum):
    """Deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    SHADOW = "shadow"


class ModelType(Enum):
    """Types of ML models"""
    PREDICTIVE_SCALING = "predictive_scaling"
    WORKLOAD_INTELLIGENCE = "workload_intelligence"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    NATURAL_LANGUAGE = "natural_language"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CONTRACT_OPTIMIZER = "contract_optimizer"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    
    # Model information
    description: str
    framework: str  # pytorch, tensorflow, sklearn, etc.
    algorithm: str
    hyperparameters: Dict[str, Any]
    
    # Training information
    training_dataset_hash: str
    training_start_time: datetime
    training_end_time: Optional[datetime]
    training_duration_seconds: Optional[float]
    
    # Model artifacts
    model_artifact_path: str
    model_size_bytes: int
    artifact_checksum: str
    
    # Feature information
    input_features: List[str]
    output_schema: Dict[str, str]
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    production_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Deployment information
    status: ModelStatus = ModelStatus.REGISTERED
    deployment_stage: Optional[DeploymentStage] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage and dependencies
    parent_model_id: Optional[str] = None
    derived_models: List[str] = field(default_factory=list)
    upstream_models: List[str] = field(default_factory=list)
    downstream_models: List[str] = field(default_factory=list)
    
    # Metadata
    owner: str = "system"
    tags: List[str] = field(default_factory=list)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Usage tracking
    prediction_count: int = 0
    last_prediction_time: Optional[datetime] = None
    error_count: int = 0
    
    # Compliance and governance
    approval_status: str = "pending"
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    compliance_checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Record of model deployment"""
    deployment_id: str
    model_id: str
    deployment_stage: DeploymentStage
    deployment_config: Dict[str, Any]
    
    # Deployment timing
    deployed_at: datetime
    undeployed_at: Optional[datetime] = None
    
    # Performance tracking
    requests_served: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Status
    is_active: bool = True
    health_status: str = "healthy"
    
    # Metadata
    deployed_by: str = "system"
    notes: str = ""


@dataclass
class ModelComparison:
    """Comparison between two models"""
    comparison_id: str
    model_a_id: str
    model_b_id: str
    comparison_type: str  # "a_b_test", "champion_challenger", "performance"
    
    # Comparison metrics
    metrics_comparison: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, bool]
    winner: Optional[str] = None
    confidence_level: float = 0.95
    
    # Test configuration
    test_start_time: datetime
    test_end_time: Optional[datetime] = None
    sample_size: int = 0
    traffic_split: Dict[str, float] = field(default_factory=dict)
    
    # Results
    results_summary: str = ""
    recommendation: str = ""
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)


class MLModelRegistry:
    """
    ML Model Registry for comprehensive model lifecycle management.
    
    Provides model versioning, artifact storage, deployment tracking,
    lineage management, and automated governance for production ML systems.
    """
    
    def __init__(self, storage_path: str = "model_registry_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage directories
        self.artifacts_path = self.storage_path / "artifacts"
        self.artifacts_path.mkdir(exist_ok=True)
        
        self.metadata_db_path = self.storage_path / "model_registry.db"
        
        # Database connection
        self.db_conn = None
        
        # In-memory caches
        self.models: Dict[str, ModelMetadata] = {}
        self.deployments: Dict[str, List[DeploymentRecord]] = defaultdict(list)
        self.comparisons: Dict[str, ModelComparison] = {}
        
        # Configuration
        self.max_versions_per_model = 10
        self.artifact_compression = True
        self.checksum_algorithm = "sha256"
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'models_registered': 0,
            'models_deployed': 0,
            'artifacts_stored': 0,
            'comparisons_created': 0,
            'last_cleanup': datetime.utcnow()
        }
        
        logger.info("ML Model Registry initialized", storage_path=str(self.storage_path))
    
    async def initialize(self):
        """Initialize model registry database and load existing data"""
        logger.info("Initializing ML Model Registry")
        
        try:
            # Initialize database
            await self._initialize_database()
            
            # Load existing models
            await self._load_models()
            
            # Load deployments
            await self._load_deployments()
            
            # Load comparisons
            await self._load_comparisons()
            
            logger.info("ML Model Registry initialization completed",
                       models=len(self.models),
                       deployments=sum(len(deps) for deps in self.deployments.values()),
                       comparisons=len(self.comparisons))
            
        except Exception as e:
            logger.error("ML Model Registry initialization failed", error=str(e))
            raise
    
    async def register_model(self, 
                           model_name: str,
                           model_type: ModelType,
                           model_object: Any,
                           metadata: Dict[str, Any],
                           training_info: Dict[str, Any],
                           performance_metrics: Dict[str, float]) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            model_type: Type of ML model
            model_object: Trained model object
            metadata: Additional model metadata
            training_info: Training information
            performance_metrics: Model performance metrics
            
        Returns:
            Model ID of registered model
        """
        logger.info("Registering model", model_name=model_name, model_type=model_type.value)
        
        try:
            # Generate model ID and version
            model_id = self._generate_model_id(model_name, model_type)
            version = self._generate_version(model_name)
            
            # Store model artifact
            artifact_path, artifact_size, checksum = await self._store_model_artifact(
                model_id, model_object
            )
            
            # Create model metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version=version,
                description=metadata.get('description', ''),
                framework=metadata.get('framework', 'unknown'),
                algorithm=metadata.get('algorithm', 'unknown'),
                hyperparameters=metadata.get('hyperparameters', {}),
                training_dataset_hash=training_info.get('dataset_hash', ''),
                training_start_time=training_info.get('start_time', datetime.utcnow()),
                training_end_time=training_info.get('end_time'),
                training_duration_seconds=training_info.get('duration_seconds'),
                model_artifact_path=str(artifact_path),
                model_size_bytes=artifact_size,
                artifact_checksum=checksum,
                input_features=metadata.get('input_features', []),
                output_schema=metadata.get('output_schema', {}),
                feature_importance=metadata.get('feature_importance', {}),
                validation_metrics=performance_metrics.get('validation', {}),
                test_metrics=performance_metrics.get('test', {}),
                owner=metadata.get('owner', 'system'),
                tags=metadata.get('tags', []),
                created_by=metadata.get('created_by', 'system')
            )
            
            # Store in database
            await self._store_model_metadata(model_metadata)
            
            # Update in-memory cache
            with self.lock:
                self.models[model_id] = model_metadata
                self.stats['models_registered'] += 1
                self.stats['artifacts_stored'] += 1
            
            # Clean up old versions if needed
            await self._cleanup_old_versions(model_name)
            
            logger.info("Model registered successfully", 
                       model_id=model_id, 
                       version=version,
                       artifact_size=artifact_size)
            
            return model_id
            
        except Exception as e:
            logger.error("Model registration failed", 
                        error=str(e), 
                        model_name=model_name)
            raise
    
    async def get_model(self, model_id: str) -> Optional[Tuple[ModelMetadata, Any]]:
        """
        Get model metadata and artifact.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (metadata, model_object) or None if not found
        """
        logger.debug("Getting model", model_id=model_id)
        
        try:
            if model_id not in self.models:
                logger.warning("Model not found", model_id=model_id)
                return None
            
            metadata = self.models[model_id]
            
            # Load model artifact
            model_object = await self._load_model_artifact(metadata.model_artifact_path)
            
            # Update access statistics
            await self._update_model_access_stats(model_id)
            
            logger.debug("Model retrieved successfully", model_id=model_id)
            return metadata, model_object
            
        except Exception as e:
            logger.error("Model retrieval failed", error=str(e), model_id=model_id)
            return None
    
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata only (without loading artifact).
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        if model_id in self.models:
            return self.models[model_id]
        return None
    
    async def list_models(self, 
                        model_type: Optional[ModelType] = None,
                        status: Optional[ModelStatus] = None,
                        owner: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List models with optional filters.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter
            owner: Optional owner filter
            tags: Optional tag filters
            
        Returns:
            List of matching model metadata
        """
        logger.info("Listing models", 
                   model_type=model_type,
                   status=status,
                   owner=owner,
                   tags=tags)
        
        try:
            matching_models = []
            
            for model_metadata in self.models.values():
                # Apply filters
                if model_type and model_metadata.model_type != model_type:
                    continue
                
                if status and model_metadata.status != status:
                    continue
                
                if owner and model_metadata.owner != owner:
                    continue
                
                if tags and not any(tag in model_metadata.tags for tag in tags):
                    continue
                
                matching_models.append(model_metadata)
            
            # Sort by creation time (newest first)
            matching_models.sort(key=lambda m: m.created_at, reverse=True)
            
            logger.info("Models listed", count=len(matching_models))
            return matching_models
            
        except Exception as e:
            logger.error("Model listing failed", error=str(e))
            return []
    
    async def deploy_model(self, 
                         model_id: str,
                         deployment_stage: DeploymentStage,
                         deployment_config: Dict[str, Any]) -> str:
        """
        Deploy model to specified stage.
        
        Args:
            model_id: Model to deploy
            deployment_stage: Target deployment stage
            deployment_config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        logger.info("Deploying model", 
                   model_id=model_id, 
                   stage=deployment_stage.value)
        
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model_metadata = self.models[model_id]
            
            # Validate model is ready for deployment
            if not await self._validate_model_for_deployment(model_metadata, deployment_stage):
                raise ValueError(f"Model not ready for deployment: {model_id}")
            
            # Generate deployment ID
            deployment_id = f"deploy_{model_id}_{deployment_stage.value}_{uuid.uuid4().hex[:8]}"
            
            # Create deployment record
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_id=model_id,
                deployment_stage=deployment_stage,
                deployment_config=deployment_config,
                deployed_at=datetime.utcnow(),
                deployed_by=deployment_config.get('deployed_by', 'system')
            )
            
            # Store deployment record
            await self._store_deployment_record(deployment_record)
            
            # Update model status
            model_metadata.status = ModelStatus.DEPLOYED
            model_metadata.deployment_stage = deployment_stage
            model_metadata.deployment_config = deployment_config
            model_metadata.updated_at = datetime.utcnow()
            
            # Update in database
            await self._update_model_metadata(model_metadata)
            
            # Update in-memory cache
            with self.lock:
                self.deployments[model_id].append(deployment_record)
                self.stats['models_deployed'] += 1
            
            logger.info("Model deployed successfully", 
                       model_id=model_id, 
                       deployment_id=deployment_id,
                       stage=deployment_stage.value)
            
            return deployment_id
            
        except Exception as e:
            logger.error("Model deployment failed", 
                        error=str(e), 
                        model_id=model_id)
            raise
    
    async def undeploy_model(self, deployment_id: str) -> bool:
        """
        Undeploy a model deployment.
        
        Args:
            deployment_id: Deployment to undeploy
            
        Returns:
            True if successful
        """
        logger.info("Undeploying model", deployment_id=deployment_id)
        
        try:
            # Find deployment record
            deployment_record = None
            model_id = None
            
            for mid, deployments in self.deployments.items():
                for deployment in deployments:
                    if deployment.deployment_id == deployment_id:
                        deployment_record = deployment
                        model_id = mid
                        break
                if deployment_record:
                    break
            
            if not deployment_record:
                logger.error("Deployment not found", deployment_id=deployment_id)
                return False
            
            # Update deployment record
            deployment_record.is_active = False
            deployment_record.undeployed_at = datetime.utcnow()
            
            # Update in database
            await self._update_deployment_record(deployment_record)
            
            # Update model status if no active deployments
            active_deployments = [
                d for d in self.deployments[model_id] 
                if d.is_active
            ]
            
            if not active_deployments:
                model_metadata = self.models[model_id]
                model_metadata.status = ModelStatus.READY
                model_metadata.deployment_stage = None
                model_metadata.updated_at = datetime.utcnow()
                
                await self._update_model_metadata(model_metadata)
            
            logger.info("Model undeployed successfully", deployment_id=deployment_id)
            return True
            
        except Exception as e:
            logger.error("Model undeployment failed", 
                        error=str(e), 
                        deployment_id=deployment_id)
            return False
    
    async def create_model_comparison(self, 
                                    model_a_id: str,
                                    model_b_id: str,
                                    comparison_type: str,
                                    test_config: Dict[str, Any]) -> str:
        """
        Create a comparison between two models.
        
        Args:
            model_a_id: First model to compare
            model_b_id: Second model to compare
            comparison_type: Type of comparison
            test_config: Test configuration
            
        Returns:
            Comparison ID
        """
        logger.info("Creating model comparison", 
                   model_a=model_a_id, 
                   model_b=model_b_id,
                   comparison_type=comparison_type)
        
        try:
            # Validate models exist
            if model_a_id not in self.models or model_b_id not in self.models:
                raise ValueError("One or both models not found")
            
            # Generate comparison ID
            comparison_id = f"comp_{model_a_id[:8]}_{model_b_id[:8]}_{uuid.uuid4().hex[:8]}"
            
            # Create comparison record
            comparison = ModelComparison(
                comparison_id=comparison_id,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                comparison_type=comparison_type,
                metrics_comparison={},
                statistical_significance={},
                test_start_time=datetime.utcnow(),
                sample_size=test_config.get('sample_size', 1000),
                traffic_split=test_config.get('traffic_split', {'model_a': 0.5, 'model_b': 0.5}),
                confidence_level=test_config.get('confidence_level', 0.95),
                created_by=test_config.get('created_by', 'system')
            )
            
            # Store comparison
            await self._store_model_comparison(comparison)
            
            # Update in-memory cache
            with self.lock:
                self.comparisons[comparison_id] = comparison
                self.stats['comparisons_created'] += 1
            
            logger.info("Model comparison created", comparison_id=comparison_id)
            return comparison_id
            
        except Exception as e:
            logger.error("Model comparison creation failed", error=str(e))
            raise
    
    async def update_comparison_results(self, 
                                      comparison_id: str,
                                      metrics_comparison: Dict[str, Dict[str, float]],
                                      winner: Optional[str] = None) -> bool:
        """
        Update comparison results.
        
        Args:
            comparison_id: Comparison to update
            metrics_comparison: Comparison metrics
            winner: Optional winner model ID
            
        Returns:
            True if successful
        """
        logger.info("Updating comparison results", comparison_id=comparison_id)
        
        try:
            if comparison_id not in self.comparisons:
                logger.error("Comparison not found", comparison_id=comparison_id)
                return False
            
            comparison = self.comparisons[comparison_id]
            
            # Update comparison results
            comparison.metrics_comparison = metrics_comparison
            comparison.winner = winner
            comparison.test_end_time = datetime.utcnow()
            
            # Calculate statistical significance (simplified)
            comparison.statistical_significance = self._calculate_statistical_significance(
                metrics_comparison, comparison.confidence_level
            )
            
            # Generate results summary
            comparison.results_summary = self._generate_comparison_summary(comparison)
            comparison.recommendation = self._generate_comparison_recommendation(comparison)
            
            # Update in database
            await self._update_model_comparison(comparison)
            
            logger.info("Comparison results updated", comparison_id=comparison_id)
            return True
            
        except Exception as e:
            logger.error("Comparison results update failed", 
                        error=str(e), 
                        comparison_id=comparison_id)
            return False
    
    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get model lineage information.
        
        Args:
            model_id: Model to get lineage for
            
        Returns:
            Dictionary with lineage information
        """
        logger.info("Getting model lineage", model_id=model_id)
        
        try:
            if model_id not in self.models:
                return {}
            
            model_metadata = self.models[model_id]
            
            lineage = {
                'model_id': model_id,
                'model_name': model_metadata.model_name,
                'version': model_metadata.version,
                'parent_model': model_metadata.parent_model_id,
                'derived_models': model_metadata.derived_models,
                'upstream_models': model_metadata.upstream_models,
                'downstream_models': model_metadata.downstream_models,
                'deployments': [],
                'comparisons': []
            }
            
            # Add deployment history
            for deployment in self.deployments.get(model_id, []):
                lineage['deployments'].append({
                    'deployment_id': deployment.deployment_id,
                    'stage': deployment.deployment_stage.value,
                    'deployed_at': deployment.deployed_at.isoformat(),
                    'is_active': deployment.is_active
                })
            
            # Add comparisons involving this model
            for comparison in self.comparisons.values():
                if model_id in [comparison.model_a_id, comparison.model_b_id]:
                    lineage['comparisons'].append({
                        'comparison_id': comparison.comparison_id,
                        'comparison_type': comparison.comparison_type,
                        'other_model': (comparison.model_b_id if comparison.model_a_id == model_id 
                                      else comparison.model_a_id),
                        'winner': comparison.winner,
                        'created_at': comparison.created_at.isoformat()
                    })
            
            logger.info("Model lineage retrieved", 
                       model_id=model_id,
                       deployments=len(lineage['deployments']),
                       comparisons=len(lineage['comparisons']))
            
            return lineage
            
        except Exception as e:
            logger.error("Model lineage retrieval failed", 
                        error=str(e), 
                        model_id=model_id)
            return {}
    
    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry metrics and statistics"""
        with self.lock:
            # Model status distribution
            status_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            for model in self.models.values():
                status_counts[model.status.value] += 1
                type_counts[model.model_type.value] += 1
            
            # Deployment statistics
            active_deployments = 0
            deployment_stage_counts = defaultdict(int)
            
            for deployments in self.deployments.values():
                for deployment in deployments:
                    if deployment.is_active:
                        active_deployments += 1
                        deployment_stage_counts[deployment.deployment_stage.value] += 1
            
            return {
                'total_models': len(self.models),
                'model_status_distribution': dict(status_counts),
                'model_type_distribution': dict(type_counts),
                'active_deployments': active_deployments,
                'deployment_stage_distribution': dict(deployment_stage_counts),
                'total_comparisons': len(self.comparisons),
                'statistics': self.stats.copy(),
                'storage_info': {
                    'artifacts_path': str(self.artifacts_path),
                    'total_artifact_size': self._calculate_total_artifact_size(),
                    'database_size': self._get_database_size()
                }
            }
    
    async def _initialize_database(self):
        """Initialize SQLite database for model registry"""
        self.db_conn = sqlite3.connect(str(self.metadata_db_path), check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=WAL")
        
        # Models table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                framework TEXT,
                algorithm TEXT,
                hyperparameters TEXT,
                training_dataset_hash TEXT,
                training_start_time TEXT,
                training_end_time TEXT,
                training_duration_seconds REAL,
                model_artifact_path TEXT,
                model_size_bytes INTEGER,
                artifact_checksum TEXT,
                input_features TEXT,
                output_schema TEXT,
                feature_importance TEXT,
                validation_metrics TEXT,
                test_metrics TEXT,
                production_metrics TEXT,
                status TEXT,
                deployment_stage TEXT,
                deployment_config TEXT,
                parent_model_id TEXT,
                derived_models TEXT,
                upstream_models TEXT,
                downstream_models TEXT,
                owner TEXT,
                tags TEXT,
                created_by TEXT,
                created_at TEXT,
                updated_at TEXT,
                prediction_count INTEGER DEFAULT 0,
                last_prediction_time TEXT,
                error_count INTEGER DEFAULT 0,
                approval_status TEXT,
                approved_by TEXT,
                approval_date TEXT,
                compliance_checks TEXT
            )
        """)
        
        # Deployments table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                deployment_stage TEXT NOT NULL,
                deployment_config TEXT,
                deployed_at TEXT NOT NULL,
                undeployed_at TEXT,
                requests_served INTEGER DEFAULT 0,
                avg_response_time_ms REAL DEFAULT 0,
                error_rate REAL DEFAULT 0,
                cpu_usage_percent REAL DEFAULT 0,
                memory_usage_mb REAL DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                health_status TEXT DEFAULT 'healthy',
                deployed_by TEXT,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        # Comparisons table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                model_a_id TEXT NOT NULL,
                model_b_id TEXT NOT NULL,
                comparison_type TEXT NOT NULL,
                metrics_comparison TEXT,
                statistical_significance TEXT,
                winner TEXT,
                confidence_level REAL,
                test_start_time TEXT,
                test_end_time TEXT,
                sample_size INTEGER,
                traffic_split TEXT,
                results_summary TEXT,
                recommendation TEXT,
                created_by TEXT,
                created_at TEXT,
                FOREIGN KEY (model_a_id) REFERENCES models (model_id),
                FOREIGN KEY (model_b_id) REFERENCES models (model_id)
            )
        """)
        
        # Create indexes
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_id)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_deployments_active ON deployments(is_active)")
        
        self.db_conn.commit()
    
    async def _store_model_artifact(self, model_id: str, model_object: Any) -> Tuple[Path, int, str]:
        """Store model artifact and return path, size, and checksum"""
        artifact_path = self.artifacts_path / f"{model_id}.pkl"
        
        # Serialize model
        with open(artifact_path, 'wb') as f:
            pickle.dump(model_object, f)
        
        # Get file size
        artifact_size = artifact_path.stat().st_size
        
        # Calculate checksum
        checksum = self._calculate_file_checksum(artifact_path)
        
        # Compress if enabled
        if self.artifact_compression:
            compressed_path = self.artifacts_path / f"{model_id}.tar.gz"
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(artifact_path, arcname=f"{model_id}.pkl")
            
            # Remove uncompressed file
            artifact_path.unlink()
            artifact_path = compressed_path
            artifact_size = artifact_path.stat().st_size
            checksum = self._calculate_file_checksum(artifact_path)
        
        return artifact_path, artifact_size, checksum
    
    async def _load_model_artifact(self, artifact_path: str) -> Any:
        """Load model artifact from path"""
        path = Path(artifact_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        
        # Handle compressed artifacts
        if path.suffix == '.gz':
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(path, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                # Find the pickle file
                pkl_files = list(Path(temp_dir).glob('*.pkl'))
                if not pkl_files:
                    raise ValueError(f"No pickle file found in compressed artifact: {artifact_path}")
                
                with open(pkl_files[0], 'rb') as f:
                    return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def _generate_model_id(self, model_name: str, model_type: ModelType) -> str:
        """Generate unique model ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"{model_name}_{model_type.value}_{timestamp}_{unique_id}"
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version number for model"""
        existing_versions = []
        
        for model in self.models.values():
            if model.model_name == model_name:
                existing_versions.append(model.version)
        
        if not existing_versions:
            return "1.0.0"
        
        # Simple version increment
        latest_version = max(existing_versions)
        major, minor, patch = map(int, latest_version.split('.'))
        return f"{major}.{minor}.{patch + 1}"
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(self.checksum_algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    async def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        cursor = self.db_conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO models VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            metadata.model_id, metadata.model_name, metadata.model_type.value, metadata.version,
            metadata.description, metadata.framework, metadata.algorithm,
            json.dumps(metadata.hyperparameters), metadata.training_dataset_hash,
            metadata.training_start_time.isoformat(),
            metadata.training_end_time.isoformat() if metadata.training_end_time else None,
            metadata.training_duration_seconds, metadata.model_artifact_path,
            metadata.model_size_bytes, metadata.artifact_checksum,
            json.dumps(metadata.input_features), json.dumps(metadata.output_schema),
            json.dumps(metadata.feature_importance), json.dumps(metadata.validation_metrics),
            json.dumps(metadata.test_metrics), json.dumps(metadata.production_metrics),
            metadata.status.value,
            metadata.deployment_stage.value if metadata.deployment_stage else None,
            json.dumps(metadata.deployment_config), metadata.parent_model_id,
            json.dumps(metadata.derived_models), json.dumps(metadata.upstream_models),
            json.dumps(metadata.downstream_models), metadata.owner, json.dumps(metadata.tags),
            metadata.created_by, metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
            metadata.prediction_count,
            metadata.last_prediction_time.isoformat() if metadata.last_prediction_time else None,
            metadata.error_count, metadata.approval_status, metadata.approved_by,
            metadata.approval_date.isoformat() if metadata.approval_date else None,
            json.dumps(metadata.compliance_checks)
        ))
        
        self.db_conn.commit()
    
    async def _load_models(self):
        """Load models from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM models")
        
        for row in cursor.fetchall():
            metadata = self._row_to_model_metadata(row)
            self.models[metadata.model_id] = metadata
    
    async def _load_deployments(self):
        """Load deployments from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM deployments")
        
        for row in cursor.fetchall():
            deployment = self._row_to_deployment_record(row)
            self.deployments[deployment.model_id].append(deployment)
    
    async def _load_comparisons(self):
        """Load comparisons from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM comparisons")
        
        for row in cursor.fetchall():
            comparison = self._row_to_model_comparison(row)
            self.comparisons[comparison.comparison_id] = comparison
    
    def _row_to_model_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            model_id=row[0], model_name=row[1], model_type=ModelType(row[2]), version=row[3],
            description=row[4], framework=row[5], algorithm=row[6],
            hyperparameters=json.loads(row[7]) if row[7] else {},
            training_dataset_hash=row[8],
            training_start_time=datetime.fromisoformat(row[9]),
            training_end_time=datetime.fromisoformat(row[10]) if row[10] else None,
            training_duration_seconds=row[11], model_artifact_path=row[12],
            model_size_bytes=row[13], artifact_checksum=row[14],
            input_features=json.loads(row[15]) if row[15] else [],
            output_schema=json.loads(row[16]) if row[16] else {},
            feature_importance=json.loads(row[17]) if row[17] else {},
            validation_metrics=json.loads(row[18]) if row[18] else {},
            test_metrics=json.loads(row[19]) if row[19] else {},
            production_metrics=json.loads(row[20]) if row[20] else {},
            status=ModelStatus(row[21]),
            deployment_stage=DeploymentStage(row[22]) if row[22] else None,
            deployment_config=json.loads(row[23]) if row[23] else {},
            parent_model_id=row[24],
            derived_models=json.loads(row[25]) if row[25] else [],
            upstream_models=json.loads(row[26]) if row[26] else [],
            downstream_models=json.loads(row[27]) if row[27] else [],
            owner=row[28], tags=json.loads(row[29]) if row[29] else [],
            created_by=row[30],
            created_at=datetime.fromisoformat(row[31]),
            updated_at=datetime.fromisoformat(row[32]),
            prediction_count=row[33],
            last_prediction_time=datetime.fromisoformat(row[34]) if row[34] else None,
            error_count=row[35], approval_status=row[36], approved_by=row[37],
            approval_date=datetime.fromisoformat(row[38]) if row[38] else None,
            compliance_checks=json.loads(row[39]) if row[39] else {}
        )
    
    def _row_to_deployment_record(self, row) -> DeploymentRecord:
        """Convert database row to DeploymentRecord"""
        return DeploymentRecord(
            deployment_id=row[0], model_id=row[1],
            deployment_stage=DeploymentStage(row[2]),
            deployment_config=json.loads(row[3]) if row[3] else {},
            deployed_at=datetime.fromisoformat(row[4]),
            undeployed_at=datetime.fromisoformat(row[5]) if row[5] else None,
            requests_served=row[6], avg_response_time_ms=row[7], error_rate=row[8],
            cpu_usage_percent=row[9], memory_usage_mb=row[10],
            is_active=bool(row[11]), health_status=row[12],
            deployed_by=row[13], notes=row[14]
        )
    
    def _row_to_model_comparison(self, row) -> ModelComparison:
        """Convert database row to ModelComparison"""
        return ModelComparison(
            comparison_id=row[0], model_a_id=row[1], model_b_id=row[2],
            comparison_type=row[3],
            metrics_comparison=json.loads(row[4]) if row[4] else {},
            statistical_significance=json.loads(row[5]) if row[5] else {},
            winner=row[6], confidence_level=row[7],
            test_start_time=datetime.fromisoformat(row[8]),
            test_end_time=datetime.fromisoformat(row[9]) if row[9] else None,
            sample_size=row[10],
            traffic_split=json.loads(row[11]) if row[11] else {},
            results_summary=row[12], recommendation=row[13],
            created_by=row[14], created_at=datetime.fromisoformat(row[15])
        )
    
    async def _validate_model_for_deployment(self, 
                                           metadata: ModelMetadata, 
                                           stage: DeploymentStage) -> bool:
        """Validate model is ready for deployment"""
        # Check model status
        if metadata.status not in [ModelStatus.READY, ModelStatus.VALIDATING]:
            return False
        
        # Check required metrics exist
        if stage == DeploymentStage.PRODUCTION:
            required_metrics = ['accuracy', 'precision', 'recall']
            for metric in required_metrics:
                if metric not in metadata.validation_metrics:
                    return False
                if metadata.validation_metrics[metric] < 0.8:  # Minimum threshold
                    return False
        
        # Check artifact exists
        if not Path(metadata.model_artifact_path).exists():
            return False
        
        return True
    
    async def _store_deployment_record(self, deployment: DeploymentRecord):
        """Store deployment record in database"""
        cursor = self.db_conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO deployments VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            deployment.deployment_id, deployment.model_id,
            deployment.deployment_stage.value,
            json.dumps(deployment.deployment_config),
            deployment.deployed_at.isoformat(),
            deployment.undeployed_at.isoformat() if deployment.undeployed_at else None,
            deployment.requests_served, deployment.avg_response_time_ms,
            deployment.error_rate, deployment.cpu_usage_percent,
            deployment.memory_usage_mb, deployment.is_active,
            deployment.health_status, deployment.deployed_by, deployment.notes
        ))
        
        self.db_conn.commit()
    
    async def _store_model_comparison(self, comparison: ModelComparison):
        """Store model comparison in database"""
        cursor = self.db_conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO comparisons VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            comparison.comparison_id, comparison.model_a_id, comparison.model_b_id,
            comparison.comparison_type,
            json.dumps(comparison.metrics_comparison),
            json.dumps(comparison.statistical_significance),
            comparison.winner, comparison.confidence_level,
            comparison.test_start_time.isoformat(),
            comparison.test_end_time.isoformat() if comparison.test_end_time else None,
            comparison.sample_size, json.dumps(comparison.traffic_split),
            comparison.results_summary, comparison.recommendation,
            comparison.created_by, comparison.created_at.isoformat()
        ))
        
        self.db_conn.commit()
    
    async def _update_model_metadata(self, metadata: ModelMetadata):
        """Update model metadata in database"""
        await self._store_model_metadata(metadata)
    
    async def _update_deployment_record(self, deployment: DeploymentRecord):
        """Update deployment record in database"""
        await self._store_deployment_record(deployment)
    
    async def _update_model_comparison(self, comparison: ModelComparison):
        """Update model comparison in database"""
        await self._store_model_comparison(comparison)
    
    async def _update_model_access_stats(self, model_id: str):
        """Update model access statistics"""
        if model_id in self.models:
            metadata = self.models[model_id]
            metadata.last_prediction_time = datetime.utcnow()
            metadata.prediction_count += 1
            
            # Update in database (simplified - in production would batch these updates)
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE models 
                SET prediction_count = ?, last_prediction_time = ?
                WHERE model_id = ?
            """, (metadata.prediction_count, 
                  metadata.last_prediction_time.isoformat(), 
                  model_id))
            
            self.db_conn.commit()
    
    async def _cleanup_old_versions(self, model_name: str):
        """Clean up old model versions"""
        same_name_models = [
            (model_id, model) for model_id, model in self.models.items()
            if model.model_name == model_name
        ]
        
        if len(same_name_models) > self.max_versions_per_model:
            # Sort by creation time and remove oldest
            same_name_models.sort(key=lambda x: x[1].created_at)
            
            models_to_remove = same_name_models[:-self.max_versions_per_model]
            
            for model_id, model in models_to_remove:
                if model.status not in [ModelStatus.DEPLOYED, ModelStatus.PRODUCTION]:
                    # Remove artifact
                    artifact_path = Path(model.model_artifact_path)
                    if artifact_path.exists():
                        artifact_path.unlink()
                    
                    # Remove from database
                    cursor = self.db_conn.cursor()
                    cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                    self.db_conn.commit()
                    
                    # Remove from memory
                    del self.models[model_id]
                    
                    logger.info("Cleaned up old model version", model_id=model_id)
    
    def _calculate_statistical_significance(self, 
                                          metrics_comparison: Dict[str, Dict[str, float]], 
                                          confidence_level: float) -> Dict[str, bool]:
        """Calculate statistical significance (simplified)"""
        # In production, this would use proper statistical tests
        significance = {}
        
        for metric_name, values in metrics_comparison.items():
            if 'model_a' in values and 'model_b' in values:
                # Simplified significance test
                difference = abs(values['model_a'] - values['model_b'])
                avg_value = (values['model_a'] + values['model_b']) / 2
                
                # Consider significant if difference > 5% of average
                significance[metric_name] = difference > (avg_value * 0.05)
        
        return significance
    
    def _generate_comparison_summary(self, comparison: ModelComparison) -> str:
        """Generate comparison results summary"""
        if not comparison.metrics_comparison:
            return "No metrics available for comparison"
        
        summary_parts = []
        
        for metric_name, values in comparison.metrics_comparison.items():
            if 'model_a' in values and 'model_b' in values:
                model_a_val = values['model_a']
                model_b_val = values['model_b']
                
                if model_a_val > model_b_val:
                    better_model = "Model A"
                    improvement = ((model_a_val - model_b_val) / model_b_val) * 100
                else:
                    better_model = "Model B"
                    improvement = ((model_b_val - model_a_val) / model_a_val) * 100
                
                is_significant = comparison.statistical_significance.get(metric_name, False)
                significance_text = "statistically significant" if is_significant else "not significant"
                
                summary_parts.append(
                    f"{metric_name}: {better_model} performs {improvement:.1f}% better ({significance_text})"
                )
        
        return "; ".join(summary_parts)
    
    def _generate_comparison_recommendation(self, comparison: ModelComparison) -> str:
        """Generate comparison recommendation"""
        if not comparison.winner:
            return "No clear winner identified. Consider additional testing."
        
        winner_name = "Model A" if comparison.winner == comparison.model_a_id else "Model B"
        
        # Count significant improvements
        significant_improvements = sum(
            1 for is_sig in comparison.statistical_significance.values() if is_sig
        )
        
        if significant_improvements >= 2:
            return f"Strong recommendation: Deploy {winner_name} based on significant improvements in multiple metrics."
        elif significant_improvements == 1:
            return f"Moderate recommendation: Consider deploying {winner_name} with additional monitoring."
        else:
            return f"Weak recommendation: {winner_name} shows marginal improvements. Consider longer testing period."
    
    def _calculate_total_artifact_size(self) -> int:
        """Calculate total size of all artifacts"""
        total_size = 0
        
        for artifact_file in self.artifacts_path.glob("*"):
            if artifact_file.is_file():
                total_size += artifact_file.stat().st_size
        
        return total_size
    
    def _get_database_size(self) -> int:
        """Get database file size"""
        try:
            return self.metadata_db_path.stat().st_size
        except Exception:
            return 0
    
    async def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
        
        logger.info("ML Model Registry closed")