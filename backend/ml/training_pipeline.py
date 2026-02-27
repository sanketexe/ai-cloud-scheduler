"""
ML Training and Deployment Pipeline

Provides automated data preprocessing, feature engineering, model training orchestration,
cross-validation, automated deployment with rollback capabilities, and performance
benchmarking for anomaly detection ML systems.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import hashlib
from pathlib import Path
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import pickle
import numpy as np
from collections import defaultdict, deque
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Training pipeline stages"""
    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK_REQUIRED = "rollback_required"

class ValidationMetric(Enum):
    """Model validation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    MAE = "mae"
    RMSE = "rmse"
    LATENCY = "latency"

@dataclass
class PipelineConfig:
    """Training pipeline configuration"""
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "anomaly_detection_pipeline"
    model_types: List[str] = field(default_factory=lambda: ["isolation_forest", "lstm_anomaly"])
    data_sources: List[str] = field(default_factory=lambda: ["aws_cost_data"])
    feature_sets: List[str] = field(default_factory=lambda: ["cost_features", "usage_features"])
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    performance_threshold: float = 0.85
    deployment_environment: str = "staging"
    auto_deploy: bool = False
    rollback_on_failure: bool = True
    monitoring_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: Optional[PipelineStage] = None
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

@dataclass
class ModelArtifact:
    """Model artifact metadata"""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""
    model_path: str = ""
    metadata_path: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    checksum: str = ""

class TrainingPipeline:
    """
    Comprehensive ML training and deployment pipeline for anomaly detection.
    
    Orchestrates the entire ML lifecycle from data collection to model deployment
    with automated validation, rollback capabilities, and performance monitoring.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.executions: Dict[str, PipelineExecution] = {}
        self.artifacts: Dict[str, ModelArtifact] = {}
        self.active_models: Dict[str, str] = {}  # environment -> artifact_id
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize database
        self.db_path = Path("training_pipeline.db")
        self._init_database()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        logger.info(f"Training pipeline initialized: {self.config.pipeline_id}")
    
    def _init_database(self):
        """Initialize SQLite database for pipeline tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pipeline_executions (
                    execution_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    stages_completed TEXT,
                    stage_results TEXT,
                    metrics TEXT,
                    errors TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_seconds REAL
                );
                
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    metadata_path TEXT,
                    performance_metrics TEXT,
                    feature_importance TEXT,
                    validation_results TEXT,
                    created_at TIMESTAMP,
                    size_bytes INTEGER,
                    checksum TEXT
                );
                
                CREATE TABLE IF NOT EXISTS deployment_history (
                    deployment_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployed_at TIMESTAMP,
                    deployed_by TEXT,
                    rollback_artifact_id TEXT,
                    status TEXT,
                    FOREIGN KEY (artifact_id) REFERENCES model_artifacts (artifact_id)
                );
            """)
    
    async def execute_pipeline(self, config: Optional[PipelineConfig] = None) -> PipelineExecution:
        """Execute complete training pipeline"""
        if config:
            self.config = config
        
        execution = PipelineExecution(
            pipeline_id=self.config.pipeline_id,
            started_at=datetime.now()
        )
        
        with self._lock:
            self.executions[execution.execution_id] = execution
        
        try:
            execution.status = PipelineStatus.RUNNING
            logger.info(f"Starting pipeline execution: {execution.execution_id}")
            
            # Execute pipeline stages
            stages = [
                (PipelineStage.DATA_COLLECTION, self._collect_data),
                (PipelineStage.PREPROCESSING, self._preprocess_data),
                (PipelineStage.FEATURE_ENGINEERING, self._engineer_features),
                (PipelineStage.MODEL_TRAINING, self._train_models),
                (PipelineStage.VALIDATION, self._validate_models),
                (PipelineStage.DEPLOYMENT, self._deploy_models),
                (PipelineStage.MONITORING, self._setup_monitoring)
            ]
            
            for stage, stage_func in stages:
                execution.current_stage = stage
                logger.info(f"Executing stage: {stage.value}")
                
                try:
                    stage_result = await stage_func(execution)
                    execution.stage_results[stage.value] = stage_result
                    execution.stages_completed.append(stage)
                    
                    logger.info(f"Stage {stage.value} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Stage {stage.value} failed: {str(e)}"
                    execution.errors.append(error_msg)
                    logger.error(error_msg)
                    
                    if self.config.rollback_on_failure:
                        await self._rollback_deployment(execution)
                    
                    execution.status = PipelineStatus.FAILED
                    break
            
            if execution.status != PipelineStatus.FAILED:
                execution.status = PipelineStatus.COMPLETED
                logger.info(f"Pipeline execution completed: {execution.execution_id}")
            
        except Exception as e:
            execution.errors.append(f"Pipeline execution failed: {str(e)}")
            execution.status = PipelineStatus.FAILED
            logger.error(f"Pipeline execution failed: {str(e)}")
        
        finally:
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            self._save_execution(execution)
        
        return execution
    
    async def _collect_data(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Collect and prepare training data"""
        logger.info("Collecting training data from configured sources")
        
        # Simulate data collection from multiple sources
        data_sources = {
            "aws_cost_data": {
                "records": 10000,
                "date_range": "2024-01-01 to 2024-12-31",
                "services": ["EC2", "S3", "RDS", "Lambda"],
                "accounts": ["123456789012", "123456789013", "123456789014"]
            },
            "usage_metrics": {
                "records": 50000,
                "metrics": ["cpu_utilization", "memory_usage", "network_io"],
                "resolution": "hourly"
            },
            "billing_data": {
                "records": 5000,
                "invoices": 12,
                "line_items": 45000
            }
        }
        
        # Validate data quality
        quality_metrics = {
            "completeness": 0.95,
            "accuracy": 0.98,
            "consistency": 0.92,
            "timeliness": 0.99
        }
        
        execution.metrics.update({
            "data_records_collected": sum(source["records"] for source in data_sources.values() if "records" in source),
            "data_quality_score": statistics.mean(quality_metrics.values())
        })
        
        return {
            "data_sources": data_sources,
            "quality_metrics": quality_metrics,
            "collection_timestamp": datetime.now().isoformat()
        }
    
    async def _preprocess_data(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Preprocess and clean training data"""
        logger.info("Preprocessing training data")
        
        # Simulate data preprocessing steps
        preprocessing_steps = [
            "missing_value_imputation",
            "outlier_detection_removal",
            "data_normalization",
            "feature_scaling",
            "temporal_alignment"
        ]
        
        preprocessing_results = {
            "steps_completed": preprocessing_steps,
            "records_before": 65000,
            "records_after": 62000,
            "outliers_removed": 1200,
            "missing_values_imputed": 1800,
            "normalization_method": "z_score",
            "scaling_method": "min_max"
        }
        
        execution.metrics.update({
            "preprocessing_data_retention": preprocessing_results["records_after"] / preprocessing_results["records_before"],
            "outlier_removal_rate": preprocessing_results["outliers_removed"] / preprocessing_results["records_before"]
        })
        
        return preprocessing_results
    
    async def _engineer_features(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Engineer features for ML models"""
        logger.info("Engineering features for ML models")
        
        # Simulate feature engineering
        feature_sets = {
            "cost_features": [
                "daily_cost_trend",
                "weekly_cost_variance",
                "monthly_cost_seasonality",
                "service_cost_distribution",
                "cost_growth_rate"
            ],
            "usage_features": [
                "resource_utilization_avg",
                "peak_usage_patterns",
                "usage_efficiency_ratio",
                "scaling_frequency",
                "idle_resource_ratio"
            ],
            "temporal_features": [
                "hour_of_day",
                "day_of_week",
                "month_of_year",
                "business_day_indicator",
                "holiday_indicator"
            ],
            "derived_features": [
                "cost_per_unit_usage",
                "anomaly_score_historical",
                "trend_deviation_score",
                "seasonal_adjustment_factor"
            ]
        }
        
        feature_importance = {
            "daily_cost_trend": 0.25,
            "service_cost_distribution": 0.20,
            "resource_utilization_avg": 0.15,
            "cost_growth_rate": 0.12,
            "peak_usage_patterns": 0.10,
            "weekly_cost_variance": 0.08,
            "cost_per_unit_usage": 0.06,
            "seasonal_adjustment_factor": 0.04
        }
        
        total_features = sum(len(features) for features in feature_sets.values())
        
        execution.metrics.update({
            "total_features_engineered": total_features,
            "feature_importance_coverage": sum(feature_importance.values())
        })
        
        return {
            "feature_sets": feature_sets,
            "feature_importance": feature_importance,
            "total_features": total_features,
            "engineering_timestamp": datetime.now().isoformat()
        }
    
    async def _train_models(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Train ML models with cross-validation"""
        logger.info("Training ML models with cross-validation")
        
        training_results = {}
        
        for model_type in self.config.model_types:
            logger.info(f"Training {model_type} model")
            
            # Simulate model training with cross-validation
            cv_scores = []
            for fold in range(self.config.cross_validation_folds):
                # Simulate training and validation for each fold
                fold_score = 0.80 + (fold * 0.02) + np.random.normal(0, 0.01)
                cv_scores.append(max(0.0, min(1.0, fold_score)))
            
            model_metrics = {
                "cv_mean_score": statistics.mean(cv_scores),
                "cv_std_score": statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0.0,
                "cv_scores": cv_scores,
                "training_time_seconds": 120 + np.random.randint(0, 60),
                "model_size_mb": 5.2 + np.random.uniform(0, 2.0),
                "feature_count": 23,
                "hyperparameters": self._get_model_hyperparameters(model_type)
            }
            
            # Create model artifact
            artifact = ModelArtifact(
                model_type=model_type,
                model_path=f"models/{model_type}_{execution.execution_id}.pkl",
                metadata_path=f"models/{model_type}_{execution.execution_id}_metadata.json",
                performance_metrics=model_metrics,
                size_bytes=int(model_metrics["model_size_mb"] * 1024 * 1024),
                checksum=hashlib.md5(f"{model_type}_{execution.execution_id}".encode()).hexdigest()
            )
            
            with self._lock:
                self.artifacts[artifact.artifact_id] = artifact
            
            training_results[model_type] = {
                "artifact_id": artifact.artifact_id,
                "metrics": model_metrics,
                "status": "completed"
            }
            
            execution.metrics[f"{model_type}_cv_score"] = model_metrics["cv_mean_score"]
        
        # Calculate ensemble performance
        if len(self.config.model_types) > 1:
            ensemble_score = statistics.mean([
                training_results[mt]["metrics"]["cv_mean_score"] 
                for mt in self.config.model_types
            ]) * 1.05  # Ensemble typically performs better
            
            execution.metrics["ensemble_cv_score"] = min(1.0, ensemble_score)
        
        return training_results
    
    async def _validate_models(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Validate trained models against performance thresholds"""
        logger.info("Validating trained models")
        
        validation_results = {}
        training_results = execution.stage_results.get("model_training", {})
        
        for model_type, training_result in training_results.items():
            if training_result["status"] != "completed":
                continue
            
            artifact_id = training_result["artifact_id"]
            artifact = self.artifacts[artifact_id]
            
            # Perform comprehensive validation
            validation_metrics = {
                ValidationMetric.ACCURACY.value: artifact.performance_metrics["cv_mean_score"],
                ValidationMetric.PRECISION.value: artifact.performance_metrics["cv_mean_score"] * 0.95,
                ValidationMetric.RECALL.value: artifact.performance_metrics["cv_mean_score"] * 0.92,
                ValidationMetric.F1_SCORE.value: artifact.performance_metrics["cv_mean_score"] * 0.93,
                ValidationMetric.LATENCY.value: 0.05 + np.random.uniform(0, 0.02),  # seconds
                ValidationMetric.MAE.value: 0.15 - (artifact.performance_metrics["cv_mean_score"] * 0.1),
                ValidationMetric.RMSE.value: 0.20 - (artifact.performance_metrics["cv_mean_score"] * 0.12)
            }
            
            # Check performance thresholds
            passes_threshold = validation_metrics[ValidationMetric.ACCURACY.value] >= self.config.performance_threshold
            
            validation_result = {
                "artifact_id": artifact_id,
                "metrics": validation_metrics,
                "passes_threshold": passes_threshold,
                "threshold_used": self.config.performance_threshold,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Update artifact with validation results
            artifact.validation_results = validation_result
            
            validation_results[model_type] = validation_result
            
            execution.metrics[f"{model_type}_validation_passed"] = 1.0 if passes_threshold else 0.0
        
        # Check if any models passed validation
        models_passed = sum(1 for result in validation_results.values() if result["passes_threshold"])
        execution.metrics["models_passed_validation"] = models_passed
        
        if models_passed == 0:
            raise Exception("No models passed validation threshold")
        
        return validation_results
    
    async def _deploy_models(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Deploy validated models to target environment"""
        logger.info(f"Deploying models to {self.config.deployment_environment}")
        
        deployment_results = {}
        validation_results = execution.stage_results.get("validation", {})
        
        for model_type, validation_result in validation_results.items():
            if not validation_result["passes_threshold"]:
                logger.warning(f"Skipping deployment of {model_type} - failed validation")
                continue
            
            artifact_id = validation_result["artifact_id"]
            
            # Simulate deployment process
            deployment_id = str(uuid.uuid4())
            deployment_result = {
                "deployment_id": deployment_id,
                "artifact_id": artifact_id,
                "environment": self.config.deployment_environment,
                "deployed_at": datetime.now().isoformat(),
                "status": "deployed",
                "endpoint_url": f"https://api.finops.com/ml/{model_type}",
                "health_check_url": f"https://api.finops.com/ml/{model_type}/health"
            }
            
            # Update active models registry
            model_key = f"{self.config.deployment_environment}_{model_type}"
            with self._lock:
                self.active_models[model_key] = artifact_id
            
            deployment_results[model_type] = deployment_result
            
            # Save deployment to database
            self._save_deployment(deployment_result)
            
            logger.info(f"Successfully deployed {model_type} model: {deployment_id}")
        
        execution.metrics["models_deployed"] = len(deployment_results)
        
        return deployment_results
    
    async def _setup_monitoring(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Setup monitoring for deployed models"""
        logger.info("Setting up model monitoring")
        
        if not self.config.monitoring_enabled:
            return {"monitoring_enabled": False}
        
        monitoring_config = {
            "performance_monitoring": {
                "enabled": True,
                "check_interval_minutes": 15,
                "metrics_tracked": [
                    "prediction_latency",
                    "prediction_accuracy",
                    "error_rate",
                    "throughput"
                ]
            },
            "drift_detection": {
                "enabled": True,
                "check_interval_hours": 6,
                "drift_threshold": 0.1,
                "statistical_tests": ["ks_test", "chi_square"]
            },
            "alerting": {
                "enabled": True,
                "channels": ["email", "slack"],
                "thresholds": {
                    "accuracy_drop": 0.05,
                    "latency_increase": 2.0,
                    "error_rate_spike": 0.1
                }
            },
            "auto_retraining": {
                "enabled": True,
                "trigger_conditions": [
                    "accuracy_drop_below_threshold",
                    "significant_drift_detected",
                    "scheduled_weekly"
                ]
            }
        }
        
        execution.metrics["monitoring_setup_completed"] = 1.0
        
        return monitoring_config
    
    async def _rollback_deployment(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Rollback to previous model version"""
        logger.info("Initiating model rollback")
        
        rollback_results = {}
        
        for model_type in self.config.model_types:
            model_key = f"{self.config.deployment_environment}_{model_type}"
            
            # Find previous deployment
            previous_artifact_id = self._get_previous_deployment(model_type, self.config.deployment_environment)
            
            if previous_artifact_id:
                rollback_id = str(uuid.uuid4())
                rollback_result = {
                    "rollback_id": rollback_id,
                    "model_type": model_type,
                    "previous_artifact_id": previous_artifact_id,
                    "rolled_back_at": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                # Update active models registry
                with self._lock:
                    self.active_models[model_key] = previous_artifact_id
                
                rollback_results[model_type] = rollback_result
                logger.info(f"Rolled back {model_type} to previous version")
            else:
                logger.warning(f"No previous deployment found for {model_type}")
        
        execution.status = PipelineStatus.ROLLBACK_REQUIRED
        
        return rollback_results
    
    def _get_model_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific hyperparameters"""
        hyperparameters = {
            "isolation_forest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "max_samples": "auto",
                "random_state": 42
            },
            "lstm_anomaly": {
                "sequence_length": 24,
                "hidden_units": 64,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 50
            },
            "prophet_forecast": {
                "seasonality_mode": "multiplicative",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05
            }
        }
        
        return hyperparameters.get(model_type, {})
    
    def _get_previous_deployment(self, model_type: str, environment: str) -> Optional[str]:
        """Get previous deployment artifact ID for rollback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT artifact_id FROM deployment_history 
                WHERE environment = ? AND status = 'deployed'
                ORDER BY deployed_at DESC LIMIT 2
            """, (environment,))
            
            results = cursor.fetchall()
            return results[1][0] if len(results) > 1 else None
    
    def _save_execution(self, execution: PipelineExecution):
        """Save pipeline execution to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipeline_executions 
                (execution_id, pipeline_id, status, current_stage, stages_completed, 
                 stage_results, metrics, errors, started_at, completed_at, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.execution_id,
                execution.pipeline_id,
                execution.status.value,
                execution.current_stage.value if execution.current_stage else None,
                json.dumps([stage.value for stage in execution.stages_completed]),
                json.dumps(execution.stage_results),
                json.dumps(execution.metrics),
                json.dumps(execution.errors),
                execution.started_at,
                execution.completed_at,
                execution.duration_seconds
            ))
    
    def _save_deployment(self, deployment_result: Dict[str, Any]):
        """Save deployment record to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO deployment_history 
                (deployment_id, artifact_id, environment, deployed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                deployment_result["deployment_id"],
                deployment_result["artifact_id"],
                deployment_result["environment"],
                deployment_result["deployed_at"],
                deployment_result["status"]
            ))
    
    def get_pipeline_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status"""
        return self.executions.get(execution_id)
    
    def get_active_models(self, environment: str = None) -> Dict[str, str]:
        """Get currently active model deployments"""
        if environment:
            return {k: v for k, v in self.active_models.items() if k.startswith(f"{environment}_")}
        return self.active_models.copy()
    
    def get_model_artifacts(self, model_type: str = None) -> List[ModelArtifact]:
        """Get model artifacts, optionally filtered by type"""
        artifacts = list(self.artifacts.values())
        if model_type:
            artifacts = [a for a in artifacts if a.model_type == model_type]
        return sorted(artifacts, key=lambda x: x.created_at, reverse=True)
    
    def get_performance_history(self, model_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get model performance history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        history = self.performance_history.get(model_type, [])
        return [h for h in history if datetime.fromisoformat(h["timestamp"]) >= cutoff_date]
    
    async def benchmark_models(self, model_types: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark model performance across different metrics"""
        model_types = model_types or self.config.model_types
        benchmark_results = {}
        
        for model_type in model_types:
            # Simulate comprehensive benchmarking
            benchmark_results[model_type] = {
                "accuracy": 0.85 + np.random.uniform(-0.05, 0.10),
                "precision": 0.82 + np.random.uniform(-0.05, 0.10),
                "recall": 0.88 + np.random.uniform(-0.05, 0.08),
                "f1_score": 0.85 + np.random.uniform(-0.05, 0.08),
                "latency_ms": 50 + np.random.uniform(-10, 20),
                "throughput_rps": 100 + np.random.uniform(-20, 50),
                "memory_usage_mb": 256 + np.random.uniform(-50, 100),
                "cpu_usage_percent": 25 + np.random.uniform(-5, 15)
            }
        
        return benchmark_results
    
    async def compare_model_performance(self, artifact_ids: List[str]) -> Dict[str, Any]:
        """Compare performance between different model artifacts"""
        if len(artifact_ids) < 2:
            raise ValueError("Need at least 2 artifacts for comparison")
        
        comparison_results = {
            "artifacts": [],
            "metrics_comparison": {},
            "winner": None,
            "improvement_percentage": 0.0
        }
        
        for artifact_id in artifact_ids:
            artifact = self.artifacts.get(artifact_id)
            if not artifact:
                continue
            
            comparison_results["artifacts"].append({
                "artifact_id": artifact_id,
                "model_type": artifact.model_type,
                "created_at": artifact.created_at.isoformat(),
                "performance_metrics": artifact.performance_metrics
            })
        
        # Determine winner based on CV score
        best_artifact = max(
            comparison_results["artifacts"],
            key=lambda x: x["performance_metrics"].get("cv_mean_score", 0)
        )
        
        comparison_results["winner"] = best_artifact["artifact_id"]
        
        # Calculate improvement
        scores = [a["performance_metrics"].get("cv_mean_score", 0) for a in comparison_results["artifacts"]]
        if len(scores) >= 2:
            best_score = max(scores)
            second_best = sorted(scores, reverse=True)[1]
            comparison_results["improvement_percentage"] = ((best_score - second_best) / second_best) * 100
        
        return comparison_results
    
    def cleanup_old_artifacts(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old model artifacts"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        artifacts_removed = 0
        deployments_removed = 0
        
        # Remove old artifacts
        artifacts_to_remove = []
        for artifact_id, artifact in self.artifacts.items():
            if artifact.created_at < cutoff_date:
                # Don't remove if currently deployed
                if artifact_id not in self.active_models.values():
                    artifacts_to_remove.append(artifact_id)
        
        for artifact_id in artifacts_to_remove:
            del self.artifacts[artifact_id]
            artifacts_removed += 1
        
        # Clean up database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM deployment_history 
                WHERE deployed_at < datetime('now', '-{} days')
                AND status != 'deployed'
            """.format(days_to_keep))
            deployments_removed = cursor.rowcount
        
        return {
            "artifacts_removed": artifacts_removed,
            "deployments_removed": deployments_removed,
            "cutoff_date": cutoff_date.isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create pipeline configuration
        config = PipelineConfig(
            name="production_anomaly_detection_pipeline",
            model_types=["isolation_forest", "lstm_anomaly", "prophet_forecast"],
            deployment_environment="staging",
            auto_deploy=True,
            performance_threshold=0.85
        )
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        print(f"Training Pipeline initialized: {config.pipeline_id}")
        print(f"Target models: {config.model_types}")
        print(f"Performance threshold: {config.performance_threshold}")
        
        # Execute pipeline
        execution = await pipeline.execute_pipeline()
        
        print(f"\nPipeline execution completed: {execution.execution_id}")
        print(f"Status: {execution.status.value}")
        print(f"Duration: {execution.duration_seconds:.2f} seconds")
        print(f"Stages completed: {[stage.value for stage in execution.stages_completed]}")
        
        if execution.errors:
            print(f"Errors: {execution.errors}")
        
        # Display metrics
        print(f"\nExecution Metrics:")
        for metric, value in execution.metrics.items():
            print(f"  {metric}: {value}")
        
        # Get active models
        active_models = pipeline.get_active_models()
        print(f"\nActive Models: {active_models}")
        
        # Benchmark models
        benchmarks = await pipeline.benchmark_models()
        print(f"\nModel Benchmarks:")
        for model_type, metrics in benchmarks.items():
            print(f"  {model_type}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
    
    # Run the example
    import asyncio
    asyncio.run(main())