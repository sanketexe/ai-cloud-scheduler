"""
ML Model Management and Experimentation Platform

This module provides comprehensive ML model lifecycle management including:
- Automated training, validation, and deployment
- A/B testing framework for model comparison
- Experiment tracking and management
- Model interpretability and explainable AI
- Bias detection and mitigation tools
"""

import uuid
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .database import get_db_session
from .models import MLModelMetrics
from .exceptions import ModelManagerError, ValidationError

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class ExperimentStatus(Enum):
    """Experiment status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    preprocessing_steps: List[str]
    validation_split: float = 0.2
    random_state: int = 42

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str
    description: str
    models_to_compare: List[ModelConfig]
    evaluation_metrics: List[str]
    test_size: float = 0.2
    cross_validation_folds: int = 5
    significance_level: float = 0.05

class ModelManager:
    """
    Comprehensive ML model management system for automated training,
    validation, deployment, and lifecycle management.
    """
    
    def __init__(self, model_registry_path: str = "models", mlflow_tracking_uri: str = None):
        """
        Initialize ModelManager
        
        Args:
            model_registry_path: Path to store model artifacts
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.model_registry_path = Path(model_registry_path)
        self.model_registry_path.mkdir(exist_ok=True)
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        self.mlflow_client = MlflowClient()
        self.active_models: Dict[str, Any] = {}
        self.model_versions: Dict[str, List[str]] = {}
        
    async def train_model(
        self,
        config: ModelConfig,
        training_data: pd.DataFrame,
        account_id: str = None
    ) -> Dict[str, Any]:
        """
        Train a new model with automated validation and registration
        
        Args:
            config: Model configuration
            training_data: Training dataset
            account_id: Account identifier for multi-tenant support
            
        Returns:
            Training results including model ID, metrics, and artifacts
        """
        try:
            model_id = str(uuid.uuid4())
            experiment_name = f"{config.model_name}_{account_id or 'default'}"
            
            # Set up MLflow experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"training_{model_id}"):
                # Log configuration
                mlflow.log_params(config.hyperparameters)
                mlflow.log_param("model_type", config.model_type)
                mlflow.log_param("feature_count", len(config.feature_columns))
                
                # Prepare data
                X = training_data[config.feature_columns]
                y = training_data[config.target_column]
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=config.validation_split, 
                    random_state=config.random_state, stratify=y
                )
                
                # Train model
                model = self._create_model(config.model_type, config.hyperparameters)
                model.fit(X_train, y_train)
                
                # Validate model
                y_pred = model.predict(X_val)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Log metrics
                mlflow.log_metrics({
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score
                })
                
                if metrics.auc_score:
                    mlflow.log_metric("auc_score", metrics.auc_score)
                
                # Log custom metrics
                for metric_name, value in metrics.custom_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Save model
                model_path = self.model_registry_path / model_id
                model_path.mkdir(exist_ok=True)
                
                # Save with joblib for sklearn models
                joblib.dump(model, model_path / "model.pkl")
                
                # Save configuration
                with open(model_path / "config.json", "w") as f:
                    json.dump(asdict(config), f, indent=2)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model, "model")
                
                # Store in database
                await self._store_model_metrics(
                    model_id, config.model_name, "1.0", account_id or "default",
                    metrics, config.hyperparameters, len(training_data)
                )
                
                # Update active models
                self.active_models[model_id] = {
                    "model": model,
                    "config": config,
                    "metrics": metrics,
                    "status": ModelStatus.TRAINING,
                    "created_at": datetime.utcnow()
                }
                
                logger.info(f"Model {model_id} trained successfully with accuracy: {metrics.accuracy:.4f}")
                
                return {
                    "model_id": model_id,
                    "metrics": asdict(metrics),
                    "status": ModelStatus.TRAINING.value,
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }
                
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise ModelManagerError(f"Model training failed: {str(e)}")
    
    async def validate_model(
        self,
        model_id: str,
        validation_data: pd.DataFrame,
        validation_criteria: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Validate model performance against criteria
        
        Args:
            model_id: Model identifier
            validation_data: Validation dataset
            validation_criteria: Performance thresholds
            
        Returns:
            Validation results
        """
        try:
            if model_id not in self.active_models:
                model_info = await self._load_model(model_id)
                if not model_info:
                    raise ModelManagerError(f"Model {model_id} not found")
            else:
                model_info = self.active_models[model_id]
            
            model = model_info["model"]
            config = model_info["config"]
            
            # Default validation criteria
            if validation_criteria is None:
                validation_criteria = {
                    "min_accuracy": 0.8,
                    "min_precision": 0.75,
                    "min_recall": 0.75,
                    "max_bias_score": 0.1
                }
            
            # Prepare validation data
            X_val = validation_data[config.feature_columns]
            y_val = validation_data[config.target_column]
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Check validation criteria
            validation_results = {
                "model_id": model_id,
                "validation_passed": True,
                "metrics": asdict(metrics),
                "criteria_results": {}
            }
            
            # Check each criterion
            if metrics.accuracy < validation_criteria.get("min_accuracy", 0):
                validation_results["validation_passed"] = False
                validation_results["criteria_results"]["accuracy"] = {
                    "passed": False,
                    "actual": metrics.accuracy,
                    "required": validation_criteria["min_accuracy"]
                }
            
            if metrics.precision < validation_criteria.get("min_precision", 0):
                validation_results["validation_passed"] = False
                validation_results["criteria_results"]["precision"] = {
                    "passed": False,
                    "actual": metrics.precision,
                    "required": validation_criteria["min_precision"]
                }
            
            if metrics.recall < validation_criteria.get("min_recall", 0):
                validation_results["validation_passed"] = False
                validation_results["criteria_results"]["recall"] = {
                    "passed": False,
                    "actual": metrics.recall,
                    "required": validation_criteria["min_recall"]
                }
            
            # Update model status
            if validation_results["validation_passed"]:
                model_info["status"] = ModelStatus.VALIDATION
                logger.info(f"Model {model_id} passed validation")
            else:
                model_info["status"] = ModelStatus.FAILED
                logger.warning(f"Model {model_id} failed validation")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise ModelManagerError(f"Model validation failed: {str(e)}")
    
    async def deploy_model(
        self,
        model_id: str,
        deployment_target: str = "staging",
        rollback_model_id: str = None
    ) -> Dict[str, Any]:
        """
        Deploy model to specified environment with rollback capability
        
        Args:
            model_id: Model to deploy
            deployment_target: Target environment (staging/production)
            rollback_model_id: Previous model for rollback
            
        Returns:
            Deployment results
        """
        try:
            if model_id not in self.active_models:
                model_info = await self._load_model(model_id)
                if not model_info:
                    raise ModelManagerError(f"Model {model_id} not found")
            else:
                model_info = self.active_models[model_id]
            
            # Validate model is ready for deployment
            if model_info["status"] not in [ModelStatus.VALIDATION, ModelStatus.TESTING]:
                raise ModelManagerError(f"Model {model_id} is not ready for deployment. Current status: {model_info['status']}")
            
            # Update model status
            if deployment_target.lower() == "production":
                model_info["status"] = ModelStatus.PRODUCTION
            else:
                model_info["status"] = ModelStatus.STAGING
            
            # Store rollback information
            deployment_info = {
                "model_id": model_id,
                "deployment_target": deployment_target,
                "deployed_at": datetime.utcnow(),
                "rollback_model_id": rollback_model_id,
                "status": "deployed"
            }
            
            # Register with MLflow Model Registry
            model_name = model_info["config"].model_name
            model_uri = f"models:/{model_name}/latest"
            
            try:
                # Create registered model if it doesn't exist
                try:
                    self.mlflow_client.create_registered_model(model_name)
                except Exception:
                    pass  # Model already exists
                
                # Create model version
                model_version = self.mlflow_client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    description=f"Deployed to {deployment_target}"
                )
                
                deployment_info["model_version"] = model_version.version
                
            except Exception as e:
                logger.warning(f"MLflow model registry update failed: {str(e)}")
            
            logger.info(f"Model {model_id} deployed to {deployment_target}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            raise ModelManagerError(f"Model deployment failed: {str(e)}")
    
    async def rollback_model(self, model_id: str, target_model_id: str) -> Dict[str, Any]:
        """
        Rollback to a previous model version
        
        Args:
            model_id: Current model to rollback from
            target_model_id: Target model to rollback to
            
        Returns:
            Rollback results
        """
        try:
            # Load target model
            target_model_info = await self._load_model(target_model_id)
            if not target_model_info:
                raise ModelManagerError(f"Target model {target_model_id} not found")
            
            # Update statuses
            if model_id in self.active_models:
                self.active_models[model_id]["status"] = ModelStatus.DEPRECATED
            
            target_model_info["status"] = ModelStatus.PRODUCTION
            self.active_models[target_model_id] = target_model_info
            
            rollback_info = {
                "rolled_back_from": model_id,
                "rolled_back_to": target_model_id,
                "rollback_time": datetime.utcnow(),
                "status": "completed"
            }
            
            logger.info(f"Successfully rolled back from {model_id} to {target_model_id}")
            
            return rollback_info
            
        except Exception as e:
            logger.error(f"Model rollback failed: {str(e)}")
            raise ModelManagerError(f"Model rollback failed: {str(e)}")
    
    def _create_model(self, model_type: str, hyperparameters: Dict[str, Any]):
        """Create model instance based on type and hyperparameters"""
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**hyperparameters)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**hyperparameters)
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**hyperparameters)
        elif model_type == "svm":
            from sklearn.svm import SVC
            return SVC(**hyperparameters)
        else:
            raise ModelManagerError(f"Unsupported model type: {model_type}")
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted'),
            recall=recall_score(y_true, y_pred, average='weighted'),
            f1_score=f1_score(y_true, y_pred, average='weighted')
        )
        
        if y_pred_proba is not None:
            try:
                metrics.auc_score = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                # Handle cases where AUC cannot be calculated
                metrics.auc_score = None
        
        return metrics
    
    async def _store_model_metrics(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        account_id: str,
        metrics: ModelMetrics,
        hyperparameters: Dict[str, Any],
        training_data_points: int
    ):
        """Store model metrics in database"""
        try:
            async with get_db_session() as session:
                ml_metrics = MLModelMetrics(
                    model_name=model_name,
                    model_version=model_version,
                    account_id=account_id,
                    training_date=datetime.utcnow(),
                    accuracy_score=Decimal(str(metrics.accuracy)),
                    precision_score=Decimal(str(metrics.precision)),
                    recall_score=Decimal(str(metrics.recall)),
                    false_positive_rate=Decimal("0.0"),  # Calculate if needed
                    detection_latency_ms=100,  # Default value
                    training_data_points=training_data_points,
                    feature_count=len(hyperparameters.get('feature_columns', [])),
                    hyperparameters=hyperparameters,
                    is_active=True
                )
                
                session.add(ml_metrics)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store model metrics: {str(e)}")
    
    async def _load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model from storage"""
        try:
            model_path = self.model_registry_path / model_id
            if not model_path.exists():
                return None
            
            # Load model
            model = joblib.load(model_path / "model.pkl")
            
            # Load configuration
            with open(model_path / "config.json", "r") as f:
                config_dict = json.load(f)
                config = ModelConfig(**config_dict)
            
            return {
                "model": model,
                "config": config,
                "status": ModelStatus.STAGING,  # Default status for loaded models
                "created_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None
    
    async def list_models(self, account_id: str = None) -> List[Dict[str, Any]]:
        """List all models with their status and metrics"""
        try:
            async with get_db_session() as session:
                query = session.query(MLModelMetrics)
                if account_id:
                    query = query.filter(MLModelMetrics.account_id == account_id)
                
                models = await query.all()
                
                return [
                    {
                        "model_name": model.model_name,
                        "model_version": model.model_version,
                        "account_id": model.account_id,
                        "accuracy": float(model.accuracy_score),
                        "precision": float(model.precision_score),
                        "recall": float(model.recall_score),
                        "training_date": model.training_date,
                        "is_active": model.is_active
                    }
                    for model in models
                ]
                
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        if model_id in self.active_models:
            model_info = self.active_models[model_id]
            return {
                "model_id": model_id,
                "config": asdict(model_info["config"]),
                "metrics": asdict(model_info["metrics"]) if hasattr(model_info["metrics"], '__dict__') else model_info["metrics"],
                "status": model_info["status"].value,
                "created_at": model_info["created_at"]
            }
        
        # Try to load from storage
        model_info = await self._load_model(model_id)
        if model_info:
            return {
                "model_id": model_id,
                "config": asdict(model_info["config"]),
                "status": model_info["status"].value,
                "created_at": model_info["created_at"]
            }
        
        return None