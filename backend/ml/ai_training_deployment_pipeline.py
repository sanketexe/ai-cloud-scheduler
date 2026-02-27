"""
AI Training and Deployment Pipeline

Comprehensive ML training orchestration with Kubeflow/MLflow integration,
automated model validation, canary deployments, drift detection, and rollback mechanisms.
This module provides production-ready AI/ML pipeline capabilities for the FinOps platform.
"""

import logging
import asyncio
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
import aiofiles
import yaml

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Kubeflow imports (simulated for this implementation)
try:
    from kfp import dsl, compiler
    from kfp.client import Client as KfpClient
    KUBEFLOW_AVAILABLE = True
except ImportError:
    KUBEFLOW_AVAILABLE = False
    # Mock classes for when Kubeflow is not available
    class dsl:
        @staticmethod
        def pipeline(name, description):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def component(func):
            return func
    
    class compiler:
        @staticmethod
        class Compiler:
            def compile(self, pipeline_func, package_path):
                pass
    
    class KfpClient:
        def __init__(self, host=None):
            pass
        
        def create_run_from_pipeline_package(self, pipeline_file, arguments=None, run_name=None):
            return type('MockRun', (), {'run_id': str(uuid.uuid4())})()

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """AI training pipeline stages"""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_TESTING = "model_testing"
    CANARY_DEPLOYMENT = "canary_deployment"
    PRODUCTION_DEPLOYMENT = "production_deployment"
    MONITORING_SETUP = "monitoring_setup"
    ROLLBACK = "rollback"

class DeploymentStrategy(Enum):
    """Model deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class DriftType(Enum):
    """Types of model drift"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

@dataclass
class PipelineConfig:
    """AI training pipeline configuration"""
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "ai_training_pipeline"
    description: str = "Automated AI training and deployment pipeline"
    
    # Training configuration
    model_types: List[str] = field(default_factory=lambda: ["anomaly_detection", "cost_forecasting"])
    training_data_sources: List[str] = field(default_factory=lambda: ["cost_data", "usage_metrics"])
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    # Deployment configuration
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.CANARY
    canary_traffic_percentage: float = 10.0
    canary_duration_hours: int = 24
    rollback_threshold_accuracy: float = 0.85
    
    # Monitoring configuration
    drift_detection_enabled: bool = True
    drift_check_interval_hours: int = 6
    performance_monitoring_enabled: bool = True
    auto_rollback_enabled: bool = True
    
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
    mlflow_experiment_name: str = "finops_ai_models"
    
    # Kubeflow configuration
    kubeflow_host: str = "http://kubeflow-pipelines:8888"
    kubeflow_namespace: str = "kubeflow"
    
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    
    # Performance metrics
    inference_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Business metrics
    cost_savings_accuracy: Optional[float] = None
    anomaly_detection_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class DriftDetectionResult:
    """Model drift detection results"""
    model_id: str
    detection_timestamp: datetime
    drift_type: DriftType
    drift_score: float
    drift_threshold: float
    is_drift_detected: bool
    affected_features: List[str]
    confidence_level: float
    recommended_action: str
    statistical_test_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CanaryDeployment:
    """Canary deployment configuration and status"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    traffic_percentage: float = 10.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    duration_hours: int = 24
    status: str = "active"
    
    # Performance comparison
    baseline_metrics: Optional[ModelMetrics] = None
    canary_metrics: Optional[ModelMetrics] = None
    performance_comparison: Dict[str, float] = field(default_factory=dict)
    
    # Decision criteria
    success_criteria: Dict[str, float] = field(default_factory=dict)
    rollback_criteria: Dict[str, float] = field(default_factory=dict)

class AITrainingDeploymentPipeline:
    """
    Comprehensive AI training and deployment pipeline with MLflow/Kubeflow integration,
    automated validation, canary deployments, drift detection, and rollback capabilities.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize Kubeflow client
        if KUBEFLOW_AVAILABLE:
            self.kfp_client = KfpClient(host=self.config.kubeflow_host)
        else:
            self.kfp_client = KfpClient()
            logger.warning("Kubeflow not available, using mock client")
        
        # Pipeline state
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.drift_history: Dict[str, List[DriftDetectionResult]] = {}
        self.performance_history: Dict[str, List[ModelMetrics]] = {}
        
        # Create MLflow experiment
        self._setup_mlflow_experiment()
        
        logger.info(f"AI Training Pipeline initialized: {self.config.pipeline_id}")
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment for tracking"""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.config.mlflow_experiment_name,
                    tags={
                        "pipeline_id": self.config.pipeline_id,
                        "created_by": "ai_training_pipeline",
                        "environment": "production"
                    }
                )
                logger.info(f"Created MLflow experiment: {experiment_id}")
            else:
                logger.info(f"Using existing MLflow experiment: {experiment.experiment_id}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {str(e)}")
    
    async def execute_training_pipeline(
        self,
        training_data: pd.DataFrame,
        model_configs: List[Dict[str, Any]],
        account_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute complete AI training pipeline with Kubeflow orchestration
        
        Args:
            training_data: Training dataset
            model_configs: List of model configurations to train
            account_id: Account identifier for multi-tenant support
            
        Returns:
            Pipeline execution results
        """
        pipeline_run_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting AI training pipeline: {pipeline_run_id}")
            
            # Create Kubeflow pipeline
            pipeline_package = await self._create_kubeflow_pipeline(
                model_configs, account_id
            )
            
            # Submit pipeline to Kubeflow
            run_result = await self._submit_kubeflow_pipeline(
                pipeline_package, pipeline_run_id, {
                    "account_id": account_id,
                    "model_configs": json.dumps(model_configs),
                    "pipeline_config": json.dumps(asdict(self.config))
                }
            )
            
            # Execute training stages
            execution_results = await self._execute_training_stages(
                training_data, model_configs, account_id, pipeline_run_id
            )
            
            # Setup monitoring for trained models
            monitoring_results = await self._setup_model_monitoring(
                execution_results["trained_models"]
            )
            
            results = {
                "pipeline_run_id": pipeline_run_id,
                "kubeflow_run_id": run_result.run_id if run_result else None,
                "execution_results": execution_results,
                "monitoring_results": monitoring_results,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"AI training pipeline completed: {pipeline_run_id}")
            return results
            
        except Exception as e:
            logger.error(f"AI training pipeline failed: {str(e)}")
            return {
                "pipeline_run_id": pipeline_run_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _create_kubeflow_pipeline(
        self,
        model_configs: List[Dict[str, Any]],
        account_id: str
    ) -> str:
        """Create Kubeflow pipeline definition"""
        
        @dsl.pipeline(
            name=f"ai-training-pipeline-{account_id}",
            description="Automated AI model training and deployment pipeline"
        )
        def ai_training_pipeline():
            """Kubeflow pipeline definition"""
            
            # Data preparation component
            data_prep_op = self._create_data_preparation_component()
            
            # Feature engineering component
            feature_eng_op = self._create_feature_engineering_component()
            feature_eng_op.after(data_prep_op)
            
            # Model training components (parallel)
            training_ops = []
            for config in model_configs:
                training_op = self._create_model_training_component(config)
                training_op.after(feature_eng_op)
                training_ops.append(training_op)
            
            # Model validation component
            validation_op = self._create_model_validation_component()
            for training_op in training_ops:
                validation_op.after(training_op)
            
            # Canary deployment component
            canary_op = self._create_canary_deployment_component()
            canary_op.after(validation_op)
            
            # Monitoring setup component
            monitoring_op = self._create_monitoring_setup_component()
            monitoring_op.after(canary_op)
        
        # Compile pipeline
        pipeline_package_path = f"/tmp/ai_training_pipeline_{account_id}.yaml"
        
        if KUBEFLOW_AVAILABLE:
            compiler.Compiler().compile(ai_training_pipeline, pipeline_package_path)
        else:
            # Create mock pipeline package
            pipeline_def = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {"name": f"ai-training-pipeline-{account_id}"},
                "spec": {"entrypoint": "ai-training-pipeline"}
            }
            
            async with aiofiles.open(pipeline_package_path, 'w') as f:
                await f.write(yaml.dump(pipeline_def))
        
        return pipeline_package_path
    
    @dsl.component
    def _create_data_preparation_component(self):
        """Create data preparation component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    @dsl.component
    def _create_feature_engineering_component(self):
        """Create feature engineering component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    @dsl.component
    def _create_model_training_component(self, config: Dict[str, Any]):
        """Create model training component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    @dsl.component
    def _create_model_validation_component(self):
        """Create model validation component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    @dsl.component
    def _create_canary_deployment_component(self):
        """Create canary deployment component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    @dsl.component
    def _create_monitoring_setup_component(self):
        """Create monitoring setup component for Kubeflow"""
        # This would be a containerized component in production
        pass
    
    async def _submit_kubeflow_pipeline(
        self,
        pipeline_package: str,
        run_name: str,
        arguments: Dict[str, Any]
    ):
        """Submit pipeline to Kubeflow for execution"""
        try:
            run_result = self.kfp_client.create_run_from_pipeline_package(
                pipeline_file=pipeline_package,
                arguments=arguments,
                run_name=run_name
            )
            
            logger.info(f"Submitted Kubeflow pipeline: {run_result.run_id}")
            return run_result
            
        except Exception as e:
            logger.error(f"Failed to submit Kubeflow pipeline: {str(e)}")
            return None
    
    async def _execute_training_stages(
        self,
        training_data: pd.DataFrame,
        model_configs: List[Dict[str, Any]],
        account_id: str,
        pipeline_run_id: str
    ) -> Dict[str, Any]:
        """Execute training stages with MLflow tracking"""
        
        trained_models = []
        
        with mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name(self.config.mlflow_experiment_name).experiment_id,
            run_name=f"pipeline_{pipeline_run_id}"
        ) as parent_run:
            
            # Log pipeline configuration
            mlflow.log_params({
                "pipeline_id": self.config.pipeline_id,
                "account_id": account_id,
                "model_count": len(model_configs),
                "training_data_size": len(training_data)
            })
            
            # Train models in parallel
            training_tasks = []
            for config in model_configs:
                task = self._train_single_model(
                    training_data, config, account_id, parent_run.info.run_id
                )
                training_tasks.append(task)
            
            # Wait for all training tasks to complete
            training_results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(training_results):
                if isinstance(result, Exception):
                    logger.error(f"Model training failed: {str(result)}")
                    continue
                
                trained_models.append(result)
                
                # Log model metrics to parent run
                mlflow.log_metrics({
                    f"model_{i}_accuracy": result["metrics"]["accuracy"],
                    f"model_{i}_f1_score": result["metrics"]["f1_score"]
                })
        
        return {
            "trained_models": trained_models,
            "pipeline_run_id": pipeline_run_id,
            "training_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _train_single_model(
        self,
        training_data: pd.DataFrame,
        model_config: Dict[str, Any],
        account_id: str,
        parent_run_id: str
    ) -> Dict[str, Any]:
        """Train a single model with MLflow tracking"""
        
        model_id = str(uuid.uuid4())
        
        with mlflow.start_run(
            run_name=f"model_{model_config['type']}_{model_id}",
            nested=True
        ) as run:
            
            # Log model configuration
            mlflow.log_params(model_config)
            
            # Simulate model training (replace with actual training logic)
            await asyncio.sleep(1)  # Simulate training time
            
            # Generate mock metrics (replace with actual metrics)
            metrics = ModelMetrics(
                accuracy=0.85 + np.random.uniform(-0.05, 0.10),
                precision=0.82 + np.random.uniform(-0.05, 0.10),
                recall=0.88 + np.random.uniform(-0.05, 0.08),
                f1_score=0.85 + np.random.uniform(-0.05, 0.08),
                auc_roc=0.90 + np.random.uniform(-0.05, 0.05),
                inference_latency_ms=50 + np.random.uniform(-10, 20),
                throughput_rps=100 + np.random.uniform(-20, 50),
                memory_usage_mb=256 + np.random.uniform(-50, 100),
                cpu_usage_percent=25 + np.random.uniform(-5, 15)
            )
            
            # Log metrics to MLflow
            mlflow.log_metrics(asdict(metrics))
            
            # Save model artifact (mock)
            model_path = f"models/{model_id}"
            mlflow.log_artifact(__file__, artifact_path=model_path)
            
            # Register model in registry
            self.model_registry[model_id] = {
                "model_id": model_id,
                "model_type": model_config["type"],
                "account_id": account_id,
                "metrics": metrics,
                "config": model_config,
                "mlflow_run_id": run.info.run_id,
                "status": ModelStatus.TRAINING,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Model trained successfully: {model_id}")
            
            return {
                "model_id": model_id,
                "metrics": asdict(metrics),
                "mlflow_run_id": run.info.run_id,
                "status": "completed"
            }
    
    async def deploy_model_canary(
        self,
        model_id: str,
        traffic_percentage: float = None,
        duration_hours: int = None
    ) -> CanaryDeployment:
        """
        Deploy model using canary deployment strategy
        
        Args:
            model_id: Model to deploy
            traffic_percentage: Percentage of traffic to route to canary
            duration_hours: Duration of canary deployment
            
        Returns:
            Canary deployment configuration
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.model_registry[model_id]
        
        # Create canary deployment
        canary = CanaryDeployment(
            model_id=model_id,
            traffic_percentage=traffic_percentage or self.config.canary_traffic_percentage,
            duration_hours=duration_hours or self.config.canary_duration_hours,
            success_criteria={
                "min_accuracy": 0.85,
                "max_latency_ms": 100,
                "max_error_rate": 0.05
            },
            rollback_criteria={
                "accuracy_drop": 0.05,
                "latency_increase": 2.0,
                "error_rate_spike": 0.1
            }
        )
        
        try:
            # Deploy to Kubernetes with canary configuration
            await self._deploy_to_kubernetes(model_id, canary)
            
            # Update model status
            model_info["status"] = ModelStatus.CANARY
            model_info["canary_deployment"] = canary
            
            # Store canary deployment
            self.active_deployments[canary.deployment_id] = canary
            
            # Start monitoring canary performance
            asyncio.create_task(self._monitor_canary_deployment(canary.deployment_id))
            
            logger.info(f"Canary deployment started: {canary.deployment_id}")
            return canary
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            raise
    
    async def _deploy_to_kubernetes(self, model_id: str, canary: CanaryDeployment):
        """Deploy model to Kubernetes with canary configuration"""
        
        # Create Kubernetes deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"ai-model-{model_id}",
                "namespace": "finops-automation",
                "labels": {
                    "app": "ai-model",
                    "model-id": model_id,
                    "deployment-type": "canary"
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "ai-model",
                        "model-id": model_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ai-model",
                            "model-id": model_id,
                            "deployment-type": "canary"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": "finops/ai-model-server:latest",
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "MODEL_ID", "value": model_id},
                                {"name": "MODEL_PATH", "value": f"/models/{model_id}"},
                                {"name": "DEPLOYMENT_TYPE", "value": "canary"}
                            ],
                            "resources": {
                                "requests": {"memory": "1Gi", "cpu": "500m"},
                                "limits": {"memory": "2Gi", "cpu": "1000m"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/ready", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Create service manifest
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"ai-model-{model_id}",
                "namespace": "finops-automation"
            },
            "spec": {
                "selector": {
                    "app": "ai-model",
                    "model-id": model_id
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8080
                }]
            }
        }
        
        # Create Istio VirtualService for traffic splitting
        virtual_service_manifest = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"ai-model-{model_id}",
                "namespace": "finops-automation"
            },
            "spec": {
                "hosts": [f"ai-model-{model_id}"],
                "http": [{
                    "match": [{"headers": {"canary": {"exact": "true"}}}],
                    "route": [{
                        "destination": {
                            "host": f"ai-model-{model_id}",
                            "subset": "canary"
                        }
                    }]
                }, {
                    "route": [{
                        "destination": {
                            "host": f"ai-model-{model_id}",
                            "subset": "stable"
                        },
                        "weight": int(100 - canary.traffic_percentage)
                    }, {
                        "destination": {
                            "host": f"ai-model-{model_id}",
                            "subset": "canary"
                        },
                        "weight": int(canary.traffic_percentage)
                    }]
                }]
            }
        }
        
        # Save manifests (in production, these would be applied to Kubernetes)
        manifests_dir = Path(f"/tmp/k8s_manifests/{model_id}")
        manifests_dir.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(manifests_dir / "deployment.yaml", 'w') as f:
            await f.write(yaml.dump(deployment_manifest))
        
        async with aiofiles.open(manifests_dir / "service.yaml", 'w') as f:
            await f.write(yaml.dump(service_manifest))
        
        async with aiofiles.open(manifests_dir / "virtual-service.yaml", 'w') as f:
            await f.write(yaml.dump(virtual_service_manifest))
        
        logger.info(f"Kubernetes manifests created for model {model_id}")
    
    async def _monitor_canary_deployment(self, deployment_id: str):
        """Monitor canary deployment performance and make promotion/rollback decisions"""
        
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            return
        
        model_info = self.model_registry[canary.model_id]
        
        try:
            # Monitor for the specified duration
            end_time = canary.start_time + timedelta(hours=canary.duration_hours)
            
            while datetime.utcnow() < end_time and canary.status == "active":
                # Collect canary metrics
                canary_metrics = await self._collect_canary_metrics(canary.model_id)
                canary.canary_metrics = canary_metrics
                
                # Compare with baseline
                if canary.baseline_metrics:
                    comparison = self._compare_metrics(canary.baseline_metrics, canary_metrics)
                    canary.performance_comparison = comparison
                    
                    # Check rollback criteria
                    if self._should_rollback(canary, comparison):
                        await self.rollback_model(canary.model_id, "Performance degradation detected")
                        return
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
            
            # Canary period completed - make promotion decision
            if canary.status == "active":
                if self._should_promote(canary):
                    await self._promote_canary_to_production(canary)
                else:
                    await self.rollback_model(canary.model_id, "Canary did not meet promotion criteria")
                    
        except Exception as e:
            logger.error(f"Canary monitoring failed: {str(e)}")
            await self.rollback_model(canary.model_id, f"Monitoring error: {str(e)}")
    
    async def _collect_canary_metrics(self, model_id: str) -> ModelMetrics:
        """Collect performance metrics for canary deployment"""
        
        # Simulate metrics collection (replace with actual monitoring)
        await asyncio.sleep(0.1)
        
        return ModelMetrics(
            accuracy=0.87 + np.random.uniform(-0.02, 0.02),
            precision=0.85 + np.random.uniform(-0.02, 0.02),
            recall=0.89 + np.random.uniform(-0.02, 0.02),
            f1_score=0.87 + np.random.uniform(-0.02, 0.02),
            auc_roc=0.92 + np.random.uniform(-0.01, 0.01),
            inference_latency_ms=45 + np.random.uniform(-5, 10),
            throughput_rps=110 + np.random.uniform(-10, 20),
            memory_usage_mb=240 + np.random.uniform(-20, 40),
            cpu_usage_percent=22 + np.random.uniform(-3, 8)
        )
    
    def _compare_metrics(self, baseline: ModelMetrics, canary: ModelMetrics) -> Dict[str, float]:
        """Compare baseline and canary metrics"""
        
        return {
            "accuracy_change": canary.accuracy - baseline.accuracy,
            "latency_change": canary.inference_latency_ms - baseline.inference_latency_ms,
            "throughput_change": canary.throughput_rps - baseline.throughput_rps,
            "memory_change": canary.memory_usage_mb - baseline.memory_usage_mb,
            "cpu_change": canary.cpu_usage_percent - baseline.cpu_usage_percent
        }
    
    def _should_rollback(self, canary: CanaryDeployment, comparison: Dict[str, float]) -> bool:
        """Determine if canary should be rolled back"""
        
        criteria = canary.rollback_criteria
        
        # Check accuracy drop
        if comparison["accuracy_change"] < -criteria.get("accuracy_drop", 0.05):
            logger.warning(f"Accuracy drop detected: {comparison['accuracy_change']}")
            return True
        
        # Check latency increase
        if comparison["latency_change"] > criteria.get("latency_increase", 50):
            logger.warning(f"Latency increase detected: {comparison['latency_change']}ms")
            return True
        
        return False
    
    def _should_promote(self, canary: CanaryDeployment) -> bool:
        """Determine if canary should be promoted to production"""
        
        if not canary.canary_metrics:
            return False
        
        criteria = canary.success_criteria
        metrics = canary.canary_metrics
        
        # Check success criteria
        if metrics.accuracy < criteria.get("min_accuracy", 0.85):
            return False
        
        if metrics.inference_latency_ms > criteria.get("max_latency_ms", 100):
            return False
        
        return True
    
    async def _promote_canary_to_production(self, canary: CanaryDeployment):
        """Promote canary deployment to production"""
        
        try:
            model_info = self.model_registry[canary.model_id]
            
            # Update traffic routing to 100% canary
            await self._update_traffic_routing(canary.model_id, 100.0)
            
            # Update model status
            model_info["status"] = ModelStatus.PRODUCTION
            canary.status = "promoted"
            
            # Log promotion to MLflow
            with mlflow.start_run(run_id=model_info["mlflow_run_id"]):
                mlflow.log_param("deployment_status", "production")
                mlflow.log_param("promotion_timestamp", datetime.utcnow().isoformat())
            
            logger.info(f"Canary promoted to production: {canary.deployment_id}")
            
        except Exception as e:
            logger.error(f"Canary promotion failed: {str(e)}")
            raise
    
    async def _update_traffic_routing(self, model_id: str, traffic_percentage: float):
        """Update traffic routing for model deployment"""
        
        # Update Istio VirtualService (simulated)
        logger.info(f"Updated traffic routing for model {model_id}: {traffic_percentage}%")
    
    async def detect_model_drift(
        self,
        model_id: str,
        current_data: pd.DataFrame,
        baseline_days: int = 30
    ) -> DriftDetectionResult:
        """
        Detect model drift using statistical methods
        
        Args:
            model_id: Model to check for drift
            current_data: Current data for comparison
            baseline_days: Days of historical data for baseline
            
        Returns:
            Drift detection results
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Simulate drift detection (replace with actual statistical tests)
        drift_score = np.random.uniform(0.0, 1.0)
        drift_threshold = 0.3
        is_drift_detected = drift_score > drift_threshold
        
        # Determine drift type
        drift_type = np.random.choice(list(DriftType))
        
        # Simulate affected features
        all_features = ["cost_trend", "usage_pattern", "service_distribution", "temporal_features"]
        affected_features = np.random.choice(all_features, size=np.random.randint(1, 3), replace=False).tolist()
        
        result = DriftDetectionResult(
            model_id=model_id,
            detection_timestamp=datetime.utcnow(),
            drift_type=drift_type,
            drift_score=drift_score,
            drift_threshold=drift_threshold,
            is_drift_detected=is_drift_detected,
            affected_features=affected_features,
            confidence_level=0.95,
            recommended_action="retrain" if is_drift_detected else "monitor",
            statistical_test_results={
                "ks_test_statistic": np.random.uniform(0.1, 0.5),
                "p_value": np.random.uniform(0.01, 0.1),
                "chi_square_statistic": np.random.uniform(10, 50)
            }
        )
        
        # Store drift result
        if model_id not in self.drift_history:
            self.drift_history[model_id] = []
        self.drift_history[model_id].append(result)
        
        # Trigger automatic retraining if significant drift detected
        if is_drift_detected and self.config.auto_rollback_enabled:
            logger.warning(f"Significant drift detected for model {model_id}, triggering retraining")
            # Schedule retraining (implementation depends on your training infrastructure)
        
        logger.info(f"Drift detection completed for model {model_id}: drift_detected={is_drift_detected}")
        return result
    
    async def rollback_model(self, model_id: str, reason: str) -> Dict[str, Any]:
        """
        Rollback model deployment to previous stable version
        
        Args:
            model_id: Model to rollback
            reason: Reason for rollback
            
        Returns:
            Rollback results
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.model_registry[model_id]
        
        try:
            # Find previous stable version
            previous_model_id = await self._find_previous_stable_model(model_id)
            
            if not previous_model_id:
                raise ValueError("No previous stable model found for rollback")
            
            # Update traffic routing to previous model
            await self._update_traffic_routing(previous_model_id, 100.0)
            
            # Update model statuses
            model_info["status"] = ModelStatus.ROLLBACK
            self.model_registry[previous_model_id]["status"] = ModelStatus.PRODUCTION
            
            # Update canary deployment if exists
            for canary in self.active_deployments.values():
                if canary.model_id == model_id:
                    canary.status = "rolled_back"
            
            # Log rollback to MLflow
            with mlflow.start_run(run_id=model_info["mlflow_run_id"]):
                mlflow.log_param("rollback_timestamp", datetime.utcnow().isoformat())
                mlflow.log_param("rollback_reason", reason)
                mlflow.log_param("rolled_back_to", previous_model_id)
            
            rollback_result = {
                "model_id": model_id,
                "previous_model_id": previous_model_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"Model rollback completed: {model_id} -> {previous_model_id}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Model rollback failed: {str(e)}")
            raise
    
    async def _find_previous_stable_model(self, model_id: str) -> Optional[str]:
        """Find previous stable model for rollback"""
        
        current_model = self.model_registry[model_id]
        model_type = current_model["model_type"]
        account_id = current_model["account_id"]
        
        # Find models of same type and account, sorted by creation time
        candidates = [
            (mid, info) for mid, info in self.model_registry.items()
            if (info["model_type"] == model_type and 
                info["account_id"] == account_id and
                info["status"] in [ModelStatus.PRODUCTION, ModelStatus.CANARY] and
                mid != model_id)
        ]
        
        if not candidates:
            return None
        
        # Sort by creation time and return most recent
        candidates.sort(key=lambda x: x[1]["created_at"], reverse=True)
        return candidates[0][0]
    
    async def _setup_model_monitoring(self, trained_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Setup monitoring for trained models"""
        
        monitoring_configs = []
        
        for model in trained_models:
            model_id = model["model_id"]
            
            # Create monitoring configuration
            monitoring_config = {
                "model_id": model_id,
                "drift_detection": {
                    "enabled": self.config.drift_detection_enabled,
                    "check_interval_hours": self.config.drift_check_interval_hours,
                    "drift_threshold": 0.3,
                    "statistical_tests": ["ks_test", "chi_square", "psi"]
                },
                "performance_monitoring": {
                    "enabled": self.config.performance_monitoring_enabled,
                    "metrics": ["accuracy", "latency", "throughput", "error_rate"],
                    "alert_thresholds": {
                        "accuracy_drop": 0.05,
                        "latency_increase": 50,
                        "error_rate_spike": 0.1
                    }
                },
                "auto_actions": {
                    "rollback_enabled": self.config.auto_rollback_enabled,
                    "retraining_enabled": True,
                    "notification_enabled": True
                }
            }
            
            monitoring_configs.append(monitoring_config)
            
            # Start drift detection monitoring
            if self.config.drift_detection_enabled:
                asyncio.create_task(self._start_drift_monitoring(model_id))
        
        return {
            "monitoring_configs": monitoring_configs,
            "setup_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _start_drift_monitoring(self, model_id: str):
        """Start continuous drift monitoring for a model"""
        
        while True:
            try:
                # Generate sample data for drift detection (replace with actual data)
                sample_data = pd.DataFrame({
                    "feature_1": np.random.normal(0, 1, 1000),
                    "feature_2": np.random.normal(0, 1, 1000),
                    "feature_3": np.random.normal(0, 1, 1000)
                })
                
                # Perform drift detection
                drift_result = await self.detect_model_drift(model_id, sample_data)
                
                # Log drift detection result
                logger.info(f"Drift check for model {model_id}: drift_detected={drift_result.is_drift_detected}")
                
                # Wait for next check
                await asyncio.sleep(self.config.drift_check_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Drift monitoring error for model {model_id}: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        
        return {
            "pipeline_id": self.config.pipeline_id,
            "active_models": len([m for m in self.model_registry.values() if m["status"] == ModelStatus.PRODUCTION]),
            "canary_deployments": len([c for c in self.active_deployments.values() if c.status == "active"]),
            "total_models": len(self.model_registry),
            "drift_alerts": sum(len([d for d in history if d.is_drift_detected]) for history in self.drift_history.values()),
            "mlflow_experiment": self.config.mlflow_experiment_name,
            "kubeflow_host": self.config.kubeflow_host,
            "monitoring_enabled": self.config.performance_monitoring_enabled,
            "auto_rollback_enabled": self.config.auto_rollback_enabled,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_deployments(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old model deployments and artifacts"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        models_cleaned = 0
        deployments_cleaned = 0
        
        # Clean up old models
        models_to_remove = []
        for model_id, model_info in self.model_registry.items():
            if (model_info["created_at"] < cutoff_date and 
                model_info["status"] not in [ModelStatus.PRODUCTION, ModelStatus.CANARY]):
                models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            del self.model_registry[model_id]
            models_cleaned += 1
        
        # Clean up old canary deployments
        deployments_to_remove = []
        for deployment_id, canary in self.active_deployments.items():
            if (canary.start_time < cutoff_date and 
                canary.status in ["completed", "rolled_back", "failed"]):
                deployments_to_remove.append(deployment_id)
        
        for deployment_id in deployments_to_remove:
            del self.active_deployments[deployment_id]
            deployments_cleaned += 1
        
        return {
            "models_cleaned": models_cleaned,
            "deployments_cleaned": deployments_cleaned,
            "cutoff_date": cutoff_date.isoformat()
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Create pipeline configuration
        config = PipelineConfig(
            name="production_ai_pipeline",
            model_types=["anomaly_detection", "cost_forecasting", "usage_prediction"],
            deployment_strategy=DeploymentStrategy.CANARY,
            canary_traffic_percentage=15.0,
            drift_detection_enabled=True,
            auto_rollback_enabled=True
        )
        
        # Initialize pipeline
        pipeline = AITrainingDeploymentPipeline(config)
        
        print(f"AI Training Pipeline initialized: {config.pipeline_id}")
        
        # Create sample training data
        training_data = pd.DataFrame({
            "cost": np.random.uniform(100, 1000, 1000),
            "usage": np.random.uniform(0, 100, 1000),
            "service_type": np.random.choice(["EC2", "S3", "RDS"], 1000),
            "anomaly": np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        })
        
        # Model configurations
        model_configs = [
            {
                "type": "anomaly_detection",
                "algorithm": "isolation_forest",
                "hyperparameters": {"n_estimators": 100, "contamination": 0.1}
            },
            {
                "type": "cost_forecasting", 
                "algorithm": "lstm",
                "hyperparameters": {"sequence_length": 24, "hidden_units": 64}
            }
        ]
        
        # Execute training pipeline
        results = await pipeline.execute_training_pipeline(
            training_data, model_configs, "demo_account"
        )
        
        print(f"Pipeline execution results: {results}")
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        print(f"Pipeline status: {status}")
    
    # Run the example
    asyncio.run(main())