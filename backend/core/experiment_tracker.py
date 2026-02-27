"""
ML Experiment Tracker

This module provides comprehensive experiment tracking and management
for ML workflows including parameter tracking, metric logging,
artifact management, and experiment comparison.
"""

import uuid
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .database import get_db_session
from .exceptions import ExperimentTrackerError, ValidationError

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ArtifactType(Enum):
    """Artifact types"""
    MODEL = "model"
    DATASET = "dataset"
    PLOT = "plot"
    REPORT = "report"
    CONFIG = "config"
    METRICS = "metrics"

@dataclass
class ExperimentMetadata:
    """Experiment metadata"""
    experiment_id: str
    name: str
    description: str
    tags: Dict[str, str]
    created_by: str
    created_at: datetime
    status: ExperimentStatus
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class RunMetadata:
    """Experiment run metadata"""
    run_id: str
    experiment_id: str
    name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metrics is None:
            self.metrics = {}
        if self.tags is None:
            self.tags = {}

@dataclass
class Artifact:
    """Experiment artifact"""
    artifact_id: str
    run_id: str
    name: str
    artifact_type: ArtifactType
    file_path: str
    size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ExperimentTracker:
    """
    Comprehensive experiment tracking system for ML workflows
    with integration to MLflow and custom artifact management.
    """
    
    def __init__(
        self,
        tracking_uri: str = None,
        artifact_store_path: str = "experiments"
    ):
        """
        Initialize Experiment Tracker
        
        Args:
            tracking_uri: MLflow tracking server URI
            artifact_store_path: Path to store experiment artifacts
        """
        # Initialize MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.mlflow_client = MlflowClient()
        self.artifact_store_path = Path(artifact_store_path)
        self.artifact_store_path.mkdir(exist_ok=True)
        
        # In-memory storage for active experiments
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self.runs: Dict[str, RunMetadata] = {}
        self.artifacts: Dict[str, List[Artifact]] = {}  # run_id -> artifacts
        
    async def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Dict[str, str] = None,
        created_by: str = "system"
    ) -> str:
        """
        Create a new experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Experiment tags
            created_by: Creator identifier
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            # Create MLflow experiment
            mlflow_experiment_id = mlflow.create_experiment(
                name=f"{name}_{experiment_id}",
                tags=tags or {}
            )
            
            # Create experiment metadata
            experiment = ExperimentMetadata(
                experiment_id=experiment_id,
                name=name,
                description=description,
                tags=tags or {},
                created_by=created_by,
                created_at=datetime.utcnow(),
                status=ExperimentStatus.CREATED
            )
            
            # Store experiment
            self.experiments[experiment_id] = experiment
            
            # Add MLflow experiment ID to tags
            experiment.tags["mlflow_experiment_id"] = str(mlflow_experiment_id)
            
            logger.info(f"Experiment {experiment_id} created: {name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise ExperimentTrackerError(f"Failed to create experiment: {str(e)}")
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: str = None,
        parameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None
    ) -> str:
        """
        Start a new experiment run
        
        Args:
            experiment_id: Experiment identifier
            run_name: Run name
            parameters: Run parameters
            tags: Run tags
            
        Returns:
            Run ID
        """
        try:
            if experiment_id not in self.experiments:
                raise ExperimentTrackerError(f"Experiment {experiment_id} not found")
            
            run_id = str(uuid.uuid4())
            experiment = self.experiments[experiment_id]
            
            # Start MLflow run
            mlflow_experiment_id = experiment.tags.get("mlflow_experiment_id")
            mlflow_run = mlflow.start_run(
                experiment_id=mlflow_experiment_id,
                run_name=run_name or f"run_{run_id}"
            )
            
            # Log parameters to MLflow
            if parameters:
                mlflow.log_params(parameters)
            
            # Log tags to MLflow
            if tags:
                mlflow.set_tags(tags)
            
            # Create run metadata
            run = RunMetadata(
                run_id=run_id,
                experiment_id=experiment_id,
                name=run_name or f"run_{run_id}",
                status=ExperimentStatus.RUNNING,
                start_time=datetime.utcnow(),
                end_time=None,
                parameters=parameters or {},
                metrics={},
                tags=tags or {}
            )
            
            # Store run
            self.runs[run_id] = run
            self.artifacts[run_id] = []
            
            # Add MLflow run ID to tags
            run.tags["mlflow_run_id"] = mlflow_run.info.run_id
            
            logger.info(f"Run {run_id} started for experiment {experiment_id}")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start run: {str(e)}")
            raise ExperimentTrackerError(f"Failed to start run: {str(e)}")
    
    async def log_parameter(self, run_id: str, key: str, value: Any):
        """
        Log a parameter for a run
        
        Args:
            run_id: Run identifier
            key: Parameter name
            value: Parameter value
        """
        try:
            if run_id not in self.runs:
                raise ExperimentTrackerError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
            run.parameters[key] = value
            
            # Log to MLflow if run is active
            mlflow_run_id = run.tags.get("mlflow_run_id")
            if mlflow_run_id and mlflow.active_run():
                mlflow.log_param(key, value)
            
        except Exception as e:
            logger.error(f"Failed to log parameter: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log parameter: {str(e)}")
    
    async def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = None,
        timestamp: datetime = None
    ):
        """
        Log a metric for a run
        
        Args:
            run_id: Run identifier
            key: Metric name
            value: Metric value
            step: Step number
            timestamp: Timestamp
        """
        try:
            if run_id not in self.runs:
                raise ExperimentTrackerError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
            run.metrics[key] = value
            
            # Log to MLflow if run is active
            mlflow_run_id = run.tags.get("mlflow_run_id")
            if mlflow_run_id and mlflow.active_run():
                mlflow.log_metric(key, value, step=step)
            
        except Exception as e:
            logger.error(f"Failed to log metric: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log metric: {str(e)}")
    
    async def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int = None):
        """
        Log multiple metrics for a run
        
        Args:
            run_id: Run identifier
            metrics: Dictionary of metrics
            step: Step number
        """
        for key, value in metrics.items():
            await self.log_metric(run_id, key, value, step)
    
    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_type: ArtifactType,
        name: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log an artifact for a run
        
        Args:
            run_id: Run identifier
            artifact_path: Path to artifact file
            artifact_type: Type of artifact
            name: Artifact name
            metadata: Artifact metadata
            
        Returns:
            Artifact ID
        """
        try:
            if run_id not in self.runs:
                raise ExperimentTrackerError(f"Run {run_id} not found")
            
            artifact_id = str(uuid.uuid4())
            artifact_file = Path(artifact_path)
            
            if not artifact_file.exists():
                raise ExperimentTrackerError(f"Artifact file not found: {artifact_path}")
            
            # Copy artifact to artifact store
            run_artifact_dir = self.artifact_store_path / run_id
            run_artifact_dir.mkdir(exist_ok=True)
            
            stored_path = run_artifact_dir / f"{artifact_id}_{artifact_file.name}"
            stored_path.write_bytes(artifact_file.read_bytes())
            
            # Create artifact metadata
            artifact = Artifact(
                artifact_id=artifact_id,
                run_id=run_id,
                name=name or artifact_file.name,
                artifact_type=artifact_type,
                file_path=str(stored_path),
                size_bytes=stored_path.stat().st_size,
                created_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Store artifact
            self.artifacts[run_id].append(artifact)
            
            # Log to MLflow
            run = self.runs[run_id]
            mlflow_run_id = run.tags.get("mlflow_run_id")
            if mlflow_run_id and mlflow.active_run():
                mlflow.log_artifact(artifact_path)
            
            logger.info(f"Artifact {artifact_id} logged for run {run_id}")
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log artifact: {str(e)}")
    
    async def log_model(
        self,
        run_id: str,
        model: Any,
        model_name: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log a trained model as an artifact
        
        Args:
            run_id: Run identifier
            model: Trained model object
            model_name: Model name
            metadata: Model metadata
            
        Returns:
            Artifact ID
        """
        try:
            # Save model to temporary file
            run_artifact_dir = self.artifact_store_path / run_id
            run_artifact_dir.mkdir(exist_ok=True)
            
            model_file = run_artifact_dir / f"{model_name}.pkl"
            
            # Save model using pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Log as artifact
            artifact_id = await self.log_artifact(
                run_id=run_id,
                artifact_path=str(model_file),
                artifact_type=ArtifactType.MODEL,
                name=model_name,
                metadata=metadata
            )
            
            # Log to MLflow
            run = self.runs[run_id]
            mlflow_run_id = run.tags.get("mlflow_run_id")
            if mlflow_run_id and mlflow.active_run():
                try:
                    mlflow.sklearn.log_model(model, model_name)
                except Exception as e:
                    logger.warning(f"Failed to log model to MLflow: {str(e)}")
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log model: {str(e)}")
    
    async def log_dataset(
        self,
        run_id: str,
        dataset: pd.DataFrame,
        dataset_name: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log a dataset as an artifact
        
        Args:
            run_id: Run identifier
            dataset: Dataset DataFrame
            dataset_name: Dataset name
            metadata: Dataset metadata
            
        Returns:
            Artifact ID
        """
        try:
            # Save dataset to CSV
            run_artifact_dir = self.artifact_store_path / run_id
            run_artifact_dir.mkdir(exist_ok=True)
            
            dataset_file = run_artifact_dir / f"{dataset_name}.csv"
            dataset.to_csv(dataset_file, index=False)
            
            # Add dataset statistics to metadata
            dataset_metadata = metadata or {}
            dataset_metadata.update({
                "shape": dataset.shape,
                "columns": list(dataset.columns),
                "dtypes": dataset.dtypes.to_dict(),
                "memory_usage": dataset.memory_usage(deep=True).sum()
            })
            
            # Log as artifact
            artifact_id = await self.log_artifact(
                run_id=run_id,
                artifact_path=str(dataset_file),
                artifact_type=ArtifactType.DATASET,
                name=dataset_name,
                metadata=dataset_metadata
            )
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to log dataset: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log dataset: {str(e)}")
    
    async def log_plot(
        self,
        run_id: str,
        figure: plt.Figure,
        plot_name: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log a matplotlib plot as an artifact
        
        Args:
            run_id: Run identifier
            figure: Matplotlib figure
            plot_name: Plot name
            metadata: Plot metadata
            
        Returns:
            Artifact ID
        """
        try:
            # Save plot to file
            run_artifact_dir = self.artifact_store_path / run_id
            run_artifact_dir.mkdir(exist_ok=True)
            
            plot_file = run_artifact_dir / f"{plot_name}.png"
            figure.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Log as artifact
            artifact_id = await self.log_artifact(
                run_id=run_id,
                artifact_path=str(plot_file),
                artifact_type=ArtifactType.PLOT,
                name=plot_name,
                metadata=metadata
            )
            
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to log plot: {str(e)}")
            raise ExperimentTrackerError(f"Failed to log plot: {str(e)}")
    
    async def end_run(self, run_id: str, status: ExperimentStatus = ExperimentStatus.COMPLETED):
        """
        End an experiment run
        
        Args:
            run_id: Run identifier
            status: Final run status
        """
        try:
            if run_id not in self.runs:
                raise ExperimentTrackerError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
            run.status = status
            run.end_time = datetime.utcnow()
            
            # End MLflow run
            mlflow_run_id = run.tags.get("mlflow_run_id")
            if mlflow_run_id and mlflow.active_run():
                mlflow.end_run(status=status.value.upper())
            
            logger.info(f"Run {run_id} ended with status: {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to end run: {str(e)}")
            raise ExperimentTrackerError(f"Failed to end run: {str(e)}")
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata"""
        return self.experiments.get(experiment_id)
    
    async def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get run metadata"""
        return self.runs.get(run_id)
    
    async def get_run_artifacts(self, run_id: str) -> List[Artifact]:
        """Get all artifacts for a run"""
        return self.artifacts.get(run_id, [])
    
    async def list_experiments(self, created_by: str = None) -> List[ExperimentMetadata]:
        """List all experiments"""
        experiments = list(self.experiments.values())
        
        if created_by:
            experiments = [exp for exp in experiments if exp.created_by == created_by]
        
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    async def list_runs(self, experiment_id: str = None) -> List[RunMetadata]:
        """List runs for an experiment"""
        runs = list(self.runs.values())
        
        if experiment_id:
            runs = [run for run in runs if run.experiment_id == experiment_id]
        
        return sorted(runs, key=lambda x: x.start_time, reverse=True)
    
    async def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        try:
            if len(run_ids) < 2:
                raise ValidationError("At least 2 runs required for comparison")
            
            runs = []
            for run_id in run_ids:
                if run_id not in self.runs:
                    raise ExperimentTrackerError(f"Run {run_id} not found")
                runs.append(self.runs[run_id])
            
            # Compare parameters
            all_params = set()
            for run in runs:
                all_params.update(run.parameters.keys())
            
            parameter_comparison = {}
            for param in all_params:
                parameter_comparison[param] = {
                    run.run_id: run.parameters.get(param, "N/A")
                    for run in runs
                }
            
            # Compare metrics
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())
            
            metric_comparison = {}
            for metric in all_metrics:
                metric_comparison[metric] = {
                    run.run_id: run.metrics.get(metric, None)
                    for run in runs
                }
            
            # Find best performing run for each metric
            best_runs = {}
            for metric in all_metrics:
                metric_values = [
                    (run.run_id, run.metrics.get(metric, float('-inf')))
                    for run in runs
                    if run.metrics.get(metric) is not None
                ]
                
                if metric_values:
                    best_run_id, best_value = max(metric_values, key=lambda x: x[1])
                    best_runs[metric] = {
                        "run_id": best_run_id,
                        "value": best_value
                    }
            
            comparison = {
                "run_ids": run_ids,
                "run_names": {run.run_id: run.name for run in runs},
                "parameter_comparison": parameter_comparison,
                "metric_comparison": metric_comparison,
                "best_runs": best_runs,
                "comparison_date": datetime.utcnow()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            raise ExperimentTrackerError(f"Failed to compare runs: {str(e)}")
    
    async def search_runs(
        self,
        experiment_id: str = None,
        filter_string: str = None,
        order_by: str = None,
        max_results: int = 100
    ) -> List[RunMetadata]:
        """
        Search runs with filters
        
        Args:
            experiment_id: Experiment ID to filter by
            filter_string: Filter string (e.g., "metrics.accuracy > 0.8")
            order_by: Order by clause (e.g., "metrics.accuracy DESC")
            max_results: Maximum number of results
            
        Returns:
            List of matching runs
        """
        try:
            runs = list(self.runs.values())
            
            # Filter by experiment
            if experiment_id:
                runs = [run for run in runs if run.experiment_id == experiment_id]
            
            # Apply filter string (simplified implementation)
            if filter_string:
                # This is a simplified filter implementation
                # In production, you'd want a more robust query parser
                filtered_runs = []
                for run in runs:
                    if self._evaluate_filter(run, filter_string):
                        filtered_runs.append(run)
                runs = filtered_runs
            
            # Apply ordering
            if order_by:
                reverse = "DESC" in order_by.upper()
                if "metrics." in order_by:
                    metric_name = order_by.split("metrics.")[1].split()[0]
                    runs.sort(
                        key=lambda x: x.metrics.get(metric_name, float('-inf')),
                        reverse=reverse
                    )
                elif "start_time" in order_by:
                    runs.sort(key=lambda x: x.start_time, reverse=reverse)
            
            # Limit results
            return runs[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search runs: {str(e)}")
            raise ExperimentTrackerError(f"Failed to search runs: {str(e)}")
    
    def _evaluate_filter(self, run: RunMetadata, filter_string: str) -> bool:
        """Evaluate filter string against a run (simplified implementation)"""
        try:
            # This is a very basic implementation
            # In production, you'd want a proper query parser
            
            if "metrics." in filter_string:
                # Extract metric name and condition
                parts = filter_string.split()
                if len(parts) >= 3:
                    metric_path = parts[0]  # e.g., "metrics.accuracy"
                    operator = parts[1]     # e.g., ">"
                    value = float(parts[2]) # e.g., "0.8"
                    
                    metric_name = metric_path.split("metrics.")[1]
                    run_value = run.metrics.get(metric_name)
                    
                    if run_value is None:
                        return False
                    
                    if operator == ">":
                        return run_value > value
                    elif operator == ">=":
                        return run_value >= value
                    elif operator == "<":
                        return run_value < value
                    elif operator == "<=":
                        return run_value <= value
                    elif operator == "=":
                        return run_value == value
            
            return True
            
        except Exception:
            return True  # If filter evaluation fails, include the run
    
    async def delete_experiment(self, experiment_id: str):
        """Delete an experiment and all its runs"""
        try:
            if experiment_id not in self.experiments:
                raise ExperimentTrackerError(f"Experiment {experiment_id} not found")
            
            # Delete all runs for this experiment
            runs_to_delete = [
                run_id for run_id, run in self.runs.items()
                if run.experiment_id == experiment_id
            ]
            
            for run_id in runs_to_delete:
                await self.delete_run(run_id)
            
            # Delete experiment
            del self.experiments[experiment_id]
            
            logger.info(f"Experiment {experiment_id} deleted")
            
        except Exception as e:
            logger.error(f"Failed to delete experiment: {str(e)}")
            raise ExperimentTrackerError(f"Failed to delete experiment: {str(e)}")
    
    async def delete_run(self, run_id: str):
        """Delete a run and its artifacts"""
        try:
            if run_id not in self.runs:
                raise ExperimentTrackerError(f"Run {run_id} not found")
            
            # Delete artifacts
            if run_id in self.artifacts:
                for artifact in self.artifacts[run_id]:
                    artifact_path = Path(artifact.file_path)
                    if artifact_path.exists():
                        artifact_path.unlink()
                
                del self.artifacts[run_id]
            
            # Delete run directory
            run_artifact_dir = self.artifact_store_path / run_id
            if run_artifact_dir.exists():
                import shutil
                shutil.rmtree(run_artifact_dir)
            
            # Delete run
            del self.runs[run_id]
            
            logger.info(f"Run {run_id} deleted")
            
        except Exception as e:
            logger.error(f"Failed to delete run: {str(e)}")
            raise ExperimentTrackerError(f"Failed to delete run: {str(e)}")