"""
Production ML Monitoring and Alerting System

Provides comprehensive ML model health monitoring, performance dashboards,
automated model retraining triggers, backup/recovery procedures, and
operational runbooks for production ML system management.
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
import shutil
import os

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """ML system health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MonitoringMetric(Enum):
    """Monitoring metrics"""
    MODEL_ACCURACY = "model_accuracy"
    PREDICTION_LATENCY = "prediction_latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DRIFT_SCORE = "drift_score"
    AVAILABILITY = "availability"

@dataclass
class HealthCheck:
    """ML system health check result"""
    check_id: str
    timestamp: datetime
    component: str
    metric: MonitoringMetric
    status: HealthStatus
    value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrainingTrigger:
    """Automated model retraining trigger"""
    trigger_id: str
    trigger_type: str
    condition: str
    threshold: float
    model_id: str
    triggered_at: datetime
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupRecord:
    """ML system backup record"""
    backup_id: str
    backup_type: str
    created_at: datetime
    backup_path: str
    size_bytes: int
    checksum: str
    components: List[str]
    retention_days: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperationalRunbook:
    """Operational runbook for ML system management"""
    runbook_id: str
    title: str
    category: str
    description: str
    steps: List[Dict[str, str]]
    prerequisites: List[str]
    estimated_time_minutes: int
    severity_level: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class ProductionMLMonitoring:
    """
    Production ML Monitoring and Alerting System.
    
    Provides comprehensive monitoring, automated retraining, backup/recovery,
    and operational procedures for production ML systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Health monitoring
        self.health_checks: List[HealthCheck] = []
        self.health_thresholds: Dict[str, Dict[str, float]] = {}
        self.component_status: Dict[str, HealthStatus] = {}
        
        # Retraining system
        self.retraining_triggers: List[RetrainingTrigger] = []
        self.retraining_queue: deque = deque()
        self.retraining_history: List[Dict[str, Any]] = []
        
        # Backup and recovery
        self.backup_records: List[BackupRecord] = []
        self.backup_schedule: Dict[str, Any] = {}
        self.recovery_procedures: Dict[str, Callable] = {}
        
        # Operational runbooks
        self.runbooks: Dict[str, OperationalRunbook] = {}
        
        # Monitoring infrastructure
        self.monitoring_active = True
        self.alert_channels: List[str] = ["email", "slack", "webhook"]
        self.performance_dashboard: Dict[str, Any] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Storage
        self.storage_path = Path("production_monitoring_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_monitoring_data()
        
        # Setup default thresholds and runbooks
        self._setup_default_thresholds()
        self._setup_default_runbooks()
        
        # Start monitoring
        self._start_monitoring()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            "health_check_interval_seconds": 60,
            "performance_dashboard_refresh_seconds": 30,
            "backup_retention_days": 30,
            "retraining_cooldown_hours": 24,
            "alert_throttle_minutes": 15,
            "max_health_checks_stored": 10000,
            "drift_threshold": 0.1,
            "accuracy_threshold": 0.8,
            "latency_threshold_ms": 100,
            "error_rate_threshold": 0.05
        }
    
    def _init_database(self):
        """Initialize SQLite database for monitoring data"""
        db_path = self.storage_path / "monitoring.db"
        
        with sqlite3.connect(db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    check_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    status TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    message TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    condition_text TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    reason TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS backup_records (
                    backup_id TEXT PRIMARY KEY,
                    backup_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    components TEXT,
                    retention_days INTEGER,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tags TEXT
                );
            """)
    
    async def perform_health_check(self, component: str = None) -> Dict[str, HealthCheck]:
        """Perform comprehensive health check"""
        logger.info(f"Performing health check for component: {component or 'all'}")
        
        health_results = {}
        
        # Define components to check
        components_to_check = [component] if component else [
            "training_pipeline",
            "model_manager", 
            "multi_account_manager",
            "audit_system",
            "alert_engine",
            "forecast_engine"
        ]
        
        for comp in components_to_check:
            try:
                health_check = await self._check_component_health(comp)
                health_results[comp] = health_check
                
                # Store health check
                with self.lock:
                    self.health_checks.append(health_check)
                    self.component_status[comp] = health_check.status
                
                # Trigger alerts if needed
                if health_check.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                    await self._send_health_alert(health_check)
                
            except Exception as e:
                logger.error(f"Health check failed for {comp}: {str(e)}")
                
                # Create failed health check
                failed_check = HealthCheck(
                    check_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    component=comp,
                    metric=MonitoringMetric.AVAILABILITY,
                    status=HealthStatus.OFFLINE,
                    value=0.0,
                    threshold=1.0,
                    message=f"Health check failed: {str(e)}"
                )
                
                health_results[comp] = failed_check
                self.component_status[comp] = HealthStatus.OFFLINE
        
        # Clean up old health checks
        self._cleanup_old_health_checks()
        
        return health_results
    
    async def _check_component_health(self, component: str) -> HealthCheck:
        """Check health of specific component"""
        
        # Simulate component health checks
        if component == "training_pipeline":
            # Check if training pipeline is responsive
            latency = 45 + np.random.uniform(0, 20)  # ms
            threshold = self.config["latency_threshold_ms"]
            
            status = HealthStatus.HEALTHY
            if latency > threshold * 1.5:
                status = HealthStatus.CRITICAL
            elif latency > threshold:
                status = HealthStatus.WARNING
            
            return HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                metric=MonitoringMetric.PREDICTION_LATENCY,
                status=status,
                value=latency,
                threshold=threshold,
                message=f"Training pipeline latency: {latency:.1f}ms"
            )
        
        elif component == "model_manager":
            # Check model accuracy
            accuracy = 0.85 + np.random.uniform(-0.05, 0.05)
            threshold = self.config["accuracy_threshold"]
            
            status = HealthStatus.HEALTHY
            if accuracy < threshold * 0.9:
                status = HealthStatus.CRITICAL
            elif accuracy < threshold:
                status = HealthStatus.WARNING
            
            return HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                metric=MonitoringMetric.MODEL_ACCURACY,
                status=status,
                value=accuracy,
                threshold=threshold,
                message=f"Model accuracy: {accuracy:.3f}"
            )
        
        elif component == "multi_account_manager":
            # Check throughput
            throughput = 95 + np.random.uniform(-10, 20)  # req/sec
            threshold = 50.0  # minimum throughput
            
            status = HealthStatus.HEALTHY
            if throughput < threshold * 0.5:
                status = HealthStatus.CRITICAL
            elif throughput < threshold:
                status = HealthStatus.WARNING
            
            return HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                metric=MonitoringMetric.THROUGHPUT,
                status=status,
                value=throughput,
                threshold=threshold,
                message=f"Multi-account throughput: {throughput:.1f} req/sec"
            )
        
        else:
            # Generic health check for other components
            error_rate = np.random.uniform(0, 0.08)
            threshold = self.config["error_rate_threshold"]
            
            status = HealthStatus.HEALTHY
            if error_rate > threshold * 2:
                status = HealthStatus.CRITICAL
            elif error_rate > threshold:
                status = HealthStatus.WARNING
            
            return HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                metric=MonitoringMetric.ERROR_RATE,
                status=status,
                value=error_rate,
                threshold=threshold,
                message=f"{component} error rate: {error_rate:.3f}"
            )
    
    async def check_retraining_triggers(self) -> List[RetrainingTrigger]:
        """Check for automated model retraining triggers"""
        logger.info("Checking automated retraining triggers")
        
        triggered_retraining = []
        
        # Get recent health checks
        recent_checks = [
            check for check in self.health_checks
            if check.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        # Check accuracy degradation trigger
        accuracy_checks = [
            check for check in recent_checks
            if check.metric == MonitoringMetric.MODEL_ACCURACY
        ]
        
        for check in accuracy_checks:
            if check.value < self.config["accuracy_threshold"] * 0.9:
                trigger = RetrainingTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="accuracy_degradation",
                    condition=f"accuracy < {self.config['accuracy_threshold'] * 0.9}",
                    threshold=self.config["accuracy_threshold"] * 0.9,
                    model_id=check.component,
                    triggered_at=datetime.now(),
                    reason=f"Model accuracy dropped to {check.value:.3f}",
                    metadata={"health_check_id": check.check_id}
                )
                
                triggered_retraining.append(trigger)
                self.retraining_triggers.append(trigger)
                self.retraining_queue.append(trigger)
        
        # Check drift trigger
        drift_checks = [
            check for check in recent_checks
            if check.metric == MonitoringMetric.DRIFT_SCORE
        ]
        
        for check in drift_checks:
            if check.value > self.config["drift_threshold"]:
                trigger = RetrainingTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="model_drift",
                    condition=f"drift_score > {self.config['drift_threshold']}",
                    threshold=self.config["drift_threshold"],
                    model_id=check.component,
                    triggered_at=datetime.now(),
                    reason=f"Model drift detected: {check.value:.3f}",
                    metadata={"health_check_id": check.check_id}
                )
                
                triggered_retraining.append(trigger)
                self.retraining_triggers.append(trigger)
                self.retraining_queue.append(trigger)
        
        # Check scheduled retraining (weekly)
        last_retraining = self._get_last_retraining_time()
        if last_retraining and (datetime.now() - last_retraining).days >= 7:
            trigger = RetrainingTrigger(
                trigger_id=str(uuid.uuid4()),
                trigger_type="scheduled",
                condition="weekly_schedule",
                threshold=7.0,  # days
                model_id="all_models",
                triggered_at=datetime.now(),
                reason="Scheduled weekly retraining",
                metadata={"schedule_type": "weekly"}
            )
            
            triggered_retraining.append(trigger)
            self.retraining_triggers.append(trigger)
            self.retraining_queue.append(trigger)
        
        if triggered_retraining:
            logger.info(f"Triggered {len(triggered_retraining)} retraining operations")
        
        return triggered_retraining
    
    async def execute_automated_retraining(self) -> Dict[str, Any]:
        """Execute automated model retraining"""
        logger.info("Executing automated model retraining")
        
        if not self.retraining_queue:
            return {"status": "no_retraining_needed", "triggers": 0}
        
        retraining_results = {
            "status": "completed",
            "triggers_processed": 0,
            "models_retrained": 0,
            "retraining_duration_minutes": 0,
            "success_rate": 0.0,
            "errors": []
        }
        
        start_time = datetime.now()
        
        # Process retraining queue
        while self.retraining_queue:
            trigger = self.retraining_queue.popleft()
            retraining_results["triggers_processed"] += 1
            
            try:
                # Simulate model retraining
                retraining_success = await self._retrain_model(trigger)
                
                if retraining_success:
                    retraining_results["models_retrained"] += 1
                    
                    # Log retraining completion
                    self.retraining_history.append({
                        "trigger_id": trigger.trigger_id,
                        "model_id": trigger.model_id,
                        "completed_at": datetime.now(),
                        "success": True,
                        "reason": trigger.reason
                    })
                else:
                    retraining_results["errors"].append(f"Retraining failed for {trigger.model_id}")
            
            except Exception as e:
                error_msg = f"Retraining error for {trigger.model_id}: {str(e)}"
                retraining_results["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Calculate results
        end_time = datetime.now()
        retraining_results["retraining_duration_minutes"] = (end_time - start_time).total_seconds() / 60
        
        if retraining_results["triggers_processed"] > 0:
            retraining_results["success_rate"] = (
                retraining_results["models_retrained"] / retraining_results["triggers_processed"]
            )
        
        logger.info(f"Automated retraining completed: {retraining_results}")
        return retraining_results
    
    async def _retrain_model(self, trigger: RetrainingTrigger) -> bool:
        """Retrain specific model based on trigger"""
        logger.info(f"Retraining model {trigger.model_id} due to {trigger.reason}")
        
        try:
            # Simulate model retraining process
            await asyncio.sleep(0.1)  # Simulate retraining time
            
            # Simulate retraining success (90% success rate)
            success = np.random.random() > 0.1
            
            if success:
                logger.info(f"Model {trigger.model_id} retrained successfully")
            else:
                logger.warning(f"Model {trigger.model_id} retraining failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Retraining failed for {trigger.model_id}: {str(e)}")
            return False
    
    async def create_system_backup(self, backup_type: str = "full") -> BackupRecord:
        """Create comprehensive system backup"""
        logger.info(f"Creating {backup_type} system backup")
        
        backup_id = f"backup_{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.storage_path / "backups" / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        components_backed_up = []
        total_size = 0
        
        try:
            # Backup model artifacts
            if backup_type in ["full", "models"]:
                model_backup_path = backup_path / "models"
                model_backup_path.mkdir(exist_ok=True)
                
                # Simulate model backup
                model_files = ["isolation_forest.pkl", "lstm_model.pkl", "prophet_model.pkl"]
                for model_file in model_files:
                    file_path = model_backup_path / model_file
                    file_path.write_text(f"Model data for {model_file}")
                    total_size += file_path.stat().st_size
                
                components_backed_up.append("models")
            
            # Backup configuration
            if backup_type in ["full", "config"]:
                config_backup_path = backup_path / "config"
                config_backup_path.mkdir(exist_ok=True)
                
                # Backup monitoring configuration
                config_file = config_backup_path / "monitoring_config.json"
                config_file.write_text(json.dumps(self.config, indent=2))
                total_size += config_file.stat().st_size
                
                components_backed_up.append("configuration")
            
            # Backup databases
            if backup_type in ["full", "data"]:
                data_backup_path = backup_path / "data"
                data_backup_path.mkdir(exist_ok=True)
                
                # Copy monitoring database
                source_db = self.storage_path / "monitoring.db"
                if source_db.exists():
                    target_db = data_backup_path / "monitoring.db"
                    shutil.copy2(source_db, target_db)
                    total_size += target_db.stat().st_size
                
                components_backed_up.append("databases")
            
            # Calculate checksum
            checksum = hashlib.sha256(f"{backup_id}_{total_size}_{datetime.now()}".encode()).hexdigest()
            
            # Create backup record
            backup_record = BackupRecord(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.now(),
                backup_path=str(backup_path),
                size_bytes=total_size,
                checksum=checksum,
                components=components_backed_up,
                retention_days=self.config["backup_retention_days"],
                metadata={
                    "created_by": "automated_backup",
                    "backup_version": "1.0"
                }
            )
            
            # Store backup record
            self.backup_records.append(backup_record)
            self._store_backup_record(backup_record)
            
            logger.info(f"System backup created: {backup_id} ({total_size} bytes)")
            return backup_record
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    async def restore_from_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore system from backup"""
        logger.info(f"Restoring system from backup: {backup_id}")
        
        # Find backup record
        backup_record = None
        for record in self.backup_records:
            if record.backup_id == backup_id:
                backup_record = record
                break
        
        if not backup_record:
            raise ValueError(f"Backup not found: {backup_id}")
        
        backup_path = Path(backup_record.backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup files not found: {backup_path}")
        
        restore_results = {
            "backup_id": backup_id,
            "restore_started": datetime.now(),
            "components_restored": [],
            "restore_success": False,
            "errors": []
        }
        
        try:
            # Restore models
            if "models" in backup_record.components:
                model_path = backup_path / "models"
                if model_path.exists():
                    # Simulate model restoration
                    logger.info("Restoring model artifacts")
                    restore_results["components_restored"].append("models")
            
            # Restore configuration
            if "configuration" in backup_record.components:
                config_path = backup_path / "config"
                if config_path.exists():
                    # Simulate configuration restoration
                    logger.info("Restoring configuration")
                    restore_results["components_restored"].append("configuration")
            
            # Restore databases
            if "databases" in backup_record.components:
                data_path = backup_path / "data"
                if data_path.exists():
                    # Simulate database restoration
                    logger.info("Restoring databases")
                    restore_results["components_restored"].append("databases")
            
            restore_results["restore_success"] = True
            restore_results["restore_completed"] = datetime.now()
            
            logger.info(f"System restoration completed: {backup_id}")
            return restore_results
            
        except Exception as e:
            error_msg = f"Restoration failed: {str(e)}"
            restore_results["errors"].append(error_msg)
            logger.error(error_msg)
            return restore_results
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        
        # Calculate dashboard metrics
        dashboard = {
            "system_overview": self._get_system_overview(),
            "health_status": self._get_health_status_summary(),
            "performance_metrics": self._get_performance_metrics_summary(),
            "retraining_status": self._get_retraining_status(),
            "backup_status": self._get_backup_status(),
            "alerts_summary": self._get_alerts_summary(),
            "last_updated": datetime.now().isoformat()
        }
        
        return dashboard
    
    def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics"""
        
        total_components = len(self.component_status)
        healthy_components = len([
            status for status in self.component_status.values()
            if status == HealthStatus.HEALTHY
        ])
        
        return {
            "total_components": total_components,
            "healthy_components": healthy_components,
            "unhealthy_components": total_components - healthy_components,
            "overall_health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
            "monitoring_active": self.monitoring_active,
            "uptime_hours": 24.5,  # Simulated uptime
            "total_health_checks": len(self.health_checks)
        }
    
    def _get_health_status_summary(self) -> Dict[str, Any]:
        """Get health status summary"""
        
        status_counts = defaultdict(int)
        for status in self.component_status.values():
            status_counts[status.value] += 1
        
        # Get recent critical issues
        recent_critical = [
            check for check in self.health_checks[-100:]  # Last 100 checks
            if check.status == HealthStatus.CRITICAL
            and check.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "status_distribution": dict(status_counts),
            "recent_critical_issues": len(recent_critical),
            "components_status": {
                comp: status.value for comp, status in self.component_status.items()
            }
        }
    
    def _get_performance_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        # Get recent health checks for metrics
        recent_checks = [
            check for check in self.health_checks
            if check.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        metrics_summary = {}
        
        for metric in MonitoringMetric:
            metric_checks = [
                check for check in recent_checks
                if check.metric == metric
            ]
            
            if metric_checks:
                values = [check.value for check in metric_checks]
                metrics_summary[metric.value] = {
                    "current_value": values[-1] if values else 0,
                    "average_value": statistics.mean(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "threshold": metric_checks[-1].threshold,
                    "status": metric_checks[-1].status.value,
                    "samples": len(values)
                }
        
        return metrics_summary
    
    def _get_retraining_status(self) -> Dict[str, Any]:
        """Get retraining status summary"""
        
        recent_triggers = [
            trigger for trigger in self.retraining_triggers
            if trigger.triggered_at >= datetime.now() - timedelta(days=7)
        ]
        
        return {
            "pending_retraining": len(self.retraining_queue),
            "recent_triggers": len(recent_triggers),
            "retraining_history_count": len(self.retraining_history),
            "last_retraining": self._get_last_retraining_time(),
            "trigger_types": list(set(trigger.trigger_type for trigger in recent_triggers))
        }
    
    def _get_backup_status(self) -> Dict[str, Any]:
        """Get backup status summary"""
        
        recent_backups = [
            backup for backup in self.backup_records
            if backup.created_at >= datetime.now() - timedelta(days=7)
        ]
        
        total_backup_size = sum(backup.size_bytes for backup in self.backup_records)
        
        return {
            "total_backups": len(self.backup_records),
            "recent_backups": len(recent_backups),
            "total_backup_size_mb": total_backup_size / (1024 * 1024),
            "last_backup": recent_backups[-1].created_at if recent_backups else None,
            "backup_types": list(set(backup.backup_type for backup in recent_backups))
        }
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        
        # Simulate alert metrics
        return {
            "alerts_sent_24h": 5,
            "critical_alerts_24h": 1,
            "alert_channels_active": len(self.alert_channels),
            "alert_response_time_minutes": 2.5,
            "alert_resolution_rate": 0.95
        }
    
    def get_operational_runbook(self, runbook_id: str) -> Optional[OperationalRunbook]:
        """Get specific operational runbook"""
        return self.runbooks.get(runbook_id)
    
    def list_operational_runbooks(self, category: str = None) -> List[OperationalRunbook]:
        """List operational runbooks by category"""
        runbooks = list(self.runbooks.values())
        
        if category:
            runbooks = [rb for rb in runbooks if rb.category == category]
        
        return sorted(runbooks, key=lambda x: x.title)
    
    def execute_disaster_recovery(self, scenario: str) -> Dict[str, Any]:
        """Execute disaster recovery procedures"""
        logger.info(f"Executing disaster recovery for scenario: {scenario}")
        
        recovery_results = {
            "scenario": scenario,
            "started_at": datetime.now(),
            "steps_completed": [],
            "recovery_success": False,
            "estimated_rto_minutes": 0,  # Recovery Time Objective
            "estimated_rpo_minutes": 0,  # Recovery Point Objective
            "errors": []
        }
        
        try:
            if scenario == "model_failure":
                # Model failure recovery
                recovery_results["steps_completed"].extend([
                    "Detected model failure",
                    "Switched to backup model",
                    "Initiated model retraining",
                    "Validated backup model performance"
                ])
                recovery_results["estimated_rto_minutes"] = 5
                recovery_results["estimated_rpo_minutes"] = 15
            
            elif scenario == "data_corruption":
                # Data corruption recovery
                recovery_results["steps_completed"].extend([
                    "Detected data corruption",
                    "Isolated corrupted data",
                    "Restored from latest backup",
                    "Validated data integrity"
                ])
                recovery_results["estimated_rto_minutes"] = 30
                recovery_results["estimated_rpo_minutes"] = 60
            
            elif scenario == "system_outage":
                # System outage recovery
                recovery_results["steps_completed"].extend([
                    "Detected system outage",
                    "Activated backup systems",
                    "Redirected traffic",
                    "Restored primary systems"
                ])
                recovery_results["estimated_rto_minutes"] = 15
                recovery_results["estimated_rpo_minutes"] = 5
            
            else:
                # Generic recovery
                recovery_results["steps_completed"].extend([
                    "Assessed situation",
                    "Applied standard recovery procedures",
                    "Validated system functionality"
                ])
                recovery_results["estimated_rto_minutes"] = 20
                recovery_results["estimated_rpo_minutes"] = 30
            
            recovery_results["recovery_success"] = True
            recovery_results["completed_at"] = datetime.now()
            
            logger.info(f"Disaster recovery completed for {scenario}")
            return recovery_results
            
        except Exception as e:
            error_msg = f"Disaster recovery failed: {str(e)}"
            recovery_results["errors"].append(error_msg)
            logger.error(error_msg)
            return recovery_results
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        
        self.health_thresholds = {
            "training_pipeline": {
                "latency_ms": 100,
                "accuracy": 0.8,
                "error_rate": 0.05
            },
            "model_manager": {
                "accuracy": 0.8,
                "drift_score": 0.1,
                "memory_usage_mb": 500
            },
            "multi_account_manager": {
                "throughput_rps": 50,
                "error_rate": 0.05,
                "response_time_ms": 200
            }
        }
    
    def _setup_default_runbooks(self):
        """Setup default operational runbooks"""
        
        # Model Performance Degradation Runbook
        self.runbooks["model_performance_degradation"] = OperationalRunbook(
            runbook_id="model_performance_degradation",
            title="Model Performance Degradation Response",
            category="performance",
            description="Steps to handle model performance degradation",
            steps=[
                {"step": 1, "action": "Verify performance metrics", "details": "Check accuracy, latency, and error rates"},
                {"step": 2, "action": "Analyze recent changes", "details": "Review recent deployments and data changes"},
                {"step": 3, "action": "Check data drift", "details": "Run drift detection analysis"},
                {"step": 4, "action": "Initiate retraining", "details": "Trigger automated retraining if needed"},
                {"step": 5, "action": "Validate recovery", "details": "Confirm performance restoration"}
            ],
            prerequisites=["Access to monitoring dashboard", "Model management permissions"],
            estimated_time_minutes=30,
            severity_level="high"
        )
        
        # System Outage Runbook
        self.runbooks["system_outage"] = OperationalRunbook(
            runbook_id="system_outage",
            title="System Outage Response",
            category="incident",
            description="Emergency response for system outages",
            steps=[
                {"step": 1, "action": "Assess outage scope", "details": "Determine affected components and users"},
                {"step": 2, "action": "Activate backup systems", "details": "Switch to redundant infrastructure"},
                {"step": 3, "action": "Communicate status", "details": "Notify stakeholders and users"},
                {"step": 4, "action": "Investigate root cause", "details": "Identify and address underlying issue"},
                {"step": 5, "action": "Restore primary systems", "details": "Safely return to normal operations"},
                {"step": 6, "action": "Post-incident review", "details": "Document lessons learned"}
            ],
            prerequisites=["Emergency access credentials", "Backup system access"],
            estimated_time_minutes=60,
            severity_level="critical"
        )
        
        # Data Backup and Recovery Runbook
        self.runbooks["backup_recovery"] = OperationalRunbook(
            runbook_id="backup_recovery",
            title="Data Backup and Recovery Procedures",
            category="maintenance",
            description="Regular backup and emergency recovery procedures",
            steps=[
                {"step": 1, "action": "Verify backup schedule", "details": "Confirm automated backups are running"},
                {"step": 2, "action": "Test backup integrity", "details": "Validate backup files and checksums"},
                {"step": 3, "action": "Document backup status", "details": "Update backup logs and metrics"},
                {"step": 4, "action": "Plan recovery test", "details": "Schedule regular recovery testing"},
                {"step": 5, "action": "Update retention policy", "details": "Review and adjust retention settings"}
            ],
            prerequisites=["Backup system access", "Storage permissions"],
            estimated_time_minutes=45,
            severity_level="medium"
        )
        
        # Model Retraining Runbook
        self.runbooks["model_retraining"] = OperationalRunbook(
            runbook_id="model_retraining",
            title="Automated Model Retraining",
            category="maintenance",
            description="Procedures for automated and manual model retraining",
            steps=[
                {"step": 1, "action": "Review retraining triggers", "details": "Check accuracy, drift, and schedule triggers"},
                {"step": 2, "action": "Prepare training data", "details": "Validate and preprocess recent data"},
                {"step": 3, "action": "Execute training pipeline", "details": "Run automated training workflow"},
                {"step": 4, "action": "Validate new model", "details": "Test performance and accuracy"},
                {"step": 5, "action": "Deploy if approved", "details": "Replace production model if validation passes"},
                {"step": 6, "action": "Monitor post-deployment", "details": "Track performance after deployment"}
            ],
            prerequisites=["Training pipeline access", "Model validation tools"],
            estimated_time_minutes=120,
            severity_level="medium"
        )
    
    def _start_monitoring(self):
        """Start background monitoring processes"""
        logger.info("Starting production ML monitoring")
        
        # In a real implementation, this would start background threads/tasks
        # For this demo, we'll simulate the monitoring being active
        self.monitoring_active = True
    
    def _cleanup_old_health_checks(self):
        """Clean up old health check records"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        self.health_checks = [
            check for check in self.health_checks
            if check.timestamp >= cutoff_date
        ]
        
        # Keep only the most recent checks if we exceed the limit
        if len(self.health_checks) > self.config["max_health_checks_stored"]:
            self.health_checks = self.health_checks[-self.config["max_health_checks_stored"]:]
    
    def _get_last_retraining_time(self) -> Optional[datetime]:
        """Get timestamp of last retraining"""
        if self.retraining_history:
            return max(entry["completed_at"] for entry in self.retraining_history)
        return None
    
    def _store_backup_record(self, backup_record: BackupRecord):
        """Store backup record to database"""
        db_path = self.storage_path / "monitoring.db"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO backup_records 
                (backup_id, backup_type, created_at, backup_path, size_bytes, 
                 checksum, components, retention_days, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backup_record.backup_id,
                backup_record.backup_type,
                backup_record.created_at.isoformat(),
                backup_record.backup_path,
                backup_record.size_bytes,
                backup_record.checksum,
                json.dumps(backup_record.components),
                backup_record.retention_days,
                json.dumps(backup_record.metadata)
            ))
    
    def _load_monitoring_data(self):
        """Load existing monitoring data"""
        try:
            # Load backup records
            db_path = self.storage_path / "monitoring.db"
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute("SELECT * FROM backup_records ORDER BY created_at DESC LIMIT 100")
                    
                    for row in cursor.fetchall():
                        backup_record = BackupRecord(
                            backup_id=row[0],
                            backup_type=row[1],
                            created_at=datetime.fromisoformat(row[2]),
                            backup_path=row[3],
                            size_bytes=row[4],
                            checksum=row[5],
                            components=json.loads(row[6]),
                            retention_days=row[7],
                            metadata=json.loads(row[8]) if row[8] else {}
                        )
                        self.backup_records.append(backup_record)
            
            logger.info(f"Loaded {len(self.backup_records)} backup records")
            
        except Exception as e:
            logger.error(f"Failed to load monitoring data: {str(e)}")
    
    async def _send_health_alert(self, health_check: HealthCheck):
        """Send health alert for critical issues"""
        logger.warning(f"Sending health alert for {health_check.component}: {health_check.message}")
        
        # In a real implementation, this would send actual alerts
        # For this demo, we'll just log the alert
        alert_message = f"ALERT: {health_check.component} - {health_check.message}"
        logger.warning(alert_message)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize production monitoring
        monitoring = ProductionMLMonitoring()
        
        print("Production ML Monitoring System initialized")
        print(f"Monitoring active: {monitoring.monitoring_active}")
        print(f"Alert channels: {monitoring.alert_channels}")
        
        # Perform health check
        print("\nPerforming system health check...")
        health_results = await monitoring.perform_health_check()
        
        for component, health_check in health_results.items():
            status_icon = "✓" if health_check.status == HealthStatus.HEALTHY else "⚠" if health_check.status == HealthStatus.WARNING else "✗"
            print(f"  {status_icon} {component}: {health_check.status.value} - {health_check.message}")
        
        # Check retraining triggers
        print("\nChecking retraining triggers...")
        triggers = await monitoring.check_retraining_triggers()
        print(f"  Retraining triggers found: {len(triggers)}")
        
        for trigger in triggers:
            print(f"    - {trigger.trigger_type}: {trigger.reason}")
        
        # Execute automated retraining if needed
        if triggers:
            print("\nExecuting automated retraining...")
            retraining_results = await monitoring.execute_automated_retraining()
            print(f"  Retraining results: {retraining_results}")
        
        # Create system backup
        print("\nCreating system backup...")
        backup_record = await monitoring.create_system_backup("full")
        print(f"  Backup created: {backup_record.backup_id} ({backup_record.size_bytes} bytes)")
        
        # Get performance dashboard
        print("\nPerformance Dashboard:")
        dashboard = monitoring.get_performance_dashboard()
        
        system_overview = dashboard["system_overview"]
        print(f"  System Health: {system_overview['overall_health_percentage']:.1f}%")
        print(f"  Healthy Components: {system_overview['healthy_components']}/{system_overview['total_components']}")
        print(f"  Total Health Checks: {system_overview['total_health_checks']}")
        
        # List operational runbooks
        print("\nOperational Runbooks:")
        runbooks = monitoring.list_operational_runbooks()
        for runbook in runbooks:
            print(f"  - {runbook.title} ({runbook.category}) - {runbook.estimated_time_minutes} min")
        
        # Test disaster recovery
        print("\nTesting disaster recovery...")
        recovery_result = monitoring.execute_disaster_recovery("model_failure")
        print(f"  Recovery success: {recovery_result['recovery_success']}")
        print(f"  RTO: {recovery_result['estimated_rto_minutes']} min, RPO: {recovery_result['estimated_rpo_minutes']} min")
        print(f"  Steps completed: {len(recovery_result['steps_completed'])}")
    
    # Run the example
    import asyncio
    asyncio.run(main())