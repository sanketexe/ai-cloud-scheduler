"""
Comprehensive Audit and Compliance System for ML Activities

Provides immutable audit logging, model versioning, performance tracking,
compliance reporting, and data retention controls for anomaly detection ML systems.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import uuid
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of auditable events"""
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    ANOMALY_DETECTION = "anomaly_detection"
    ALERT_GENERATION = "alert_generation"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_RETENTION = "data_retention"

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPTED = "exempted"
    UNKNOWN = "unknown"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class RetentionPolicy(Enum):
    """Data retention policies"""
    SHORT_TERM = 30      # 30 days
    MEDIUM_TERM = 365    # 1 year
    LONG_TERM = 2555     # 7 years
    PERMANENT = -1       # Never delete

@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Event details
    description: str
    component: str
    action: str
    resource_id: Optional[str]
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security and integrity
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    # Compliance
    data_classification: DataClassification = DataClassification.INTERNAL
    retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM
    
    def __post_init__(self):
        """Generate checksum for integrity verification"""
        if not self.checksum:
            self.checksum = self._generate_checksum()
    
    def _generate_checksum(self) -> str:
        """Generate SHA-256 checksum for event integrity"""
        event_data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'description': self.description,
            'component': self.component,
            'action': self.action,
            'resource_id': self.resource_id,
            'context': self.context,
            'metadata': self.metadata
        }
        
        event_json = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        return self.checksum == self._generate_checksum()

@dataclass
class ModelVersion:
    """Model version tracking"""
    version_id: str
    model_name: str
    version_number: str
    created_at: datetime
    created_by: str
    
    # Model metadata
    model_type: str
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    # Deployment info
    deployment_status: str = "pending"
    deployed_at: Optional[datetime] = None
    deployment_environment: Optional[str] = None
    
    # Compliance
    approval_status: str = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Lifecycle
    is_active: bool = True
    retired_at: Optional[datetime] = None
    retirement_reason: Optional[str] = None

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    category: str
    severity: str  # critical, high, medium, low
    
    # Rule logic
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    
    # Metadata
    created_at: datetime
    created_by: str
    is_active: bool = True
    last_updated: Optional[datetime] = None

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Overall status
    overall_status: ComplianceStatus
    compliance_score: float  # 0-100
    
    # Detailed results
    rule_results: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Metrics
    total_events_audited: int
    compliant_events: int
    non_compliant_events: int
    
    # Metadata
    generated_by: str
    report_type: str = "periodic"
class AuditComplianceSystem:
    """
    Comprehensive audit and compliance system for ML operations.
    
    Provides immutable audit logging, model versioning, performance tracking,
    compliance reporting, and data retention management.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "audit_compliance_data"
        
        # Storage
        self.audit_events: List[AuditEvent] = []
        self.model_versions: Dict[str, ModelVersion] = {}
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.compliance_reports: Dict[str, ComplianceReport] = {}
        
        # Database connection
        self.db_path = Path(self.storage_path) / "audit_compliance.db"
        self.db_lock = threading.Lock()
        
        # Configuration
        self.max_events_in_memory = 10000
        self.auto_archive_days = 90
        self.compliance_check_interval_hours = 24
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Load existing data
        self._load_audit_data()
        self._setup_default_compliance_rules()
    
    def log_audit_event(
        self,
        event_type: AuditEventType,
        description: str,
        component: str,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        data_classification: DataClassification = DataClassification.INTERNAL
    ) -> AuditEvent:
        """Log immutable audit event"""
        
        event_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            description=description,
            component=component,
            action=action,
            resource_id=resource_id,
            context=context or {},
            metadata=metadata or {},
            data_classification=data_classification
        )
        
        # Store event
        self._store_audit_event(event)
        
        # Add to in-memory cache
        self.audit_events.append(event)
        
        # Manage memory usage
        if len(self.audit_events) > self.max_events_in_memory:
            self.audit_events = self.audit_events[-self.max_events_in_memory:]
        
        logger.debug(f"Logged audit event: {event_id}")
        return event
    
    def create_model_version(
        self,
        model_name: str,
        version_number: str,
        model_type: str,
        training_data_hash: str,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        created_by: str
    ) -> ModelVersion:
        """Create new model version with tracking"""
        
        version_id = f"model_{model_name}_{version_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version_number=version_number,
            created_at=datetime.now(),
            created_by=created_by,
            model_type=model_type,
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics
        )
        
        # Store model version
        self.model_versions[version_id] = model_version
        self._store_model_version(model_version)
        
        # Log audit event
        self.log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            description=f"Created model version {version_number} for {model_name}",
            component="model_management",
            action="create_version",
            user_id=created_by,
            resource_id=version_id,
            context={
                'model_name': model_name,
                'version_number': version_number,
                'model_type': model_type,
                'performance_metrics': performance_metrics
            }
        )
        
        logger.info(f"Created model version: {version_id}")
        return model_version
    
    def deploy_model_version(
        self,
        version_id: str,
        environment: str,
        deployed_by: str
    ) -> bool:
        """Deploy model version with audit trail"""
        
        model_version = self.model_versions.get(version_id)
        if not model_version:
            logger.error(f"Model version not found: {version_id}")
            return False
        
        # Update deployment status
        model_version.deployment_status = "deployed"
        model_version.deployed_at = datetime.now()
        model_version.deployment_environment = environment
        
        # Update storage
        self._store_model_version(model_version)
        
        # Log audit event
        self.log_audit_event(
            event_type=AuditEventType.MODEL_DEPLOYMENT,
            description=f"Deployed model version {model_version.version_number} to {environment}",
            component="model_management",
            action="deploy",
            user_id=deployed_by,
            resource_id=version_id,
            context={
                'model_name': model_version.model_name,
                'version_number': model_version.version_number,
                'environment': environment,
                'deployment_timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Deployed model version {version_id} to {environment}")
        return True
    
    def log_anomaly_detection(
        self,
        anomaly_id: str,
        model_version_id: str,
        detection_result: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AuditEvent:
        """Log anomaly detection activity"""
        
        return self.log_audit_event(
            event_type=AuditEventType.ANOMALY_DETECTION,
            description=f"Anomaly detection performed for {anomaly_id}",
            component="anomaly_detector",
            action="detect_anomaly",
            user_id=user_id,
            resource_id=anomaly_id,
            context={
                'model_version_id': model_version_id,
                'confidence_score': detection_result.get('confidence', 0),
                'anomaly_score': detection_result.get('anomaly_score', 0),
                'features_analyzed': detection_result.get('features_analyzed', [])
            },
            metadata={
                'detection_timestamp': datetime.now().isoformat(),
                'model_performance': detection_result.get('model_performance', {})
            }
        )
    
    def log_alert_generation(
        self,
        alert_id: str,
        anomaly_id: str,
        alert_details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AuditEvent:
        """Log alert generation activity"""
        
        return self.log_audit_event(
            event_type=AuditEventType.ALERT_GENERATION,
            description=f"Alert generated for anomaly {anomaly_id}",
            component="alert_engine",
            action="generate_alert",
            user_id=user_id,
            resource_id=alert_id,
            context={
                'anomaly_id': anomaly_id,
                'alert_severity': alert_details.get('severity'),
                'notification_channels': alert_details.get('channels', []),
                'escalation_level': alert_details.get('escalation_level', 1)
            },
            metadata={
                'alert_timestamp': datetime.now().isoformat(),
                'delivery_status': alert_details.get('delivery_status', {})
            }
        )
    
    def log_configuration_change(
        self,
        component: str,
        configuration_type: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        changed_by: str,
        reason: Optional[str] = None
    ) -> AuditEvent:
        """Log configuration changes"""
        
        # Calculate configuration diff
        config_diff = self._calculate_config_diff(old_config, new_config)
        
        return self.log_audit_event(
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            description=f"Configuration changed for {component}",
            component=component,
            action="update_configuration",
            user_id=changed_by,
            context={
                'configuration_type': configuration_type,
                'changes': config_diff,
                'reason': reason
            },
            metadata={
                'old_config_hash': hashlib.sha256(json.dumps(old_config, sort_keys=True).encode()).hexdigest(),
                'new_config_hash': hashlib.sha256(json.dumps(new_config, sort_keys=True).encode()).hexdigest(),
                'change_timestamp': datetime.now().isoformat()
            }
        )
    
    def add_compliance_rule(
        self,
        name: str,
        description: str,
        category: str,
        severity: str,
        conditions: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        created_by: str
    ) -> ComplianceRule:
        """Add new compliance rule"""
        
        rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        rule = ComplianceRule(
            rule_id=rule_id,
            name=name,
            description=description,
            category=category,
            severity=severity,
            conditions=conditions,
            actions=actions,
            created_at=datetime.now(),
            created_by=created_by
        )
        
        self.compliance_rules[rule_id] = rule
        self._store_compliance_rule(rule)
        
        # Log audit event
        self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            description=f"Added compliance rule: {name}",
            component="compliance_system",
            action="add_rule",
            user_id=created_by,
            resource_id=rule_id,
            context={
                'rule_name': name,
                'category': category,
                'severity': severity
            }
        )
        
        logger.info(f"Added compliance rule: {rule_id}")
        return rule
    
    def run_compliance_check(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        generated_by: str = "system"
    ) -> ComplianceReport:
        """Run comprehensive compliance check"""
        
        if not period_end:
            period_end = datetime.now()
        if not period_start:
            period_start = period_end - timedelta(days=30)
        
        report_id = f"compliance_{period_end.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Get events in period
        events_in_period = self._get_events_in_period(period_start, period_end)
        
        # Run compliance checks
        rule_results = []
        violations = []
        compliant_count = 0
        non_compliant_count = 0
        
        for rule in self.compliance_rules.values():
            if not rule.is_active:
                continue
            
            rule_result = self._evaluate_compliance_rule(rule, events_in_period)
            rule_results.append(rule_result)
            
            if rule_result['status'] == 'compliant':
                compliant_count += rule_result.get('compliant_events', 0)
            else:
                non_compliant_count += rule_result.get('non_compliant_events', 0)
                violations.extend(rule_result.get('violations', []))
        
        # Calculate overall compliance
        total_events = len(events_in_period)
        compliance_score = (compliant_count / total_events * 100) if total_events > 0 else 100
        
        overall_status = ComplianceStatus.COMPLIANT
        if compliance_score < 95:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif compliance_score < 98:
            overall_status = ComplianceStatus.PENDING_REVIEW
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(violations, rule_results)
        
        # Create report
        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            overall_status=overall_status,
            compliance_score=compliance_score,
            rule_results=rule_results,
            violations=violations,
            recommendations=recommendations,
            total_events_audited=total_events,
            compliant_events=compliant_count,
            non_compliant_events=non_compliant_count,
            generated_by=generated_by
        )
        
        # Store report
        self.compliance_reports[report_id] = report
        self._store_compliance_report(report)
        
        # Log audit event
        self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            description=f"Compliance check completed with {compliance_score:.1f}% score",
            component="compliance_system",
            action="run_compliance_check",
            user_id=generated_by,
            resource_id=report_id,
            context={
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'compliance_score': compliance_score,
                'overall_status': overall_status.value,
                'total_events': total_events,
                'violations_count': len(violations)
            }
        )
        
        logger.info(f"Compliance check completed: {report_id} (Score: {compliance_score:.1f}%)")
        return report
    
    def apply_data_retention(
        self,
        dry_run: bool = True,
        retention_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Apply data retention policies"""
        
        if not retention_date:
            retention_date = datetime.now()
        
        retention_results = {
            'events_processed': 0,
            'events_archived': 0,
            'events_deleted': 0,
            'models_archived': 0,
            'reports_archived': 0,
            'dry_run': dry_run
        }
        
        # Process audit events
        events_to_process = self._get_events_for_retention(retention_date)
        retention_results['events_processed'] = len(events_to_process)
        
        for event in events_to_process:
            retention_policy = event.retention_policy
            
            if retention_policy == RetentionPolicy.PERMANENT:
                continue
            
            event_age_days = (retention_date - event.timestamp).days
            
            if event_age_days > retention_policy.value:
                if not dry_run:
                    self._archive_or_delete_event(event)
                
                if retention_policy in [RetentionPolicy.SHORT_TERM, RetentionPolicy.MEDIUM_TERM]:
                    retention_results['events_archived'] += 1
                else:
                    retention_results['events_deleted'] += 1
        
        # Process model versions
        for model_version in self.model_versions.values():
            if model_version.retired_at:
                age_days = (retention_date - model_version.retired_at).days
                if age_days > RetentionPolicy.LONG_TERM.value:
                    if not dry_run:
                        self._archive_model_version(model_version)
                    retention_results['models_archived'] += 1
        
        # Log retention activity
        if not dry_run:
            self.log_audit_event(
                event_type=AuditEventType.DATA_RETENTION,
                description=f"Data retention applied: {retention_results['events_archived']} events archived, {retention_results['events_deleted']} deleted",
                component="compliance_system",
                action="apply_retention",
                context=retention_results
            )
        
        logger.info(f"Data retention {'simulation' if dry_run else 'application'} completed: {retention_results}")
        return retention_results
    
    def get_audit_trail(
        self,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Get filtered audit trail"""
        
        return self._query_audit_events(
            resource_id=resource_id,
            user_id=user_id,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    def get_model_performance_history(
        self,
        model_name: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get model performance history"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get model versions in period
        model_versions = [
            mv for mv in self.model_versions.values()
            if mv.model_name == model_name and start_date <= mv.created_at <= end_date
        ]
        
        # Get performance events
        performance_events = self._query_audit_events(
            event_type=AuditEventType.ANOMALY_DETECTION,
            start_date=start_date,
            end_date=end_date
        )
        
        # Aggregate performance data
        performance_history = []
        
        for version in model_versions:
            version_events = [
                e for e in performance_events
                if e.context.get('model_version_id') == version.version_id
            ]
            
            if version_events:
                avg_confidence = sum(e.context.get('confidence_score', 0) for e in version_events) / len(version_events)
                avg_anomaly_score = sum(e.context.get('anomaly_score', 0) for e in version_events) / len(version_events)
                
                performance_history.append({
                    'version_id': version.version_id,
                    'version_number': version.version_number,
                    'created_at': version.created_at,
                    'detections_count': len(version_events),
                    'avg_confidence': avg_confidence,
                    'avg_anomaly_score': avg_anomaly_score,
                    'performance_metrics': version.performance_metrics
                })
        
        return sorted(performance_history, key=lambda x: x['created_at'])
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        
        # Get recent compliance reports
        recent_reports = sorted(
            self.compliance_reports.values(),
            key=lambda r: r.generated_at,
            reverse=True
        )[:5]
        
        # Calculate compliance trends
        compliance_trend = []
        for report in reversed(recent_reports):
            compliance_trend.append({
                'date': report.generated_at.date(),
                'score': report.compliance_score,
                'status': report.overall_status.value
            })
        
        # Get violation summary
        violation_summary = {}
        for report in recent_reports[:1]:  # Latest report
            for violation in report.violations:
                category = violation.get('category', 'unknown')
                violation_summary[category] = violation_summary.get(category, 0) + 1
        
        # Get audit activity summary
        recent_events = self._query_audit_events(
            start_date=datetime.now() - timedelta(days=7),
            limit=10000
        )
        
        activity_summary = {}
        for event in recent_events:
            event_type = event.event_type.value
            activity_summary[event_type] = activity_summary.get(event_type, 0) + 1
        
        return {
            'compliance_trend': compliance_trend,
            'latest_compliance_score': recent_reports[0].compliance_score if recent_reports else 0,
            'violation_summary': violation_summary,
            'activity_summary': activity_summary,
            'total_audit_events': len(recent_events),
            'active_compliance_rules': len([r for r in self.compliance_rules.values() if r.is_active]),
            'model_versions_tracked': len(self.model_versions),
            'last_updated': datetime.now()
        }
    def _initialize_database(self):
        """Initialize SQLite database for audit storage"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Audit events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    description TEXT NOT NULL,
                    component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT,
                    context TEXT,
                    metadata TEXT,
                    checksum TEXT NOT NULL,
                    data_classification TEXT,
                    retention_policy TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    training_data_hash TEXT NOT NULL,
                    hyperparameters TEXT,
                    performance_metrics TEXT,
                    deployment_status TEXT DEFAULT 'pending',
                    deployed_at TEXT,
                    deployment_environment TEXT,
                    approval_status TEXT DEFAULT 'pending',
                    approved_by TEXT,
                    approved_at TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    retired_at TEXT,
                    retirement_reason TEXT
                )
            ''')
            
            # Compliance rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    conditions TEXT,
                    actions TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    last_updated TEXT
                )
            ''')
            
            # Compliance reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    compliance_score REAL NOT NULL,
                    rule_results TEXT,
                    violations TEXT,
                    recommendations TEXT,
                    total_events_audited INTEGER,
                    compliant_events INTEGER,
                    non_compliant_events INTEGER,
                    generated_by TEXT NOT NULL,
                    report_type TEXT DEFAULT 'periodic'
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_events_resource ON audit_events(resource_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(model_name)')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper locking"""
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_events (
                    event_id, event_type, timestamp, user_id, session_id,
                    description, component, action, resource_id, context,
                    metadata, checksum, data_classification, retention_policy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.session_id,
                event.description,
                event.component,
                event.action,
                event.resource_id,
                json.dumps(event.context),
                json.dumps(event.metadata),
                event.checksum,
                event.data_classification.value,
                event.retention_policy.name if hasattr(event.retention_policy, 'name') else str(event.retention_policy.value)
            ))
            conn.commit()
    
    def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO model_versions (
                    version_id, model_name, version_number, created_at, created_by,
                    model_type, training_data_hash, hyperparameters, performance_metrics,
                    deployment_status, deployed_at, deployment_environment,
                    approval_status, approved_by, approved_at, is_active,
                    retired_at, retirement_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version.version_id,
                model_version.model_name,
                model_version.version_number,
                model_version.created_at.isoformat(),
                model_version.created_by,
                model_version.model_type,
                model_version.training_data_hash,
                json.dumps(model_version.hyperparameters),
                json.dumps(model_version.performance_metrics),
                model_version.deployment_status,
                model_version.deployed_at.isoformat() if model_version.deployed_at else None,
                model_version.deployment_environment,
                model_version.approval_status,
                model_version.approved_by,
                model_version.approved_at.isoformat() if model_version.approved_at else None,
                model_version.is_active,
                model_version.retired_at.isoformat() if model_version.retired_at else None,
                model_version.retirement_reason
            ))
            conn.commit()
    
    def _store_compliance_rule(self, rule: ComplianceRule):
        """Store compliance rule in database"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO compliance_rules (
                    rule_id, name, description, category, severity,
                    conditions, actions, created_at, created_by, is_active, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.name,
                rule.description,
                rule.category,
                rule.severity,
                json.dumps(rule.conditions),
                json.dumps(rule.actions),
                rule.created_at.isoformat(),
                rule.created_by,
                rule.is_active,
                rule.last_updated.isoformat() if rule.last_updated else None
            ))
            conn.commit()
    
    def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO compliance_reports (
                    report_id, generated_at, period_start, period_end,
                    overall_status, compliance_score, rule_results, violations,
                    recommendations, total_events_audited, compliant_events,
                    non_compliant_events, generated_by, report_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id,
                report.generated_at.isoformat(),
                report.period_start.isoformat(),
                report.period_end.isoformat(),
                report.overall_status.value,
                report.compliance_score,
                json.dumps(report.rule_results),
                json.dumps(report.violations),
                json.dumps(report.recommendations),
                report.total_events_audited,
                report.compliant_events,
                report.non_compliant_events,
                report.generated_by,
                report.report_type
            ))
            conn.commit()
    
    def _query_audit_events(
        self,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events from database"""
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if resource_id:
            query += " AND resource_id = ?"
            params.append(resource_id)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                event = AuditEvent(
                    event_id=row['event_id'],
                    event_type=AuditEventType(row['event_type']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    description=row['description'],
                    component=row['component'],
                    action=row['action'],
                    resource_id=row['resource_id'],
                    context=json.loads(row['context']) if row['context'] else {},
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    checksum=row['checksum'],
                    data_classification=DataClassification(row['data_classification']),
                    retention_policy=RetentionPolicy(int(row['retention_policy'])) if row['retention_policy'].isdigit() else RetentionPolicy.MEDIUM_TERM
                )
                events.append(event)
        
        return events
    
    def _load_audit_data(self):
        """Load recent audit data into memory"""
        
        # Load recent events
        recent_events = self._query_audit_events(
            start_date=datetime.now() - timedelta(days=7),
            limit=self.max_events_in_memory
        )
        self.audit_events = recent_events
        
        # Load model versions
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE is_active = 1")
            
            for row in cursor.fetchall():
                model_version = ModelVersion(
                    version_id=row['version_id'],
                    model_name=row['model_name'],
                    version_number=row['version_number'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    model_type=row['model_type'],
                    training_data_hash=row['training_data_hash'],
                    hyperparameters=json.loads(row['hyperparameters']) if row['hyperparameters'] else {},
                    performance_metrics=json.loads(row['performance_metrics']) if row['performance_metrics'] else {},
                    deployment_status=row['deployment_status'],
                    deployed_at=datetime.fromisoformat(row['deployed_at']) if row['deployed_at'] else None,
                    deployment_environment=row['deployment_environment'],
                    approval_status=row['approval_status'],
                    approved_by=row['approved_by'],
                    approved_at=datetime.fromisoformat(row['approved_at']) if row['approved_at'] else None,
                    is_active=bool(row['is_active']),
                    retired_at=datetime.fromisoformat(row['retired_at']) if row['retired_at'] else None,
                    retirement_reason=row['retirement_reason']
                )
                self.model_versions[model_version.version_id] = model_version
        
        # Load compliance rules
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM compliance_rules WHERE is_active = 1")
            
            for row in cursor.fetchall():
                rule = ComplianceRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    category=row['category'],
                    severity=row['severity'],
                    conditions=json.loads(row['conditions']) if row['conditions'] else [],
                    actions=json.loads(row['actions']) if row['actions'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    created_by=row['created_by'],
                    is_active=bool(row['is_active']),
                    last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None
                )
                self.compliance_rules[rule.rule_id] = rule
        
        logger.info(f"Loaded audit data: {len(self.audit_events)} events, {len(self.model_versions)} model versions, {len(self.compliance_rules)} compliance rules")
    
    def _setup_default_compliance_rules(self):
        """Setup default compliance rules"""
        
        if self.compliance_rules:
            return  # Rules already exist
        
        default_rules = [
            {
                'name': 'Model Deployment Approval',
                'description': 'All model deployments must be approved',
                'category': 'model_governance',
                'severity': 'high',
                'conditions': [
                    {'event_type': 'model_deployment', 'approval_required': True}
                ],
                'actions': [
                    {'type': 'require_approval', 'approver_role': 'ml_engineer'}
                ]
            },
            {
                'name': 'Anomaly Detection Logging',
                'description': 'All anomaly detections must be logged',
                'category': 'audit_trail',
                'severity': 'medium',
                'conditions': [
                    {'event_type': 'anomaly_detection', 'logging_required': True}
                ],
                'actions': [
                    {'type': 'log_event', 'retention_period': 'long_term'}
                ]
            },
            {
                'name': 'Configuration Change Authorization',
                'description': 'Configuration changes must be authorized',
                'category': 'change_management',
                'severity': 'high',
                'conditions': [
                    {'event_type': 'configuration_change', 'authorization_required': True}
                ],
                'actions': [
                    {'type': 'require_authorization', 'authorized_roles': ['admin', 'ml_engineer']}
                ]
            }
        ]
        
        for rule_config in default_rules:
            self.add_compliance_rule(
                name=rule_config['name'],
                description=rule_config['description'],
                category=rule_config['category'],
                severity=rule_config['severity'],
                conditions=rule_config['conditions'],
                actions=rule_config['actions'],
                created_by='system'
            )
    
    def _calculate_config_diff(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate configuration differences"""
        
        diff = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added and modified keys
        for key, new_value in new_config.items():
            if key not in old_config:
                diff['added'][key] = new_value
            elif old_config[key] != new_value:
                diff['modified'][key] = {
                    'old': old_config[key],
                    'new': new_value
                }
        
        # Find removed keys
        for key, old_value in old_config.items():
            if key not in new_config:
                diff['removed'][key] = old_value
        
        return diff
    
    def _get_events_in_period(self, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Get audit events in specified period"""
        
        return self._query_audit_events(
            start_date=start_date,
            end_date=end_date,
            limit=100000  # Large limit for compliance checks
        )
    
    def _evaluate_compliance_rule(self, rule: ComplianceRule, events: List[AuditEvent]) -> Dict[str, Any]:
        """Evaluate compliance rule against events"""
        
        result = {
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'status': 'compliant',
            'compliant_events': 0,
            'non_compliant_events': 0,
            'violations': []
        }
        
        # Simple rule evaluation (would be more sophisticated in production)
        for event in events:
            rule_applies = False
            
            # Check if rule applies to this event
            for condition in rule.conditions:
                if condition.get('event_type') == event.event_type.value:
                    rule_applies = True
                    break
            
            if rule_applies:
                # Check compliance (simplified logic)
                is_compliant = True
                
                # Example compliance checks
                if rule.name == 'Model Deployment Approval':
                    if event.event_type == AuditEventType.MODEL_DEPLOYMENT:
                        # Check if deployment was approved
                        model_version_id = event.resource_id
                        model_version = self.model_versions.get(model_version_id)
                        if model_version and model_version.approval_status != 'approved':
                            is_compliant = False
                
                elif rule.name == 'Anomaly Detection Logging':
                    if event.event_type == AuditEventType.ANOMALY_DETECTION:
                        # Check if proper logging occurred
                        if not event.context or not event.metadata:
                            is_compliant = False
                
                elif rule.name == 'Configuration Change Authorization':
                    if event.event_type == AuditEventType.CONFIGURATION_CHANGE:
                        # Check if user is authorized
                        authorized_users = ['admin', 'ml_engineer', 'system']
                        if event.user_id not in authorized_users:
                            is_compliant = False
                
                if is_compliant:
                    result['compliant_events'] += 1
                else:
                    result['non_compliant_events'] += 1
                    result['status'] = 'non_compliant'
                    result['violations'].append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'description': f"Rule violation: {rule.name}",
                        'severity': rule.severity,
                        'category': rule.category
                    })
        
        return result
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]], rule_results: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # Analyze violations by category
        violation_categories = {}
        for violation in violations:
            category = violation.get('category', 'unknown')
            violation_categories[category] = violation_categories.get(category, 0) + 1
        
        # Generate category-specific recommendations
        if violation_categories.get('model_governance', 0) > 0:
            recommendations.append("Implement mandatory approval workflow for model deployments")
        
        if violation_categories.get('audit_trail', 0) > 0:
            recommendations.append("Enhance audit logging to capture all required metadata")
        
        if violation_categories.get('change_management', 0) > 0:
            recommendations.append("Restrict configuration changes to authorized personnel only")
        
        # General recommendations
        if len(violations) > 10:
            recommendations.append("Conduct compliance training for all ML operations team members")
        
        if not recommendations:
            recommendations.append("Maintain current compliance practices - no issues detected")
        
        return recommendations
    
    def _get_events_for_retention(self, retention_date: datetime) -> List[AuditEvent]:
        """Get events that may need retention processing"""
        
        # Get events older than shortest retention period
        cutoff_date = retention_date - timedelta(days=RetentionPolicy.SHORT_TERM.value)
        
        return self._query_audit_events(
            end_date=cutoff_date,
            limit=100000
        )
    
    def _archive_or_delete_event(self, event: AuditEvent):
        """Archive or delete event based on retention policy"""
        
        # In production, this would move to archive storage or delete
        logger.debug(f"Processing retention for event {event.event_id}")
    
    def _archive_model_version(self, model_version: ModelVersion):
        """Archive retired model version"""
        
        # In production, this would move to archive storage
        logger.debug(f"Archiving model version {model_version.version_id}")

# Global audit compliance system instance
audit_compliance_system = AuditComplianceSystem()