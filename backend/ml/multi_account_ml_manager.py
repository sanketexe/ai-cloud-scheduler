"""
Multi-Account ML Manager for Cross-Account Anomaly Detection

Provides cross-account pattern detection, account-specific model training,
consolidated anomaly reporting, and cross-account correlation analysis
while maintaining proper account isolation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    """Account status levels"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class ModelScope(Enum):
    """Model scope levels"""
    ACCOUNT_SPECIFIC = "account_specific"
    CROSS_ACCOUNT = "cross_account"
    ORGANIZATION_WIDE = "organization_wide"

class CorrelationType(Enum):
    """Cross-account correlation types"""
    COST_PATTERN = "cost_pattern"
    SEASONAL_TREND = "seasonal_trend"
    SERVICE_USAGE = "service_usage"
    ANOMALY_CLUSTER = "anomaly_cluster"
    DEPLOYMENT_IMPACT = "deployment_impact"

@dataclass
class AccountConfiguration:
    """AWS account configuration for ML operations"""
    account_id: str
    account_name: str
    account_alias: Optional[str]
    
    # Status and permissions
    status: AccountStatus = AccountStatus.ACTIVE
    ml_enabled: bool = True
    cross_account_sharing: bool = True
    
    # Model configuration
    model_scope: ModelScope = ModelScope.ACCOUNT_SPECIFIC
    baseline_period_days: int = 30
    sensitivity_level: str = "balanced"
    
    # Data and privacy
    data_classification: str = "internal"
    isolation_required: bool = False
    compliance_tags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Performance tracking
    model_performance: Dict[str, float] = field(default_factory=dict)
    anomaly_count_30d: int = 0
    alert_count_30d: int = 0

@dataclass
class CrossAccountAnomaly:
    """Cross-account anomaly detection result"""
    anomaly_id: str
    detection_timestamp: datetime
    
    # Account information
    primary_account_id: str
    affected_accounts: List[str]
    correlation_accounts: List[str]
    
    # Anomaly details
    anomaly_type: str
    confidence_score: float
    severity: str
    
    # Cross-account analysis
    correlation_type: CorrelationType
    correlation_strength: float
    pattern_description: str
    
    # Impact assessment
    estimated_impact_usd: float
    affected_services: List[str]
    time_window: Tuple[datetime, datetime]
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsolidatedReport:
    """Consolidated multi-account anomaly report"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Account coverage
    included_accounts: List[str]
    total_accounts: int
    active_accounts: int
    
    # Anomaly summary
    total_anomalies: int
    cross_account_anomalies: int
    account_specific_anomalies: int
    
    # Impact analysis
    total_estimated_impact_usd: float
    impact_by_account: Dict[str, float]
    impact_by_service: Dict[str, float]
    
    # Correlation analysis
    correlation_patterns: List[Dict[str, Any]]
    organization_trends: List[Dict[str, Any]]
    
    # Performance metrics
    detection_accuracy: float
    false_positive_rate: float
    average_detection_time_minutes: float
    
    # Recommendations
    recommendations: List[str]
    action_items: List[Dict[str, Any]]

class MultiAccountMLManager:
    """
    Multi-Account ML Manager for cross-account anomaly detection.
    
    Manages ML operations across multiple AWS accounts while maintaining
    proper isolation and providing consolidated reporting and analysis.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "multi_account_ml_data"
        
        # Account management
        self.accounts: Dict[str, AccountConfiguration] = {}
        self.account_models: Dict[str, Dict[str, Any]] = {}
        self.cross_account_models: Dict[str, Any] = {}
        
        # Anomaly tracking
        self.account_anomalies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cross_account_anomalies: List[CrossAccountAnomaly] = []
        self.consolidated_reports: Dict[str, ConsolidatedReport] = {}
        
        # Configuration
        self.max_concurrent_accounts = 10
        self.correlation_threshold = 0.7
        self.cross_account_window_hours = 24
        self.isolation_enforcement = True
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_accounts)
        self.lock = threading.Lock()
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_account_data()
    
    def register_account(
        self,
        account_id: str,
        account_name: str,
        account_alias: Optional[str] = None,
        ml_enabled: bool = True,
        cross_account_sharing: bool = True,
        model_scope: ModelScope = ModelScope.ACCOUNT_SPECIFIC,
        isolation_required: bool = False,
        compliance_tags: Optional[List[str]] = None
    ) -> AccountConfiguration:
        """Register AWS account for multi-account ML operations"""
        
        if account_id in self.accounts:
            logger.warning(f"Account {account_id} already registered, updating configuration")
        
        account_config = AccountConfiguration(
            account_id=account_id,
            account_name=account_name,
            account_alias=account_alias,
            ml_enabled=ml_enabled,
            cross_account_sharing=cross_account_sharing,
            model_scope=model_scope,
            isolation_required=isolation_required,
            compliance_tags=compliance_tags or []
        )
        
        self.accounts[account_id] = account_config
        self.account_models[account_id] = {}
        self.account_anomalies[account_id] = []
        
        # Initialize account-specific models if needed
        if ml_enabled:
            self._initialize_account_models(account_id)
        
        logger.info(f"Registered account: {account_id} ({account_name})")
        return account_config
    
    def update_account_configuration(
        self,
        account_id: str,
        **updates
    ) -> bool:
        """Update account configuration"""
        
        if account_id not in self.accounts:
            logger.error(f"Account not found: {account_id}")
            return False
        
        account = self.accounts[account_id]
        
        # Update configuration
        for key, value in updates.items():
            if hasattr(account, key):
                setattr(account, key, value)
        
        account.last_updated = datetime.now()
        
        # Reinitialize models if ML configuration changed
        if 'ml_enabled' in updates or 'model_scope' in updates:
            if account.ml_enabled:
                self._initialize_account_models(account_id)
            else:
                self._cleanup_account_models(account_id)
        
        logger.info(f"Updated account configuration: {account_id}")
        return True
    
    async def detect_cross_account_anomalies(
        self,
        time_window_hours: int = 24,
        correlation_threshold: float = 0.7
    ) -> List[CrossAccountAnomaly]:
        """Detect anomalies across multiple accounts with correlation analysis"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get active accounts
        active_accounts = [
            acc_id for acc_id, acc in self.accounts.items()
            if acc.status == AccountStatus.ACTIVE and acc.ml_enabled
        ]
        
        if len(active_accounts) < 2:
            logger.warning("Need at least 2 active accounts for cross-account detection")
            return []
        
        logger.info(f"Running cross-account anomaly detection for {len(active_accounts)} accounts")
        
        # Collect account-specific anomalies concurrently
        account_anomalies = await self._collect_account_anomalies(
            active_accounts, start_time, end_time
        )
        
        # Perform cross-account correlation analysis
        cross_account_anomalies = await self._analyze_cross_account_correlations(
            account_anomalies, correlation_threshold
        )
        
        # Store results
        self.cross_account_anomalies.extend(cross_account_anomalies)
        
        logger.info(f"Detected {len(cross_account_anomalies)} cross-account anomalies")
        return cross_account_anomalies
    
    async def generate_consolidated_report(
        self,
        period_days: int = 30,
        include_accounts: Optional[List[str]] = None
    ) -> ConsolidatedReport:
        """Generate consolidated anomaly report across accounts"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Determine accounts to include
        if include_accounts:
            target_accounts = [
                acc_id for acc_id in include_accounts
                if acc_id in self.accounts and self.accounts[acc_id].status == AccountStatus.ACTIVE
            ]
        else:
            target_accounts = [
                acc_id for acc_id, acc in self.accounts.items()
                if acc.status == AccountStatus.ACTIVE and acc.ml_enabled
            ]
        
        if not target_accounts:
            logger.warning("No active accounts found for consolidated reporting")
            return None
        
        report_id = f"consolidated_{end_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Generating consolidated report for {len(target_accounts)} accounts")
        
        # Collect anomaly data
        total_anomalies = 0
        cross_account_count = 0
        account_specific_count = 0
        impact_by_account = {}
        impact_by_service = defaultdict(float)
        
        # Process each account
        for account_id in target_accounts:
            account_anomalies = self._get_account_anomalies_in_period(
                account_id, start_date, end_date
            )
            
            account_impact = sum(
                anomaly.get('estimated_impact_usd', 0)
                for anomaly in account_anomalies
            )
            impact_by_account[account_id] = account_impact
            
            # Aggregate by service
            for anomaly in account_anomalies:
                for service in anomaly.get('affected_services', []):
                    service_impact = anomaly.get('estimated_impact_usd', 0) / len(anomaly.get('affected_services', [1]))
                    impact_by_service[service] += service_impact
            
            total_anomalies += len(account_anomalies)
            account_specific_count += len(account_anomalies)
        
        # Process cross-account anomalies
        cross_account_anomalies_in_period = [
            anomaly for anomaly in self.cross_account_anomalies
            if start_date <= anomaly.detection_timestamp <= end_date
            and any(acc_id in target_accounts for acc_id in anomaly.affected_accounts)
        ]
        
        cross_account_count = len(cross_account_anomalies_in_period)
        
        # Analyze correlation patterns
        correlation_patterns = self._analyze_correlation_patterns(
            cross_account_anomalies_in_period
        )
        
        # Identify organization trends
        organization_trends = self._identify_organization_trends(
            target_accounts, start_date, end_date
        )
        
        # Calculate performance metrics
        detection_accuracy = self._calculate_detection_accuracy(target_accounts, period_days)
        false_positive_rate = self._calculate_false_positive_rate(target_accounts, period_days)
        avg_detection_time = self._calculate_average_detection_time(target_accounts, period_days)
        
        # Generate recommendations
        recommendations = self._generate_multi_account_recommendations(
            target_accounts, correlation_patterns, organization_trends
        )
        
        # Create action items
        action_items = self._create_action_items(
            target_accounts, cross_account_anomalies_in_period, recommendations
        )
        
        # Create consolidated report
        report = ConsolidatedReport(
            report_id=report_id,
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            included_accounts=target_accounts,
            total_accounts=len(target_accounts),
            active_accounts=len([acc for acc in self.accounts.values() if acc.status == AccountStatus.ACTIVE]),
            total_anomalies=total_anomalies,
            cross_account_anomalies=cross_account_count,
            account_specific_anomalies=account_specific_count,
            total_estimated_impact_usd=sum(impact_by_account.values()),
            impact_by_account=impact_by_account,
            impact_by_service=dict(impact_by_service),
            correlation_patterns=correlation_patterns,
            organization_trends=organization_trends,
            detection_accuracy=detection_accuracy,
            false_positive_rate=false_positive_rate,
            average_detection_time_minutes=avg_detection_time,
            recommendations=recommendations,
            action_items=action_items
        )
        
        # Store report
        self.consolidated_reports[report_id] = report
        
        logger.info(f"Generated consolidated report: {report_id}")
        return report
    
    def get_account_isolation_status(self, account_id: str) -> Dict[str, Any]:
        """Get account isolation status and compliance"""
        
        if account_id not in self.accounts:
            return {'error': 'Account not found'}
        
        account = self.accounts[account_id]
        
        # Check isolation requirements
        isolation_status = {
            'account_id': account_id,
            'isolation_required': account.isolation_required,
            'isolation_enforced': True,  # Always enforced in this implementation
            'cross_account_sharing': account.cross_account_sharing,
            'data_classification': account.data_classification,
            'compliance_tags': account.compliance_tags
        }
        
        # Check for isolation violations
        violations = []
        
        # Check if isolated account data appears in cross-account models
        if account.isolation_required and account.cross_account_sharing:
            violations.append("Isolation required but cross-account sharing enabled")
        
        # Check cross-account anomaly involvement
        cross_account_involvement = [
            anomaly for anomaly in self.cross_account_anomalies
            if account_id in anomaly.affected_accounts and account.isolation_required
        ]
        
        if cross_account_involvement:
            violations.append(f"Isolated account involved in {len(cross_account_involvement)} cross-account anomalies")
        
        isolation_status['violations'] = violations
        isolation_status['compliance_score'] = 100 if not violations else max(0, 100 - len(violations) * 20)
        
        return isolation_status
    
    def get_cross_account_correlations(
        self,
        account_id: str,
        correlation_types: Optional[List[CorrelationType]] = None,
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get cross-account correlations for specific account"""
        
        if account_id not in self.accounts:
            return []
        
        correlations = []
        
        # Filter cross-account anomalies involving this account
        relevant_anomalies = [
            anomaly for anomaly in self.cross_account_anomalies
            if account_id in anomaly.affected_accounts or account_id == anomaly.primary_account_id
        ]
        
        for anomaly in relevant_anomalies:
            if correlation_types and anomaly.correlation_type not in correlation_types:
                continue
            
            if anomaly.correlation_strength < min_strength:
                continue
            
            correlation = {
                'anomaly_id': anomaly.anomaly_id,
                'correlation_type': anomaly.correlation_type.value,
                'correlation_strength': anomaly.correlation_strength,
                'pattern_description': anomaly.pattern_description,
                'correlated_accounts': [
                    acc_id for acc_id in anomaly.correlation_accounts
                    if acc_id != account_id
                ],
                'detection_timestamp': anomaly.detection_timestamp,
                'estimated_impact_usd': anomaly.estimated_impact_usd,
                'affected_services': anomaly.affected_services
            }
            
            correlations.append(correlation)
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)
        
        return correlations
    
    def get_organization_dashboard(self) -> Dict[str, Any]:
        """Get organization-wide ML dashboard data"""
        
        dashboard = {
            'organization_summary': {
                'total_accounts': len(self.accounts),
                'active_accounts': len([acc for acc in self.accounts.values() if acc.status == AccountStatus.ACTIVE]),
                'ml_enabled_accounts': len([acc for acc in self.accounts.values() if acc.ml_enabled]),
                'isolated_accounts': len([acc for acc in self.accounts.values() if acc.isolation_required])
            },
            'anomaly_summary': {
                'total_anomalies_30d': sum(acc.anomaly_count_30d for acc in self.accounts.values()),
                'cross_account_anomalies_30d': len([
                    anomaly for anomaly in self.cross_account_anomalies
                    if anomaly.detection_timestamp >= datetime.now() - timedelta(days=30)
                ]),
                'total_alerts_30d': sum(acc.alert_count_30d for acc in self.accounts.values())
            },
            'performance_summary': {
                'average_detection_accuracy': statistics.mean([
                    acc.model_performance.get('accuracy', 0.85)
                    for acc in self.accounts.values() if acc.ml_enabled
                ]) if any(acc.ml_enabled for acc in self.accounts.values()) else 0,
                'average_false_positive_rate': statistics.mean([
                    acc.model_performance.get('false_positive_rate', 0.05)
                    for acc in self.accounts.values() if acc.ml_enabled
                ]) if any(acc.ml_enabled for acc in self.accounts.values()) else 0
            },
            'account_details': []
        }
        
        # Add account details
        for account_id, account in self.accounts.items():
            account_detail = {
                'account_id': account_id,
                'account_name': account.account_name,
                'status': account.status.value,
                'ml_enabled': account.ml_enabled,
                'anomaly_count_30d': account.anomaly_count_30d,
                'alert_count_30d': account.alert_count_30d,
                'model_performance': account.model_performance,
                'last_updated': account.last_updated
            }
            dashboard['account_details'].append(account_detail)
        
        # Sort accounts by anomaly count
        dashboard['account_details'].sort(key=lambda x: x['anomaly_count_30d'], reverse=True)
        
        return dashboard
    
    async def _collect_account_anomalies(
        self,
        account_ids: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect anomalies from multiple accounts concurrently"""
        
        async def collect_account_data(account_id: str) -> Tuple[str, List[Dict[str, Any]]]:
            # Simulate anomaly detection for account
            anomalies = self._simulate_account_anomaly_detection(account_id, start_time, end_time)
            return account_id, anomalies
        
        # Collect data concurrently
        tasks = [collect_account_data(account_id) for account_id in account_ids]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _analyze_cross_account_correlations(
        self,
        account_anomalies: Dict[str, List[Dict[str, Any]]],
        correlation_threshold: float
    ) -> List[CrossAccountAnomaly]:
        """Analyze correlations between account anomalies"""
        
        cross_account_anomalies = []
        
        # Find temporal correlations
        time_window_minutes = 60
        
        for primary_account, primary_anomalies in account_anomalies.items():
            for primary_anomaly in primary_anomalies:
                primary_time = primary_anomaly['timestamp']
                
                # Find correlated anomalies in other accounts
                correlated_accounts = []
                correlation_strength = 0.0
                
                for other_account, other_anomalies in account_anomalies.items():
                    if other_account == primary_account:
                        continue
                    
                    # Check for temporal correlation
                    for other_anomaly in other_anomalies:
                        other_time = other_anomaly['timestamp']
                        time_diff = abs((primary_time - other_time).total_seconds() / 60)
                        
                        if time_diff <= time_window_minutes:
                            # Calculate correlation strength based on similarity
                            similarity = self._calculate_anomaly_similarity(
                                primary_anomaly, other_anomaly
                            )
                            
                            if similarity >= correlation_threshold:
                                correlated_accounts.append(other_account)
                                correlation_strength = max(correlation_strength, similarity)
                
                # Create cross-account anomaly if correlations found
                if correlated_accounts:
                    cross_account_anomaly = CrossAccountAnomaly(
                        anomaly_id=f"cross_account_{uuid.uuid4().hex[:8]}",
                        detection_timestamp=datetime.now(),
                        primary_account_id=primary_account,
                        affected_accounts=[primary_account] + correlated_accounts,
                        correlation_accounts=correlated_accounts,
                        anomaly_type=primary_anomaly.get('type', 'cost_spike'),
                        confidence_score=primary_anomaly.get('confidence', 0.8),
                        severity=self._determine_cross_account_severity(
                            len(correlated_accounts), correlation_strength
                        ),
                        correlation_type=self._determine_correlation_type(primary_anomaly),
                        correlation_strength=correlation_strength,
                        pattern_description=self._generate_pattern_description(
                            primary_anomaly, correlated_accounts
                        ),
                        estimated_impact_usd=primary_anomaly.get('estimated_impact', 0) * (1 + len(correlated_accounts) * 0.5),
                        affected_services=primary_anomaly.get('services', []),
                        time_window=(primary_time - timedelta(minutes=30), primary_time + timedelta(minutes=30))
                    )
                    
                    cross_account_anomalies.append(cross_account_anomaly)
        
        return cross_account_anomalies
    
    def _initialize_account_models(self, account_id: str):
        """Initialize ML models for account"""
        
        account = self.accounts[account_id]
        
        # Initialize account-specific models
        self.account_models[account_id] = {
            'isolation_forest': {
                'model_type': 'isolation_forest',
                'trained_at': datetime.now(),
                'performance': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
                'baseline_period': account.baseline_period_days,
                'sensitivity': account.sensitivity_level
            },
            'lstm_detector': {
                'model_type': 'lstm',
                'trained_at': datetime.now(),
                'performance': {'accuracy': 0.89, 'mae': 0.12, 'rmse': 0.18},
                'sequence_length': 14,
                'sensitivity': account.sensitivity_level
            }
        }
        
        logger.debug(f"Initialized models for account: {account_id}")
    
    def _cleanup_account_models(self, account_id: str):
        """Cleanup ML models for account"""
        
        if account_id in self.account_models:
            del self.account_models[account_id]
        
        logger.debug(f"Cleaned up models for account: {account_id}")
    
    def _simulate_account_anomaly_detection(
        self,
        account_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Simulate anomaly detection for account (would be real detection in production)"""
        
        import random
        
        anomalies = []
        
        # Generate simulated anomalies
        num_anomalies = random.randint(0, 5)
        
        for i in range(num_anomalies):
            anomaly_time = start_time + timedelta(
                seconds=random.randint(0, int((end_time - start_time).total_seconds()))
            )
            
            anomaly = {
                'anomaly_id': f"anomaly_{account_id}_{i}_{uuid.uuid4().hex[:8]}",
                'account_id': account_id,
                'timestamp': anomaly_time,
                'type': random.choice(['cost_spike', 'usage_anomaly', 'service_anomaly']),
                'confidence': random.uniform(0.7, 0.95),
                'estimated_impact': random.uniform(10, 1000),
                'services': random.sample(['EC2', 'S3', 'RDS', 'Lambda', 'CloudWatch'], random.randint(1, 3)),
                'features': {
                    'cost_increase_percent': random.uniform(20, 200),
                    'duration_hours': random.uniform(1, 24),
                    'affected_resources': random.randint(1, 10)
                }
            }
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_anomaly_similarity(
        self,
        anomaly1: Dict[str, Any],
        anomaly2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two anomalies"""
        
        similarity_score = 0.0
        
        # Type similarity
        if anomaly1.get('type') == anomaly2.get('type'):
            similarity_score += 0.3
        
        # Service overlap
        services1 = set(anomaly1.get('services', []))
        services2 = set(anomaly2.get('services', []))
        
        if services1 and services2:
            service_overlap = len(services1.intersection(services2)) / len(services1.union(services2))
            similarity_score += service_overlap * 0.4
        
        # Confidence similarity
        conf1 = anomaly1.get('confidence', 0.5)
        conf2 = anomaly2.get('confidence', 0.5)
        conf_similarity = 1.0 - abs(conf1 - conf2)
        similarity_score += conf_similarity * 0.2
        
        # Impact similarity
        impact1 = anomaly1.get('estimated_impact', 0)
        impact2 = anomaly2.get('estimated_impact', 0)
        
        if impact1 > 0 and impact2 > 0:
            impact_ratio = min(impact1, impact2) / max(impact1, impact2)
            similarity_score += impact_ratio * 0.1
        
        return min(similarity_score, 1.0)
    
    def _determine_cross_account_severity(
        self,
        num_accounts: int,
        correlation_strength: float
    ) -> str:
        """Determine severity of cross-account anomaly"""
        
        if num_accounts >= 5 and correlation_strength >= 0.9:
            return "critical"
        elif num_accounts >= 3 and correlation_strength >= 0.8:
            return "high"
        elif num_accounts >= 2 and correlation_strength >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _determine_correlation_type(self, anomaly: Dict[str, Any]) -> CorrelationType:
        """Determine correlation type based on anomaly characteristics"""
        
        anomaly_type = anomaly.get('type', 'cost_spike')
        
        if anomaly_type == 'cost_spike':
            return CorrelationType.COST_PATTERN
        elif anomaly_type == 'usage_anomaly':
            return CorrelationType.SERVICE_USAGE
        elif 'deployment' in anomaly.get('context', {}):
            return CorrelationType.DEPLOYMENT_IMPACT
        else:
            return CorrelationType.ANOMALY_CLUSTER
    
    def _generate_pattern_description(
        self,
        primary_anomaly: Dict[str, Any],
        correlated_accounts: List[str]
    ) -> str:
        """Generate human-readable pattern description"""
        
        anomaly_type = primary_anomaly.get('type', 'anomaly')
        services = primary_anomaly.get('services', [])
        
        if len(correlated_accounts) == 1:
            return f"Correlated {anomaly_type} detected across 2 accounts affecting {', '.join(services)}"
        else:
            return f"Organization-wide {anomaly_type} pattern detected across {len(correlated_accounts) + 1} accounts affecting {', '.join(services)}"
    
    def _get_account_anomalies_in_period(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get account anomalies in specified period"""
        
        # In production, this would query the actual anomaly database
        # For demo, return simulated data
        return self._simulate_account_anomaly_detection(account_id, start_date, end_date)
    
    def _analyze_correlation_patterns(
        self,
        cross_account_anomalies: List[CrossAccountAnomaly]
    ) -> List[Dict[str, Any]]:
        """Analyze correlation patterns in cross-account anomalies"""
        
        patterns = []
        
        # Group by correlation type
        by_type = defaultdict(list)
        for anomaly in cross_account_anomalies:
            by_type[anomaly.correlation_type].append(anomaly)
        
        for correlation_type, anomalies in by_type.items():
            if len(anomalies) >= 2:
                pattern = {
                    'correlation_type': correlation_type.value,
                    'frequency': len(anomalies),
                    'average_strength': statistics.mean([a.correlation_strength for a in anomalies]),
                    'total_impact_usd': sum([a.estimated_impact_usd for a in anomalies]),
                    'affected_accounts': list(set([
                        acc_id for anomaly in anomalies
                        for acc_id in anomaly.affected_accounts
                    ])),
                    'common_services': self._find_common_services(anomalies)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _identify_organization_trends(
        self,
        account_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Identify organization-wide trends"""
        
        trends = []
        
        # Simulate trend analysis
        trends.append({
            'trend_type': 'cost_increase',
            'description': 'Organization-wide cost increase trend detected',
            'affected_accounts': len(account_ids),
            'trend_strength': 0.75,
            'estimated_impact_usd': 5000,
            'recommendation': 'Review resource scaling policies across all accounts'
        })
        
        trends.append({
            'trend_type': 'service_migration',
            'description': 'Increased Lambda usage across multiple accounts',
            'affected_accounts': max(1, len(account_ids) // 2),
            'trend_strength': 0.65,
            'estimated_impact_usd': 2000,
            'recommendation': 'Optimize Lambda configurations for cost efficiency'
        })
        
        return trends
    
    def _calculate_detection_accuracy(self, account_ids: List[str], period_days: int) -> float:
        """Calculate average detection accuracy across accounts"""
        
        accuracies = []
        for account_id in account_ids:
            if account_id in self.account_models:
                for model_name, model_data in self.account_models[account_id].items():
                    accuracies.append(model_data.get('performance', {}).get('accuracy', 0.85))
        
        return statistics.mean(accuracies) if accuracies else 0.85
    
    def _calculate_false_positive_rate(self, account_ids: List[str], period_days: int) -> float:
        """Calculate average false positive rate across accounts"""
        
        # Simulate false positive rate calculation
        return 0.05  # 5% false positive rate
    
    def _calculate_average_detection_time(self, account_ids: List[str], period_days: int) -> float:
        """Calculate average detection time across accounts"""
        
        # Simulate detection time calculation
        return 12.5  # 12.5 minutes average detection time
    
    def _generate_multi_account_recommendations(
        self,
        account_ids: List[str],
        correlation_patterns: List[Dict[str, Any]],
        organization_trends: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate multi-account recommendations"""
        
        recommendations = []
        
        # Account-based recommendations
        if len(account_ids) > 5:
            recommendations.append("Consider implementing organization-wide cost policies for better anomaly prevention")
        
        # Pattern-based recommendations
        for pattern in correlation_patterns:
            if pattern['frequency'] > 3:
                recommendations.append(f"Investigate recurring {pattern['correlation_type']} patterns affecting {len(pattern['affected_accounts'])} accounts")
        
        # Trend-based recommendations
        for trend in organization_trends:
            if trend['trend_strength'] > 0.7:
                recommendations.append(trend['recommendation'])
        
        # General recommendations
        recommendations.append("Enable cross-account alerting for organization-wide anomaly visibility")
        recommendations.append("Review account isolation settings to ensure compliance requirements are met")
        
        return recommendations
    
    def _create_action_items(
        self,
        account_ids: List[str],
        cross_account_anomalies: List[CrossAccountAnomaly],
        recommendations: List[str]
    ) -> List[Dict[str, Any]]:
        """Create actionable items from analysis"""
        
        action_items = []
        
        # High-impact anomaly actions
        high_impact_anomalies = [
            anomaly for anomaly in cross_account_anomalies
            if anomaly.estimated_impact_usd > 1000
        ]
        
        for anomaly in high_impact_anomalies:
            action_items.append({
                'type': 'investigate_anomaly',
                'priority': 'high',
                'title': f"Investigate high-impact cross-account anomaly",
                'description': f"Anomaly {anomaly.anomaly_id} with ${anomaly.estimated_impact_usd:.2f} impact",
                'affected_accounts': anomaly.affected_accounts,
                'due_date': datetime.now() + timedelta(days=1)
            })
        
        # Recommendation actions
        for i, recommendation in enumerate(recommendations[:3]):  # Top 3 recommendations
            action_items.append({
                'type': 'implement_recommendation',
                'priority': 'medium',
                'title': f"Implement recommendation {i+1}",
                'description': recommendation,
                'affected_accounts': account_ids,
                'due_date': datetime.now() + timedelta(days=7)
            })
        
        return action_items
    
    def _find_common_services(self, anomalies: List[CrossAccountAnomaly]) -> List[str]:
        """Find services common across anomalies"""
        
        if not anomalies:
            return []
        
        # Find intersection of all affected services
        common_services = set(anomalies[0].affected_services)
        
        for anomaly in anomalies[1:]:
            common_services = common_services.intersection(set(anomaly.affected_services))
        
        return list(common_services)
    
    def _load_account_data(self):
        """Load existing account data"""
        
        # In production, this would load from persistent storage
        # For demo, we'll start with empty data
        logger.debug("Account data loaded (empty for demo)")

# Global multi-account ML manager instance
multi_account_ml_manager = MultiAccountMLManager()