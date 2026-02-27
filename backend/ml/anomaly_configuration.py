"""
Anomaly Configuration Management System

Provides configurable anomaly detection with sensitivity level adjustment,
custom thresholds, baseline period configuration, service exclusions,
and maintenance window support for flexible anomaly detection tuning.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class SensitivityLevel(Enum):
    """Anomaly detection sensitivity levels"""
    CONSERVATIVE = "conservative"    # Low false positives, may miss subtle anomalies
    BALANCED = "balanced"           # Balanced approach - default
    AGGRESSIVE = "aggressive"       # High sensitivity, may have more false positives

class ThresholdType(Enum):
    """Types of anomaly thresholds"""
    PERCENTAGE = "percentage"       # Percentage deviation from baseline
    ABSOLUTE = "absolute"          # Absolute dollar amount deviation
    STANDARD_DEVIATION = "std_dev" # Standard deviation multiplier

class BaselinePeriod(Enum):
    """Baseline period options for anomaly detection"""
    THIRTY_DAYS = 30
    SIXTY_DAYS = 60
    NINETY_DAYS = 90

@dataclass
class ThresholdConfig:
    """Threshold configuration for anomaly detection"""
    threshold_type: ThresholdType
    value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def validate_threshold(self, cost_value: float, baseline_value: float) -> bool:
        """Validate if cost value exceeds threshold"""
        if self.threshold_type == ThresholdType.PERCENTAGE:
            deviation_pct = abs(cost_value - baseline_value) / baseline_value * 100
            return deviation_pct >= self.value
        elif self.threshold_type == ThresholdType.ABSOLUTE:
            deviation_abs = abs(cost_value - baseline_value)
            return deviation_abs >= self.value
        elif self.threshold_type == ThresholdType.STANDARD_DEVIATION:
            # Requires standard deviation calculation
            return True  # Placeholder - actual implementation in detector
        return False

@dataclass
class MaintenanceWindow:
    """Maintenance window configuration"""
    name: str
    start_time: time
    end_time: time
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    
    def is_in_window(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within maintenance window"""
        # Check date range if specified
        if self.start_date and timestamp < self.start_date:
            return False
        if self.end_date and timestamp > self.end_date:
            return False
        
        # Check day of week
        if timestamp.weekday() not in self.days_of_week:
            return False
        
        # Check time range
        current_time = timestamp.time()
        if self.start_time <= self.end_time:
            # Same day window
            return self.start_time <= current_time <= self.end_time
        else:
            # Overnight window
            return current_time >= self.start_time or current_time <= self.end_time

@dataclass
class ServiceExclusion:
    """Service exclusion configuration"""
    service_name: str
    resource_patterns: List[str] = field(default_factory=list)
    tag_filters: Dict[str, str] = field(default_factory=dict)
    reason: Optional[str] = None
    temporary: bool = False
    expires_at: Optional[datetime] = None
    
    def matches_resource(self, service: str, resource_id: str, tags: Dict[str, str] = None) -> bool:
        """Check if resource matches exclusion criteria"""
        if service != self.service_name:
            return False
        
        # Check if exclusion has expired
        if self.temporary and self.expires_at and datetime.now() > self.expires_at:
            return False
        
        # Check resource patterns
        if self.resource_patterns:
            import re
            for pattern in self.resource_patterns:
                if re.match(pattern, resource_id):
                    return True
        
        # Check tag filters
        if self.tag_filters and tags:
            for tag_key, tag_value in self.tag_filters.items():
                if tags.get(tag_key) == tag_value:
                    return True
        
        # If no specific patterns/tags, exclude entire service
        return not self.resource_patterns and not self.tag_filters

@dataclass
class AnomalyConfiguration:
    """Complete anomaly detection configuration"""
    config_id: str
    name: str
    account_id: str
    
    # Core configuration
    sensitivity_level: SensitivityLevel = SensitivityLevel.BALANCED
    baseline_period: BaselinePeriod = BaselinePeriod.THIRTY_DAYS
    
    # Threshold configuration
    primary_threshold: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(
        threshold_type=ThresholdType.PERCENTAGE,
        value=20.0
    ))
    secondary_threshold: Optional[ThresholdConfig] = None
    
    # Service and resource configuration
    service_exclusions: List[ServiceExclusion] = field(default_factory=list)
    maintenance_windows: List[MaintenanceWindow] = field(default_factory=list)
    
    # Advanced configuration
    min_cost_threshold: float = 1.0  # Minimum cost to trigger anomaly detection
    confidence_threshold: float = 0.7  # Minimum confidence for anomaly alerts
    alert_cooldown_minutes: int = 60  # Cooldown between similar alerts
    
    # Service-specific overrides
    service_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    is_active: bool = True
    
    def get_sensitivity_parameters(self) -> Dict[str, float]:
        """Get sensitivity parameters based on level"""
        sensitivity_params = {
            SensitivityLevel.CONSERVATIVE: {
                'anomaly_threshold': 0.8,
                'confidence_multiplier': 1.2,
                'noise_tolerance': 0.3,
                'min_deviation_pct': 30.0
            },
            SensitivityLevel.BALANCED: {
                'anomaly_threshold': 0.7,
                'confidence_multiplier': 1.0,
                'noise_tolerance': 0.2,
                'min_deviation_pct': 20.0
            },
            SensitivityLevel.AGGRESSIVE: {
                'anomaly_threshold': 0.5,
                'confidence_multiplier': 0.8,
                'noise_tolerance': 0.1,
                'min_deviation_pct': 10.0
            }
        }
        
        return sensitivity_params[self.sensitivity_level]
    
    def should_exclude_resource(self, service: str, resource_id: str, tags: Dict[str, str] = None) -> Tuple[bool, Optional[str]]:
        """Check if resource should be excluded from anomaly detection"""
        for exclusion in self.service_exclusions:
            if exclusion.matches_resource(service, resource_id, tags):
                return True, exclusion.reason
        return False, None
    
    def is_in_maintenance_window(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Check if timestamp is within any maintenance window"""
        for window in self.maintenance_windows:
            if window.is_in_window(timestamp):
                return True, window.name
        return False, None
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get service-specific configuration overrides"""
        return self.service_specific_configs.get(service_name, {})
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()

class AnomalyConfigurationManager:
    """
    Manager for anomaly detection configurations.
    
    Handles creation, storage, retrieval, and management of anomaly
    detection configurations with support for multiple accounts and
    service-specific settings.
    """
    
    def __init__(self, config_storage_path: Optional[str] = None):
        self.config_storage_path = config_storage_path or "anomaly_configs"
        self.configurations: Dict[str, AnomalyConfiguration] = {}
        self.default_configs: Dict[str, AnomalyConfiguration] = {}
        
        # Ensure storage directory exists
        Path(self.config_storage_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing configurations
        self._load_configurations()
        
        # Create default configurations
        self._create_default_configurations()
    
    def create_configuration(
        self,
        name: str,
        account_id: str,
        sensitivity_level: SensitivityLevel = SensitivityLevel.BALANCED,
        baseline_period: BaselinePeriod = BaselinePeriod.THIRTY_DAYS,
        created_by: Optional[str] = None
    ) -> AnomalyConfiguration:
        """Create new anomaly detection configuration"""
        
        config_id = f"config_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = AnomalyConfiguration(
            config_id=config_id,
            name=name,
            account_id=account_id,
            sensitivity_level=sensitivity_level,
            baseline_period=baseline_period,
            created_by=created_by
        )
        
        self.configurations[config_id] = config
        self._save_configuration(config)
        
        logger.info(f"Created anomaly configuration {config_id} for account {account_id}")
        return config
    
    def update_configuration(
        self,
        config_id: str,
        updates: Dict[str, Any]
    ) -> AnomalyConfiguration:
        """Update existing configuration"""
        
        if config_id not in self.configurations:
            raise ValueError(f"Configuration {config_id} not found")
        
        config = self.configurations[config_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.update_timestamp()
        self._save_configuration(config)
        
        logger.info(f"Updated anomaly configuration {config_id}")
        return config
    
    def get_configuration(self, config_id: str) -> Optional[AnomalyConfiguration]:
        """Get configuration by ID"""
        return self.configurations.get(config_id)
    
    def get_account_configurations(self, account_id: str) -> List[AnomalyConfiguration]:
        """Get all configurations for an account"""
        return [
            config for config in self.configurations.values()
            if config.account_id == account_id and config.is_active
        ]
    
    def get_active_configuration(self, account_id: str) -> Optional[AnomalyConfiguration]:
        """Get the active configuration for an account"""
        account_configs = self.get_account_configurations(account_id)
        
        if not account_configs:
            # Return default configuration
            return self.get_default_configuration(account_id)
        
        # Return the most recently updated active configuration
        return max(account_configs, key=lambda c: c.updated_at)
    
    def add_service_exclusion(
        self,
        config_id: str,
        service_name: str,
        resource_patterns: List[str] = None,
        tag_filters: Dict[str, str] = None,
        reason: str = None,
        temporary: bool = False,
        expires_at: datetime = None
    ) -> AnomalyConfiguration:
        """Add service exclusion to configuration"""
        
        config = self.get_configuration(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        exclusion = ServiceExclusion(
            service_name=service_name,
            resource_patterns=resource_patterns or [],
            tag_filters=tag_filters or {},
            reason=reason,
            temporary=temporary,
            expires_at=expires_at
        )
        
        config.service_exclusions.append(exclusion)
        config.update_timestamp()
        self._save_configuration(config)
        
        logger.info(f"Added service exclusion for {service_name} to config {config_id}")
        return config
    
    def add_maintenance_window(
        self,
        config_id: str,
        name: str,
        start_time: time,
        end_time: time,
        days_of_week: List[int],
        start_date: datetime = None,
        end_date: datetime = None,
        description: str = None
    ) -> AnomalyConfiguration:
        """Add maintenance window to configuration"""
        
        config = self.get_configuration(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        window = MaintenanceWindow(
            name=name,
            start_time=start_time,
            end_time=end_time,
            days_of_week=days_of_week,
            start_date=start_date,
            end_date=end_date,
            description=description
        )
        
        config.maintenance_windows.append(window)
        config.update_timestamp()
        self._save_configuration(config)
        
        logger.info(f"Added maintenance window '{name}' to config {config_id}")
        return config
    
    def set_service_specific_config(
        self,
        config_id: str,
        service_name: str,
        service_config: Dict[str, Any]
    ) -> AnomalyConfiguration:
        """Set service-specific configuration overrides"""
        
        config = self.get_configuration(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        config.service_specific_configs[service_name] = service_config
        config.update_timestamp()
        self._save_configuration(config)
        
        logger.info(f"Set service-specific config for {service_name} in config {config_id}")
        return config
    
    def update_sensitivity_level(
        self,
        config_id: str,
        sensitivity_level: SensitivityLevel
    ) -> AnomalyConfiguration:
        """Update sensitivity level for configuration"""
        
        return self.update_configuration(config_id, {
            'sensitivity_level': sensitivity_level
        })
    
    def update_thresholds(
        self,
        config_id: str,
        primary_threshold: ThresholdConfig,
        secondary_threshold: ThresholdConfig = None
    ) -> AnomalyConfiguration:
        """Update threshold configuration"""
        
        updates = {'primary_threshold': primary_threshold}
        if secondary_threshold:
            updates['secondary_threshold'] = secondary_threshold
        
        return self.update_configuration(config_id, updates)
    
    def update_baseline_period(
        self,
        config_id: str,
        baseline_period: BaselinePeriod
    ) -> AnomalyConfiguration:
        """Update baseline period for configuration"""
        
        return self.update_configuration(config_id, {
            'baseline_period': baseline_period
        })
    
    def delete_configuration(self, config_id: str) -> bool:
        """Delete configuration"""
        
        if config_id not in self.configurations:
            return False
        
        # Mark as inactive instead of deleting
        config = self.configurations[config_id]
        config.is_active = False
        config.update_timestamp()
        self._save_configuration(config)
        
        logger.info(f"Deactivated anomaly configuration {config_id}")
        return True
    
    def get_default_configuration(self, account_id: str) -> AnomalyConfiguration:
        """Get or create default configuration for account"""
        
        if account_id not in self.default_configs:
            self.default_configs[account_id] = self.create_configuration(
                name=f"Default Configuration - {account_id}",
                account_id=account_id,
                sensitivity_level=SensitivityLevel.BALANCED,
                baseline_period=BaselinePeriod.THIRTY_DAYS,
                created_by="system"
            )
        
        return self.default_configs[account_id]
    
    def list_configurations(
        self,
        account_id: Optional[str] = None,
        active_only: bool = True
    ) -> List[AnomalyConfiguration]:
        """List configurations with optional filtering"""
        
        configs = list(self.configurations.values())
        
        if account_id:
            configs = [c for c in configs if c.account_id == account_id]
        
        if active_only:
            configs = [c for c in configs if c.is_active]
        
        return sorted(configs, key=lambda c: c.updated_at, reverse=True)
    
    def validate_configuration(self, config: AnomalyConfiguration) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        # Validate thresholds
        if config.primary_threshold.value <= 0:
            issues.append("Primary threshold value must be positive")
        
        if config.secondary_threshold and config.secondary_threshold.value <= 0:
            issues.append("Secondary threshold value must be positive")
        
        # Validate maintenance windows
        for window in config.maintenance_windows:
            if not window.days_of_week:
                issues.append(f"Maintenance window '{window.name}' has no days specified")
            
            if any(day < 0 or day > 6 for day in window.days_of_week):
                issues.append(f"Maintenance window '{window.name}' has invalid day of week")
        
        # Validate service exclusions
        for exclusion in config.service_exclusions:
            if exclusion.temporary and not exclusion.expires_at:
                issues.append(f"Temporary exclusion for {exclusion.service_name} missing expiration date")
        
        # Validate configuration values
        if config.confidence_threshold < 0 or config.confidence_threshold > 1:
            issues.append("Confidence threshold must be between 0 and 1")
        
        if config.alert_cooldown_minutes < 0:
            issues.append("Alert cooldown must be non-negative")
        
        return issues
    
    def _load_configurations(self):
        """Load configurations from storage"""
        
        try:
            config_dir = Path(self.config_storage_path)
            
            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    config = self._deserialize_configuration(config_data)
                    self.configurations[config.config_id] = config
                    
                except Exception as e:
                    logger.error(f"Failed to load configuration from {config_file}: {e}")
            
            logger.info(f"Loaded {len(self.configurations)} anomaly configurations")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    def _save_configuration(self, config: AnomalyConfiguration):
        """Save configuration to storage"""
        
        try:
            config_file = Path(self.config_storage_path) / f"{config.config_id}.json"
            config_data = self._serialize_configuration(config)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to save configuration {config.config_id}: {e}")
    
    def _serialize_configuration(self, config: AnomalyConfiguration) -> Dict[str, Any]:
        """Serialize configuration to dictionary"""
        
        return {
            'config_id': config.config_id,
            'name': config.name,
            'account_id': config.account_id,
            'sensitivity_level': config.sensitivity_level.value,
            'baseline_period': config.baseline_period.value,
            'primary_threshold': {
                'threshold_type': config.primary_threshold.threshold_type.value,
                'value': config.primary_threshold.value,
                'min_value': config.primary_threshold.min_value,
                'max_value': config.primary_threshold.max_value
            },
            'secondary_threshold': {
                'threshold_type': config.secondary_threshold.threshold_type.value,
                'value': config.secondary_threshold.value,
                'min_value': config.secondary_threshold.min_value,
                'max_value': config.secondary_threshold.max_value
            } if config.secondary_threshold else None,
            'service_exclusions': [
                {
                    'service_name': exc.service_name,
                    'resource_patterns': exc.resource_patterns,
                    'tag_filters': exc.tag_filters,
                    'reason': exc.reason,
                    'temporary': exc.temporary,
                    'expires_at': exc.expires_at.isoformat() if exc.expires_at else None
                }
                for exc in config.service_exclusions
            ],
            'maintenance_windows': [
                {
                    'name': window.name,
                    'start_time': window.start_time.isoformat(),
                    'end_time': window.end_time.isoformat(),
                    'days_of_week': window.days_of_week,
                    'start_date': window.start_date.isoformat() if window.start_date else None,
                    'end_date': window.end_date.isoformat() if window.end_date else None,
                    'description': window.description
                }
                for window in config.maintenance_windows
            ],
            'min_cost_threshold': config.min_cost_threshold,
            'confidence_threshold': config.confidence_threshold,
            'alert_cooldown_minutes': config.alert_cooldown_minutes,
            'service_specific_configs': config.service_specific_configs,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat(),
            'created_by': config.created_by,
            'is_active': config.is_active
        }
    
    def _deserialize_configuration(self, config_data: Dict[str, Any]) -> AnomalyConfiguration:
        """Deserialize configuration from dictionary"""
        
        # Parse primary threshold
        primary_threshold_data = config_data['primary_threshold']
        primary_threshold = ThresholdConfig(
            threshold_type=ThresholdType(primary_threshold_data['threshold_type']),
            value=primary_threshold_data['value'],
            min_value=primary_threshold_data.get('min_value'),
            max_value=primary_threshold_data.get('max_value')
        )
        
        # Parse secondary threshold if present
        secondary_threshold = None
        if config_data.get('secondary_threshold'):
            secondary_threshold_data = config_data['secondary_threshold']
            secondary_threshold = ThresholdConfig(
                threshold_type=ThresholdType(secondary_threshold_data['threshold_type']),
                value=secondary_threshold_data['value'],
                min_value=secondary_threshold_data.get('min_value'),
                max_value=secondary_threshold_data.get('max_value')
            )
        
        # Parse service exclusions
        service_exclusions = []
        for exc_data in config_data.get('service_exclusions', []):
            exclusion = ServiceExclusion(
                service_name=exc_data['service_name'],
                resource_patterns=exc_data.get('resource_patterns', []),
                tag_filters=exc_data.get('tag_filters', {}),
                reason=exc_data.get('reason'),
                temporary=exc_data.get('temporary', False),
                expires_at=datetime.fromisoformat(exc_data['expires_at']) if exc_data.get('expires_at') else None
            )
            service_exclusions.append(exclusion)
        
        # Parse maintenance windows
        maintenance_windows = []
        for window_data in config_data.get('maintenance_windows', []):
            window = MaintenanceWindow(
                name=window_data['name'],
                start_time=time.fromisoformat(window_data['start_time']),
                end_time=time.fromisoformat(window_data['end_time']),
                days_of_week=window_data['days_of_week'],
                start_date=datetime.fromisoformat(window_data['start_date']) if window_data.get('start_date') else None,
                end_date=datetime.fromisoformat(window_data['end_date']) if window_data.get('end_date') else None,
                description=window_data.get('description')
            )
            maintenance_windows.append(window)
        
        return AnomalyConfiguration(
            config_id=config_data['config_id'],
            name=config_data['name'],
            account_id=config_data['account_id'],
            sensitivity_level=SensitivityLevel(config_data['sensitivity_level']),
            baseline_period=BaselinePeriod(config_data['baseline_period']),
            primary_threshold=primary_threshold,
            secondary_threshold=secondary_threshold,
            service_exclusions=service_exclusions,
            maintenance_windows=maintenance_windows,
            min_cost_threshold=config_data.get('min_cost_threshold', 1.0),
            confidence_threshold=config_data.get('confidence_threshold', 0.7),
            alert_cooldown_minutes=config_data.get('alert_cooldown_minutes', 60),
            service_specific_configs=config_data.get('service_specific_configs', {}),
            created_at=datetime.fromisoformat(config_data['created_at']),
            updated_at=datetime.fromisoformat(config_data['updated_at']),
            created_by=config_data.get('created_by'),
            is_active=config_data.get('is_active', True)
        )
    
    def _create_default_configurations(self):
        """Create default configurations for common use cases"""
        
        # These will be created on-demand per account
        pass

# Global configuration manager instance
config_manager = AnomalyConfigurationManager()