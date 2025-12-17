"""
Compliance Manager for Automated Cost Optimization

This module provides comprehensive compliance management for the automated cost optimization system,
including configurable retention policies, data anonymization, audit trail export, and regulatory
compliance reporting.

Requirements: 6.2, 6.3, 6.4, 6.5
"""

import json
import csv
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import re

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .database import get_db_session
from .automation_models import (
    OptimizationAction, AutomationAuditLog, ActionApproval, SafetyCheckResult,
    ActionType, ActionStatus, RiskLevel, ApprovalStatus
)
from .compliance_framework import (
    AuditLogger, ComplianceReporter, ComplianceStandard, AuditEventType,
    DataClassification, GeographicRegion, log_audit_event
)

logger = logging.getLogger(__name__)


class RetentionPeriod(Enum):
    """Standard retention periods for compliance"""
    DAYS_30 = 30
    DAYS_90 = 90
    DAYS_180 = 180
    YEAR_1 = 365
    YEARS_3 = 1095
    YEARS_6 = 2190  # GDPR/HIPAA
    YEARS_7 = 2555  # SOC2


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"


class AnonymizationLevel(Enum):
    """Levels of data anonymization"""
    NONE = "none"
    PARTIAL = "partial"  # Hash sensitive fields
    FULL = "full"       # Remove all PII/sensitive data
    PSEUDONYMIZE = "pseudonymize"  # Replace with consistent fake data


@dataclass
class RetentionPolicy:
    """Data retention policy configuration"""
    policy_id: str
    name: str
    data_types: List[str]  # Types of data this policy applies to
    retention_period_days: int
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    auto_delete: bool = False
    archive_before_delete: bool = True
    anonymize_after_days: Optional[int] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnonymizationRule:
    """Data anonymization rule"""
    rule_id: str
    field_pattern: str  # Regex pattern for field names
    anonymization_method: str  # 'hash', 'remove', 'mask', 'pseudonymize'
    applies_to_data_types: List[str]
    preserve_format: bool = False  # Keep original format (e.g., email structure)


@dataclass
class ComplianceAuditTrail:
    """Comprehensive audit trail for compliance reporting"""
    trail_id: str
    action_id: str
    event_sequence: List[Dict[str, Any]]
    compliance_metadata: Dict[str, Any]
    data_classification: Optional[DataClassification]
    retention_policy_id: str
    anonymization_applied: bool = False
    export_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ComplianceManager:
    """
    Comprehensive compliance management system for automated cost optimization.
    
    Provides:
    - Configurable retention policies
    - Data anonymization for sensitive information
    - Audit trail export in standard formats
    - Compliance reporting with regulatory support
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or next(get_db_session())
        
        # Initialize audit logger with a temporary directory for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(storage_path=temp_dir)
        self.compliance_reporter = ComplianceReporter(self.audit_logger)
        
        # Storage paths
        self.compliance_storage = Path("compliance_data")
        self.compliance_storage.mkdir(exist_ok=True, mode=0o700)
        
        # Initialize default policies and rules
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        self.anonymization_rules: Dict[str, AnonymizationRule] = {}
        self._initialize_default_policies()
        self._initialize_anonymization_rules()
        
        logger.info("ComplianceManager initialized")
    
    def _initialize_default_policies(self):
        """Initialize default retention policies for different compliance standards"""
        
        # SOC2 Policy - 7 years retention
        soc2_policy = RetentionPolicy(
            policy_id="soc2_default",
            name="SOC2 Default Retention",
            data_types=["optimization_action", "audit_log", "safety_check", "approval"],
            retention_period_days=RetentionPeriod.YEARS_7.value,
            compliance_standards=[ComplianceStandard.SOC2],
            auto_delete=False,
            archive_before_delete=True,
            anonymize_after_days=RetentionPeriod.YEARS_3.value
        )
        self.retention_policies[soc2_policy.policy_id] = soc2_policy
        
        # GDPR Policy - 6 years retention with anonymization
        gdpr_policy = RetentionPolicy(
            policy_id="gdpr_default",
            name="GDPR Data Protection Retention",
            data_types=["user_data", "pii_containing_logs"],
            retention_period_days=RetentionPeriod.YEARS_6.value,
            compliance_standards=[ComplianceStandard.GDPR],
            auto_delete=True,
            archive_before_delete=True,
            anonymize_after_days=RetentionPeriod.YEAR_1.value
        )
        self.retention_policies[gdpr_policy.policy_id] = gdpr_policy
        
        # HIPAA Policy - 6 years retention
        hipaa_policy = RetentionPolicy(
            policy_id="hipaa_default",
            name="HIPAA Healthcare Data Retention",
            data_types=["phi_containing_logs", "healthcare_actions"],
            retention_period_days=RetentionPeriod.YEARS_6.value,
            compliance_standards=[ComplianceStandard.HIPAA],
            auto_delete=False,
            archive_before_delete=True,
            anonymize_after_days=RetentionPeriod.YEARS_3.value
        )
        self.retention_policies[hipaa_policy.policy_id] = hipaa_policy
        
        # Short-term operational policy
        operational_policy = RetentionPolicy(
            policy_id="operational_default",
            name="Operational Data Retention",
            data_types=["system_logs", "performance_metrics"],
            retention_period_days=RetentionPeriod.DAYS_90.value,
            compliance_standards=[],
            auto_delete=True,
            archive_before_delete=False,
            anonymize_after_days=RetentionPeriod.DAYS_30.value
        )
        self.retention_policies[operational_policy.policy_id] = operational_policy
    
    def _initialize_anonymization_rules(self):
        """Initialize default anonymization rules"""
        
        # User identification anonymization
        user_rule = AnonymizationRule(
            rule_id="user_identification",
            field_pattern=r"(user_id|username|email|user_name)",
            anonymization_method="hash",
            applies_to_data_types=["audit_log", "optimization_action", "approval"],
            preserve_format=False
        )
        self.anonymization_rules[user_rule.rule_id] = user_rule
        
        # IP address anonymization
        ip_rule = AnonymizationRule(
            rule_id="ip_addresses",
            field_pattern=r"(ip_address|client_ip|remote_addr)",
            anonymization_method="mask",
            applies_to_data_types=["audit_log", "system_logs"],
            preserve_format=True
        )
        self.anonymization_rules[ip_rule.rule_id] = ip_rule
        
        # Resource ID partial anonymization
        resource_rule = AnonymizationRule(
            rule_id="resource_identifiers",
            field_pattern=r"(resource_id|instance_id|volume_id)",
            anonymization_method="pseudonymize",
            applies_to_data_types=["optimization_action", "audit_log"],
            preserve_format=True
        )
        self.anonymization_rules[resource_rule.rule_id] = resource_rule
        
        # Sensitive metadata removal
        metadata_rule = AnonymizationRule(
            rule_id="sensitive_metadata",
            field_pattern=r"(password|secret|key|token|credential)",
            anonymization_method="remove",
            applies_to_data_types=["audit_log", "system_logs", "optimization_action"],
            preserve_format=False
        )
        self.anonymization_rules[metadata_rule.rule_id] = metadata_rule
    
    def add_retention_policy(self, policy: RetentionPolicy) -> bool:
        """Add or update a retention policy"""
        try:
            self.retention_policies[policy.policy_id] = policy
            
            # Log the policy change
            log_audit_event(
                AuditEventType.CONFIGURATION_CHANGE,
                None,  # System action
                "retention_policy",
                policy.policy_id,
                "policy_added",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={
                    "policy_name": policy.name,
                    "retention_days": policy.retention_period_days,
                    "compliance_standards": [s.value for s in policy.compliance_standards]
                }
            )
            
            logger.info(f"Added retention policy: {policy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add retention policy {policy.policy_id}: {e}")
            return False
    
    def add_anonymization_rule(self, rule: AnonymizationRule) -> bool:
        """Add or update an anonymization rule"""
        try:
            self.anonymization_rules[rule.rule_id] = rule
            
            # Log the rule change
            log_audit_event(
                AuditEventType.CONFIGURATION_CHANGE,
                None,  # System action
                "anonymization_rule",
                rule.rule_id,
                "rule_added",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={
                    "field_pattern": rule.field_pattern,
                    "method": rule.anonymization_method,
                    "data_types": rule.applies_to_data_types
                }
            )
            
            logger.info(f"Added anonymization rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add anonymization rule {rule.rule_id}: {e}")
            return False
    
    def anonymize_data(self, data: Dict[str, Any], data_type: str, 
                      anonymization_level: AnonymizationLevel = AnonymizationLevel.PARTIAL) -> Dict[str, Any]:
        """
        Anonymize sensitive data according to configured rules
        
        Args:
            data: Data dictionary to anonymize
            data_type: Type of data (e.g., 'audit_log', 'optimization_action')
            anonymization_level: Level of anonymization to apply
            
        Returns:
            Anonymized data dictionary
        """
        if anonymization_level == AnonymizationLevel.NONE:
            return data.copy()
        
        anonymized_data = data.copy()
        
        # Apply anonymization rules
        for rule in self.anonymization_rules.values():
            if data_type not in rule.applies_to_data_types:
                continue
            
            # Find matching fields
            pattern = re.compile(rule.field_pattern, re.IGNORECASE)
            
            def anonymize_recursive(obj, path=""):
                if isinstance(obj, dict):
                    # Create a copy of items to avoid modification during iteration
                    items = list(obj.items())
                    for key, value in items:
                        current_path = f"{path}.{key}" if path else key
                        
                        if pattern.search(key):
                            obj[key] = self._apply_anonymization_method(
                                value, rule.anonymization_method, rule.preserve_format
                            )
                        elif isinstance(value, (dict, list)):
                            anonymize_recursive(value, current_path)
                
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            anonymize_recursive(item, f"{path}[{i}]")
            
            anonymize_recursive(anonymized_data)
        
        # Apply level-specific anonymization
        if anonymization_level == AnonymizationLevel.FULL:
            anonymized_data = self._apply_full_anonymization(anonymized_data)
        
        return anonymized_data
    
    def _apply_anonymization_method(self, value: Any, method: str, preserve_format: bool) -> Any:
        """Apply specific anonymization method to a value"""
        if value is None:
            return None
        
        str_value = str(value)
        
        # Ensure consistent anonymization by using the original value as seed
        if method == "hash":
            # Create consistent hash using SHA256
            hash_obj = hashlib.sha256(str_value.encode('utf-8'))
            return hash_obj.hexdigest()[:16]
        
        elif method == "remove":
            return "[REDACTED]"
        
        elif method == "mask":
            if preserve_format:
                if "@" in str_value:  # Email format
                    parts = str_value.split("@")
                    if len(parts) == 2:
                        return f"***@{parts[1]}"
                elif "." in str_value and len(str_value.split(".")) == 4:  # IP format
                    parts = str_value.split(".")
                    return f"{parts[0]}.{parts[1]}.***.***.***"
            
            # Default masking
            if len(str_value) <= 4:
                return "***"
            return str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]
        
        elif method == "pseudonymize":
            # Generate consistent pseudonym based on MD5 hash
            hash_obj = hashlib.md5(str_value.encode('utf-8'))
            hash_val = hash_obj.hexdigest()
            
            if preserve_format:
                if str_value.startswith("i-"):  # EC2 instance ID
                    return f"i-{hash_val[:17]}"
                elif str_value.startswith("vol-"):  # EBS volume ID
                    return f"vol-{hash_val[:17]}"
                elif str_value.startswith("sg-"):  # Security group ID
                    return f"sg-{hash_val[:17]}"
            
            return f"anon_{hash_val[:12]}"
        
        return value
    
    def _apply_full_anonymization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply full anonymization - remove all potentially sensitive data"""
        
        # Fields to completely remove in full anonymization
        sensitive_fields = {
            'user_id', 'username', 'email', 'ip_address', 'session_id',
            'user_agent', 'resource_id', 'instance_id', 'volume_id',
            'security_group_id', 'subnet_id', 'vpc_id'
        }
        
        def remove_sensitive_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_sensitive_recursive(v) 
                    for k, v in obj.items() 
                    if k.lower() not in sensitive_fields
                }
            elif isinstance(obj, list):
                return [remove_sensitive_recursive(item) for item in obj]
            else:
                return obj
        
        return remove_sensitive_recursive(data)
    
    def create_audit_trail(self, action_id: str, include_related_data: bool = True) -> ComplianceAuditTrail:
        """
        Create comprehensive audit trail for a specific optimization action
        
        Args:
            action_id: ID of the optimization action
            include_related_data: Whether to include related safety checks, approvals, etc.
            
        Returns:
            Complete audit trail for compliance reporting
        """
        try:
            # Get the main action
            action = self.db.query(OptimizationAction).filter(
                OptimizationAction.id == action_id
            ).first()
            
            if not action:
                raise ValueError(f"Action {action_id} not found")
            
            # Build event sequence
            event_sequence = []
            
            # Add action creation event
            event_sequence.append({
                "timestamp": action.created_at.isoformat(),
                "event_type": "action_created",
                "data": {
                    "action_type": action.action_type.value,
                    "resource_id": action.resource_id,
                    "resource_type": action.resource_type,
                    "estimated_savings": float(action.estimated_monthly_savings),
                    "risk_level": action.risk_level.value
                }
            })
            
            # Add audit log events
            audit_logs = self.db.query(AutomationAuditLog).filter(
                AutomationAuditLog.action_id == action_id
            ).order_by(AutomationAuditLog.timestamp).all()
            
            for log in audit_logs:
                event_sequence.append({
                    "timestamp": log.timestamp.isoformat(),
                    "event_type": log.event_type,
                    "data": log.event_data,
                    "user_context": log.user_context,
                    "system_context": log.system_context,
                    "correlation_id": log.correlation_id
                })
            
            if include_related_data:
                # Add safety check results
                safety_checks = self.db.query(SafetyCheckResult).filter(
                    SafetyCheckResult.action_id == action_id
                ).order_by(SafetyCheckResult.checked_at).all()
                
                for check in safety_checks:
                    event_sequence.append({
                        "timestamp": check.checked_at.isoformat(),
                        "event_type": "safety_check",
                        "data": {
                            "check_name": check.check_name,
                            "result": check.check_result,
                            "details": check.check_details
                        }
                    })
                
                # Add approval events
                approvals = self.db.query(ActionApproval).filter(
                    ActionApproval.action_id == action_id
                ).order_by(ActionApproval.requested_at).all()
                
                for approval in approvals:
                    event_sequence.append({
                        "timestamp": approval.requested_at.isoformat(),
                        "event_type": "approval_requested",
                        "data": {
                            "requested_by": approval.requested_by,
                            "status": approval.approval_status.value,
                            "approved_by": str(approval.approved_by) if approval.approved_by else None,
                            "approved_at": approval.approved_at.isoformat() if approval.approved_at else None,
                            "rejection_reason": approval.rejection_reason
                        }
                    })
            
            # Sort events by timestamp
            event_sequence.sort(key=lambda x: x["timestamp"])
            
            # Determine data classification
            data_classification = None
            if any("user" in str(event.get("data", {})).lower() for event in event_sequence):
                data_classification = DataClassification.PII
            
            # Create compliance metadata
            compliance_metadata = {
                "action_summary": {
                    "total_events": len(event_sequence),
                    "action_type": action.action_type.value,
                    "final_status": action.execution_status.value,
                    "total_savings": float(action.actual_savings or 0),
                    "risk_level": action.risk_level.value
                },
                "compliance_tags": self._determine_compliance_tags(action, event_sequence),
                "retention_requirements": self._determine_retention_requirements(action, data_classification),
                "geographic_context": self._determine_geographic_context(action)
            }
            
            # Determine applicable retention policy
            retention_policy_id = self._select_retention_policy(action, data_classification)
            
            trail = ComplianceAuditTrail(
                trail_id=secrets.token_urlsafe(16),
                action_id=action_id,
                event_sequence=event_sequence,
                compliance_metadata=compliance_metadata,
                data_classification=data_classification,
                retention_policy_id=retention_policy_id
            )
            
            # Log audit trail creation
            log_audit_event(
                AuditEventType.DATA_ACCESS,
                None,  # System action
                "audit_trail",
                trail.trail_id,
                "trail_created",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={
                    "action_id": action_id,
                    "event_count": len(event_sequence),
                    "data_classification": data_classification.value if data_classification else None
                }
            )
            
            return trail
            
        except Exception as e:
            logger.error(f"Failed to create audit trail for action {action_id}: {e}")
            raise
    
    def _determine_compliance_tags(self, action: OptimizationAction, 
                                 event_sequence: List[Dict[str, Any]]) -> List[str]:
        """Determine which compliance standards apply to this action"""
        tags = []
        
        # SOC2 applies to all automated actions
        tags.append(ComplianceStandard.SOC2.value)
        
        # Check for PII/PHI data in events
        has_pii = any(
            "user" in str(event.get("data", {})).lower() or
            "email" in str(event.get("data", {})).lower()
            for event in event_sequence
        )
        
        if has_pii:
            tags.append(ComplianceStandard.GDPR.value)
        
        # Check for healthcare-related resources (simplified check)
        if "health" in action.resource_type.lower() or "medical" in action.resource_type.lower():
            tags.append(ComplianceStandard.HIPAA.value)
        
        return tags
    
    def _determine_retention_requirements(self, action: OptimizationAction,
                                        data_classification: Optional[DataClassification]) -> Dict[str, Any]:
        """Determine retention requirements based on action and data classification"""
        
        requirements = {
            "minimum_retention_days": RetentionPeriod.YEARS_7.value,  # SOC2 default
            "anonymization_after_days": RetentionPeriod.YEARS_3.value,
            "applicable_standards": [ComplianceStandard.SOC2.value]
        }
        
        if data_classification == DataClassification.PII:
            requirements["minimum_retention_days"] = max(
                requirements["minimum_retention_days"],
                RetentionPeriod.YEARS_6.value
            )
            requirements["applicable_standards"].append(ComplianceStandard.GDPR.value)
            requirements["anonymization_after_days"] = RetentionPeriod.YEAR_1.value
        
        if data_classification == DataClassification.PHI:
            requirements["minimum_retention_days"] = max(
                requirements["minimum_retention_days"],
                RetentionPeriod.YEARS_6.value
            )
            requirements["applicable_standards"].append(ComplianceStandard.HIPAA.value)
        
        return requirements
    
    def _determine_geographic_context(self, action: OptimizationAction) -> Dict[str, Any]:
        """Determine geographic context for data residency compliance"""
        
        # Extract region from resource metadata or ID
        resource_region = None
        
        if action.resource_metadata and "region" in action.resource_metadata:
            resource_region = action.resource_metadata["region"]
        elif action.resource_id:
            # Try to extract region from AWS resource ID patterns
            if "us-east" in action.resource_id:
                resource_region = "us-east-1"
            elif "us-west" in action.resource_id:
                resource_region = "us-west-2"
            elif "eu-west" in action.resource_id:
                resource_region = "eu-west-1"
        
        return {
            "resource_region": resource_region,
            "data_residency_requirements": self._get_data_residency_requirements(resource_region),
            "cross_border_restrictions": self._get_cross_border_restrictions(resource_region)
        }
    
    def _get_data_residency_requirements(self, region: Optional[str]) -> List[str]:
        """Get data residency requirements for a region"""
        if not region:
            return []
        
        requirements = []
        
        if region.startswith("eu-"):
            requirements.append("GDPR data residency - EU data must remain in EU")
        
        if region.startswith("us-"):
            requirements.append("US data sovereignty requirements")
        
        return requirements
    
    def _get_cross_border_restrictions(self, region: Optional[str]) -> List[str]:
        """Get cross-border data transfer restrictions"""
        if not region:
            return []
        
        restrictions = []
        
        if region.startswith("eu-"):
            restrictions.append("GDPR Article 44-49 - Transfers to third countries restricted")
        
        return restrictions
    
    def _select_retention_policy(self, action: OptimizationAction,
                               data_classification: Optional[DataClassification]) -> str:
        """Select appropriate retention policy for the action"""
        
        # Default to SOC2 policy
        selected_policy = "soc2_default"
        
        # Override based on data classification
        if data_classification == DataClassification.PII:
            selected_policy = "gdpr_default"
        elif data_classification == DataClassification.PHI:
            selected_policy = "hipaa_default"
        
        return selected_policy
    
    def export_audit_trail(self, trail: ComplianceAuditTrail, 
                          export_format: ExportFormat = ExportFormat.JSON,
                          anonymization_level: AnonymizationLevel = AnonymizationLevel.PARTIAL,
                          include_metadata: bool = True) -> str:
        """
        Export audit trail in specified format with optional anonymization
        
        Args:
            trail: Audit trail to export
            export_format: Format for export (JSON, CSV, XML, PDF)
            anonymization_level: Level of anonymization to apply
            include_metadata: Whether to include compliance metadata
            
        Returns:
            Exported data as string
        """
        try:
            # Apply anonymization
            anonymized_trail = self._anonymize_audit_trail(trail, anonymization_level)
            
            # Record export event
            export_event = {
                "timestamp": datetime.now().isoformat(),
                "format": export_format.value,
                "anonymization_level": anonymization_level.value,
                "exported_by": "system",  # Could be enhanced to track actual user
                "record_count": len(anonymized_trail.event_sequence)
            }
            
            # Update export history
            trail.export_history.append(export_event)
            
            # Generate export based on format
            if export_format == ExportFormat.JSON:
                export_data = self._export_json(anonymized_trail, include_metadata)
            elif export_format == ExportFormat.CSV:
                export_data = self._export_csv(anonymized_trail, include_metadata)
            elif export_format == ExportFormat.XML:
                export_data = self._export_xml(anonymized_trail, include_metadata)
            else:
                raise ValueError(f"Unsupported export format: {export_format.value}")
            
            # Log export event
            log_audit_event(
                AuditEventType.DATA_EXPORT,
                None,  # System action
                "audit_trail",
                trail.trail_id,
                "trail_exported",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={
                    "format": export_format.value,
                    "anonymization_level": anonymization_level.value,
                    "event_count": len(trail.event_sequence)
                }
            )
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export audit trail {trail.trail_id}: {e}")
            
            # Log export failure
            log_audit_event(
                AuditEventType.DATA_EXPORT,
                None,
                "audit_trail",
                trail.trail_id,
                "trail_export_failed",
                "FAILURE",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={"error": str(e)}
            )
            
            raise
    
    def _anonymize_audit_trail(self, trail: ComplianceAuditTrail,
                             anonymization_level: AnonymizationLevel) -> ComplianceAuditTrail:
        """Apply anonymization to audit trail"""
        
        if anonymization_level == AnonymizationLevel.NONE:
            return trail
        
        # Create copy for anonymization
        anonymized_trail = ComplianceAuditTrail(
            trail_id=trail.trail_id,
            action_id=trail.action_id,
            event_sequence=[],
            compliance_metadata=trail.compliance_metadata.copy(),
            data_classification=trail.data_classification,
            retention_policy_id=trail.retention_policy_id,
            anonymization_applied=True,
            export_history=trail.export_history.copy(),
            created_at=trail.created_at
        )
        
        # Anonymize each event in sequence
        for event in trail.event_sequence:
            anonymized_event = self.anonymize_data(
                event, "audit_log", anonymization_level
            )
            anonymized_trail.event_sequence.append(anonymized_event)
        
        # Anonymize compliance metadata if full anonymization
        if anonymization_level == AnonymizationLevel.FULL:
            anonymized_trail.compliance_metadata = self.anonymize_data(
                trail.compliance_metadata, "compliance_metadata", anonymization_level
            )
        
        return anonymized_trail
    
    def _export_json(self, trail: ComplianceAuditTrail, include_metadata: bool) -> str:
        """Export audit trail as JSON"""
        
        export_data = {
            "audit_trail_id": trail.trail_id,
            "action_id": trail.action_id,
            "created_at": trail.created_at.isoformat(),
            "anonymization_applied": trail.anonymization_applied,
            "event_sequence": trail.event_sequence
        }
        
        if include_metadata:
            export_data["compliance_metadata"] = trail.compliance_metadata
            export_data["data_classification"] = trail.data_classification.value if trail.data_classification else None
            export_data["retention_policy_id"] = trail.retention_policy_id
            export_data["export_history"] = trail.export_history
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_csv(self, trail: ComplianceAuditTrail, include_metadata: bool) -> str:
        """Export audit trail as CSV"""
        
        output = []
        
        # CSV headers
        headers = [
            "timestamp", "event_type", "action_id", "trail_id"
        ]
        
        if include_metadata:
            headers.extend(["data_classification", "retention_policy"])
        
        headers.extend(["event_data", "user_context", "system_context"])
        output.append(",".join(headers))
        
        # CSV rows
        for event in trail.event_sequence:
            row = [
                event.get("timestamp", ""),
                event.get("event_type", ""),
                trail.action_id,
                trail.trail_id
            ]
            
            if include_metadata:
                row.extend([
                    trail.data_classification.value if trail.data_classification else "",
                    trail.retention_policy_id
                ])
            
            row.extend([
                json.dumps(event.get("data", {})),
                json.dumps(event.get("user_context", {})),
                json.dumps(event.get("system_context", {}))
            ])
            
            # Escape commas and quotes in CSV
            escaped_row = []
            for field in row:
                field_str = str(field)
                if "," in field_str or '"' in field_str:
                    field_str = '"' + field_str.replace('"', '""') + '"'
                escaped_row.append(field_str)
            
            output.append(",".join(escaped_row))
        
        return "\n".join(output)
    
    def _export_xml(self, trail: ComplianceAuditTrail, include_metadata: bool) -> str:
        """Export audit trail as XML"""
        
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<audit_trail>')
        xml_lines.append(f'  <trail_id>{trail.trail_id}</trail_id>')
        xml_lines.append(f'  <action_id>{trail.action_id}</action_id>')
        xml_lines.append(f'  <created_at>{trail.created_at.isoformat()}</created_at>')
        xml_lines.append(f'  <anonymization_applied>{trail.anonymization_applied}</anonymization_applied>')
        
        if include_metadata:
            xml_lines.append('  <compliance_metadata>')
            xml_lines.append(f'    <data_classification>{trail.data_classification.value if trail.data_classification else ""}</data_classification>')
            xml_lines.append(f'    <retention_policy_id>{trail.retention_policy_id}</retention_policy_id>')
            xml_lines.append('  </compliance_metadata>')
        
        xml_lines.append('  <events>')
        for event in trail.event_sequence:
            xml_lines.append('    <event>')
            xml_lines.append(f'      <timestamp>{event.get("timestamp", "")}</timestamp>')
            xml_lines.append(f'      <event_type>{event.get("event_type", "")}</event_type>')
            xml_lines.append(f'      <data><![CDATA[{json.dumps(event.get("data", {}))}]]></data>')
            xml_lines.append('    </event>')
        xml_lines.append('  </events>')
        
        xml_lines.append('</audit_trail>')
        
        return "\n".join(xml_lines)
    
    def generate_compliance_report(self, standard: ComplianceStandard,
                                 start_date: datetime, end_date: datetime,
                                 include_audit_trails: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for automated cost optimization
        
        Args:
            standard: Compliance standard to report on
            start_date: Start date for report period
            end_date: End date for report period
            include_audit_trails: Whether to include detailed audit trails
            
        Returns:
            Comprehensive compliance report
        """
        try:
            # Get optimization actions in period
            actions = self.db.query(OptimizationAction).filter(
                and_(
                    OptimizationAction.created_at >= start_date,
                    OptimizationAction.created_at <= end_date
                )
            ).all()
            
            # Generate base compliance report using framework
            base_report = self.compliance_reporter.generate_report(
                standard, start_date, end_date, "ComplianceManager"
            )
            
            # Enhance with automation-specific data
            automation_summary = {
                "total_actions": len(actions),
                "actions_by_type": self._summarize_actions_by_type(actions),
                "actions_by_status": self._summarize_actions_by_status(actions),
                "total_savings": sum(float(a.actual_savings or 0) for a in actions),
                "safety_check_summary": self._summarize_safety_checks(actions),
                "approval_workflow_summary": self._summarize_approvals(actions)
            }
            
            # Create comprehensive report
            compliance_report = {
                "report_metadata": {
                    "report_id": base_report.report_id,
                    "standard": standard.value,
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "generated_at": base_report.generated_at.isoformat(),
                    "generated_by": base_report.generated_by
                },
                "compliance_status": {
                    "overall_status": base_report.overall_status,
                    "checks_passed": len([c for c in base_report.checks if c.status == "PASS"]),
                    "checks_failed": len([c for c in base_report.checks if c.status == "FAIL"]),
                    "checks_warning": len([c for c in base_report.checks if c.status == "WARNING"])
                },
                "automation_summary": automation_summary,
                "compliance_checks": [
                    {
                        "check_id": check.check_id,
                        "requirement": check.requirement,
                        "status": check.status,
                        "evidence": check.evidence,
                        "remediation": check.remediation
                    }
                    for check in base_report.checks
                ],
                "recommendations": base_report.recommendations,
                "retention_compliance": self._assess_retention_compliance(actions, standard),
                "data_privacy_compliance": self._assess_data_privacy_compliance(actions, standard)
            }
            
            # Include audit trails if requested
            if include_audit_trails:
                audit_trails = []
                for action in actions[:10]:  # Limit to first 10 for performance
                    try:
                        trail = self.create_audit_trail(str(action.id))
                        audit_trails.append({
                            "action_id": str(action.id),
                            "trail_id": trail.trail_id,
                            "event_count": len(trail.event_sequence),
                            "compliance_tags": trail.compliance_metadata.get("compliance_tags", []),
                            "data_classification": trail.data_classification.value if trail.data_classification else None
                        })
                    except Exception as e:
                        logger.warning(f"Failed to create audit trail for action {action.id}: {e}")
                
                compliance_report["audit_trails_summary"] = {
                    "total_trails": len(audit_trails),
                    "trails": audit_trails
                }
            
            # Log report generation
            log_audit_event(
                AuditEventType.DATA_ACCESS,
                None,
                "compliance_report",
                base_report.report_id,
                "report_generated",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details={
                    "standard": standard.value,
                    "actions_analyzed": len(actions),
                    "period_days": (end_date - start_date).days
                }
            )
            
            return compliance_report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report for {standard.value}: {e}")
            raise
    
    def _summarize_actions_by_type(self, actions: List[OptimizationAction]) -> Dict[str, int]:
        """Summarize actions by type"""
        return dict(Counter(action.action_type.value for action in actions))
    
    def _summarize_actions_by_status(self, actions: List[OptimizationAction]) -> Dict[str, int]:
        """Summarize actions by status"""
        return dict(Counter(action.execution_status.value for action in actions))
    
    def _summarize_safety_checks(self, actions: List[OptimizationAction]) -> Dict[str, Any]:
        """Summarize safety check results"""
        
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        
        for action in actions:
            safety_checks = self.db.query(SafetyCheckResult).filter(
                SafetyCheckResult.action_id == action.id
            ).all()
            
            total_checks += len(safety_checks)
            passed_checks += len([c for c in safety_checks if c.check_result])
            failed_checks += len([c for c in safety_checks if not c.check_result])
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "pass_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
    
    def _summarize_approvals(self, actions: List[OptimizationAction]) -> Dict[str, Any]:
        """Summarize approval workflow results"""
        
        total_approvals = 0
        approved_count = 0
        rejected_count = 0
        pending_count = 0
        
        for action in actions:
            approvals = self.db.query(ActionApproval).filter(
                ActionApproval.action_id == action.id
            ).all()
            
            total_approvals += len(approvals)
            approved_count += len([a for a in approvals if a.approval_status == ApprovalStatus.APPROVED])
            rejected_count += len([a for a in approvals if a.approval_status == ApprovalStatus.REJECTED])
            pending_count += len([a for a in approvals if a.approval_status == ApprovalStatus.PENDING])
        
        return {
            "total_approvals": total_approvals,
            "approved": approved_count,
            "rejected": rejected_count,
            "pending": pending_count,
            "approval_rate": (approved_count / total_approvals * 100) if total_approvals > 0 else 0
        }
    
    def _assess_retention_compliance(self, actions: List[OptimizationAction],
                                   standard: ComplianceStandard) -> Dict[str, Any]:
        """Assess compliance with retention policies"""
        
        # Get applicable retention policy
        policy = None
        for p in self.retention_policies.values():
            if standard in p.compliance_standards:
                policy = p
                break
        
        if not policy:
            return {"status": "NO_POLICY", "message": "No retention policy found for standard"}
        
        # Check if any actions are approaching retention limits
        cutoff_date = datetime.now() - timedelta(days=policy.retention_period_days)
        old_actions = [a for a in actions if a.created_at < cutoff_date]
        
        # Check anonymization requirements
        anonymization_cutoff = None
        if policy.anonymize_after_days:
            anonymization_cutoff = datetime.now() - timedelta(days=policy.anonymize_after_days)
            needs_anonymization = [a for a in actions if a.created_at < anonymization_cutoff]
        else:
            needs_anonymization = []
        
        return {
            "status": "COMPLIANT" if not old_actions else "ATTENTION_REQUIRED",
            "policy_name": policy.name,
            "retention_period_days": policy.retention_period_days,
            "actions_requiring_deletion": len(old_actions),
            "actions_requiring_anonymization": len(needs_anonymization),
            "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def _assess_data_privacy_compliance(self, actions: List[OptimizationAction],
                                      standard: ComplianceStandard) -> Dict[str, Any]:
        """Assess data privacy compliance"""
        
        privacy_assessment = {
            "status": "COMPLIANT",
            "issues": [],
            "recommendations": []
        }
        
        # Check for PII in action metadata
        pii_actions = []
        for action in actions:
            if self._contains_pii(action.resource_metadata):
                pii_actions.append(action)
        
        if pii_actions and standard == ComplianceStandard.GDPR:
            privacy_assessment["issues"].append(
                f"Found {len(pii_actions)} actions with potential PII in metadata"
            )
            privacy_assessment["recommendations"].append(
                "Implement automatic PII detection and anonymization"
            )
            privacy_assessment["status"] = "ATTENTION_REQUIRED"
        
        # Check anonymization rules coverage
        if not self.anonymization_rules:
            privacy_assessment["issues"].append("No anonymization rules configured")
            privacy_assessment["recommendations"].append("Configure anonymization rules for sensitive data")
            privacy_assessment["status"] = "NON_COMPLIANT"
        
        return privacy_assessment
    
    def _contains_pii(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata contains potential PII"""
        if not metadata:
            return False
        
        pii_indicators = ["email", "user", "name", "phone", "address", "ssn"]
        metadata_str = json.dumps(metadata).lower()
        
        return any(indicator in metadata_str for indicator in pii_indicators)
    
    def cleanup_expired_data(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up expired data according to retention policies
        
        Args:
            dry_run: If True, only report what would be cleaned up without actually deleting
            
        Returns:
            Summary of cleanup actions taken or planned
        """
        cleanup_summary = {
            "dry_run": dry_run,
            "actions_to_delete": [],
            "actions_to_anonymize": [],
            "total_actions_processed": 0,
            "errors": []
        }
        
        try:
            for policy in self.retention_policies.values():
                if not policy.is_active:
                    continue
                
                # Find actions subject to this policy
                cutoff_date = datetime.now() - timedelta(days=policy.retention_period_days)
                
                # Query actions that match this policy's data types
                # This is a simplified implementation - in practice, you'd need more sophisticated matching
                actions_to_process = self.db.query(OptimizationAction).filter(
                    OptimizationAction.created_at < cutoff_date
                ).all()
                
                cleanup_summary["total_actions_processed"] += len(actions_to_process)
                
                for action in actions_to_process:
                    if policy.auto_delete:
                        cleanup_summary["actions_to_delete"].append({
                            "action_id": str(action.id),
                            "created_at": action.created_at.isoformat(),
                            "policy": policy.name,
                            "age_days": (datetime.now() - action.created_at).days
                        })
                        
                        if not dry_run:
                            # Archive before delete if required
                            if policy.archive_before_delete:
                                self._archive_action(action)
                            
                            # Delete the action (in practice, you might soft-delete)
                            # self.db.delete(action)
                    
                    # Check for anonymization requirements
                    if policy.anonymize_after_days:
                        anonymize_cutoff = datetime.now() - timedelta(days=policy.anonymize_after_days)
                        
                        if action.created_at < anonymize_cutoff:
                            cleanup_summary["actions_to_anonymize"].append({
                                "action_id": str(action.id),
                                "created_at": action.created_at.isoformat(),
                                "policy": policy.name
                            })
                            
                            if not dry_run:
                                self._anonymize_action_data(action)
            
            if not dry_run:
                self.db.commit()
            
            # Log cleanup operation
            log_audit_event(
                AuditEventType.DATA_DELETION if not dry_run else AuditEventType.DATA_ACCESS,
                None,
                "data_cleanup",
                None,
                "cleanup_executed" if not dry_run else "cleanup_planned",
                "SUCCESS",
                "127.0.0.1",
                "ComplianceManager/1.0",
                details=cleanup_summary
            )
            
        except Exception as e:
            cleanup_summary["errors"].append(str(e))
            logger.error(f"Data cleanup failed: {e}")
            
            if not dry_run:
                self.db.rollback()
        
        return cleanup_summary
    
    def _archive_action(self, action: OptimizationAction):
        """Archive action data before deletion"""
        # Create audit trail and export it
        trail = self.create_audit_trail(str(action.id))
        
        # Export as JSON for archival
        archived_data = self.export_audit_trail(
            trail, 
            ExportFormat.JSON, 
            AnonymizationLevel.PARTIAL
        )
        
        # Save to archive storage
        archive_path = self.compliance_storage / "archives" / f"{action.id}.json"
        archive_path.parent.mkdir(exist_ok=True)
        
        with open(archive_path, 'w') as f:
            f.write(archived_data)
        
        logger.info(f"Archived action {action.id} to {archive_path}")
    
    def _anonymize_action_data(self, action: OptimizationAction):
        """Anonymize action data in place"""
        # Anonymize resource metadata
        if action.resource_metadata:
            action.resource_metadata = self.anonymize_data(
                action.resource_metadata, 
                "optimization_action", 
                AnonymizationLevel.PARTIAL
            )
        
        # Mark as anonymized (you might add a flag to the model)
        # action.anonymized = True
        
        logger.info(f"Anonymized action {action.id}")
    
    def get_retention_policies(self) -> List[RetentionPolicy]:
        """Get all configured retention policies"""
        return list(self.retention_policies.values())
    
    def get_anonymization_rules(self) -> List[AnonymizationRule]:
        """Get all configured anonymization rules"""
        return list(self.anonymization_rules.values())
    
    def validate_compliance_configuration(self) -> Dict[str, Any]:
        """Validate current compliance configuration"""
        
        validation_result = {
            "status": "VALID",
            "issues": [],
            "recommendations": [],
            "policies_count": len(self.retention_policies),
            "rules_count": len(self.anonymization_rules)
        }
        
        # Check for required policies
        required_standards = [ComplianceStandard.SOC2, ComplianceStandard.GDPR]
        
        for standard in required_standards:
            has_policy = any(
                standard in policy.compliance_standards 
                for policy in self.retention_policies.values()
            )
            
            if not has_policy:
                validation_result["issues"].append(
                    f"No retention policy configured for {standard.value}"
                )
                validation_result["status"] = "INCOMPLETE"
        
        # Check for anonymization rules
        if not self.anonymization_rules:
            validation_result["issues"].append("No anonymization rules configured")
            validation_result["recommendations"].append("Configure anonymization rules for PII protection")
            validation_result["status"] = "INCOMPLETE"
        
        return validation_result