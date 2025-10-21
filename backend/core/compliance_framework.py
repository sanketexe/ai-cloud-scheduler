"""
Compliance Framework and Audit Logging

This module provides comprehensive compliance management including:
- Detailed audit logging
- Compliance reporting for SOC2, GDPR, HIPAA
- Data residency controls
- Geographic restriction enforcement
- Compliance monitoring and alerting
- Automated compliance checks
"""

import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
import secrets
from collections import defaultdict, Counter
import re

# Configure compliance logging
compliance_logger = logging.getLogger('compliance')
compliance_logger.setLevel(logging.INFO)

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry data

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    PERMISSION_CHANGE = "permission_change"
    SYSTEM_ACCESS = "system_access"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"
    DATABASE_QUERY = "database_query"
    BACKUP_OPERATION = "backup_operation"
    SECURITY_EVENT = "security_event"

class GeographicRegion(Enum):
    """Geographic regions for data residency"""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    CANADA = "canada"
    AUSTRALIA = "australia"

@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    outcome: str  # SUCCESS, FAILURE, PARTIAL
    details: Dict[str, Any] = field(default_factory=dict)
    data_classification: Optional[DataClassification] = None
    geographic_region: Optional[GeographicRegion] = None
    compliance_tags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    retention_period_days: int = 2555  # 7 years default

@dataclass
class DataResidencyRule:
    """Data residency and geographic restriction rule"""
    rule_id: str
    name: str
    data_types: List[DataClassification]
    allowed_regions: List[GeographicRegion]
    prohibited_regions: List[GeographicRegion] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceCheck:
    """Individual compliance check result"""
    check_id: str
    standard: ComplianceStandard
    requirement: str
    description: str
    status: str  # PASS, FAIL, WARNING, NOT_APPLICABLE
    evidence: List[str] = field(default_factory=list)
    remediation: Optional[str] = None
    last_checked: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    standard: ComplianceStandard
    report_period: Tuple[datetime, datetime]
    generated_at: datetime
    generated_by: str
    overall_status: str  # COMPLIANT, NON_COMPLIANT, PARTIAL
    checks: List[ComplianceCheck] = field(default_factory=list)
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    next_review_date: Optional[datetime] = None

class AuditLogger:
    """Enhanced audit logging system with compliance features"""
    
    def __init__(self, storage_path: str = "audit_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, mode=0o700)
        
        # In-memory storage for recent events (for performance)
        self.recent_events: List[AuditEvent] = []
        self.max_recent_events = 10000
        
        # Data residency rules
        self.residency_rules: Dict[str, DataResidencyRule] = {}
        self._initialize_default_rules()
        
        # Compliance configurations
        self.compliance_configs = self._load_compliance_configs()
    
    def _initialize_default_rules(self):
        """Initialize default data residency rules"""
        
        # GDPR rule - EU data must stay in EU
        gdpr_rule = DataResidencyRule(
            rule_id="gdpr_eu_residency",
            name="GDPR EU Data Residency",
            data_types=[DataClassification.PII],
            allowed_regions=[GeographicRegion.EU_WEST, GeographicRegion.EU_CENTRAL],
            prohibited_regions=[GeographicRegion.US_EAST, GeographicRegion.US_WEST],
            compliance_standards=[ComplianceStandard.GDPR]
        )
        self.residency_rules[gdpr_rule.rule_id] = gdpr_rule
        
        # HIPAA rule - PHI restrictions
        hipaa_rule = DataResidencyRule(
            rule_id="hipaa_phi_residency",
            name="HIPAA PHI Data Residency",
            data_types=[DataClassification.PHI],
            allowed_regions=[GeographicRegion.US_EAST, GeographicRegion.US_WEST],
            compliance_standards=[ComplianceStandard.HIPAA]
        )
        self.residency_rules[hipaa_rule.rule_id] = hipaa_rule
    
    def _load_compliance_configs(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Load compliance standard configurations"""
        return {
            ComplianceStandard.SOC2: {
                'retention_days': 2555,  # 7 years
                'required_events': [
                    AuditEventType.USER_LOGIN,
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.CONFIGURATION_CHANGE,
                    AuditEventType.PERMISSION_CHANGE
                ],
                'monitoring_requirements': {
                    'failed_logins': {'threshold': 5, 'window_minutes': 15},
                    'privileged_access': {'log_all': True},
                    'data_changes': {'log_all': True}
                }
            },
            ComplianceStandard.GDPR: {
                'retention_days': 2190,  # 6 years
                'required_events': [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.DATA_DELETION,
                    AuditEventType.DATA_EXPORT
                ],
                'data_subject_rights': {
                    'right_to_access': True,
                    'right_to_rectification': True,
                    'right_to_erasure': True,
                    'right_to_portability': True
                }
            },
            ComplianceStandard.HIPAA: {
                'retention_days': 2190,  # 6 years
                'required_events': [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.SYSTEM_ACCESS
                ],
                'phi_requirements': {
                    'minimum_necessary': True,
                    'access_logging': True,
                    'encryption_required': True
                }
            }
        }
    
    def log_event(self, event_type: AuditEventType, user_id: Optional[str],
                 resource_type: str, resource_id: Optional[str], action: str,
                 outcome: str, ip_address: str, user_agent: str,
                 session_id: Optional[str] = None,
                 details: Dict[str, Any] = None,
                 data_classification: Optional[DataClassification] = None,
                 geographic_region: Optional[GeographicRegion] = None) -> str:
        """Log an audit event"""
        
        event_id = secrets.token_urlsafe(16)
        
        # Determine compliance tags based on data classification and event type
        compliance_tags = self._determine_compliance_tags(
            event_type, data_classification, resource_type
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            event_type, outcome, data_classification, details or {}
        )
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            data_classification=data_classification,
            geographic_region=geographic_region,
            compliance_tags=compliance_tags,
            risk_score=risk_score
        )
        
        # Add to recent events
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]
        
        # Persist to storage
        self._persist_event(event)
        
        # Check data residency compliance
        if data_classification and geographic_region:
            self._check_data_residency_compliance(event)
        
        compliance_logger.info(f"Audit event logged: {event_id} - {event_type.value}")
        return event_id
    
    def _determine_compliance_tags(self, event_type: AuditEventType,
                                 data_classification: Optional[DataClassification],
                                 resource_type: str) -> List[str]:
        """Determine which compliance standards apply to this event"""
        tags = []
        
        # SOC2 applies to most system events
        if event_type in [AuditEventType.USER_LOGIN, AuditEventType.DATA_ACCESS,
                         AuditEventType.CONFIGURATION_CHANGE, AuditEventType.PERMISSION_CHANGE]:
            tags.append(ComplianceStandard.SOC2.value)
        
        # GDPR applies to PII data events
        if data_classification == DataClassification.PII:
            tags.append(ComplianceStandard.GDPR.value)
        
        # HIPAA applies to PHI data events
        if data_classification == DataClassification.PHI:
            tags.append(ComplianceStandard.HIPAA.value)
        
        # PCI DSS applies to payment data
        if data_classification == DataClassification.PCI:
            tags.append(ComplianceStandard.PCI_DSS.value)
        
        return tags
    
    def _calculate_risk_score(self, event_type: AuditEventType, outcome: str,
                            data_classification: Optional[DataClassification],
                            details: Dict[str, Any]) -> float:
        """Calculate risk score for the event"""
        base_scores = {
            AuditEventType.USER_LOGIN: 0.1,
            AuditEventType.DATA_ACCESS: 0.3,
            AuditEventType.DATA_MODIFICATION: 0.5,
            AuditEventType.DATA_DELETION: 0.7,
            AuditEventType.DATA_EXPORT: 0.6,
            AuditEventType.CONFIGURATION_CHANGE: 0.8,
            AuditEventType.PERMISSION_CHANGE: 0.9,
            AuditEventType.SECURITY_EVENT: 0.8
        }
        
        risk_score = base_scores.get(event_type, 0.3)
        
        # Increase risk for sensitive data
        if data_classification in [DataClassification.PII, DataClassification.PHI, DataClassification.PCI]:
            risk_score += 0.2
        
        # Increase risk for failures
        if outcome == "FAILURE":
            risk_score += 0.3
        
        # Increase risk for bulk operations
        if details.get('bulk_operation', False):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _persist_event(self, event: AuditEvent):
        """Persist audit event to storage"""
        # Create daily log file
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit_{date_str}.jsonl"
        
        # Append event to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(event), default=str) + '\n')
        
        # Set secure permissions
        log_file.chmod(0o600)
    
    def _check_data_residency_compliance(self, event: AuditEvent):
        """Check if event complies with data residency rules"""
        if not event.data_classification or not event.geographic_region:
            return
        
        violations = []
        
        for rule in self.residency_rules.values():
            if not rule.is_active:
                continue
            
            if event.data_classification in rule.data_types:
                # Check if region is allowed
                if (rule.allowed_regions and 
                    event.geographic_region not in rule.allowed_regions):
                    violations.append(f"Data residency violation: {rule.name}")
                
                # Check if region is prohibited
                if event.geographic_region in rule.prohibited_regions:
                    violations.append(f"Data residency violation: {rule.name}")
        
        if violations:
            # Log compliance violation
            self.log_event(
                AuditEventType.SECURITY_EVENT,
                event.user_id,
                "compliance",
                "data_residency",
                "violation_detected",
                "FAILURE",
                event.ip_address,
                event.user_agent,
                details={
                    'violations': violations,
                    'original_event_id': event.event_id
                }
            )
    
    def get_events(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  event_types: Optional[List[AuditEventType]] = None,
                  user_id: Optional[str] = None,
                  resource_type: Optional[str] = None,
                  compliance_standard: Optional[ComplianceStandard] = None,
                  limit: int = 1000) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        
        # Start with recent events for performance
        events = self.recent_events.copy()
        
        # Load from persistent storage if needed
        if start_time or end_time:
            events.extend(self._load_events_from_storage(start_time, end_time))
        
        # Apply filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if event_types:
            event_type_values = [et.value for et in event_types]
            events = [e for e in events if e.event_type.value in event_type_values]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        
        if compliance_standard:
            events = [e for e in events if compliance_standard.value in e.compliance_tags]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    def _load_events_from_storage(self, start_time: Optional[datetime],
                                end_time: Optional[datetime]) -> List[AuditEvent]:
        """Load events from persistent storage"""
        events = []
        
        # Determine date range for files to check
        start_date = start_time.date() if start_time else datetime.now().date() - timedelta(days=30)
        end_date = end_time.date() if end_time else datetime.now().date()
        
        current_date = start_date
        while current_date <= end_date:
            log_file = self.storage_path / f"audit_{current_date.strftime('%Y-%m-%d')}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            event_data = json.loads(line.strip())
                            
                            # Convert string timestamps back to datetime
                            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                            if event_data.get('created_at'):
                                event_data['created_at'] = datetime.fromisoformat(event_data['created_at'])
                            
                            # Convert enum strings back to enums
                            event_data['event_type'] = AuditEventType(event_data['event_type'])
                            if event_data.get('data_classification'):
                                event_data['data_classification'] = DataClassification(event_data['data_classification'])
                            if event_data.get('geographic_region'):
                                event_data['geographic_region'] = GeographicRegion(event_data['geographic_region'])
                            
                            event = AuditEvent(**event_data)
                            events.append(event)
                            
                except Exception as e:
                    compliance_logger.error(f"Error loading events from {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return events

class ComplianceReporter:
    """Generates compliance reports for various standards"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.report_templates = self._load_report_templates()
    
    def _load_report_templates(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Load compliance report templates"""
        return {
            ComplianceStandard.SOC2: {
                'checks': [
                    {
                        'id': 'soc2_cc6_1',
                        'requirement': 'CC6.1 - Logical and Physical Access Controls',
                        'description': 'System implements logical and physical access controls'
                    },
                    {
                        'id': 'soc2_cc6_2',
                        'requirement': 'CC6.2 - Authentication and Authorization',
                        'description': 'System implements authentication and authorization controls'
                    },
                    {
                        'id': 'soc2_cc6_7',
                        'requirement': 'CC6.7 - Data Transmission and Disposal',
                        'description': 'System protects data during transmission and disposal'
                    },
                    {
                        'id': 'soc2_cc7_1',
                        'requirement': 'CC7.1 - System Monitoring',
                        'description': 'System implements monitoring controls'
                    }
                ]
            },
            ComplianceStandard.GDPR: {
                'checks': [
                    {
                        'id': 'gdpr_art_5',
                        'requirement': 'Article 5 - Principles of Processing',
                        'description': 'Personal data processed lawfully, fairly and transparently'
                    },
                    {
                        'id': 'gdpr_art_25',
                        'requirement': 'Article 25 - Data Protection by Design',
                        'description': 'Data protection by design and by default'
                    },
                    {
                        'id': 'gdpr_art_30',
                        'requirement': 'Article 30 - Records of Processing',
                        'description': 'Records of processing activities maintained'
                    },
                    {
                        'id': 'gdpr_art_32',
                        'requirement': 'Article 32 - Security of Processing',
                        'description': 'Appropriate technical and organizational measures'
                    }
                ]
            },
            ComplianceStandard.HIPAA: {
                'checks': [
                    {
                        'id': 'hipaa_164_308',
                        'requirement': '164.308 - Administrative Safeguards',
                        'description': 'Administrative safeguards implemented'
                    },
                    {
                        'id': 'hipaa_164_310',
                        'requirement': '164.310 - Physical Safeguards',
                        'description': 'Physical safeguards implemented'
                    },
                    {
                        'id': 'hipaa_164_312',
                        'requirement': '164.312 - Technical Safeguards',
                        'description': 'Technical safeguards implemented'
                    },
                    {
                        'id': 'hipaa_164_314',
                        'requirement': '164.314 - Organizational Requirements',
                        'description': 'Organizational requirements met'
                    }
                ]
            }
        }
    
    def generate_report(self, standard: ComplianceStandard,
                       start_date: datetime, end_date: datetime,
                       generated_by: str) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        report_id = secrets.token_urlsafe(16)
        
        # Get relevant audit events
        events = self.audit_logger.get_events(
            start_time=start_date,
            end_time=end_date,
            compliance_standard=standard,
            limit=10000
        )
        
        # Perform compliance checks
        checks = self._perform_compliance_checks(standard, events)
        
        # Generate audit summary
        audit_summary = self._generate_audit_summary(events)
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(standard, checks, events)
        
        # Calculate next review date
        next_review_date = end_date + timedelta(days=90)  # Quarterly reviews
        
        report = ComplianceReport(
            report_id=report_id,
            standard=standard,
            report_period=(start_date, end_date),
            generated_at=datetime.now(),
            generated_by=generated_by,
            overall_status=overall_status,
            checks=checks,
            audit_summary=audit_summary,
            recommendations=recommendations,
            next_review_date=next_review_date
        )
        
        return report
    
    def _perform_compliance_checks(self, standard: ComplianceStandard,
                                 events: List[AuditEvent]) -> List[ComplianceCheck]:
        """Perform compliance checks based on audit events"""
        checks = []
        template = self.report_templates.get(standard, {})
        
        for check_template in template.get('checks', []):
            check = ComplianceCheck(
                check_id=check_template['id'],
                standard=standard,
                requirement=check_template['requirement'],
                description=check_template['description'],
                status="PASS"  # Default to pass, will be updated based on analysis
            )
            
            # Perform specific checks based on standard and requirement
            if standard == ComplianceStandard.SOC2:
                check = self._check_soc2_requirement(check, events)
            elif standard == ComplianceStandard.GDPR:
                check = self._check_gdpr_requirement(check, events)
            elif standard == ComplianceStandard.HIPAA:
                check = self._check_hipaa_requirement(check, events)
            
            checks.append(check)
        
        return checks
    
    def _check_soc2_requirement(self, check: ComplianceCheck,
                              events: List[AuditEvent]) -> ComplianceCheck:
        """Check SOC2 specific requirements"""
        
        if 'cc6_1' in check.check_id.lower():
            # Check for access control events
            access_events = [e for e in events if e.event_type in [
                AuditEventType.USER_LOGIN, AuditEventType.PERMISSION_CHANGE
            ]]
            
            if access_events:
                check.status = "PASS"
                check.evidence = [f"Found {len(access_events)} access control events"]
            else:
                check.status = "FAIL"
                check.remediation = "Implement comprehensive access logging"
        
        elif 'cc6_2' in check.check_id.lower():
            # Check for authentication events
            auth_events = [e for e in events if e.event_type == AuditEventType.USER_LOGIN]
            failed_auth = [e for e in auth_events if e.outcome == "FAILURE"]
            
            if auth_events:
                check.status = "PASS"
                check.evidence = [
                    f"Found {len(auth_events)} authentication events",
                    f"Failed authentication attempts: {len(failed_auth)}"
                ]
            else:
                check.status = "WARNING"
                check.remediation = "Ensure all authentication attempts are logged"
        
        elif 'cc7_1' in check.check_id.lower():
            # Check for monitoring events
            if events:
                check.status = "PASS"
                check.evidence = [f"System monitoring active with {len(events)} events logged"]
            else:
                check.status = "FAIL"
                check.remediation = "Implement comprehensive system monitoring"
        
        return check
    
    def _check_gdpr_requirement(self, check: ComplianceCheck,
                              events: List[AuditEvent]) -> ComplianceCheck:
        """Check GDPR specific requirements"""
        
        if 'art_30' in check.check_id.lower():
            # Check for records of processing (data access events)
            pii_events = [e for e in events if e.data_classification == DataClassification.PII]
            
            if pii_events:
                check.status = "PASS"
                check.evidence = [f"Found {len(pii_events)} PII processing events"]
            else:
                check.status = "WARNING"
                check.remediation = "Ensure all PII processing is logged"
        
        elif 'art_32' in check.check_id.lower():
            # Check for security measures
            security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_EVENT]
            
            if security_events:
                check.status = "PASS"
                check.evidence = [f"Found {len(security_events)} security events"]
            else:
                check.status = "PASS"  # Absence of security events might be good
                check.evidence = ["No security incidents detected"]
        
        return check
    
    def _check_hipaa_requirement(self, check: ComplianceCheck,
                               events: List[AuditEvent]) -> ComplianceCheck:
        """Check HIPAA specific requirements"""
        
        if '164_312' in check.check_id.lower():
            # Check for technical safeguards (PHI access)
            phi_events = [e for e in events if e.data_classification == DataClassification.PHI]
            
            if phi_events:
                check.status = "PASS"
                check.evidence = [f"Found {len(phi_events)} PHI access events"]
            else:
                check.status = "NOT_APPLICABLE"
                check.evidence = ["No PHI processing detected"]
        
        return check
    
    def _generate_audit_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate summary statistics from audit events"""
        
        event_counts = Counter(e.event_type.value for e in events)
        outcome_counts = Counter(e.outcome for e in events)
        user_counts = Counter(e.user_id for e in events if e.user_id)
        
        # Risk analysis
        high_risk_events = [e for e in events if e.risk_score > 0.7]
        
        return {
            'total_events': len(events),
            'event_types': dict(event_counts),
            'outcomes': dict(outcome_counts),
            'unique_users': len(user_counts),
            'high_risk_events': len(high_risk_events),
            'average_risk_score': sum(e.risk_score for e in events) / len(events) if events else 0,
            'date_range': {
                'start': min(e.timestamp for e in events).isoformat() if events else None,
                'end': max(e.timestamp for e in events).isoformat() if events else None
            }
        }
    
    def _determine_overall_status(self, checks: List[ComplianceCheck]) -> str:
        """Determine overall compliance status"""
        
        if not checks:
            return "NOT_APPLICABLE"
        
        failed_checks = [c for c in checks if c.status == "FAIL"]
        warning_checks = [c for c in checks if c.status == "WARNING"]
        
        if failed_checks:
            return "NON_COMPLIANT"
        elif warning_checks:
            return "PARTIAL"
        else:
            return "COMPLIANT"
    
    def _generate_recommendations(self, standard: ComplianceStandard,
                                checks: List[ComplianceCheck],
                                events: List[AuditEvent]) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # Check for failed or warning checks
        for check in checks:
            if check.status in ["FAIL", "WARNING"] and check.remediation:
                recommendations.append(check.remediation)
        
        # General recommendations based on event analysis
        if len(events) < 100:
            recommendations.append("Increase audit logging coverage")
        
        high_risk_events = [e for e in events if e.risk_score > 0.7]
        if len(high_risk_events) > len(events) * 0.1:  # More than 10% high risk
            recommendations.append("Review and mitigate high-risk activities")
        
        # Standard-specific recommendations
        if standard == ComplianceStandard.GDPR:
            pii_events = [e for e in events if e.data_classification == DataClassification.PII]
            if not pii_events:
                recommendations.append("Implement PII data classification and logging")
        
        return recommendations
    
    def export_report(self, report: ComplianceReport, format: str = "json") -> str:
        """Export compliance report in specified format"""
        
        if format.lower() == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        
        elif format.lower() == "csv":
            # Export checks as CSV
            output = []
            output.append("Check ID,Standard,Requirement,Status,Evidence,Remediation")
            
            for check in report.checks:
                evidence = "; ".join(check.evidence)
                remediation = check.remediation or ""
                output.append(f"{check.check_id},{check.standard.value},{check.requirement},{check.status},{evidence},{remediation}")
            
            return "\n".join(output)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global instances
audit_logger = AuditLogger()
compliance_reporter = ComplianceReporter(audit_logger)

# Convenience functions
def log_audit_event(event_type: AuditEventType, user_id: Optional[str],
                   resource_type: str, resource_id: Optional[str], action: str,
                   outcome: str, ip_address: str, user_agent: str, **kwargs) -> str:
    """Log an audit event"""
    return audit_logger.log_event(
        event_type, user_id, resource_type, resource_id, action,
        outcome, ip_address, user_agent, **kwargs
    )

def generate_compliance_report(standard: ComplianceStandard, start_date: datetime,
                             end_date: datetime, generated_by: str) -> ComplianceReport:
    """Generate compliance report"""
    return compliance_reporter.generate_report(standard, start_date, end_date, generated_by)

def get_audit_events(start_time: Optional[datetime] = None, **kwargs) -> List[AuditEvent]:
    """Get audit events with filtering"""
    return audit_logger.get_events(start_time=start_time, **kwargs)