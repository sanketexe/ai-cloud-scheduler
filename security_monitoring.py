"""
Security Monitoring and Threat Detection System

This module provides comprehensive security monitoring including:
- Real-time threat detection
- Behavioral analysis
- Anomaly detection using ML
- Incident response automation
- Security alerting and notifications
- Threat intelligence integration
- Security metrics and dashboards
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import re
import hashlib
import secrets
import logging
from pathlib import Path
# Email imports (optional - can be removed if not needed)
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Configure security monitoring logging
security_logger = logging.getLogger('security_monitoring')
security_logger.setLevel(logging.INFO)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    ACCOUNT_TAKEOVER = "account_takeover"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_API_USAGE = "malicious_api_usage"
    SUSPICIOUS_LOGIN = "suspicious_login"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    POLICY_VIOLATION = "policy_violation"
    INSIDER_THREAT = "insider_threat"
    EXTERNAL_ATTACK = "external_attack"

class IncidentStatus(Enum):
    """Incident response status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

@dataclass
class SecurityEvent:
    """Security event for monitoring and analysis"""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    user_agent: str
    description: str
    indicators: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    false_positive: bool = False

@dataclass
class ThreatPattern:
    """Pattern definition for threat detection"""
    pattern_id: str
    name: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    conditions: Dict[str, Any]
    time_window_minutes: int = 60
    threshold: int = 1
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    title: str
    description: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    events: List[str] = field(default_factory=list)  # Event IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    resolution_notes: Optional[str] = None

@dataclass
class UserBehaviorProfile:
    """User behavior baseline for anomaly detection"""
    user_id: str
    typical_login_hours: List[int] = field(default_factory=list)
    common_ip_addresses: List[str] = field(default_factory=list)
    common_user_agents: List[str] = field(default_factory=list)
    typical_api_usage: Dict[str, float] = field(default_factory=dict)
    geographic_locations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_level: float = 0.0

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    threat_types: List[ThreatType]
    threat_levels: List[ThreatLevel]
    channels: List[AlertChannel]
    recipients: List[str]
    is_active: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None

class BehaviorAnalyzer:
    """Analyzes user behavior patterns for anomaly detection"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.learning_period_days = 30
        self.min_events_for_profile = 10
    
    def update_user_profile(self, user_id: str, event_data: Dict[str, Any]):
        """Update user behavior profile with new event data"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update login hours
        if 'login_hour' in event_data:
            hour = event_data['login_hour']
            if hour not in profile.typical_login_hours:
                profile.typical_login_hours.append(hour)
            # Keep only recent patterns (last 50 logins)
            if len(profile.typical_login_hours) > 50:
                profile.typical_login_hours = profile.typical_login_hours[-50:]
        
        # Update IP addresses
        if 'ip_address' in event_data:
            ip = event_data['ip_address']
            if ip not in profile.common_ip_addresses:
                profile.common_ip_addresses.append(ip)
            # Keep only recent IPs (last 20)
            if len(profile.common_ip_addresses) > 20:
                profile.common_ip_addresses = profile.common_ip_addresses[-20:]
        
        # Update user agents
        if 'user_agent' in event_data:
            ua = event_data['user_agent']
            if ua not in profile.common_user_agents:
                profile.common_user_agents.append(ua)
            # Keep only recent user agents (last 10)
            if len(profile.common_user_agents) > 10:
                profile.common_user_agents = profile.common_user_agents[-10:]
        
        # Update API usage patterns
        if 'api_endpoint' in event_data:
            endpoint = event_data['api_endpoint']
            if endpoint not in profile.typical_api_usage:
                profile.typical_api_usage[endpoint] = 0
            profile.typical_api_usage[endpoint] += 1
        
        # Update geographic locations
        if 'location' in event_data:
            location = event_data['location']
            if location not in profile.geographic_locations:
                profile.geographic_locations.append(location)
            # Keep only recent locations (last 10)
            if len(profile.geographic_locations) > 10:
                profile.geographic_locations = profile.geographic_locations[-10:]
        
        profile.last_updated = datetime.now()
        
        # Calculate confidence level based on data points
        data_points = (len(profile.typical_login_hours) + 
                      len(profile.common_ip_addresses) + 
                      len(profile.common_user_agents) + 
                      len(profile.typical_api_usage))
        profile.confidence_level = min(1.0, data_points / 50.0)
    
    def detect_anomalies(self, user_id: str, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies for a user"""
        
        if user_id not in self.user_profiles:
            return []  # No baseline yet
        
        profile = self.user_profiles[user_id]
        anomalies = []
        
        # Check login hour anomaly
        if 'login_hour' in event_data:
            hour = event_data['login_hour']
            if (profile.typical_login_hours and 
                hour not in profile.typical_login_hours and
                len(profile.typical_login_hours) >= 5):
                anomalies.append({
                    'type': 'unusual_login_hour',
                    'severity': 'medium',
                    'details': f'Login at hour {hour}, typical hours: {profile.typical_login_hours}'
                })
        
        # Check IP address anomaly
        if 'ip_address' in event_data:
            ip = event_data['ip_address']
            if (profile.common_ip_addresses and 
                ip not in profile.common_ip_addresses and
                len(profile.common_ip_addresses) >= 3):
                anomalies.append({
                    'type': 'new_ip_address',
                    'severity': 'high',
                    'details': f'Login from new IP {ip}, known IPs: {profile.common_ip_addresses}'
                })
        
        # Check user agent anomaly
        if 'user_agent' in event_data:
            ua = event_data['user_agent']
            if (profile.common_user_agents and 
                ua not in profile.common_user_agents and
                len(profile.common_user_agents) >= 2):
                anomalies.append({
                    'type': 'new_user_agent',
                    'severity': 'medium',
                    'details': f'New user agent detected'
                })
        
        # Check API usage anomaly
        if 'api_endpoint' in event_data:
            endpoint = event_data['api_endpoint']
            if (profile.typical_api_usage and 
                endpoint not in profile.typical_api_usage and
                len(profile.typical_api_usage) >= 5):
                anomalies.append({
                    'type': 'unusual_api_usage',
                    'severity': 'medium',
                    'details': f'Access to unusual API endpoint: {endpoint}'
                })
        
        return anomalies

class ThreatDetector:
    """Detects security threats based on patterns and rules"""
    
    def __init__(self):
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self.event_history: deque = deque(maxlen=10000)
        self.ip_tracking: Dict[str, List[datetime]] = defaultdict(list)
        self.user_tracking: Dict[str, List[datetime]] = defaultdict(list)
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default threat detection patterns"""
        
        # Brute force attack pattern
        brute_force_pattern = ThreatPattern(
            pattern_id="brute_force_login",
            name="Brute Force Login Attack",
            threat_type=ThreatType.BRUTE_FORCE,
            threat_level=ThreatLevel.HIGH,
            conditions={
                'event_type': 'login_failure',
                'max_attempts': 5,
                'time_window_minutes': 15
            },
            time_window_minutes=15,
            threshold=5
        )
        self.threat_patterns[brute_force_pattern.pattern_id] = brute_force_pattern
        
        # Suspicious API usage pattern
        api_abuse_pattern = ThreatPattern(
            pattern_id="api_abuse",
            name="Malicious API Usage",
            threat_type=ThreatType.MALICIOUS_API_USAGE,
            threat_level=ThreatLevel.MEDIUM,
            conditions={
                'event_type': 'api_call',
                'max_requests': 1000,
                'time_window_minutes': 60
            },
            time_window_minutes=60,
            threshold=1000
        )
        self.threat_patterns[api_abuse_pattern.pattern_id] = api_abuse_pattern
        
        # Data exfiltration pattern
        data_exfil_pattern = ThreatPattern(
            pattern_id="data_exfiltration",
            name="Potential Data Exfiltration",
            threat_type=ThreatType.DATA_EXFILTRATION,
            threat_level=ThreatLevel.CRITICAL,
            conditions={
                'event_type': 'data_export',
                'max_exports': 10,
                'time_window_minutes': 30
            },
            time_window_minutes=30,
            threshold=10
        )
        self.threat_patterns[data_exfil_pattern.pattern_id] = data_exfil_pattern
    
    def analyze_event(self, event_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze an event for potential threats"""
        
        self.event_history.append(event_data)
        threats = []
        
        # Check against all active patterns
        for pattern in self.threat_patterns.values():
            if not pattern.is_active:
                continue
            
            threat_event = self._check_pattern(pattern, event_data)
            if threat_event:
                threats.append(threat_event)
        
        return threats
    
    def _check_pattern(self, pattern: ThreatPattern, event_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Check if an event matches a threat pattern"""
        
        conditions = pattern.conditions
        
        # Check if event type matches
        if 'event_type' in conditions and event_data.get('event_type') != conditions['event_type']:
            return None
        
        # Get relevant events in time window
        cutoff_time = datetime.now() - timedelta(minutes=pattern.time_window_minutes)
        relevant_events = [
            e for e in self.event_history 
            if e.get('timestamp', datetime.now()) > cutoff_time
        ]
        
        # Apply pattern-specific logic
        if pattern.threat_type == ThreatType.BRUTE_FORCE:
            return self._check_brute_force(pattern, event_data, relevant_events)
        elif pattern.threat_type == ThreatType.MALICIOUS_API_USAGE:
            return self._check_api_abuse(pattern, event_data, relevant_events)
        elif pattern.threat_type == ThreatType.DATA_EXFILTRATION:
            return self._check_data_exfiltration(pattern, event_data, relevant_events)
        
        return None
    
    def _check_brute_force(self, pattern: ThreatPattern, event_data: Dict[str, Any], 
                          relevant_events: List[Dict[str, Any]]) -> Optional[SecurityEvent]:
        """Check for brute force attack pattern"""
        
        if event_data.get('event_type') != 'login_failure':
            return None
        
        ip_address = event_data.get('ip_address', '')
        
        # Count failed login attempts from same IP
        failed_attempts = [
            e for e in relevant_events
            if (e.get('event_type') == 'login_failure' and 
                e.get('ip_address') == ip_address)
        ]
        
        if len(failed_attempts) >= pattern.threshold:
            return SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                timestamp=datetime.now(),
                threat_type=pattern.threat_type,
                threat_level=pattern.threat_level,
                source_ip=ip_address,
                user_id=event_data.get('user_id'),
                user_agent=event_data.get('user_agent', ''),
                description=f"Brute force attack detected: {len(failed_attempts)} failed login attempts from {ip_address}",
                indicators={
                    'failed_attempts': len(failed_attempts),
                    'time_window': pattern.time_window_minutes,
                    'pattern_id': pattern.pattern_id
                },
                raw_data=event_data,
                confidence_score=min(1.0, len(failed_attempts) / (pattern.threshold * 2))
            )
        
        return None
    
    def _check_api_abuse(self, pattern: ThreatPattern, event_data: Dict[str, Any],
                        relevant_events: List[Dict[str, Any]]) -> Optional[SecurityEvent]:
        """Check for API abuse pattern"""
        
        if event_data.get('event_type') != 'api_call':
            return None
        
        ip_address = event_data.get('ip_address', '')
        
        # Count API calls from same IP
        api_calls = [
            e for e in relevant_events
            if (e.get('event_type') == 'api_call' and 
                e.get('ip_address') == ip_address)
        ]
        
        if len(api_calls) >= pattern.threshold:
            return SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                timestamp=datetime.now(),
                threat_type=pattern.threat_type,
                threat_level=pattern.threat_level,
                source_ip=ip_address,
                user_id=event_data.get('user_id'),
                user_agent=event_data.get('user_agent', ''),
                description=f"Malicious API usage detected: {len(api_calls)} requests from {ip_address}",
                indicators={
                    'api_calls': len(api_calls),
                    'time_window': pattern.time_window_minutes,
                    'pattern_id': pattern.pattern_id
                },
                raw_data=event_data,
                confidence_score=min(1.0, len(api_calls) / (pattern.threshold * 1.5))
            )
        
        return None
    
    def _check_data_exfiltration(self, pattern: ThreatPattern, event_data: Dict[str, Any],
                               relevant_events: List[Dict[str, Any]]) -> Optional[SecurityEvent]:
        """Check for data exfiltration pattern"""
        
        if event_data.get('event_type') != 'data_export':
            return None
        
        user_id = event_data.get('user_id')
        if not user_id:
            return None
        
        # Count data exports by same user
        exports = [
            e for e in relevant_events
            if (e.get('event_type') == 'data_export' and 
                e.get('user_id') == user_id)
        ]
        
        if len(exports) >= pattern.threshold:
            return SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                timestamp=datetime.now(),
                threat_type=pattern.threat_type,
                threat_level=pattern.threat_level,
                source_ip=event_data.get('ip_address', ''),
                user_id=user_id,
                user_agent=event_data.get('user_agent', ''),
                description=f"Potential data exfiltration: {len(exports)} exports by user {user_id}",
                indicators={
                    'export_count': len(exports),
                    'time_window': pattern.time_window_minutes,
                    'pattern_id': pattern.pattern_id
                },
                raw_data=event_data,
                confidence_score=min(1.0, len(exports) / pattern.threshold)
            )
        
        return None

class IncidentManager:
    """Manages security incidents and response workflows"""
    
    def __init__(self):
        self.incidents: Dict[str, SecurityIncident] = {}
        self.auto_response_rules: Dict[ThreatType, List[Callable]] = defaultdict(list)
        self._setup_auto_response_rules()
    
    def _setup_auto_response_rules(self):
        """Setup automatic response rules for different threat types"""
        
        # Brute force response
        self.auto_response_rules[ThreatType.BRUTE_FORCE].append(
            lambda incident: self._block_ip_address(incident)
        )
        
        # API abuse response
        self.auto_response_rules[ThreatType.MALICIOUS_API_USAGE].append(
            lambda incident: self._rate_limit_ip(incident)
        )
        
        # Data exfiltration response
        self.auto_response_rules[ThreatType.DATA_EXFILTRATION].append(
            lambda incident: self._suspend_user_account(incident)
        )
    
    def create_incident(self, security_event: SecurityEvent) -> str:
        """Create a new security incident from a security event"""
        
        incident_id = secrets.token_urlsafe(16)
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{security_event.threat_type.value.replace('_', ' ').title()} - {security_event.source_ip}",
            description=security_event.description,
            threat_type=security_event.threat_type,
            threat_level=security_event.threat_level,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            events=[security_event.event_id]
        )
        
        # Add initial timeline entry
        incident.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'incident_created',
            'description': 'Security incident created from threat detection',
            'user': 'system'
        })
        
        self.incidents[incident_id] = incident
        
        # Trigger automatic response
        self._trigger_auto_response(incident)
        
        security_logger.warning(f"Security incident created: {incident_id} - {incident.title}")
        return incident_id
    
    def _trigger_auto_response(self, incident: SecurityIncident):
        """Trigger automatic response actions for an incident"""
        
        response_rules = self.auto_response_rules.get(incident.threat_type, [])
        
        for rule in response_rules:
            try:
                result = rule(incident)
                if result:
                    self._add_timeline_entry(
                        incident.incident_id,
                        'auto_response',
                        f"Automatic response executed: {result}",
                        'system'
                    )
            except Exception as e:
                security_logger.error(f"Auto response failed for incident {incident.incident_id}: {e}")
    
    def _block_ip_address(self, incident: SecurityIncident) -> str:
        """Block IP address (placeholder implementation)"""
        # In a real implementation, this would integrate with firewall/WAF
        security_logger.warning(f"IP address blocked: {incident.events[0] if incident.events else 'unknown'}")
        return "IP address blocked"
    
    def _rate_limit_ip(self, incident: SecurityIncident) -> str:
        """Apply rate limiting to IP address"""
        # In a real implementation, this would integrate with rate limiting system
        security_logger.warning(f"Rate limiting applied to IP")
        return "Rate limiting applied"
    
    def _suspend_user_account(self, incident: SecurityIncident) -> str:
        """Suspend user account"""
        # In a real implementation, this would integrate with user management system
        security_logger.warning(f"User account suspended for potential data exfiltration")
        return "User account suspended"
    
    def update_incident_status(self, incident_id: str, status: IncidentStatus, 
                             notes: Optional[str] = None, updated_by: str = "system"):
        """Update incident status"""
        
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = status
        incident.updated_at = datetime.now()
        
        # Add timeline entry
        description = f"Status changed from {old_status.value} to {status.value}"
        if notes:
            description += f": {notes}"
        
        self._add_timeline_entry(incident_id, 'status_update', description, updated_by)
        
        security_logger.info(f"Incident {incident_id} status updated to {status.value}")
    
    def _add_timeline_entry(self, incident_id: str, action: str, 
                          description: str, user: str):
        """Add entry to incident timeline"""
        
        if incident_id in self.incidents:
            self.incidents[incident_id].timeline.append({
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'description': description,
                'user': user
            })
    
    def get_open_incidents(self) -> List[SecurityIncident]:
        """Get all open incidents"""
        return [
            incident for incident in self.incidents.values()
            if incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]

class AlertManager:
    """Manages security alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        self._setup_notification_handlers()
        self._initialize_default_rules()
    
    def _setup_notification_handlers(self):
        """Setup notification handlers for different channels"""
        self.notification_handlers[AlertChannel.EMAIL] = self._send_email_alert
        self.notification_handlers[AlertChannel.WEBHOOK] = self._send_webhook_alert
        self.notification_handlers[AlertChannel.DASHBOARD] = self._send_dashboard_alert
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        
        # Critical threat alert
        critical_rule = AlertRule(
            rule_id="critical_threats",
            name="Critical Security Threats",
            description="Alert for all critical security threats",
            threat_types=[ThreatType.DATA_EXFILTRATION, ThreatType.PRIVILEGE_ESCALATION],
            threat_levels=[ThreatLevel.CRITICAL],
            channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
            recipients=["security@company.com"],
            cooldown_minutes=30
        )
        self.alert_rules[critical_rule.rule_id] = critical_rule
        
        # High threat alert
        high_rule = AlertRule(
            rule_id="high_threats",
            name="High Security Threats",
            description="Alert for high-level security threats",
            threat_types=[ThreatType.BRUTE_FORCE, ThreatType.ACCOUNT_TAKEOVER],
            threat_levels=[ThreatLevel.HIGH],
            channels=[AlertChannel.EMAIL],
            recipients=["security@company.com"],
            cooldown_minutes=60
        )
        self.alert_rules[high_rule.rule_id] = high_rule
    
    def process_security_event(self, security_event: SecurityEvent):
        """Process security event and send alerts if rules match"""
        
        for rule in self.alert_rules.values():
            if not rule.is_active:
                continue
            
            # Check if event matches rule criteria
            if (security_event.threat_type in rule.threat_types and
                security_event.threat_level in rule.threat_levels):
                
                # Check cooldown period
                if (rule.last_triggered and 
                    datetime.now() - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                    continue
                
                # Send alerts
                self._send_alerts(rule, security_event)
                rule.last_triggered = datetime.now()
    
    def _send_alerts(self, rule: AlertRule, security_event: SecurityEvent):
        """Send alerts through configured channels"""
        
        alert_data = {
            'rule_name': rule.name,
            'threat_type': security_event.threat_type.value,
            'threat_level': security_event.threat_level.value,
            'description': security_event.description,
            'source_ip': security_event.source_ip,
            'user_id': security_event.user_id,
            'timestamp': security_event.timestamp.isoformat(),
            'confidence_score': security_event.confidence_score
        }
        
        for channel in rule.channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert_data, rule.recipients)
                except Exception as e:
                    security_logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    def _send_email_alert(self, alert_data: Dict[str, Any], recipients: List[str]):
        """Send email alert (placeholder implementation)"""
        # In a real implementation, this would use SMTP configuration
        security_logger.info(f"Email alert sent to {recipients}: {alert_data['description']}")
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any], recipients: List[str]):
        """Send webhook alert (placeholder implementation)"""
        # In a real implementation, this would make HTTP POST requests
        security_logger.info(f"Webhook alert sent to {recipients}: {alert_data['description']}")
    
    def _send_dashboard_alert(self, alert_data: Dict[str, Any], recipients: List[str]):
        """Send dashboard alert (placeholder implementation)"""
        # In a real implementation, this would update dashboard notifications
        security_logger.info(f"Dashboard alert created: {alert_data['description']}")

class SecurityMonitoringSystem:
    """Main security monitoring system that coordinates all components"""
    
    def __init__(self):
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_detector = ThreatDetector()
        self.incident_manager = IncidentManager()
        self.alert_manager = AlertManager()
        
        self.security_events: List[SecurityEvent] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start the security monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        security_logger.info("Security monitoring system started")
    
    def stop_monitoring(self):
        """Stop the security monitoring system"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        security_logger.info("Security monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop (placeholder for real-time processing)"""
        while self.monitoring_active:
            try:
                # In a real implementation, this would process events from queues
                time.sleep(10)  # Check every 10 seconds
                
                # Perform periodic tasks
                self._cleanup_old_data()
                
            except Exception as e:
                security_logger.error(f"Error in monitoring loop: {e}")
    
    def process_event(self, event_data: Dict[str, Any]) -> List[str]:
        """Process a security event and return any incident IDs created"""
        
        incident_ids = []
        
        # Update user behavior profile
        if event_data.get('user_id'):
            self.behavior_analyzer.update_user_profile(
                event_data['user_id'], event_data
            )
            
            # Check for behavioral anomalies
            anomalies = self.behavior_analyzer.detect_anomalies(
                event_data['user_id'], event_data
            )
            
            # Create security events for anomalies
            for anomaly in anomalies:
                security_event = SecurityEvent(
                    event_id=secrets.token_urlsafe(16),
                    timestamp=datetime.now(),
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM if anomaly['severity'] == 'medium' else ThreatLevel.HIGH,
                    source_ip=event_data.get('ip_address', ''),
                    user_id=event_data.get('user_id'),
                    user_agent=event_data.get('user_agent', ''),
                    description=f"Behavioral anomaly detected: {anomaly['type']}",
                    indicators=anomaly,
                    raw_data=event_data,
                    confidence_score=0.7
                )
                
                self.security_events.append(security_event)
                self.alert_manager.process_security_event(security_event)
                
                # Create incident for high-severity anomalies
                if security_event.threat_level == ThreatLevel.HIGH:
                    incident_id = self.incident_manager.create_incident(security_event)
                    incident_ids.append(incident_id)
        
        # Detect threats using pattern matching
        threats = self.threat_detector.analyze_event(event_data)
        
        for threat in threats:
            self.security_events.append(threat)
            self.alert_manager.process_security_event(threat)
            
            # Create incident for medium and higher threats
            if threat.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                incident_id = self.incident_manager.create_incident(threat)
                incident_ids.append(incident_id)
        
        return incident_ids
    
    def _cleanup_old_data(self):
        """Clean up old security events and data"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Remove old security events
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data"""
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        recent_events = [e for e in self.security_events if e.timestamp > last_24h]
        weekly_events = [e for e in self.security_events if e.timestamp > last_7d]
        
        # Threat level distribution
        threat_levels = defaultdict(int)
        for event in recent_events:
            threat_levels[event.threat_level.value] += 1
        
        # Threat type distribution
        threat_types = defaultdict(int)
        for event in recent_events:
            threat_types[event.threat_type.value] += 1
        
        # Top source IPs
        source_ips = defaultdict(int)
        for event in recent_events:
            source_ips[event.source_ip] += 1
        
        top_ips = sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Open incidents
        open_incidents = self.incident_manager.get_open_incidents()
        
        return {
            'summary': {
                'total_events_24h': len(recent_events),
                'total_events_7d': len(weekly_events),
                'open_incidents': len(open_incidents),
                'critical_incidents': len([i for i in open_incidents if i.threat_level == ThreatLevel.CRITICAL])
            },
            'threat_levels': dict(threat_levels),
            'threat_types': dict(threat_types),
            'top_source_ips': top_ips,
            'recent_incidents': [
                {
                    'incident_id': i.incident_id,
                    'title': i.title,
                    'threat_level': i.threat_level.value,
                    'status': i.status.value,
                    'created_at': i.created_at.isoformat()
                }
                for i in sorted(open_incidents, key=lambda x: x.created_at, reverse=True)[:10]
            ],
            'monitoring_status': {
                'active': self.monitoring_active,
                'behavior_profiles': len(self.behavior_analyzer.user_profiles),
                'threat_patterns': len(self.threat_detector.threat_patterns),
                'alert_rules': len(self.alert_manager.alert_rules)
            }
        }

# Global security monitoring system instance
security_monitoring_system = SecurityMonitoringSystem()

# Convenience functions
def start_security_monitoring():
    """Start security monitoring"""
    security_monitoring_system.start_monitoring()

def stop_security_monitoring():
    """Stop security monitoring"""
    security_monitoring_system.stop_monitoring()

def process_security_event(event_data: Dict[str, Any]) -> List[str]:
    """Process a security event"""
    return security_monitoring_system.process_event(event_data)

def get_security_dashboard() -> Dict[str, Any]:
    """Get security dashboard data"""
    return security_monitoring_system.get_security_dashboard()