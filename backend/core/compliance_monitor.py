"""
Compliance Monitoring and Detection System

This module provides real-time resource scanning, violation detection,
and compliance scoring capabilities for tagging governance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
from collections import defaultdict, deque

from tagging_policy_manager import TagPolicyManager, TaggingPolicy

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of compliance violations"""
    MISSING_MANDATORY_TAG = "missing_mandatory_tag"
    FORBIDDEN_TAG_PRESENT = "forbidden_tag_present"
    INVALID_TAG_VALUE = "invalid_tag_value"
    POLICY_CONFLICT = "policy_conflict"
    UNTAGGED_RESOURCE = "untagged_resource"


class ViolationSeverity(Enum):
    """Severity levels for violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceStatus(Enum):
    """Overall compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class CloudResource:
    """Represents a cloud resource for compliance monitoring"""
    resource_id: str
    resource_type: str
    provider: str
    region: str
    tags: Dict[str, str]
    attributes: Dict[str, Any]
    created_at: datetime
    last_modified: datetime
    
    def get_resource_attributes(self) -> Dict[str, Any]:
        """Get resource attributes for policy matching"""
        return {
            "resource_type": self.resource_type,
            "provider": self.provider,
            "region": self.region,
            **self.attributes
        }


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    violation_id: str
    resource_id: str
    violation_type: ViolationType
    severity: ViolationSeverity
    policy_id: str
    tag_key: Optional[str]
    tag_value: Optional[str]
    description: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[str] = None
    
    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None


@dataclass
class ComplianceScore:
    """Compliance scoring metrics"""
    overall_score: float  # 0-100
    total_resources: int
    compliant_resources: int
    non_compliant_resources: int
    violation_count: int
    critical_violations: int
    high_violations: int
    medium_violations: int
    low_violations: int
    score_trend: List[float] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def compliance_rate(self) -> float:
        if self.total_resources == 0:
            return 100.0
        return (self.compliant_resources / self.total_resources) * 100


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    generated_at: datetime
    time_period: Dict[str, datetime]
    compliance_score: ComplianceScore
    violations: List[ComplianceViolation]
    resource_summary: Dict[str, Any]
    policy_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ScanConfiguration:
    """Configuration for compliance scanning"""
    scan_interval: int = 3600  # seconds
    batch_size: int = 100
    parallel_workers: int = 5
    include_resource_types: Optional[List[str]] = None
    exclude_resource_types: Optional[List[str]] = None
    include_regions: Optional[List[str]] = None
    exclude_regions: Optional[List[str]] = None
    severity_threshold: ViolationSeverity = ViolationSeverity.LOW


class ComplianceMonitor:
    """
    Real-time compliance monitoring system that scans resources,
    detects violations, and provides compliance scoring and analytics.
    """
    
    def __init__(self, policy_manager: TagPolicyManager):
        self.policy_manager = policy_manager
        self.violations: Dict[str, ComplianceViolation] = {}
        self.compliance_history: deque = deque(maxlen=1000)  # Store last 1000 scores
        self.scan_config = ScanConfiguration()
        self.resource_cache: Dict[str, CloudResource] = {}
        self.last_scan_time: Optional[datetime] = None
        self.scan_statistics = {
            "total_scans": 0,
            "resources_scanned": 0,
            "violations_detected": 0,
            "scan_duration_avg": 0.0
        }
        
        # Callbacks for real-time notifications
        self.violation_callbacks: List[Callable[[ComplianceViolation], None]] = []
        self.compliance_callbacks: List[Callable[[ComplianceScore], None]] = []
    
    def configure_scanning(self, config: ScanConfiguration):
        """Configure compliance scanning parameters"""
        self.scan_config = config
        logger.info(f"Updated scan configuration: interval={config.scan_interval}s, batch_size={config.batch_size}")
    
    def add_violation_callback(self, callback: Callable[[ComplianceViolation], None]):
        """Add callback for violation notifications"""
        self.violation_callbacks.append(callback)
    
    def add_compliance_callback(self, callback: Callable[[ComplianceScore], None]):
        """Add callback for compliance score updates"""
        self.compliance_callbacks.append(callback)
    
    async def scan_resources(self, resources: List[CloudResource]) -> ComplianceScore:
        """
        Scan a list of resources for compliance violations
        """
        scan_start = datetime.now()
        
        try:
            # Filter resources based on configuration
            filtered_resources = self._filter_resources(resources)
            
            # Process resources in batches
            all_violations = []
            
            with ThreadPoolExecutor(max_workers=self.scan_config.parallel_workers) as executor:
                # Split resources into batches
                batches = [
                    filtered_resources[i:i + self.scan_config.batch_size]
                    for i in range(0, len(filtered_resources), self.scan_config.batch_size)
                ]
                
                # Process batches concurrently
                futures = [
                    executor.submit(self._scan_resource_batch, batch)
                    for batch in batches
                ]
                
                # Collect results
                for future in futures:
                    batch_violations = future.result()
                    all_violations.extend(batch_violations)
            
            # Update violation store
            for violation in all_violations:
                self.violations[violation.violation_id] = violation
                
                # Notify callbacks
                for callback in self.violation_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"Violation callback failed: {str(e)}")
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(filtered_resources, all_violations)
            
            # Update statistics
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self._update_scan_statistics(len(filtered_resources), len(all_violations), scan_duration)
            
            # Store compliance history
            self.compliance_history.append(compliance_score)
            
            # Notify compliance callbacks
            for callback in self.compliance_callbacks:
                try:
                    callback(compliance_score)
                except Exception as e:
                    logger.error(f"Compliance callback failed: {str(e)}")
            
            self.last_scan_time = datetime.now()
            logger.info(f"Compliance scan completed: {len(filtered_resources)} resources, "
                       f"{len(all_violations)} violations, {scan_duration:.2f}s")
            
            return compliance_score
            
        except Exception as e:
            logger.error(f"Compliance scan failed: {str(e)}")
            raise
    
    def _filter_resources(self, resources: List[CloudResource]) -> List[CloudResource]:
        """Filter resources based on scan configuration"""
        filtered = resources
        
        if self.scan_config.include_resource_types:
            filtered = [r for r in filtered if r.resource_type in self.scan_config.include_resource_types]
        
        if self.scan_config.exclude_resource_types:
            filtered = [r for r in filtered if r.resource_type not in self.scan_config.exclude_resource_types]
        
        if self.scan_config.include_regions:
            filtered = [r for r in filtered if r.region in self.scan_config.include_regions]
        
        if self.scan_config.exclude_regions:
            filtered = [r for r in filtered if r.region not in self.scan_config.exclude_regions]
        
        return filtered
    
    def _scan_resource_batch(self, resources: List[CloudResource]) -> List[ComplianceViolation]:
        """Scan a batch of resources for violations"""
        violations = []
        
        for resource in resources:
            try:
                resource_violations = self._scan_single_resource(resource)
                violations.extend(resource_violations)
                
                # Update resource cache
                self.resource_cache[resource.resource_id] = resource
                
            except Exception as e:
                logger.error(f"Failed to scan resource {resource.resource_id}: {str(e)}")
        
        return violations
    
    def _scan_single_resource(self, resource: CloudResource) -> List[ComplianceViolation]:
        """Scan a single resource for compliance violations"""
        violations = []
        
        # Get applicable policies
        resource_attributes = resource.get_resource_attributes()
        applicable_policies = self.policy_manager.get_applicable_policies(resource_attributes)
        
        if not applicable_policies:
            # No policies apply - this might be a violation itself
            if not resource.tags:  # Completely untagged resource
                violations.append(ComplianceViolation(
                    violation_id=f"{resource.resource_id}_untagged_{datetime.now().timestamp()}",
                    resource_id=resource.resource_id,
                    violation_type=ViolationType.UNTAGGED_RESOURCE,
                    severity=ViolationSeverity.MEDIUM,
                    policy_id="system",
                    tag_key=None,
                    tag_value=None,
                    description="Resource has no tags and no applicable policies",
                    detected_at=datetime.now()
                ))
            return violations
        
        # Validate against each applicable policy
        for policy in applicable_policies:
            policy_violations = self._validate_resource_against_policy(resource, policy)
            violations.extend(policy_violations)
        
        return violations
    
    def _validate_resource_against_policy(self, resource: CloudResource, 
                                        policy: TaggingPolicy) -> List[ComplianceViolation]:
        """Validate a resource against a specific policy"""
        violations = []
        
        # Use policy manager's validation
        validation_result = self.policy_manager.validate_resource_tags(
            resource.get_resource_attributes(),
            resource.tags
        )
        
        # Convert validation results to violations
        timestamp = datetime.now()
        
        # Missing mandatory tags
        for missing in validation_result.get("missing_mandatory", []):
            violations.append(ComplianceViolation(
                violation_id=f"{resource.resource_id}_{missing['tag']}_missing_{timestamp.timestamp()}",
                resource_id=resource.resource_id,
                violation_type=ViolationType.MISSING_MANDATORY_TAG,
                severity=ViolationSeverity.HIGH,
                policy_id=missing["policy"],
                tag_key=missing["tag"],
                tag_value=None,
                description=f"Missing mandatory tag: {missing['tag']} - {missing['description']}",
                detected_at=timestamp
            ))
        
        # Forbidden tags present
        for forbidden in validation_result.get("forbidden_present", []):
            violations.append(ComplianceViolation(
                violation_id=f"{resource.resource_id}_{forbidden['tag']}_forbidden_{timestamp.timestamp()}",
                resource_id=resource.resource_id,
                violation_type=ViolationType.FORBIDDEN_TAG_PRESENT,
                severity=ViolationSeverity.MEDIUM,
                policy_id=forbidden["policy"],
                tag_key=forbidden["tag"],
                tag_value=forbidden["value"],
                description=f"Forbidden tag present: {forbidden['tag']}={forbidden['value']}",
                detected_at=timestamp
            ))
        
        # Invalid tag values
        for violation in validation_result.get("violations", []):
            violations.append(ComplianceViolation(
                violation_id=f"{resource.resource_id}_{violation['tag']}_invalid_{timestamp.timestamp()}",
                resource_id=resource.resource_id,
                violation_type=ViolationType.INVALID_TAG_VALUE,
                severity=ViolationSeverity.MEDIUM,
                policy_id=violation["policy"],
                tag_key=violation["tag"],
                tag_value=violation["value"],
                description=violation["error"],
                detected_at=timestamp
            ))
        
        return violations
    
    def _calculate_compliance_score(self, resources: List[CloudResource], 
                                  violations: List[ComplianceViolation]) -> ComplianceScore:
        """Calculate overall compliance score"""
        total_resources = len(resources)
        
        if total_resources == 0:
            return ComplianceScore(
                overall_score=100.0,
                total_resources=0,
                compliant_resources=0,
                non_compliant_resources=0,
                violation_count=0,
                critical_violations=0,
                high_violations=0,
                medium_violations=0,
                low_violations=0
            )
        
        # Count violations by severity
        violation_counts = {
            ViolationSeverity.CRITICAL: 0,
            ViolationSeverity.HIGH: 0,
            ViolationSeverity.MEDIUM: 0,
            ViolationSeverity.LOW: 0
        }
        
        for violation in violations:
            if violation.severity in violation_counts:
                violation_counts[violation.severity] += 1
        
        # Count resources with violations
        resources_with_violations = set(v.resource_id for v in violations)
        non_compliant_resources = len(resources_with_violations)
        compliant_resources = total_resources - non_compliant_resources
        
        # Calculate weighted score
        # Critical violations have more impact on score
        severity_weights = {
            ViolationSeverity.CRITICAL: 10,
            ViolationSeverity.HIGH: 5,
            ViolationSeverity.MEDIUM: 2,
            ViolationSeverity.LOW: 1
        }
        
        total_violation_weight = sum(
            violation_counts[severity] * weight
            for severity, weight in severity_weights.items()
        )
        
        # Calculate score (0-100)
        if total_violation_weight == 0:
            overall_score = 100.0
        else:
            # Normalize by total resources and apply logarithmic scaling
            max_possible_weight = total_resources * severity_weights[ViolationSeverity.CRITICAL]
            raw_score = max(0, 100 - (total_violation_weight / max_possible_weight * 100))
            overall_score = max(0, min(100, raw_score))
        
        # Add trend data
        score_trend = [score.overall_score for score in list(self.compliance_history)[-10:]]
        
        return ComplianceScore(
            overall_score=overall_score,
            total_resources=total_resources,
            compliant_resources=compliant_resources,
            non_compliant_resources=non_compliant_resources,
            violation_count=len(violations),
            critical_violations=violation_counts[ViolationSeverity.CRITICAL],
            high_violations=violation_counts[ViolationSeverity.HIGH],
            medium_violations=violation_counts[ViolationSeverity.MEDIUM],
            low_violations=violation_counts[ViolationSeverity.LOW],
            score_trend=score_trend
        )
    
    def _update_scan_statistics(self, resources_scanned: int, violations_detected: int, duration: float):
        """Update scanning statistics"""
        self.scan_statistics["total_scans"] += 1
        self.scan_statistics["resources_scanned"] += resources_scanned
        self.scan_statistics["violations_detected"] += violations_detected
        
        # Update average duration
        current_avg = self.scan_statistics["scan_duration_avg"]
        total_scans = self.scan_statistics["total_scans"]
        self.scan_statistics["scan_duration_avg"] = (
            (current_avg * (total_scans - 1) + duration) / total_scans
        )
    
    def get_violations(self, resource_id: Optional[str] = None, 
                      severity: Optional[ViolationSeverity] = None,
                      resolved: Optional[bool] = None) -> List[ComplianceViolation]:
        """Get violations with optional filtering"""
        violations = list(self.violations.values())
        
        if resource_id:
            violations = [v for v in violations if v.resource_id == resource_id]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        if resolved is not None:
            violations = [v for v in violations if v.is_resolved == resolved]
        
        return sorted(violations, key=lambda v: v.detected_at, reverse=True)
    
    def resolve_violation(self, violation_id: str, resolution_action: str) -> bool:
        """Mark a violation as resolved"""
        if violation_id not in self.violations:
            return False
        
        violation = self.violations[violation_id]
        violation.resolved_at = datetime.now()
        violation.resolution_action = resolution_action
        
        logger.info(f"Resolved violation {violation_id}: {resolution_action}")
        return True
    
    def get_compliance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get compliance trends over specified period"""
        if not self.compliance_history:
            return {"error": "No compliance history available"}
        
        # Get recent scores
        recent_scores = list(self.compliance_history)[-days:] if len(self.compliance_history) >= days else list(self.compliance_history)
        
        if len(recent_scores) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        scores = [score.overall_score for score in recent_scores]
        
        # Calculate trend metrics
        trend_analysis = {
            "current_score": scores[-1],
            "average_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_variance": statistics.variance(scores) if len(scores) > 1 else 0,
            "trend_direction": "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable",
            "data_points": len(scores),
            "period_days": days
        }
        
        # Calculate improvement rate
        if len(scores) >= 2:
            score_change = scores[-1] - scores[0]
            trend_analysis["improvement_rate"] = score_change / len(scores)
        
        return trend_analysis
    
    def generate_compliance_report(self, time_period: Optional[Dict[str, datetime]] = None) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        report_id = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not time_period:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            time_period = {"start": start_time, "end": end_time}
        
        # Filter violations by time period
        period_violations = [
            v for v in self.violations.values()
            if time_period["start"] <= v.detected_at <= time_period["end"]
        ]
        
        # Get current compliance score
        current_score = self.compliance_history[-1] if self.compliance_history else ComplianceScore(
            overall_score=0, total_resources=0, compliant_resources=0,
            non_compliant_resources=0, violation_count=0,
            critical_violations=0, high_violations=0, medium_violations=0, low_violations=0
        )
        
        # Generate resource summary
        resource_summary = self._generate_resource_summary(period_violations)
        
        # Generate policy summary
        policy_summary = self._generate_policy_summary(period_violations)
        
        # Generate trend analysis
        trend_analysis = self.get_compliance_trends()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_score, period_violations)
        
        return ComplianceReport(
            report_id=report_id,
            generated_at=datetime.now(),
            time_period=time_period,
            compliance_score=current_score,
            violations=period_violations,
            resource_summary=resource_summary,
            policy_summary=policy_summary,
            trend_analysis=trend_analysis,
            recommendations=recommendations
        )
    
    def _generate_resource_summary(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate resource-level summary from violations"""
        resource_stats = defaultdict(lambda: {
            "violation_count": 0,
            "severity_breakdown": defaultdict(int),
            "violation_types": defaultdict(int)
        })
        
        for violation in violations:
            stats = resource_stats[violation.resource_id]
            stats["violation_count"] += 1
            stats["severity_breakdown"][violation.severity.value] += 1
            stats["violation_types"][violation.violation_type.value] += 1
        
        return {
            "total_resources_with_violations": len(resource_stats),
            "resource_details": dict(resource_stats),
            "most_violated_resources": sorted(
                resource_stats.items(),
                key=lambda x: x[1]["violation_count"],
                reverse=True
            )[:10]
        }
    
    def _generate_policy_summary(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate policy-level summary from violations"""
        policy_stats = defaultdict(lambda: {
            "violation_count": 0,
            "affected_resources": set(),
            "violation_types": defaultdict(int)
        })
        
        for violation in violations:
            stats = policy_stats[violation.policy_id]
            stats["violation_count"] += 1
            stats["affected_resources"].add(violation.resource_id)
            stats["violation_types"][violation.violation_type.value] += 1
        
        # Convert sets to counts for JSON serialization
        for policy_id, stats in policy_stats.items():
            stats["affected_resources"] = len(stats["affected_resources"])
        
        return {
            "policies_with_violations": len(policy_stats),
            "policy_details": dict(policy_stats),
            "most_violated_policies": sorted(
                policy_stats.items(),
                key=lambda x: x[1]["violation_count"],
                reverse=True
            )[:10]
        }
    
    def _generate_recommendations(self, score: ComplianceScore, 
                                violations: List[ComplianceViolation]) -> List[str]:
        """Generate actionable recommendations based on compliance analysis"""
        recommendations = []
        
        if score.overall_score < 70:
            recommendations.append("Compliance score is below acceptable threshold (70%). Immediate action required.")
        
        if score.critical_violations > 0:
            recommendations.append(f"Address {score.critical_violations} critical violations immediately.")
        
        if score.high_violations > 5:
            recommendations.append(f"High number of high-severity violations ({score.high_violations}). Consider policy review.")
        
        # Analyze violation patterns
        violation_types = defaultdict(int)
        for violation in violations:
            violation_types[violation.violation_type] += 1
        
        if violation_types[ViolationType.MISSING_MANDATORY_TAG] > 10:
            recommendations.append("High number of missing mandatory tags. Consider automated tagging solutions.")
        
        if violation_types[ViolationType.UNTAGGED_RESOURCE] > 5:
            recommendations.append("Multiple untagged resources detected. Implement tagging policies for all resource types.")
        
        # Trend-based recommendations
        if len(score.score_trend) >= 3:
            recent_trend = score.score_trend[-3:]
            if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                recommendations.append("Compliance score is declining. Review recent changes and enforcement procedures.")
        
        if not recommendations:
            recommendations.append("Compliance is good. Continue monitoring and maintain current practices.")
        
        return recommendations
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanning performance statistics"""
        return {
            **self.scan_statistics,
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "cached_resources": len(self.resource_cache),
            "active_violations": len([v for v in self.violations.values() if not v.is_resolved]),
            "resolved_violations": len([v for v in self.violations.values() if v.is_resolved])
        }


# Example usage and testing
if __name__ == "__main__":
    from tagging_policy_manager import TagPolicyManager, TaggingPolicy, TagRule, TagRequirement, ValidationRule, PolicyScope
    
    # Initialize components
    policy_manager = TagPolicyManager()
    monitor = ComplianceMonitor(policy_manager)
    
    # Create a test policy
    test_policy = TaggingPolicy(
        policy_id="test_policy",
        name="Test Policy",
        description="Test tagging policy",
        scope=PolicyScope.GLOBAL,
        scope_filter={},
        tag_rules=[
            TagRule(
                tag_key="Environment",
                requirement=TagRequirement.MANDATORY,
                validation_rules={
                    ValidationRule.ALLOWED_VALUES: ["dev", "test", "prod"]
                }
            ),
            TagRule(
                tag_key="Owner",
                requirement=TagRequirement.MANDATORY,
                validation_rules={
                    ValidationRule.REGEX_PATTERN: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }
            )
        ]
    )
    
    policy_manager.create_policy(test_policy)
    
    # Create test resources
    test_resources = [
        CloudResource(
            resource_id="resource-1",
            resource_type="ec2_instance",
            provider="aws",
            region="us-east-1",
            tags={"Environment": "prod", "Owner": "john@company.com"},
            attributes={"instance_type": "t3.micro"},
            created_at=datetime.now() - timedelta(days=1),
            last_modified=datetime.now()
        ),
        CloudResource(
            resource_id="resource-2",
            resource_type="ec2_instance",
            provider="aws",
            region="us-east-1",
            tags={"Environment": "invalid"},  # Invalid value
            attributes={"instance_type": "t3.small"},
            created_at=datetime.now() - timedelta(days=2),
            last_modified=datetime.now()
        ),
        CloudResource(
            resource_id="resource-3",
            resource_type="s3_bucket",
            provider="aws",
            region="us-east-1",
            tags={},  # Missing mandatory tags
            attributes={"bucket_name": "test-bucket"},
            created_at=datetime.now() - timedelta(days=3),
            last_modified=datetime.now()
        )
    ]
    
    # Run compliance scan
    async def test_scan():
        compliance_score = await monitor.scan_resources(test_resources)
        print(f"Compliance Score: {compliance_score.overall_score:.2f}")
        print(f"Violations: {compliance_score.violation_count}")
        
        # Get violations
        violations = monitor.get_violations()
        for violation in violations:
            print(f"Violation: {violation.description}")
        
        # Generate report
        report = monitor.generate_compliance_report()
        print(f"Report generated: {report.report_id}")
        print(f"Recommendations: {len(report.recommendations)}")
    
    # Run test
    import asyncio
    asyncio.run(test_scan())