"""
Governance Validation System for Cloud Migration Advisor

This module provides categorization compliance checking, governance rule validation,
and violation detection and reporting.

Requirements: 6.6
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, ResourceType
from .auto_categorization_engine import (
    CategorizedResources,
    ResourceCategorization,
    OwnershipStatus
)
from .organizational_structure_manager import (
    OrganizationalStructure,
    DimensionType
)


logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Severity levels for governance violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleType(Enum):
    """Types of governance rules"""
    CATEGORIZATION_REQUIRED = "categorization_required"
    TAG_REQUIRED = "tag_required"
    NAMING_CONVENTION = "naming_convention"
    OWNERSHIP_REQUIRED = "ownership_required"
    DIMENSION_REQUIRED = "dimension_required"
    CUSTOM = "custom"


@dataclass
class GovernanceRule:
    """
    Definition of a governance rule
    
    Requirements: 6.6
    """
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    severity: ViolationSeverity
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure enums are properly set"""
        if isinstance(self.rule_type, str):
            self.rule_type = RuleType(self.rule_type)
        if isinstance(self.severity, str):
            self.severity = ViolationSeverity(self.severity)


@dataclass
class GovernanceViolation:
    """
    Represents a governance violation
    
    Requirements: 6.6
    """
    violation_id: str
    rule_id: str
    rule_name: str
    resource_id: str
    resource_name: Optional[str]
    severity: ViolationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ValidationResult:
    """Result of governance validation"""
    total_resources: int
    compliant_resources: int
    non_compliant_resources: int
    violations: List[GovernanceViolation] = field(default_factory=list)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    
    def get_compliance_rate(self) -> float:
        """Calculate compliance rate as percentage"""
        if self.total_resources == 0:
            return 100.0
        return (self.compliant_resources / self.total_resources) * 100


@dataclass
class GovernancePolicy:
    """Collection of governance rules"""
    policy_id: str
    name: str
    description: str
    rules: List[GovernanceRule] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_rule(self, rule: GovernanceRule) -> None:
        """Add a rule to the policy"""
        self.rules.append(rule)
        self.updated_at = datetime.utcnow()
    
    def get_enabled_rules(self) -> List[GovernanceRule]:
        """Get all enabled rules"""
        return [rule for rule in self.rules if rule.enabled]


class GovernanceValidationSystem:
    """
    System for validating governance compliance
    
    Requirements: 6.6
    """
    
    def __init__(self):
        """Initialize the governance validation system"""
        self._policies: Dict[str, GovernancePolicy] = {}
        logger.info("Governance Validation System initialized")
    
    def create_policy(
        self,
        policy_id: str,
        name: str,
        description: str
    ) -> GovernancePolicy:
        """
        Create a new governance policy
        
        Args:
            policy_id: Unique policy identifier
            name: Policy name
            description: Policy description
            
        Returns:
            New GovernancePolicy
        """
        logger.info(f"Creating governance policy: {name}")
        
        policy = GovernancePolicy(
            policy_id=policy_id,
            name=name,
            description=description
        )
        
        self._policies[policy_id] = policy
        
        return policy
    
    def add_rule_to_policy(
        self,
        policy_id: str,
        rule: GovernanceRule
    ) -> GovernancePolicy:
        """Add a rule to a policy"""
        policy = self._policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy not found: {policy_id}")
        
        policy.add_rule(rule)
        logger.debug(f"Added rule {rule.rule_id} to policy {policy_id}")
        
        return policy

    
    def validate_resources(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        policy: GovernancePolicy
    ) -> ValidationResult:
        """
        Validate resources against a governance policy
        
        Args:
            resources: List of cloud resources to validate
            categorizations: Resource categorizations
            structure: Organizational structure
            policy: Governance policy to validate against
            
        Returns:
            ValidationResult with violations
        """
        logger.info(f"Validating {len(resources)} resources against policy: {policy.name}")
        start_time = datetime.utcnow()
        
        violations = []
        compliant_count = 0
        
        # Get enabled rules
        enabled_rules = policy.get_enabled_rules()
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            resource_violations = []
            
            # Validate against each rule
            for rule in enabled_rules:
                rule_violations = self._validate_rule(
                    resource,
                    categorization,
                    structure,
                    rule
                )
                resource_violations.extend(rule_violations)
            
            if resource_violations:
                violations.extend(resource_violations)
            else:
                compliant_count += 1
        
        # Calculate statistics
        violations_by_severity = self._count_by_severity(violations)
        violations_by_rule = self._count_by_rule(violations)
        
        end_time = datetime.utcnow()
        validation_time = (end_time - start_time).total_seconds() * 1000
        
        result = ValidationResult(
            total_resources=len(resources),
            compliant_resources=compliant_count,
            non_compliant_resources=len(resources) - compliant_count,
            violations=violations,
            violations_by_severity=violations_by_severity,
            violations_by_rule=violations_by_rule,
            validation_time_ms=validation_time
        )
        
        logger.info(
            f"Validation complete: {result.compliant_resources}/{result.total_resources} "
            f"compliant ({result.get_compliance_rate():.1f}%), "
            f"{len(violations)} violations found"
        )
        
        return result
    
    def validate_categorization_completeness(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        required_dimensions: List[DimensionType]
    ) -> ValidationResult:
        """
        Validate that all resources have required categorizations
        
        Args:
            resources: Resources to validate
            categorizations: Resource categorizations
            required_dimensions: Dimensions that must be categorized
            
        Returns:
            ValidationResult
        """
        logger.info(f"Validating categorization completeness for {len(resources)} resources")
        
        violations = []
        compliant_count = 0
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            if not categorization:
                violation = self._create_violation(
                    rule_id="cat_required",
                    rule_name="Categorization Required",
                    resource=resource,
                    severity=ViolationSeverity.ERROR,
                    message="Resource is not categorized"
                )
                violations.append(violation)
                continue
            
            # Check each required dimension
            missing_dimensions = []
            for dimension in required_dimensions:
                if not self._has_dimension_value(categorization, dimension):
                    missing_dimensions.append(dimension.value)
            
            if missing_dimensions:
                violation = self._create_violation(
                    rule_id="dim_required",
                    rule_name="Required Dimensions",
                    resource=resource,
                    severity=ViolationSeverity.WARNING,
                    message=f"Missing required dimensions: {', '.join(missing_dimensions)}",
                    details={"missing_dimensions": missing_dimensions}
                )
                violations.append(violation)
            else:
                compliant_count += 1
        
        return ValidationResult(
            total_resources=len(resources),
            compliant_resources=compliant_count,
            non_compliant_resources=len(resources) - compliant_count,
            violations=violations,
            violations_by_severity=self._count_by_severity(violations),
            violations_by_rule=self._count_by_rule(violations)
        )

    
    def validate_ownership(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> ValidationResult:
        """
        Validate that all resources have ownership assigned
        
        Args:
            resources: Resources to validate
            categorizations: Resource categorizations
            
        Returns:
            ValidationResult
        """
        logger.info(f"Validating ownership for {len(resources)} resources")
        
        violations = []
        compliant_count = 0
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            if not categorization or categorization.ownership_status == OwnershipStatus.UNASSIGNED:
                violation = self._create_violation(
                    rule_id="ownership_required",
                    rule_name="Ownership Required",
                    resource=resource,
                    severity=ViolationSeverity.WARNING,
                    message="Resource has no assigned owner"
                )
                violations.append(violation)
            else:
                compliant_count += 1
        
        return ValidationResult(
            total_resources=len(resources),
            compliant_resources=compliant_count,
            non_compliant_resources=len(resources) - compliant_count,
            violations=violations,
            violations_by_severity=self._count_by_severity(violations),
            violations_by_rule=self._count_by_rule(violations)
        )
    
    def validate_tagging(
        self,
        resources: List[CloudResource],
        required_tags: List[str]
    ) -> ValidationResult:
        """
        Validate that resources have required tags
        
        Args:
            resources: Resources to validate
            required_tags: List of required tag keys
            
        Returns:
            ValidationResult
        """
        logger.info(f"Validating tags for {len(resources)} resources")
        
        violations = []
        compliant_count = 0
        
        for resource in resources:
            missing_tags = []
            
            for tag_key in required_tags:
                if tag_key not in resource.tags or not resource.tags[tag_key]:
                    missing_tags.append(tag_key)
            
            if missing_tags:
                violation = self._create_violation(
                    rule_id="tags_required",
                    rule_name="Required Tags",
                    resource=resource,
                    severity=ViolationSeverity.WARNING,
                    message=f"Missing required tags: {', '.join(missing_tags)}",
                    details={"missing_tags": missing_tags}
                )
                violations.append(violation)
            else:
                compliant_count += 1
        
        return ValidationResult(
            total_resources=len(resources),
            compliant_resources=compliant_count,
            non_compliant_resources=len(resources) - compliant_count,
            violations=violations,
            violations_by_severity=self._count_by_severity(violations),
            violations_by_rule=self._count_by_rule(violations)
        )
    
    def validate_naming_convention(
        self,
        resources: List[CloudResource],
        pattern: str
    ) -> ValidationResult:
        """
        Validate resource names against a naming convention pattern
        
        Args:
            resources: Resources to validate
            pattern: Regex pattern for naming convention
            
        Returns:
            ValidationResult
        """
        import re
        
        logger.info(f"Validating naming convention for {len(resources)} resources")
        
        violations = []
        compliant_count = 0
        compiled_pattern = re.compile(pattern)
        
        for resource in resources:
            if not compiled_pattern.match(resource.resource_name):
                violation = self._create_violation(
                    rule_id="naming_convention",
                    rule_name="Naming Convention",
                    resource=resource,
                    severity=ViolationSeverity.INFO,
                    message=f"Resource name does not match pattern: {pattern}",
                    details={"pattern": pattern, "actual_name": resource.resource_name}
                )
                violations.append(violation)
            else:
                compliant_count += 1
        
        return ValidationResult(
            total_resources=len(resources),
            compliant_resources=compliant_count,
            non_compliant_resources=len(resources) - compliant_count,
            violations=violations,
            violations_by_severity=self._count_by_severity(violations),
            violations_by_rule=self._count_by_rule(violations)
        )
    
    def get_violations_by_severity(
        self,
        result: ValidationResult,
        severity: ViolationSeverity
    ) -> List[GovernanceViolation]:
        """Get violations filtered by severity"""
        return [v for v in result.violations if v.severity == severity]
    
    def get_violations_by_resource(
        self,
        result: ValidationResult,
        resource_id: str
    ) -> List[GovernanceViolation]:
        """Get violations for a specific resource"""
        return [v for v in result.violations if v.resource_id == resource_id]
    
    def generate_compliance_report(
        self,
        result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Generate a compliance report from validation results
        
        Args:
            result: Validation result
            
        Returns:
            Dictionary with compliance report data
        """
        return {
            "summary": {
                "total_resources": result.total_resources,
                "compliant_resources": result.compliant_resources,
                "non_compliant_resources": result.non_compliant_resources,
                "compliance_rate": result.get_compliance_rate(),
                "total_violations": len(result.violations)
            },
            "violations_by_severity": result.violations_by_severity,
            "violations_by_rule": result.violations_by_rule,
            "critical_violations": len(self.get_violations_by_severity(
                result, ViolationSeverity.CRITICAL
            )),
            "error_violations": len(self.get_violations_by_severity(
                result, ViolationSeverity.ERROR
            )),
            "warning_violations": len(self.get_violations_by_severity(
                result, ViolationSeverity.WARNING
            )),
            "info_violations": len(self.get_violations_by_severity(
                result, ViolationSeverity.INFO
            ))
        }

    
    # Private helper methods
    
    def _validate_rule(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization],
        structure: OrganizationalStructure,
        rule: GovernanceRule
    ) -> List[GovernanceViolation]:
        """Validate a single rule against a resource"""
        violations = []
        
        if rule.rule_type == RuleType.CATEGORIZATION_REQUIRED:
            if not categorization:
                violations.append(self._create_violation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    resource=resource,
                    severity=rule.severity,
                    message="Resource is not categorized"
                ))
        
        elif rule.rule_type == RuleType.TAG_REQUIRED:
            required_tags = rule.parameters.get('required_tags', [])
            missing_tags = [
                tag for tag in required_tags
                if tag not in resource.tags or not resource.tags[tag]
            ]
            if missing_tags:
                violations.append(self._create_violation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    resource=resource,
                    severity=rule.severity,
                    message=f"Missing required tags: {', '.join(missing_tags)}",
                    details={"missing_tags": missing_tags}
                ))
        
        elif rule.rule_type == RuleType.NAMING_CONVENTION:
            import re
            pattern = rule.parameters.get('pattern', '.*')
            if not re.match(pattern, resource.resource_name):
                violations.append(self._create_violation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    resource=resource,
                    severity=rule.severity,
                    message=f"Name does not match pattern: {pattern}"
                ))
        
        elif rule.rule_type == RuleType.OWNERSHIP_REQUIRED:
            if not categorization or categorization.ownership_status == OwnershipStatus.UNASSIGNED:
                violations.append(self._create_violation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    resource=resource,
                    severity=rule.severity,
                    message="Resource has no assigned owner"
                ))
        
        elif rule.rule_type == RuleType.DIMENSION_REQUIRED:
            if categorization:
                required_dimensions = rule.parameters.get('required_dimensions', [])
                missing_dimensions = []
                for dim_str in required_dimensions:
                    dim = DimensionType(dim_str)
                    if not self._has_dimension_value(categorization, dim):
                        missing_dimensions.append(dim.value)
                
                if missing_dimensions:
                    violations.append(self._create_violation(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        resource=resource,
                        severity=rule.severity,
                        message=f"Missing required dimensions: {', '.join(missing_dimensions)}",
                        details={"missing_dimensions": missing_dimensions}
                    ))
        
        return violations
    
    def _has_dimension_value(
        self,
        categorization: ResourceCategorization,
        dimension: DimensionType
    ) -> bool:
        """Check if categorization has a value for the dimension"""
        if dimension == DimensionType.TEAM:
            return bool(categorization.team)
        elif dimension == DimensionType.PROJECT:
            return bool(categorization.project)
        elif dimension == DimensionType.ENVIRONMENT:
            return bool(categorization.environment)
        elif dimension == DimensionType.REGION:
            return bool(categorization.region)
        elif dimension == DimensionType.COST_CENTER:
            return bool(categorization.cost_center)
        
        return False
    
    def _create_violation(
        self,
        rule_id: str,
        rule_name: str,
        resource: CloudResource,
        severity: ViolationSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> GovernanceViolation:
        """Create a governance violation"""
        import hashlib
        
        violation_id = hashlib.md5(
            f"{rule_id}_{resource.resource_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        return GovernanceViolation(
            violation_id=violation_id,
            rule_id=rule_id,
            rule_name=rule_name,
            resource_id=resource.resource_id,
            resource_name=resource.resource_name,
            severity=severity,
            message=message,
            details=details or {}
        )
    
    def _count_by_severity(
        self,
        violations: List[GovernanceViolation]
    ) -> Dict[str, int]:
        """Count violations by severity"""
        counts = {
            ViolationSeverity.CRITICAL.value: 0,
            ViolationSeverity.ERROR.value: 0,
            ViolationSeverity.WARNING.value: 0,
            ViolationSeverity.INFO.value: 0
        }
        
        for violation in violations:
            counts[violation.severity.value] += 1
        
        return counts
    
    def _count_by_rule(
        self,
        violations: List[GovernanceViolation]
    ) -> Dict[str, int]:
        """Count violations by rule"""
        from collections import defaultdict
        counts = defaultdict(int)
        
        for violation in violations:
            counts[violation.rule_id] += 1
        
        return dict(counts)


# Predefined governance rules

def create_standard_governance_policy() -> GovernancePolicy:
    """Create a standard governance policy with common rules"""
    policy = GovernancePolicy(
        policy_id="standard_policy",
        name="Standard Governance Policy",
        description="Standard governance rules for cloud resources"
    )
    
    # Categorization required
    policy.add_rule(GovernanceRule(
        rule_id="cat_001",
        rule_type=RuleType.CATEGORIZATION_REQUIRED,
        name="Categorization Required",
        description="All resources must be categorized",
        severity=ViolationSeverity.ERROR
    ))
    
    # Team assignment required
    policy.add_rule(GovernanceRule(
        rule_id="dim_001",
        rule_type=RuleType.DIMENSION_REQUIRED,
        name="Team Assignment Required",
        description="All resources must be assigned to a team",
        severity=ViolationSeverity.WARNING,
        parameters={"required_dimensions": [DimensionType.TEAM.value]}
    ))
    
    # Environment assignment required
    policy.add_rule(GovernanceRule(
        rule_id="dim_002",
        rule_type=RuleType.DIMENSION_REQUIRED,
        name="Environment Assignment Required",
        description="All resources must be assigned to an environment",
        severity=ViolationSeverity.WARNING,
        parameters={"required_dimensions": [DimensionType.ENVIRONMENT.value]}
    ))
    
    # Ownership required
    policy.add_rule(GovernanceRule(
        rule_id="own_001",
        rule_type=RuleType.OWNERSHIP_REQUIRED,
        name="Ownership Required",
        description="All resources must have an assigned owner",
        severity=ViolationSeverity.WARNING
    ))
    
    # Required tags
    policy.add_rule(GovernanceRule(
        rule_id="tag_001",
        rule_type=RuleType.TAG_REQUIRED,
        name="Required Tags",
        description="Resources must have required tags",
        severity=ViolationSeverity.INFO,
        parameters={"required_tags": ["Name", "Environment", "Owner"]}
    ))
    
    return policy
