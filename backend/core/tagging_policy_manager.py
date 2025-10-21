"""
Tagging Policy Management System

This module provides comprehensive tagging policy management and enforcement
capabilities for cloud resources, supporting flexible policy definition,
validation, and conflict resolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PolicyScope(Enum):
    """Defines the scope of a tagging policy"""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    DEPARTMENT = "department"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    RESOURCE_TYPE = "resource_type"


class TagRequirement(Enum):
    """Defines tag requirement levels"""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


class ValidationRule(Enum):
    """Types of tag validation rules"""
    REGEX_PATTERN = "regex_pattern"
    ALLOWED_VALUES = "allowed_values"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    FORMAT_VALIDATION = "format_validation"


@dataclass
class TagRule:
    """Defines a single tag rule within a policy"""
    tag_key: str
    requirement: TagRequirement
    validation_rules: Dict[ValidationRule, Any] = field(default_factory=dict)
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    def validate_tag_value(self, value: str) -> tuple[bool, str]:
        """Validate a tag value against this rule's validation rules"""
        if not value and self.requirement == TagRequirement.MANDATORY:
            return False, f"Tag '{self.tag_key}' is mandatory but missing"
        
        if not value:
            return True, ""  # Optional or recommended tags can be empty
        
        # Apply validation rules
        for rule_type, rule_value in self.validation_rules.items():
            if rule_type == ValidationRule.REGEX_PATTERN:
                if not re.match(rule_value, value):
                    return False, f"Tag '{self.tag_key}' value '{value}' doesn't match pattern '{rule_value}'"
            
            elif rule_type == ValidationRule.ALLOWED_VALUES:
                if value not in rule_value:
                    return False, f"Tag '{self.tag_key}' value '{value}' not in allowed values: {rule_value}"
            
            elif rule_type == ValidationRule.MIN_LENGTH:
                if len(value) < rule_value:
                    return False, f"Tag '{self.tag_key}' value too short (min: {rule_value})"
            
            elif rule_type == ValidationRule.MAX_LENGTH:
                if len(value) > rule_value:
                    return False, f"Tag '{self.tag_key}' value too long (max: {rule_value})"
        
        return True, ""


@dataclass
class TaggingPolicy:
    """Comprehensive tagging policy definition"""
    policy_id: str
    name: str
    description: str
    scope: PolicyScope
    scope_filter: Dict[str, Any]  # Filters for applying policy (e.g., resource_type, environment)
    tag_rules: List[TagRule]
    priority: int = 100  # Higher number = higher priority
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    def get_mandatory_tags(self) -> List[str]:
        """Get list of mandatory tag keys"""
        return [rule.tag_key for rule in self.tag_rules if rule.requirement == TagRequirement.MANDATORY]
    
    def get_forbidden_tags(self) -> List[str]:
        """Get list of forbidden tag keys"""
        return [rule.tag_key for rule in self.tag_rules if rule.requirement == TagRequirement.FORBIDDEN]
    
    def applies_to_resource(self, resource_attributes: Dict[str, Any]) -> bool:
        """Check if this policy applies to a given resource"""
        if not self.active:
            return False
        
        # Check scope filters
        for filter_key, filter_value in self.scope_filter.items():
            resource_value = resource_attributes.get(filter_key)
            
            if isinstance(filter_value, list):
                if resource_value not in filter_value:
                    return False
            elif isinstance(filter_value, str):
                if resource_value != filter_value:
                    return False
            elif isinstance(filter_value, dict) and "regex" in filter_value:
                if not re.match(filter_value["regex"], str(resource_value)):
                    return False
        
        return True


@dataclass
class PolicyTemplate:
    """Template for creating common tagging policies"""
    template_id: str
    name: str
    description: str
    category: str  # e.g., "financial", "security", "operational"
    tag_rules: List[TagRule]
    default_scope: PolicyScope
    customizable_fields: List[str] = field(default_factory=list)
    
    def create_policy(self, policy_id: str, name: str, scope_filter: Dict[str, Any]) -> TaggingPolicy:
        """Create a policy instance from this template"""
        return TaggingPolicy(
            policy_id=policy_id,
            name=name,
            description=self.description,
            scope=self.default_scope,
            scope_filter=scope_filter,
            tag_rules=self.tag_rules.copy(),
            created_by="template"
        )


@dataclass
class PolicyConflict:
    """Represents a conflict between tagging policies"""
    conflict_id: str
    policy_1: str
    policy_2: str
    conflict_type: str
    description: str
    severity: str  # "high", "medium", "low"
    resolution_suggestion: str
    detected_at: datetime = field(default_factory=datetime.now)


class TagPolicyManager:
    """
    Comprehensive tagging policy management system that provides flexible
    policy definition, validation, and conflict resolution capabilities.
    """
    
    def __init__(self):
        self.policies: Dict[str, TaggingPolicy] = {}
        self.templates: Dict[str, PolicyTemplate] = {}
        self.policy_conflicts: List[PolicyConflict] = []
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize common policy templates"""
        
        # Financial governance template
        financial_template = PolicyTemplate(
            template_id="financial_governance",
            name="Financial Governance",
            description="Standard financial tagging for cost attribution and chargeback",
            category="financial",
            default_scope=PolicyScope.GLOBAL,
            tag_rules=[
                TagRule(
                    tag_key="CostCenter",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.REGEX_PATTERN: r"^CC-\d{4}$"
                    },
                    description="Cost center for chargeback allocation",
                    examples=["CC-1001", "CC-2045"]
                ),
                TagRule(
                    tag_key="Project",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.MIN_LENGTH: 3,
                        ValidationRule.MAX_LENGTH: 50
                    },
                    description="Project name for cost tracking"
                ),
                TagRule(
                    tag_key="Environment",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["dev", "test", "staging", "prod"]
                    },
                    description="Environment classification"
                ),
                TagRule(
                    tag_key="Owner",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.REGEX_PATTERN: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                    },
                    description="Resource owner email address"
                )
            ]
        )
        
        # Security compliance template
        security_template = PolicyTemplate(
            template_id="security_compliance",
            name="Security Compliance",
            description="Security and compliance tagging requirements",
            category="security",
            default_scope=PolicyScope.GLOBAL,
            tag_rules=[
                TagRule(
                    tag_key="DataClassification",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["public", "internal", "confidential", "restricted"]
                    },
                    description="Data classification level"
                ),
                TagRule(
                    tag_key="ComplianceFramework",
                    requirement=TagRequirement.RECOMMENDED,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["SOX", "HIPAA", "PCI-DSS", "GDPR", "SOC2"]
                    },
                    description="Applicable compliance frameworks"
                ),
                TagRule(
                    tag_key="BackupRequired",
                    requirement=TagRequirement.MANDATORY,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["true", "false"]
                    },
                    description="Whether resource requires backup"
                )
            ]
        )
        
        # Operational management template
        operational_template = PolicyTemplate(
            template_id="operational_management",
            name="Operational Management",
            description="Operational tagging for lifecycle and maintenance",
            category="operational",
            default_scope=PolicyScope.GLOBAL,
            tag_rules=[
                TagRule(
                    tag_key="Schedule",
                    requirement=TagRequirement.RECOMMENDED,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["24x7", "business-hours", "weekend-only", "on-demand"]
                    },
                    description="Resource operational schedule"
                ),
                TagRule(
                    tag_key="AutoShutdown",
                    requirement=TagRequirement.OPTIONAL,
                    validation_rules={
                        ValidationRule.ALLOWED_VALUES: ["enabled", "disabled"]
                    },
                    description="Auto-shutdown configuration"
                ),
                TagRule(
                    tag_key="MaintenanceWindow",
                    requirement=TagRequirement.RECOMMENDED,
                    validation_rules={
                        ValidationRule.REGEX_PATTERN: r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)-\d{2}:\d{2}$"
                    },
                    description="Preferred maintenance window",
                    examples=["Sun-02:00", "Sat-23:30"]
                )
            ]
        )
        
        self.templates = {
            "financial_governance": financial_template,
            "security_compliance": security_template,
            "operational_management": operational_template
        }
    
    def create_policy(self, policy: TaggingPolicy) -> bool:
        """Create a new tagging policy"""
        try:
            # Validate policy
            if not self._validate_policy(policy):
                return False
            
            # Check for conflicts
            conflicts = self._detect_policy_conflicts(policy)
            if conflicts:
                logger.warning(f"Policy conflicts detected for {policy.policy_id}: {len(conflicts)} conflicts")
                self.policy_conflicts.extend(conflicts)
            
            # Store policy
            self.policies[policy.policy_id] = policy
            logger.info(f"Created tagging policy: {policy.policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create policy {policy.policy_id}: {str(e)}")
            return False
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing tagging policy"""
        if policy_id not in self.policies:
            logger.error(f"Policy {policy_id} not found")
            return False
        
        try:
            policy = self.policies[policy_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            policy.updated_at = datetime.now()
            
            # Re-validate after updates
            if not self._validate_policy(policy):
                return False
            
            logger.info(f"Updated tagging policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy {policy_id}: {str(e)}")
            return False
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a tagging policy"""
        if policy_id not in self.policies:
            logger.error(f"Policy {policy_id} not found")
            return False
        
        try:
            del self.policies[policy_id]
            
            # Remove related conflicts
            self.policy_conflicts = [
                conflict for conflict in self.policy_conflicts
                if conflict.policy_1 != policy_id and conflict.policy_2 != policy_id
            ]
            
            logger.info(f"Deleted tagging policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {str(e)}")
            return False
    
    def get_policy(self, policy_id: str) -> Optional[TaggingPolicy]:
        """Get a specific tagging policy"""
        return self.policies.get(policy_id)
    
    def list_policies(self, scope: Optional[PolicyScope] = None, active_only: bool = True) -> List[TaggingPolicy]:
        """List tagging policies with optional filtering"""
        policies = list(self.policies.values())
        
        if active_only:
            policies = [p for p in policies if p.active]
        
        if scope:
            policies = [p for p in policies if p.scope == scope]
        
        # Sort by priority (higher first)
        return sorted(policies, key=lambda p: p.priority, reverse=True)
    
    def create_policy_from_template(self, template_id: str, policy_id: str, 
                                  name: str, scope_filter: Dict[str, Any],
                                  customizations: Dict[str, Any] = None) -> Optional[TaggingPolicy]:
        """Create a policy from a template"""
        if template_id not in self.templates:
            logger.error(f"Template {template_id} not found")
            return None
        
        try:
            template = self.templates[template_id]
            policy = template.create_policy(policy_id, name, scope_filter)
            
            # Apply customizations
            if customizations:
                for key, value in customizations.items():
                    if key in template.customizable_fields and hasattr(policy, key):
                        setattr(policy, key, value)
            
            if self.create_policy(policy):
                return policy
            return None
            
        except Exception as e:
            logger.error(f"Failed to create policy from template {template_id}: {str(e)}")
            return None
    
    def get_applicable_policies(self, resource_attributes: Dict[str, Any]) -> List[TaggingPolicy]:
        """Get all policies that apply to a given resource"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if policy.applies_to_resource(resource_attributes):
                applicable_policies.append(policy)
        
        # Sort by priority (higher first)
        return sorted(applicable_policies, key=lambda p: p.priority, reverse=True)
    
    def validate_resource_tags(self, resource_attributes: Dict[str, Any], 
                             resource_tags: Dict[str, str]) -> Dict[str, Any]:
        """Validate resource tags against applicable policies"""
        applicable_policies = self.get_applicable_policies(resource_attributes)
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "missing_mandatory": [],
            "forbidden_present": [],
            "policy_count": len(applicable_policies)
        }
        
        # Collect all rules from applicable policies
        all_rules = {}
        for policy in applicable_policies:
            for rule in policy.tag_rules:
                if rule.tag_key not in all_rules or policy.priority > all_rules[rule.tag_key][1]:
                    all_rules[rule.tag_key] = (rule, policy.priority, policy.policy_id)
        
        # Validate against rules
        for tag_key, (rule, priority, policy_id) in all_rules.items():
            tag_value = resource_tags.get(tag_key, "")
            
            if rule.requirement == TagRequirement.MANDATORY and not tag_value:
                validation_result["missing_mandatory"].append({
                    "tag": tag_key,
                    "policy": policy_id,
                    "description": rule.description
                })
                validation_result["compliant"] = False
            
            elif rule.requirement == TagRequirement.FORBIDDEN and tag_value:
                validation_result["forbidden_present"].append({
                    "tag": tag_key,
                    "value": tag_value,
                    "policy": policy_id,
                    "description": rule.description
                })
                validation_result["compliant"] = False
            
            elif tag_value:  # Validate non-empty values
                is_valid, error_message = rule.validate_tag_value(tag_value)
                if not is_valid:
                    validation_result["violations"].append({
                        "tag": tag_key,
                        "value": tag_value,
                        "error": error_message,
                        "policy": policy_id
                    })
                    validation_result["compliant"] = False
            
            elif rule.requirement == TagRequirement.RECOMMENDED and not tag_value:
                validation_result["warnings"].append({
                    "tag": tag_key,
                    "message": f"Recommended tag '{tag_key}' is missing",
                    "policy": policy_id,
                    "description": rule.description
                })
        
        return validation_result
    
    def _validate_policy(self, policy: TaggingPolicy) -> bool:
        """Validate a policy definition"""
        if not policy.policy_id or not policy.name:
            logger.error("Policy must have ID and name")
            return False
        
        if not policy.tag_rules:
            logger.error("Policy must have at least one tag rule")
            return False
        
        # Check for duplicate tag keys within policy
        tag_keys = [rule.tag_key for rule in policy.tag_rules]
        if len(tag_keys) != len(set(tag_keys)):
            logger.error("Policy contains duplicate tag keys")
            return False
        
        return True
    
    def _detect_policy_conflicts(self, new_policy: TaggingPolicy) -> List[PolicyConflict]:
        """Detect conflicts between policies"""
        conflicts = []
        
        for existing_policy in self.policies.values():
            if existing_policy.policy_id == new_policy.policy_id:
                continue
            
            # Check for overlapping scope and conflicting rules
            if self._policies_overlap(existing_policy, new_policy):
                policy_conflicts = self._find_rule_conflicts(existing_policy, new_policy)
                conflicts.extend(policy_conflicts)
        
        return conflicts
    
    def _policies_overlap(self, policy1: TaggingPolicy, policy2: TaggingPolicy) -> bool:
        """Check if two policies have overlapping scope"""
        # Simplified overlap detection - in practice, this would be more sophisticated
        return (policy1.scope == policy2.scope or 
                policy1.scope == PolicyScope.GLOBAL or 
                policy2.scope == PolicyScope.GLOBAL)
    
    def _find_rule_conflicts(self, policy1: TaggingPolicy, policy2: TaggingPolicy) -> List[PolicyConflict]:
        """Find conflicts between rules in two policies"""
        conflicts = []
        
        # Create maps of tag rules
        rules1 = {rule.tag_key: rule for rule in policy1.tag_rules}
        rules2 = {rule.tag_key: rule for rule in policy2.tag_rules}
        
        # Check for conflicts on same tag keys
        common_tags = set(rules1.keys()) & set(rules2.keys())
        
        for tag_key in common_tags:
            rule1 = rules1[tag_key]
            rule2 = rules2[tag_key]
            
            # Check for requirement conflicts
            if (rule1.requirement == TagRequirement.MANDATORY and 
                rule2.requirement == TagRequirement.FORBIDDEN):
                conflicts.append(PolicyConflict(
                    conflict_id=f"{policy1.policy_id}_{policy2.policy_id}_{tag_key}_requirement",
                    policy_1=policy1.policy_id,
                    policy_2=policy2.policy_id,
                    conflict_type="requirement_conflict",
                    description=f"Tag '{tag_key}' is mandatory in {policy1.policy_id} but forbidden in {policy2.policy_id}",
                    severity="high",
                    resolution_suggestion="Review policy scopes or modify requirements"
                ))
            
            # Check for validation rule conflicts
            if (ValidationRule.ALLOWED_VALUES in rule1.validation_rules and 
                ValidationRule.ALLOWED_VALUES in rule2.validation_rules):
                values1 = set(rule1.validation_rules[ValidationRule.ALLOWED_VALUES])
                values2 = set(rule2.validation_rules[ValidationRule.ALLOWED_VALUES])
                
                if not values1 & values2:  # No common values
                    conflicts.append(PolicyConflict(
                        conflict_id=f"{policy1.policy_id}_{policy2.policy_id}_{tag_key}_values",
                        policy_1=policy1.policy_id,
                        policy_2=policy2.policy_id,
                        conflict_type="validation_conflict",
                        description=f"Tag '{tag_key}' has incompatible allowed values between policies",
                        severity="medium",
                        resolution_suggestion="Align allowed values or adjust policy scopes"
                    ))
        
        return conflicts
    
    def get_policy_conflicts(self) -> List[PolicyConflict]:
        """Get all detected policy conflicts"""
        return self.policy_conflicts
    
    def resolve_conflict(self, conflict_id: str, resolution_action: str) -> bool:
        """Mark a conflict as resolved"""
        for i, conflict in enumerate(self.policy_conflicts):
            if conflict.conflict_id == conflict_id:
                logger.info(f"Resolved conflict {conflict_id} with action: {resolution_action}")
                del self.policy_conflicts[i]
                return True
        return False
    
    def export_policies(self, policy_ids: List[str] = None) -> Dict[str, Any]:
        """Export policies to JSON format"""
        policies_to_export = self.policies
        if policy_ids:
            policies_to_export = {pid: self.policies[pid] for pid in policy_ids if pid in self.policies}
        
        return {
            "policies": {
                pid: {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "description": policy.description,
                    "scope": policy.scope.value,
                    "scope_filter": policy.scope_filter,
                    "tag_rules": [
                        {
                            "tag_key": rule.tag_key,
                            "requirement": rule.requirement.value,
                            "validation_rules": {vr.value: val for vr, val in rule.validation_rules.items()},
                            "description": rule.description,
                            "examples": rule.examples
                        }
                        for rule in policy.tag_rules
                    ],
                    "priority": policy.priority,
                    "active": policy.active,
                    "created_at": policy.created_at.isoformat(),
                    "updated_at": policy.updated_at.isoformat(),
                    "created_by": policy.created_by
                }
                for pid, policy in policies_to_export.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_policies(self, policy_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, bool]:
        """Import policies from JSON format"""
        results = {}
        
        for policy_id, policy_dict in policy_data.get("policies", {}).items():
            try:
                if policy_id in self.policies and not overwrite:
                    results[policy_id] = False
                    continue
                
                # Reconstruct policy object
                tag_rules = []
                for rule_dict in policy_dict["tag_rules"]:
                    validation_rules = {
                        ValidationRule(vr): val 
                        for vr, val in rule_dict["validation_rules"].items()
                    }
                    
                    tag_rules.append(TagRule(
                        tag_key=rule_dict["tag_key"],
                        requirement=TagRequirement(rule_dict["requirement"]),
                        validation_rules=validation_rules,
                        description=rule_dict["description"],
                        examples=rule_dict["examples"]
                    ))
                
                policy = TaggingPolicy(
                    policy_id=policy_dict["policy_id"],
                    name=policy_dict["name"],
                    description=policy_dict["description"],
                    scope=PolicyScope(policy_dict["scope"]),
                    scope_filter=policy_dict["scope_filter"],
                    tag_rules=tag_rules,
                    priority=policy_dict["priority"],
                    active=policy_dict["active"],
                    created_at=datetime.fromisoformat(policy_dict["created_at"]),
                    updated_at=datetime.fromisoformat(policy_dict["updated_at"]),
                    created_by=policy_dict["created_by"]
                )
                
                results[policy_id] = self.create_policy(policy)
                
            except Exception as e:
                logger.error(f"Failed to import policy {policy_id}: {str(e)}")
                results[policy_id] = False
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize policy manager
    manager = TagPolicyManager()
    
    # Create a custom policy
    custom_policy = TaggingPolicy(
        policy_id="dev_team_policy",
        name="Development Team Policy",
        description="Tagging requirements for development resources",
        scope=PolicyScope.DEPARTMENT,
        scope_filter={"department": "engineering", "environment": ["dev", "test"]},
        tag_rules=[
            TagRule(
                tag_key="Team",
                requirement=TagRequirement.MANDATORY,
                validation_rules={
                    ValidationRule.ALLOWED_VALUES: ["backend", "frontend", "devops", "qa"]
                },
                description="Development team responsible for the resource"
            ),
            TagRule(
                tag_key="Sprint",
                requirement=TagRequirement.RECOMMENDED,
                validation_rules={
                    ValidationRule.REGEX_PATTERN: r"^S\d{4}-\d{2}$"
                },
                description="Sprint identifier",
                examples=["S2024-01", "S2024-15"]
            )
        ],
        priority=150
    )
    
    # Create policy
    success = manager.create_policy(custom_policy)
    print(f"Policy creation: {'Success' if success else 'Failed'}")
    
    # Create policy from template
    financial_policy = manager.create_policy_from_template(
        template_id="financial_governance",
        policy_id="prod_financial_policy",
        name="Production Financial Governance",
        scope_filter={"environment": "prod"}
    )
    
    if financial_policy:
        print(f"Created policy from template: {financial_policy.policy_id}")
    
    # Test resource validation
    resource_attrs = {
        "department": "engineering",
        "environment": "dev",
        "resource_type": "ec2_instance"
    }
    
    resource_tags = {
        "Team": "backend",
        "CostCenter": "CC-1001",
        "Project": "user-service",
        "Environment": "dev",
        "Owner": "john.doe@company.com"
    }
    
    validation_result = manager.validate_resource_tags(resource_attrs, resource_tags)
    print(f"Validation result: {validation_result}")
    
    # List policies
    policies = manager.list_policies()
    print(f"Total active policies: {len(policies)}")
    
    # Check for conflicts
    conflicts = manager.get_policy_conflicts()
    print(f"Policy conflicts detected: {len(conflicts)}")