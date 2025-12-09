"""
Advanced Filtering System for Cloud Migration Advisor

This module provides advanced filtering capabilities with complex queries,
multi-condition filtering, and logical operators.

Requirements: 6.2
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .resource_discovery_engine import CloudResource, ResourceType
from .auto_categorization_engine import CategorizedResources, ResourceCategorization
from .organizational_structure_manager import DimensionType


logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """Logical operators for combining filter conditions"""
    AND = "and"
    OR = "or"
    NOT = "not"


class ComparisonOperator(Enum):
    """Comparison operators for filter conditions"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class FilterField(Enum):
    """Fields that can be filtered on"""
    RESOURCE_ID = "resource_id"
    RESOURCE_NAME = "resource_name"
    RESOURCE_TYPE = "resource_type"
    PROVIDER = "provider"
    REGION = "region"
    TEAM = "team"
    PROJECT = "project"
    ENVIRONMENT = "environment"
    COST_CENTER = "cost_center"
    TAGS = "tags"
    CREATED_DATE = "created_date"
    OWNERSHIP_STATUS = "ownership_status"
    CUSTOM_ATTRIBUTE = "custom_attribute"


@dataclass
class FilterCondition:
    """
    Single filter condition
    
    Requirements: 6.2
    """
    field: FilterField
    operator: ComparisonOperator
    value: Any
    field_path: Optional[str] = None  # For nested fields like tags.environment
    
    def __post_init__(self):
        """Ensure enums are properly set"""
        if isinstance(self.field, str):
            self.field = FilterField(self.field)
        if isinstance(self.operator, str):
            self.operator = ComparisonOperator(self.operator)
    
    def evaluate(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization]
    ) -> bool:
        """
        Evaluate this condition against a resource
        
        Args:
            resource: Cloud resource to evaluate
            categorization: Optional resource categorization
            
        Returns:
            True if condition matches, False otherwise
        """
        # Get the field value
        field_value = self._get_field_value(resource, categorization)
        
        # Apply operator
        return self._apply_operator(field_value, self.operator, self.value)
    
    def _get_field_value(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization]
    ) -> Any:
        """Extract field value from resource or categorization"""
        if self.field == FilterField.RESOURCE_ID:
            return resource.resource_id
        elif self.field == FilterField.RESOURCE_NAME:
            return resource.resource_name
        elif self.field == FilterField.RESOURCE_TYPE:
            return resource.resource_type.value if hasattr(resource.resource_type, 'value') else resource.resource_type
        elif self.field == FilterField.PROVIDER:
            return resource.provider.value if hasattr(resource.provider, 'value') else resource.provider
        elif self.field == FilterField.REGION:
            return resource.region
        elif self.field == FilterField.TAGS:
            if self.field_path:
                # Access specific tag
                return resource.tags.get(self.field_path)
            return resource.tags
        elif self.field == FilterField.CREATED_DATE:
            return resource.created_date
        
        # Fields from categorization
        if categorization:
            if self.field == FilterField.TEAM:
                return categorization.team
            elif self.field == FilterField.PROJECT:
                return categorization.project
            elif self.field == FilterField.ENVIRONMENT:
                return categorization.environment
            elif self.field == FilterField.COST_CENTER:
                return categorization.cost_center
            elif self.field == FilterField.OWNERSHIP_STATUS:
                return categorization.ownership_status
            elif self.field == FilterField.CUSTOM_ATTRIBUTE:
                if self.field_path:
                    return categorization.custom_attributes.get(self.field_path)
                return categorization.custom_attributes
        
        return None
    
    def _apply_operator(self, field_value: Any, operator: ComparisonOperator, target_value: Any) -> bool:
        """Apply comparison operator"""
        try:
            if operator == ComparisonOperator.EQUALS:
                return field_value == target_value
            elif operator == ComparisonOperator.NOT_EQUALS:
                return field_value != target_value
            elif operator == ComparisonOperator.GREATER_THAN:
                return field_value > target_value
            elif operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
                return field_value >= target_value
            elif operator == ComparisonOperator.LESS_THAN:
                return field_value < target_value
            elif operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
                return field_value <= target_value
            elif operator == ComparisonOperator.CONTAINS:
                if isinstance(field_value, str):
                    return target_value in field_value
                elif isinstance(field_value, (list, dict)):
                    return target_value in field_value
                return False
            elif operator == ComparisonOperator.NOT_CONTAINS:
                if isinstance(field_value, str):
                    return target_value not in field_value
                elif isinstance(field_value, (list, dict)):
                    return target_value not in field_value
                return True
            elif operator == ComparisonOperator.STARTS_WITH:
                return isinstance(field_value, str) and field_value.startswith(target_value)
            elif operator == ComparisonOperator.ENDS_WITH:
                return isinstance(field_value, str) and field_value.endswith(target_value)
            elif operator == ComparisonOperator.IN:
                return field_value in target_value
            elif operator == ComparisonOperator.NOT_IN:
                return field_value not in target_value
            elif operator == ComparisonOperator.REGEX:
                if isinstance(field_value, str):
                    return bool(re.match(target_value, field_value))
                return False
            elif operator == ComparisonOperator.EXISTS:
                return field_value is not None
            elif operator == ComparisonOperator.NOT_EXISTS:
                return field_value is None
        except Exception as e:
            logger.error(f"Error applying operator {operator.value}: {e}")
            return False
        
        return False


@dataclass
class FilterExpression:
    """
    Complex filter expression with logical operators
    
    Requirements: 6.2
    """
    conditions: List[FilterCondition] = field(default_factory=list)
    logical_operator: LogicalOperator = LogicalOperator.AND
    nested_expressions: List['FilterExpression'] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure logical operator is enum"""
        if isinstance(self.logical_operator, str):
            self.logical_operator = LogicalOperator(self.logical_operator)
    
    def evaluate(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization]
    ) -> bool:
        """
        Evaluate this expression against a resource
        
        Args:
            resource: Cloud resource to evaluate
            categorization: Optional resource categorization
            
        Returns:
            True if expression matches, False otherwise
        """
        # Evaluate all conditions
        condition_results = [
            cond.evaluate(resource, categorization)
            for cond in self.conditions
        ]
        
        # Evaluate nested expressions
        nested_results = [
            expr.evaluate(resource, categorization)
            for expr in self.nested_expressions
        ]
        
        # Combine all results
        all_results = condition_results + nested_results
        
        if not all_results:
            return True  # Empty expression matches everything
        
        # Apply logical operator
        if self.logical_operator == LogicalOperator.AND:
            return all(all_results)
        elif self.logical_operator == LogicalOperator.OR:
            return any(all_results)
        elif self.logical_operator == LogicalOperator.NOT:
            # NOT operator negates the first result
            return not all_results[0] if all_results else True
        
        return False
    
    def add_condition(self, condition: FilterCondition) -> 'FilterExpression':
        """Add a condition to this expression"""
        self.conditions.append(condition)
        return self
    
    def add_nested_expression(self, expression: 'FilterExpression') -> 'FilterExpression':
        """Add a nested expression"""
        self.nested_expressions.append(expression)
        return self


@dataclass
class FilterResult:
    """Result of applying a filter"""
    matched_resources: List[CloudResource] = field(default_factory=list)
    total_matched: int = 0
    total_evaluated: int = 0
    filter_expression: Optional[FilterExpression] = None
    execution_time_ms: float = 0.0


class AdvancedFilteringSystem:
    """
    Advanced filtering system with complex query support
    
    Requirements: 6.2
    """
    
    def __init__(self):
        """Initialize the advanced filtering system"""
        logger.info("Advanced Filtering System initialized")
    
    def filter_resources(
        self,
        resources: List[CloudResource],
        filter_expression: FilterExpression,
        categorizations: Optional[CategorizedResources] = None
    ) -> FilterResult:
        """
        Filter resources using a filter expression
        
        Args:
            resources: List of resources to filter
            filter_expression: Filter expression to apply
            categorizations: Optional resource categorizations
            
        Returns:
            FilterResult with matched resources
        """
        logger.info(f"Filtering {len(resources)} resources")
        start_time = datetime.utcnow()
        
        matched_resources = []
        
        for resource in resources:
            # Get categorization if available
            categorization = None
            if categorizations:
                categorization = categorizations.get_categorization(resource.resource_id)
            
            # Evaluate filter expression
            if filter_expression.evaluate(resource, categorization):
                matched_resources.append(resource)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        result = FilterResult(
            matched_resources=matched_resources,
            total_matched=len(matched_resources),
            total_evaluated=len(resources),
            filter_expression=filter_expression,
            execution_time_ms=execution_time
        )
        
        logger.info(
            f"Filter matched {result.total_matched}/{result.total_evaluated} resources "
            f"in {execution_time:.2f}ms"
        )
        
        return result
    
    def create_simple_filter(
        self,
        field: FilterField,
        operator: ComparisonOperator,
        value: Any,
        field_path: Optional[str] = None
    ) -> FilterExpression:
        """
        Create a simple filter expression with a single condition
        
        Args:
            field: Field to filter on
            operator: Comparison operator
            value: Value to compare against
            field_path: Optional path for nested fields
            
        Returns:
            FilterExpression with single condition
        """
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value,
            field_path=field_path
        )
        
        return FilterExpression(conditions=[condition])
    
    def create_and_filter(self, conditions: List[FilterCondition]) -> FilterExpression:
        """Create an AND filter expression"""
        return FilterExpression(
            conditions=conditions,
            logical_operator=LogicalOperator.AND
        )
    
    def create_or_filter(self, conditions: List[FilterCondition]) -> FilterExpression:
        """Create an OR filter expression"""
        return FilterExpression(
            conditions=conditions,
            logical_operator=LogicalOperator.OR
        )
    
    def create_not_filter(self, condition: FilterCondition) -> FilterExpression:
        """Create a NOT filter expression"""
        return FilterExpression(
            conditions=[condition],
            logical_operator=LogicalOperator.NOT
        )
    
    def combine_filters(
        self,
        expressions: List[FilterExpression],
        operator: LogicalOperator
    ) -> FilterExpression:
        """
        Combine multiple filter expressions with a logical operator
        
        Args:
            expressions: List of filter expressions to combine
            operator: Logical operator to use
            
        Returns:
            Combined FilterExpression
        """
        return FilterExpression(
            nested_expressions=expressions,
            logical_operator=operator
        )
    
    def filter_by_team(
        self,
        resources: List[CloudResource],
        team: str,
        categorizations: CategorizedResources
    ) -> FilterResult:
        """Filter resources by team"""
        filter_expr = self.create_simple_filter(
            FilterField.TEAM,
            ComparisonOperator.EQUALS,
            team
        )
        return self.filter_resources(resources, filter_expr, categorizations)
    
    def filter_by_project(
        self,
        resources: List[CloudResource],
        project: str,
        categorizations: CategorizedResources
    ) -> FilterResult:
        """Filter resources by project"""
        filter_expr = self.create_simple_filter(
            FilterField.PROJECT,
            ComparisonOperator.EQUALS,
            project
        )
        return self.filter_resources(resources, filter_expr, categorizations)
    
    def filter_by_environment(
        self,
        resources: List[CloudResource],
        environment: str,
        categorizations: CategorizedResources
    ) -> FilterResult:
        """Filter resources by environment"""
        filter_expr = self.create_simple_filter(
            FilterField.ENVIRONMENT,
            ComparisonOperator.EQUALS,
            environment
        )
        return self.filter_resources(resources, filter_expr, categorizations)
    
    def filter_by_region(
        self,
        resources: List[CloudResource],
        region: str
    ) -> FilterResult:
        """Filter resources by region"""
        filter_expr = self.create_simple_filter(
            FilterField.REGION,
            ComparisonOperator.EQUALS,
            region
        )
        return self.filter_resources(resources, filter_expr)
    
    def filter_by_resource_type(
        self,
        resources: List[CloudResource],
        resource_type: Union[str, ResourceType]
    ) -> FilterResult:
        """Filter resources by type"""
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        
        filter_expr = self.create_simple_filter(
            FilterField.RESOURCE_TYPE,
            ComparisonOperator.EQUALS,
            resource_type
        )
        return self.filter_resources(resources, filter_expr)
    
    def filter_by_tag(
        self,
        resources: List[CloudResource],
        tag_key: str,
        tag_value: Optional[str] = None
    ) -> FilterResult:
        """
        Filter resources by tag
        
        Args:
            resources: Resources to filter
            tag_key: Tag key to filter on
            tag_value: Optional tag value (if None, checks for tag existence)
            
        Returns:
            FilterResult
        """
        if tag_value is None:
            # Check for tag existence
            filter_expr = self.create_simple_filter(
                FilterField.TAGS,
                ComparisonOperator.EXISTS,
                None,
                field_path=tag_key
            )
        else:
            # Check for specific tag value
            filter_expr = self.create_simple_filter(
                FilterField.TAGS,
                ComparisonOperator.EQUALS,
                tag_value,
                field_path=tag_key
            )
        
        return self.filter_resources(resources, filter_expr)
    
    def filter_by_multiple_dimensions(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        region: Optional[str] = None
    ) -> FilterResult:
        """
        Filter resources by multiple dimensions (AND logic)
        
        Args:
            resources: Resources to filter
            categorizations: Resource categorizations
            team: Optional team filter
            project: Optional project filter
            environment: Optional environment filter
            region: Optional region filter
            
        Returns:
            FilterResult
        """
        conditions = []
        
        if team:
            conditions.append(FilterCondition(
                field=FilterField.TEAM,
                operator=ComparisonOperator.EQUALS,
                value=team
            ))
        
        if project:
            conditions.append(FilterCondition(
                field=FilterField.PROJECT,
                operator=ComparisonOperator.EQUALS,
                value=project
            ))
        
        if environment:
            conditions.append(FilterCondition(
                field=FilterField.ENVIRONMENT,
                operator=ComparisonOperator.EQUALS,
                value=environment
            ))
        
        if region:
            conditions.append(FilterCondition(
                field=FilterField.REGION,
                operator=ComparisonOperator.EQUALS,
                value=region
            ))
        
        if not conditions:
            # No filters, return all resources
            return FilterResult(
                matched_resources=resources,
                total_matched=len(resources),
                total_evaluated=len(resources)
            )
        
        filter_expr = self.create_and_filter(conditions)
        return self.filter_resources(resources, filter_expr, categorizations)
    
    def filter_unassigned_resources(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources
    ) -> FilterResult:
        """Filter resources that are not assigned to any team"""
        filter_expr = self.create_simple_filter(
            FilterField.TEAM,
            ComparisonOperator.NOT_EXISTS,
            None
        )
        return self.filter_resources(resources, filter_expr, categorizations)
    
    def parse_filter_query(self, query_string: str) -> FilterExpression:
        """
        Parse a filter query string into a FilterExpression
        
        Simple query format: "field operator value"
        Examples:
            - "team eq engineering"
            - "region in us-east-1,us-west-2"
            - "resource_type eq compute"
        
        Args:
            query_string: Query string to parse
            
        Returns:
            FilterExpression
        """
        logger.debug(f"Parsing filter query: {query_string}")
        
        # Simple parser for basic queries
        parts = query_string.strip().split(maxsplit=2)
        
        if len(parts) < 3:
            raise ValueError(f"Invalid query format: {query_string}")
        
        field_str, operator_str, value_str = parts
        
        # Parse field
        try:
            field = FilterField(field_str)
        except ValueError:
            raise ValueError(f"Invalid field: {field_str}")
        
        # Parse operator
        try:
            operator = ComparisonOperator(operator_str)
        except ValueError:
            raise ValueError(f"Invalid operator: {operator_str}")
        
        # Parse value
        value = self._parse_value(value_str, operator)
        
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value
        )
        
        return FilterExpression(conditions=[condition])
    
    def _parse_value(self, value_str: str, operator: ComparisonOperator) -> Any:
        """Parse value string based on operator"""
        # Handle list values for IN/NOT_IN operators
        if operator in [ComparisonOperator.IN, ComparisonOperator.NOT_IN]:
            return [v.strip() for v in value_str.split(',')]
        
        # Handle boolean values
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Handle numeric values
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
