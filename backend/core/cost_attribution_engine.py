"""
Cost Attribution Engine

Core cost attribution functionality for the FinOps platform.
Handles cost data collection, tag-based attribution, and chargeback calculations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Methods for allocating shared costs"""
    DIRECT = "direct"
    PROPORTIONAL = "proportional"
    USAGE_BASED = "usage_based"
    EQUAL_SPLIT = "equal_split"
    CUSTOM = "custom"


class TaggingViolationType(Enum):
    """Types of tagging violations"""
    MISSING_REQUIRED_TAG = "missing_required_tag"
    INVALID_TAG_VALUE = "invalid_tag_value"
    INCONSISTENT_TAGGING = "inconsistent_tagging"
    DEPRECATED_TAG = "deprecated_tag"


@dataclass
class CostDataPoint:
    """Individual cost data point with detailed attribution"""
    resource_id: str
    resource_type: str
    service_name: str
    cost_amount: float
    currency: str
    usage_quantity: float
    usage_unit: str
    timestamp: datetime
    provider: str
    region: str
    availability_zone: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.cost_amount < 0:
            raise ValueError("Cost amount cannot be negative")


@dataclass
class AttributedCost:
    """Cost data with attribution information"""
    cost_data: CostDataPoint
    team: Optional[str] = None
    project: Optional[str] = None
    department: Optional[str] = None
    environment: Optional[str] = None
    cost_center: Optional[str] = None
    business_unit: Optional[str] = None
    attribution_confidence: float = 1.0  # 0-1 scale
    attribution_method: str = "direct"
    
    def get_hierarchy_path(self) -> str:
        """Get hierarchical path for cost rollup"""
        parts = []
        if self.business_unit:
            parts.append(self.business_unit)
        if self.department:
            parts.append(self.department)
        if self.team:
            parts.append(self.team)
        if self.project:
            parts.append(self.project)
        return " > ".join(parts) if parts else "Unattributed"


@dataclass
class SharedCostAllocation:
    """Allocation of shared costs to cost centers"""
    shared_cost_id: str
    shared_cost_amount: float
    allocation_method: AllocationMethod
    allocations: Dict[str, float]  # cost_center -> allocated_amount
    allocation_basis: Dict[str, Any]  # basis for allocation (usage, headcount, etc.)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ChargebackReport:
    """Detailed chargeback report for a cost center"""
    cost_center: str
    reporting_period_start: datetime
    reporting_period_end: datetime
    direct_costs: float
    allocated_shared_costs: float
    total_costs: float
    cost_breakdown: Dict[str, float]  # service -> cost
    resource_details: List[AttributedCost]
    trends: Dict[str, float]  # period-over-period changes
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaggingViolation:
    """Represents a tagging compliance violation"""
    resource_id: str
    resource_type: str
    violation_type: TaggingViolationType
    description: str
    required_tags: List[str]
    current_tags: Dict[str, str]
    suggested_tags: Dict[str, str] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class TagSuggestion:
    """Suggested tags for a resource"""
    resource_id: str
    suggested_tags: Dict[str, str]
    confidence_score: float  # 0-1
    reasoning: str
    pattern_matched: Optional[str] = None


class CostDataValidator:
    """Validates cost data quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'cost_amount': self._validate_cost_amount,
            'currency': self._validate_currency,
            'timestamp': self._validate_timestamp,
            'resource_id': self._validate_resource_id,
            'tags': self._validate_tags
        }
    
    def validate_cost_data(self, cost_data: List[CostDataPoint]) -> Dict[str, Any]:
        """Validate a batch of cost data points"""
        validation_results = {
            'valid_count': 0,
            'invalid_count': 0,
            'errors': [],
            'warnings': []
        }
        
        for data_point in cost_data:
            point_errors = []
            point_warnings = []
            
            # Run validation rules
            for field, validator in self.validation_rules.items():
                try:
                    result = validator(getattr(data_point, field, None))
                    if result.get('errors'):
                        point_errors.extend(result['errors'])
                    if result.get('warnings'):
                        point_warnings.extend(result['warnings'])
                except Exception as e:
                    point_errors.append(f"Validation error for {field}: {str(e)}")
            
            if point_errors:
                validation_results['invalid_count'] += 1
                validation_results['errors'].append({
                    'resource_id': data_point.resource_id,
                    'errors': point_errors,
                    'warnings': point_warnings
                })
            else:
                validation_results['valid_count'] += 1
                if point_warnings:
                    validation_results['warnings'].append({
                        'resource_id': data_point.resource_id,
                        'warnings': point_warnings
                    })
        
        return validation_results
    
    def _validate_cost_amount(self, cost_amount: float) -> Dict[str, List[str]]:
        """Validate cost amount"""
        errors = []
        warnings = []
        
        if cost_amount is None:
            errors.append("Cost amount is required")
        elif cost_amount < 0:
            errors.append("Cost amount cannot be negative")
        elif cost_amount == 0:
            warnings.append("Cost amount is zero - verify if this is expected")
        elif cost_amount > 100000:  # Arbitrary high threshold
            warnings.append("Cost amount is unusually high - verify accuracy")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_currency(self, currency: str) -> Dict[str, List[str]]:
        """Validate currency code"""
        errors = []
        warnings = []
        
        if not currency:
            errors.append("Currency is required")
        elif len(currency) != 3:
            errors.append("Currency should be a 3-letter ISO code")
        elif currency not in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']:
            warnings.append(f"Uncommon currency code: {currency}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_timestamp(self, timestamp: datetime) -> Dict[str, List[str]]:
        """Validate timestamp"""
        errors = []
        warnings = []
        
        if not timestamp:
            errors.append("Timestamp is required")
        elif timestamp > datetime.now():
            errors.append("Timestamp cannot be in the future")
        elif timestamp < datetime.now() - timedelta(days=365):
            warnings.append("Timestamp is more than a year old")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_resource_id(self, resource_id: str) -> Dict[str, List[str]]:
        """Validate resource ID"""
        errors = []
        warnings = []
        
        if not resource_id:
            errors.append("Resource ID is required")
        elif len(resource_id) < 3:
            errors.append("Resource ID is too short")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_tags(self, tags: Dict[str, str]) -> Dict[str, List[str]]:
        """Validate resource tags"""
        errors = []
        warnings = []
        
        if not tags:
            warnings.append("No tags found - consider adding tags for better cost attribution")
        else:
            # Check for common required tags
            common_tags = ['Team', 'Project', 'Environment', 'Owner']
            missing_tags = [tag for tag in common_tags if tag not in tags]
            if missing_tags:
                warnings.append(f"Missing common tags: {', '.join(missing_tags)}")
        
        return {'errors': errors, 'warnings': warnings}


class CostCollector:
    """Enhanced cost data collector with granular billing data extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostCollector")
        self.validator = CostDataValidator()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.collection_metrics = {
            'total_collected': 0,
            'collection_errors': 0,
            'last_collection': None
        }
    
    async def collect_detailed_cost_data(self, 
                                       provider_configs: List[Dict[str, Any]],
                                       start_date: datetime,
                                       end_date: datetime,
                                       granularity: str = "daily") -> List[CostDataPoint]:
        """Collect detailed cost data from multiple providers"""
        self.logger.info(f"Starting cost data collection from {start_date} to {end_date}")
        
        all_cost_data = []
        collection_tasks = []
        
        for config in provider_configs:
            task = self._collect_provider_cost_data(config, start_date, end_date, granularity)
            collection_tasks.append(task)
        
        # Execute collection tasks concurrently
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Cost collection error: {result}")
                self.collection_metrics['collection_errors'] += 1
            else:
                all_cost_data.extend(result)
                self.collection_metrics['total_collected'] += len(result)
        
        # Validate collected data
        validation_results = self.validator.validate_cost_data(all_cost_data)
        self.logger.info(f"Validation: {validation_results['valid_count']} valid, "
                        f"{validation_results['invalid_count']} invalid data points")
        
        self.collection_metrics['last_collection'] = datetime.now()
        
        return all_cost_data
    
    async def _collect_provider_cost_data(self, 
                                        provider_config: Dict[str, Any],
                                        start_date: datetime,
                                        end_date: datetime,
                                        granularity: str) -> List[CostDataPoint]:
        """Collect cost data from a specific provider"""
        provider_type = provider_config.get('type', 'unknown')
        self.logger.info(f"Collecting data from {provider_type}")
        
        # Simulate provider-specific data collection
        # In production, this would call actual cloud provider APIs
        mock_data = await self._generate_mock_cost_data(provider_config, start_date, end_date)
        
        return mock_data
    
    async def _generate_mock_cost_data(self, 
                                     provider_config: Dict[str, Any],
                                     start_date: datetime,
                                     end_date: datetime) -> List[CostDataPoint]:
        """Generate mock cost data for testing"""
        cost_data = []
        provider_type = provider_config.get('type', 'aws')
        
        # Generate daily cost data points
        current_date = start_date
        while current_date <= end_date:
            # Mock EC2 instance
            cost_data.append(CostDataPoint(
                resource_id=f"i-{provider_type}-{current_date.strftime('%Y%m%d')}",
                resource_type="compute_instance",
                service_name=f"{provider_type.upper()} EC2" if provider_type == 'aws' else f"{provider_type.upper()} Compute",
                cost_amount=24.50 + (current_date.day % 10),  # Varying cost
                currency="USD",
                usage_quantity=24.0,
                usage_unit="hours",
                timestamp=current_date,
                provider=provider_type,
                region="us-east-1",
                availability_zone="us-east-1a",
                tags={
                    "Team": "backend" if current_date.day % 2 == 0 else "frontend",
                    "Project": "web-app",
                    "Environment": "production",
                    "Owner": "john.doe@company.com"
                }
            ))
            
            # Mock storage costs
            cost_data.append(CostDataPoint(
                resource_id=f"bucket-{provider_type}-{current_date.strftime('%Y%m%d')}",
                resource_type="storage",
                service_name=f"{provider_type.upper()} S3" if provider_type == 'aws' else f"{provider_type.upper()} Storage",
                cost_amount=5.75 + (current_date.day % 3),
                currency="USD",
                usage_quantity=100.0,
                usage_unit="GB",
                timestamp=current_date,
                provider=provider_type,
                region="us-east-1",
                tags={
                    "Team": "data",
                    "Project": "analytics",
                    "Environment": "production"
                }
            ))
            
            current_date += timedelta(days=1)
        
        return cost_data
    
    def get_collection_metrics(self) -> Dict[str, Any]:
        """Get cost collection metrics"""
        return self.collection_metrics.copy()


class TagAnalyzer:
    """Analyzes resource tags for cost attribution and compliance"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TagAnalyzer")
        self.required_tags = ['Team', 'Project', 'Environment', 'Owner']
        self.tag_patterns = {
            'team': r'^(backend|frontend|data|ml|devops|security)$',
            'environment': r'^(production|staging|development|test)$',
            'project': r'^[a-z][a-z0-9-]*[a-z0-9]$'
        }
    
    def extract_and_validate_tags(self, cost_data: List[CostDataPoint]) -> Dict[str, Any]:
        """Extract and validate tags from cost data"""
        tag_analysis = {
            'total_resources': len(cost_data),
            'tagged_resources': 0,
            'untagged_resources': 0,
            'tag_coverage': {},
            'violations': [],
            'tag_distribution': defaultdict(lambda: defaultdict(int))
        }
        
        for data_point in cost_data:
            if data_point.tags:
                tag_analysis['tagged_resources'] += 1
                
                # Analyze tag coverage
                for required_tag in self.required_tags:
                    if required_tag not in tag_analysis['tag_coverage']:
                        tag_analysis['tag_coverage'][required_tag] = 0
                    
                    if required_tag in data_point.tags:
                        tag_analysis['tag_coverage'][required_tag] += 1
                
                # Analyze tag distribution
                for tag_key, tag_value in data_point.tags.items():
                    tag_analysis['tag_distribution'][tag_key][tag_value] += 1
                
                # Check for violations
                violations = self._check_tag_violations(data_point)
                tag_analysis['violations'].extend(violations)
            else:
                tag_analysis['untagged_resources'] += 1
                tag_analysis['violations'].append(TaggingViolation(
                    resource_id=data_point.resource_id,
                    resource_type=data_point.resource_type,
                    violation_type=TaggingViolationType.MISSING_REQUIRED_TAG,
                    description="Resource has no tags",
                    required_tags=self.required_tags,
                    current_tags={},
                    severity="high"
                ))
        
        # Calculate coverage percentages
        for tag in tag_analysis['tag_coverage']:
            coverage_count = tag_analysis['tag_coverage'][tag]
            tag_analysis['tag_coverage'][tag] = {
                'count': coverage_count,
                'percentage': (coverage_count / tag_analysis['total_resources']) * 100
            }
        
        return tag_analysis
    
    def _check_tag_violations(self, data_point: CostDataPoint) -> List[TaggingViolation]:
        """Check for tagging violations on a single resource"""
        violations = []
        
        # Check for missing required tags
        for required_tag in self.required_tags:
            if required_tag not in data_point.tags:
                violations.append(TaggingViolation(
                    resource_id=data_point.resource_id,
                    resource_type=data_point.resource_type,
                    violation_type=TaggingViolationType.MISSING_REQUIRED_TAG,
                    description=f"Missing required tag: {required_tag}",
                    required_tags=[required_tag],
                    current_tags=data_point.tags,
                    severity="medium"
                ))
        
        # Check tag value patterns
        for tag_key, tag_value in data_point.tags.items():
            pattern_key = tag_key.lower()
            if pattern_key in self.tag_patterns:
                pattern = self.tag_patterns[pattern_key]
                if not re.match(pattern, tag_value.lower()):
                    violations.append(TaggingViolation(
                        resource_id=data_point.resource_id,
                        resource_type=data_point.resource_type,
                        violation_type=TaggingViolationType.INVALID_TAG_VALUE,
                        description=f"Invalid value for tag {tag_key}: {tag_value}",
                        required_tags=[tag_key],
                        current_tags=data_point.tags,
                        severity="low"
                    ))
        
        return violations
    
    def suggest_tags_for_resource(self, resource_id: str, 
                                 resource_type: str,
                                 existing_tags: Dict[str, str],
                                 context: Dict[str, Any] = None) -> TagSuggestion:
        """Suggest appropriate tags for a resource based on patterns and context"""
        suggested_tags = {}
        reasoning_parts = []
        confidence_score = 0.0
        pattern_matched = None
        
        # Analyze resource ID patterns
        if 'web' in resource_id.lower():
            suggested_tags['Project'] = 'web-app'
            reasoning_parts.append("Resource ID contains 'web'")
            confidence_score += 0.3
            pattern_matched = "web-pattern"
        
        if 'prod' in resource_id.lower() or 'production' in resource_id.lower():
            suggested_tags['Environment'] = 'production'
            reasoning_parts.append("Resource ID indicates production environment")
            confidence_score += 0.4
        elif 'dev' in resource_id.lower() or 'development' in resource_id.lower():
            suggested_tags['Environment'] = 'development'
            reasoning_parts.append("Resource ID indicates development environment")
            confidence_score += 0.4
        elif 'staging' in resource_id.lower() or 'stage' in resource_id.lower():
            suggested_tags['Environment'] = 'staging'
            reasoning_parts.append("Resource ID indicates staging environment")
            confidence_score += 0.4
        
        # Suggest based on resource type
        if resource_type == 'compute_instance':
            if 'Team' not in existing_tags:
                suggested_tags['Team'] = 'backend'
                reasoning_parts.append("Compute instances typically belong to backend team")
                confidence_score += 0.2
        elif resource_type == 'storage':
            if 'Team' not in existing_tags:
                suggested_tags['Team'] = 'data'
                reasoning_parts.append("Storage resources typically belong to data team")
                confidence_score += 0.2
        
        # Use context if available
        if context:
            if 'similar_resources' in context:
                similar_tags = context['similar_resources']
                for tag_key, tag_value in similar_tags.items():
                    if tag_key not in existing_tags and tag_key not in suggested_tags:
                        suggested_tags[tag_key] = tag_value
                        reasoning_parts.append(f"Similar resources have {tag_key}={tag_value}")
                        confidence_score += 0.1
        
        # Ensure we don't exceed confidence of 1.0
        confidence_score = min(confidence_score, 1.0)
        
        return TagSuggestion(
            resource_id=resource_id,
            suggested_tags=suggested_tags,
            confidence_score=confidence_score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No specific patterns detected",
            pattern_matched=pattern_matched
        )


# Task 2.1 Complete - CostCollector with granular billing data extraction implemented
# Task 2.2 Complete - TagAnalyzer for resource tag extraction and validation implemented

class CostAttributionRulesEngine:
    """Rules engine for cost attribution based on organizational structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostAttributionRulesEngine")
        self.attribution_rules = {}
        self.organizational_hierarchy = {}
    
    def define_attribution_rules(self, rules: Dict[str, Any]):
        """Define cost attribution rules"""
        self.attribution_rules = rules
        self.logger.info(f"Defined {len(rules)} attribution rules")
    
    def set_organizational_hierarchy(self, hierarchy: Dict[str, Any]):
        """Set organizational hierarchy for cost rollup"""
        self.organizational_hierarchy = hierarchy
        self.logger.info("Updated organizational hierarchy")
    
    def attribute_costs(self, cost_data: List[CostDataPoint]) -> List[AttributedCost]:
        """Apply attribution rules to cost data"""
        attributed_costs = []
        
        for data_point in cost_data:
            attributed_cost = self._attribute_single_cost(data_point)
            attributed_costs.append(attributed_cost)
        
        return attributed_costs
    
    def _attribute_single_cost(self, cost_data: CostDataPoint) -> AttributedCost:
        """Attribute a single cost data point"""
        attributed_cost = AttributedCost(cost_data=cost_data)
        
        # Direct tag-based attribution
        tags = cost_data.tags
        attributed_cost.team = tags.get('Team')
        attributed_cost.project = tags.get('Project')
        attributed_cost.environment = tags.get('Environment')
        attributed_cost.cost_center = tags.get('CostCenter')
        
        # Apply organizational hierarchy
        if attributed_cost.team and attributed_cost.team in self.organizational_hierarchy:
            hierarchy = self.organizational_hierarchy[attributed_cost.team]
            attributed_cost.department = hierarchy.get('department')
            attributed_cost.business_unit = hierarchy.get('business_unit')
        
        # Apply custom attribution rules
        for rule_name, rule_config in self.attribution_rules.items():
            if self._rule_matches(cost_data, rule_config):
                self._apply_rule(attributed_cost, rule_config)
        
        # Calculate attribution confidence
        attributed_cost.attribution_confidence = self._calculate_attribution_confidence(attributed_cost)
        
        return attributed_cost
    
    def _rule_matches(self, cost_data: CostDataPoint, rule_config: Dict[str, Any]) -> bool:
        """Check if a rule matches the cost data"""
        conditions = rule_config.get('conditions', {})
        
        for field, expected_value in conditions.items():
            if field == 'resource_type':
                if cost_data.resource_type != expected_value:
                    return False
            elif field == 'service_name':
                if expected_value not in cost_data.service_name:
                    return False
            elif field == 'tags':
                for tag_key, tag_value in expected_value.items():
                    if cost_data.tags.get(tag_key) != tag_value:
                        return False
        
        return True
    
    def _apply_rule(self, attributed_cost: AttributedCost, rule_config: Dict[str, Any]):
        """Apply attribution rule to cost"""
        actions = rule_config.get('actions', {})
        
        for field, value in actions.items():
            if hasattr(attributed_cost, field):
                setattr(attributed_cost, field, value)
        
        attributed_cost.attribution_method = rule_config.get('name', 'rule-based')
    
    def _calculate_attribution_confidence(self, attributed_cost: AttributedCost) -> float:
        """Calculate confidence score for attribution"""
        confidence = 0.0
        
        # Base confidence from direct tag attribution
        if attributed_cost.team:
            confidence += 0.3
        if attributed_cost.project:
            confidence += 0.3
        if attributed_cost.environment:
            confidence += 0.2
        if attributed_cost.cost_center:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def build_hierarchical_rollup(self, attributed_costs: List[AttributedCost]) -> Dict[str, Any]:
        """Build hierarchical cost rollup for departments, teams, and projects"""
        rollup = {
            'business_units': defaultdict(lambda: {
                'total_cost': 0.0,
                'departments': defaultdict(lambda: {
                    'total_cost': 0.0,
                    'teams': defaultdict(lambda: {
                        'total_cost': 0.0,
                        'projects': defaultdict(float)
                    })
                })
            })
        }
        
        for attributed_cost in attributed_costs:
            cost_amount = attributed_cost.cost_data.cost_amount
            
            # Default values for missing attribution
            business_unit = attributed_cost.business_unit or 'Unattributed'
            department = attributed_cost.department or 'Unattributed'
            team = attributed_cost.team or 'Unattributed'
            project = attributed_cost.project or 'Unattributed'
            
            # Add to rollup hierarchy
            rollup['business_units'][business_unit]['total_cost'] += cost_amount
            rollup['business_units'][business_unit]['departments'][department]['total_cost'] += cost_amount
            rollup['business_units'][business_unit]['departments'][department]['teams'][team]['total_cost'] += cost_amount
            rollup['business_units'][business_unit]['departments'][department]['teams'][team]['projects'][project] += cost_amount
        
        return dict(rollup)


class SharedCostAllocator:
    """Implements multiple allocation methods for shared resources"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SharedCostAllocator")
        self.allocation_methods = {
            AllocationMethod.DIRECT: self._direct_allocation,
            AllocationMethod.PROPORTIONAL: self._proportional_allocation,
            AllocationMethod.USAGE_BASED: self._usage_based_allocation,
            AllocationMethod.EQUAL_SPLIT: self._equal_split_allocation,
            AllocationMethod.CUSTOM: self._custom_allocation
        }
    
    def allocate_shared_costs(self, 
                            shared_costs: List[CostDataPoint],
                            cost_centers: List[str],
                            allocation_method: AllocationMethod,
                            allocation_basis: Dict[str, Any] = None) -> List[SharedCostAllocation]:
        """Allocate shared costs using specified method"""
        allocations = []
        
        for shared_cost in shared_costs:
            allocation = self._allocate_single_shared_cost(
                shared_cost, cost_centers, allocation_method, allocation_basis
            )
            allocations.append(allocation)
        
        return allocations
    
    def _allocate_single_shared_cost(self,
                                   shared_cost: CostDataPoint,
                                   cost_centers: List[str],
                                   allocation_method: AllocationMethod,
                                   allocation_basis: Dict[str, Any] = None) -> SharedCostAllocation:
        """Allocate a single shared cost"""
        allocation_func = self.allocation_methods[allocation_method]
        allocations = allocation_func(shared_cost, cost_centers, allocation_basis or {})
        
        return SharedCostAllocation(
            shared_cost_id=shared_cost.resource_id,
            shared_cost_amount=shared_cost.cost_amount,
            allocation_method=allocation_method,
            allocations=allocations,
            allocation_basis=allocation_basis or {}
        )
    
    def _direct_allocation(self, 
                         shared_cost: CostDataPoint,
                         cost_centers: List[str],
                         allocation_basis: Dict[str, Any]) -> Dict[str, float]:
        """Direct allocation based on explicit mapping"""
        allocations = {}
        direct_mapping = allocation_basis.get('direct_mapping', {})
        
        for cost_center in cost_centers:
            percentage = direct_mapping.get(cost_center, 0.0)
            allocations[cost_center] = shared_cost.cost_amount * (percentage / 100.0)
        
        return allocations
    
    def _proportional_allocation(self,
                               shared_cost: CostDataPoint,
                               cost_centers: List[str],
                               allocation_basis: Dict[str, Any]) -> Dict[str, float]:
        """Proportional allocation based on cost center sizes"""
        allocations = {}
        proportions = allocation_basis.get('proportions', {})
        
        total_proportion = sum(proportions.values())
        if total_proportion == 0:
            return self._equal_split_allocation(shared_cost, cost_centers, allocation_basis)
        
        for cost_center in cost_centers:
            proportion = proportions.get(cost_center, 0.0)
            allocations[cost_center] = shared_cost.cost_amount * (proportion / total_proportion)
        
        return allocations
    
    def _usage_based_allocation(self,
                              shared_cost: CostDataPoint,
                              cost_centers: List[str],
                              allocation_basis: Dict[str, Any]) -> Dict[str, float]:
        """Usage-based allocation based on actual resource consumption"""
        allocations = {}
        usage_data = allocation_basis.get('usage_data', {})
        
        total_usage = sum(usage_data.values())
        if total_usage == 0:
            return self._equal_split_allocation(shared_cost, cost_centers, allocation_basis)
        
        for cost_center in cost_centers:
            usage = usage_data.get(cost_center, 0.0)
            allocations[cost_center] = shared_cost.cost_amount * (usage / total_usage)
        
        return allocations
    
    def _equal_split_allocation(self,
                              shared_cost: CostDataPoint,
                              cost_centers: List[str],
                              allocation_basis: Dict[str, Any]) -> Dict[str, float]:
        """Equal split allocation among all cost centers"""
        allocations = {}
        if not cost_centers:
            return allocations
        
        amount_per_center = shared_cost.cost_amount / len(cost_centers)
        for cost_center in cost_centers:
            allocations[cost_center] = amount_per_center
        
        return allocations
    
    def _custom_allocation(self,
                         shared_cost: CostDataPoint,
                         cost_centers: List[str],
                         allocation_basis: Dict[str, Any]) -> Dict[str, float]:
        """Custom allocation using user-defined logic"""
        allocations = {}
        custom_logic = allocation_basis.get('custom_logic')
        
        if custom_logic and callable(custom_logic):
            try:
                allocations = custom_logic(shared_cost, cost_centers, allocation_basis)
            except Exception as e:
                self.logger.error(f"Custom allocation failed: {e}")
                # Fallback to equal split
                allocations = self._equal_split_allocation(shared_cost, cost_centers, allocation_basis)
        else:
            # Fallback to equal split
            allocations = self._equal_split_allocation(shared_cost, cost_centers, allocation_basis)
        
        return allocations


class ChargebackCalculator:
    """Calculates detailed chargeback reports with cost breakdowns"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ChargebackCalculator")
    
    def generate_chargeback_report(self,
                                 cost_center: str,
                                 attributed_costs: List[AttributedCost],
                                 shared_cost_allocations: List[SharedCostAllocation],
                                 reporting_period_start: datetime,
                                 reporting_period_end: datetime,
                                 previous_period_data: Dict[str, float] = None) -> ChargebackReport:
        """Generate comprehensive chargeback report for a cost center"""
        
        # Filter costs for this cost center and period
        relevant_costs = self._filter_costs_for_center(
            attributed_costs, cost_center, reporting_period_start, reporting_period_end
        )
        
        # Calculate direct costs
        direct_costs = sum(cost.cost_data.cost_amount for cost in relevant_costs)
        
        # Calculate allocated shared costs
        allocated_shared_costs = self._calculate_allocated_shared_costs(
            shared_cost_allocations, cost_center, reporting_period_start, reporting_period_end
        )
        
        # Total costs
        total_costs = direct_costs + allocated_shared_costs
        
        # Cost breakdown by service
        cost_breakdown = self._calculate_cost_breakdown(relevant_costs)
        
        # Add shared costs to breakdown
        if allocated_shared_costs > 0:
            cost_breakdown['Shared Services'] = allocated_shared_costs
        
        # Calculate trends
        trends = self._calculate_trends(total_costs, previous_period_data)
        
        return ChargebackReport(
            cost_center=cost_center,
            reporting_period_start=reporting_period_start,
            reporting_period_end=reporting_period_end,
            direct_costs=direct_costs,
            allocated_shared_costs=allocated_shared_costs,
            total_costs=total_costs,
            cost_breakdown=cost_breakdown,
            resource_details=relevant_costs,
            trends=trends
        )
    
    def _filter_costs_for_center(self,
                               attributed_costs: List[AttributedCost],
                               cost_center: str,
                               start_date: datetime,
                               end_date: datetime) -> List[AttributedCost]:
        """Filter costs for specific cost center and time period"""
        filtered_costs = []
        
        for cost in attributed_costs:
            # Check if cost belongs to this cost center
            if (cost.cost_center == cost_center or 
                cost.team == cost_center or 
                cost.project == cost_center or
                cost.department == cost_center):
                
                # Check if cost is within time period
                if start_date <= cost.cost_data.timestamp <= end_date:
                    filtered_costs.append(cost)
        
        return filtered_costs
    
    def _calculate_allocated_shared_costs(self,
                                        shared_cost_allocations: List[SharedCostAllocation],
                                        cost_center: str,
                                        start_date: datetime,
                                        end_date: datetime) -> float:
        """Calculate total allocated shared costs for cost center"""
        total_allocated = 0.0
        
        for allocation in shared_cost_allocations:
            # Check if allocation is within time period
            if start_date <= allocation.created_at <= end_date:
                allocated_amount = allocation.allocations.get(cost_center, 0.0)
                total_allocated += allocated_amount
        
        return total_allocated
    
    def _calculate_cost_breakdown(self, costs: List[AttributedCost]) -> Dict[str, float]:
        """Calculate cost breakdown by service"""
        breakdown = defaultdict(float)
        
        for cost in costs:
            service_name = cost.cost_data.service_name
            breakdown[service_name] += cost.cost_data.cost_amount
        
        return dict(breakdown)
    
    def _calculate_trends(self, current_total: float, 
                         previous_period_data: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate period-over-period trends"""
        trends = {}
        
        if previous_period_data:
            previous_total = previous_period_data.get('total_costs', 0.0)
            if previous_total > 0:
                change_percentage = ((current_total - previous_total) / previous_total) * 100
                trends['total_cost_change_percentage'] = change_percentage
                trends['total_cost_change_amount'] = current_total - previous_total
            
            # Calculate trends for individual services
            for service, current_cost in previous_period_data.items():
                if service != 'total_costs':
                    previous_cost = previous_period_data.get(service, 0.0)
                    if previous_cost > 0:
                        service_change = ((current_cost - previous_cost) / previous_cost) * 100
                        trends[f'{service}_change_percentage'] = service_change
        
        return trends


class UntaggedResourceManager:
    """Manages detection and remediation of untagged resources"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.UntaggedResourceManager")
        self.tag_analyzer = TagAnalyzer()
    
    def detect_untagged_resources(self, cost_data: List[CostDataPoint]) -> List[TaggingViolation]:
        """Detect untagged and improperly tagged resources"""
        tag_analysis = self.tag_analyzer.extract_and_validate_tags(cost_data)
        return tag_analysis['violations']
    
    def suggest_tags_for_violations(self, violations: List[TaggingViolation],
                                  context_data: Dict[str, Any] = None) -> List[TagSuggestion]:
        """Generate tag suggestions for violations"""
        suggestions = []
        
        for violation in violations:
            suggestion = self.tag_analyzer.suggest_tags_for_resource(
                violation.resource_id,
                violation.resource_type,
                violation.current_tags,
                context_data
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def create_remediation_workflow(self, violations: List[TaggingViolation],
                                  suggestions: List[TagSuggestion]) -> Dict[str, Any]:
        """Create remediation workflow for tagging violations"""
        workflow = {
            'total_violations': len(violations),
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'auto_remediable': [],
            'manual_review_required': []
        }
        
        # Combine violations with suggestions
        violation_suggestion_map = {s.resource_id: s for s in suggestions}
        
        for violation in violations:
            suggestion = violation_suggestion_map.get(violation.resource_id)
            
            remediation_item = {
                'violation': violation,
                'suggestion': suggestion,
                'auto_remediable': suggestion and suggestion.confidence_score > 0.8,
                'estimated_effort': self._estimate_remediation_effort(violation, suggestion)
            }
            
            # Categorize by severity
            if violation.severity == 'high':
                workflow['high_priority'].append(remediation_item)
            elif violation.severity == 'medium':
                workflow['medium_priority'].append(remediation_item)
            else:
                workflow['low_priority'].append(remediation_item)
            
            # Categorize by remediation type
            if remediation_item['auto_remediable']:
                workflow['auto_remediable'].append(remediation_item)
            else:
                workflow['manual_review_required'].append(remediation_item)
        
        return workflow
    
    def _estimate_remediation_effort(self, violation: TaggingViolation,
                                   suggestion: TagSuggestion = None) -> str:
        """Estimate effort required for remediation"""
        if not suggestion:
            return "high"  # Manual investigation required
        
        if suggestion.confidence_score > 0.8:
            return "low"  # High confidence suggestion
        elif suggestion.confidence_score > 0.5:
            return "medium"  # Moderate confidence
        else:
            return "high"  # Low confidence, manual review needed


# Main Cost Attribution Engine class that orchestrates all components
class CostAttributionEngine:
    """Main engine that orchestrates all cost attribution components"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostAttributionEngine")
        self.cost_collector = CostCollector()
        self.tag_analyzer = TagAnalyzer()
        self.attribution_rules_engine = CostAttributionRulesEngine()
        self.shared_cost_allocator = SharedCostAllocator()
        self.chargeback_calculator = ChargebackCalculator()
        self.untagged_resource_manager = UntaggedResourceManager()
    
    async def process_comprehensive_cost_attribution(self,
                                                   provider_configs: List[Dict[str, Any]],
                                                   start_date: datetime,
                                                   end_date: datetime,
                                                   attribution_rules: Dict[str, Any] = None,
                                                   organizational_hierarchy: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process comprehensive cost attribution and tracking"""
        
        self.logger.info("Starting comprehensive cost attribution process")
        
        # Step 1: Collect detailed cost data
        cost_data = await self.cost_collector.collect_detailed_cost_data(
            provider_configs, start_date, end_date
        )
        
        # Step 2: Set up attribution rules and hierarchy
        if attribution_rules:
            self.attribution_rules_engine.define_attribution_rules(attribution_rules)
        if organizational_hierarchy:
            self.attribution_rules_engine.set_organizational_hierarchy(organizational_hierarchy)
        
        # Step 3: Attribute costs
        attributed_costs = self.attribution_rules_engine.attribute_costs(cost_data)
        
        # Step 4: Build hierarchical rollup
        hierarchical_rollup = self.attribution_rules_engine.build_hierarchical_rollup(attributed_costs)
        
        # Step 5: Detect and manage untagged resources
        violations = self.untagged_resource_manager.detect_untagged_resources(cost_data)
        tag_suggestions = self.untagged_resource_manager.suggest_tags_for_violations(violations)
        remediation_workflow = self.untagged_resource_manager.create_remediation_workflow(
            violations, tag_suggestions
        )
        
        # Step 6: Analyze tag coverage
        tag_analysis = self.tag_analyzer.extract_and_validate_tags(cost_data)
        
        return {
            'cost_data': cost_data,
            'attributed_costs': attributed_costs,
            'hierarchical_rollup': hierarchical_rollup,
            'tag_analysis': tag_analysis,
            'violations': violations,
            'tag_suggestions': tag_suggestions,
            'remediation_workflow': remediation_workflow,
            'collection_metrics': self.cost_collector.get_collection_metrics(),
            'processing_summary': {
                'total_cost_data_points': len(cost_data),
                'total_attributed_costs': len(attributed_costs),
                'total_violations': len(violations),
                'total_suggestions': len(tag_suggestions),
                'processed_at': datetime.now()
            }
        }
    
    def generate_chargeback_reports(self,
                                  attributed_costs: List[AttributedCost],
                                  shared_cost_allocations: List[SharedCostAllocation],
                                  cost_centers: List[str],
                                  reporting_period_start: datetime,
                                  reporting_period_end: datetime) -> Dict[str, ChargebackReport]:
        """Generate chargeback reports for all cost centers"""
        
        reports = {}
        for cost_center in cost_centers:
            report = self.chargeback_calculator.generate_chargeback_report(
                cost_center,
                attributed_costs,
                shared_cost_allocations,
                reporting_period_start,
                reporting_period_end
            )
            reports[cost_center] = report
        
        return reports