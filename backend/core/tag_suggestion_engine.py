"""
Intelligent Tag Suggestion Engine

This module provides ML-powered tag suggestions based on resource patterns,
naming conventions, and organizational context for efficient remediation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class SuggestionConfidence(Enum):
    """Confidence levels for tag suggestions"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 70-89%
    MEDIUM = "medium"       # 50-69%
    LOW = "low"            # 30-49%
    VERY_LOW = "very_low"  # 0-29%


class SuggestionSource(Enum):
    """Sources of tag suggestions"""
    PATTERN_MATCHING = "pattern_matching"
    ML_PREDICTION = "ml_prediction"
    SIMILAR_RESOURCES = "similar_resources"
    NAMING_CONVENTION = "naming_convention"
    ORGANIZATIONAL_RULE = "organizational_rule"
    USER_HISTORY = "user_history"


@dataclass
class TagSuggestion:
    """Represents a tag suggestion"""
    tag_key: str
    suggested_value: str
    confidence: SuggestionConfidence
    confidence_score: float  # 0.0 to 1.0
    source: SuggestionSource
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    alternative_values: List[str] = field(default_factory=list)


@dataclass
class ResourceContext:
    """Context information for tag suggestions"""
    resource_id: str
    resource_type: str
    resource_name: Optional[str]
    provider: str
    region: str
    account_id: Optional[str]
    existing_tags: Dict[str, str]
    resource_attributes: Dict[str, Any]
    creation_time: Optional[datetime] = None
    similar_resources: List[str] = field(default_factory=list)


@dataclass
class BulkTaggingJob:
    """Represents a bulk tagging operation"""
    job_id: str
    resource_ids: List[str]
    tag_operations: List[Dict[str, Any]]  # List of {action, tag_key, tag_value}
    created_at: datetime
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternRule:
    """Rule for pattern-based tag suggestions"""
    rule_id: str
    name: str
    description: str
    pattern_type: str  # "regex", "contains", "starts_with", "ends_with"
    pattern: str
    target_attribute: str  # "resource_name", "resource_type", etc.
    suggested_tag_key: str
    suggested_tag_value: str
    confidence_modifier: float = 1.0
    active: bool = True


class TagSuggestionEngine:
    """
    Intelligent tag suggestion system using pattern recognition,
    ML predictions, and organizational context.
    """
    
    def __init__(self):
        self.pattern_rules: Dict[str, PatternRule] = {}
        self.ml_models: Dict[str, Any] = {}
        self.resource_similarity_cache: Dict[str, List[str]] = {}
        self.tag_frequency_cache: Dict[str, Counter] = {}
        self.organizational_mappings: Dict[str, Dict[str, str]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.bulk_jobs: Dict[str, BulkTaggingJob] = {}
        
        # Initialize default pattern rules
        self._initialize_default_patterns()
        
        # Load ML models if available
        self._load_ml_models()
    
    def _initialize_default_patterns(self):
        """Initialize common pattern-based tagging rules"""
        
        # Environment detection patterns
        env_patterns = [
            PatternRule(
                rule_id="env_dev_pattern",
                name="Development Environment Detection",
                description="Detect development resources by name",
                pattern_type="regex",
                pattern=r".*(dev|development|test|testing|sandbox).*",
                target_attribute="resource_name",
                suggested_tag_key="Environment",
                suggested_tag_value="dev",
                confidence_modifier=0.8
            ),
            PatternRule(
                rule_id="env_prod_pattern",
                name="Production Environment Detection",
                description="Detect production resources by name",
                pattern_type="regex",
                pattern=r".*(prod|production|live|main).*",
                target_attribute="resource_name",
                suggested_tag_key="Environment",
                suggested_tag_value="prod",
                confidence_modifier=0.9
            ),
            PatternRule(
                rule_id="env_staging_pattern",
                name="Staging Environment Detection",
                description="Detect staging resources by name",
                pattern_type="regex",
                pattern=r".*(staging|stage|uat|preprod).*",
                target_attribute="resource_name",
                suggested_tag_key="Environment",
                suggested_tag_value="staging",
                confidence_modifier=0.85
            )
        ]
        
        # Project detection patterns
        project_patterns = [
            PatternRule(
                rule_id="project_prefix_pattern",
                name="Project Prefix Detection",
                description="Extract project name from resource prefix",
                pattern_type="regex",
                pattern=r"^([a-zA-Z0-9-]+)-(.*)",
                target_attribute="resource_name",
                suggested_tag_key="Project",
                suggested_tag_value="$1",  # First capture group
                confidence_modifier=0.7
            )
        ]
        
        # Team detection patterns
        team_patterns = [
            PatternRule(
                rule_id="team_backend_pattern",
                name="Backend Team Detection",
                description="Detect backend team resources",
                pattern_type="regex",
                pattern=r".*(api|backend|service|microservice|database|db).*",
                target_attribute="resource_name",
                suggested_tag_key="Team",
                suggested_tag_value="backend",
                confidence_modifier=0.6
            ),
            PatternRule(
                rule_id="team_frontend_pattern",
                name="Frontend Team Detection",
                description="Detect frontend team resources",
                pattern_type="regex",
                pattern=r".*(frontend|web|ui|app|client).*",
                target_attribute="resource_name",
                suggested_tag_key="Team",
                suggested_tag_value="frontend",
                confidence_modifier=0.6
            ),
            PatternRule(
                rule_id="team_data_pattern",
                name="Data Team Detection",
                description="Detect data team resources",
                pattern_type="regex",
                pattern=r".*(data|analytics|etl|warehouse|lake|pipeline).*",
                target_attribute="resource_name",
                suggested_tag_key="Team",
                suggested_tag_value="data",
                confidence_modifier=0.7
            )
        ]
        
        # Combine all patterns
        all_patterns = env_patterns + project_patterns + team_patterns
        
        for pattern in all_patterns:
            self.pattern_rules[pattern.rule_id] = pattern
    
    def _load_ml_models(self):
        """Load pre-trained ML models for tag prediction"""
        model_dir = Path("models/tagging")
        
        if model_dir.exists():
            try:
                # Load tag value prediction models
                for model_file in model_dir.glob("*.pkl"):
                    tag_key = model_file.stem
                    with open(model_file, 'rb') as f:
                        self.ml_models[tag_key] = pickle.load(f)
                    logger.info(f"Loaded ML model for tag: {tag_key}")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {str(e)}")
    
    def suggest_tags(self, context: ResourceContext, 
                    required_tags: Optional[List[str]] = None) -> List[TagSuggestion]:
        """
        Generate intelligent tag suggestions for a resource
        """
        suggestions = []
        
        # Get pattern-based suggestions
        pattern_suggestions = self._get_pattern_suggestions(context)
        suggestions.extend(pattern_suggestions)
        
        # Get ML-based suggestions
        ml_suggestions = self._get_ml_suggestions(context)
        suggestions.extend(ml_suggestions)
        
        # Get similarity-based suggestions
        similarity_suggestions = self._get_similarity_suggestions(context)
        suggestions.extend(similarity_suggestions)
        
        # Get organizational rule suggestions
        org_suggestions = self._get_organizational_suggestions(context)
        suggestions.extend(org_suggestions)
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_and_rank_suggestions(
            suggestions, context, required_tags
        )
        
        return filtered_suggestions
    
    def _get_pattern_suggestions(self, context: ResourceContext) -> List[TagSuggestion]:
        """Generate suggestions based on pattern matching"""
        suggestions = []
        
        for rule in self.pattern_rules.values():
            if not rule.active:
                continue
            
            # Get target attribute value
            target_value = self._get_attribute_value(context, rule.target_attribute)
            if not target_value:
                continue
            
            # Check if pattern matches
            match = self._check_pattern_match(rule, target_value)
            if match:
                # Extract suggested value (handle regex capture groups)
                suggested_value = self._extract_suggested_value(rule, target_value, match)
                
                if suggested_value:
                    confidence_score = 0.7 * rule.confidence_modifier
                    confidence = self._score_to_confidence(confidence_score)
                    
                    suggestions.append(TagSuggestion(
                        tag_key=rule.suggested_tag_key,
                        suggested_value=suggested_value,
                        confidence=confidence,
                        confidence_score=confidence_score,
                        source=SuggestionSource.PATTERN_MATCHING,
                        reasoning=f"Matched pattern rule: {rule.name}",
                        supporting_evidence=[f"Pattern: {rule.pattern}", f"Matched: {target_value}"]
                    ))
        
        return suggestions
    
    def _get_ml_suggestions(self, context: ResourceContext) -> List[TagSuggestion]:
        """Generate suggestions using ML models"""
        suggestions = []
        
        for tag_key, model in self.ml_models.items():
            try:
                # Prepare features for ML model
                features = self._extract_ml_features(context)
                
                # Get prediction
                prediction = model.predict([features])[0]
                confidence_score = model.predict_proba([features])[0].max()
                
                if confidence_score > 0.3:  # Minimum confidence threshold
                    confidence = self._score_to_confidence(confidence_score)
                    
                    suggestions.append(TagSuggestion(
                        tag_key=tag_key,
                        suggested_value=prediction,
                        confidence=confidence,
                        confidence_score=confidence_score,
                        source=SuggestionSource.ML_PREDICTION,
                        reasoning=f"ML model prediction based on resource attributes",
                        supporting_evidence=[f"Model confidence: {confidence_score:.2f}"]
                    ))
            
            except Exception as e:
                logger.warning(f"ML prediction failed for tag {tag_key}: {str(e)}")
        
        return suggestions
    
    def _get_similarity_suggestions(self, context: ResourceContext) -> List[TagSuggestion]:
        """Generate suggestions based on similar resources"""
        suggestions = []
        
        # Find similar resources
        similar_resources = self._find_similar_resources(context)
        
        if similar_resources:
            # Analyze tags from similar resources
            tag_frequency = defaultdict(Counter)
            
            for similar_resource in similar_resources:
                # In a real implementation, you would fetch tags from similar resources
                # For now, we'll simulate this
                similar_tags = self._get_resource_tags(similar_resource)
                
                for tag_key, tag_value in similar_tags.items():
                    if tag_key not in context.existing_tags:
                        tag_frequency[tag_key][tag_value] += 1
            
            # Generate suggestions from frequent tags
            for tag_key, value_counts in tag_frequency.items():
                if value_counts:
                    most_common_value, frequency = value_counts.most_common(1)[0]
                    total_similar = len(similar_resources)
                    
                    confidence_score = frequency / total_similar
                    
                    if confidence_score > 0.3:
                        confidence = self._score_to_confidence(confidence_score)
                        
                        suggestions.append(TagSuggestion(
                            tag_key=tag_key,
                            suggested_value=most_common_value,
                            confidence=confidence,
                            confidence_score=confidence_score,
                            source=SuggestionSource.SIMILAR_RESOURCES,
                            reasoning=f"Common tag among {frequency}/{total_similar} similar resources",
                            supporting_evidence=[f"Similar resources: {len(similar_resources)}"]
                        ))
        
        return suggestions
    
    def _get_organizational_suggestions(self, context: ResourceContext) -> List[TagSuggestion]:
        """Generate suggestions based on organizational rules and mappings"""
        suggestions = []
        
        # Account-based mappings
        if context.account_id and context.account_id in self.organizational_mappings:
            mappings = self.organizational_mappings[context.account_id]
            
            for tag_key, tag_value in mappings.items():
                if tag_key not in context.existing_tags:
                    suggestions.append(TagSuggestion(
                        tag_key=tag_key,
                        suggested_value=tag_value,
                        confidence=SuggestionConfidence.HIGH,
                        confidence_score=0.8,
                        source=SuggestionSource.ORGANIZATIONAL_RULE,
                        reasoning=f"Organizational mapping for account {context.account_id}",
                        supporting_evidence=[f"Account mapping rule"]
                    ))
        
        # Region-based suggestions
        region_mappings = {
            "us-east-1": {"DataResidency": "US", "ComplianceZone": "US-East"},
            "us-west-2": {"DataResidency": "US", "ComplianceZone": "US-West"},
            "eu-west-1": {"DataResidency": "EU", "ComplianceZone": "EU-West"},
            "ap-southeast-1": {"DataResidency": "APAC", "ComplianceZone": "APAC-SE"}
        }
        
        if context.region in region_mappings:
            for tag_key, tag_value in region_mappings[context.region].items():
                if tag_key not in context.existing_tags:
                    suggestions.append(TagSuggestion(
                        tag_key=tag_key,
                        suggested_value=tag_value,
                        confidence=SuggestionConfidence.VERY_HIGH,
                        confidence_score=0.95,
                        source=SuggestionSource.ORGANIZATIONAL_RULE,
                        reasoning=f"Standard regional mapping for {context.region}",
                        supporting_evidence=[f"Region: {context.region}"]
                    ))
        
        return suggestions
    
    def _filter_and_rank_suggestions(self, suggestions: List[TagSuggestion], 
                                   context: ResourceContext,
                                   required_tags: Optional[List[str]] = None) -> List[TagSuggestion]:
        """Filter and rank suggestions by relevance and confidence"""
        
        # Remove duplicates (keep highest confidence)
        unique_suggestions = {}
        for suggestion in suggestions:
            key = (suggestion.tag_key, suggestion.suggested_value)
            if key not in unique_suggestions or suggestion.confidence_score > unique_suggestions[key].confidence_score:
                unique_suggestions[key] = suggestion
        
        filtered_suggestions = list(unique_suggestions.values())
        
        # Prioritize required tags
        if required_tags:
            required_suggestions = [s for s in filtered_suggestions if s.tag_key in required_tags]
            optional_suggestions = [s for s in filtered_suggestions if s.tag_key not in required_tags]
            filtered_suggestions = required_suggestions + optional_suggestions
        
        # Sort by confidence score (descending)
        filtered_suggestions.sort(key=lambda s: s.confidence_score, reverse=True)
        
        # Limit to top suggestions
        return filtered_suggestions[:20]
    
    def create_bulk_tagging_job(self, resource_contexts: List[ResourceContext],
                              auto_apply_threshold: float = 0.8) -> BulkTaggingJob:
        """Create a bulk tagging job for multiple resources"""
        job_id = f"bulk_tag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tag_operations = []
        
        for context in resource_contexts:
            suggestions = self.suggest_tags(context)
            
            # Auto-apply high-confidence suggestions
            for suggestion in suggestions:
                if suggestion.confidence_score >= auto_apply_threshold:
                    tag_operations.append({
                        "resource_id": context.resource_id,
                        "action": "add_tag",
                        "tag_key": suggestion.tag_key,
                        "tag_value": suggestion.suggested_value,
                        "confidence": suggestion.confidence_score,
                        "source": suggestion.source.value
                    })
        
        job = BulkTaggingJob(
            job_id=job_id,
            resource_ids=[ctx.resource_id for ctx in resource_contexts],
            tag_operations=tag_operations,
            created_at=datetime.now()
        )
        
        self.bulk_jobs[job_id] = job
        return job
    
    def execute_bulk_tagging_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a bulk tagging job"""
        if job_id not in self.bulk_jobs:
            return {"error": "Job not found"}
        
        job = self.bulk_jobs[job_id]
        job.status = "running"
        
        results = {
            "successful_operations": 0,
            "failed_operations": 0,
            "operations": []
        }
        
        try:
            for i, operation in enumerate(job.tag_operations):
                try:
                    # In a real implementation, this would call cloud provider APIs
                    # For now, we'll simulate the operation
                    success = self._apply_tag_operation(operation)
                    
                    if success:
                        results["successful_operations"] += 1
                        results["operations"].append({
                            "resource_id": operation["resource_id"],
                            "status": "success",
                            "operation": operation
                        })
                    else:
                        results["failed_operations"] += 1
                        results["operations"].append({
                            "resource_id": operation["resource_id"],
                            "status": "failed",
                            "error": "Simulated failure",
                            "operation": operation
                        })
                    
                    # Update progress
                    job.progress = (i + 1) / len(job.tag_operations)
                    
                except Exception as e:
                    results["failed_operations"] += 1
                    results["operations"].append({
                        "resource_id": operation["resource_id"],
                        "status": "error",
                        "error": str(e),
                        "operation": operation
                    })
            
            job.status = "completed"
            job.results = results
            
        except Exception as e:
            job.status = "failed"
            job.results = {"error": str(e)}
            logger.error(f"Bulk tagging job {job_id} failed: {str(e)}")
        
        return results
    
    def add_pattern_rule(self, rule: PatternRule) -> bool:
        """Add a new pattern rule"""
        try:
            self.pattern_rules[rule.rule_id] = rule
            logger.info(f"Added pattern rule: {rule.rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add pattern rule: {str(e)}")
            return False
    
    def update_organizational_mappings(self, account_id: str, mappings: Dict[str, str]):
        """Update organizational tag mappings for an account"""
        self.organizational_mappings[account_id] = mappings
        logger.info(f"Updated organizational mappings for account: {account_id}")
    
    def train_ml_model(self, tag_key: str, training_data: List[Dict[str, Any]]) -> bool:
        """Train ML model for tag prediction (simplified implementation)"""
        try:
            # In a real implementation, this would use scikit-learn or similar
            # For now, we'll create a simple frequency-based model
            
            value_counts = Counter()
            feature_patterns = defaultdict(Counter)
            
            for data_point in training_data:
                tag_value = data_point.get("tag_value")
                features = data_point.get("features", {})
                
                if tag_value:
                    value_counts[tag_value] += 1
                    
                    # Build feature patterns
                    for feature_name, feature_value in features.items():
                        feature_patterns[feature_name][feature_value] += 1
            
            # Create simple model
            model = {
                "type": "frequency_based",
                "value_counts": value_counts,
                "feature_patterns": feature_patterns,
                "most_common_value": value_counts.most_common(1)[0][0] if value_counts else None
            }
            
            self.ml_models[tag_key] = model
            logger.info(f"Trained ML model for tag: {tag_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML model for {tag_key}: {str(e)}")
            return False
    
    def get_suggestion_statistics(self) -> Dict[str, Any]:
        """Get statistics about tag suggestions"""
        return {
            "pattern_rules": len(self.pattern_rules),
            "active_pattern_rules": len([r for r in self.pattern_rules.values() if r.active]),
            "ml_models": len(self.ml_models),
            "organizational_mappings": len(self.organizational_mappings),
            "bulk_jobs": len(self.bulk_jobs),
            "completed_jobs": len([j for j in self.bulk_jobs.values() if j.status == "completed"]),
            "cached_similarities": len(self.resource_similarity_cache)
        }
    
    # Helper methods
    
    def _get_attribute_value(self, context: ResourceContext, attribute_name: str) -> Optional[str]:
        """Get attribute value from resource context"""
        if attribute_name == "resource_name":
            return context.resource_name
        elif attribute_name == "resource_type":
            return context.resource_type
        elif attribute_name == "provider":
            return context.provider
        elif attribute_name == "region":
            return context.region
        else:
            return context.resource_attributes.get(attribute_name)
    
    def _check_pattern_match(self, rule: PatternRule, value: str) -> Optional[re.Match]:
        """Check if a pattern rule matches a value"""
        if rule.pattern_type == "regex":
            return re.search(rule.pattern, value, re.IGNORECASE)
        elif rule.pattern_type == "contains":
            return value.lower().find(rule.pattern.lower()) != -1
        elif rule.pattern_type == "starts_with":
            return value.lower().startswith(rule.pattern.lower())
        elif rule.pattern_type == "ends_with":
            return value.lower().endswith(rule.pattern.lower())
        return None
    
    def _extract_suggested_value(self, rule: PatternRule, value: str, match: Any) -> Optional[str]:
        """Extract suggested tag value from pattern match"""
        suggested_value = rule.suggested_tag_value
        
        # Handle regex capture groups
        if rule.pattern_type == "regex" and hasattr(match, 'groups') and match.groups():
            # Replace $1, $2, etc. with capture groups
            for i, group in enumerate(match.groups(), 1):
                suggested_value = suggested_value.replace(f"${i}", group or "")
        
        return suggested_value.strip() if suggested_value else None
    
    def _extract_ml_features(self, context: ResourceContext) -> List[float]:
        """Extract features for ML model (simplified)"""
        # In a real implementation, this would extract meaningful features
        features = [
            len(context.resource_name or ""),
            len(context.existing_tags),
            hash(context.resource_type) % 1000,
            hash(context.provider) % 100,
            hash(context.region) % 50
        ]
        return features
    
    def _find_similar_resources(self, context: ResourceContext) -> List[str]:
        """Find resources similar to the given context"""
        # Simplified similarity based on resource type and name patterns
        similar_resources = []
        
        # In a real implementation, this would use more sophisticated similarity algorithms
        cache_key = f"{context.resource_type}_{context.provider}_{context.region}"
        
        if cache_key in self.resource_similarity_cache:
            return self.resource_similarity_cache[cache_key]
        
        # Simulate finding similar resources
        similar_resources = [f"similar-resource-{i}" for i in range(3)]
        self.resource_similarity_cache[cache_key] = similar_resources
        
        return similar_resources
    
    def _get_resource_tags(self, resource_id: str) -> Dict[str, str]:
        """Get tags for a resource (simulated)"""
        # In a real implementation, this would fetch from cloud provider or database
        simulated_tags = {
            "Environment": "prod",
            "Team": "backend",
            "Project": "user-service"
        }
        return simulated_tags
    
    def _score_to_confidence(self, score: float) -> SuggestionConfidence:
        """Convert numeric score to confidence enum"""
        if score >= 0.9:
            return SuggestionConfidence.VERY_HIGH
        elif score >= 0.7:
            return SuggestionConfidence.HIGH
        elif score >= 0.5:
            return SuggestionConfidence.MEDIUM
        elif score >= 0.3:
            return SuggestionConfidence.LOW
        else:
            return SuggestionConfidence.VERY_LOW
    
    def _apply_tag_operation(self, operation: Dict[str, Any]) -> bool:
        """Apply a tag operation (simulated)"""
        # In a real implementation, this would call cloud provider APIs
        # For now, we'll simulate success/failure
        import random
        return random.random() > 0.1  # 90% success rate


# Example usage and testing
if __name__ == "__main__":
    # Initialize tag suggestion engine
    engine = TagSuggestionEngine()
    
    # Create test resource context
    context = ResourceContext(
        resource_id="i-1234567890abcdef0",
        resource_type="ec2_instance",
        resource_name="backend-api-prod-server",
        provider="aws",
        region="us-east-1",
        account_id="123456789012",
        existing_tags={"Name": "backend-api-prod-server"},
        resource_attributes={
            "instance_type": "t3.medium",
            "vpc_id": "vpc-12345678",
            "subnet_id": "subnet-12345678"
        }
    )
    
    # Get tag suggestions
    suggestions = engine.suggest_tags(context, required_tags=["Environment", "Team", "Project"])
    
    print(f"Generated {len(suggestions)} tag suggestions:")
    for suggestion in suggestions:
        print(f"  {suggestion.tag_key}: {suggestion.suggested_value} "
              f"({suggestion.confidence.value}, {suggestion.confidence_score:.2f}) "
              f"- {suggestion.reasoning}")
    
    # Test bulk tagging
    contexts = [context]  # In practice, this would be many resources
    bulk_job = engine.create_bulk_tagging_job(contexts, auto_apply_threshold=0.7)
    print(f"\nCreated bulk job: {bulk_job.job_id} with {len(bulk_job.tag_operations)} operations")
    
    # Execute bulk job
    results = engine.execute_bulk_tagging_job(bulk_job.job_id)
    print(f"Bulk job results: {results['successful_operations']} successful, {results['failed_operations']} failed")
    
    # Get statistics
    stats = engine.get_suggestion_statistics()
    print(f"\nEngine statistics: {stats}")