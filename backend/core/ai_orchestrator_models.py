"""
AI Orchestrator Data Models

This module contains shared data models used by the AI orchestration system
to avoid circular imports.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AISystemType(Enum):
    """Types of AI systems in the orchestration"""
    PREDICTIVE_SCALING = "predictive_scaling"
    WORKLOAD_INTELLIGENCE = "workload_intelligence"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NATURAL_LANGUAGE = "natural_language"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    SMART_CONTRACT = "smart_contract"


class CoordinationStrategy(Enum):
    """Coordination strategies for multi-system optimization"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    PRIORITY_BASED = "priority_based"


class OptimizationScope(Enum):
    """Scope of optimization operations"""
    RESOURCE_LEVEL = "resource_level"
    SERVICE_LEVEL = "service_level"
    ACCOUNT_LEVEL = "account_level"
    ORGANIZATION_LEVEL = "organization_level"


@dataclass
class OptimizationContext:
    """Context for optimization operations"""
    user_id: str
    account_id: str
    resource_ids: List[str]
    optimization_goals: List[str]
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]
    historical_feedback: List[Dict[str, Any]]


@dataclass
class SystemRecommendation:
    """Recommendation from an AI system"""
    system_type: AISystemType
    recommendation_id: str
    resource_id: str
    action_type: str
    confidence: float
    expected_impact: Dict[str, float]
    rationale: str
    dependencies: List[str]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CoordinatedRecommendation:
    """Coordinated recommendation from multiple systems"""
    coordination_id: str
    primary_recommendation: SystemRecommendation
    supporting_recommendations: List[SystemRecommendation]
    coordination_strategy: CoordinationStrategy
    overall_confidence: float
    combined_impact: Dict[str, float]
    implementation_plan: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]


@dataclass
class AISystemStatus:
    """Status information for an AI system"""
    system_type: AISystemType
    is_active: bool
    last_update: datetime
    performance_metrics: Dict[str, float]
    error_count: int
    success_rate: float
    resource_usage: Dict[str, float]


@dataclass
class CoordinationRequest:
    """Request for multi-system coordination"""
    request_id: str
    requesting_system: AISystemType
    target_systems: List[AISystemType]
    coordination_type: str
    parameters: Dict[str, Any]
    priority: int
    created_at: datetime = field(default_factory=datetime.now)