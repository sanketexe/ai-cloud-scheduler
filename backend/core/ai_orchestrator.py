"""
AI Orchestrator for Advanced AI & ML Features

This module implements the central AI orchestration system that coordinates
all AI/ML systems, manages system-wide learning, and provides intelligent
resource coordination across multiple AI components.

Key Components:
- AIOrchestrator: Central coordinator for all AI systems
- IntelligentResourceCoordinator: Multi-system optimization coordinator
- AdaptiveOptimizationAlgorithm: Continuous improvement algorithm
- ContextualRecommendationEngine: Personalized recommendation system
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Import shared models
from .ai_orchestrator_models import (
    AISystemType, CoordinationStrategy, OptimizationScope,
    OptimizationContext, SystemRecommendation, CoordinatedRecommendation,
    AISystemStatus, CoordinationRequest
)

# Import existing AI/ML systems
from .predictive_scaling_engine import PredictiveScalingEngine, DemandForecast, ScalingRecommendation
from .workload_intelligence_system import WorkloadIntelligenceSystem, WorkloadProfile, PlacementRecommendation
from .reinforcement_learning_agent import ReinforcementLearningAgent, SystemState, OptimizationAction, Experience
from .natural_language_interface import NaturalLanguageInterface, QueryResponse, ConversationContext
from .graph_neural_network_system import GraphNeuralNetworkSystem, ResourceGraph, DependencyAnalysis
from .predictive_maintenance_system import PredictiveMaintenanceSystem, HealthAssessment, MaintenanceRecommendation
from .smart_contract_optimizer import SmartContractOptimizer, OptimizationResult
from .models import BaseModel
from .exceptions import FinOpsException

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """
    Central AI orchestrator that coordinates all AI/ML systems and manages
    system-wide learning and optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AIOrchestrator")
        
        # AI System instances
        self.ai_systems: Dict[AISystemType, Any] = {}
        self.system_status: Dict[AISystemType, AISystemStatus] = {}
        
        # Coordination components
        self.resource_coordinator = IntelligentResourceCoordinator()
        self.optimization_algorithm = AdaptiveOptimizationAlgorithm()
        
        # Import here to avoid circular import
        from .contextual_recommendation_engine import ContextualRecommendationEngine
        self.recommendation_engine = ContextualRecommendationEngine()
        
        # Coordination state
        self.active_coordinations: Dict[str, CoordinationRequest] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.system_interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.learning_metrics: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("AI Orchestrator initialized")
    
    async def initialize_systems(self) -> bool:
        """Initialize all AI/ML systems"""
        try:
            self.logger.info("Initializing AI/ML systems")
            
            # Initialize individual AI systems
            self.ai_systems[AISystemType.PREDICTIVE_SCALING] = PredictiveScalingEngine(None)
            self.ai_systems[AISystemType.WORKLOAD_INTELLIGENCE] = WorkloadIntelligenceSystem()
            self.ai_systems[AISystemType.REINFORCEMENT_LEARNING] = ReinforcementLearningAgent()
            self.ai_systems[AISystemType.NATURAL_LANGUAGE] = NaturalLanguageInterface()
            self.ai_systems[AISystemType.GRAPH_NEURAL_NETWORK] = GraphNeuralNetworkSystem()
            self.ai_systems[AISystemType.PREDICTIVE_MAINTENANCE] = PredictiveMaintenanceSystem()
            self.ai_systems[AISystemType.SMART_CONTRACT] = SmartContractOptimizer()
            
            # Initialize system status tracking
            for system_type in AISystemType:
                self.system_status[system_type] = AISystemStatus(
                    system_type=system_type,
                    is_active=True,
                    last_update=datetime.now(),
                    performance_metrics={},
                    error_count=0,
                    success_rate=1.0,
                    resource_usage={}
                )
            
            # Initialize coordination components
            await self.resource_coordinator.initialize(self.ai_systems)
            await self.optimization_algorithm.initialize()
            await self.recommendation_engine.initialize()
            
            self.logger.info("All AI/ML systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI systems: {str(e)}")
            return False
    
    async def coordinate_optimization(self, context: OptimizationContext) -> CoordinatedRecommendation:
        """
        Coordinate optimization across multiple AI systems.
        
        Args:
            context: Optimization context with goals and constraints
            
        Returns:
            Coordinated recommendation from multiple systems
        """
        self.logger.info(f"Starting coordinated optimization for user {context.user_id}")
        
        try:
            # Determine relevant AI systems for this optimization
            relevant_systems = await self._determine_relevant_systems(context)
            
            # Gather recommendations from each system
            system_recommendations = await self._gather_system_recommendations(
                relevant_systems, context
            )
            
            # Coordinate recommendations using intelligent resource coordinator
            coordinated_recommendation = await self.resource_coordinator.coordinate_recommendations(
                system_recommendations, context
            )
            
            # Apply adaptive optimization algorithm
            optimized_recommendation = await self.optimization_algorithm.optimize_recommendation(
                coordinated_recommendation, context
            )
            
            # Add contextual personalization
            personalized_recommendation = await self.recommendation_engine.personalize_recommendation(
                optimized_recommendation, context
            )
            
            # Update system learning
            await self._update_system_learning(personalized_recommendation, context)
            
            # Track coordination metrics
            await self._track_coordination_metrics(personalized_recommendation)
            
            self.logger.info(f"Coordinated optimization completed with confidence {personalized_recommendation.overall_confidence:.2f}")
            
            return personalized_recommendation
            
        except Exception as e:
            self.logger.error(f"Coordination optimization failed: {str(e)}")
            raise FinOpsException(f"AI coordination failed: {str(e)}")
    
    async def _determine_relevant_systems(self, context: OptimizationContext) -> List[AISystemType]:
        """Determine which AI systems are relevant for the optimization context"""
        relevant_systems = []
        
        # Always include reinforcement learning for decision making
        relevant_systems.append(AISystemType.REINFORCEMENT_LEARNING)
        
        # Include systems based on optimization goals
        for goal in context.optimization_goals:
            if "scaling" in goal.lower() or "capacity" in goal.lower():
                relevant_systems.append(AISystemType.PREDICTIVE_SCALING)
            
            if "placement" in goal.lower() or "migration" in goal.lower():
                relevant_systems.append(AISystemType.WORKLOAD_INTELLIGENCE)
            
            if "dependency" in goal.lower() or "relationship" in goal.lower():
                relevant_systems.append(AISystemType.GRAPH_NEURAL_NETWORK)
            
            if "maintenance" in goal.lower() or "health" in goal.lower():
                relevant_systems.append(AISystemType.PREDICTIVE_MAINTENANCE)
            
            if "commitment" in goal.lower() or "reserved" in goal.lower():
                relevant_systems.append(AISystemType.SMART_CONTRACT)
        
        # Include natural language if user preferences indicate conversational interface
        if context.preferences.get("interface_type") == "conversational":
            relevant_systems.append(AISystemType.NATURAL_LANGUAGE)
        
        return list(set(relevant_systems))  # Remove duplicates
    
    async def _gather_system_recommendations(self, 
                                           systems: List[AISystemType], 
                                           context: OptimizationContext) -> List[SystemRecommendation]:
        """Gather recommendations from multiple AI systems"""
        recommendations = []
        
        # Use ThreadPoolExecutor for parallel system calls
        with ThreadPoolExecutor(max_workers=len(systems)) as executor:
            tasks = []
            
            for system_type in systems:
                if system_type in self.ai_systems:
                    task = executor.submit(
                        asyncio.run,
                        self._get_system_recommendation(system_type, context)
                    )
                    tasks.append((system_type, task))
            
            # Collect results
            for system_type, task in tasks:
                try:
                    recommendation = task.result(timeout=30)  # 30 second timeout
                    if recommendation:
                        recommendations.append(recommendation)
                except Exception as e:
                    self.logger.warning(f"Failed to get recommendation from {system_type.value}: {str(e)}")
                    self._update_system_error(system_type)
        
        return recommendations
    
    async def _get_system_recommendation(self, 
                                       system_type: AISystemType, 
                                       context: OptimizationContext) -> Optional[SystemRecommendation]:
        """Get recommendation from a specific AI system"""
        try:
            system = self.ai_systems.get(system_type)
            if not system:
                return None
            
            recommendation = None
            
            if system_type == AISystemType.PREDICTIVE_SCALING:
                # Mock call to predictive scaling system
                recommendation = SystemRecommendation(
                    system_type=system_type,
                    recommendation_id=str(uuid.uuid4()),
                    resource_id=context.resource_ids[0] if context.resource_ids else "unknown",
                    action_type="scale_up",
                    confidence=0.85,
                    expected_impact={"cost_change": -100.0, "performance_improvement": 0.2},
                    rationale="Predicted demand increase requires scaling",
                    dependencies=[]
                )
            
            elif system_type == AISystemType.WORKLOAD_INTELLIGENCE:
                # Mock call to workload intelligence system
                recommendation = SystemRecommendation(
                    system_type=system_type,
                    recommendation_id=str(uuid.uuid4()),
                    resource_id=context.resource_ids[0] if context.resource_ids else "unknown",
                    action_type="migrate_workload",
                    confidence=0.78,
                    expected_impact={"cost_savings": 200.0, "performance_change": 0.1},
                    rationale="Optimal placement analysis suggests migration",
                    dependencies=[]
                )
            
            elif system_type == AISystemType.REINFORCEMENT_LEARNING:
                # Mock call to RL agent
                recommendation = SystemRecommendation(
                    system_type=system_type,
                    recommendation_id=str(uuid.uuid4()),
                    resource_id=context.resource_ids[0] if context.resource_ids else "unknown",
                    action_type="optimize_configuration",
                    confidence=0.92,
                    expected_impact={"cost_savings": 150.0, "risk_reduction": 0.15},
                    rationale="RL agent learned optimal configuration",
                    dependencies=[]
                )
            
            # Update system status
            self._update_system_success(system_type)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error getting recommendation from {system_type.value}: {str(e)}")
            self._update_system_error(system_type)
            return None
    
    def _update_system_success(self, system_type: AISystemType):
        """Update system status on successful operation"""
        with self._lock:
            if system_type in self.system_status:
                status = self.system_status[system_type]
                status.last_update = datetime.now()
                # Update success rate (exponential moving average)
                status.success_rate = status.success_rate * 0.9 + 0.1
    
    def _update_system_error(self, system_type: AISystemType):
        """Update system status on error"""
        with self._lock:
            if system_type in self.system_status:
                status = self.system_status[system_type]
                status.error_count += 1
                status.last_update = datetime.now()
                # Update success rate (exponential moving average)
                status.success_rate = status.success_rate * 0.9
    
    async def _update_system_learning(self, recommendation: CoordinatedRecommendation, context: OptimizationContext):
        """Update system-wide learning based on recommendation outcomes"""
        try:
            # Record system interactions
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "coordination_id": recommendation.coordination_id,
                "systems_involved": [rec.system_type.value for rec in 
                                   [recommendation.primary_recommendation] + recommendation.supporting_recommendations],
                "context": asdict(context),
                "outcome_confidence": recommendation.overall_confidence
            }
            
            interaction_key = f"{context.user_id}_{context.account_id}"
            self.system_interactions[interaction_key].append(interaction)
            
            # Update learning metrics
            self.learning_metrics["total_coordinations"] = self.learning_metrics.get("total_coordinations", 0) + 1
            self.learning_metrics["average_confidence"] = (
                self.learning_metrics.get("average_confidence", 0.5) * 0.9 + 
                recommendation.overall_confidence * 0.1
            )
            
            # Share learning across systems (simplified)
            await self.optimization_algorithm.update_learning_metrics(self.learning_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to update system learning: {str(e)}")
    
    async def _track_coordination_metrics(self, recommendation: CoordinatedRecommendation):
        """Track metrics for coordination performance"""
        try:
            metrics = {
                "coordination_time": datetime.now(),
                "systems_count": len([recommendation.primary_recommendation] + recommendation.supporting_recommendations),
                "overall_confidence": recommendation.overall_confidence,
                "expected_cost_impact": recommendation.combined_impact.get("cost_savings", 0),
                "coordination_strategy": recommendation.coordination_strategy.value
            }
            
            # Store metrics for analysis
            self.performance_metrics["coordination_confidence"].append(recommendation.overall_confidence)
            self.performance_metrics["systems_involved"].append(metrics["systems_count"])
            
        except Exception as e:
            self.logger.error(f"Failed to track coordination metrics: {str(e)}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all AI systems"""
        health_status = {}
        
        for system_type, status in self.system_status.items():
            health_status[system_type.value] = {
                "is_active": status.is_active,
                "last_update": status.last_update.isoformat(),
                "success_rate": status.success_rate,
                "error_count": status.error_count,
                "performance_metrics": status.performance_metrics
            }
        
        # Add overall orchestrator health
        health_status["orchestrator"] = {
            "total_coordinations": self.learning_metrics.get("total_coordinations", 0),
            "average_confidence": self.learning_metrics.get("average_confidence", 0.0),
            "active_coordinations": len(self.active_coordinations)
        }
        
        return health_status

class IntelligentResourceCoordinator:
    """
    Coordinates resource optimization across multiple AI systems,
    resolving conflicts and optimizing for global objectives.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".IntelligentResourceCoordinator")
        self.ai_systems: Dict[AISystemType, Any] = {}
        self.coordination_strategies: Dict[str, CoordinationStrategy] = {}
        
    async def initialize(self, ai_systems: Dict[AISystemType, Any]):
        """Initialize the resource coordinator with AI systems"""
        self.ai_systems = ai_systems
        self.logger.info("Intelligent Resource Coordinator initialized")
    
    async def coordinate_recommendations(self, 
                                       recommendations: List[SystemRecommendation],
                                       context: OptimizationContext) -> CoordinatedRecommendation:
        """
        Coordinate multiple system recommendations into a unified plan.
        
        Args:
            recommendations: List of recommendations from different AI systems
            context: Optimization context
            
        Returns:
            Coordinated recommendation with implementation plan
        """
        self.logger.info(f"Coordinating {len(recommendations)} recommendations")
        
        if not recommendations:
            raise FinOpsException("No recommendations to coordinate")
        
        # Analyze recommendation conflicts and synergies
        conflicts, synergies = await self._analyze_recommendation_relationships(recommendations)
        
        # Select coordination strategy
        strategy = await self._select_coordination_strategy(recommendations, conflicts, synergies)
        
        # Resolve conflicts
        resolved_recommendations = await self._resolve_conflicts(recommendations, conflicts, strategy)
        
        # Select primary recommendation
        primary_recommendation = await self._select_primary_recommendation(resolved_recommendations, context)
        
        # Identify supporting recommendations
        supporting_recommendations = [rec for rec in resolved_recommendations if rec != primary_recommendation]
        
        # Calculate overall confidence
        overall_confidence = await self._calculate_overall_confidence(
            primary_recommendation, supporting_recommendations
        )
        
        # Calculate combined impact
        combined_impact = await self._calculate_combined_impact(resolved_recommendations)
        
        # Create implementation plan
        implementation_plan = await self._create_implementation_plan(
            resolved_recommendations, strategy
        )
        
        # Assess risks
        risk_assessment = await self._assess_coordination_risks(resolved_recommendations, context)
        
        coordinated_recommendation = CoordinatedRecommendation(
            coordination_id=str(uuid.uuid4()),
            primary_recommendation=primary_recommendation,
            supporting_recommendations=supporting_recommendations,
            coordination_strategy=strategy,
            overall_confidence=overall_confidence,
            combined_impact=combined_impact,
            implementation_plan=implementation_plan,
            risk_assessment=risk_assessment
        )
        
        self.logger.info(f"Coordination completed with strategy {strategy.value}")
        
        return coordinated_recommendation
    
    async def _analyze_recommendation_relationships(self, 
                                                  recommendations: List[SystemRecommendation]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Analyze conflicts and synergies between recommendations"""
        conflicts = []
        synergies = []
        
        for i, rec1 in enumerate(recommendations):
            for j, rec2 in enumerate(recommendations):
                if i >= j:
                    continue
                
                # Check for conflicts (same resource, conflicting actions)
                if (rec1.resource_id == rec2.resource_id and 
                    self._are_actions_conflicting(rec1.action_type, rec2.action_type)):
                    conflicts.append((i, j))
                
                # Check for synergies (complementary actions)
                elif self._are_actions_synergistic(rec1, rec2):
                    synergies.append((i, j))
        
        return conflicts, synergies
    
    def _are_actions_conflicting(self, action1: str, action2: str) -> bool:
        """Check if two actions conflict with each other"""
        conflicting_pairs = [
            ("scale_up", "scale_down"),
            ("migrate_workload", "optimize_configuration"),
            ("increase_capacity", "reduce_capacity")
        ]
        
        for pair in conflicting_pairs:
            if (action1 in pair and action2 in pair) and action1 != action2:
                return True
        
        return False
    
    def _are_actions_synergistic(self, rec1: SystemRecommendation, rec2: SystemRecommendation) -> bool:
        """Check if two recommendations are synergistic"""
        # Actions that work well together
        synergistic_pairs = [
            ("optimize_configuration", "scale_up"),
            ("migrate_workload", "optimize_configuration"),
            ("predictive_maintenance", "optimize_configuration")
        ]
        
        for pair in synergistic_pairs:
            if rec1.action_type in pair and rec2.action_type in pair:
                return True
        
        return False
    
    async def _select_coordination_strategy(self, 
                                          recommendations: List[SystemRecommendation],
                                          conflicts: List[Tuple[int, int]],
                                          synergies: List[Tuple[int, int]]) -> CoordinationStrategy:
        """Select the best coordination strategy"""
        
        # If many conflicts, use consensus strategy
        if len(conflicts) > len(recommendations) * 0.3:
            return CoordinationStrategy.CONSENSUS
        
        # If many synergies, use parallel strategy
        elif len(synergies) > len(recommendations) * 0.5:
            return CoordinationStrategy.PARALLEL
        
        # If recommendations have clear priority differences, use priority-based
        elif self._has_clear_priority_differences(recommendations):
            return CoordinationStrategy.PRIORITY_BASED
        
        # Default to hierarchical
        else:
            return CoordinationStrategy.HIERARCHICAL
    
    def _has_clear_priority_differences(self, recommendations: List[SystemRecommendation]) -> bool:
        """Check if recommendations have clear priority differences"""
        confidences = [rec.confidence for rec in recommendations]
        if not confidences:
            return False
        
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        
        return (max_confidence - min_confidence) > 0.3  # 30% difference threshold
    
    async def _resolve_conflicts(self, 
                               recommendations: List[SystemRecommendation],
                               conflicts: List[Tuple[int, int]],
                               strategy: CoordinationStrategy) -> List[SystemRecommendation]:
        """Resolve conflicts between recommendations"""
        if not conflicts:
            return recommendations
        
        resolved_recommendations = recommendations.copy()
        
        for i, j in conflicts:
            rec1 = recommendations[i]
            rec2 = recommendations[j]
            
            # Resolve based on strategy
            if strategy == CoordinationStrategy.PRIORITY_BASED:
                # Keep the higher confidence recommendation
                if rec1.confidence > rec2.confidence:
                    if rec2 in resolved_recommendations:
                        resolved_recommendations.remove(rec2)
                else:
                    if rec1 in resolved_recommendations:
                        resolved_recommendations.remove(rec1)
            
            elif strategy == CoordinationStrategy.CONSENSUS:
                # Try to find a compromise action
                compromise_rec = await self._create_compromise_recommendation(rec1, rec2)
                if compromise_rec:
                    if rec1 in resolved_recommendations:
                        resolved_recommendations.remove(rec1)
                    if rec2 in resolved_recommendations:
                        resolved_recommendations.remove(rec2)
                    resolved_recommendations.append(compromise_rec)
        
        return resolved_recommendations
    
    async def _create_compromise_recommendation(self, 
                                              rec1: SystemRecommendation, 
                                              rec2: SystemRecommendation) -> Optional[SystemRecommendation]:
        """Create a compromise recommendation from two conflicting ones"""
        try:
            # Simple compromise: average the expected impacts and use lower confidence
            combined_impact = {}
            for key in set(rec1.expected_impact.keys()) | set(rec2.expected_impact.keys()):
                val1 = rec1.expected_impact.get(key, 0)
                val2 = rec2.expected_impact.get(key, 0)
                combined_impact[key] = (val1 + val2) / 2
            
            compromise_rec = SystemRecommendation(
                system_type=rec1.system_type,  # Use first system as primary
                recommendation_id=str(uuid.uuid4()),
                resource_id=rec1.resource_id,
                action_type="optimize_configuration",  # Generic compromise action
                confidence=min(rec1.confidence, rec2.confidence) * 0.8,  # Reduce confidence
                expected_impact=combined_impact,
                rationale=f"Compromise between {rec1.system_type.value} and {rec2.system_type.value}",
                dependencies=list(set(rec1.dependencies + rec2.dependencies))
            )
            
            return compromise_rec
            
        except Exception as e:
            self.logger.error(f"Failed to create compromise recommendation: {str(e)}")
            return None
    
    async def _select_primary_recommendation(self, 
                                           recommendations: List[SystemRecommendation],
                                           context: OptimizationContext) -> SystemRecommendation:
        """Select the primary recommendation from the list"""
        if not recommendations:
            raise FinOpsException("No recommendations to select from")
        
        # Score recommendations based on confidence, impact, and context alignment
        scored_recommendations = []
        
        for rec in recommendations:
            score = rec.confidence * 0.4  # Base confidence score
            
            # Add impact score
            cost_impact = abs(rec.expected_impact.get("cost_savings", 0))
            score += min(cost_impact / 1000, 0.3)  # Normalize and cap at 0.3
            
            # Add context alignment score
            alignment_score = await self._calculate_context_alignment(rec, context)
            score += alignment_score * 0.3
            
            scored_recommendations.append((score, rec))
        
        # Sort by score and return the highest
        scored_recommendations.sort(key=lambda x: x[0], reverse=True)
        
        return scored_recommendations[0][1]
    
    async def _calculate_context_alignment(self, 
                                         recommendation: SystemRecommendation, 
                                         context: OptimizationContext) -> float:
        """Calculate how well a recommendation aligns with the context"""
        alignment_score = 0.0
        
        # Check goal alignment
        for goal in context.optimization_goals:
            if goal.lower() in recommendation.rationale.lower():
                alignment_score += 0.2
        
        # Check constraint compliance
        constraints = context.constraints
        if "max_cost" in constraints:
            cost_impact = recommendation.expected_impact.get("cost_savings", 0)
            if cost_impact <= constraints["max_cost"]:
                alignment_score += 0.1
        
        return min(alignment_score, 1.0)
    
    async def _calculate_overall_confidence(self, 
                                          primary: SystemRecommendation,
                                          supporting: List[SystemRecommendation]) -> float:
        """Calculate overall confidence for coordinated recommendation"""
        # Start with primary recommendation confidence
        overall_confidence = primary.confidence * 0.6
        
        # Add weighted supporting confidences
        if supporting:
            supporting_confidence = sum(rec.confidence for rec in supporting) / len(supporting)
            overall_confidence += supporting_confidence * 0.4
        
        return min(overall_confidence, 1.0)
    
    async def _calculate_combined_impact(self, recommendations: List[SystemRecommendation]) -> Dict[str, float]:
        """Calculate combined impact of all recommendations"""
        combined_impact = defaultdict(float)
        
        for rec in recommendations:
            for key, value in rec.expected_impact.items():
                combined_impact[key] += value
        
        return dict(combined_impact)
    
    async def _create_implementation_plan(self, 
                                        recommendations: List[SystemRecommendation],
                                        strategy: CoordinationStrategy) -> List[Dict[str, Any]]:
        """Create implementation plan for coordinated recommendations"""
        implementation_plan = []
        
        if strategy == CoordinationStrategy.SEQUENTIAL:
            # Order by dependencies and confidence
            ordered_recs = sorted(recommendations, key=lambda x: (len(x.dependencies), -x.confidence))
            
            for i, rec in enumerate(ordered_recs):
                implementation_plan.append({
                    "step": i + 1,
                    "action": rec.action_type,
                    "resource_id": rec.resource_id,
                    "system": rec.system_type.value,
                    "dependencies": rec.dependencies,
                    "estimated_duration": "30 minutes"  # Mock duration
                })
        
        elif strategy == CoordinationStrategy.PARALLEL:
            # Group by resource and execute in parallel
            for i, rec in enumerate(recommendations):
                implementation_plan.append({
                    "step": 1,  # All parallel
                    "action": rec.action_type,
                    "resource_id": rec.resource_id,
                    "system": rec.system_type.value,
                    "parallel_group": i,
                    "estimated_duration": "20 minutes"
                })
        
        else:
            # Default hierarchical approach
            primary_rec = recommendations[0] if recommendations else None
            if primary_rec:
                implementation_plan.append({
                    "step": 1,
                    "action": primary_rec.action_type,
                    "resource_id": primary_rec.resource_id,
                    "system": primary_rec.system_type.value,
                    "priority": "high",
                    "estimated_duration": "25 minutes"
                })
                
                for i, rec in enumerate(recommendations[1:], 2):
                    implementation_plan.append({
                        "step": i,
                        "action": rec.action_type,
                        "resource_id": rec.resource_id,
                        "system": rec.system_type.value,
                        "priority": "medium",
                        "estimated_duration": "15 minutes"
                    })
        
        return implementation_plan
    
    async def _assess_coordination_risks(self, 
                                       recommendations: List[SystemRecommendation],
                                       context: OptimizationContext) -> Dict[str, float]:
        """Assess risks of the coordinated recommendations"""
        risk_assessment = {
            "execution_risk": 0.0,
            "performance_risk": 0.0,
            "cost_risk": 0.0,
            "dependency_risk": 0.0
        }
        
        # Execution risk based on confidence
        confidences = [rec.confidence for rec in recommendations]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            risk_assessment["execution_risk"] = 1.0 - avg_confidence
        
        # Performance risk based on expected impacts
        performance_impacts = [rec.expected_impact.get("performance_change", 0) for rec in recommendations]
        negative_impacts = [impact for impact in performance_impacts if impact < 0]
        if negative_impacts:
            risk_assessment["performance_risk"] = abs(min(negative_impacts)) / 100.0
        
        # Cost risk based on cost changes
        cost_changes = [rec.expected_impact.get("cost_change", 0) for rec in recommendations]
        positive_costs = [cost for cost in cost_changes if cost > 0]
        if positive_costs:
            risk_assessment["cost_risk"] = sum(positive_costs) / 10000.0  # Normalize
        
        # Dependency risk based on number of dependencies
        total_dependencies = sum(len(rec.dependencies) for rec in recommendations)
        risk_assessment["dependency_risk"] = min(total_dependencies / 10.0, 1.0)
        
        return risk_assessment
class AdaptiveOptimizationAlgorithm:
    """
    Adaptive optimization algorithm that continuously improves based on
    historical performance and user feedback.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AdaptiveOptimizationAlgorithm")
        self.learning_history: List[Dict[str, Any]] = []
        self.optimization_patterns: Dict[str, Dict[str, float]] = {}
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    async def initialize(self):
        """Initialize the adaptive optimization algorithm"""
        self.logger.info("Adaptive Optimization Algorithm initialized")
    
    async def optimize_recommendation(self, 
                                    recommendation: CoordinatedRecommendation,
                                    context: OptimizationContext) -> CoordinatedRecommendation:
        """
        Apply adaptive optimization to improve the recommendation.
        
        Args:
            recommendation: Initial coordinated recommendation
            context: Optimization context
            
        Returns:
            Optimized recommendation
        """
        self.logger.info(f"Applying adaptive optimization to recommendation {recommendation.coordination_id}")
        
        # Learn from historical patterns
        historical_adjustments = await self._learn_from_history(recommendation, context)
        
        # Apply pattern-based optimizations
        pattern_optimizations = await self._apply_pattern_optimizations(recommendation, context)
        
        # Adjust confidence based on learning
        adjusted_confidence = await self._adjust_confidence_based_on_learning(
            recommendation, historical_adjustments
        )
        
        # Optimize implementation plan
        optimized_plan = await self._optimize_implementation_plan(
            recommendation.implementation_plan, pattern_optimizations
        )
        
        # Update recommendation with optimizations
        optimized_recommendation = CoordinatedRecommendation(
            coordination_id=recommendation.coordination_id,
            primary_recommendation=recommendation.primary_recommendation,
            supporting_recommendations=recommendation.supporting_recommendations,
            coordination_strategy=recommendation.coordination_strategy,
            overall_confidence=adjusted_confidence,
            combined_impact=recommendation.combined_impact,
            implementation_plan=optimized_plan,
            risk_assessment=recommendation.risk_assessment
        )
        
        # Record learning
        await self._record_optimization_learning(recommendation, optimized_recommendation, context)
        
        self.logger.info(f"Adaptive optimization completed, confidence adjusted to {adjusted_confidence:.2f}")
        
        return optimized_recommendation
    
    async def _learn_from_history(self, 
                                recommendation: CoordinatedRecommendation,
                                context: OptimizationContext) -> Dict[str, float]:
        """Learn adjustments from historical performance"""
        adjustments = {}
        
        # Find similar historical recommendations
        similar_recommendations = await self._find_similar_recommendations(recommendation, context)
        
        if similar_recommendations:
            # Calculate average performance of similar recommendations
            performance_scores = [rec.get("actual_performance", 0.5) for rec in similar_recommendations]
            avg_performance = sum(performance_scores) / len(performance_scores)
            
            # Adjust based on historical performance
            if avg_performance > 0.8:
                adjustments["confidence_boost"] = 0.1
            elif avg_performance < 0.4:
                adjustments["confidence_penalty"] = -0.1
            
            # Learn from implementation success rates
            success_rates = [rec.get("implementation_success", 0.5) for rec in similar_recommendations]
            avg_success = sum(success_rates) / len(success_rates)
            
            adjustments["implementation_adjustment"] = (avg_success - 0.5) * 0.2
        
        return adjustments
    
    async def _find_similar_recommendations(self, 
                                          recommendation: CoordinatedRecommendation,
                                          context: OptimizationContext) -> List[Dict[str, Any]]:
        """Find similar historical recommendations for learning"""
        similar_recommendations = []
        
        for historical_rec in self.learning_history:
            similarity_score = await self._calculate_recommendation_similarity(
                recommendation, historical_rec, context
            )
            
            if similarity_score > 0.7:  # 70% similarity threshold
                similar_recommendations.append(historical_rec)
        
        return similar_recommendations[-10:]  # Return last 10 similar recommendations
    
    async def _calculate_recommendation_similarity(self, 
                                                 current: CoordinatedRecommendation,
                                                 historical: Dict[str, Any],
                                                 context: OptimizationContext) -> float:
        """Calculate similarity between current and historical recommendations"""
        similarity_score = 0.0
        
        # Compare action types
        current_actions = set([current.primary_recommendation.action_type] + 
                            [rec.action_type for rec in current.supporting_recommendations])
        historical_actions = set(historical.get("action_types", []))
        
        if current_actions & historical_actions:  # Intersection
            similarity_score += 0.3
        
        # Compare resource types
        current_resources = set([current.primary_recommendation.resource_id] + 
                              [rec.resource_id for rec in current.supporting_recommendations])
        historical_resources = set(historical.get("resource_ids", []))
        
        if current_resources & historical_resources:
            similarity_score += 0.2
        
        # Compare context goals
        current_goals = set(context.optimization_goals)
        historical_goals = set(historical.get("optimization_goals", []))
        
        goal_overlap = len(current_goals & historical_goals) / max(len(current_goals), 1)
        similarity_score += goal_overlap * 0.3
        
        # Compare coordination strategy
        if current.coordination_strategy.value == historical.get("coordination_strategy"):
            similarity_score += 0.2
        
        return similarity_score
    
    async def _apply_pattern_optimizations(self, 
                                         recommendation: CoordinatedRecommendation,
                                         context: OptimizationContext) -> Dict[str, Any]:
        """Apply learned pattern optimizations"""
        optimizations = {}
        
        # Check for known successful patterns
        pattern_key = f"{recommendation.coordination_strategy.value}_{len(recommendation.supporting_recommendations)}"
        
        if pattern_key in self.optimization_patterns:
            pattern_data = self.optimization_patterns[pattern_key]
            
            # Apply successful timing adjustments
            if "optimal_timing" in pattern_data:
                optimizations["timing_adjustment"] = pattern_data["optimal_timing"]
            
            # Apply resource allocation optimizations
            if "resource_allocation" in pattern_data:
                optimizations["resource_optimization"] = pattern_data["resource_allocation"]
        
        return optimizations
    
    async def _adjust_confidence_based_on_learning(self, 
                                                 recommendation: CoordinatedRecommendation,
                                                 adjustments: Dict[str, float]) -> float:
        """Adjust confidence based on learning adjustments"""
        adjusted_confidence = recommendation.overall_confidence
        
        # Apply confidence adjustments
        confidence_boost = adjustments.get("confidence_boost", 0)
        confidence_penalty = adjustments.get("confidence_penalty", 0)
        
        adjusted_confidence += confidence_boost + confidence_penalty
        
        # Apply implementation success adjustment
        impl_adjustment = adjustments.get("implementation_adjustment", 0)
        adjusted_confidence += impl_adjustment
        
        # Ensure confidence stays within bounds
        return max(0.1, min(1.0, adjusted_confidence))
    
    async def _optimize_implementation_plan(self, 
                                          original_plan: List[Dict[str, Any]],
                                          optimizations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize the implementation plan based on learned patterns"""
        optimized_plan = original_plan.copy()
        
        # Apply timing optimizations
        if "timing_adjustment" in optimizations:
            timing_factor = optimizations["timing_adjustment"]
            for step in optimized_plan:
                if "estimated_duration" in step:
                    # Parse duration and adjust (simplified)
                    duration_str = step["estimated_duration"]
                    if "minutes" in duration_str:
                        minutes = int(duration_str.split()[0])
                        adjusted_minutes = int(minutes * timing_factor)
                        step["estimated_duration"] = f"{adjusted_minutes} minutes"
        
        # Apply resource optimizations
        if "resource_optimization" in optimizations:
            # Add resource optimization hints to the plan
            for step in optimized_plan:
                step["optimization_hints"] = optimizations["resource_optimization"]
        
        return optimized_plan
    
    async def _record_optimization_learning(self, 
                                          original: CoordinatedRecommendation,
                                          optimized: CoordinatedRecommendation,
                                          context: OptimizationContext):
        """Record learning from optimization process"""
        learning_record = {
            "timestamp": datetime.now().isoformat(),
            "original_confidence": original.overall_confidence,
            "optimized_confidence": optimized.overall_confidence,
            "action_types": [original.primary_recommendation.action_type] + 
                          [rec.action_type for rec in original.supporting_recommendations],
            "resource_ids": [original.primary_recommendation.resource_id] + 
                          [rec.resource_id for rec in original.supporting_recommendations],
            "coordination_strategy": original.coordination_strategy.value,
            "optimization_goals": context.optimization_goals,
            "user_id": context.user_id,
            "account_id": context.account_id
        }
        
        self.learning_history.append(learning_record)
        
        # Update performance metrics
        confidence_improvement = optimized.overall_confidence - original.overall_confidence
        self.performance_metrics["confidence_improvements"].append(confidence_improvement)
    
    async def update_learning_metrics(self, metrics: Dict[str, float]):
        """Update learning metrics from external sources"""
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)