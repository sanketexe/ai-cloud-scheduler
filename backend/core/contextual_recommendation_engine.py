"""
Contextual Recommendation Engine

This module provides personalized recommendations based on user context,
preferences, and historical feedback.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .ai_orchestrator_models import CoordinatedRecommendation, OptimizationContext

logger = logging.getLogger(__name__)


class ContextualRecommendationEngine:
    """
    Provides personalized recommendations based on user context,
    preferences, and historical feedback.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ContextualRecommendationEngine")
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.recommendation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.preference_patterns: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize the contextual recommendation engine"""
        self.logger.info("Contextual Recommendation Engine initialized")
    
    async def personalize_recommendation(self, 
                                       recommendation: CoordinatedRecommendation,
                                       context: OptimizationContext) -> CoordinatedRecommendation:
        """
        Personalize recommendation based on user context and preferences.
        
        Args:
            recommendation: Base coordinated recommendation
            context: Optimization context with user preferences
            
        Returns:
            Personalized recommendation
        """
        self.logger.info(f"Personalizing recommendation for user {context.user_id}")
        
        # Get or create user profile
        user_profile = await self._get_user_profile(context.user_id)
        
        # Analyze user preferences
        preference_adjustments = await self._analyze_user_preferences(
            recommendation, context, user_profile
        )
        
        # Apply contextual adjustments
        contextual_adjustments = await self._apply_contextual_adjustments(
            recommendation, context, user_profile
        )
        
        # Adjust recommendation based on historical feedback
        feedback_adjustments = await self._apply_feedback_adjustments(
            recommendation, context, user_profile
        )
        
        # Create personalized recommendation
        personalized_recommendation = await self._create_personalized_recommendation(
            recommendation, preference_adjustments, contextual_adjustments, feedback_adjustments
        )
        
        # Update user profile with new interaction
        await self._update_user_profile(context.user_id, personalized_recommendation, context)
        
        self.logger.info(f"Recommendation personalized with confidence {personalized_recommendation.overall_confidence:.2f}")
        
        return personalized_recommendation
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "preferences": {
                    "risk_tolerance": 0.5,  # 0 = risk averse, 1 = risk seeking
                    "cost_priority": 0.7,   # 0 = performance first, 1 = cost first
                    "automation_level": 0.6, # 0 = manual approval, 1 = full automation
                    "notification_frequency": 0.5  # 0 = minimal, 1 = frequent
                },
                "interaction_history": [],
                "feedback_patterns": {},
                "optimization_goals": [],
                "constraint_patterns": {}
            }
        
        return self.user_profiles[user_id]
    
    async def _analyze_user_preferences(self, 
                                      recommendation: CoordinatedRecommendation,
                                      context: OptimizationContext,
                                      user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Analyze user preferences and calculate adjustments"""
        adjustments = {}
        preferences = user_profile.get("preferences", {})
        
        # Risk tolerance adjustments
        risk_tolerance = preferences.get("risk_tolerance", 0.5)
        overall_risk = sum(recommendation.risk_assessment.values()) / len(recommendation.risk_assessment)
        
        if risk_tolerance < 0.3 and overall_risk > 0.6:  # Risk averse user, high risk recommendation
            adjustments["confidence_penalty"] = -0.2
        elif risk_tolerance > 0.7 and overall_risk < 0.3:  # Risk seeking user, low risk recommendation
            adjustments["confidence_boost"] = 0.1
        
        # Cost priority adjustments
        cost_priority = preferences.get("cost_priority", 0.7)
        cost_savings = recommendation.combined_impact.get("cost_savings", 0)
        
        if cost_priority > 0.8 and cost_savings > 0:  # High cost priority, positive savings
            adjustments["cost_preference_boost"] = 0.15
        elif cost_priority < 0.3 and cost_savings < 0:  # Low cost priority, cost increase
            adjustments["performance_preference_boost"] = 0.1
        
        # Automation level adjustments
        automation_level = preferences.get("automation_level", 0.6)
        if automation_level < 0.4:  # Prefers manual approval
            adjustments["manual_approval_required"] = True
        
        return adjustments
    
    async def _apply_contextual_adjustments(self, 
                                          recommendation: CoordinatedRecommendation,
                                          context: OptimizationContext,
                                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contextual adjustments based on current situation"""
        adjustments = {}
        
        # Time-based adjustments
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:  # Night time
            adjustments["timing_preference"] = "delay_to_business_hours"
        
        # Account-specific adjustments
        account_history = user_profile.get("interaction_history", [])
        account_specific_history = [
            interaction for interaction in account_history 
            if interaction.get("account_id") == context.account_id
        ]
        
        if len(account_specific_history) > 10:
            # Calculate success rate for this account
            successful_interactions = [
                interaction for interaction in account_specific_history
                if interaction.get("outcome_rating", 0) > 0.7
            ]
            success_rate = len(successful_interactions) / len(account_specific_history)
            
            if success_rate > 0.8:
                adjustments["account_confidence_boost"] = 0.1
            elif success_rate < 0.4:
                adjustments["account_confidence_penalty"] = -0.1
        
        # Resource-specific adjustments
        resource_patterns = user_profile.get("constraint_patterns", {})
        for resource_id in [recommendation.primary_recommendation.resource_id]:
            if resource_id in resource_patterns:
                pattern = resource_patterns[resource_id]
                if pattern.get("frequent_changes", False):
                    adjustments["resource_stability_concern"] = True
        
        return adjustments
    
    async def _apply_feedback_adjustments(self, 
                                        recommendation: CoordinatedRecommendation,
                                        context: OptimizationContext,
                                        user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Apply adjustments based on historical feedback"""
        adjustments = {}
        feedback_patterns = user_profile.get("feedback_patterns", {})
        
        # Analyze feedback for similar recommendation types
        primary_action = recommendation.primary_recommendation.action_type
        if primary_action in feedback_patterns:
            pattern = feedback_patterns[primary_action]
            avg_rating = pattern.get("average_rating", 0.5)
            
            if avg_rating > 0.8:
                adjustments["positive_feedback_boost"] = 0.15
            elif avg_rating < 0.3:
                adjustments["negative_feedback_penalty"] = -0.2
        
        # Analyze feedback for coordination strategies
        strategy = recommendation.coordination_strategy.value
        strategy_key = f"strategy_{strategy}"
        if strategy_key in feedback_patterns:
            pattern = feedback_patterns[strategy_key]
            avg_rating = pattern.get("average_rating", 0.5)
            
            if avg_rating > 0.7:
                adjustments["strategy_preference_boost"] = 0.1
            elif avg_rating < 0.4:
                adjustments["strategy_preference_penalty"] = -0.1
        
        return adjustments
    
    async def _create_personalized_recommendation(self, 
                                                original: CoordinatedRecommendation,
                                                preference_adjustments: Dict[str, float],
                                                contextual_adjustments: Dict[str, Any],
                                                feedback_adjustments: Dict[str, float]) -> CoordinatedRecommendation:
        """Create personalized recommendation with all adjustments applied"""
        
        # Calculate adjusted confidence
        adjusted_confidence = original.overall_confidence
        
        # Apply preference adjustments
        adjusted_confidence += preference_adjustments.get("confidence_penalty", 0)
        adjusted_confidence += preference_adjustments.get("confidence_boost", 0)
        adjusted_confidence += preference_adjustments.get("cost_preference_boost", 0)
        adjusted_confidence += preference_adjustments.get("performance_preference_boost", 0)
        
        # Apply contextual adjustments
        adjusted_confidence += contextual_adjustments.get("account_confidence_boost", 0)
        adjusted_confidence += contextual_adjustments.get("account_confidence_penalty", 0)
        
        # Apply feedback adjustments
        adjusted_confidence += feedback_adjustments.get("positive_feedback_boost", 0)
        adjusted_confidence += feedback_adjustments.get("negative_feedback_penalty", 0)
        adjusted_confidence += feedback_adjustments.get("strategy_preference_boost", 0)
        adjusted_confidence += feedback_adjustments.get("strategy_preference_penalty", 0)
        
        # Ensure confidence stays within bounds
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
        
        # Modify implementation plan based on contextual adjustments
        modified_plan = original.implementation_plan.copy()
        
        # Add timing preferences
        if contextual_adjustments.get("timing_preference") == "delay_to_business_hours":
            for step in modified_plan:
                step["scheduling_preference"] = "business_hours_only"
        
        # Add approval requirements
        if preference_adjustments.get("manual_approval_required"):
            for step in modified_plan:
                step["requires_manual_approval"] = True
        
        # Add resource stability concerns
        if contextual_adjustments.get("resource_stability_concern"):
            for step in modified_plan:
                step["stability_check_required"] = True
        
        # Create personalized recommendation
        personalized_recommendation = CoordinatedRecommendation(
            coordination_id=original.coordination_id,
            primary_recommendation=original.primary_recommendation,
            supporting_recommendations=original.supporting_recommendations,
            coordination_strategy=original.coordination_strategy,
            overall_confidence=adjusted_confidence,
            combined_impact=original.combined_impact,
            implementation_plan=modified_plan,
            risk_assessment=original.risk_assessment
        )
        
        return personalized_recommendation
    
    async def _update_user_profile(self, 
                                 user_id: str, 
                                 recommendation: CoordinatedRecommendation,
                                 context: OptimizationContext):
        """Update user profile with new interaction"""
        if user_id not in self.user_profiles:
            return
        
        user_profile = self.user_profiles[user_id]
        
        # Add to interaction history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "coordination_id": recommendation.coordination_id,
            "account_id": context.account_id,
            "optimization_goals": context.optimization_goals,
            "primary_action": recommendation.primary_recommendation.action_type,
            "coordination_strategy": recommendation.coordination_strategy.value,
            "confidence": recommendation.overall_confidence,
            "expected_impact": recommendation.combined_impact
        }
        
        user_profile["interaction_history"].append(interaction)
        
        # Keep only last 100 interactions
        if len(user_profile["interaction_history"]) > 100:
            user_profile["interaction_history"] = user_profile["interaction_history"][-100:]
        
        # Update optimization goals patterns
        for goal in context.optimization_goals:
            if goal not in user_profile["optimization_goals"]:
                user_profile["optimization_goals"].append(goal)
        
        # Update constraint patterns
        for resource_id in [recommendation.primary_recommendation.resource_id]:
            if resource_id not in user_profile["constraint_patterns"]:
                user_profile["constraint_patterns"][resource_id] = {
                    "interaction_count": 0,
                    "frequent_changes": False
                }
            
            user_profile["constraint_patterns"][resource_id]["interaction_count"] += 1
            
            # Mark as frequently changed if more than 5 interactions
            if user_profile["constraint_patterns"][resource_id]["interaction_count"] > 5:
                user_profile["constraint_patterns"][resource_id]["frequent_changes"] = True
    
    async def record_feedback(self, 
                            user_id: str, 
                            coordination_id: str, 
                            rating: float, 
                            feedback_text: str = ""):
        """Record user feedback for a recommendation"""
        if user_id not in self.user_profiles:
            return
        
        user_profile = self.user_profiles[user_id]
        
        # Find the interaction
        interaction = None
        for hist_interaction in user_profile["interaction_history"]:
            if hist_interaction.get("coordination_id") == coordination_id:
                interaction = hist_interaction
                break
        
        if not interaction:
            return
        
        # Update interaction with feedback
        interaction["outcome_rating"] = rating
        interaction["feedback_text"] = feedback_text
        interaction["feedback_timestamp"] = datetime.now().isoformat()
        
        # Update feedback patterns
        primary_action = interaction.get("primary_action")
        if primary_action:
            if primary_action not in user_profile["feedback_patterns"]:
                user_profile["feedback_patterns"][primary_action] = {
                    "ratings": [],
                    "average_rating": 0.0
                }
            
            pattern = user_profile["feedback_patterns"][primary_action]
            pattern["ratings"].append(rating)
            
            # Keep only last 20 ratings
            if len(pattern["ratings"]) > 20:
                pattern["ratings"] = pattern["ratings"][-20:]
            
            # Update average
            pattern["average_rating"] = sum(pattern["ratings"]) / len(pattern["ratings"])
        
        # Update strategy feedback patterns
        strategy = interaction.get("coordination_strategy")
        if strategy:
            strategy_key = f"strategy_{strategy}"
            if strategy_key not in user_profile["feedback_patterns"]:
                user_profile["feedback_patterns"][strategy_key] = {
                    "ratings": [],
                    "average_rating": 0.0
                }
            
            pattern = user_profile["feedback_patterns"][strategy_key]
            pattern["ratings"].append(rating)
            
            # Keep only last 20 ratings
            if len(pattern["ratings"]) > 20:
                pattern["ratings"] = pattern["ratings"][-20:]
            
            # Update average
            pattern["average_rating"] = sum(pattern["ratings"]) / len(pattern["ratings"])
        
        self.logger.info(f"Recorded feedback for user {user_id}, coordination {coordination_id}: {rating}")
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences and patterns"""
        if user_id not in self.user_profiles:
            return {}
        
        user_profile = self.user_profiles[user_id]
        
        insights = {
            "user_id": user_id,
            "total_interactions": len(user_profile["interaction_history"]),
            "preferences": user_profile["preferences"],
            "common_optimization_goals": user_profile["optimization_goals"][-10:],  # Last 10 goals
            "feedback_summary": {},
            "account_performance": {}
        }
        
        # Analyze feedback patterns
        for pattern_key, pattern_data in user_profile["feedback_patterns"].items():
            insights["feedback_summary"][pattern_key] = {
                "average_rating": pattern_data["average_rating"],
                "total_feedback": len(pattern_data["ratings"])
            }
        
        # Analyze account performance
        account_performance = defaultdict(list)
        for interaction in user_profile["interaction_history"]:
            if "outcome_rating" in interaction:
                account_id = interaction.get("account_id", "unknown")
                account_performance[account_id].append(interaction["outcome_rating"])
        
        for account_id, ratings in account_performance.items():
            insights["account_performance"][account_id] = {
                "average_rating": sum(ratings) / len(ratings),
                "total_interactions": len(ratings)
            }
        
        return insights