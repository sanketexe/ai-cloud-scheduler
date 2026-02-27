"""
Reinforcement Learning Optimization Service

This service integrates the reinforcement learning agent with the existing
FinOps platform, providing a high-level interface for cost optimization
using reinforcement learning techniques.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .reinforcement_learning_agent import (
    ReinforcementLearningAgent, SystemState, OptimizationAction, 
    ActionOutcome, Experience, ActionType, RiskLevel
)
from .rl_training_pipeline import ContinuousLearningPipeline, TrainingConfig
from .database import get_db_session
from .models import (
    CostData, ResourceMetrics, OptimizationRecommendation, 
    AuditLog, ScalingEvent
)
from .cloud_providers import CloudProvider
from .safety_checker import SafetyChecker

logger = logging.getLogger(__name__)

class RLOptimizationService:
    """Main service for reinforcement learning-based cost optimization"""
    
    def __init__(self, safety_checker: SafetyChecker):
        self.safety_checker = safety_checker
        self.logger = logging.getLogger(__name__ + ".RLOptimizationService")
        
        # Initialize RL agent
        self.agent = ReinforcementLearningAgent()
        
        # Initialize training pipeline
        training_config = TrainingConfig(
            training_interval_hours=6,  # Train every 6 hours
            min_experiences_for_training=50,
            validation_split=0.2
        )
        self.training_pipeline = ContinuousLearningPipeline(training_config)
        
        # State tracking
        self.is_initialized = False
        self.active_optimizations: Dict[str, Dict] = {}
        self.optimization_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_cost_savings': 0.0,
            'average_confidence': 0.0
        }
    
    async def initialize(self):
        """Initialize the RL optimization service"""
        
        try:
            self.logger.info("Initializing RL Optimization Service")
            
            # Load existing model if available
            await self._load_existing_model()
            
            # Start training pipeline
            asyncio.create_task(self.training_pipeline.start_pipeline(self.agent))
            
            self.is_initialized = True
            self.logger.info("RL Optimization Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RL Optimization Service: {str(e)}")
            raise
    
    async def optimize_resource(self, resource_id: str, 
                              provider: CloudProvider) -> OptimizationRecommendation:
        """Generate optimization recommendation for a specific resource"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            self.logger.info(f"Generating optimization recommendation for resource {resource_id}")
            
            # Gather current system state
            current_state = await self._gather_system_state(resource_id)
            
            # Get optimization action from RL agent
            action = await self.agent.select_action(current_state)
            
            # Convert to optimization recommendation
            recommendation = await self._create_optimization_recommendation(
                action, current_state, provider
            )
            
            # Track active optimization
            self.active_optimizations[resource_id] = {
                'recommendation': recommendation,
                'action': action,
                'state': current_state,
                'timestamp': datetime.now(),
                'provider': provider
            }
            
            self.logger.info(f"Generated recommendation for {resource_id}: "
                           f"{action.action_type.value} with confidence {action.confidence:.3f}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error optimizing resource {resource_id}: {str(e)}")
            raise
    
    async def execute_optimization(self, resource_id: str, 
                                 recommendation_id: str) -> Dict[str, Any]:
        """Execute an optimization recommendation"""
        
        if resource_id not in self.active_optimizations:
            raise ValueError(f"No active optimization found for resource {resource_id}")
        
        optimization = self.active_optimizations[resource_id]
        action = optimization['action']
        provider = optimization['provider']
        
        try:
            self.logger.info(f"Executing optimization for resource {resource_id}: "
                           f"{action.action_type.value}")
            
            # Execute the action through appropriate service
            execution_result = await self._execute_action(action, provider)
            
            # Create action outcome
            outcome = ActionOutcome(
                action_id=action.action_id,
                success=execution_result['success'],
                actual_cost_savings=execution_result.get('cost_savings', 0.0),
                actual_performance_change=execution_result.get('performance_change', 0.0),
                actual_availability_change=execution_result.get('availability_change', 0.0),
                execution_time=datetime.now(),
                side_effects=execution_result.get('side_effects', []),
                user_feedback=None  # Will be updated later if provided
            )
            
            # Calculate reward and update agent
            reward = await self.agent.evaluate_action(action, outcome)
            
            # Create experience for learning
            # Note: next_state would be gathered after action execution in practice
            next_state = await self._gather_system_state(resource_id)
            
            experience = Experience(
                state=optimization['state'],
                action=action,
                reward=reward,
                next_state=next_state,
                done=True,  # Single-step optimization
                timestamp=datetime.now()
            )
            
            # Update agent policy
            await self.agent.update_policy(experience)
            
            # Update metrics
            self._update_metrics(outcome)
            
            # Log to audit trail
            await self._log_optimization_execution(resource_id, action, outcome)
            
            # Clean up active optimization
            del self.active_optimizations[resource_id]
            
            # Add to history
            self.optimization_history.append({
                'resource_id': resource_id,
                'action': action,
                'outcome': outcome,
                'reward': reward,
                'timestamp': datetime.now()
            })
            
            result = {
                'success': outcome.success,
                'action_type': action.action_type.value,
                'cost_savings': outcome.actual_cost_savings,
                'performance_change': outcome.actual_performance_change,
                'reward': reward,
                'execution_time': outcome.execution_time.isoformat()
            }
            
            if not outcome.success:
                result['error_message'] = execution_result.get('error_message', 'Unknown error')
            
            self.logger.info(f"Optimization execution completed for {resource_id}: "
                           f"success={outcome.success}, reward={reward:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing optimization for {resource_id}: {str(e)}")
            
            # Create failed outcome for learning
            failed_outcome = ActionOutcome(
                action_id=action.action_id,
                success=False,
                actual_cost_savings=0.0,
                actual_performance_change=0.0,
                actual_availability_change=0.0,
                execution_time=datetime.now(),
                side_effects=[f"Execution error: {str(e)}"],
                user_feedback=None
            )
            
            # Still learn from failures
            reward = await self.agent.evaluate_action(action, failed_outcome)
            
            raise
    
    async def provide_user_feedback(self, resource_id: str, action_id: str, 
                                  feedback_score: float, comments: str = ""):
        """Provide user feedback on an optimization action"""
        
        try:
            # Find the optimization in history
            for opt in self.optimization_history:
                if (opt['resource_id'] == resource_id and 
                    opt['action'].action_id == action_id):
                    
                    # Update outcome with user feedback
                    opt['outcome'].user_feedback = feedback_score
                    
                    # Re-evaluate action with user feedback
                    new_reward = await self.agent.evaluate_action(opt['action'], opt['outcome'])
                    
                    # Create new experience with updated reward
                    experience = Experience(
                        state=opt.get('state'),  # Would need to store state in history
                        action=opt['action'],
                        reward=new_reward,
                        next_state=opt.get('next_state'),  # Would need to store next_state
                        done=True,
                        timestamp=datetime.now()
                    )
                    
                    # Update agent with feedback
                    await self.agent.update_policy(experience)
                    
                    self.logger.info(f"User feedback provided for {resource_id}: "
                                   f"score={feedback_score}, new_reward={new_reward:.3f}")
                    
                    return
            
            self.logger.warning(f"Optimization not found for feedback: {resource_id}, {action_id}")
            
        except Exception as e:
            self.logger.error(f"Error providing user feedback: {str(e)}")
    
    async def get_optimization_recommendations(self, account_id: str, 
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Get optimization recommendations for all resources in an account"""
        
        try:
            recommendations = []
            
            # Get resources that need optimization
            resources = await self._get_optimization_candidates(account_id, limit)
            
            for resource in resources:
                try:
                    # Generate recommendation for each resource
                    state = await self._gather_system_state(resource['resource_id'])
                    action = await self.agent.select_action(state)
                    
                    # Get confidence score
                    confidence = await self.agent.get_action_confidence(state, action)
                    
                    recommendation = {
                        'resource_id': resource['resource_id'],
                        'resource_type': resource['resource_type'],
                        'action_type': action.action_type.value,
                        'expected_cost_savings': action.expected_impact.get('cost_savings', 0.0),
                        'expected_performance_change': action.expected_impact.get('performance_change', 0.0),
                        'risk_level': action.risk_level.value,
                        'confidence': confidence,
                        'reasoning': self._generate_reasoning(action, state),
                        'parameters': action.parameters
                    }
                    
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    self.logger.error(f"Error generating recommendation for {resource['resource_id']}: {str(e)}")
                    continue
            
            # Sort by expected cost savings and confidence
            recommendations.sort(
                key=lambda x: (x['expected_cost_savings'] * x['confidence']), 
                reverse=True
            )
            
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {str(e)}")
            return []
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        
        agent_metrics = self.agent.get_learning_metrics()
        pipeline_status = self.training_pipeline.get_pipeline_status()
        
        return {
            'service_metrics': self.metrics,
            'agent_metrics': agent_metrics,
            'training_pipeline': pipeline_status,
            'active_optimizations': len(self.active_optimizations),
            'optimization_history_length': len(self.optimization_history),
            'is_initialized': self.is_initialized
        }
    
    async def _gather_system_state(self, resource_id: str) -> SystemState:
        """Gather current system state for a resource"""
        
        try:
            async with get_db_session() as session:
                # Get recent resource metrics
                metrics_query = """
                    SELECT * FROM resource_metrics 
                    WHERE resource_id = :resource_id 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """
                
                result = await session.execute(metrics_query, {'resource_id': resource_id})
                metrics_data = result.fetchall()
                
                # Get recent cost data
                cost_query = """
                    SELECT * FROM cost_data 
                    WHERE resource_id = :resource_id 
                    ORDER BY cost_date DESC 
                    LIMIT 7
                """
                
                result = await session.execute(cost_query, {'resource_id': resource_id})
                cost_data = result.fetchall()
                
                # Calculate aggregated metrics
                resource_utilization = self._calculate_resource_utilization(metrics_data)
                cost_metrics = self._calculate_cost_metrics(cost_data)
                performance_metrics = self._calculate_performance_metrics(metrics_data)
                external_factors = self._calculate_external_factors()
                budget_status = await self._calculate_budget_status(resource_id)
                
                return SystemState(
                    timestamp=datetime.now(),
                    resource_utilization=resource_utilization,
                    cost_metrics=cost_metrics,
                    performance_metrics=performance_metrics,
                    external_factors=external_factors,
                    budget_status=budget_status
                )
                
        except Exception as e:
            self.logger.error(f"Error gathering system state: {str(e)}")
            # Return default state
            return self._create_default_state()
    
    def _calculate_resource_utilization(self, metrics_data: List) -> Dict[str, float]:
        """Calculate resource utilization metrics"""
        
        if not metrics_data:
            return {
                'cpu_avg': 0.5,
                'memory_avg': 0.5,
                'network_in': 0.0,
                'network_out': 0.0,
                'disk_io': 0.0
            }
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame([dict(row) for row in metrics_data])
        
        return {
            'cpu_avg': df['cpu_utilization'].mean() / 100.0 if 'cpu_utilization' in df else 0.5,
            'memory_avg': df['memory_utilization'].mean() / 100.0 if 'memory_utilization' in df else 0.5,
            'network_in': df['network_in'].mean() if 'network_in' in df else 0.0,
            'network_out': df['network_out'].mean() if 'network_out' in df else 0.0,
            'disk_io': (df['disk_read'].mean() + df['disk_write'].mean()) if 'disk_read' in df else 0.0
        }
    
    def _calculate_cost_metrics(self, cost_data: List) -> Dict[str, float]:
        """Calculate cost-related metrics"""
        
        if not cost_data:
            return {
                'current_hourly_cost': 10.0,
                'cost_trend_24h': 0.0,
                'cost_variance': 0.0,
                'cost_per_request': 0.001
            }
        
        df = pd.DataFrame([dict(row) for row in cost_data])
        
        # Calculate daily costs
        daily_costs = df['cost_amount'].values
        
        current_cost = daily_costs[0] if len(daily_costs) > 0 else 10.0
        
        # Calculate trend (simple linear regression slope)
        if len(daily_costs) > 1:
            x = np.arange(len(daily_costs))
            trend = np.polyfit(x, daily_costs, 1)[0]
        else:
            trend = 0.0
        
        return {
            'current_hourly_cost': float(current_cost / 24),  # Convert daily to hourly
            'cost_trend_24h': float(trend),
            'cost_variance': float(np.var(daily_costs)) if len(daily_costs) > 1 else 0.0,
            'cost_per_request': float(current_cost / 1000)  # Estimate
        }
    
    def _calculate_performance_metrics(self, metrics_data: List) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        if not metrics_data:
            return {
                'avg_response_time': 500.0,
                'error_rate': 0.01,
                'throughput': 100.0,
                'availability': 0.99
            }
        
        df = pd.DataFrame([dict(row) for row in metrics_data])
        
        return {
            'avg_response_time': df['response_time'].mean() if 'response_time' in df else 500.0,
            'error_rate': df['error_rate'].mean() / 100.0 if 'error_rate' in df else 0.01,
            'throughput': df['request_count'].mean() if 'request_count' in df else 100.0,
            'availability': 0.99  # Would calculate from uptime data
        }
    
    def _calculate_external_factors(self) -> Dict[str, Any]:
        """Calculate external factors affecting optimization"""
        
        now = datetime.now()
        
        return {
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            'seasonal_factor': 1.0 + 0.1 * np.sin(2 * np.pi * now.timetuple().tm_yday / 365)
        }
    
    async def _calculate_budget_status(self, resource_id: str) -> Dict[str, float]:
        """Calculate budget-related status"""
        
        # This would integrate with budget management system
        # For now, return placeholder values
        
        return {
            'utilization_percent': 65.0,
            'remaining_days': 15.0,
            'projected_overrun': 0.1
        }
    
    def _create_default_state(self) -> SystemState:
        """Create a default system state"""
        
        return SystemState(
            timestamp=datetime.now(),
            resource_utilization={'cpu_avg': 0.5, 'memory_avg': 0.5, 'network_in': 0.0, 'network_out': 0.0, 'disk_io': 0.0},
            cost_metrics={'current_hourly_cost': 10.0, 'cost_trend_24h': 0.0, 'cost_variance': 0.0, 'cost_per_request': 0.001},
            performance_metrics={'avg_response_time': 500.0, 'error_rate': 0.01, 'throughput': 100.0, 'availability': 0.99},
            external_factors={'hour_of_day': 12, 'day_of_week': 2, 'is_weekend': 0.0, 'seasonal_factor': 1.0},
            budget_status={'utilization_percent': 50.0, 'remaining_days': 20.0, 'projected_overrun': 0.0}
        )
    
    async def _create_optimization_recommendation(self, action: OptimizationAction, 
                                               state: SystemState, 
                                               provider: CloudProvider) -> OptimizationRecommendation:
        """Create an optimization recommendation from an RL action"""
        
        # This would be stored in the database
        recommendation = OptimizationRecommendation(
            provider_id=provider.id if hasattr(provider, 'id') else None,
            resource_id=action.resource_id,
            resource_type="compute",  # Would determine from resource
            recommendation_type=self._map_action_to_recommendation_type(action.action_type),
            current_cost=state.cost_metrics.get('current_hourly_cost', 0.0) * 24,
            optimized_cost=(state.cost_metrics.get('current_hourly_cost', 0.0) * 24) - action.expected_impact.get('cost_savings', 0.0),
            potential_savings=action.expected_impact.get('cost_savings', 0.0),
            confidence_score=action.confidence,
            risk_level=action.risk_level,
            recommendation_text=self._generate_recommendation_text(action),
            implementation_details=action.parameters,
            valid_until=datetime.now() + timedelta(hours=24)
        )
        
        return recommendation
    
    def _map_action_to_recommendation_type(self, action_type: ActionType):
        """Map RL action type to recommendation type"""
        
        mapping = {
            ActionType.SCALE_UP: "rightsizing",
            ActionType.SCALE_DOWN: "rightsizing", 
            ActionType.RIGHTSIZING: "rightsizing",
            ActionType.RESERVED_INSTANCE: "reserved_instance",
            ActionType.UNUSED_RESOURCE: "unused_resource",
            ActionType.WORKLOAD_MIGRATION: "rightsizing"
        }
        
        return mapping.get(action_type, "rightsizing")
    
    def _generate_recommendation_text(self, action: OptimizationAction) -> str:
        """Generate human-readable recommendation text"""
        
        action_descriptions = {
            ActionType.SCALE_UP: f"Scale up resource to handle increased demand. Expected cost increase: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.SCALE_DOWN: f"Scale down resource to reduce costs. Expected savings: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.RIGHTSIZING: f"Rightsize resource for optimal cost-performance ratio. Expected savings: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.RESERVED_INSTANCE: f"Purchase reserved instances for long-term savings. Expected savings: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.UNUSED_RESOURCE: f"Terminate unused resource. Expected savings: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.WORKLOAD_MIGRATION: f"Migrate workload to more cost-effective provider. Expected savings: ${action.expected_impact.get('cost_savings', 0.0):.2f}",
            ActionType.NO_ACTION: "No optimization needed at this time."
        }
        
        return action_descriptions.get(action.action_type, "Optimization recommendation generated by RL agent.")
    
    async def _execute_action(self, action: OptimizationAction, 
                            provider: CloudProvider) -> Dict[str, Any]:
        """Execute an optimization action"""
        
        # This would integrate with actual cloud provider APIs
        # For now, simulate execution
        
        try:
            # Simulate different success rates based on action type and risk
            success_probability = {
                RiskLevel.LOW: 0.95,
                RiskLevel.MEDIUM: 0.85,
                RiskLevel.HIGH: 0.70
            }.get(action.risk_level, 0.80)
            
            success = np.random.random() < success_probability
            
            if success:
                # Simulate actual results with some variance from expected
                variance_factor = np.random.normal(1.0, 0.2)  # ±20% variance
                
                actual_cost_savings = action.expected_impact.get('cost_savings', 0.0) * variance_factor
                actual_performance_change = action.expected_impact.get('performance_change', 0.0) * variance_factor
                actual_availability_change = action.expected_impact.get('availability_change', 0.0) * variance_factor
                
                return {
                    'success': True,
                    'cost_savings': actual_cost_savings,
                    'performance_change': actual_performance_change,
                    'availability_change': actual_availability_change,
                    'side_effects': []
                }
            else:
                return {
                    'success': False,
                    'cost_savings': 0.0,
                    'performance_change': 0.0,
                    'availability_change': 0.0,
                    'side_effects': ['Execution failed'],
                    'error_message': 'Simulated execution failure'
                }
                
        except Exception as e:
            return {
                'success': False,
                'cost_savings': 0.0,
                'performance_change': 0.0,
                'availability_change': 0.0,
                'side_effects': [f'Exception: {str(e)}'],
                'error_message': str(e)
            }
    
    async def _get_optimization_candidates(self, account_id: str, 
                                        limit: int) -> List[Dict[str, Any]]:
        """Get resources that are candidates for optimization"""
        
        # This would query the database for resources needing optimization
        # For now, return mock data
        
        candidates = []
        for i in range(min(limit, 5)):  # Limit to 5 for demo
            candidates.append({
                'resource_id': f'resource_{account_id}_{i}',
                'resource_type': 'compute',
                'current_cost': 100.0 + i * 20,
                'utilization': 0.3 + i * 0.1
            })
        
        return candidates
    
    def _generate_reasoning(self, action: OptimizationAction, state: SystemState) -> str:
        """Generate reasoning for the recommendation"""
        
        cpu_util = state.resource_utilization.get('cpu_avg', 0.5)
        cost_trend = state.cost_metrics.get('cost_trend_24h', 0.0)
        
        if action.action_type == ActionType.SCALE_DOWN:
            return f"Low CPU utilization ({cpu_util:.1%}) indicates over-provisioning. Scaling down can reduce costs."
        elif action.action_type == ActionType.SCALE_UP:
            return f"High CPU utilization ({cpu_util:.1%}) may impact performance. Scaling up recommended."
        elif action.action_type == ActionType.RESERVED_INSTANCE:
            return f"Consistent usage pattern detected. Reserved instances can provide significant savings."
        else:
            return f"RL agent recommends {action.action_type.value} based on current system state and learned patterns."
    
    def _update_metrics(self, outcome: ActionOutcome):
        """Update service performance metrics"""
        
        self.metrics['total_optimizations'] += 1
        
        if outcome.success:
            self.metrics['successful_optimizations'] += 1
            self.metrics['total_cost_savings'] += outcome.actual_cost_savings
        
        # Update average confidence (rolling average)
        if hasattr(self, '_confidence_history'):
            self._confidence_history.append(outcome.success)
            if len(self._confidence_history) > 100:
                self._confidence_history.pop(0)
            self.metrics['average_confidence'] = np.mean(self._confidence_history)
        else:
            self._confidence_history = [outcome.success]
            self.metrics['average_confidence'] = outcome.success
    
    async def _log_optimization_execution(self, resource_id: str, 
                                        action: OptimizationAction, 
                                        outcome: ActionOutcome):
        """Log optimization execution to audit trail"""
        
        try:
            async with get_db_session() as session:
                audit_log = AuditLog(
                    user_id=None,  # System action
                    action=f"rl_optimization_{action.action_type.value}",
                    resource_type="optimization",
                    resource_id=resource_id,
                    old_values={},
                    new_values={
                        'action_id': action.action_id,
                        'action_type': action.action_type.value,
                        'expected_savings': action.expected_impact.get('cost_savings', 0.0),
                        'actual_savings': outcome.actual_cost_savings,
                        'success': outcome.success,
                        'confidence': action.confidence
                    },
                    correlation_id=action.action_id
                )
                
                session.add(audit_log)
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging optimization execution: {str(e)}")
    
    async def _load_existing_model(self):
        """Load existing trained model if available"""
        
        try:
            model_path = "models/rl_agent_latest.pt"
            if os.path.exists(model_path):
                self.agent.load_model(model_path)
                self.logger.info("Loaded existing RL model")
            else:
                self.logger.info("No existing model found, starting with fresh model")
                
        except Exception as e:
            self.logger.error(f"Error loading existing model: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the service gracefully"""
        
        try:
            # Stop training pipeline
            self.training_pipeline.stop_pipeline()
            
            # Save current model
            model_path = "models/rl_agent_latest.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.agent.save_model(model_path)
            
            self.logger.info("RL Optimization Service shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

# Factory function
def create_rl_optimization_service(safety_checker: SafetyChecker) -> RLOptimizationService:
    """Create an RL optimization service"""
    return RLOptimizationService(safety_checker)