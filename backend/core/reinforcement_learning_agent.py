"""
Reinforcement Learning Agent for Cost Optimization

This module implements a reinforcement learning agent that continuously improves
optimization strategies through trial, feedback, and learning. The agent uses
policy and value networks to select optimal cost optimization actions and learns
from the outcomes to improve future decision-making.

Key Components:
- ReinforcementLearningAgent: Main RL agent with policy and value networks
- ExperienceReplay: Buffer for storing and replaying past experiences
- RewardCalculator: Computes rewards based on cost savings and performance
- ABTestingFramework: Framework for comparing optimization strategies
- PolicyNetwork & ValueNetwork: Neural networks for decision-making
"""

import asyncio
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .database import get_db_session
from .models import OptimizationRecommendation, AuditLog
from .cloud_providers import CloudProvider

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of optimization actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RIGHTSIZING = "rightsizing"
    RESERVED_INSTANCE = "reserved_instance"
    UNUSED_RESOURCE = "unused_resource"
    WORKLOAD_MIGRATION = "workload_migration"
    NO_ACTION = "no_action"

class RiskLevel(Enum):
    """Risk levels for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class SystemState:
    """Represents the current state of the system for RL decision-making"""
    timestamp: datetime
    resource_utilization: Dict[str, float]  # CPU, memory, network, etc.
    cost_metrics: Dict[str, float]  # Current costs, trends, etc.
    performance_metrics: Dict[str, float]  # Response times, error rates, etc.
    external_factors: Dict[str, Any]  # Time of day, seasonality, etc.
    budget_status: Dict[str, float]  # Budget utilization, remaining, etc.
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for neural network input"""
        features = []
        
        # Resource utilization features
        features.extend([
            self.resource_utilization.get('cpu_avg', 0.0),
            self.resource_utilization.get('memory_avg', 0.0),
            self.resource_utilization.get('network_in', 0.0),
            self.resource_utilization.get('network_out', 0.0),
            self.resource_utilization.get('disk_io', 0.0)
        ])
        
        # Cost metrics features
        features.extend([
            self.cost_metrics.get('current_hourly_cost', 0.0),
            self.cost_metrics.get('cost_trend_24h', 0.0),
            self.cost_metrics.get('cost_variance', 0.0),
            self.cost_metrics.get('cost_per_request', 0.0)
        ])
        
        # Performance metrics features
        features.extend([
            self.performance_metrics.get('avg_response_time', 0.0),
            self.performance_metrics.get('error_rate', 0.0),
            self.performance_metrics.get('throughput', 0.0),
            self.performance_metrics.get('availability', 1.0)
        ])
        
        # External factors features
        features.extend([
            self.external_factors.get('hour_of_day', 0.0) / 24.0,  # Normalize to 0-1
            self.external_factors.get('day_of_week', 0.0) / 7.0,   # Normalize to 0-1
            self.external_factors.get('is_weekend', 0.0),
            self.external_factors.get('seasonal_factor', 1.0)
        ])
        
        # Budget status features
        features.extend([
            self.budget_status.get('utilization_percent', 0.0) / 100.0,
            self.budget_status.get('remaining_days', 30.0) / 30.0,
            self.budget_status.get('projected_overrun', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)

@dataclass
class OptimizationAction:
    """Represents an optimization action to be taken"""
    action_id: str
    action_type: ActionType
    resource_id: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    risk_level: RiskLevel
    confidence: float
    
    def to_vector(self) -> np.ndarray:
        """Convert action to numerical vector"""
        # One-hot encode action type
        action_encoding = [0.0] * len(ActionType)
        action_encoding[list(ActionType).index(self.action_type)] = 1.0
        
        # Risk level encoding
        risk_encoding = [0.0] * len(RiskLevel)
        risk_encoding[list(RiskLevel).index(self.risk_level)] = 1.0
        
        # Expected impact features
        impact_features = [
            self.expected_impact.get('cost_savings', 0.0),
            self.expected_impact.get('performance_change', 0.0),
            self.expected_impact.get('availability_change', 0.0)
        ]
        
        features = action_encoding + risk_encoding + impact_features + [self.confidence]
        return np.array(features, dtype=np.float32)

@dataclass
class Experience:
    """Represents a single experience for learning"""
    state: SystemState
    action: OptimizationAction
    reward: float
    next_state: SystemState
    done: bool
    timestamp: datetime

@dataclass
class ActionOutcome:
    """Represents the outcome of an executed action"""
    action_id: str
    success: bool
    actual_cost_savings: float
    actual_performance_change: float
    actual_availability_change: float
    execution_time: datetime
    side_effects: List[str]
    user_feedback: Optional[float]  # User satisfaction score 0-1

class PolicyNetwork(nn.Module):
    """Neural network for policy (action selection)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    """Neural network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        return self.network(state)

class ExperienceReplay:
    """Experience replay buffer for storing and sampling past experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.logger = logging.getLogger(__name__ + ".ExperienceReplay")
    
    def add(self, experience: Experience):
        """Add an experience to the buffer"""
        self.buffer.append(experience)
        self.logger.debug(f"Added experience to buffer. Buffer size: {len(self.buffer)}")
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(list(self.buffer), batch_size)
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.logger.info("Experience buffer cleared")

class RewardCalculator:
    """Calculates rewards based on cost savings and performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".RewardCalculator")
        
        # Reward weights
        self.cost_weight = 0.4
        self.performance_weight = 0.3
        self.availability_weight = 0.2
        self.user_feedback_weight = 0.1
        
        # Penalty factors
        self.failure_penalty = -1.0
        self.risk_penalty_factors = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: -0.1,
            RiskLevel.HIGH: -0.2
        }
    
    def calculate_reward(self, action: OptimizationAction, outcome: ActionOutcome) -> float:
        """Calculate reward based on action and outcome"""
        
        if not outcome.success:
            # Heavy penalty for failed actions
            reward = self.failure_penalty
            self.logger.debug(f"Action {action.action_id} failed, reward: {reward}")
            return reward
        
        # Cost savings reward (normalized)
        expected_savings = action.expected_impact.get('cost_savings', 0.0)
        actual_savings = outcome.actual_cost_savings
        
        if expected_savings > 0:
            cost_reward = (actual_savings / expected_savings) * self.cost_weight
        else:
            cost_reward = max(0, actual_savings) * self.cost_weight
        
        # Performance reward
        expected_perf = action.expected_impact.get('performance_change', 0.0)
        actual_perf = outcome.actual_performance_change
        
        if expected_perf != 0:
            perf_reward = (actual_perf / abs(expected_perf)) * self.performance_weight
        else:
            perf_reward = max(0, actual_perf) * self.performance_weight
        
        # Availability reward
        expected_avail = action.expected_impact.get('availability_change', 0.0)
        actual_avail = outcome.actual_availability_change
        
        if expected_avail != 0:
            avail_reward = (actual_avail / abs(expected_avail)) * self.availability_weight
        else:
            avail_reward = max(0, actual_avail) * self.availability_weight
        
        # User feedback reward
        feedback_reward = 0.0
        if outcome.user_feedback is not None:
            feedback_reward = outcome.user_feedback * self.user_feedback_weight
        
        # Risk penalty
        risk_penalty = self.risk_penalty_factors.get(action.risk_level, 0.0)
        
        # Total reward
        total_reward = cost_reward + perf_reward + avail_reward + feedback_reward + risk_penalty
        
        self.logger.debug(f"Reward calculation for action {action.action_id}: "
                         f"cost={cost_reward:.3f}, perf={perf_reward:.3f}, "
                         f"avail={avail_reward:.3f}, feedback={feedback_reward:.3f}, "
                         f"risk_penalty={risk_penalty:.3f}, total={total_reward:.3f}")
        
        return float(total_reward)
    
    def calculate_immediate_reward(self, action: OptimizationAction, 
                                 state: SystemState) -> float:
        """Calculate immediate reward for action selection (before execution)"""
        
        # Reward based on expected impact and current state needs
        cost_urgency = self._calculate_cost_urgency(state)
        performance_urgency = self._calculate_performance_urgency(state)
        
        expected_cost_impact = action.expected_impact.get('cost_savings', 0.0)
        expected_perf_impact = action.expected_impact.get('performance_change', 0.0)
        
        # Reward actions that address urgent needs
        immediate_reward = (
            cost_urgency * expected_cost_impact * 0.5 +
            performance_urgency * expected_perf_impact * 0.3 +
            action.confidence * 0.2
        )
        
        # Apply risk penalty
        risk_penalty = self.risk_penalty_factors.get(action.risk_level, 0.0)
        immediate_reward += risk_penalty
        
        return float(immediate_reward)
    
    def _calculate_cost_urgency(self, state: SystemState) -> float:
        """Calculate urgency of cost optimization based on state"""
        
        budget_util = state.budget_status.get('utilization_percent', 0.0) / 100.0
        cost_trend = state.cost_metrics.get('cost_trend_24h', 0.0)
        
        # Higher urgency if budget utilization is high or costs are trending up
        urgency = min(1.0, budget_util + max(0, cost_trend))
        
        return urgency
    
    def _calculate_performance_urgency(self, state: SystemState) -> float:
        """Calculate urgency of performance optimization based on state"""
        
        error_rate = state.performance_metrics.get('error_rate', 0.0)
        response_time = state.performance_metrics.get('avg_response_time', 0.0)
        availability = state.performance_metrics.get('availability', 1.0)
        
        # Higher urgency if performance metrics are poor
        urgency = (
            min(1.0, error_rate * 10) +  # Scale error rate
            min(1.0, max(0, response_time - 1000) / 5000) +  # Response time > 1s
            max(0, 1.0 - availability)  # Availability issues
        ) / 3.0
        
        return urgency

class ABTestingFramework:
    """Framework for A/B testing different optimization strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ABTestingFramework")
        self.active_tests: Dict[str, Dict] = {}
        self.test_results: Dict[str, Dict] = {}
    
    def create_test(self, test_id: str, strategy_a: str, strategy_b: str,
                   traffic_split: float = 0.5, duration_hours: int = 24) -> bool:
        """Create a new A/B test"""
        
        if test_id in self.active_tests:
            self.logger.warning(f"Test {test_id} already exists")
            return False
        
        test_config = {
            'test_id': test_id,
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=duration_hours),
            'results_a': [],
            'results_b': []
        }
        
        self.active_tests[test_id] = test_config
        self.logger.info(f"Created A/B test {test_id}: {strategy_a} vs {strategy_b}")
        
        return True
    
    def assign_strategy(self, test_id: str, resource_id: str) -> Optional[str]:
        """Assign a strategy for a resource in an A/B test"""
        
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Check if test is still active
        if datetime.now() > test['end_time']:
            return None
        
        # Deterministic assignment based on resource_id hash
        hash_value = hash(resource_id) % 100
        threshold = int(test['traffic_split'] * 100)
        
        if hash_value < threshold:
            return test['strategy_a']
        else:
            return test['strategy_b']
    
    def record_result(self, test_id: str, strategy: str, outcome: ActionOutcome):
        """Record the result of an action in an A/B test"""
        
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        result = {
            'timestamp': outcome.execution_time,
            'success': outcome.success,
            'cost_savings': outcome.actual_cost_savings,
            'performance_change': outcome.actual_performance_change,
            'user_feedback': outcome.user_feedback
        }
        
        if strategy == test['strategy_a']:
            test['results_a'].append(result)
        elif strategy == test['strategy_b']:
            test['results_b'].append(result)
        
        self.logger.debug(f"Recorded result for test {test_id}, strategy {strategy}")
    
    def analyze_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Analyze the results of an A/B test"""
        
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        results_a = test['results_a']
        results_b = test['results_b']
        
        if len(results_a) == 0 or len(results_b) == 0:
            return None
        
        # Calculate metrics for strategy A
        metrics_a = self._calculate_strategy_metrics(results_a)
        
        # Calculate metrics for strategy B
        metrics_b = self._calculate_strategy_metrics(results_b)
        
        # Determine winner
        winner = self._determine_winner(metrics_a, metrics_b)
        
        analysis = {
            'test_id': test_id,
            'strategy_a': test['strategy_a'],
            'strategy_b': test['strategy_b'],
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'winner': winner,
            'confidence': self._calculate_confidence(metrics_a, metrics_b),
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b)
        }
        
        self.test_results[test_id] = analysis
        self.logger.info(f"Analyzed test {test_id}, winner: {winner}")
        
        return analysis
    
    def _calculate_strategy_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for a strategy"""
        
        if not results:
            return {}
        
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_cost_savings = np.mean([r['cost_savings'] for r in results])
        avg_performance_change = np.mean([r['performance_change'] for r in results])
        
        # User feedback (only from results that have it)
        feedback_results = [r['user_feedback'] for r in results if r['user_feedback'] is not None]
        avg_user_feedback = np.mean(feedback_results) if feedback_results else 0.0
        
        return {
            'success_rate': success_rate,
            'avg_cost_savings': avg_cost_savings,
            'avg_performance_change': avg_performance_change,
            'avg_user_feedback': avg_user_feedback,
            'total_cost_savings': sum(r['cost_savings'] for r in results)
        }
    
    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict) -> str:
        """Determine the winning strategy based on metrics"""
        
        # Weighted score calculation
        score_a = (
            metrics_a.get('success_rate', 0) * 0.3 +
            max(0, metrics_a.get('avg_cost_savings', 0)) * 0.4 +
            max(0, metrics_a.get('avg_performance_change', 0)) * 0.2 +
            metrics_a.get('avg_user_feedback', 0) * 0.1
        )
        
        score_b = (
            metrics_b.get('success_rate', 0) * 0.3 +
            max(0, metrics_b.get('avg_cost_savings', 0)) * 0.4 +
            max(0, metrics_b.get('avg_performance_change', 0)) * 0.2 +
            metrics_b.get('avg_user_feedback', 0) * 0.1
        )
        
        if score_a > score_b:
            return 'strategy_a'
        elif score_b > score_a:
            return 'strategy_b'
        else:
            return 'tie'
    
    def _calculate_confidence(self, metrics_a: Dict, metrics_b: Dict) -> float:
        """Calculate confidence in the test results"""
        
        # Simple confidence calculation based on sample size and difference
        # In a real implementation, this would use proper statistical tests
        
        diff = abs(metrics_a.get('avg_cost_savings', 0) - metrics_b.get('avg_cost_savings', 0))
        
        # Higher difference and larger sample sizes increase confidence
        confidence = min(0.95, diff * 0.1 + 0.5)
        
        return confidence
    
    def end_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """End an A/B test and return final analysis"""
        
        if test_id not in self.active_tests:
            return None
        
        analysis = self.analyze_test(test_id)
        
        # Move test to completed tests
        if analysis:
            del self.active_tests[test_id]
            self.logger.info(f"Ended A/B test {test_id}")
        
        return analysis

class ReinforcementLearningAgent:
    """Main reinforcement learning agent for cost optimization"""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 7, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_network = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Components
        self.experience_replay = ExperienceReplay()
        self.reward_calculator = RewardCalculator()
        self.ab_testing = ABTestingFramework()
        
        # State preprocessing
        self.state_scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # Learning parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.batch_size = 32
        
        # Performance tracking
        self.action_history: List[Dict] = []
        self.learning_metrics: Dict[str, List[float]] = {
            'policy_loss': [],
            'value_loss': [],
            'average_reward': [],
            'success_rate': []
        }
        
        self.logger = logging.getLogger(__name__ + ".ReinforcementLearningAgent")
        self.logger.info("Reinforcement Learning Agent initialized")
    
    async def select_action(self, state: SystemState) -> OptimizationAction:
        """Select an optimization action based on current state"""
        
        try:
            # Preprocess state
            state_vector = self._preprocess_state(state)
            
            # Get action probabilities from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                action_probs = self.policy_network(state_tensor)
                action_probs = action_probs.cpu().numpy()[0]
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_dim - 1)
                self.logger.debug("Selected random action for exploration")
            else:
                action_idx = np.argmax(action_probs)
                self.logger.debug(f"Selected action {action_idx} with probability {action_probs[action_idx]:.3f}")
            
            # Convert action index to optimization action
            action = self._create_optimization_action(action_idx, state, action_probs[action_idx])
            
            # Calculate immediate reward for action selection
            immediate_reward = self.reward_calculator.calculate_immediate_reward(action, state)
            
            # Record action selection
            self.action_history.append({
                'timestamp': datetime.now(),
                'state': asdict(state),
                'action': asdict(action),
                'immediate_reward': immediate_reward,
                'action_probabilities': action_probs.tolist()
            })
            
            self.logger.info(f"Selected action {action.action_type.value} for resource {action.resource_id} "
                           f"with confidence {action.confidence:.3f}")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error selecting action: {str(e)}")
            # Return safe default action
            return self._create_default_action(state)
    
    async def update_policy(self, experience: Experience) -> Dict[str, float]:
        """Update policy based on experience"""
        
        try:
            # Add experience to replay buffer
            self.experience_replay.add(experience)
            
            # Only update if we have enough experiences
            if self.experience_replay.size() < self.batch_size:
                return {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Sample batch of experiences
            batch = self.experience_replay.sample(self.batch_size)
            
            # Prepare batch data
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for exp in batch:
                states.append(self._preprocess_state(exp.state))
                actions.append(self._action_to_index(exp.action))
                rewards.append(exp.reward)
                next_states.append(self._preprocess_state(exp.next_state))
                dones.append(exp.done)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            
            # Calculate target values
            with torch.no_grad():
                next_values = self.value_network(next_states).squeeze()
                targets = rewards + self.gamma * next_values * (~dones)
            
            # Update value network
            current_values = self.value_network(states).squeeze()
            value_loss = F.mse_loss(current_values, targets)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Update policy network
            action_probs = self.policy_network(states)
            selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Calculate advantages
            advantages = targets - current_values.detach()
            
            # Policy loss (negative log likelihood weighted by advantages)
            policy_loss = -torch.mean(torch.log(selected_action_probs + 1e-8) * advantages)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Record metrics
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            
            self.learning_metrics['policy_loss'].append(policy_loss_val)
            self.learning_metrics['value_loss'].append(value_loss_val)
            
            self.logger.debug(f"Updated policy: policy_loss={policy_loss_val:.4f}, "
                            f"value_loss={value_loss_val:.4f}")
            
            return {
                'policy_loss': policy_loss_val,
                'value_loss': value_loss_val,
                'batch_size': len(batch)
            }
            
        except Exception as e:
            self.logger.error(f"Error updating policy: {str(e)}")
            return {'policy_loss': 0.0, 'value_loss': 0.0}
    
    async def evaluate_action(self, action: OptimizationAction, outcome: ActionOutcome) -> float:
        """Evaluate an action based on its outcome and calculate reward"""
        
        try:
            reward = self.reward_calculator.calculate_reward(action, outcome)
            
            # Update success rate metric
            recent_outcomes = [h for h in self.action_history[-100:] if 'outcome' in h]
            if recent_outcomes:
                success_rate = sum(1 for h in recent_outcomes if h['outcome']['success']) / len(recent_outcomes)
                self.learning_metrics['success_rate'].append(success_rate)
            
            # Update average reward metric
            recent_rewards = self.learning_metrics['average_reward'][-100:]
            recent_rewards.append(reward)
            avg_reward = np.mean(recent_rewards)
            self.learning_metrics['average_reward'].append(avg_reward)
            
            # Find corresponding action in history and update with outcome
            for i in range(len(self.action_history) - 1, -1, -1):
                if self.action_history[i].get('action', {}).get('action_id') == action.action_id:
                    self.action_history[i]['outcome'] = asdict(outcome)
                    self.action_history[i]['reward'] = reward
                    break
            
            self.logger.info(f"Evaluated action {action.action_id}: reward={reward:.3f}, "
                           f"success={outcome.success}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error evaluating action: {str(e)}")
            return 0.0
    
    async def get_action_confidence(self, state: SystemState, action: OptimizationAction) -> float:
        """Get confidence score for a specific action in a given state"""
        
        try:
            # Preprocess state
            state_vector = self._preprocess_state(state)
            
            # Get action probabilities
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                action_probs = self.policy_network(state_tensor)
                action_probs = action_probs.cpu().numpy()[0]
            
            # Get probability for the specific action
            action_idx = self._action_to_index(action)
            confidence = float(action_probs[action_idx])
            
            # Adjust confidence based on historical performance
            historical_confidence = self._get_historical_confidence(action.action_type)
            
            # Weighted average of network confidence and historical performance
            final_confidence = 0.7 * confidence + 0.3 * historical_confidence
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating action confidence: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def _preprocess_state(self, state: SystemState) -> np.ndarray:
        """Preprocess state for neural network input"""
        
        state_vector = state.to_vector()
        
        # Fit scaler on first state if not fitted
        if not self.is_scaler_fitted:
            self.state_scaler.fit(state_vector.reshape(1, -1))
            self.is_scaler_fitted = True
        
        # Scale state vector
        scaled_state = self.state_scaler.transform(state_vector.reshape(1, -1))[0]
        
        return scaled_state
    
    def _create_optimization_action(self, action_idx: int, state: SystemState, 
                                  confidence: float) -> OptimizationAction:
        """Create an optimization action from action index"""
        
        action_types = list(ActionType)
        action_type = action_types[action_idx]
        
        # Generate action parameters based on type and state
        parameters = self._generate_action_parameters(action_type, state)
        
        # Estimate expected impact
        expected_impact = self._estimate_action_impact(action_type, state, parameters)
        
        # Determine risk level
        risk_level = self._determine_risk_level(action_type, parameters)
        
        return OptimizationAction(
            action_id=f"rl_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            action_type=action_type,
            resource_id=parameters.get('resource_id', 'unknown'),
            parameters=parameters,
            expected_impact=expected_impact,
            risk_level=risk_level,
            confidence=confidence
        )
    
    def _generate_action_parameters(self, action_type: ActionType, state: SystemState) -> Dict[str, Any]:
        """Generate parameters for an action based on type and state"""
        
        # This would be more sophisticated in a real implementation
        # For now, generate basic parameters based on action type
        
        base_params = {
            'resource_id': f"resource_{random.randint(1, 100)}",
            'timestamp': datetime.now().isoformat()
        }
        
        if action_type == ActionType.SCALE_UP:
            base_params.update({
                'scale_factor': min(2.0, 1.0 + state.resource_utilization.get('cpu_avg', 0.5)),
                'target_capacity': int(10 * (1 + state.resource_utilization.get('cpu_avg', 0.5)))
            })
        
        elif action_type == ActionType.SCALE_DOWN:
            base_params.update({
                'scale_factor': max(0.5, state.resource_utilization.get('cpu_avg', 0.5)),
                'target_capacity': max(1, int(10 * state.resource_utilization.get('cpu_avg', 0.5)))
            })
        
        elif action_type == ActionType.RIGHTSIZING:
            base_params.update({
                'new_instance_type': 't3.medium',  # Example
                'cpu_optimization': True,
                'memory_optimization': True
            })
        
        elif action_type == ActionType.RESERVED_INSTANCE:
            base_params.update({
                'commitment_term': '1year',
                'payment_option': 'partial_upfront',
                'instance_count': random.randint(1, 5)
            })
        
        elif action_type == ActionType.UNUSED_RESOURCE:
            base_params.update({
                'action': 'terminate',
                'grace_period_hours': 24
            })
        
        elif action_type == ActionType.WORKLOAD_MIGRATION:
            base_params.update({
                'target_provider': random.choice(['aws', 'gcp', 'azure']),
                'migration_strategy': 'blue_green'
            })
        
        return base_params
    
    def _estimate_action_impact(self, action_type: ActionType, state: SystemState, 
                              parameters: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the expected impact of an action"""
        
        # Simplified impact estimation - would be more sophisticated in practice
        current_cost = state.cost_metrics.get('current_hourly_cost', 100.0)
        
        if action_type == ActionType.SCALE_UP:
            return {
                'cost_savings': -current_cost * 0.2,  # Negative = cost increase
                'performance_change': 0.3,
                'availability_change': 0.1
            }
        
        elif action_type == ActionType.SCALE_DOWN:
            return {
                'cost_savings': current_cost * 0.3,
                'performance_change': -0.1,
                'availability_change': -0.05
            }
        
        elif action_type == ActionType.RIGHTSIZING:
            return {
                'cost_savings': current_cost * 0.2,
                'performance_change': 0.1,
                'availability_change': 0.0
            }
        
        elif action_type == ActionType.RESERVED_INSTANCE:
            return {
                'cost_savings': current_cost * 0.4,
                'performance_change': 0.0,
                'availability_change': 0.0
            }
        
        elif action_type == ActionType.UNUSED_RESOURCE:
            return {
                'cost_savings': current_cost * 1.0,  # Full cost savings
                'performance_change': 0.0,
                'availability_change': 0.0
            }
        
        elif action_type == ActionType.WORKLOAD_MIGRATION:
            return {
                'cost_savings': current_cost * 0.15,
                'performance_change': 0.05,
                'availability_change': -0.02  # Temporary during migration
            }
        
        else:  # NO_ACTION
            return {
                'cost_savings': 0.0,
                'performance_change': 0.0,
                'availability_change': 0.0
            }
    
    def _determine_risk_level(self, action_type: ActionType, parameters: Dict[str, Any]) -> RiskLevel:
        """Determine risk level for an action"""
        
        high_risk_actions = [ActionType.WORKLOAD_MIGRATION, ActionType.UNUSED_RESOURCE]
        medium_risk_actions = [ActionType.SCALE_DOWN, ActionType.RIGHTSIZING]
        
        if action_type in high_risk_actions:
            return RiskLevel.HIGH
        elif action_type in medium_risk_actions:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _action_to_index(self, action: OptimizationAction) -> int:
        """Convert optimization action to index"""
        action_types = list(ActionType)
        return action_types.index(action.action_type)
    
    def _create_default_action(self, state: SystemState) -> OptimizationAction:
        """Create a safe default action"""
        return OptimizationAction(
            action_id=f"default_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            action_type=ActionType.NO_ACTION,
            resource_id="unknown",
            parameters={},
            expected_impact={'cost_savings': 0.0, 'performance_change': 0.0, 'availability_change': 0.0},
            risk_level=RiskLevel.LOW,
            confidence=1.0
        )
    
    def _get_historical_confidence(self, action_type: ActionType) -> float:
        """Get historical confidence for an action type based on past performance"""
        
        # Filter actions by type from history
        type_actions = [
            h for h in self.action_history 
            if h.get('action', {}).get('action_type') == action_type.value and 'outcome' in h
        ]
        
        if not type_actions:
            return 0.5  # Default confidence
        
        # Calculate success rate for this action type
        successes = sum(1 for h in type_actions if h['outcome']['success'])
        success_rate = successes / len(type_actions)
        
        return success_rate
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics and performance statistics"""
        
        metrics = {
            'total_actions': len(self.action_history),
            'experience_buffer_size': self.experience_replay.size(),
            'recent_policy_loss': self.learning_metrics['policy_loss'][-10:] if self.learning_metrics['policy_loss'] else [],
            'recent_value_loss': self.learning_metrics['value_loss'][-10:] if self.learning_metrics['value_loss'] else [],
            'recent_average_reward': self.learning_metrics['average_reward'][-10:] if self.learning_metrics['average_reward'] else [],
            'recent_success_rate': self.learning_metrics['success_rate'][-10:] if self.learning_metrics['success_rate'] else [],
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
        
        # Action type distribution
        action_types = [h.get('action', {}).get('action_type') for h in self.action_history]
        action_distribution = {}
        for action_type in ActionType:
            action_distribution[action_type.value] = action_types.count(action_type.value)
        
        metrics['action_distribution'] = action_distribution
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'state_scaler': self.state_scaler,
                'learning_metrics': self.learning_metrics,
                'epsilon': self.epsilon
            }, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            self.state_scaler = checkpoint['state_scaler']
            self.is_scaler_fitted = True
            self.learning_metrics = checkpoint['learning_metrics']
            self.epsilon = checkpoint['epsilon']
            
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")