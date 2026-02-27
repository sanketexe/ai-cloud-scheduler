"""
Reinforcement Learning Training Pipeline

This module implements the continuous learning and model improvement pipeline
for the reinforcement learning agent. It handles automated training, model
evaluation, hyperparameter tuning, and deployment of improved models.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

from .reinforcement_learning_agent import (
    ReinforcementLearningAgent, Experience, SystemState, 
    OptimizationAction, ActionOutcome
)
from .database import get_db_session
from .models import MLModelMetrics

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    training_interval_hours: int = 24
    min_experiences_for_training: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    max_training_epochs: int = 100
    learning_rate_decay: float = 0.95
    model_save_path: str = "models/rl_agent"
    backup_retention_days: int = 30

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    average_reward: float
    success_rate: float
    cost_savings_rate: float
    training_loss: float
    validation_loss: float

class ModelEvaluator:
    """Evaluates model performance and determines if updates are needed"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ModelEvaluator")
        
        # Performance thresholds for model updates
        self.min_improvement_threshold = 0.05  # 5% improvement required
        self.min_success_rate = 0.7
        self.min_cost_savings_rate = 0.1
    
    async def evaluate_model(self, agent: ReinforcementLearningAgent, 
                           validation_experiences: List[Experience]) -> ModelPerformance:
        """Evaluate model performance on validation data"""
        
        if not validation_experiences:
            return self._create_default_performance()
        
        try:
            # Collect predictions and actual outcomes
            predictions = []
            actual_rewards = []
            success_count = 0
            total_cost_savings = 0.0
            
            for experience in validation_experiences:
                # Get model's action prediction
                predicted_action = await agent.select_action(experience.state)
                
                # Compare with actual action taken
                action_match = predicted_action.action_type == experience.action.action_type
                predictions.append(1 if action_match else 0)
                
                # Collect reward information
                actual_rewards.append(experience.reward)
                
                # Track success and cost savings (from action outcomes if available)
                if hasattr(experience, 'outcome') and experience.outcome:
                    if experience.outcome.success:
                        success_count += 1
                    total_cost_savings += experience.outcome.actual_cost_savings
            
            # Calculate metrics
            accuracy = np.mean(predictions) if predictions else 0.0
            average_reward = np.mean(actual_rewards) if actual_rewards else 0.0
            success_rate = success_count / len(validation_experiences) if validation_experiences else 0.0
            cost_savings_rate = total_cost_savings / len(validation_experiences) if validation_experiences else 0.0
            
            # Get training metrics from agent
            learning_metrics = agent.get_learning_metrics()
            training_loss = np.mean(learning_metrics.get('recent_policy_loss', [0.0]))
            validation_loss = training_loss * 1.1  # Approximate validation loss
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=accuracy,  # Simplified for now
                recall=accuracy,     # Simplified for now
                f1_score=accuracy,   # Simplified for now
                average_reward=average_reward,
                success_rate=success_rate,
                cost_savings_rate=cost_savings_rate,
                training_loss=training_loss,
                validation_loss=validation_loss
            )
            
            self.logger.info(f"Model evaluation completed: accuracy={accuracy:.3f}, "
                           f"success_rate={success_rate:.3f}, avg_reward={average_reward:.3f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return self._create_default_performance()
    
    def should_update_model(self, current_performance: ModelPerformance, 
                          previous_performance: Optional[ModelPerformance]) -> bool:
        """Determine if model should be updated based on performance"""
        
        # Always update if no previous performance data
        if previous_performance is None:
            return True
        
        # Check minimum performance thresholds
        if (current_performance.success_rate < self.min_success_rate or
            current_performance.cost_savings_rate < self.min_cost_savings_rate):
            self.logger.warning("Model performance below minimum thresholds")
            return False
        
        # Check for improvement
        reward_improvement = (current_performance.average_reward - 
                            previous_performance.average_reward) / abs(previous_performance.average_reward)
        
        success_improvement = (current_performance.success_rate - 
                             previous_performance.success_rate)
        
        cost_improvement = (current_performance.cost_savings_rate - 
                          previous_performance.cost_savings_rate)
        
        # Weighted improvement score
        improvement_score = (
            reward_improvement * 0.4 +
            success_improvement * 0.3 +
            cost_improvement * 0.3
        )
        
        should_update = improvement_score >= self.min_improvement_threshold
        
        self.logger.info(f"Model update decision: improvement_score={improvement_score:.3f}, "
                        f"threshold={self.min_improvement_threshold}, update={should_update}")
        
        return should_update
    
    def _create_default_performance(self) -> ModelPerformance:
        """Create default performance metrics"""
        return ModelPerformance(
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            average_reward=0.0,
            success_rate=0.5,
            cost_savings_rate=0.0,
            training_loss=1.0,
            validation_loss=1.0
        )

class HyperparameterTuner:
    """Tunes hyperparameters for optimal model performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".HyperparameterTuner")
        
        # Hyperparameter search space
        self.search_space = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'gamma': [0.95, 0.99, 0.995],
            'epsilon': [0.05, 0.1, 0.2],
            'batch_size': [16, 32, 64],
            'hidden_dim': [64, 128, 256]
        }
    
    async def tune_hyperparameters(self, training_experiences: List[Experience],
                                 validation_experiences: List[Experience],
                                 max_trials: int = 10) -> Dict[str, Any]:
        """Tune hyperparameters using random search"""
        
        best_params = None
        best_performance = 0.0
        trial_results = []
        
        self.logger.info(f"Starting hyperparameter tuning with {max_trials} trials")
        
        for trial in range(max_trials):
            try:
                # Sample random hyperparameters
                params = self._sample_hyperparameters()
                
                self.logger.info(f"Trial {trial + 1}/{max_trials}: {params}")
                
                # Create and train agent with these parameters
                agent = ReinforcementLearningAgent(
                    learning_rate=params['learning_rate']
                )
                
                # Set other parameters
                agent.gamma = params['gamma']
                agent.epsilon = params['epsilon']
                agent.batch_size = params['batch_size']
                
                # Train on subset of data
                await self._train_agent_subset(agent, training_experiences[:500])  # Limit for speed
                
                # Evaluate performance
                evaluator = ModelEvaluator()
                performance = await evaluator.evaluate_model(agent, validation_experiences[:100])
                
                # Calculate composite score
                score = self._calculate_composite_score(performance)
                
                trial_results.append({
                    'trial': trial + 1,
                    'params': params,
                    'performance': performance,
                    'score': score
                })
                
                # Update best parameters
                if score > best_performance:
                    best_performance = score
                    best_params = params
                    self.logger.info(f"New best score: {score:.3f} with params: {params}")
                
            except Exception as e:
                self.logger.error(f"Error in trial {trial + 1}: {str(e)}")
                continue
        
        self.logger.info(f"Hyperparameter tuning completed. Best score: {best_performance:.3f}")
        
        return {
            'best_params': best_params,
            'best_score': best_performance,
            'trial_results': trial_results
        }
    
    def _sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample random hyperparameters from search space"""
        
        params = {}
        for param_name, values in self.search_space.items():
            params[param_name] = np.random.choice(values)
        
        return params
    
    async def _train_agent_subset(self, agent: ReinforcementLearningAgent, 
                                experiences: List[Experience]):
        """Train agent on a subset of experiences"""
        
        for experience in experiences:
            await agent.update_policy(experience)
    
    def _calculate_composite_score(self, performance: ModelPerformance) -> float:
        """Calculate composite performance score"""
        
        score = (
            performance.accuracy * 0.2 +
            performance.average_reward * 0.3 +
            performance.success_rate * 0.3 +
            performance.cost_savings_rate * 0.2
        )
        
        return score

class ContinuousLearningPipeline:
    """Main pipeline for continuous learning and model improvement"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".ContinuousLearningPipeline")
        
        # Components
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()
        
        # State tracking
        self.is_running = False
        self.last_training_time = None
        self.model_version = 1
        self.performance_history: List[ModelPerformance] = []
        
        # Ensure model directory exists
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    async def start_pipeline(self, agent: ReinforcementLearningAgent):
        """Start the continuous learning pipeline"""
        
        self.is_running = True
        self.logger.info("Starting continuous learning pipeline")
        
        while self.is_running:
            try:
                await self._run_training_cycle(agent)
                
                # Wait for next training interval
                await asyncio.sleep(self.config.training_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in training cycle: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    def stop_pipeline(self):
        """Stop the continuous learning pipeline"""
        self.is_running = False
        self.logger.info("Stopping continuous learning pipeline")
    
    async def _run_training_cycle(self, agent: ReinforcementLearningAgent):
        """Run a single training cycle"""
        
        self.logger.info("Starting training cycle")
        
        # Check if enough new experiences are available
        if agent.experience_replay.size() < self.config.min_experiences_for_training:
            self.logger.info(f"Insufficient experiences for training: "
                           f"{agent.experience_replay.size()} < {self.config.min_experiences_for_training}")
            return
        
        # Prepare training and validation data
        experiences = agent.experience_replay.sample(agent.experience_replay.size())
        train_experiences, val_experiences = self._split_experiences(experiences)
        
        # Evaluate current model performance
        current_performance = await self.evaluator.evaluate_model(agent, val_experiences)
        
        # Check if model update is needed
        previous_performance = self.performance_history[-1] if self.performance_history else None
        
        if not self.evaluator.should_update_model(current_performance, previous_performance):
            self.logger.info("Model update not needed based on performance evaluation")
            return
        
        # Perform hyperparameter tuning periodically
        should_tune = (len(self.performance_history) % 10 == 0 or 
                      current_performance.success_rate < 0.6)
        
        if should_tune:
            self.logger.info("Performing hyperparameter tuning")
            tuning_results = await self.tuner.tune_hyperparameters(
                train_experiences, val_experiences
            )
            
            if tuning_results['best_params']:
                await self._apply_hyperparameters(agent, tuning_results['best_params'])
        
        # Train the model
        await self._train_model(agent, train_experiences)
        
        # Evaluate updated model
        updated_performance = await self.evaluator.evaluate_model(agent, val_experiences)
        
        # Save model if performance improved
        if (previous_performance is None or 
            updated_performance.average_reward > previous_performance.average_reward):
            
            await self._save_model_version(agent, updated_performance)
            self.performance_history.append(updated_performance)
            
            # Log performance metrics to database
            await self._log_performance_metrics(updated_performance)
            
            self.logger.info(f"Model updated successfully. New performance: "
                           f"reward={updated_performance.average_reward:.3f}, "
                           f"success_rate={updated_performance.success_rate:.3f}")
        else:
            self.logger.info("Model performance did not improve, keeping previous version")
        
        # Clean up old model versions
        await self._cleanup_old_models()
        
        self.last_training_time = datetime.now()
    
    def _split_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[Experience]]:
        """Split experiences into training and validation sets"""
        
        split_idx = int(len(experiences) * (1 - self.config.validation_split))
        
        # Shuffle experiences
        shuffled = experiences.copy()
        np.random.shuffle(shuffled)
        
        train_experiences = shuffled[:split_idx]
        val_experiences = shuffled[split_idx:]
        
        return train_experiences, val_experiences
    
    async def _apply_hyperparameters(self, agent: ReinforcementLearningAgent, 
                                   params: Dict[str, Any]):
        """Apply new hyperparameters to the agent"""
        
        # Update learning rate
        if 'learning_rate' in params:
            for param_group in agent.policy_optimizer.param_groups:
                param_group['lr'] = params['learning_rate']
            for param_group in agent.value_optimizer.param_groups:
                param_group['lr'] = params['learning_rate']
        
        # Update other parameters
        if 'gamma' in params:
            agent.gamma = params['gamma']
        
        if 'epsilon' in params:
            agent.epsilon = params['epsilon']
        
        if 'batch_size' in params:
            agent.batch_size = params['batch_size']
        
        self.logger.info(f"Applied hyperparameters: {params}")
    
    async def _train_model(self, agent: ReinforcementLearningAgent, 
                         train_experiences: List[Experience]):
        """Train the model on training experiences"""
        
        self.logger.info(f"Training model on {len(train_experiences)} experiences")
        
        # Train for multiple epochs
        for epoch in range(min(self.config.max_training_epochs, 50)):
            
            # Shuffle experiences for each epoch
            np.random.shuffle(train_experiences)
            
            epoch_losses = []
            
            # Train on batches
            for i in range(0, len(train_experiences), agent.batch_size):
                batch = train_experiences[i:i + agent.batch_size]
                
                for experience in batch:
                    result = await agent.update_policy(experience)
                    epoch_losses.append(result.get('policy_loss', 0.0))
            
            # Log epoch progress
            if epoch % 10 == 0:
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                self.logger.debug(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
            
            # Early stopping check (simplified)
            if len(epoch_losses) > 0 and np.mean(epoch_losses) < 0.01:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
    
    async def _save_model_version(self, agent: ReinforcementLearningAgent, 
                                performance: ModelPerformance):
        """Save a new model version"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.config.model_save_path}_v{self.model_version}_{timestamp}.pt"
        
        # Save model
        agent.save_model(model_path)
        
        # Save performance metadata
        metadata = {
            'version': self.model_version,
            'timestamp': timestamp,
            'performance': {
                'accuracy': performance.accuracy,
                'average_reward': performance.average_reward,
                'success_rate': performance.success_rate,
                'cost_savings_rate': performance.cost_savings_rate
            }
        }
        
        metadata_path = f"{self.config.model_save_path}_v{self.model_version}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.model_version += 1
        self.logger.info(f"Saved model version {self.model_version - 1} to {model_path}")
    
    async def _log_performance_metrics(self, performance: ModelPerformance):
        """Log performance metrics to database"""
        
        try:
            async with get_db_session() as session:
                metrics = MLModelMetrics(
                    model_name="reinforcement_learning_agent",
                    model_version=f"v{self.model_version}",
                    account_id="system",
                    training_date=datetime.now(),
                    accuracy_score=performance.accuracy,
                    precision_score=performance.precision,
                    recall_score=performance.recall,
                    false_positive_rate=1.0 - performance.precision,
                    detection_latency_ms=100,  # Placeholder
                    training_data_points=self.config.min_experiences_for_training,
                    feature_count=21,  # State dimension
                    hyperparameters={
                        'learning_rate': 0.001,  # Would get from agent
                        'gamma': 0.99,
                        'epsilon': 0.1
                    },
                    performance_notes=f"Continuous learning cycle. Success rate: {performance.success_rate:.3f}",
                    is_active=True
                )
                
                session.add(metrics)
                await session.commit()
                
                self.logger.debug("Performance metrics logged to database")
                
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {str(e)}")
    
    async def _cleanup_old_models(self):
        """Clean up old model files"""
        
        try:
            model_dir = Path(self.config.model_save_path).parent
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            for file_path in model_dir.glob("*.pt"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    self.logger.debug(f"Deleted old model file: {file_path}")
            
            for file_path in model_dir.glob("*_metadata.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    self.logger.debug(f"Deleted old metadata file: {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old models: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        
        return {
            'is_running': self.is_running,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_version': self.model_version,
            'performance_history_length': len(self.performance_history),
            'latest_performance': (
                {
                    'accuracy': self.performance_history[-1].accuracy,
                    'average_reward': self.performance_history[-1].average_reward,
                    'success_rate': self.performance_history[-1].success_rate,
                    'cost_savings_rate': self.performance_history[-1].cost_savings_rate
                } if self.performance_history else None
            ),
            'config': {
                'training_interval_hours': self.config.training_interval_hours,
                'min_experiences_for_training': self.config.min_experiences_for_training,
                'validation_split': self.config.validation_split
            }
        }

# Factory function for creating pipeline with default config
def create_training_pipeline(config: Optional[TrainingConfig] = None) -> ContinuousLearningPipeline:
    """Create a continuous learning pipeline with default or custom config"""
    
    if config is None:
        config = TrainingConfig()
    
    return ContinuousLearningPipeline(config)