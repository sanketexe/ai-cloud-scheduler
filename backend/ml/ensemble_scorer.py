"""
Ensemble Scoring Algorithm for ML Model Combination

Combines predictions from multiple anomaly detection models (Isolation Forest, LSTM, Prophet)
using weighted ensemble methods to improve overall accuracy and reduce false positives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from enum import Enum
import json

from .isolation_forest_detector import IsolationForestDetector, AnomalyScore
from .lstm_anomaly_detector import LSTMAnomalyDetector, LSTMPrediction
from .prophet_forecaster import ProphetForecaster, AnomalyDetection

logger = structlog.get_logger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


@dataclass
class ModelWeight:
    """Weight configuration for individual models"""
    isolation_forest: float = 0.4
    lstm: float = 0.35
    prophet: float = 0.25


@dataclass
class EnsembleResult:
    """Combined ensemble prediction result"""
    timestamp: datetime
    ensemble_score: float  # 0 to 1, higher = more anomalous
    confidence: float      # 0 to 1, confidence in prediction
    is_anomaly: bool
    severity: str          # 'low', 'medium', 'high', 'critical'
    
    # Individual model contributions
    isolation_forest_score: Optional[float] = None
    lstm_score: Optional[float] = None
    prophet_score: Optional[float] = None
    
    # Model agreement metrics
    model_agreement: float = 0.0  # 0 to 1, how much models agree
    consensus_strength: float = 0.0  # Strength of consensus
    
    # Additional context
    contributing_models: List[str] = field(default_factory=list)
    model_weights_used: Dict[str, float] = field(default_factory=dict)
    raw_predictions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble scoring"""
    method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    weights: ModelWeight = field(default_factory=ModelWeight)
    
    # Thresholds
    anomaly_threshold: float = 0.7  # Ensemble score threshold for anomaly
    confidence_threshold: float = 0.6  # Minimum confidence for prediction
    
    # Agreement requirements
    min_model_agreement: float = 0.5  # Minimum agreement between models
    require_consensus: bool = False  # Require majority consensus
    
    # Dynamic weighting parameters
    performance_window: int = 100  # Number of recent predictions for performance calculation
    adaptation_rate: float = 0.1   # Rate of weight adaptation
    
    # Severity thresholds
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.9,
        'high': 0.8,
        'medium': 0.7,
        'low': 0.6
    })


class EnsembleScorer:
    """
    Ensemble scoring system for combining multiple anomaly detection models.
    
    Combines predictions from Isolation Forest, LSTM, and Prophet models using
    various ensemble methods to improve accuracy and reduce false positives.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize ensemble scorer.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        
        # Model instances
        self.isolation_forest = None
        self.lstm_detector = None
        self.prophet_forecaster = None
        
        # Performance tracking
        self.model_performance_history = {
            'isolation_forest': [],
            'lstm': [],
            'prophet': []
        }
        
        # Dynamic weights (start with configured weights)
        self.current_weights = {
            'isolation_forest': self.config.weights.isolation_forest,
            'lstm': self.config.weights.lstm,
            'prophet': self.config.weights.prophet
        }
        
        # Ensemble history
        self.prediction_history = []
        self.performance_metrics = {}
        
    def set_models(self, 
                   isolation_forest: Optional[IsolationForestDetector] = None,
                   lstm_detector: Optional[LSTMAnomalyDetector] = None,
                   prophet_forecaster: Optional[ProphetForecaster] = None):
        """
        Set the individual models for ensemble.
        
        Args:
            isolation_forest: Trained Isolation Forest detector
            lstm_detector: Trained LSTM anomaly detector
            prophet_forecaster: Trained Prophet forecaster
        """
        self.isolation_forest = isolation_forest
        self.lstm_detector = lstm_detector
        self.prophet_forecaster = prophet_forecaster
        
        logger.info(
            "Ensemble models configured",
            isolation_forest=isolation_forest is not None,
            lstm=lstm_detector is not None,
            prophet=prophet_forecaster is not None
        )
    
    def predict_ensemble(self, data: pd.DataFrame) -> List[EnsembleResult]:
        """
        Generate ensemble predictions combining all available models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            List of ensemble prediction results
        """
        logger.info("Generating ensemble predictions", samples=len(data))
        
        try:
            if data.empty:
                return []
            
            # Get predictions from individual models
            model_predictions = self._get_individual_predictions(data)
            
            # Combine predictions using configured method
            if self.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
                ensemble_results = self._weighted_average_ensemble(model_predictions, data)
            elif self.config.method == EnsembleMethod.VOTING:
                ensemble_results = self._voting_ensemble(model_predictions, data)
            elif self.config.method == EnsembleMethod.DYNAMIC_WEIGHTING:
                ensemble_results = self._dynamic_weighting_ensemble(model_predictions, data)
            else:
                # Default to weighted average
                ensemble_results = self._weighted_average_ensemble(model_predictions, data)
            
            # Update prediction history
            self.prediction_history.extend(ensemble_results)
            
            # Trim history to maintain performance
            if len(self.prediction_history) > self.config.performance_window * 2:
                self.prediction_history = self.prediction_history[-self.config.performance_window:]
            
            logger.info(
                "Ensemble predictions completed",
                total_predictions=len(ensemble_results),
                anomalies_detected=sum(1 for r in ensemble_results if r.is_anomaly)
            )
            
            return ensemble_results
            
        except Exception as e:
            logger.error("Ensemble prediction failed", error=str(e))
            raise
    
    def _get_individual_predictions(self, data: pd.DataFrame) -> Dict[str, List[Any]]:
        """Get predictions from all available individual models"""
        predictions = {}
        
        # Isolation Forest predictions
        if self.isolation_forest and self.isolation_forest.is_trained:
            try:
                if_predictions = self.isolation_forest.predict(data)
                predictions['isolation_forest'] = if_predictions
                logger.debug("Isolation Forest predictions obtained", count=len(if_predictions))
            except Exception as e:
                logger.warning("Isolation Forest prediction failed", error=str(e))
                predictions['isolation_forest'] = []
        
        # LSTM predictions
        if self.lstm_detector and self.lstm_detector.is_trained:
            try:
                lstm_predictions = self.lstm_detector.predict(data)
                predictions['lstm'] = lstm_predictions
                logger.debug("LSTM predictions obtained", count=len(lstm_predictions))
            except Exception as e:
                logger.warning("LSTM prediction failed", error=str(e))
                predictions['lstm'] = []
        
        # Prophet predictions (requires forecast comparison)
        if self.prophet_forecaster and self.prophet_forecaster.is_trained:
            try:
                prophet_predictions = self.prophet_forecaster.detect_forecast_deviations(data)
                predictions['prophet'] = prophet_predictions
                logger.debug("Prophet predictions obtained", count=len(prophet_predictions))
            except Exception as e:
                logger.warning("Prophet prediction failed", error=str(e))
                predictions['prophet'] = []
        
        return predictions
    
    def _weighted_average_ensemble(self, 
                                 model_predictions: Dict[str, List[Any]], 
                                 data: pd.DataFrame) -> List[EnsembleResult]:
        """Combine predictions using weighted average"""
        
        results = []
        max_length = max([len(preds) for preds in model_predictions.values()] + [0])
        
        if max_length == 0:
            return results
        
        for i in range(max_length):
            # Get timestamp
            timestamp = data.iloc[min(i, len(data) - 1)].get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Collect scores from available models
            scores = {}
            confidences = {}
            raw_preds = {}
            
            # Isolation Forest
            if 'isolation_forest' in model_predictions and i < len(model_predictions['isolation_forest']):
                if_pred = model_predictions['isolation_forest'][i]
                # Convert decision function score to 0-1 scale
                if_score = max(0, min(1, (-if_pred.anomaly_score + 1) / 2))
                scores['isolation_forest'] = if_score
                confidences['isolation_forest'] = if_pred.confidence
                raw_preds['isolation_forest'] = if_pred
            
            # LSTM
            if 'lstm' in model_predictions and i < len(model_predictions['lstm']):
                lstm_pred = model_predictions['lstm'][i]
                scores['lstm'] = lstm_pred.anomaly_score
                confidences['lstm'] = lstm_pred.confidence
                raw_preds['lstm'] = lstm_pred
            
            # Prophet
            if 'prophet' in model_predictions and i < len(model_predictions['prophet']):
                prophet_pred = model_predictions['prophet'][i]
                # Convert deviation to 0-1 score
                prophet_score = min(1.0, abs(prophet_pred.deviation_percentage) / 100)
                scores['prophet'] = prophet_score
                confidences['prophet'] = prophet_pred.confidence
                raw_preds['prophet'] = prophet_pred
            
            # Calculate weighted ensemble score
            ensemble_score = 0.0
            total_weight = 0.0
            contributing_models = []
            weights_used = {}
            
            for model_name, score in scores.items():
                weight = self.current_weights.get(model_name, 0)
                if weight > 0:
                    ensemble_score += weight * score
                    total_weight += weight
                    contributing_models.append(model_name)
                    weights_used[model_name] = weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            # Calculate ensemble confidence
            ensemble_confidence = 0.0
            if confidences:
                ensemble_confidence = np.mean(list(confidences.values()))
            
            # Calculate model agreement
            model_agreement = self._calculate_model_agreement(scores)
            
            # Determine if anomaly
            is_anomaly = (
                ensemble_score >= self.config.anomaly_threshold and
                ensemble_confidence >= self.config.confidence_threshold and
                (not self.config.require_consensus or model_agreement >= self.config.min_model_agreement)
            )
            
            # Calculate severity
            severity = self._calculate_severity(ensemble_score)
            
            # Calculate consensus strength
            consensus_strength = model_agreement * ensemble_confidence
            
            result = EnsembleResult(
                timestamp=timestamp,
                ensemble_score=ensemble_score,
                confidence=ensemble_confidence,
                is_anomaly=is_anomaly,
                severity=severity,
                isolation_forest_score=scores.get('isolation_forest'),
                lstm_score=scores.get('lstm'),
                prophet_score=scores.get('prophet'),
                model_agreement=model_agreement,
                consensus_strength=consensus_strength,
                contributing_models=contributing_models,
                model_weights_used=weights_used,
                raw_predictions=raw_preds
            )
            
            results.append(result)
        
        return results
    
    def _voting_ensemble(self, 
                        model_predictions: Dict[str, List[Any]], 
                        data: pd.DataFrame) -> List[EnsembleResult]:
        """Combine predictions using majority voting"""
        
        results = []
        max_length = max([len(preds) for preds in model_predictions.values()] + [0])
        
        for i in range(max_length):
            timestamp = data.iloc[min(i, len(data) - 1)].get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Collect votes from models
            votes = {}
            scores = {}
            confidences = {}
            raw_preds = {}
            
            # Isolation Forest vote
            if 'isolation_forest' in model_predictions and i < len(model_predictions['isolation_forest']):
                if_pred = model_predictions['isolation_forest'][i]
                votes['isolation_forest'] = if_pred.is_anomaly
                scores['isolation_forest'] = max(0, min(1, (-if_pred.anomaly_score + 1) / 2))
                confidences['isolation_forest'] = if_pred.confidence
                raw_preds['isolation_forest'] = if_pred
            
            # LSTM vote
            if 'lstm' in model_predictions and i < len(model_predictions['lstm']):
                lstm_pred = model_predictions['lstm'][i]
                votes['lstm'] = lstm_pred.is_anomaly
                scores['lstm'] = lstm_pred.anomaly_score
                confidences['lstm'] = lstm_pred.confidence
                raw_preds['lstm'] = lstm_pred
            
            # Prophet vote
            if 'prophet' in model_predictions and i < len(model_predictions['prophet']):
                prophet_pred = model_predictions['prophet'][i]
                votes['prophet'] = prophet_pred.is_anomaly
                scores['prophet'] = min(1.0, abs(prophet_pred.deviation_percentage) / 100)
                confidences['prophet'] = prophet_pred.confidence
                raw_preds['prophet'] = prophet_pred
            
            # Count votes
            anomaly_votes = sum(votes.values())
            total_votes = len(votes)
            
            # Majority decision
            is_anomaly = anomaly_votes > (total_votes / 2) if total_votes > 0 else False
            
            # Calculate ensemble score as average of contributing scores
            ensemble_score = np.mean(list(scores.values())) if scores else 0.0
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(list(confidences.values())) if confidences else 0.0
            
            # Calculate model agreement
            model_agreement = self._calculate_vote_agreement(votes)
            
            # Calculate severity
            severity = self._calculate_severity(ensemble_score)
            
            result = EnsembleResult(
                timestamp=timestamp,
                ensemble_score=ensemble_score,
                confidence=ensemble_confidence,
                is_anomaly=is_anomaly,
                severity=severity,
                isolation_forest_score=scores.get('isolation_forest'),
                lstm_score=scores.get('lstm'),
                prophet_score=scores.get('prophet'),
                model_agreement=model_agreement,
                consensus_strength=model_agreement * ensemble_confidence,
                contributing_models=list(votes.keys()),
                model_weights_used={model: 1.0 for model in votes.keys()},
                raw_predictions=raw_preds
            )
            
            results.append(result)
        
        return results
    
    def _dynamic_weighting_ensemble(self, 
                                  model_predictions: Dict[str, List[Any]], 
                                  data: pd.DataFrame) -> List[EnsembleResult]:
        """Combine predictions using dynamic weighting based on recent performance"""
        
        # Update weights based on recent performance
        self._update_dynamic_weights()
        
        # Use weighted average with updated weights
        return self._weighted_average_ensemble(model_predictions, data)
    
    def _calculate_model_agreement(self, scores: Dict[str, float]) -> float:
        """Calculate agreement between model scores"""
        if len(scores) < 2:
            return 1.0
        
        score_values = list(scores.values())
        
        # Calculate pairwise agreements
        agreements = []
        for i in range(len(score_values)):
            for j in range(i + 1, len(score_values)):
                # Agreement based on score similarity
                diff = abs(score_values[i] - score_values[j])
                agreement = 1.0 - diff  # Higher agreement for smaller differences
                agreements.append(max(0, agreement))
        
        return np.mean(agreements) if agreements else 1.0
    
    def _calculate_vote_agreement(self, votes: Dict[str, bool]) -> float:
        """Calculate agreement between model votes"""
        if len(votes) < 2:
            return 1.0
        
        vote_values = list(votes.values())
        
        # Calculate percentage of models that agree with majority
        majority_vote = sum(vote_values) > (len(vote_values) / 2)
        agreements = [vote == majority_vote for vote in vote_values]
        
        return sum(agreements) / len(agreements)
    
    def _calculate_severity(self, ensemble_score: float) -> str:
        """Calculate severity based on ensemble score"""
        thresholds = self.config.severity_thresholds
        
        if ensemble_score >= thresholds['critical']:
            return 'critical'
        elif ensemble_score >= thresholds['high']:
            return 'high'
        elif ensemble_score >= thresholds['medium']:
            return 'medium'
        elif ensemble_score >= thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def _update_dynamic_weights(self):
        """Update model weights based on recent performance"""
        if len(self.prediction_history) < 10:
            return  # Need sufficient history
        
        # Calculate recent performance for each model
        recent_predictions = self.prediction_history[-self.config.performance_window:]
        
        model_accuracies = {}
        
        for model_name in ['isolation_forest', 'lstm', 'prophet']:
            # Calculate accuracy based on consensus with other models
            correct_predictions = 0
            total_predictions = 0
            
            for pred in recent_predictions:
                if model_name in pred.contributing_models:
                    # Check if this model's prediction agrees with ensemble decision
                    model_score = getattr(pred, f"{model_name}_score", None)
                    if model_score is not None:
                        model_anomaly = model_score >= self.config.anomaly_threshold
                        if model_anomaly == pred.is_anomaly:
                            correct_predictions += 1
                        total_predictions += 1
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                model_accuracies[model_name] = accuracy
            else:
                model_accuracies[model_name] = self.current_weights[model_name]
        
        # Update weights based on performance
        if model_accuracies:
            total_accuracy = sum(model_accuracies.values())
            
            if total_accuracy > 0:
                for model_name in self.current_weights:
                    if model_name in model_accuracies:
                        new_weight = model_accuracies[model_name] / total_accuracy
                        
                        # Smooth weight updates
                        self.current_weights[model_name] = (
                            (1 - self.config.adaptation_rate) * self.current_weights[model_name] +
                            self.config.adaptation_rate * new_weight
                        )
        
        logger.debug("Dynamic weights updated", weights=self.current_weights)
    
    def evaluate_ensemble_performance(self, 
                                    ground_truth: List[bool],
                                    predictions: List[EnsembleResult]) -> Dict[str, float]:
        """
        Evaluate ensemble performance against ground truth.
        
        Args:
            ground_truth: List of true anomaly labels
            predictions: List of ensemble predictions
            
        Returns:
            Performance metrics
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have same length")
        
        # Calculate metrics
        true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) 
                           if gt and pred.is_anomaly)
        false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) 
                            if not gt and pred.is_anomaly)
        true_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) 
                           if not gt and not pred.is_anomaly)
        false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) 
                            if gt and not pred.is_anomaly)
        
        # Calculate performance metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(ground_truth)
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Calculate ensemble-specific metrics
        avg_confidence = np.mean([pred.confidence for pred in predictions])
        avg_agreement = np.mean([pred.model_agreement for pred in predictions])
        avg_consensus_strength = np.mean([pred.consensus_strength for pred in predictions])
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'false_positive_rate': false_positive_rate,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'avg_confidence': avg_confidence,
            'avg_model_agreement': avg_agreement,
            'avg_consensus_strength': avg_consensus_strength
        }
        
        # Store performance metrics
        self.performance_metrics = metrics
        
        logger.info(
            "Ensemble performance evaluated",
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy
        )
        
        return metrics
    
    def get_model_contributions(self, predictions: List[EnsembleResult]) -> Dict[str, Any]:
        """Analyze individual model contributions to ensemble decisions"""
        
        contributions = {
            'isolation_forest': {'anomalies': 0, 'total': 0, 'avg_score': 0},
            'lstm': {'anomalies': 0, 'total': 0, 'avg_score': 0},
            'prophet': {'anomalies': 0, 'total': 0, 'avg_score': 0}
        }
        
        for pred in predictions:
            for model in contributions:
                score = getattr(pred, f"{model}_score", None)
                if score is not None:
                    contributions[model]['total'] += 1
                    contributions[model]['avg_score'] += score
                    
                    if model in pred.contributing_models and pred.is_anomaly:
                        contributions[model]['anomalies'] += 1
        
        # Calculate averages
        for model in contributions:
            if contributions[model]['total'] > 0:
                contributions[model]['avg_score'] /= contributions[model]['total']
                contributions[model]['anomaly_rate'] = contributions[model]['anomalies'] / contributions[model]['total']
            else:
                contributions[model]['anomaly_rate'] = 0
        
        return contributions
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information"""
        return {
            'ensemble_method': self.config.method.value,
            'current_weights': self.current_weights,
            'config': {
                'anomaly_threshold': self.config.anomaly_threshold,
                'confidence_threshold': self.config.confidence_threshold,
                'min_model_agreement': self.config.min_model_agreement,
                'require_consensus': self.config.require_consensus,
                'severity_thresholds': self.config.severity_thresholds
            },
            'models_available': {
                'isolation_forest': self.isolation_forest is not None and self.isolation_forest.is_trained,
                'lstm': self.lstm_detector is not None and self.lstm_detector.is_trained,
                'prophet': self.prophet_forecaster is not None and self.prophet_forecaster.is_trained
            },
            'prediction_history_count': len(self.prediction_history),
            'performance_metrics': self.performance_metrics
        }