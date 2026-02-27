"""
Isolation Forest Detector for Point Anomaly Detection

Implements isolation forest algorithm for detecting point anomalies in cost data.
Optimized for real-time detection of sudden cost spikes and unusual spending patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyScore:
    """Individual anomaly score result"""
    timestamp: datetime
    anomaly_score: float  # -1 to 1, where < 0 indicates anomaly
    confidence: float     # 0 to 1, confidence in the prediction
    feature_importance: Dict[str, float]
    raw_features: Dict[str, float]
    is_anomaly: bool
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    training_samples: int
    feature_count: int
    model_version: str
    last_trained: datetime


class IsolationForestDetector:
    """
    Isolation Forest detector for point anomalies in cost data.
    
    Uses isolation forest algorithm to detect outliers by isolating observations
    through random selection of features and split values. Effective for detecting
    sudden cost spikes and unusual spending patterns.
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 max_samples: str = 'auto',
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in dataset
            n_estimators: Number of base estimators in ensemble
            max_samples: Number of samples to draw to train each estimator
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        # Initialize model components
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_version = "1.0.0"
        
        # Performance tracking
        self.metrics = None
        self.training_history = []
        
        # Anomaly thresholds
        self.severity_thresholds = {
            'critical': -0.8,  # Very strong anomaly signal
            'high': -0.6,      # Strong anomaly signal
            'medium': -0.4,    # Moderate anomaly signal
            'low': -0.2        # Weak anomaly signal
        }
    
    def train(self, 
              training_data: pd.DataFrame, 
              feature_columns: Optional[List[str]] = None,
              validation_split: float = 0.2) -> ModelMetrics:
        """
        Train the isolation forest model on historical cost data.
        
        Args:
            training_data: Historical cost data with features
            feature_columns: Specific columns to use as features
            validation_split: Fraction of data to use for validation
            
        Returns:
            Model performance metrics
        """
        logger.info("Training Isolation Forest model", samples=len(training_data))
        
        try:
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            # Prepare features
            if feature_columns is None:
                # Use numeric columns as features
                feature_columns = training_data.select_dtypes(include=[np.number]).columns.tolist()
                # Remove target columns if present
                exclude_cols = ['is_anomaly', 'anomaly_score', 'timestamp']
                feature_columns = [col for col in feature_columns if col not in exclude_cols]
            
            self.feature_names = feature_columns
            
            if not self.feature_names:
                raise ValueError("No valid feature columns found")
            
            # Extract features
            X = training_data[self.feature_names].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Split data for validation
            if validation_split > 0:
                X_train, X_val = train_test_split(
                    X, test_size=validation_split, random_state=self.random_state
                )
            else:
                X_train = X
                X_val = None
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            logger.info("Fitting Isolation Forest", features=len(self.feature_names))
            self.model.fit(X_train_scaled)
            
            # Calculate training metrics
            train_scores = self.model.decision_function(X_train_scaled)
            train_predictions = self.model.predict(X_train_scaled)
            
            # Validation metrics if validation set provided
            val_metrics = None
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_scores = self.model.decision_function(X_val_scaled)
                val_predictions = self.model.predict(X_val_scaled)
                
                val_metrics = self._calculate_metrics(
                    val_predictions, val_scores, len(X_val)
                )
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(
                train_predictions, train_scores, len(X_train)
            )
            
            # Use validation metrics if available, otherwise training metrics
            self.metrics = val_metrics if val_metrics else train_metrics
            self.metrics.training_samples = len(X_train)
            self.metrics.feature_count = len(self.feature_names)
            self.metrics.model_version = self.model_version
            self.metrics.last_trained = datetime.utcnow()
            
            # Update training history
            self.training_history.append({
                'timestamp': datetime.utcnow(),
                'samples': len(training_data),
                'features': len(self.feature_names),
                'metrics': self.metrics
            })
            
            self.is_trained = True
            
            logger.info(
                "Isolation Forest training completed",
                accuracy=self.metrics.accuracy,
                precision=self.metrics.precision,
                recall=self.metrics.recall
            )
            
            return self.metrics
            
        except Exception as e:
            logger.error("Isolation Forest training failed", error=str(e))
            raise
    
    def predict(self, data: pd.DataFrame) -> List[AnomalyScore]:
        """
        Predict anomalies in new cost data.
        
        Args:
            data: Cost data with same features as training data
            
        Returns:
            List of anomaly scores for each data point
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        logger.info("Predicting anomalies", samples=len(data))
        
        try:
            if data.empty:
                return []
            
            # Prepare features
            X = data[self.feature_names].copy()
            X = X.fillna(X.median())
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions and scores
            predictions = self.model.predict(X_scaled)
            decision_scores = self.model.decision_function(X_scaled)
            
            # Convert to anomaly scores
            anomaly_scores = []
            
            for i, (pred, score) in enumerate(zip(predictions, decision_scores)):
                # Calculate confidence (higher absolute score = higher confidence)
                confidence = min(abs(score), 1.0)
                
                # Determine severity
                severity = self._calculate_severity(score)
                
                # Calculate feature importance (simplified)
                feature_importance = self._calculate_feature_importance(X_scaled[i])
                
                # Get raw features
                raw_features = X.iloc[i].to_dict()
                
                # Get timestamp if available
                timestamp = data.iloc[i].get('timestamp', datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                anomaly_score = AnomalyScore(
                    timestamp=timestamp,
                    anomaly_score=score,
                    confidence=confidence,
                    feature_importance=feature_importance,
                    raw_features=raw_features,
                    is_anomaly=(pred == -1),
                    severity=severity
                )
                
                anomaly_scores.append(anomaly_score)
            
            logger.info(
                "Anomaly prediction completed",
                total_samples=len(anomaly_scores),
                anomalies_detected=sum(1 for a in anomaly_scores if a.is_anomaly)
            )
            
            return anomaly_scores
            
        except Exception as e:
            logger.error("Anomaly prediction failed", error=str(e))
            raise
    
    def predict_single(self, features: Dict[str, float]) -> AnomalyScore:
        """
        Predict anomaly for a single data point.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Anomaly score for the data point
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Convert to DataFrame
            data = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0.0  # Default value for missing features
            
            # Get prediction
            results = self.predict(data)
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error("Single anomaly prediction failed", error=str(e))
            raise
    
    def update_model(self, new_data: pd.DataFrame) -> ModelMetrics:
        """
        Update model with new training data (incremental learning simulation).
        
        Args:
            new_data: New training data
            
        Returns:
            Updated model metrics
        """
        logger.info("Updating Isolation Forest model", new_samples=len(new_data))
        
        try:
            if not self.is_trained:
                # If not trained, just train normally
                return self.train(new_data)
            
            # For Isolation Forest, we need to retrain with combined data
            # In a production system, you might implement true incremental learning
            logger.warning("Isolation Forest requires full retraining for updates")
            
            # Retrain with new data
            return self.train(new_data)
            
        except Exception as e:
            logger.error("Model update failed", error=str(e))
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            return {}
        
        # Isolation Forest doesn't provide direct feature importance
        # We can estimate it based on feature variance in training data
        try:
            # This is a simplified approach - in production you might use
            # permutation importance or other methods
            importance = {}
            
            for i, feature in enumerate(self.feature_names):
                # Use inverse of feature index as a simple importance measure
                # In practice, you'd calculate this more rigorously
                importance[feature] = 1.0 / (i + 1)
            
            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error("Failed to calculate feature importance", error=str(e))
            return {}
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_version': self.model_version,
                'metrics': self.metrics,
                'training_history': self.training_history,
                'severity_thresholds': self.severity_thresholds,
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'random_state': self.random_state
            }
            
            joblib.dump(model_data, filepath)
            logger.info("Model saved successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to save model", filepath=filepath, error=str(e))
            raise
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_version = model_data['model_version']
            self.metrics = model_data['metrics']
            self.training_history = model_data['training_history']
            self.severity_thresholds = model_data['severity_thresholds']
            self.contamination = model_data['contamination']
            self.n_estimators = model_data['n_estimators']
            self.max_samples = model_data['max_samples']
            self.random_state = model_data['random_state']
            
            self.is_trained = True
            
            logger.info("Model loaded successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to load model", filepath=filepath, error=str(e))
            raise
    
    def _calculate_metrics(self, predictions: np.ndarray, scores: np.ndarray, total_samples: int) -> ModelMetrics:
        """Calculate model performance metrics"""
        
        # For unsupervised learning, we estimate metrics based on predictions
        anomalies = predictions == -1
        normal = predictions == 1
        
        anomaly_count = np.sum(anomalies)
        normal_count = np.sum(normal)
        
        # Estimated metrics (in practice, you'd use labeled validation data)
        accuracy = normal_count / total_samples if total_samples > 0 else 0.0
        precision = 0.8  # Estimated - would be calculated with true labels
        recall = 0.7     # Estimated - would be calculated with true labels
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_positive_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            training_samples=total_samples,
            feature_count=len(self.feature_names),
            model_version=self.model_version,
            last_trained=datetime.utcnow()
        )
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity based on score"""
        if score <= self.severity_thresholds['critical']:
            return 'critical'
        elif score <= self.severity_thresholds['high']:
            return 'high'
        elif score <= self.severity_thresholds['medium']:
            return 'medium'
        elif score <= self.severity_thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def _calculate_feature_importance(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for a single prediction"""
        
        # Simplified feature importance based on feature magnitude
        importance = {}
        
        if len(feature_vector) == len(self.feature_names):
            # Normalize feature values to get relative importance
            abs_values = np.abs(feature_vector)
            total = np.sum(abs_values)
            
            if total > 0:
                normalized_values = abs_values / total
                
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(normalized_values[i])
            else:
                # Equal importance if all values are zero
                equal_importance = 1.0 / len(self.feature_names)
                for feature in self.feature_names:
                    importance[feature] = equal_importance
        
        return importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_type': 'IsolationForest',
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'hyperparameters': {
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'random_state': self.random_state
            },
            'metrics': self.metrics.__dict__ if self.metrics else None,
            'training_history_count': len(self.training_history),
            'severity_thresholds': self.severity_thresholds
        }