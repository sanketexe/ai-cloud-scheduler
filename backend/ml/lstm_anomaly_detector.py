"""
LSTM Anomaly Detector for Time Series Pattern Analysis

Implements LSTM neural network for detecting sequential anomalies in time series cost data.
Specialized for identifying unusual temporal patterns and trend deviations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
import json

# TensorFlow/Keras imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # Fallback for environments without TensorFlow
    TENSORFLOW_AVAILABLE = False
    tf = None

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logger = structlog.get_logger(__name__)


@dataclass
class LSTMPrediction:
    """LSTM prediction result"""
    timestamp: datetime
    predicted_value: float
    actual_value: Optional[float]
    reconstruction_error: float
    anomaly_score: float  # 0 to 1, higher = more anomalous
    confidence: float     # 0 to 1, confidence in prediction
    is_anomaly: bool
    severity: str  # 'low', 'medium', 'high', 'critical'
    sequence_context: List[float]  # Input sequence that led to prediction


@dataclass
class LSTMModelConfig:
    """LSTM model configuration"""
    sequence_length: int = 24  # Hours of history to consider
    lstm_units: int = 50
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    anomaly_threshold: float = 0.95  # Percentile threshold for anomaly detection


class LSTMAnomalyDetector:
    """
    LSTM-based anomaly detector for time series cost data.
    
    Uses autoencoder architecture to learn normal patterns in sequential cost data
    and detect deviations from expected temporal behavior. Effective for identifying
    gradual changes, trend shifts, and seasonal pattern violations.
    """
    
    def __init__(self, config: LSTMModelConfig = None):
        """
        Initialize LSTM anomaly detector.
        
        Args:
            config: Model configuration parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM anomaly detection. "
                "Install with: pip install tensorflow"
            )
        
        self.config = config or LSTMModelConfig()
        
        # Model components
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = MinMaxScaler()
        
        # Training state
        self.is_trained = False
        self.feature_names = []
        self.model_version = "1.0.0"
        
        # Performance tracking
        self.training_history = []
        self.validation_losses = []
        self.anomaly_threshold_value = None
        
        # Anomaly detection parameters
        self.severity_thresholds = {
            'critical': 0.99,  # 99th percentile
            'high': 0.95,      # 95th percentile
            'medium': 0.90,    # 90th percentile
            'low': 0.80        # 80th percentile
        }
        
        # Set TensorFlow logging level
        if TENSORFLOW_AVAILABLE:
            tf.get_logger().setLevel('ERROR')
    
    def _build_autoencoder_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build LSTM autoencoder model for anomaly detection.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM autoencoder model", input_shape=input_shape)
        
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # Encoder
        encoded = LSTM(
            self.config.lstm_units, 
            activation='relu', 
            return_sequences=False,
            name='encoder_lstm'
        )(input_layer)
        encoded = Dropout(self.config.dropout_rate)(encoded)
        
        # Bottleneck (compressed representation)
        bottleneck = Dense(self.config.lstm_units // 2, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = RepeatVector(input_shape[0])(bottleneck)
        decoded = LSTM(
            self.config.lstm_units, 
            activation='relu', 
            return_sequences=True,
            name='decoder_lstm'
        )(decoded)
        decoded = Dropout(self.config.dropout_rate)(decoded)
        
        # Output layer
        output_layer = TimeDistributed(
            Dense(input_shape[1], activation='linear'),
            name='output'
        )(decoded)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer, name='lstm_autoencoder')
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data into sequences for LSTM training.
        
        Args:
            data: Time series data array
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.sequence_length):
            # Input sequence
            seq = data[i:i + self.config.sequence_length]
            sequences.append(seq)
            
            # Target sequence (same as input for autoencoder)
            targets.append(seq)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, 
              training_data: pd.DataFrame,
              target_column: str = 'cost_amount',
              feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train LSTM autoencoder on historical time series data.
        
        Args:
            training_data: Historical time series data
            target_column: Column to use as primary target
            feature_columns: Additional feature columns to include
            
        Returns:
            Training metrics and history
        """
        logger.info("Training LSTM anomaly detector", samples=len(training_data))
        
        try:
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            # Prepare features
            if feature_columns is None:
                # Use numeric columns
                numeric_cols = training_data.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
            
            # Ensure target column is included
            all_columns = [target_column] + [col for col in feature_columns if col != target_column]
            self.feature_names = all_columns
            
            # Sort by timestamp if available
            if 'timestamp' in training_data.columns:
                training_data = training_data.sort_values('timestamp')
            
            # Extract and prepare data
            data = training_data[self.feature_names].values
            
            # Handle missing values
            data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._prepare_sequences(data_scaled)
            
            if len(X) < self.config.batch_size:
                raise ValueError(f"Insufficient data for training. Need at least {self.config.batch_size} sequences")
            
            logger.info(
                "Prepared training sequences",
                sequences=len(X),
                sequence_length=self.config.sequence_length,
                features=len(self.feature_names)
            )
            
            # Build model
            input_shape = (self.config.sequence_length, len(self.feature_names))
            self.model = self._build_autoencoder_model(input_shape)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            logger.info("Starting LSTM model training")
            history = self.model.fit(
                X, y,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Don't shuffle time series data
            )
            
            # Calculate reconstruction errors for threshold determination
            predictions = self.model.predict(X, verbose=0)
            reconstruction_errors = np.mean(np.square(X - predictions), axis=(1, 2))
            
            # Set anomaly threshold
            self.anomaly_threshold_value = np.percentile(
                reconstruction_errors, 
                self.config.anomaly_threshold * 100
            )
            
            # Store training history
            training_metrics = {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'min_val_loss': float(min(history.history['val_loss'])),
                'epochs_trained': len(history.history['loss']),
                'anomaly_threshold': float(self.anomaly_threshold_value),
                'training_samples': len(X),
                'sequence_length': self.config.sequence_length,
                'features': len(self.feature_names)
            }
            
            self.training_history.append({
                'timestamp': datetime.utcnow(),
                'metrics': training_metrics,
                'config': self.config.__dict__
            })
            
            self.validation_losses = history.history['val_loss']
            self.is_trained = True
            
            logger.info(
                "LSTM training completed",
                final_loss=training_metrics['final_loss'],
                final_val_loss=training_metrics['final_val_loss'],
                epochs=training_metrics['epochs_trained']
            )
            
            return training_metrics
            
        except Exception as e:
            logger.error("LSTM training failed", error=str(e))
            raise
    
    def predict(self, data: pd.DataFrame) -> List[LSTMPrediction]:
        """
        Predict anomalies in time series data.
        
        Args:
            data: Time series data for prediction
            
        Returns:
            List of LSTM predictions with anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        logger.info("Predicting with LSTM model", samples=len(data))
        
        try:
            if data.empty:
                return []
            
            # Sort by timestamp if available
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
            
            # Prepare data
            input_data = data[self.feature_names].values
            input_data = pd.DataFrame(input_data).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale data
            data_scaled = self.scaler.transform(input_data)
            
            # Create sequences
            X, _ = self._prepare_sequences(data_scaled)
            
            if len(X) == 0:
                logger.warning("Insufficient data for sequence creation")
                return []
            
            # Get predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(X - predictions), axis=(1, 2))
            
            # Convert to anomaly scores and predictions
            results = []
            
            for i, error in enumerate(reconstruction_errors):
                # Calculate anomaly score (0-1, higher = more anomalous)
                anomaly_score = min(error / (self.anomaly_threshold_value + 1e-8), 1.0)
                
                # Determine if anomaly
                is_anomaly = error > self.anomaly_threshold_value
                
                # Calculate severity
                severity = self._calculate_severity(anomaly_score)
                
                # Calculate confidence (inverse of uncertainty)
                confidence = 1.0 - min(error / (self.anomaly_threshold_value * 2 + 1e-8), 1.0)
                
                # Get timestamp
                sequence_start_idx = i
                sequence_end_idx = i + self.config.sequence_length
                
                if sequence_end_idx < len(data):
                    timestamp = data.iloc[sequence_end_idx].get('timestamp', datetime.utcnow())
                    if isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp)
                else:
                    timestamp = datetime.utcnow()
                
                # Get predicted and actual values (use last value in sequence)
                predicted_value = float(predictions[i, -1, 0])  # Last timestep, first feature
                actual_value = float(X[i, -1, 0]) if len(X[i]) > 0 else None
                
                # Get sequence context
                sequence_context = X[i, :, 0].tolist()  # First feature across time
                
                prediction = LSTMPrediction(
                    timestamp=timestamp,
                    predicted_value=predicted_value,
                    actual_value=actual_value,
                    reconstruction_error=float(error),
                    anomaly_score=float(anomaly_score),
                    confidence=float(confidence),
                    is_anomaly=is_anomaly,
                    severity=severity,
                    sequence_context=sequence_context
                )
                
                results.append(prediction)
            
            logger.info(
                "LSTM prediction completed",
                total_predictions=len(results),
                anomalies_detected=sum(1 for r in results if r.is_anomaly)
            )
            
            return results
            
        except Exception as e:
            logger.error("LSTM prediction failed", error=str(e))
            raise
    
    def predict_next_values(self, 
                           recent_data: pd.DataFrame, 
                           steps_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Predict future values based on recent data.
        
        Args:
            recent_data: Recent time series data
            steps_ahead: Number of time steps to predict ahead
            
        Returns:
            List of future predictions with confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        logger.info("Predicting future values", steps=steps_ahead)
        
        try:
            # Prepare recent data
            input_data = recent_data[self.feature_names].values
            input_data = pd.DataFrame(input_data).fillna(method='ffill').fillna(method='bfill').values
            
            # Take last sequence_length points
            if len(input_data) < self.config.sequence_length:
                raise ValueError(f"Need at least {self.config.sequence_length} data points for prediction")
            
            last_sequence = input_data[-self.config.sequence_length:]
            
            # Scale data
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Predict future values
            future_predictions = []
            current_sequence = last_sequence_scaled.copy()
            
            for step in range(steps_ahead):
                # Reshape for model input
                model_input = current_sequence.reshape(1, self.config.sequence_length, len(self.feature_names))
                
                # Get prediction
                prediction = self.model.predict(model_input, verbose=0)
                
                # Extract next value (last timestep of prediction)
                next_value = prediction[0, -1, :]
                
                # Calculate prediction confidence (based on reconstruction error)
                reconstruction_error = np.mean(np.square(model_input - prediction))
                confidence = 1.0 - min(reconstruction_error / (self.anomaly_threshold_value + 1e-8), 1.0)
                
                # Inverse transform to original scale
                next_value_original = self.scaler.inverse_transform(
                    next_value.reshape(1, -1)
                )[0]
                
                # Store prediction
                future_predictions.append({
                    'step': step + 1,
                    'predicted_values': next_value_original.tolist(),
                    'confidence': float(confidence),
                    'reconstruction_error': float(reconstruction_error),
                    'feature_names': self.feature_names
                })
                
                # Update sequence for next prediction (shift and append)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_value
            
            logger.info("Future value prediction completed", predictions=len(future_predictions))
            return future_predictions
            
        except Exception as e:
            logger.error("Future value prediction failed", error=str(e))
            raise
    
    def detect_sequence_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in sequential patterns.
        
        Args:
            data: Time series data
            
        Returns:
            List of sequence-level anomaly detections
        """
        predictions = self.predict(data)
        
        # Group predictions into sequences and analyze patterns
        sequence_anomalies = []
        
        # Look for patterns in consecutive anomalies
        anomaly_sequences = []
        current_sequence = []
        
        for pred in predictions:
            if pred.is_anomaly:
                current_sequence.append(pred)
            else:
                if len(current_sequence) > 0:
                    anomaly_sequences.append(current_sequence)
                    current_sequence = []
        
        # Add final sequence if it exists
        if len(current_sequence) > 0:
            anomaly_sequences.append(current_sequence)
        
        # Analyze each anomaly sequence
        for i, seq in enumerate(anomaly_sequences):
            if len(seq) > 1:  # Multi-point anomaly sequence
                avg_score = np.mean([p.anomaly_score for p in seq])
                max_score = max([p.anomaly_score for p in seq])
                duration = len(seq)
                
                sequence_anomalies.append({
                    'sequence_id': i,
                    'start_timestamp': seq[0].timestamp,
                    'end_timestamp': seq[-1].timestamp,
                    'duration': duration,
                    'average_anomaly_score': float(avg_score),
                    'max_anomaly_score': float(max_score),
                    'severity': seq[0].severity,  # Use first point's severity
                    'pattern_type': 'sustained_anomaly',
                    'predictions': seq
                })
        
        return sequence_anomalies
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate anomaly severity based on score"""
        if anomaly_score >= self.severity_thresholds['critical']:
            return 'critical'
        elif anomaly_score >= self.severity_thresholds['high']:
            return 'high'
        elif anomaly_score >= self.severity_thresholds['medium']:
            return 'medium'
        elif anomaly_score >= self.severity_thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            # Save Keras model
            model_path = f"{filepath}_model.h5"
            self.model.save(model_path)
            
            # Save other components
            components = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config.__dict__,
                'model_version': self.model_version,
                'training_history': self.training_history,
                'validation_losses': self.validation_losses,
                'anomaly_threshold_value': self.anomaly_threshold_value,
                'severity_thresholds': self.severity_thresholds
            }
            
            components_path = f"{filepath}_components.pkl"
            joblib.dump(components, components_path)
            
            logger.info("LSTM model saved successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to save LSTM model", filepath=filepath, error=str(e))
            raise
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            # Load Keras model
            model_path = f"{filepath}_model.h5"
            self.model = tf.keras.models.load_model(model_path)
            
            # Load other components
            components_path = f"{filepath}_components.pkl"
            components = joblib.load(components_path)
            
            self.scaler = components['scaler']
            self.feature_names = components['feature_names']
            self.config = LSTMModelConfig(**components['config'])
            self.model_version = components['model_version']
            self.training_history = components['training_history']
            self.validation_losses = components['validation_losses']
            self.anomaly_threshold_value = components['anomaly_threshold_value']
            self.severity_thresholds = components['severity_thresholds']
            
            self.is_trained = True
            
            logger.info("LSTM model loaded successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to load LSTM model", filepath=filepath, error=str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_type': 'LSTM_Autoencoder',
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'config': self.config.__dict__,
            'anomaly_threshold': self.anomaly_threshold_value,
            'severity_thresholds': self.severity_thresholds,
            'training_history_count': len(self.training_history)
        }
        
        if self.is_trained and self.model:
            info['model_summary'] = {
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'layers': len(self.model.layers)
            }
        
        return info