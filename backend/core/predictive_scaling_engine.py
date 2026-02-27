"""
Predictive Scaling Engine with Time Series Forecasting

This module implements AI-powered predictive resource scaling using:
- LSTM and Transformer models for demand forecasting
- Multi-horizon forecasting (1h, 24h, 7d)
- Safety checks and gradual scaling
- Cross-provider resource optimization
- Seasonal pattern detection and adjustment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

from .database import get_db_session
from .models import ResourceMetrics, ScalingEvent
from .cloud_providers import CloudProvider
from .safety_checker import SafetyChecker

logger = logging.getLogger(__name__)

class ScalingActionType(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"

class ForecastHorizon(Enum):
    """Forecast time horizons"""
    ONE_HOUR = "1h"
    TWENTY_FOUR_HOURS = "24h"
    SEVEN_DAYS = "7d"

@dataclass
class DemandPoint:
    """Single demand prediction point"""
    timestamp: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    contributing_factors: Dict[str, float]

@dataclass
class DemandForecast:
    """Complete demand forecast result"""
    resource_id: str
    forecast_horizon: ForecastHorizon
    predictions: List[DemandPoint]
    confidence_intervals: List[Tuple[float, float]]
    seasonal_factors: Dict[str, float]
    external_factors: List[str]
    accuracy_score: float
    model_used: str

@dataclass
class ScalingRecommendation:
    """Scaling action recommendation"""
    resource_id: str
    action_type: ScalingActionType
    target_capacity: int
    current_capacity: int
    confidence: float
    reasoning: str
    expected_impact: Dict[str, float]
    execution_time: datetime
    safety_checks_passed: bool

@dataclass
class ScalingResult:
    """Result of scaling action execution"""
    resource_id: str
    action_executed: ScalingActionType
    success: bool
    previous_capacity: int
    new_capacity: int
    execution_time: datetime
    error_message: Optional[str]
    cost_impact: Optional[float]

class LSTMForecaster(nn.Module):
    """LSTM neural network for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class TransformerForecaster(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 3, output_size: int = 1, dropout: float = 0.1):
        super(TransformerForecaster, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, d_model: int, max_len: int = 5000):
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Take the last output and project to output size
        last_output = transformer_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.output_projection(output)
        
        return output

class DemandPredictor:
    """Multi-horizon demand forecasting using LSTM and Transformer models"""
    
    def __init__(self, sequence_length: int = 24, feature_size: int = 10):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.scaler = MinMaxScaler()
        
        # Initialize models
        self.lstm_model = LSTMForecaster(
            input_size=feature_size,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        self.transformer_model = TransformerForecaster(
            input_size=feature_size,
            d_model=64,
            nhead=8,
            num_layers=3,
            output_size=1
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model.to(self.device)
        self.transformer_model.to(self.device)
        
        self.is_trained = False
        self.logger = logging.getLogger(__name__ + ".DemandPredictor")
    
    def prepare_data(self, metrics_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training"""
        
        # Sort by timestamp
        metrics_data = metrics_data.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features
        feature_columns = [
            'cpu_utilization', 'memory_utilization', 'network_in', 'network_out',
            'disk_read', 'disk_write', 'request_count', 'response_time',
            'hour_of_day', 'day_of_week'
        ]
        
        # Create features that exist in the data
        available_features = [col for col in feature_columns if col in metrics_data.columns]
        
        if not available_features:
            # Create basic features from timestamp
            metrics_data['hour_of_day'] = pd.to_datetime(metrics_data['timestamp']).dt.hour
            metrics_data['day_of_week'] = pd.to_datetime(metrics_data['timestamp']).dt.dayofweek
            available_features = ['hour_of_day', 'day_of_week']
        
        # Add time-based features if timestamp exists
        if 'timestamp' in metrics_data.columns:
            timestamps = pd.to_datetime(metrics_data['timestamp'])
            if 'hour_of_day' not in available_features:
                metrics_data['hour_of_day'] = timestamps.dt.hour
                available_features.append('hour_of_day')
            if 'day_of_week' not in available_features:
                metrics_data['day_of_week'] = timestamps.dt.dayofweek
                available_features.append('day_of_week')
        
        # Fill missing values
        for col in available_features:
            metrics_data[col] = metrics_data[col].fillna(metrics_data[col].mean())
        
        # Update feature size to match available features
        self.feature_size = len(available_features)
        
        # Recreate models with correct input size
        self.lstm_model = LSTMForecaster(
            input_size=self.feature_size,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        self.transformer_model = TransformerForecaster(
            input_size=self.feature_size,
            d_model=64,
            nhead=8,
            num_layers=3,
            output_size=1
        )
        
        self.lstm_model.to(self.device)
        self.transformer_model.to(self.device)
        
        # Scale features
        features = metrics_data[available_features].values
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:(i + self.sequence_length)])
            # Predict the first feature (e.g., CPU utilization)
            y.append(features_scaled[i + self.sequence_length, 0])
        
        return np.array(X), np.array(y)
    
    async def train_models(self, resource_id: str, historical_data: pd.DataFrame) -> bool:
        """Train LSTM and Transformer models on historical data"""
        
        try:
            self.logger.info(f"Training demand prediction models for resource {resource_id}")
            
            if historical_data.empty:
                self.logger.warning("No historical data provided for training")
                return False
            
            # Prepare training data
            X, y = self.prepare_data(historical_data)
            
            if len(X) == 0:
                self.logger.warning("Insufficient data for training")
                return False
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train LSTM model
            await self._train_model(self.lstm_model, dataloader, "LSTM")
            
            # Train Transformer model
            await self._train_model(self.transformer_model, dataloader, "Transformer")
            
            self.is_trained = True
            self.logger.info(f"Models trained successfully for resource {resource_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    async def _train_model(self, model: nn.Module, dataloader: DataLoader, model_name: str):
        """Train a specific model"""
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        epochs = 50
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.debug(f"{model_name} Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    async def predict_demand(self, resource_id: str, recent_data: pd.DataFrame, 
                           horizon: ForecastHorizon) -> Optional[DemandForecast]:
        """Generate demand forecast for specified horizon"""
        
        if not self.is_trained:
            self.logger.warning("Models not trained")
            return None
        
        try:
            # Prepare input data
            X, _ = self.prepare_data(recent_data)
            
            if len(X) == 0:
                return None
            
            # Use the last sequence for prediction
            last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
            
            # Get predictions from both models
            self.lstm_model.eval()
            self.transformer_model.eval()
            
            with torch.no_grad():
                lstm_pred = self.lstm_model(last_sequence).cpu().numpy()[0, 0]
                transformer_pred = self.transformer_model(last_sequence).cpu().numpy()[0, 0]
            
            # Ensemble prediction (average of both models)
            ensemble_pred = float((lstm_pred + transformer_pred) / 2)
            
            # Generate forecast points based on horizon
            forecast_points = self._generate_forecast_points(
                ensemble_pred, horizon, recent_data
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                float(lstm_pred), float(transformer_pred), len(forecast_points)
            )
            
            # Detect seasonal factors
            seasonal_factors = self._detect_seasonal_patterns(recent_data)
            
            return DemandForecast(
                resource_id=resource_id,
                forecast_horizon=horizon,
                predictions=forecast_points,
                confidence_intervals=confidence_intervals,
                seasonal_factors=seasonal_factors,
                external_factors=["historical_patterns", "time_of_day", "day_of_week"],
                accuracy_score=0.85,  # This would be calculated from validation data
                model_used="LSTM+Transformer Ensemble"
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting demand: {str(e)}")
            return None
    
    def _generate_forecast_points(self, base_prediction: float, horizon: ForecastHorizon, 
                                recent_data: pd.DataFrame) -> List[DemandPoint]:
        """Generate forecast points for the specified horizon"""
        
        points = []
        
        # Determine number of points based on horizon
        if horizon == ForecastHorizon.ONE_HOUR:
            num_points = 12  # 5-minute intervals
            interval_minutes = 5
        elif horizon == ForecastHorizon.TWENTY_FOUR_HOURS:
            num_points = 24  # Hourly intervals
            interval_minutes = 60
        else:  # 7 days
            num_points = 7  # Daily intervals
            interval_minutes = 1440
        
        base_time = datetime.now()
        
        for i in range(num_points):
            timestamp = base_time + timedelta(minutes=i * interval_minutes)
            
            # Add some variation based on time patterns
            time_factor = self._get_time_factor(timestamp)
            predicted_value = float(base_prediction * time_factor)
            
            # Calculate confidence bounds (±20% for now)
            confidence_range = predicted_value * 0.2
            
            point = DemandPoint(
                timestamp=timestamp,
                predicted_value=predicted_value,
                confidence_lower=predicted_value - confidence_range,
                confidence_upper=predicted_value + confidence_range,
                contributing_factors={
                    "base_prediction": 0.6,
                    "time_pattern": 0.3,
                    "seasonal_adjustment": 0.1
                }
            )
            
            points.append(point)
        
        return points
    
    def _get_time_factor(self, timestamp: datetime) -> float:
        """Get time-based adjustment factor"""
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Business hours adjustment
        if 9 <= hour <= 17 and day_of_week < 5:  # Business hours, weekday
            return 1.2
        elif 22 <= hour or hour <= 6:  # Night hours
            return 0.7
        elif day_of_week >= 5:  # Weekend
            return 0.8
        else:
            return 1.0
    
    def _calculate_confidence_intervals(self, lstm_pred: float, transformer_pred: float, 
                                      num_points: int) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        
        # Use model disagreement as uncertainty measure
        disagreement = abs(lstm_pred - transformer_pred)
        base_uncertainty = max(0.1, disagreement)
        
        intervals = []
        for i in range(num_points):
            # Uncertainty increases with prediction distance
            uncertainty = base_uncertainty * (1 + i * 0.1)
            
            lower = lstm_pred - uncertainty
            upper = transformer_pred + uncertainty
            
            intervals.append((lower, upper))
        
        return intervals
    
    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect seasonal patterns in the data"""
        
        if data.empty or 'timestamp' not in data.columns:
            return {}
        
        try:
            # Convert timestamp to datetime if it's not already
            data = data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Extract time components
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            
            seasonal_factors = {}
            
            # Get first numeric column for analysis
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return seasonal_factors
            
            target_column = numeric_columns[0]
            
            # Calculate hourly patterns
            if len(data) > 24:
                hourly_avg = data.groupby('hour')[target_column].mean()
                if len(hourly_avg) > 0:
                    peak_hour = hourly_avg.idxmax()
                    seasonal_factors['peak_hour'] = float(peak_hour)
                    seasonal_factors['hourly_variation'] = float(hourly_avg.std())
            
            # Calculate weekly patterns
            if len(data) > 7:
                weekly_avg = data.groupby('day_of_week')[target_column].mean()
                if len(weekly_avg) > 0:
                    peak_day = weekly_avg.idxmax()
                    seasonal_factors['peak_day'] = float(peak_day)
                    seasonal_factors['weekly_variation'] = float(weekly_avg.std())
            
            return seasonal_factors
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal patterns: {str(e)}")
            return {}

class ScalingController:
    """Controls scaling actions with safety checks and gradual scaling"""
    
    def __init__(self, safety_checker: SafetyChecker):
        self.safety_checker = safety_checker
        self.logger = logging.getLogger(__name__ + ".ScalingController")
        
        # Scaling parameters
        self.max_scale_up_percent = 50  # Maximum 50% increase at once
        self.max_scale_down_percent = 30  # Maximum 30% decrease at once
        self.min_capacity = 1
        self.cooldown_period = timedelta(minutes=15)
        
        # Track recent scaling actions
        self.recent_actions: Dict[str, datetime] = {}
    
    async def recommend_scaling_action(self, forecast: DemandForecast, 
                                     current_capacity: int) -> ScalingRecommendation:
        """Recommend scaling action based on demand forecast"""
        
        try:
            # Analyze forecast to determine scaling need
            scaling_need = self._analyze_scaling_need(forecast, current_capacity)
            
            if scaling_need['action'] == ScalingActionType.NO_ACTION:
                return ScalingRecommendation(
                    resource_id=forecast.resource_id,
                    action_type=ScalingActionType.NO_ACTION,
                    target_capacity=current_capacity,
                    current_capacity=current_capacity,
                    confidence=scaling_need['confidence'],
                    reasoning="No scaling needed based on forecast",
                    expected_impact={},
                    execution_time=datetime.now(),
                    safety_checks_passed=True
                )
            
            # Calculate target capacity with safety limits
            target_capacity = self._calculate_safe_target_capacity(
                current_capacity, scaling_need
            )
            
            # Perform safety checks
            safety_result = await self.safety_checker.check_scaling_safety(
                forecast.resource_id, scaling_need['action'], target_capacity
            )
            
            return ScalingRecommendation(
                resource_id=forecast.resource_id,
                action_type=scaling_need['action'],
                target_capacity=target_capacity,
                current_capacity=current_capacity,
                confidence=scaling_need['confidence'],
                reasoning=scaling_need['reasoning'],
                expected_impact=scaling_need['expected_impact'],
                execution_time=datetime.now() + timedelta(minutes=5),  # Small delay for preparation
                safety_checks_passed=safety_result.passed
            )
            
        except Exception as e:
            self.logger.error(f"Error recommending scaling action: {str(e)}")
            return ScalingRecommendation(
                resource_id=forecast.resource_id,
                action_type=ScalingActionType.NO_ACTION,
                target_capacity=current_capacity,
                current_capacity=current_capacity,
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                expected_impact={},
                execution_time=datetime.now(),
                safety_checks_passed=False
            )
    
    def _analyze_scaling_need(self, forecast: DemandForecast, 
                            current_capacity: int) -> Dict[str, Any]:
        """Analyze forecast to determine scaling requirements"""
        
        if not forecast.predictions:
            return {
                'action': ScalingActionType.NO_ACTION,
                'confidence': 0.0,
                'reasoning': "No forecast data available",
                'expected_impact': {}
            }
        
        # Calculate average predicted demand
        avg_demand = np.mean([p.predicted_value for p in forecast.predictions])
        max_demand = max([p.predicted_value for p in forecast.predictions])
        min_demand = min([p.predicted_value for p in forecast.predictions])
        
        # Define thresholds (these could be configurable)
        scale_up_threshold = 0.8  # Scale up if demand > 80% of capacity
        scale_down_threshold = 0.4  # Scale down if demand < 40% of capacity
        
        current_utilization = avg_demand / current_capacity if current_capacity > 0 else 0
        max_utilization = max_demand / current_capacity if current_capacity > 0 else 0
        
        # Determine scaling action
        if max_utilization > scale_up_threshold:
            action = ScalingActionType.SCALE_UP
            confidence = min(0.9, max_utilization - scale_up_threshold + 0.5)
            reasoning = f"Peak demand ({max_demand:.2f}) exceeds {scale_up_threshold*100}% of capacity ({current_capacity})"
            
        elif current_utilization < scale_down_threshold:
            action = ScalingActionType.SCALE_DOWN
            confidence = min(0.9, scale_down_threshold - current_utilization + 0.5)
            reasoning = f"Average demand ({avg_demand:.2f}) is below {scale_down_threshold*100}% of capacity ({current_capacity})"
            
        else:
            action = ScalingActionType.NO_ACTION
            confidence = 0.7
            reasoning = f"Demand ({avg_demand:.2f}) is within acceptable range for capacity ({current_capacity})"
        
        # Calculate expected impact
        expected_impact = {
            'cost_change_percent': self._estimate_cost_impact(action, current_capacity),
            'performance_improvement': self._estimate_performance_impact(action),
            'availability_improvement': 0.05 if action == ScalingActionType.SCALE_UP else 0.0
        }
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'expected_impact': expected_impact
        }
    
    def _calculate_safe_target_capacity(self, current_capacity: int, 
                                      scaling_need: Dict[str, Any]) -> int:
        """Calculate safe target capacity with gradual scaling limits"""
        
        action = scaling_need['action']
        
        if action == ScalingActionType.SCALE_UP:
            max_increase = max(1, int(current_capacity * self.max_scale_up_percent / 100))
            target_capacity = current_capacity + max_increase
            
        elif action == ScalingActionType.SCALE_DOWN:
            max_decrease = max(1, int(current_capacity * self.max_scale_down_percent / 100))
            target_capacity = max(self.min_capacity, current_capacity - max_decrease)
            
        else:
            target_capacity = current_capacity
        
        return target_capacity
    
    def _estimate_cost_impact(self, action: ScalingActionType, current_capacity: int) -> float:
        """Estimate cost impact of scaling action"""
        
        if action == ScalingActionType.SCALE_UP:
            return self.max_scale_up_percent  # Positive percentage increase
        elif action == ScalingActionType.SCALE_DOWN:
            return -self.max_scale_down_percent  # Negative percentage decrease
        else:
            return 0.0
    
    def _estimate_performance_impact(self, action: ScalingActionType) -> float:
        """Estimate performance impact of scaling action"""
        
        if action == ScalingActionType.SCALE_UP:
            return 0.2  # 20% performance improvement
        elif action == ScalingActionType.SCALE_DOWN:
            return -0.1  # 10% performance decrease
        else:
            return 0.0
    
    async def execute_scaling(self, recommendation: ScalingRecommendation, 
                            cloud_provider: CloudProvider) -> ScalingResult:
        """Execute scaling action through cloud provider"""
        
        if not recommendation.safety_checks_passed:
            return ScalingResult(
                resource_id=recommendation.resource_id,
                action_executed=ScalingActionType.NO_ACTION,
                success=False,
                previous_capacity=recommendation.current_capacity,
                new_capacity=recommendation.current_capacity,
                execution_time=datetime.now(),
                error_message="Safety checks failed",
                cost_impact=None
            )
        
        # Check cooldown period
        if self._is_in_cooldown(recommendation.resource_id):
            return ScalingResult(
                resource_id=recommendation.resource_id,
                action_executed=ScalingActionType.NO_ACTION,
                success=False,
                previous_capacity=recommendation.current_capacity,
                new_capacity=recommendation.current_capacity,
                execution_time=datetime.now(),
                error_message="Resource is in cooldown period",
                cost_impact=None
            )
        
        try:
            # Execute scaling through cloud provider
            success = await cloud_provider.scale_resource(
                recommendation.resource_id,
                recommendation.target_capacity
            )
            
            if success:
                # Update cooldown tracking
                self.recent_actions[recommendation.resource_id] = datetime.now()
                
                # Calculate cost impact
                cost_impact = self._calculate_actual_cost_impact(
                    recommendation.current_capacity,
                    recommendation.target_capacity
                )
                
                return ScalingResult(
                    resource_id=recommendation.resource_id,
                    action_executed=recommendation.action_type,
                    success=True,
                    previous_capacity=recommendation.current_capacity,
                    new_capacity=recommendation.target_capacity,
                    execution_time=datetime.now(),
                    error_message=None,
                    cost_impact=cost_impact
                )
            else:
                return ScalingResult(
                    resource_id=recommendation.resource_id,
                    action_executed=ScalingActionType.NO_ACTION,
                    success=False,
                    previous_capacity=recommendation.current_capacity,
                    new_capacity=recommendation.current_capacity,
                    execution_time=datetime.now(),
                    error_message="Cloud provider scaling failed",
                    cost_impact=None
                )
                
        except Exception as e:
            self.logger.error(f"Error executing scaling: {str(e)}")
            return ScalingResult(
                resource_id=recommendation.resource_id,
                action_executed=ScalingActionType.NO_ACTION,
                success=False,
                previous_capacity=recommendation.current_capacity,
                new_capacity=recommendation.current_capacity,
                execution_time=datetime.now(),
                error_message=str(e),
                cost_impact=None
            )
    
    def _is_in_cooldown(self, resource_id: str) -> bool:
        """Check if resource is in cooldown period"""
        
        if resource_id not in self.recent_actions:
            return False
        
        last_action = self.recent_actions[resource_id]
        return datetime.now() - last_action < self.cooldown_period
    
    def _calculate_actual_cost_impact(self, previous_capacity: int, new_capacity: int) -> float:
        """Calculate actual cost impact of scaling action"""
        
        if previous_capacity == 0:
            return 0.0
        
        capacity_change = (new_capacity - previous_capacity) / previous_capacity
        
        # Assume linear cost relationship (this could be more sophisticated)
        # Cost per unit capacity (example: $0.10 per hour per unit)
        cost_per_unit_hour = 0.10
        
        # Calculate hourly cost impact
        cost_impact = (new_capacity - previous_capacity) * cost_per_unit_hour
        
        return cost_impact

class ResourceOptimizer:
    """Cross-provider resource allocation optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ResourceOptimizer")
        
        # Provider cost factors (relative to AWS = 1.0)
        self.provider_cost_factors = {
            'aws': 1.0,
            'azure': 0.95,
            'gcp': 0.90,
            'alibaba': 0.85
        }
        
        # Provider performance factors
        self.provider_performance_factors = {
            'aws': 1.0,
            'azure': 0.98,
            'gcp': 1.02,
            'alibaba': 0.95
        }
    
    async def optimize_resource_allocation(self, demand_forecasts: Dict[str, DemandForecast],
                                         available_providers: List[str]) -> Dict[str, Any]:
        """Optimize resource allocation across cloud providers"""
        
        try:
            optimization_results = {}
            
            for resource_id, forecast in demand_forecasts.items():
                # Calculate optimal allocation for this resource
                optimal_allocation = await self._optimize_single_resource(
                    resource_id, forecast, available_providers
                )
                
                optimization_results[resource_id] = optimal_allocation
            
            # Calculate overall optimization summary
            summary = self._calculate_optimization_summary(optimization_results)
            
            return {
                'resource_allocations': optimization_results,
                'optimization_summary': summary,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing resource allocation: {str(e)}")
            return {}
    
    async def _optimize_single_resource(self, resource_id: str, forecast: DemandForecast,
                                      available_providers: List[str]) -> Dict[str, Any]:
        """Optimize allocation for a single resource"""
        
        # Calculate demand statistics
        avg_demand = np.mean([p.predicted_value for p in forecast.predictions])
        max_demand = max([p.predicted_value for p in forecast.predictions])
        
        # Evaluate each provider
        provider_scores = {}
        
        for provider in available_providers:
            score = self._calculate_provider_score(
                provider, avg_demand, max_demand, forecast.seasonal_factors
            )
            provider_scores[provider] = score
        
        # Select best provider
        best_provider = max(provider_scores, key=provider_scores.get)
        
        # Calculate recommended capacity
        recommended_capacity = self._calculate_optimal_capacity(
            avg_demand, max_demand, best_provider
        )
        
        return {
            'resource_id': resource_id,
            'recommended_provider': best_provider,
            'recommended_capacity': recommended_capacity,
            'provider_scores': provider_scores,
            'expected_cost_savings': self._calculate_cost_savings(
                provider_scores, best_provider
            ),
            'confidence': forecast.accuracy_score
        }
    
    def _calculate_provider_score(self, provider: str, avg_demand: float, 
                                max_demand: float, seasonal_factors: Dict[str, float]) -> float:
        """Calculate score for a provider based on cost and performance"""
        
        # Get provider factors
        cost_factor = self.provider_cost_factors.get(provider, 1.0)
        performance_factor = self.provider_performance_factors.get(provider, 1.0)
        
        # Calculate base score (lower cost is better)
        cost_score = 1.0 / cost_factor
        
        # Performance score
        performance_score = performance_factor
        
        # Seasonal adjustment
        seasonal_score = 1.0
        if seasonal_factors.get('weekly_variation', 0) > 0.3:
            # High variation favors providers with better auto-scaling
            if provider in ['aws', 'gcp']:
                seasonal_score = 1.1
        
        # Combine scores (weighted average)
        total_score = (
            cost_score * 0.4 +
            performance_score * 0.4 +
            seasonal_score * 0.2
        )
        
        return total_score
    
    def _calculate_optimal_capacity(self, avg_demand: float, max_demand: float, 
                                  provider: str) -> int:
        """Calculate optimal capacity for the provider"""
        
        # Base capacity on average demand with buffer for peaks
        buffer_factor = 1.2  # 20% buffer
        
        if provider in ['aws', 'gcp']:
            # These providers have better auto-scaling, so less buffer needed
            buffer_factor = 1.15
        
        optimal_capacity = int(avg_demand * buffer_factor)
        
        # Ensure minimum capacity
        return max(1, optimal_capacity)
    
    def _calculate_cost_savings(self, provider_scores: Dict[str, float], 
                              best_provider: str) -> float:
        """Calculate expected cost savings from optimal provider selection"""
        
        if not provider_scores:
            return 0.0
        
        # Compare best provider cost to average cost
        best_score = provider_scores[best_provider]
        avg_score = np.mean(list(provider_scores.values()))
        
        # Convert score difference to cost savings percentage
        savings_percent = (best_score - avg_score) / avg_score * 100
        
        return max(0.0, savings_percent)
    
    def _calculate_optimization_summary(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization summary"""
        
        if not optimization_results:
            return {}
        
        # Count provider recommendations
        provider_counts = {}
        total_savings = 0.0
        total_confidence = 0.0
        
        for result in optimization_results.values():
            provider = result['recommended_provider']
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            total_savings += result['expected_cost_savings']
            total_confidence += result['confidence']
        
        num_resources = len(optimization_results)
        
        return {
            'total_resources_optimized': num_resources,
            'average_cost_savings_percent': total_savings / num_resources if num_resources > 0 else 0.0,
            'average_confidence': total_confidence / num_resources if num_resources > 0 else 0.0,
            'provider_distribution': provider_counts,
            'optimization_timestamp': datetime.now().isoformat()
        }

class PredictiveScalingEngine:
    """Main predictive scaling engine coordinating all components"""
    
    def __init__(self, safety_checker: SafetyChecker):
        self.demand_predictor = DemandPredictor()
        self.scaling_controller = ScalingController(safety_checker)
        self.resource_optimizer = ResourceOptimizer()
        self.logger = logging.getLogger(__name__ + ".PredictiveScalingEngine")
        
        # Track trained resources
        self.trained_resources: set = set()
    
    async def initialize_resource_monitoring(self, resource_id: str, 
                                           historical_data: pd.DataFrame) -> bool:
        """Initialize predictive scaling for a resource"""
        
        try:
            self.logger.info(f"Initializing predictive scaling for resource {resource_id}")
            
            # Train demand prediction models
            success = await self.demand_predictor.train_models(resource_id, historical_data)
            
            if success:
                self.trained_resources.add(resource_id)
                self.logger.info(f"Predictive scaling initialized for resource {resource_id}")
            else:
                self.logger.warning(f"Failed to initialize predictive scaling for resource {resource_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing resource monitoring: {str(e)}")
            return False
    
    async def forecast_demand(self, resource_id: str, horizon: ForecastHorizon,
                            recent_data: pd.DataFrame) -> Optional[DemandForecast]:
        """Generate demand forecast for a resource"""
        
        if resource_id not in self.trained_resources:
            self.logger.warning(f"Resource {resource_id} not trained for prediction")
            return None
        
        try:
            forecast = await self.demand_predictor.predict_demand(
                resource_id, recent_data, horizon
            )
            
            if forecast:
                self.logger.info(f"Generated {horizon.value} forecast for resource {resource_id}")
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error forecasting demand: {str(e)}")
            return None
    
    async def recommend_scaling_action(self, forecast: DemandForecast,
                                     current_capacity: int) -> ScalingRecommendation:
        """Recommend scaling action based on forecast"""
        
        try:
            recommendation = await self.scaling_controller.recommend_scaling_action(
                forecast, current_capacity
            )
            
            self.logger.info(f"Generated scaling recommendation for resource {forecast.resource_id}: "
                           f"{recommendation.action_type.value}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error recommending scaling action: {str(e)}")
            # Return safe default recommendation
            return ScalingRecommendation(
                resource_id=forecast.resource_id,
                action_type=ScalingActionType.NO_ACTION,
                target_capacity=current_capacity,
                current_capacity=current_capacity,
                confidence=0.0,
                reasoning=f"Error in recommendation: {str(e)}",
                expected_impact={},
                execution_time=datetime.now(),
                safety_checks_passed=False
            )
    
    async def execute_scaling(self, recommendation: ScalingRecommendation,
                            cloud_provider: CloudProvider) -> ScalingResult:
        """Execute scaling action"""
        
        try:
            result = await self.scaling_controller.execute_scaling(
                recommendation, cloud_provider
            )
            
            self.logger.info(f"Scaling execution result for resource {recommendation.resource_id}: "
                           f"{'Success' if result.success else 'Failed'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing scaling: {str(e)}")
            return ScalingResult(
                resource_id=recommendation.resource_id,
                action_executed=ScalingActionType.NO_ACTION,
                success=False,
                previous_capacity=recommendation.current_capacity,
                new_capacity=recommendation.current_capacity,
                execution_time=datetime.now(),
                error_message=str(e),
                cost_impact=None
            )
    
    async def optimize_multi_resource_allocation(self, resource_ids: List[str],
                                               horizon: ForecastHorizon,
                                               available_providers: List[str]) -> Dict[str, Any]:
        """Optimize allocation across multiple resources and providers"""
        
        try:
            # Generate forecasts for all resources
            forecasts = {}
            
            for resource_id in resource_ids:
                if resource_id in self.trained_resources:
                    # This would need recent data - simplified for now
                    recent_data = pd.DataFrame()  # Would be fetched from monitoring system
                    
                    forecast = await self.forecast_demand(resource_id, horizon, recent_data)
                    if forecast:
                        forecasts[resource_id] = forecast
            
            if not forecasts:
                return {}
            
            # Optimize allocation across providers
            optimization_result = await self.resource_optimizer.optimize_resource_allocation(
                forecasts, available_providers
            )
            
            self.logger.info(f"Optimized allocation for {len(forecasts)} resources "
                           f"across {len(available_providers)} providers")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing multi-resource allocation: {str(e)}")
            return {}
    
    async def update_model(self, resource_id: str, feedback_data: pd.DataFrame) -> bool:
        """Update models with new feedback data"""
        
        try:
            if resource_id not in self.trained_resources:
                return False
            
            # Retrain models with new data
            success = await self.demand_predictor.train_models(resource_id, feedback_data)
            
            if success:
                self.logger.info(f"Updated models for resource {resource_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False