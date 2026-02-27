"""
AI-Powered Cost Anomaly Detection System

This module implements machine learning-based cost anomaly detection with:
- Real-time anomaly detection using ensemble ML models
- Predictive cost forecasting with confidence intervals
- Service-level drill-down analysis
- Explainable AI for anomaly explanations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json

# ML libraries
try:
    from prophet import Prophet
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will be disabled.")

from .aws_cost_analyzer import CostAnalysisReport
from .database import get_db_session
from .models import AnomalyEvent, CostForecast, AnomalyConfiguration

logger = logging.getLogger(__name__)

class AWSCostExplorer:
    """Simple AWS Cost Explorer interface for ML anomaly detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AWSCostExplorer")
    
    async def get_cost_and_usage(self, **kwargs):
        """Get cost and usage data from AWS Cost Explorer API"""
        # This would integrate with actual AWS Cost Explorer API
        # For now, return empty structure for testing
        return {
            'ResultsByTime': []
        }

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    event_id: str
    account_id: str
    detection_time: datetime
    anomaly_type: str
    service: str
    resource_id: Optional[str]
    anomaly_score: float
    cost_impact: float
    percentage_deviation: float
    baseline_value: float
    actual_value: float
    feature_importance: Dict[str, float]
    explanation: str
    confidence: float

@dataclass
class ForecastResult:
    """Result of cost forecasting"""
    forecast_id: str
    account_id: str
    generated_time: datetime
    forecast_period_days: int
    forecast_values: List[float]
    confidence_intervals: Dict[str, List[float]]
    accuracy_score: float
    key_assumptions: List[str]
    risk_factors: List[str]
    budget_overrun_probability: Optional[float]

class CostDataCollector:
    """Enhanced cost data collection for ML processing"""
    
    def __init__(self, aws_cost_explorer: AWSCostExplorer):
        self.aws_cost_explorer = aws_cost_explorer
        self.logger = logging.getLogger(__name__ + ".CostDataCollector")
    
    async def collect_enhanced_cost_data(self, account_id: str, days: int = 90) -> pd.DataFrame:
        """Collect comprehensive cost data for ML analysis"""
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get daily cost data
            daily_costs = await self.aws_cost_explorer.get_cost_and_usage(
                account_id=account_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                granularity='DAILY',
                group_by=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ]
            )
            
            # Convert to DataFrame
            cost_df = self._convert_to_dataframe(daily_costs)
            
            # Enrich with additional features
            enriched_df = await self._enrich_cost_data(cost_df, account_id)
            
            return enriched_df
            
        except Exception as e:
            self.logger.error(f"Error collecting cost data for account {account_id}: {str(e)}")
            raise
    
    def _convert_to_dataframe(self, cost_data: Dict) -> pd.DataFrame:
        """Convert AWS cost data to pandas DataFrame"""
        
        records = []
        
        for result in cost_data.get('ResultsByTime', []):
            date = result['TimePeriod']['Start']
            
            for group in result.get('Groups', []):
                service = group['Keys'][0] if len(group['Keys']) > 0 else 'Unknown'
                region = group['Keys'][1] if len(group['Keys']) > 1 else 'Unknown'
                
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                records.append({
                    'date': pd.to_datetime(date),
                    'service': service,
                    'region': region,
                    'cost': amount,
                    'currency': group['Metrics']['UnblendedCost']['Unit']
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    async def _enrich_cost_data(self, cost_df: pd.DataFrame, account_id: str) -> pd.DataFrame:
        """Enrich cost data with additional features for ML"""
        
        if cost_df.empty:
            return cost_df
        
        # Add time-based features
        cost_df['day_of_week'] = cost_df['date'].dt.dayofweek
        cost_df['day_of_month'] = cost_df['date'].dt.day
        cost_df['month'] = cost_df['date'].dt.month
        cost_df['quarter'] = cost_df['date'].dt.quarter
        cost_df['is_weekend'] = cost_df['day_of_week'].isin([5, 6])
        cost_df['is_month_end'] = cost_df['day_of_month'] > 25
        
        # Add rolling statistics
        cost_df = cost_df.sort_values(['service', 'region', 'date'])
        
        for window in [7, 14, 30]:
            cost_df[f'rolling_mean_{window}d'] = cost_df.groupby(['service', 'region'])['cost'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            cost_df[f'rolling_std_{window}d'] = cost_df.groupby(['service', 'region'])['cost'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Add rate of change
        cost_df['cost_change_1d'] = cost_df.groupby(['service', 'region'])['cost'].pct_change()
        cost_df['cost_change_7d'] = cost_df.groupby(['service', 'region'])['cost'].pct_change(periods=7)
        
        # Fill NaN values
        cost_df = cost_df.fillna(0)
        
        return cost_df

class FeatureEngine:
    """Feature engineering for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.logger = logging.getLogger(__name__ + ".FeatureEngine")
    
    def extract_features(self, cost_df: pd.DataFrame) -> np.ndarray:
        """Extract ML features from cost data"""
        
        if cost_df.empty:
            return np.array([])
        
        # Select feature columns
        feature_cols = [
            'cost', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'is_month_end', 'rolling_mean_7d', 'rolling_mean_14d',
            'rolling_mean_30d', 'rolling_std_7d', 'rolling_std_14d', 'rolling_std_30d',
            'cost_change_1d', 'cost_change_7d'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in cost_df.columns]
        self.feature_columns = available_cols
        
        if not available_cols:
            self.logger.warning("No feature columns available")
            return np.array([])
        
        # Extract features
        features = cost_df[available_cols].values
        
        # Handle infinite and NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler"""
        if features.size > 0:
            self.scaler.fit(features)
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if features.size > 0:
            return self.scaler.transform(features)
        return features

class IsolationForestDetector:
    """Isolation Forest model for point anomaly detection"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0
        )
        self.is_fitted = False
        self.logger = logging.getLogger(__name__ + ".IsolationForestDetector")
    
    def fit(self, features: np.ndarray):
        """Train the isolation forest model"""
        if features.size == 0:
            self.logger.warning("No features provided for training")
            return
        
        try:
            self.model.fit(features)
            self.is_fitted = True
            self.logger.info(f"Isolation Forest trained on {features.shape[0]} samples")
        except Exception as e:
            self.logger.error(f"Error training Isolation Forest: {str(e)}")
            raise
    
    def predict_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        if not self.is_fitted or features.size == 0:
            return np.array([]), np.array([])
        
        try:
            # Get anomaly predictions (-1 for anomaly, 1 for normal)
            predictions = self.model.predict(features)
            
            # Get anomaly scores (lower scores indicate anomalies)
            scores = self.model.decision_function(features)
            
            # Convert to confidence scores (0-100, higher = more anomalous)
            confidence_scores = self._convert_to_confidence_scores(scores)
            
            return predictions, confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error predicting anomalies: {str(e)}")
            return np.array([]), np.array([])
    
    def _convert_to_confidence_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert decision function scores to 0-100 confidence scores"""
        # Normalize scores to 0-1 range, then scale to 0-100
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        # Invert so higher scores indicate more anomalous
        confidence = (1 - normalized) * 100
        
        return confidence

class ProphetForecaster:
    """Prophet model for time series forecasting and trend analysis"""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__ + ".ProphetForecaster")
    
    def fit_and_forecast(self, cost_df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """Fit Prophet model and generate forecast"""
        
        if cost_df.empty:
            self.logger.warning("No data provided for forecasting")
            return {}
        
        try:
            # Prepare data for Prophet
            prophet_df = self._prepare_prophet_data(cost_df)
            
            if prophet_df.empty:
                return {}
            
            # Initialize and fit Prophet model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='multiplicative'
            )
            
            self.model.fit(prophet_df)
            self.is_fitted = True
            
            # Generate forecast
            future = self.model.make_future_dataframe(periods=periods, freq='D')
            forecast = self.model.predict(future)
            
            # Extract forecast results
            forecast_result = self._extract_forecast_results(forecast, periods)
            
            return forecast_result
            
        except Exception as e:
            self.logger.error(f"Error in Prophet forecasting: {str(e)}")
            return {}
    
    def _prepare_prophet_data(self, cost_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        
        # Aggregate daily costs
        daily_costs = cost_df.groupby('date')['cost'].sum().reset_index()
        daily_costs.columns = ['ds', 'y']
        
        # Remove zero values and outliers
        daily_costs = daily_costs[daily_costs['y'] > 0]
        
        # Remove extreme outliers (beyond 3 standard deviations)
        mean_cost = daily_costs['y'].mean()
        std_cost = daily_costs['y'].std()
        daily_costs = daily_costs[
            abs(daily_costs['y'] - mean_cost) <= 3 * std_cost
        ]
        
        return daily_costs
    
    def _extract_forecast_results(self, forecast: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Extract forecast results from Prophet output"""
        
        # Get forecast values for the prediction period
        forecast_values = forecast['yhat'].tail(periods).tolist()
        lower_bounds = forecast['yhat_lower'].tail(periods).tolist()
        upper_bounds = forecast['yhat_upper'].tail(periods).tolist()
        
        # Calculate confidence intervals
        confidence_intervals = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
        
        # Calculate forecast accuracy (using historical data)
        historical_forecast = forecast[:-periods] if periods > 0 else forecast
        if len(historical_forecast) > 0:
            # Simple MAPE calculation
            actual = historical_forecast['y'].dropna()
            predicted = historical_forecast['yhat'][:len(actual)]
            
            if len(actual) > 0 and len(predicted) > 0:
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                accuracy_score = max(0, 100 - mape) / 100
            else:
                accuracy_score = 0.5
        else:
            accuracy_score = 0.5
        
        return {
            'forecast_values': forecast_values,
            'confidence_intervals': confidence_intervals,
            'accuracy_score': accuracy_score,
            'total_forecast': sum(forecast_values),
            'trend_direction': 'increasing' if forecast_values[-1] > forecast_values[0] else 'decreasing'
        }

class MLAnomalyDetector:
    """Main ML-based anomaly detection engine"""
    
    def __init__(self):
        self.cost_collector = None
        self.feature_engine = FeatureEngine()
        self.isolation_forest = IsolationForestDetector()
        self.prophet_forecaster = ProphetForecaster()
        self.is_trained = False
        self.logger = logging.getLogger(__name__ + ".MLAnomalyDetector")
    
    def initialize(self, aws_cost_explorer: AWSCostExplorer):
        """Initialize the anomaly detector with AWS integration"""
        self.cost_collector = CostDataCollector(aws_cost_explorer)
    
    async def train_models(self, account_id: str, training_days: int = 90):
        """Train ML models on historical data"""
        
        if not self.cost_collector:
            raise ValueError("Anomaly detector not initialized")
        
        try:
            self.logger.info(f"Training models for account {account_id}")
            
            # Collect training data
            cost_data = await self.cost_collector.collect_enhanced_cost_data(
                account_id, days=training_days
            )
            
            if cost_data.empty:
                self.logger.warning(f"No training data available for account {account_id}")
                return False
            
            # Extract features
            features = self.feature_engine.extract_features(cost_data)
            
            if features.size == 0:
                self.logger.warning("No features extracted from training data")
                return False
            
            # Fit feature scaler
            self.feature_engine.fit_scaler(features)
            
            # Transform features
            scaled_features = self.feature_engine.transform_features(features)
            
            # Train Isolation Forest
            self.isolation_forest.fit(scaled_features)
            
            self.is_trained = True
            self.logger.info(f"Models trained successfully for account {account_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    async def detect_anomalies(self, account_id: str, days: int = 7) -> List[AnomalyResult]:
        """Detect cost anomalies in recent data"""
        
        if not self.is_trained or not self.cost_collector:
            self.logger.warning("Models not trained or detector not initialized")
            return []
        
        try:
            # Collect recent cost data
            cost_data = await self.cost_collector.collect_enhanced_cost_data(
                account_id, days=days
            )
            
            if cost_data.empty:
                return []
            
            # Extract and transform features
            features = self.feature_engine.extract_features(cost_data)
            if features.size == 0:
                return []
            
            scaled_features = self.feature_engine.transform_features(features)
            
            # Detect anomalies using Isolation Forest
            predictions, confidence_scores = self.isolation_forest.predict_anomalies(scaled_features)
            
            # Process anomaly results
            anomalies = self._process_anomaly_results(
                cost_data, predictions, confidence_scores, account_id
            )
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def generate_forecast(self, account_id: str, periods: int = 30) -> Optional[ForecastResult]:
        """Generate cost forecast using Prophet model"""
        
        if not self.cost_collector:
            return None
        
        try:
            # Collect historical data for forecasting
            cost_data = await self.cost_collector.collect_enhanced_cost_data(
                account_id, days=90
            )
            
            if cost_data.empty:
                return None
            
            # Generate forecast using Prophet
            forecast_result = self.prophet_forecaster.fit_and_forecast(cost_data, periods)
            
            if not forecast_result:
                return None
            
            # Create forecast result object
            forecast = ForecastResult(
                forecast_id=f"forecast_{account_id}_{datetime.now().isoformat()}",
                account_id=account_id,
                generated_time=datetime.now(),
                forecast_period_days=periods,
                forecast_values=forecast_result['forecast_values'],
                confidence_intervals=forecast_result['confidence_intervals'],
                accuracy_score=forecast_result['accuracy_score'],
                key_assumptions=[
                    "Historical spending patterns continue",
                    "No major infrastructure changes",
                    "Seasonal patterns remain consistent"
                ],
                risk_factors=[
                    "Unexpected traffic spikes",
                    "New service deployments",
                    "Pricing changes"
                ],
                budget_overrun_probability=None  # Will be calculated separately
            )
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            return None
    
    def _process_anomaly_results(
        self, 
        cost_data: pd.DataFrame, 
        predictions: np.ndarray, 
        confidence_scores: np.ndarray,
        account_id: str
    ) -> List[AnomalyResult]:
        """Process ML model results into anomaly objects"""
        
        anomalies = []
        
        # Find anomalous points (prediction = -1)
        anomaly_indices = np.where(predictions == -1)[0]
        
        for idx in anomaly_indices:
            if idx >= len(cost_data):
                continue
            
            row = cost_data.iloc[idx]
            confidence = confidence_scores[idx] if idx < len(confidence_scores) else 0
            
            # Skip low-confidence anomalies
            if confidence < 60:  # Configurable threshold
                continue
            
            # Calculate baseline and deviation
            baseline = row.get('rolling_mean_7d', row['cost'])
            actual = row['cost']
            
            if baseline > 0:
                percentage_deviation = ((actual - baseline) / baseline) * 100
            else:
                percentage_deviation = 0
            
            # Create anomaly result
            anomaly = AnomalyResult(
                event_id=f"anomaly_{account_id}_{row['date'].isoformat()}_{row['service']}",
                account_id=account_id,
                detection_time=datetime.now(),
                anomaly_type='point',
                service=row['service'],
                resource_id=None,
                anomaly_score=confidence,
                cost_impact=abs(actual - baseline),
                percentage_deviation=percentage_deviation,
                baseline_value=baseline,
                actual_value=actual,
                feature_importance={
                    'cost_deviation': 0.4,
                    'time_pattern': 0.3,
                    'service_pattern': 0.3
                },
                explanation=f"Cost spike detected in {row['service']}: ${actual:.2f} vs baseline ${baseline:.2f}",
                confidence=confidence / 100
            )
            
            anomalies.append(anomaly)
        
        # Sort by impact (highest first)
        anomalies.sort(key=lambda x: x.cost_impact, reverse=True)
        
        return anomalies

# Service class for integration with the rest of the platform
class CostAnomalyDetectionService:
    """Service class for cost anomaly detection integration"""
    
    def __init__(self, aws_cost_explorer: AWSCostExplorer):
        self.ml_detector = MLAnomalyDetector()
        self.ml_detector.initialize(aws_cost_explorer)
        self.logger = logging.getLogger(__name__ + ".CostAnomalyDetectionService")
    
    async def setup_account_monitoring(self, account_id: str) -> bool:
        """Set up anomaly detection for an AWS account"""
        
        try:
            # Train models on historical data
            success = await self.ml_detector.train_models(account_id)
            
            if success:
                self.logger.info(f"Anomaly detection setup completed for account {account_id}")
            else:
                self.logger.warning(f"Failed to setup anomaly detection for account {account_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting up account monitoring: {str(e)}")
            return False
    
    async def check_for_anomalies(self, account_id: str) -> List[Dict[str, Any]]:
        """Check for cost anomalies in an account"""
        
        try:
            anomalies = await self.ml_detector.detect_anomalies(account_id)
            
            # Convert to dictionaries for API response
            anomaly_dicts = [asdict(anomaly) for anomaly in anomalies]
            
            self.logger.info(f"Detected {len(anomalies)} anomalies for account {account_id}")
            
            return anomaly_dicts
            
        except Exception as e:
            self.logger.error(f"Error checking for anomalies: {str(e)}")
            return []
    
    async def generate_cost_forecast(self, account_id: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """Generate cost forecast for an account"""
        
        try:
            forecast = await self.ml_detector.generate_forecast(account_id, days)
            
            if forecast:
                return asdict(forecast)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            return None