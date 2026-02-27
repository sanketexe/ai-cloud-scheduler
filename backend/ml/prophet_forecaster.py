"""
Prophet Forecaster for Seasonal Trend Analysis

Implements Facebook Prophet for seasonal pattern detection and trend forecasting
in cost data. Specialized for handling seasonality, holidays, and trend changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import structlog
import json

# Prophet imports with fallback
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from fbprophet import Prophet
        from fbprophet.diagnostics import cross_validation, performance_metrics
        from fbprophet.plot import plot_cross_validation_metric
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        Prophet = None

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logger = structlog.get_logger(__name__)


@dataclass
class ForecastResult:
    """Prophet forecast result"""
    timestamp: datetime
    forecasted_value: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float
    trend: float
    seasonal: float
    weekly: Optional[float] = None
    yearly: Optional[float] = None
    holiday: Optional[float] = None
    uncertainty: float = 0.0


@dataclass
class AnomalyDetection:
    """Prophet-based anomaly detection result"""
    timestamp: datetime
    actual_value: float
    forecasted_value: float
    deviation: float
    deviation_percentage: float
    is_anomaly: bool
    severity: str
    confidence: float
    components: Dict[str, float]


@dataclass
class ProphetConfig:
    """Prophet model configuration"""
    growth: str = 'linear'  # 'linear' or 'logistic'
    yearly_seasonality: Union[bool, str, int] = 'auto'
    weekly_seasonality: Union[bool, str, int] = 'auto'
    daily_seasonality: Union[bool, str, int] = False
    seasonality_mode: str = 'additive'  # 'additive' or 'multiplicative'
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    mcmc_samples: int = 0
    interval_width: float = 0.80
    uncertainty_samples: int = 1000


class ProphetForecaster:
    """
    Prophet-based forecaster for seasonal trend analysis and anomaly detection.
    
    Uses Facebook Prophet to model seasonal patterns, trends, and holiday effects
    in cost data. Provides forecasting capabilities and detects deviations from
    expected seasonal behavior.
    """
    
    def __init__(self, config: ProphetConfig = None):
        """
        Initialize Prophet forecaster.
        
        Args:
            config: Prophet model configuration
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is required for seasonal forecasting. "
                "Install with: pip install prophet"
            )
        
        self.config = config or ProphetConfig()
        
        # Model components
        self.model = None
        self.is_trained = False
        self.model_version = "1.0.0"
        
        # Training data and results
        self.training_data = None
        self.forecast_results = None
        self.cross_validation_results = None
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = {}
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            'critical': 3.0,   # 3 standard deviations
            'high': 2.5,       # 2.5 standard deviations
            'medium': 2.0,     # 2 standard deviations
            'low': 1.5         # 1.5 standard deviations
        }
        
        # Holiday definitions (simplified - would use comprehensive calendar in production)
        self.holidays = self._create_holiday_dataframe()
    
    def _create_holiday_dataframe(self) -> pd.DataFrame:
        """Create holiday dataframe for Prophet"""
        holidays = []
        
        # Define major holidays for multiple years
        years = range(2020, 2030)
        
        for year in years:
            holidays.extend([
                {'holiday': 'New Year', 'ds': f'{year}-01-01', 'lower_window': 0, 'upper_window': 1},
                {'holiday': 'Independence Day', 'ds': f'{year}-07-04', 'lower_window': 0, 'upper_window': 1},
                {'holiday': 'Christmas', 'ds': f'{year}-12-25', 'lower_window': -1, 'upper_window': 1},
                {'holiday': 'Thanksgiving', 'ds': f'{year}-11-{self._get_thanksgiving_date(year)}', 'lower_window': -1, 'upper_window': 1},
                {'holiday': 'Black Friday', 'ds': f'{year}-11-{self._get_thanksgiving_date(year) + 1}', 'lower_window': 0, 'upper_window': 1},
            ])
        
        return pd.DataFrame(holidays)
    
    def _get_thanksgiving_date(self, year: int) -> int:
        """Get Thanksgiving date (4th Thursday of November)"""
        # Simplified calculation - in production would use proper date library
        return 22 + (3 - datetime(year, 11, 1).weekday()) % 7
    
    def train(self, 
              training_data: pd.DataFrame,
              target_column: str = 'cost_amount',
              timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Train Prophet model on historical time series data.
        
        Args:
            training_data: Historical time series data
            target_column: Column containing values to forecast
            timestamp_column: Column containing timestamps
            
        Returns:
            Training metrics and model information
        """
        logger.info("Training Prophet forecaster", samples=len(training_data))
        
        try:
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_data = pd.DataFrame()
            prophet_data['ds'] = pd.to_datetime(training_data[timestamp_column])
            prophet_data['y'] = training_data[target_column]
            
            # Remove any missing values
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:
                raise ValueError("Insufficient data points for training (need at least 10)")
            
            # Sort by timestamp
            prophet_data = prophet_data.sort_values('ds')
            
            # Store training data
            self.training_data = prophet_data.copy()
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.config.growth,
                yearly_seasonality=self.config.yearly_seasonality,
                weekly_seasonality=self.config.weekly_seasonality,
                daily_seasonality=self.config.daily_seasonality,
                seasonality_mode=self.config.seasonality_mode,
                changepoint_prior_scale=self.config.changepoint_prior_scale,
                seasonality_prior_scale=self.config.seasonality_prior_scale,
                holidays_prior_scale=self.config.holidays_prior_scale,
                mcmc_samples=self.config.mcmc_samples,
                interval_width=self.config.interval_width,
                uncertainty_samples=self.config.uncertainty_samples,
                holidays=self.holidays
            )
            
            # Add custom seasonalities if needed
            self._add_custom_seasonalities()
            
            # Fit model
            logger.info("Fitting Prophet model")
            self.model.fit(prophet_data)
            
            # Generate in-sample forecast for validation
            future = self.model.make_future_dataframe(periods=0, freq='H')
            forecast = self.model.predict(future)
            
            # Calculate training metrics
            training_metrics = self._calculate_training_metrics(prophet_data, forecast)
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.utcnow(),
                'samples': len(prophet_data),
                'metrics': training_metrics,
                'config': self.config.__dict__
            })
            
            self.is_trained = True
            
            logger.info(
                "Prophet training completed",
                mae=training_metrics.get('mae', 0),
                mape=training_metrics.get('mape', 0),
                rmse=training_metrics.get('rmse', 0)
            )
            
            return training_metrics
            
        except Exception as e:
            logger.error("Prophet training failed", error=str(e))
            raise
    
    def _add_custom_seasonalities(self):
        """Add custom seasonality patterns"""
        # Add monthly seasonality
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Add quarterly seasonality
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=3
        )
    
    def generate_forecast(self, 
                         periods: int = 30,
                         freq: str = 'H',
                         include_history: bool = True) -> List[ForecastResult]:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecast ('H' for hourly, 'D' for daily)
            include_history: Whether to include historical fitted values
            
        Returns:
            List of forecast results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        logger.info("Generating Prophet forecast", periods=periods, freq=freq)
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods, 
                freq=freq, 
                include_history=include_history
            )
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Convert to ForecastResult objects
            results = []
            
            for _, row in forecast.iterrows():
                # Calculate confidence interval width
                confidence_interval = row['yhat_upper'] - row['yhat_lower']
                
                # Calculate uncertainty
                uncertainty = confidence_interval / (2 * row['yhat']) if row['yhat'] != 0 else 0
                
                result = ForecastResult(
                    timestamp=row['ds'],
                    forecasted_value=row['yhat'],
                    lower_bound=row['yhat_lower'],
                    upper_bound=row['yhat_upper'],
                    confidence_interval=confidence_interval,
                    trend=row['trend'],
                    seasonal=row.get('seasonal', 0),
                    weekly=row.get('weekly', None),
                    yearly=row.get('yearly', None),
                    holiday=row.get('holidays', None),
                    uncertainty=abs(uncertainty)
                )
                
                results.append(result)
            
            # Store forecast results
            self.forecast_results = forecast
            
            logger.info("Prophet forecast completed", forecast_points=len(results))
            return results
            
        except Exception as e:
            logger.error("Prophet forecasting failed", error=str(e))
            raise
    
    def detect_forecast_deviations(self, 
                                 actual_data: pd.DataFrame,
                                 timestamp_column: str = 'timestamp',
                                 value_column: str = 'cost_amount') -> List[AnomalyDetection]:
        """
        Detect deviations from forecast (anomalies).
        
        Args:
            actual_data: Actual observed data
            timestamp_column: Column containing timestamps
            value_column: Column containing actual values
            
        Returns:
            List of anomaly detections
        """
        if not self.is_trained or self.forecast_results is None:
            raise ValueError("Model must be trained and forecast generated before anomaly detection")
        
        logger.info("Detecting forecast deviations", samples=len(actual_data))
        
        try:
            # Prepare actual data
            actual_df = pd.DataFrame()
            actual_df['ds'] = pd.to_datetime(actual_data[timestamp_column])
            actual_df['actual'] = actual_data[value_column]
            
            # Merge with forecast results
            merged = pd.merge(
                actual_df,
                self.forecast_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']],
                on='ds',
                how='inner'
            )
            
            if merged.empty:
                logger.warning("No matching timestamps between actual data and forecast")
                return []
            
            # Calculate deviations
            anomalies = []
            
            for _, row in merged.iterrows():
                actual_value = row['actual']
                forecasted_value = row['yhat']
                lower_bound = row['yhat_lower']
                upper_bound = row['yhat_upper']
                
                # Calculate deviation
                deviation = actual_value - forecasted_value
                deviation_percentage = (deviation / forecasted_value * 100) if forecasted_value != 0 else 0
                
                # Check if outside confidence interval
                is_outside_bounds = actual_value < lower_bound or actual_value > upper_bound
                
                # Calculate standardized deviation
                forecast_std = (upper_bound - lower_bound) / (2 * 1.96)  # Approximate std from 95% CI
                standardized_deviation = abs(deviation) / forecast_std if forecast_std > 0 else 0
                
                # Determine if anomaly and severity
                is_anomaly = is_outside_bounds or standardized_deviation > self.anomaly_thresholds['low']
                severity = self._calculate_severity(standardized_deviation)
                
                # Calculate confidence (inverse of uncertainty)
                confidence = 1.0 - min(abs(deviation_percentage) / 100, 1.0)
                
                # Get forecast components
                components = {
                    'trend': row['trend'],
                    'seasonal': forecasted_value - row['trend'],
                    'forecast_std': forecast_std,
                    'standardized_deviation': standardized_deviation
                }
                
                anomaly = AnomalyDetection(
                    timestamp=row['ds'],
                    actual_value=actual_value,
                    forecasted_value=forecasted_value,
                    deviation=deviation,
                    deviation_percentage=deviation_percentage,
                    is_anomaly=is_anomaly,
                    severity=severity,
                    confidence=confidence,
                    components=components
                )
                
                anomalies.append(anomaly)
            
            logger.info(
                "Forecast deviation detection completed",
                total_points=len(anomalies),
                anomalies_detected=sum(1 for a in anomalies if a.is_anomaly)
            )
            
            return anomalies
            
        except Exception as e:
            logger.error("Forecast deviation detection failed", error=str(e))
            raise
    
    def cross_validate_model(self, 
                           initial: str = '730 days',
                           period: str = '180 days',
                           horizon: str = '30 days') -> Dict[str, Any]:
        """
        Perform cross-validation on the trained model.
        
        Args:
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            Cross-validation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        logger.info("Performing Prophet cross-validation")
        
        try:
            # Perform cross-validation
            cv_results = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel='processes'
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            # Store results
            self.cross_validation_results = cv_results
            self.performance_metrics = {
                'mae': float(metrics['mae'].mean()),
                'mape': float(metrics['mape'].mean()),
                'rmse': float(metrics['rmse'].mean()),
                'coverage': float(metrics['coverage'].mean()) if 'coverage' in metrics else None
            }
            
            logger.info(
                "Cross-validation completed",
                mae=self.performance_metrics['mae'],
                mape=self.performance_metrics['mape'],
                rmse=self.performance_metrics['rmse']
            )
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error("Cross-validation failed", error=str(e))
            raise
    
    def detect_changepoints(self) -> List[Dict[str, Any]]:
        """
        Detect significant changepoints in the time series.
        
        Returns:
            List of detected changepoints with metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before changepoint detection")
        
        try:
            # Get changepoints from model
            changepoints = self.model.changepoints
            changepoint_deltas = self.model.params['delta'].mean(axis=0)
            
            # Filter significant changepoints
            significant_changepoints = []
            
            for i, (cp_date, delta) in enumerate(zip(changepoints, changepoint_deltas)):
                if abs(delta) > 0.01:  # Threshold for significance
                    significant_changepoints.append({
                        'date': cp_date,
                        'delta': float(delta),
                        'significance': 'high' if abs(delta) > 0.05 else 'medium',
                        'direction': 'increase' if delta > 0 else 'decrease'
                    })
            
            logger.info("Changepoint detection completed", changepoints=len(significant_changepoints))
            return significant_changepoints
            
        except Exception as e:
            logger.error("Changepoint detection failed", error=str(e))
            raise
    
    def analyze_seasonality(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in the data.
        
        Returns:
            Seasonality analysis results
        """
        if not self.is_trained or self.forecast_results is None:
            raise ValueError("Model must be trained and forecast generated")
        
        try:
            seasonality_analysis = {}
            
            # Analyze different seasonal components
            if 'weekly' in self.forecast_results.columns:
                weekly_strength = self.forecast_results['weekly'].std()
                seasonality_analysis['weekly'] = {
                    'strength': float(weekly_strength),
                    'peak_day': int(self.forecast_results.groupby(
                        self.forecast_results['ds'].dt.dayofweek
                    )['weekly'].mean().idxmax()),
                    'amplitude': float(self.forecast_results['weekly'].max() - self.forecast_results['weekly'].min())
                }
            
            if 'yearly' in self.forecast_results.columns:
                yearly_strength = self.forecast_results['yearly'].std()
                seasonality_analysis['yearly'] = {
                    'strength': float(yearly_strength),
                    'peak_month': int(self.forecast_results.groupby(
                        self.forecast_results['ds'].dt.month
                    )['yearly'].mean().idxmax()),
                    'amplitude': float(self.forecast_results['yearly'].max() - self.forecast_results['yearly'].min())
                }
            
            # Overall seasonality strength
            total_variance = self.forecast_results['yhat'].var()
            seasonal_variance = (
                self.forecast_results.get('weekly', pd.Series([0])).var() +
                self.forecast_results.get('yearly', pd.Series([0])).var()
            )
            
            seasonality_analysis['overall'] = {
                'seasonal_strength': float(seasonal_variance / total_variance) if total_variance > 0 else 0,
                'trend_strength': float(self.forecast_results['trend'].var() / total_variance) if total_variance > 0 else 0
            }
            
            return seasonality_analysis
            
        except Exception as e:
            logger.error("Seasonality analysis failed", error=str(e))
            raise
    
    def _calculate_training_metrics(self, actual_data: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Calculate training performance metrics"""
        
        # Merge actual and forecast data
        merged = pd.merge(actual_data, forecast[['ds', 'yhat']], on='ds', how='inner')
        
        if merged.empty:
            return {}
        
        actual = merged['y']
        predicted = merged['yhat']
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'samples': len(merged)
        }
    
    def _calculate_severity(self, standardized_deviation: float) -> str:
        """Calculate anomaly severity based on standardized deviation"""
        if standardized_deviation >= self.anomaly_thresholds['critical']:
            return 'critical'
        elif standardized_deviation >= self.anomaly_thresholds['high']:
            return 'high'
        elif standardized_deviation >= self.anomaly_thresholds['medium']:
            return 'medium'
        elif standardized_deviation >= self.anomaly_thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            import pickle
            
            model_data = {
                'model': self.model,
                'config': self.config.__dict__,
                'model_version': self.model_version,
                'training_data': self.training_data,
                'forecast_results': self.forecast_results,
                'cross_validation_results': self.cross_validation_results,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'anomaly_thresholds': self.anomaly_thresholds,
                'holidays': self.holidays
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Prophet model saved successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to save Prophet model", filepath=filepath, error=str(e))
            raise
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.config = ProphetConfig(**model_data['config'])
            self.model_version = model_data['model_version']
            self.training_data = model_data['training_data']
            self.forecast_results = model_data['forecast_results']
            self.cross_validation_results = model_data['cross_validation_results']
            self.training_history = model_data['training_history']
            self.performance_metrics = model_data['performance_metrics']
            self.anomaly_thresholds = model_data['anomaly_thresholds']
            self.holidays = model_data['holidays']
            
            self.is_trained = True
            
            logger.info("Prophet model loaded successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to load Prophet model", filepath=filepath, error=str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_type': 'Prophet',
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'prophet_available': PROPHET_AVAILABLE,
            'config': self.config.__dict__,
            'anomaly_thresholds': self.anomaly_thresholds,
            'training_history_count': len(self.training_history),
            'performance_metrics': self.performance_metrics
        }
        
        if self.is_trained and self.training_data is not None:
            info['training_data_info'] = {
                'samples': len(self.training_data),
                'date_range': {
                    'start': self.training_data['ds'].min().isoformat(),
                    'end': self.training_data['ds'].max().isoformat()
                },
                'value_stats': {
                    'mean': float(self.training_data['y'].mean()),
                    'std': float(self.training_data['y'].std()),
                    'min': float(self.training_data['y'].min()),
                    'max': float(self.training_data['y'].max())
                }
            }
        
        return info