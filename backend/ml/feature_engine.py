"""
Feature Engineering Engine for Cost Anomaly Detection

Transforms raw cost data into ML-ready features for anomaly detection models.
Generates time-based, statistical, service-specific, and contextual features
optimized for detecting various types of cost anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

logger = structlog.get_logger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: pd.DataFrame
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_days: int = 30
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    percentiles: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 0.95, 0.99])
    seasonal_periods: List[int] = field(default_factory=lambda: [7, 30, 365])  # Weekly, monthly, yearly
    include_holidays: bool = True
    normalize_features: bool = True
    pca_components: Optional[int] = None


class FeatureEngine:
    """
    Advanced feature engineering engine for cost anomaly detection.
    
    Generates comprehensive feature sets including:
    - Time-based patterns (hourly, daily, seasonal)
    - Statistical indicators (rolling stats, percentiles)
    - Service-specific metrics (utilization, efficiency)
    - Contextual features (deployments, scaling events)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
        # Feature scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.pca = PCA(n_components=self.config.pca_components) if self.config.pca_components else None
        
        # Holiday calendar (simplified - would use proper holiday library in production)
        self.holidays = self._initialize_holiday_calendar()
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_correlations = {}
    
    def extract_features(self, cost_data: pd.DataFrame, context: Dict[str, Any] = None) -> FeatureSet:
        """
        Extract comprehensive feature set from cost data.
        
        Args:
            cost_data: Normalized cost data DataFrame
            context: Additional context (deployments, scaling events, etc.)
            
        Returns:
            FeatureSet with engineered features
        """
        logger.info("Starting feature extraction", rows=len(cost_data))
        
        try:
            if cost_data.empty:
                return FeatureSet(features=pd.DataFrame(), feature_names=[])
            
            # Ensure timestamp column is datetime
            if 'timestamp' in cost_data.columns:
                cost_data['timestamp'] = pd.to_datetime(cost_data['timestamp'])
                cost_data = cost_data.sort_values('timestamp')
            
            # Extract different feature types
            time_features = self._extract_time_features(cost_data)
            statistical_features = self._extract_statistical_features(cost_data)
            service_features = self._extract_service_features(cost_data)
            contextual_features = self._extract_contextual_features(cost_data, context)
            
            # Combine all features
            all_features = pd.concat([
                time_features,
                statistical_features,
                service_features,
                contextual_features
            ], axis=1)
            
            # Handle missing values
            all_features = self._handle_missing_values(all_features)
            
            # Normalize features if configured
            if self.config.normalize_features:
                all_features = self._normalize_features(all_features)
            
            # Apply PCA if configured
            if self.pca:
                all_features = self._apply_pca(all_features)
            
            feature_names = list(all_features.columns)
            
            logger.info(
                "Feature extraction completed",
                features=len(feature_names),
                samples=len(all_features)
            )
            
            return FeatureSet(
                features=all_features,
                feature_names=feature_names,
                metadata={
                    'extraction_time': datetime.utcnow(),
                    'config': self.config,
                    'data_range': {
                        'start': cost_data['timestamp'].min() if 'timestamp' in cost_data.columns else None,
                        'end': cost_data['timestamp'].max() if 'timestamp' in cost_data.columns else None
                    }
                }
            )
            
        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            raise
    
    def extract_training_features(self, historical_data: pd.DataFrame) -> FeatureSet:
        """
        Extract features optimized for model training.
        
        Args:
            historical_data: Historical cost data for training
            
        Returns:
            FeatureSet optimized for training ML models
        """
        logger.info("Extracting training features", data_points=len(historical_data))
        
        try:
            # Extract base features
            feature_set = self.extract_features(historical_data)
            
            # Add training-specific features
            training_features = self._add_training_features(feature_set.features, historical_data)
            
            # Calculate feature importance and correlations
            self._calculate_feature_statistics(training_features)
            
            # Select most important features
            selected_features = self._select_important_features(training_features)
            
            return FeatureSet(
                features=selected_features,
                feature_names=list(selected_features.columns),
                metadata={
                    **feature_set.metadata,
                    'training_mode': True,
                    'feature_importance': self.feature_importance,
                    'feature_correlations': self.feature_correlations
                }
            )
            
        except Exception as e:
            logger.error("Training feature extraction failed", error=str(e))
            raise
    
    def extract_real_time_features(self, recent_data: pd.DataFrame) -> FeatureSet:
        """
        Extract features for real-time anomaly detection.
        
        Args:
            recent_data: Recent cost data for real-time analysis
            
        Returns:
            FeatureSet optimized for real-time inference
        """
        logger.info("Extracting real-time features", data_points=len(recent_data))
        
        try:
            # Extract base features with reduced lookback for speed
            config = FeatureConfig(
                lookback_days=7,  # Shorter lookback for real-time
                rolling_windows=[3, 7],  # Fewer windows
                normalize_features=True
            )
            
            # Temporarily use reduced config
            original_config = self.config
            self.config = config
            
            feature_set = self.extract_features(recent_data)
            
            # Restore original config
            self.config = original_config
            
            # Add real-time specific features
            rt_features = self._add_real_time_features(feature_set.features, recent_data)
            
            return FeatureSet(
                features=rt_features,
                feature_names=list(rt_features.columns),
                metadata={
                    **feature_set.metadata,
                    'real_time_mode': True,
                    'processing_latency_ms': (datetime.utcnow() - feature_set.timestamp).total_seconds() * 1000
                }
            )
            
        except Exception as e:
            logger.error("Real-time feature extraction failed", error=str(e))
            raise
    
    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        logger.debug("Extracting time-based features")
        
        time_features = pd.DataFrame(index=data.index)
        
        if 'timestamp' not in data.columns:
            return time_features
        
        timestamps = pd.to_datetime(data['timestamp'])
        
        # Basic time features
        time_features['hour'] = timestamps.dt.hour
        time_features['day_of_week'] = timestamps.dt.dayofweek
        time_features['day_of_month'] = timestamps.dt.day
        time_features['month'] = timestamps.dt.month
        time_features['quarter'] = timestamps.dt.quarter
        time_features['year'] = timestamps.dt.year
        
        # Cyclical encoding for better ML performance
        time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
        time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7)
        time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7)
        time_features['month_sin'] = np.sin(2 * np.pi * time_features['month'] / 12)
        time_features['month_cos'] = np.cos(2 * np.pi * time_features['month'] / 12)
        
        # Business time indicators
        time_features['is_weekend'] = (time_features['day_of_week'] >= 5).astype(int)
        time_features['is_business_hours'] = ((time_features['hour'] >= 9) & (time_features['hour'] <= 17)).astype(int)
        time_features['is_night'] = ((time_features['hour'] >= 22) | (time_features['hour'] <= 6)).astype(int)
        
        # Holiday indicators
        if self.config.include_holidays:
            time_features['is_holiday'] = timestamps.dt.date.isin(self.holidays).astype(int)
            time_features['days_since_holiday'] = self._calculate_days_since_holiday(timestamps)
            time_features['days_until_holiday'] = self._calculate_days_until_holiday(timestamps)
        
        # Seasonal patterns
        for period in self.config.seasonal_periods:
            day_of_period = (timestamps - timestamps.min()).dt.days % period
            time_features[f'seasonal_{period}_sin'] = np.sin(2 * np.pi * day_of_period / period)
            time_features[f'seasonal_{period}_cos'] = np.cos(2 * np.pi * day_of_period / period)
        
        return time_features
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from cost data"""
        logger.debug("Extracting statistical features")
        
        stat_features = pd.DataFrame(index=data.index)
        
        if 'cost_amount' not in data.columns:
            return stat_features
        
        cost_series = data['cost_amount']
        
        # Rolling statistics
        for window in self.config.rolling_windows:
            window_data = cost_series.rolling(window=window, min_periods=1)
            
            stat_features[f'rolling_mean_{window}d'] = window_data.mean()
            stat_features[f'rolling_std_{window}d'] = window_data.std()
            stat_features[f'rolling_min_{window}d'] = window_data.min()
            stat_features[f'rolling_max_{window}d'] = window_data.max()
            stat_features[f'rolling_median_{window}d'] = window_data.median()
            
            # Rate of change
            stat_features[f'rate_of_change_{window}d'] = cost_series.pct_change(periods=window)
            
            # Z-score (standardized deviation from rolling mean)
            rolling_mean = window_data.mean()
            rolling_std = window_data.std()
            stat_features[f'z_score_{window}d'] = (cost_series - rolling_mean) / (rolling_std + 1e-10)
        
        # Percentile features
        for percentile in self.config.percentiles:
            for window in self.config.rolling_windows:
                stat_features[f'rolling_p{int(percentile*100)}_{window}d'] = (
                    cost_series.rolling(window=window, min_periods=1).quantile(percentile)
                )
        
        # Lag features
        for lag in [1, 7, 24]:  # 1 hour, 1 day, 1 day (if hourly data)
            if lag < len(cost_series):
                stat_features[f'lag_{lag}'] = cost_series.shift(lag)
                stat_features[f'diff_lag_{lag}'] = cost_series - cost_series.shift(lag)
        
        # Volatility measures
        stat_features['volatility_7d'] = cost_series.rolling(window=7*24, min_periods=1).std()
        stat_features['volatility_30d'] = cost_series.rolling(window=30*24, min_periods=1).std()
        
        # Trend indicators
        stat_features['trend_7d'] = self._calculate_trend(cost_series, window=7*24)
        stat_features['trend_30d'] = self._calculate_trend(cost_series, window=30*24)
        
        return stat_features
    
    def _extract_service_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract service-specific features"""
        logger.debug("Extracting service-specific features")
        
        service_features = pd.DataFrame(index=data.index)
        
        if 'service' not in data.columns:
            return service_features
        
        # Service distribution features
        service_counts = data.groupby('service')['cost_amount'].sum()
        total_cost = service_counts.sum()
        
        for service in service_counts.index:
            service_mask = data['service'] == service
            service_cost_ratio = service_counts[service] / total_cost if total_cost > 0 else 0
            
            service_features[f'service_{service}_ratio'] = service_mask.astype(float) * service_cost_ratio
            service_features[f'service_{service}_cost'] = data['cost_amount'] * service_mask
        
        # Cross-service correlations
        if len(service_counts) > 1:
            service_features['service_diversity'] = len(service_counts)
            service_features['service_concentration'] = (service_counts / total_cost).max() if total_cost > 0 else 0
        
        # Usage efficiency features
        if 'usage_amount' in data.columns:
            service_features['cost_per_unit'] = data['cost_amount'] / (data['usage_amount'] + 1e-10)
            service_features['usage_efficiency'] = data['usage_amount'] / (data['cost_amount'] + 1e-10)
        
        return service_features
    
    def _extract_contextual_features(self, data: pd.DataFrame, context: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract contextual features from external events"""
        logger.debug("Extracting contextual features")
        
        contextual_features = pd.DataFrame(index=data.index)
        
        if not context:
            return contextual_features
        
        # Deployment events
        deployments = context.get('deployments', [])
        if deployments:
            contextual_features['days_since_deployment'] = self._calculate_days_since_events(
                data.get('timestamp', pd.Series()), deployments
            )
            contextual_features['deployment_impact'] = self._calculate_event_impact(
                data.get('timestamp', pd.Series()), deployments, decay_days=7
            )
        
        # Scaling events
        scaling_events = context.get('scaling_events', [])
        if scaling_events:
            contextual_features['days_since_scaling'] = self._calculate_days_since_events(
                data.get('timestamp', pd.Series()), scaling_events
            )
            contextual_features['scaling_impact'] = self._calculate_event_impact(
                data.get('timestamp', pd.Series()), scaling_events, decay_days=3
            )
        
        # Maintenance windows
        maintenance_windows = context.get('maintenance_windows', [])
        if maintenance_windows:
            contextual_features['in_maintenance'] = self._calculate_maintenance_indicator(
                data.get('timestamp', pd.Series()), maintenance_windows
            )
        
        # External factors (market conditions, etc.)
        external_factors = context.get('external_factors', {})
        for factor_name, factor_values in external_factors.items():
            if isinstance(factor_values, (list, pd.Series)):
                contextual_features[f'external_{factor_name}'] = factor_values[:len(data)]
        
        return contextual_features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        logger.debug("Handling missing values", missing_count=features.isnull().sum().sum())
        
        # Forward fill for time series continuity
        features = features.ffill()
        
        # Backward fill for remaining missing values
        features = features.bfill()
        
        # Fill remaining with zeros (for new features)
        features = features.fillna(0)
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML models"""
        logger.debug("Normalizing features")
        
        # Use robust scaler for better handling of outliers
        normalized_features = pd.DataFrame(
            self.robust_scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return normalized_features
    
    def _apply_pca(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        logger.debug("Applying PCA", original_dims=features.shape[1])
        
        pca_features = pd.DataFrame(
            self.pca.fit_transform(features),
            columns=[f'pca_{i}' for i in range(self.pca.n_components_)],
            index=features.index
        )
        
        logger.debug("PCA completed", reduced_dims=pca_features.shape[1])
        return pca_features
    
    def _add_training_features(self, features: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to training mode"""
        training_features = features.copy()
        
        # Add anomaly labels if available
        if 'is_anomaly' in historical_data.columns:
            training_features['anomaly_label'] = historical_data['is_anomaly']
        
        # Add historical context features
        if 'timestamp' in historical_data.columns:
            timestamps = pd.to_datetime(historical_data['timestamp'])
            training_features['days_from_start'] = (timestamps - timestamps.min()).dt.days
            training_features['data_age'] = (timestamps.max() - timestamps).dt.days
        
        return training_features
    
    def _add_real_time_features(self, features: pd.DataFrame, recent_data: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to real-time mode"""
        rt_features = features.copy()
        
        # Add recency indicators
        if 'timestamp' in recent_data.columns:
            timestamps = pd.to_datetime(recent_data['timestamp'])
            rt_features['minutes_since_latest'] = (
                (timestamps.max() - timestamps).dt.total_seconds() / 60
            )
        
        # Add data quality indicators
        rt_features['data_completeness'] = (~features.isnull()).sum(axis=1) / len(features.columns)
        
        return rt_features
    
    def _calculate_feature_statistics(self, features: pd.DataFrame):
        """Calculate feature importance and correlations"""
        # Feature variance (higher variance = more informative)
        feature_variance = features.var()
        self.feature_importance = feature_variance.to_dict()
        
        # Feature correlations
        correlation_matrix = features.corr()
        self.feature_correlations = correlation_matrix.to_dict()
    
    def _select_important_features(self, features: pd.DataFrame, top_k: int = None) -> pd.DataFrame:
        """Select most important features based on variance"""
        if not self.feature_importance or top_k is None:
            return features
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top k features
        selected_feature_names = [name for name, _ in sorted_features[:top_k]]
        
        return features[selected_feature_names]
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend using linear regression slope"""
        def trend_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window=window, min_periods=2).apply(trend_slope)
    
    def _initialize_holiday_calendar(self) -> List[datetime.date]:
        """Initialize holiday calendar (simplified)"""
        # In production, would use proper holiday library
        holidays = []
        current_year = datetime.now().year
        
        # Add major US holidays
        for year in range(current_year - 2, current_year + 2):
            holidays.extend([
                datetime(year, 1, 1).date(),   # New Year's Day
                datetime(year, 7, 4).date(),   # Independence Day
                datetime(year, 12, 25).date(), # Christmas Day
            ])
        
        return holidays
    
    def _calculate_days_since_holiday(self, timestamps: pd.Series) -> pd.Series:
        """Calculate days since last holiday"""
        days_since = []
        
        for ts in timestamps:
            ts_date = ts.date()
            past_holidays = [h for h in self.holidays if h <= ts_date]
            
            if past_holidays:
                last_holiday = max(past_holidays)
                days_since.append((ts_date - last_holiday).days)
            else:
                days_since.append(365)  # Default if no past holidays
        
        return pd.Series(days_since, index=timestamps.index)
    
    def _calculate_days_until_holiday(self, timestamps: pd.Series) -> pd.Series:
        """Calculate days until next holiday"""
        days_until = []
        
        for ts in timestamps:
            ts_date = ts.date()
            future_holidays = [h for h in self.holidays if h > ts_date]
            
            if future_holidays:
                next_holiday = min(future_holidays)
                days_until.append((next_holiday - ts_date).days)
            else:
                days_until.append(365)  # Default if no future holidays
        
        return pd.Series(days_until, index=timestamps.index)
    
    def _calculate_days_since_events(self, timestamps: pd.Series, events: List[Dict[str, Any]]) -> pd.Series:
        """Calculate days since events"""
        days_since = []
        
        for ts in timestamps:
            event_dates = [
                datetime.fromisoformat(event['timestamp']).date()
                for event in events
                if 'timestamp' in event
            ]
            
            past_events = [e for e in event_dates if e <= ts.date()]
            
            if past_events:
                last_event = max(past_events)
                days_since.append((ts.date() - last_event).days)
            else:
                days_since.append(999)  # Large number if no past events
        
        return pd.Series(days_since, index=timestamps.index)
    
    def _calculate_event_impact(self, timestamps: pd.Series, events: List[Dict[str, Any]], decay_days: int = 7) -> pd.Series:
        """Calculate decaying impact of events"""
        impact = []
        
        for ts in timestamps:
            total_impact = 0
            
            for event in events:
                if 'timestamp' not in event:
                    continue
                
                event_date = datetime.fromisoformat(event['timestamp']).date()
                days_diff = (ts.date() - event_date).days
                
                if 0 <= days_diff <= decay_days:
                    # Exponential decay
                    event_impact = np.exp(-days_diff / decay_days)
                    total_impact += event_impact
            
            impact.append(total_impact)
        
        return pd.Series(impact, index=timestamps.index)
    
    def _calculate_maintenance_indicator(self, timestamps: pd.Series, maintenance_windows: List[Dict[str, Any]]) -> pd.Series:
        """Calculate maintenance window indicator"""
        in_maintenance = []
        
        for ts in timestamps:
            is_maintenance = False
            
            for window in maintenance_windows:
                start_time = datetime.fromisoformat(window['start'])
                end_time = datetime.fromisoformat(window['end'])
                
                if start_time <= ts <= end_time:
                    is_maintenance = True
                    break
            
            in_maintenance.append(int(is_maintenance))
        
        return pd.Series(in_maintenance, index=timestamps.index)