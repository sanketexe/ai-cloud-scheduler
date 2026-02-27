"""
Data Pipeline for AI Cost Anomaly Detection

Orchestrates the complete data pipeline from raw AWS cost data collection
through feature engineering to ML-ready datasets. Handles data validation,
preprocessing, and preparation for anomaly detection models.
"""

import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import structlog

from .cost_data_collector import CostDataCollector, CostDataPoint
from .feature_engine import FeatureEngine, FeatureSet, FeatureConfig
from .time_series_db import TimeSeriesDB, TimeSeriesPoint, DataPreprocessor

logger = structlog.get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    collection_interval_hours: int = 1
    feature_extraction_interval_hours: int = 6
    data_retention_days: int = 365
    batch_size: int = 1000
    enable_real_time: bool = True
    enable_preprocessing: bool = True
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring"""
    data_points_collected: int = 0
    data_points_processed: int = 0
    features_extracted: int = 0
    processing_time_ms: float = 0
    error_count: int = 0
    last_run_time: Optional[datetime] = None


class DataPipeline:
    """
    Complete data pipeline for AI cost anomaly detection.
    
    Orchestrates data collection, preprocessing, feature engineering,
    and storage for ML models. Provides both batch and real-time
    processing capabilities.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.data_collector = CostDataCollector()
        self.feature_engine = FeatureEngine(self.config.feature_config)
        self.time_series_db = TimeSeriesDB()
        self.preprocessor = DataPreprocessor()
        
        # Pipeline state
        self.metrics = PipelineMetrics()
        self.is_running = False
        self.last_collection_time = None
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 60  # seconds
    
    async def initialize(self):
        """Initialize the data pipeline"""
        logger.info("Initializing data pipeline")
        
        try:
            await self.time_series_db.initialize()
            logger.info("Data pipeline initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize data pipeline", error=str(e))
            raise
    
    async def run_batch_pipeline(self,
                                start_date: date,
                                end_date: date,
                                account_ids: Optional[List[str]] = None,
                                services: Optional[List[str]] = None) -> PipelineMetrics:
        """
        Run complete batch pipeline for historical data processing.
        
        Args:
            start_date: Start date for batch processing
            end_date: End date for batch processing
            account_ids: Optional account ID filters
            services: Optional service filters
            
        Returns:
            Pipeline metrics for the batch run
        """
        logger.info(
            "Starting batch pipeline",
            date_range=f"{start_date} to {end_date}",
            accounts=len(account_ids) if account_ids else "all",
            services=len(services) if services else "all"
        )
        
        start_time = datetime.utcnow()
        batch_metrics = PipelineMetrics(last_run_time=start_time)
        
        try:
            # Step 1: Collect raw cost data
            logger.info("Step 1: Collecting cost data")
            cost_data = await self._collect_batch_data(start_date, end_date, account_ids, services)
            batch_metrics.data_points_collected = len(cost_data)
            
            # Step 2: Preprocess and store data
            logger.info("Step 2: Preprocessing and storing data")
            processed_count = await self._process_and_store_data(cost_data)
            batch_metrics.data_points_processed = processed_count
            
            # Step 3: Extract features
            logger.info("Step 3: Extracting features")
            feature_count = await self._extract_batch_features(start_date, end_date)
            batch_metrics.features_extracted = feature_count
            
            # Calculate processing time
            end_time = datetime.utcnow()
            batch_metrics.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.info(
                "Batch pipeline completed successfully",
                metrics=batch_metrics
            )
            
            return batch_metrics
            
        except Exception as e:
            batch_metrics.error_count += 1
            logger.error("Batch pipeline failed", error=str(e))
            raise
    
    async def run_real_time_pipeline(self) -> PipelineMetrics:
        """
        Run real-time pipeline for continuous data processing.
        
        Returns:
            Pipeline metrics for the real-time run
        """
        logger.info("Starting real-time pipeline")
        
        start_time = datetime.utcnow()
        rt_metrics = PipelineMetrics(last_run_time=start_time)
        
        try:
            # Determine time range for real-time collection
            end_time = datetime.utcnow()
            start_time_data = end_time - timedelta(hours=self.config.collection_interval_hours)
            
            # Step 1: Collect recent cost data
            logger.info("Collecting recent cost data")
            cost_data = await self._collect_real_time_data(start_time_data, end_time)
            rt_metrics.data_points_collected = len(cost_data)
            
            # Step 2: Process and store data
            logger.info("Processing and storing recent data")
            processed_count = await self._process_and_store_data(cost_data)
            rt_metrics.data_points_processed = processed_count
            
            # Step 3: Extract features for recent data
            logger.info("Extracting real-time features")
            feature_count = await self._extract_real_time_features(start_time_data, end_time)
            rt_metrics.features_extracted = feature_count
            
            # Update metrics
            end_time_processing = datetime.utcnow()
            rt_metrics.processing_time_ms = (end_time_processing - start_time).total_seconds() * 1000
            
            logger.info(
                "Real-time pipeline completed successfully",
                metrics=rt_metrics
            )
            
            return rt_metrics
            
        except Exception as e:
            rt_metrics.error_count += 1
            logger.error("Real-time pipeline failed", error=str(e))
            raise
    
    async def start_continuous_pipeline(self):
        """Start continuous real-time pipeline"""
        logger.info("Starting continuous pipeline")
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Run real-time pipeline
                metrics = await self.run_real_time_pipeline()
                
                # Update overall metrics
                self.metrics.data_points_collected += metrics.data_points_collected
                self.metrics.data_points_processed += metrics.data_points_processed
                self.metrics.features_extracted += metrics.features_extracted
                self.metrics.last_run_time = metrics.last_run_time
                
                # Wait for next interval
                await asyncio.sleep(self.config.collection_interval_hours * 3600)
                
            except Exception as e:
                self.metrics.error_count += 1
                logger.error("Continuous pipeline iteration failed", error=str(e))
                
                # Wait before retrying
                await asyncio.sleep(self.retry_delay)
    
    def stop_continuous_pipeline(self):
        """Stop continuous pipeline"""
        logger.info("Stopping continuous pipeline")
        self.is_running = False
    
    async def get_ml_training_data(self,
                                 start_date: date,
                                 end_date: date,
                                 include_features: bool = True) -> Tuple[pd.DataFrame, Optional[FeatureSet]]:
        """
        Get ML training data with optional feature engineering.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            include_features: Whether to include engineered features
            
        Returns:
            Tuple of (raw_data, feature_set)
        """
        logger.info("Preparing ML training data", date_range=f"{start_date} to {end_date}")
        
        try:
            # Get raw training data from time series DB
            raw_data = await self.time_series_db.get_ml_training_data(start_date, end_date)
            
            feature_set = None
            if include_features and not raw_data.empty:
                # Extract features for training
                feature_set = self.feature_engine.extract_training_features(raw_data)
            
            logger.info(
                "ML training data prepared",
                raw_rows=len(raw_data),
                features=len(feature_set.feature_names) if feature_set else 0
            )
            
            return raw_data, feature_set
            
        except Exception as e:
            logger.error("Failed to prepare ML training data", error=str(e))
            raise
    
    async def get_real_time_inference_data(self, lookback_hours: int = 24) -> Tuple[pd.DataFrame, FeatureSet]:
        """
        Get real-time data for ML inference.
        
        Args:
            lookback_hours: Hours of historical data to include
            
        Returns:
            Tuple of (raw_data, feature_set)
        """
        logger.info("Preparing real-time inference data", lookback_hours=lookback_hours)
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Query recent data
            raw_data = await self.time_series_db.query_cost_data(start_time, end_time)
            
            # Extract real-time features
            feature_set = self.feature_engine.extract_real_time_features(raw_data)
            
            logger.info(
                "Real-time inference data prepared",
                raw_rows=len(raw_data),
                features=len(feature_set.feature_names)
            )
            
            return raw_data, feature_set
            
        except Exception as e:
            logger.error("Failed to prepare real-time inference data", error=str(e))
            raise
    
    async def validate_data_quality(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Validate data quality for a given time period.
        
        Args:
            start_date: Start date for validation
            end_date: End date for validation
            
        Returns:
            Data quality report
        """
        logger.info("Validating data quality", date_range=f"{start_date} to {end_date}")
        
        try:
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date, datetime.max.time())
            
            # Get data statistics
            statistics = await self.time_series_db.get_cost_statistics(start_time, end_time)
            
            # Query raw data for quality checks
            raw_data = await self.time_series_db.query_cost_data(start_time, end_time)
            
            quality_report = {
                'time_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'data_statistics': statistics,
                'quality_metrics': {}
            }
            
            if not raw_data.empty:
                # Calculate quality metrics
                quality_report['quality_metrics'] = {
                    'total_records': len(raw_data),
                    'missing_values': raw_data.isnull().sum().to_dict(),
                    'duplicate_records': raw_data.duplicated().sum(),
                    'data_completeness': (1 - raw_data.isnull().sum() / len(raw_data)).to_dict(),
                    'time_coverage': {
                        'first_timestamp': raw_data['timestamp'].min().isoformat() if 'timestamp' in raw_data.columns else None,
                        'last_timestamp': raw_data['timestamp'].max().isoformat() if 'timestamp' in raw_data.columns else None,
                        'time_gaps': self._detect_time_gaps(raw_data)
                    }
                }
            
            logger.info("Data quality validation completed", report=quality_report)
            return quality_report
            
        except Exception as e:
            logger.error("Data quality validation failed", error=str(e))
            raise
    
    async def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up old data beyond retention period.
        
        Returns:
            Cleanup statistics
        """
        logger.info("Starting data cleanup", retention_days=self.config.data_retention_days)
        
        try:
            # Cleanup time series data
            ts_deleted = await self.time_series_db.cleanup_old_data(self.config.data_retention_days)
            
            cleanup_stats = {
                'time_series_deleted': ts_deleted,
                'retention_days': self.config.data_retention_days,
                'cleanup_date': datetime.utcnow().isoformat()
            }
            
            logger.info("Data cleanup completed", stats=cleanup_stats)
            return cleanup_stats
            
        except Exception as e:
            logger.error("Data cleanup failed", error=str(e))
            raise
    
    async def _collect_batch_data(self,
                                start_date: date,
                                end_date: date,
                                account_ids: Optional[List[str]],
                                services: Optional[List[str]]) -> List[CostDataPoint]:
        """Collect batch cost data"""
        try:
            cost_data = await self.data_collector.collect_hourly_costs(
                start_date, end_date, account_ids, services
            )
            return cost_data
            
        except Exception as e:
            logger.error("Batch data collection failed", error=str(e))
            raise
    
    async def _collect_real_time_data(self, start_time: datetime, end_time: datetime) -> List[CostDataPoint]:
        """Collect real-time cost data"""
        try:
            # For real-time, collect data for the last few hours
            start_date = start_time.date()
            end_date = end_time.date()
            
            cost_data = await self.data_collector.collect_hourly_costs(start_date, end_date)
            
            # Filter to exact time range
            filtered_data = [
                point for point in cost_data
                if start_time <= point.timestamp <= end_time
            ]
            
            return filtered_data
            
        except Exception as e:
            logger.error("Real-time data collection failed", error=str(e))
            raise
    
    async def _process_and_store_data(self, cost_data: List[CostDataPoint]) -> int:
        """Process and store cost data"""
        try:
            if not cost_data:
                return 0
            
            # Convert to DataFrame for preprocessing
            df_data = []
            for point in cost_data:
                row = {
                    'timestamp': point.timestamp,
                    'account_id': point.account_id,
                    'service': point.service,
                    'region': point.region,
                    'cost_amount': float(point.cost_amount),
                    'usage_amount': float(point.usage_amount) if point.usage_amount else 0.0,
                    'usage_unit': point.usage_unit,
                    'currency': point.currency
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Preprocess data if enabled
            if self.config.enable_preprocessing:
                df = self.preprocessor.preprocess_cost_data(df)
            
            # Convert back to TimeSeriesPoint format
            time_series_points = []
            for _, row in df.iterrows():
                point = TimeSeriesPoint(
                    timestamp=row['timestamp'],
                    account_id=row['account_id'],
                    service=row['service'],
                    region=row['region'],
                    metric_name='cost',
                    value=row['cost_amount'],
                    tags={},
                    metadata={
                        'usage_amount': row.get('usage_amount', 0),
                        'usage_unit': row.get('usage_unit', ''),
                        'currency': row.get('currency', 'USD')
                    }
                )
                time_series_points.append(point)
            
            # Store in time series database
            stored_count = await self.time_series_db.store_cost_data(time_series_points)
            return stored_count
            
        except Exception as e:
            logger.error("Data processing and storage failed", error=str(e))
            raise
    
    async def _extract_batch_features(self, start_date: date, end_date: date) -> int:
        """Extract features for batch data"""
        try:
            # Get data from time series DB
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date, datetime.max.time())
            
            raw_data = await self.time_series_db.query_cost_data(start_time, end_time)
            
            if raw_data.empty:
                return 0
            
            # Extract features
            feature_set = self.feature_engine.extract_training_features(raw_data)
            
            return len(feature_set.feature_names)
            
        except Exception as e:
            logger.error("Batch feature extraction failed", error=str(e))
            raise
    
    async def _extract_real_time_features(self, start_time: datetime, end_time: datetime) -> int:
        """Extract features for real-time data"""
        try:
            # Get recent data
            raw_data = await self.time_series_db.query_cost_data(start_time, end_time)
            
            if raw_data.empty:
                return 0
            
            # Extract real-time features
            feature_set = self.feature_engine.extract_real_time_features(raw_data)
            
            return len(feature_set.feature_names)
            
        except Exception as e:
            logger.error("Real-time feature extraction failed", error=str(e))
            raise
    
    def _detect_time_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect gaps in time series data"""
        if 'timestamp' not in data.columns or data.empty:
            return []
        
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        timestamps = pd.to_datetime(data_sorted['timestamp'])
        
        # Calculate time differences
        time_diffs = timestamps.diff()
        
        # Expected interval (assume hourly data)
        expected_interval = pd.Timedelta(hours=1)
        gap_threshold = expected_interval * 2  # Allow some tolerance
        
        # Find gaps
        gaps = []
        gap_indices = time_diffs > gap_threshold
        
        for idx in gap_indices[gap_indices].index:
            if idx > 0:
                gap_start = timestamps.iloc[idx - 1]
                gap_end = timestamps.iloc[idx]
                gap_duration = gap_end - gap_start
                
                gaps.append({
                    'gap_start': gap_start.isoformat(),
                    'gap_end': gap_end.isoformat(),
                    'duration_hours': gap_duration.total_seconds() / 3600
                })
        
        return gaps
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics"""
        return {
            'is_running': self.is_running,
            'config': {
                'collection_interval_hours': self.config.collection_interval_hours,
                'feature_extraction_interval_hours': self.config.feature_extraction_interval_hours,
                'data_retention_days': self.config.data_retention_days,
                'enable_real_time': self.config.enable_real_time,
                'enable_preprocessing': self.config.enable_preprocessing
            },
            'metrics': {
                'data_points_collected': self.metrics.data_points_collected,
                'data_points_processed': self.metrics.data_points_processed,
                'features_extracted': self.metrics.features_extracted,
                'processing_time_ms': self.metrics.processing_time_ms,
                'error_count': self.metrics.error_count,
                'last_run_time': self.metrics.last_run_time.isoformat() if self.metrics.last_run_time else None
            },
            'last_collection_time': self.last_collection_time.isoformat() if self.last_collection_time else None
        }
    
    async def shutdown(self):
        """Shutdown the data pipeline"""
        logger.info("Shutting down data pipeline")
        
        # Stop continuous pipeline if running
        if self.is_running:
            self.stop_continuous_pipeline()
        
        # Close database connections
        await self.time_series_db.close()
        
        logger.info("Data pipeline shutdown completed")