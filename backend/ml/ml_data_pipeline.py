"""
Advanced ML Data Pipeline for Real-time Feature Extraction

Provides real-time feature extraction from cloud metrics, streaming data processing,
and ML-ready data preparation for advanced AI/ML features including predictive scaling,
workload intelligence, and reinforcement learning systems.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict
import hashlib

logger = structlog.get_logger(__name__)


class DataSourceType(Enum):
    """Types of data sources for ML pipeline"""
    CLOUD_METRICS = "cloud_metrics"
    COST_DATA = "cost_data"
    PERFORMANCE_METRICS = "performance_metrics"
    RESOURCE_UTILIZATION = "resource_utilization"
    DEPLOYMENT_EVENTS = "deployment_events"
    SCALING_EVENTS = "scaling_events"
    EXTERNAL_FACTORS = "external_factors"


@dataclass
class MLDataPoint:
    """Individual ML data point with metadata"""
    timestamp: datetime
    source_type: DataSourceType
    account_id: str
    resource_id: str
    metric_name: str
    value: float
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0


@dataclass
class FeatureVector:
    """ML feature vector with context"""
    timestamp: datetime
    resource_id: str
    features: Dict[str, float]
    feature_names: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""
    batch_size: int = 1000
    processing_interval_seconds: int = 30
    buffer_size: int = 10000
    enable_real_time: bool = True
    quality_threshold: float = 0.8
    feature_extraction_window_minutes: int = 60


class MLDataPipeline:
    """
    Advanced ML Data Pipeline for real-time feature extraction.
    
    Provides streaming data ingestion, real-time feature extraction,
    data quality monitoring, and ML-ready data preparation for
    advanced AI/ML systems.
    """
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        
        # Data buffers for streaming processing
        self.data_buffers: Dict[DataSourceType, deque] = {
            source_type: deque(maxlen=self.config.buffer_size)
            for source_type in DataSourceType
        }
        
        # Feature extraction state
        self.feature_extractors: Dict[str, Any] = {}
        self.feature_cache: Dict[str, FeatureVector] = {}
        self.feature_schemas: Dict[str, List[str]] = {}
        
        # Processing state
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metrics and monitoring
        self.processing_metrics = {
            'data_points_ingested': 0,
            'features_extracted': 0,
            'processing_errors': 0,
            'quality_violations': 0,
            'last_processing_time': None
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("ML Data Pipeline initialized", config=self.config)
    
    async def start_streaming(self):
        """Start streaming data processing"""
        logger.info("Starting ML data pipeline streaming")
        
        self.is_running = True
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._stream_processor()),
            asyncio.create_task(self._feature_extractor()),
            asyncio.create_task(self._quality_monitor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        logger.info("ML data pipeline streaming started")
    
    async def stop_streaming(self):
        """Stop streaming data processing"""
        logger.info("Stopping ML data pipeline streaming")
        
        self.is_running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ML data pipeline streaming stopped")
    
    async def ingest_data_point(self, data_point: MLDataPoint) -> bool:
        """
        Ingest a single data point into the pipeline.
        
        Args:
            data_point: ML data point to ingest
            
        Returns:
            True if ingested successfully
        """
        try:
            # Validate data quality
            if not self._validate_data_quality(data_point):
                self.processing_metrics['quality_violations'] += 1
                logger.warning("Data quality validation failed", data_point=data_point)
                return False
            
            # Add to appropriate buffer
            with self.lock:
                self.data_buffers[data_point.source_type].append(data_point)
                self.processing_metrics['data_points_ingested'] += 1
            
            return True
            
        except Exception as e:
            self.processing_metrics['processing_errors'] += 1
            logger.error("Failed to ingest data point", error=str(e), data_point=data_point)
            return False
    
    async def ingest_batch(self, data_points: List[MLDataPoint]) -> int:
        """
        Ingest a batch of data points.
        
        Args:
            data_points: List of ML data points to ingest
            
        Returns:
            Number of successfully ingested points
        """
        logger.info("Ingesting data batch", size=len(data_points))
        
        successful_ingests = 0
        
        for data_point in data_points:
            if await self.ingest_data_point(data_point):
                successful_ingests += 1
        
        logger.info("Batch ingestion completed", 
                   successful=successful_ingests, 
                   total=len(data_points))
        
        return successful_ingests
    
    async def extract_features_real_time(self, 
                                       resource_id: str,
                                       lookback_minutes: int = 60) -> Optional[FeatureVector]:
        """
        Extract features for a specific resource in real-time.
        
        Args:
            resource_id: Resource identifier
            lookback_minutes: Minutes of historical data to consider
            
        Returns:
            Feature vector or None if insufficient data
        """
        logger.debug("Extracting real-time features", 
                    resource_id=resource_id, 
                    lookback_minutes=lookback_minutes)
        
        try:
            # Get recent data for resource
            cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
            resource_data = []
            
            with self.lock:
                for source_type, buffer in self.data_buffers.items():
                    for data_point in buffer:
                        if (data_point.resource_id == resource_id and 
                            data_point.timestamp >= cutoff_time):
                            resource_data.append(data_point)
            
            if not resource_data:
                logger.debug("No recent data found for resource", resource_id=resource_id)
                return None
            
            # Extract features
            features = await self._extract_resource_features(resource_data)
            
            if not features:
                return None
            
            feature_vector = FeatureVector(
                timestamp=datetime.utcnow(),
                resource_id=resource_id,
                features=features,
                feature_names=list(features.keys()),
                context={
                    'lookback_minutes': lookback_minutes,
                    'data_points_used': len(resource_data),
                    'extraction_method': 'real_time'
                }
            )
            
            # Cache feature vector
            self.feature_cache[resource_id] = feature_vector
            
            logger.debug("Real-time features extracted", 
                        resource_id=resource_id, 
                        feature_count=len(features))
            
            return feature_vector
            
        except Exception as e:
            logger.error("Real-time feature extraction failed", 
                        error=str(e), 
                        resource_id=resource_id)
            return None
    
    async def get_feature_stream(self, 
                               resource_ids: Optional[List[str]] = None,
                               feature_names: Optional[List[str]] = None) -> AsyncGenerator[FeatureVector, None]:
        """
        Get streaming feature vectors.
        
        Args:
            resource_ids: Optional resource ID filters
            feature_names: Optional feature name filters
            
        Yields:
            Feature vectors as they are generated
        """
        logger.info("Starting feature stream", 
                   resource_filters=len(resource_ids) if resource_ids else "all",
                   feature_filters=len(feature_names) if feature_names else "all")
        
        last_check = datetime.utcnow()
        
        while self.is_running:
            try:
                # Check for new feature vectors
                current_time = datetime.utcnow()
                
                # Get cached features updated since last check
                for resource_id, feature_vector in self.feature_cache.items():
                    if feature_vector.timestamp > last_check:
                        # Apply filters
                        if resource_ids and resource_id not in resource_ids:
                            continue
                        
                        if feature_names:
                            filtered_features = {
                                name: value for name, value in feature_vector.features.items()
                                if name in feature_names
                            }
                            if not filtered_features:
                                continue
                            
                            # Create filtered feature vector
                            filtered_vector = FeatureVector(
                                timestamp=feature_vector.timestamp,
                                resource_id=feature_vector.resource_id,
                                features=filtered_features,
                                feature_names=list(filtered_features.keys()),
                                context=feature_vector.context
                            )
                            yield filtered_vector
                        else:
                            yield feature_vector
                
                last_check = current_time
                await asyncio.sleep(self.config.processing_interval_seconds)
                
            except Exception as e:
                logger.error("Feature stream error", error=str(e))
                await asyncio.sleep(1)
    
    async def get_training_dataset(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 resource_ids: Optional[List[str]] = None,
                                 feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate training dataset for ML models.
        
        Args:
            start_time: Start time for dataset
            end_time: End time for dataset
            resource_ids: Optional resource ID filters
            feature_names: Optional feature name filters
            
        Returns:
            DataFrame with training data
        """
        logger.info("Generating training dataset", 
                   time_range=f"{start_time} to {end_time}",
                   resource_filters=len(resource_ids) if resource_ids else "all")
        
        try:
            # Collect historical data points
            historical_data = []
            
            with self.lock:
                for source_type, buffer in self.data_buffers.items():
                    for data_point in buffer:
                        if (start_time <= data_point.timestamp <= end_time):
                            if not resource_ids or data_point.resource_id in resource_ids:
                                historical_data.append(data_point)
            
            if not historical_data:
                logger.warning("No historical data found for training dataset")
                return pd.DataFrame()
            
            # Group by resource and time windows
            resource_groups = defaultdict(list)
            for data_point in historical_data:
                resource_groups[data_point.resource_id].append(data_point)
            
            # Extract features for each resource group
            training_rows = []
            
            for resource_id, resource_data in resource_groups.items():
                # Sort by timestamp
                resource_data.sort(key=lambda x: x.timestamp)
                
                # Extract features in time windows
                window_size = timedelta(minutes=self.config.feature_extraction_window_minutes)
                current_window_start = start_time
                
                while current_window_start < end_time:
                    window_end = current_window_start + window_size
                    
                    # Get data points in window
                    window_data = [
                        dp for dp in resource_data
                        if current_window_start <= dp.timestamp < window_end
                    ]
                    
                    if window_data:
                        # Extract features for window
                        features = await self._extract_resource_features(window_data)
                        
                        if features:
                            # Filter features if specified
                            if feature_names:
                                features = {
                                    name: value for name, value in features.items()
                                    if name in feature_names
                                }
                            
                            if features:
                                row = {
                                    'timestamp': current_window_start,
                                    'resource_id': resource_id,
                                    **features
                                }
                                training_rows.append(row)
                    
                    current_window_start = window_end
            
            # Create DataFrame
            if training_rows:
                df = pd.DataFrame(training_rows)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values(['resource_id', 'timestamp'])
                
                logger.info("Training dataset generated", 
                           rows=len(df), 
                           features=len(df.columns) - 2)  # Exclude timestamp and resource_id
                
                return df
            else:
                logger.warning("No features extracted for training dataset")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("Training dataset generation failed", error=str(e))
            return pd.DataFrame()
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline processing metrics"""
        with self.lock:
            buffer_sizes = {
                source_type.value: len(buffer)
                for source_type, buffer in self.data_buffers.items()
            }
            
            return {
                'is_running': self.is_running,
                'processing_metrics': self.processing_metrics.copy(),
                'buffer_sizes': buffer_sizes,
                'feature_cache_size': len(self.feature_cache),
                'active_tasks': len([t for t in self.processing_tasks if not t.done()]),
                'config': {
                    'batch_size': self.config.batch_size,
                    'processing_interval_seconds': self.config.processing_interval_seconds,
                    'buffer_size': self.config.buffer_size,
                    'enable_real_time': self.config.enable_real_time
                }
            }
    
    async def _stream_processor(self):
        """Background stream processing task"""
        logger.info("Stream processor started")
        
        while self.is_running:
            try:
                # Process data from buffers
                for source_type, buffer in self.data_buffers.items():
                    if len(buffer) >= self.config.batch_size:
                        # Extract batch for processing
                        with self.lock:
                            batch = list(buffer)[:self.config.batch_size]
                            # Clear processed items
                            for _ in range(min(self.config.batch_size, len(buffer))):
                                buffer.popleft()
                        
                        # Process batch
                        await self._process_data_batch(batch, source_type)
                
                await asyncio.sleep(self.config.processing_interval_seconds)
                
            except Exception as e:
                logger.error("Stream processor error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info("Stream processor stopped")
    
    async def _feature_extractor(self):
        """Background feature extraction task"""
        logger.info("Feature extractor started")
        
        while self.is_running:
            try:
                # Extract features for resources with recent data
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(minutes=self.config.feature_extraction_window_minutes)
                
                # Find resources with recent activity
                active_resources = set()
                
                with self.lock:
                    for buffer in self.data_buffers.values():
                        for data_point in buffer:
                            if data_point.timestamp >= cutoff_time:
                                active_resources.add(data_point.resource_id)
                
                # Extract features for active resources
                for resource_id in active_resources:
                    await self.extract_features_real_time(
                        resource_id, 
                        self.config.feature_extraction_window_minutes
                    )
                
                await asyncio.sleep(self.config.processing_interval_seconds * 2)
                
            except Exception as e:
                logger.error("Feature extractor error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info("Feature extractor stopped")
    
    async def _quality_monitor(self):
        """Background data quality monitoring task"""
        logger.info("Quality monitor started")
        
        while self.is_running:
            try:
                # Monitor data quality metrics
                current_time = datetime.utcnow()
                
                # Check for data freshness
                for source_type, buffer in self.data_buffers.items():
                    if buffer:
                        latest_timestamp = max(dp.timestamp for dp in buffer)
                        age_minutes = (current_time - latest_timestamp).total_seconds() / 60
                        
                        if age_minutes > 30:  # Alert if data is older than 30 minutes
                            logger.warning("Stale data detected", 
                                         source_type=source_type.value,
                                         age_minutes=age_minutes)
                
                # Check buffer utilization
                for source_type, buffer in self.data_buffers.items():
                    utilization = len(buffer) / self.config.buffer_size
                    if utilization > 0.9:
                        logger.warning("Buffer near capacity", 
                                     source_type=source_type.value,
                                     utilization=utilization)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Quality monitor error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info("Quality monitor stopped")
    
    async def _metrics_collector(self):
        """Background metrics collection task"""
        logger.info("Metrics collector started")
        
        while self.is_running:
            try:
                # Update processing metrics
                self.processing_metrics['last_processing_time'] = datetime.utcnow()
                
                # Log periodic metrics
                metrics = self.get_pipeline_metrics()
                logger.info("Pipeline metrics", metrics=metrics['processing_metrics'])
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error("Metrics collector error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info("Metrics collector stopped")
    
    async def _process_data_batch(self, batch: List[MLDataPoint], source_type: DataSourceType):
        """Process a batch of data points"""
        try:
            logger.debug("Processing data batch", 
                        size=len(batch), 
                        source_type=source_type.value)
            
            # Group by resource for efficient processing
            resource_groups = defaultdict(list)
            for data_point in batch:
                resource_groups[data_point.resource_id].append(data_point)
            
            # Process each resource group
            for resource_id, resource_data in resource_groups.items():
                await self._process_resource_data(resource_id, resource_data)
            
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
    
    async def _process_resource_data(self, resource_id: str, data_points: List[MLDataPoint]):
        """Process data points for a specific resource"""
        try:
            # Sort by timestamp
            data_points.sort(key=lambda x: x.timestamp)
            
            # Apply data transformations
            transformed_data = await self._apply_transformations(data_points)
            
            # Update resource state (if needed)
            await self._update_resource_state(resource_id, transformed_data)
            
        except Exception as e:
            logger.error("Resource data processing failed", 
                        error=str(e), 
                        resource_id=resource_id)
    
    async def _apply_transformations(self, data_points: List[MLDataPoint]) -> List[MLDataPoint]:
        """Apply data transformations"""
        # For now, return data as-is
        # In production, this would apply normalization, filtering, etc.
        return data_points
    
    async def _update_resource_state(self, resource_id: str, data_points: List[MLDataPoint]):
        """Update resource state based on processed data"""
        # For now, just log
        logger.debug("Resource state updated", 
                    resource_id=resource_id, 
                    data_points=len(data_points))
    
    async def _extract_resource_features(self, data_points: List[MLDataPoint]) -> Dict[str, float]:
        """Extract features from resource data points"""
        try:
            if not data_points:
                return {}
            
            # Group by metric name
            metric_groups = defaultdict(list)
            for dp in data_points:
                metric_groups[dp.metric_name].append(dp.value)
            
            features = {}
            
            # Extract statistical features for each metric
            for metric_name, values in metric_groups.items():
                if values:
                    features[f"{metric_name}_mean"] = np.mean(values)
                    features[f"{metric_name}_std"] = np.std(values)
                    features[f"{metric_name}_min"] = np.min(values)
                    features[f"{metric_name}_max"] = np.max(values)
                    features[f"{metric_name}_median"] = np.median(values)
                    features[f"{metric_name}_count"] = len(values)
                    
                    # Rate of change
                    if len(values) > 1:
                        features[f"{metric_name}_rate_of_change"] = (values[-1] - values[0]) / len(values)
                    
                    # Percentiles
                    features[f"{metric_name}_p95"] = np.percentile(values, 95)
                    features[f"{metric_name}_p99"] = np.percentile(values, 99)
            
            # Time-based features
            timestamps = [dp.timestamp for dp in data_points]
            if timestamps:
                time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60
                features['time_span_minutes'] = time_span_minutes
                features['data_frequency'] = len(data_points) / max(time_span_minutes, 1)
            
            # Quality features
            quality_scores = [dp.quality_score for dp in data_points]
            features['avg_quality_score'] = np.mean(quality_scores)
            features['min_quality_score'] = np.min(quality_scores)
            
            return features
            
        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            return {}
    
    def _validate_data_quality(self, data_point: MLDataPoint) -> bool:
        """Validate data point quality"""
        try:
            # Check required fields
            if not data_point.resource_id or not data_point.metric_name:
                return False
            
            # Check timestamp is reasonable
            now = datetime.utcnow()
            if data_point.timestamp > now + timedelta(minutes=5):  # Future data
                return False
            
            if data_point.timestamp < now - timedelta(days=7):  # Very old data
                return False
            
            # Check value is numeric and reasonable
            if not isinstance(data_point.value, (int, float)):
                return False
            
            if np.isnan(data_point.value) or np.isinf(data_point.value):
                return False
            
            # Check quality score
            if data_point.quality_score < self.config.quality_threshold:
                return False
            
            return True
            
        except Exception:
            return False