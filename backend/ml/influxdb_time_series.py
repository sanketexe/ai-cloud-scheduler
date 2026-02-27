"""
InfluxDB Time Series Database for Advanced ML Features

Provides high-performance time series data storage and retrieval optimized
for ML workloads. Supports real-time data ingestion, efficient querying,
and ML-ready data preparation for advanced AI/ML systems.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
import threading
from collections import defaultdict
import sqlite3  # Using SQLite as InfluxDB alternative for demo
from pathlib import Path

logger = structlog.get_logger(__name__)


class DataRetentionPolicy(Enum):
    """Data retention policies"""
    REAL_TIME = "1h"      # 1 hour for real-time data
    SHORT_TERM = "7d"     # 7 days for short-term analysis
    MEDIUM_TERM = "30d"   # 30 days for medium-term analysis
    LONG_TERM = "365d"    # 1 year for long-term analysis
    PERMANENT = "inf"     # Permanent retention


@dataclass
class TimeSeriesPoint:
    """Time series data point"""
    measurement: str
    timestamp: datetime
    fields: Dict[str, Union[float, int, str, bool]]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Time series query result"""
    measurement: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_time_ms: float = 0.0


@dataclass
class AggregationConfig:
    """Configuration for data aggregation"""
    function: str  # mean, sum, min, max, count, etc.
    window: str    # 1m, 5m, 1h, 1d, etc.
    fill: str = "null"  # null, previous, linear, zero


class InfluxDBTimeSeriesDB:
    """
    InfluxDB-compatible Time Series Database for ML workloads.
    
    Provides high-performance time series storage, real-time ingestion,
    efficient querying, and ML-optimized data preparation.
    
    Note: This implementation uses SQLite as a demonstration.
    In production, this would connect to actual InfluxDB.
    """
    
    def __init__(self, 
                 db_path: str = "influxdb_timeseries.db",
                 retention_policies: Optional[Dict[str, DataRetentionPolicy]] = None):
        self.db_path = Path(db_path)
        self.db_conn = None
        
        # Retention policies for different measurements
        self.retention_policies = retention_policies or {
            'cpu_metrics': DataRetentionPolicy.MEDIUM_TERM,
            'memory_metrics': DataRetentionPolicy.MEDIUM_TERM,
            'cost_metrics': DataRetentionPolicy.LONG_TERM,
            'performance_metrics': DataRetentionPolicy.MEDIUM_TERM,
            'scaling_events': DataRetentionPolicy.LONG_TERM,
            'deployment_events': DataRetentionPolicy.LONG_TERM
        }
        
        # Configuration
        self.batch_size = 1000
        self.max_query_points = 100000
        self.compression_enabled = True
        
        # Caching
        self.query_cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self.cache_ttl_minutes = 5
        self.max_cache_size = 100
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'points_written': 0,
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_write_time': None,
            'last_query_time': None
        }
        
        logger.info("InfluxDB Time Series DB initialized", db_path=str(self.db_path))
    
    async def initialize(self):
        """Initialize time series database"""
        logger.info("Initializing InfluxDB Time Series Database")
        
        try:
            # Create database connection
            self.db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.db_conn.execute("PRAGMA journal_mode=WAL")
            self.db_conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create tables for different measurements
            await self._create_tables()
            
            # Create indexes for performance
            await self._create_indexes()
            
            logger.info("InfluxDB Time Series Database initialized successfully")
            
        except Exception as e:
            logger.error("InfluxDB Time Series Database initialization failed", error=str(e))
            raise
    
    async def write_points(self, points: List[TimeSeriesPoint]) -> int:
        """
        Write time series points to database.
        
        Args:
            points: List of time series points to write
            
        Returns:
            Number of points written successfully
        """
        logger.debug("Writing time series points", count=len(points))
        
        try:
            if not points:
                return 0
            
            written_count = 0
            
            # Group points by measurement for efficient insertion
            measurement_groups = defaultdict(list)
            for point in points:
                measurement_groups[point.measurement].append(point)
            
            # Write each measurement group
            for measurement, measurement_points in measurement_groups.items():
                count = await self._write_measurement_points(measurement, measurement_points)
                written_count += count
            
            # Update statistics
            with self.lock:
                self.stats['points_written'] += written_count
                self.stats['last_write_time'] = datetime.utcnow()
            
            logger.debug("Time series points written successfully", 
                        written=written_count, 
                        total=len(points))
            
            return written_count
            
        except Exception as e:
            logger.error("Time series points write failed", error=str(e))
            return 0
    
    async def query(self, 
                   measurement: str,
                   start_time: datetime,
                   end_time: datetime,
                   fields: Optional[List[str]] = None,
                   tags: Optional[Dict[str, str]] = None,
                   aggregation: Optional[AggregationConfig] = None,
                   limit: Optional[int] = None) -> QueryResult:
        """
        Query time series data.
        
        Args:
            measurement: Measurement name to query
            start_time: Start time for query
            end_time: End time for query
            fields: Optional field filters
            tags: Optional tag filters
            aggregation: Optional aggregation configuration
            limit: Optional result limit
            
        Returns:
            Query result with data
        """
        logger.debug("Querying time series data", 
                    measurement=measurement,
                    time_range=f"{start_time} to {end_time}")
        
        try:
            query_start = datetime.utcnow()
            
            # Check cache first
            cache_key = self._generate_cache_key(
                measurement, start_time, end_time, fields, tags, aggregation, limit
            )
            
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                with self.lock:
                    self.stats['cache_hits'] += 1
                return cached_result
            
            with self.lock:
                self.stats['cache_misses'] += 1
            
            # Execute query
            data = await self._execute_query(
                measurement, start_time, end_time, fields, tags, aggregation, limit
            )
            
            # Calculate query time
            query_time_ms = (datetime.utcnow() - query_start).total_seconds() * 1000
            
            # Create result
            result = QueryResult(
                measurement=measurement,
                data=data,
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'fields': fields,
                    'tags': tags,
                    'aggregation': aggregation,
                    'limit': limit,
                    'rows_returned': len(data)
                },
                query_time_ms=query_time_ms
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            with self.lock:
                self.stats['queries_executed'] += 1
                self.stats['last_query_time'] = datetime.utcnow()
            
            logger.debug("Time series query completed", 
                        measurement=measurement,
                        rows=len(data),
                        query_time_ms=query_time_ms)
            
            return result
            
        except Exception as e:
            logger.error("Time series query failed", 
                        error=str(e), 
                        measurement=measurement)
            return QueryResult(measurement=measurement, data=pd.DataFrame())
    
    async def query_ml_features(self,
                              measurements: List[str],
                              start_time: datetime,
                              end_time: datetime,
                              window: str = "1h",
                              aggregations: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query time series data optimized for ML feature extraction.
        
        Args:
            measurements: List of measurements to query
            start_time: Start time for query
            end_time: End time for query
            window: Aggregation window
            aggregations: List of aggregation functions
            
        Returns:
            DataFrame with ML-ready features
        """
        logger.info("Querying ML features", 
                   measurements=len(measurements),
                   time_range=f"{start_time} to {end_time}",
                   window=window)
        
        try:
            if not aggregations:
                aggregations = ['mean', 'min', 'max', 'std']
            
            # Query each measurement
            feature_dataframes = []
            
            for measurement in measurements:
                for agg_func in aggregations:
                    agg_config = AggregationConfig(
                        function=agg_func,
                        window=window,
                        fill="previous"
                    )
                    
                    result = await self.query(
                        measurement=measurement,
                        start_time=start_time,
                        end_time=end_time,
                        aggregation=agg_config
                    )
                    
                    if not result.data.empty:
                        # Rename columns to include measurement and aggregation
                        renamed_data = result.data.copy()
                        for col in renamed_data.columns:
                            if col != 'timestamp':
                                renamed_data = renamed_data.rename(
                                    columns={col: f"{measurement}_{col}_{agg_func}"}
                                )
                        
                        feature_dataframes.append(renamed_data)
            
            # Combine all features
            if feature_dataframes:
                # Merge on timestamp
                ml_features = feature_dataframes[0]
                for df in feature_dataframes[1:]:
                    ml_features = pd.merge(ml_features, df, on='timestamp', how='outer')
                
                # Sort by timestamp
                ml_features = ml_features.sort_values('timestamp')
                
                # Forward fill missing values
                ml_features = ml_features.ffill()
                
                logger.info("ML features query completed", 
                           rows=len(ml_features),
                           features=len(ml_features.columns) - 1)  # Exclude timestamp
                
                return ml_features
            else:
                logger.warning("No ML features found for query")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("ML features query failed", error=str(e))
            return pd.DataFrame()
    
    async def get_latest_values(self, 
                              measurement: str,
                              tags: Optional[Dict[str, str]] = None,
                              fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get latest values for a measurement.
        
        Args:
            measurement: Measurement name
            tags: Optional tag filters
            fields: Optional field filters
            
        Returns:
            Dictionary with latest values
        """
        logger.debug("Getting latest values", measurement=measurement)
        
        try:
            # Query last hour of data and get the most recent point
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            result = await self.query(
                measurement=measurement,
                start_time=start_time,
                end_time=end_time,
                fields=fields,
                tags=tags,
                limit=1
            )
            
            if result.data.empty:
                return {}
            
            # Get the latest row
            latest_row = result.data.iloc[-1]
            latest_values = latest_row.to_dict()
            
            logger.debug("Latest values retrieved", 
                        measurement=measurement,
                        values=len(latest_values))
            
            return latest_values
            
        except Exception as e:
            logger.error("Latest values retrieval failed", 
                        error=str(e), 
                        measurement=measurement)
            return {}
    
    async def get_measurement_statistics(self, 
                                       measurement: str,
                                       start_time: datetime,
                                       end_time: datetime) -> Dict[str, Any]:
        """
        Get statistics for a measurement over time period.
        
        Args:
            measurement: Measurement name
            start_time: Start time for statistics
            end_time: End time for statistics
            
        Returns:
            Dictionary with measurement statistics
        """
        logger.info("Getting measurement statistics", 
                   measurement=measurement,
                   time_range=f"{start_time} to {end_time}")
        
        try:
            # Query raw data
            result = await self.query(
                measurement=measurement,
                start_time=start_time,
                end_time=end_time
            )
            
            if result.data.empty:
                return {
                    'measurement': measurement,
                    'data_points': 0,
                    'time_range': {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat()
                    }
                }
            
            data = result.data
            
            # Calculate statistics
            statistics = {
                'measurement': measurement,
                'data_points': len(data),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'actual_start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                    'actual_end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
                },
                'field_statistics': {}
            }
            
            # Calculate statistics for numeric fields
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column != 'timestamp':
                    field_stats = {
                        'count': len(data[column].dropna()),
                        'mean': float(data[column].mean()),
                        'std': float(data[column].std()),
                        'min': float(data[column].min()),
                        'max': float(data[column].max()),
                        'median': float(data[column].median()),
                        'p95': float(data[column].quantile(0.95)),
                        'p99': float(data[column].quantile(0.99))
                    }
                    statistics['field_statistics'][column] = field_stats
            
            logger.info("Measurement statistics calculated", 
                       measurement=measurement,
                       data_points=statistics['data_points'],
                       fields=len(statistics['field_statistics']))
            
            return statistics
            
        except Exception as e:
            logger.error("Measurement statistics calculation failed", 
                        error=str(e), 
                        measurement=measurement)
            return {}
    
    async def cleanup_old_data(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up old data based on retention policies.
        
        Args:
            retention_days: Optional override for retention period
            
        Returns:
            Dictionary with cleanup statistics
        """
        logger.info("Cleaning up old time series data")
        
        try:
            cleanup_stats = {}
            total_deleted = 0
            
            # Get all measurements
            measurements = await self._get_all_measurements()
            
            for measurement in measurements:
                # Determine retention period
                if retention_days:
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                else:
                    retention_policy = self.retention_policies.get(
                        measurement, 
                        DataRetentionPolicy.MEDIUM_TERM
                    )
                    cutoff_date = self._calculate_cutoff_date(retention_policy)
                
                # Delete old data
                deleted_count = await self._delete_old_measurement_data(measurement, cutoff_date)
                cleanup_stats[measurement] = deleted_count
                total_deleted += deleted_count
            
            # Clean up query cache
            self._cleanup_query_cache()
            
            logger.info("Time series data cleanup completed", 
                       total_deleted=total_deleted,
                       measurements_cleaned=len(cleanup_stats))
            
            return {
                'total_deleted': total_deleted,
                'measurements_cleaned': cleanup_stats,
                'cleanup_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Time series data cleanup failed", error=str(e))
            return {}
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database metrics and statistics"""
        with self.lock:
            # Calculate cache statistics
            cache_hit_rate = 0.0
            total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            if total_cache_requests > 0:
                cache_hit_rate = self.stats['cache_hits'] / total_cache_requests
            
            return {
                'database_path': str(self.db_path),
                'database_size_bytes': self._get_database_size(),
                'retention_policies': {
                    measurement: policy.value 
                    for measurement, policy in self.retention_policies.items()
                },
                'configuration': {
                    'batch_size': self.batch_size,
                    'max_query_points': self.max_query_points,
                    'compression_enabled': self.compression_enabled,
                    'cache_ttl_minutes': self.cache_ttl_minutes,
                    'max_cache_size': self.max_cache_size
                },
                'statistics': {
                    **self.stats,
                    'cache_hit_rate': cache_hit_rate,
                    'cache_size': len(self.query_cache)
                }
            }
    
    async def _create_tables(self):
        """Create tables for time series measurements"""
        # Generic time series table structure
        # In production InfluxDB, this would be handled automatically
        
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_value REAL,
                field_value_str TEXT,
                field_value_bool INTEGER,
                tags TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_conn.commit()
    
    async def _create_indexes(self):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_measurement_time ON time_series_data(measurement, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_measurement_field ON time_series_data(measurement, field_name)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON time_series_data(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_measurement ON time_series_data(measurement)"
        ]
        
        for index_sql in indexes:
            self.db_conn.execute(index_sql)
        
        self.db_conn.commit()
    
    async def _write_measurement_points(self, 
                                      measurement: str, 
                                      points: List[TimeSeriesPoint]) -> int:
        """Write points for a specific measurement"""
        try:
            records = []
            
            for point in points:
                for field_name, field_value in point.fields.items():
                    # Determine field type and value
                    field_value_real = None
                    field_value_str = None
                    field_value_bool = None
                    
                    if isinstance(field_value, (int, float)):
                        field_value_real = float(field_value)
                    elif isinstance(field_value, bool):
                        field_value_bool = int(field_value)
                    else:
                        field_value_str = str(field_value)
                    
                    record = (
                        measurement,
                        point.timestamp.isoformat(),
                        field_name,
                        field_value_real,
                        field_value_str,
                        field_value_bool,
                        json.dumps(point.tags),
                        json.dumps(point.metadata)
                    )
                    records.append(record)
            
            # Batch insert
            cursor = self.db_conn.cursor()
            cursor.executemany("""
                INSERT INTO time_series_data 
                (measurement, timestamp, field_name, field_value, field_value_str, field_value_bool, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            self.db_conn.commit()
            
            return len(records)
            
        except Exception as e:
            logger.error("Measurement points write failed", 
                        error=str(e), 
                        measurement=measurement)
            self.db_conn.rollback()
            return 0
    
    async def _execute_query(self,
                           measurement: str,
                           start_time: datetime,
                           end_time: datetime,
                           fields: Optional[List[str]],
                           tags: Optional[Dict[str, str]],
                           aggregation: Optional[AggregationConfig],
                           limit: Optional[int]) -> pd.DataFrame:
        """Execute time series query"""
        try:
            # Build query
            query = """
                SELECT timestamp, field_name, field_value, field_value_str, field_value_bool, tags
                FROM time_series_data
                WHERE measurement = ? AND timestamp >= ? AND timestamp <= ?
            """
            params = [measurement, start_time.isoformat(), end_time.isoformat()]
            
            # Add field filters
            if fields:
                placeholders = ','.join(['?' for _ in fields])
                query += f" AND field_name IN ({placeholders})"
                params.extend(fields)
            
            # Add tag filters (simplified)
            if tags:
                for tag_key, tag_value in tags.items():
                    query += " AND tags LIKE ?"
                    params.append(f'%"{tag_key}":"{tag_value}"%')
            
            # Add ordering and limit
            query += " ORDER BY timestamp"
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            cursor = self.db_conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_records = []
            for row in rows:
                timestamp_str, field_name, field_value, field_value_str, field_value_bool, tags_json = row
                
                # Determine actual field value
                if field_value is not None:
                    value = field_value
                elif field_value_bool is not None:
                    value = bool(field_value_bool)
                else:
                    value = field_value_str
                
                data_records.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'field_name': field_name,
                    'value': value,
                    'tags': json.loads(tags_json) if tags_json else {}
                })
            
            df = pd.DataFrame(data_records)
            
            if df.empty:
                return df
            
            # Pivot to get fields as columns
            pivot_df = df.pivot_table(
                index='timestamp',
                columns='field_name',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Flatten column names
            pivot_df.columns.name = None
            
            # Apply aggregation if specified
            if aggregation:
                pivot_df = await self._apply_aggregation(pivot_df, aggregation)
            
            return pivot_df
            
        except Exception as e:
            logger.error("Query execution failed", error=str(e))
            return pd.DataFrame()
    
    async def _apply_aggregation(self, 
                               data: pd.DataFrame, 
                               aggregation: AggregationConfig) -> pd.DataFrame:
        """Apply aggregation to query results"""
        try:
            if data.empty or 'timestamp' not in data.columns:
                return data
            
            # Convert window to pandas frequency
            freq_map = {
                '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
                '1h': '1H', '2h': '2H', '6h': '6H', '12h': '12H',
                '1d': '1D', '1w': '1W', '1M': '1M'
            }
            
            freq = freq_map.get(aggregation.window, '1H')
            
            # Set timestamp as index
            data = data.set_index('timestamp')
            
            # Apply aggregation function
            if aggregation.function == 'mean':
                aggregated = data.resample(freq).mean()
            elif aggregation.function == 'sum':
                aggregated = data.resample(freq).sum()
            elif aggregation.function == 'min':
                aggregated = data.resample(freq).min()
            elif aggregation.function == 'max':
                aggregated = data.resample(freq).max()
            elif aggregation.function == 'count':
                aggregated = data.resample(freq).count()
            elif aggregation.function == 'std':
                aggregated = data.resample(freq).std()
            else:
                # Default to mean
                aggregated = data.resample(freq).mean()
            
            # Handle fill method
            if aggregation.fill == 'previous':
                aggregated = aggregated.ffill()
            elif aggregation.fill == 'linear':
                aggregated = aggregated.interpolate()
            elif aggregation.fill == 'zero':
                aggregated = aggregated.fillna(0)
            
            # Reset index
            aggregated = aggregated.reset_index()
            
            return aggregated
            
        except Exception as e:
            logger.error("Aggregation application failed", error=str(e))
            return data
    
    def _generate_cache_key(self, 
                          measurement: str,
                          start_time: datetime,
                          end_time: datetime,
                          fields: Optional[List[str]],
                          tags: Optional[Dict[str, str]],
                          aggregation: Optional[AggregationConfig],
                          limit: Optional[int]) -> str:
        """Generate cache key for query"""
        key_parts = [
            measurement,
            start_time.isoformat(),
            end_time.isoformat(),
            str(sorted(fields)) if fields else "all_fields",
            str(sorted(tags.items())) if tags else "no_tags",
            f"{aggregation.function}_{aggregation.window}" if aggregation else "no_agg",
            str(limit) if limit else "no_limit"
        ]
        
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result if valid"""
        with self.lock:
            if cache_key in self.query_cache:
                result, cached_at = self.query_cache[cache_key]
                
                # Check if cache is still valid
                age_minutes = (datetime.utcnow() - cached_at).total_seconds() / 60
                if age_minutes < self.cache_ttl_minutes:
                    return result
                else:
                    # Remove expired cache entry
                    del self.query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result"""
        with self.lock:
            # Check cache size limit
            if len(self.query_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k][1])
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = (result, datetime.utcnow())
    
    def _cleanup_query_cache(self):
        """Clean up expired cache entries"""
        with self.lock:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for cache_key, (result, cached_at) in self.query_cache.items():
                age_minutes = (current_time - cached_at).total_seconds() / 60
                if age_minutes >= self.cache_ttl_minutes:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.query_cache[key]
    
    async def _get_all_measurements(self) -> List[str]:
        """Get all measurement names"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT DISTINCT measurement FROM time_series_data")
        
        measurements = [row[0] for row in cursor.fetchall()]
        return measurements
    
    def _calculate_cutoff_date(self, retention_policy: DataRetentionPolicy) -> datetime:
        """Calculate cutoff date based on retention policy"""
        current_time = datetime.utcnow()
        
        if retention_policy == DataRetentionPolicy.REAL_TIME:
            return current_time - timedelta(hours=1)
        elif retention_policy == DataRetentionPolicy.SHORT_TERM:
            return current_time - timedelta(days=7)
        elif retention_policy == DataRetentionPolicy.MEDIUM_TERM:
            return current_time - timedelta(days=30)
        elif retention_policy == DataRetentionPolicy.LONG_TERM:
            return current_time - timedelta(days=365)
        else:  # PERMANENT
            return datetime.min
    
    async def _delete_old_measurement_data(self, 
                                         measurement: str, 
                                         cutoff_date: datetime) -> int:
        """Delete old data for a measurement"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                DELETE FROM time_series_data 
                WHERE measurement = ? AND timestamp < ?
            """, (measurement, cutoff_date.isoformat()))
            
            deleted_count = cursor.rowcount
            self.db_conn.commit()
            
            return deleted_count
            
        except Exception as e:
            logger.error("Old measurement data deletion failed", 
                        error=str(e), 
                        measurement=measurement)
            self.db_conn.rollback()
            return 0
    
    def _get_database_size(self) -> int:
        """Get database file size in bytes"""
        try:
            return self.db_path.stat().st_size if self.db_path.exists() else 0
        except Exception:
            return 0
    
    async def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
        
        logger.info("InfluxDB Time Series Database connection closed")