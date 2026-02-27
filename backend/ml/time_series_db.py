"""
Time Series Database for Cost Anomaly Detection

Specialized database layer for storing and querying time series cost data
optimized for ML workloads. Provides efficient storage, retrieval, and
aggregation of historical cost data for anomaly detection models.
"""

import asyncio
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date
from decimal import Decimal
from dataclasses import dataclass, asdict
import structlog
import json

logger = structlog.get_logger(__name__)


@dataclass
class TimeSeriesPoint:
    """Individual time series data point"""
    timestamp: datetime
    account_id: str
    service: str
    region: str
    metric_name: str
    value: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class AggregationConfig:
    """Configuration for time series aggregation"""
    interval: str  # '1h', '1d', '1w', '1m'
    aggregation_func: str  # 'sum', 'avg', 'min', 'max', 'count'
    fill_method: str = 'forward'  # 'forward', 'backward', 'zero', 'interpolate'


class TimeSeriesDB:
    """
    Time series database optimized for cost anomaly detection.
    
    Provides efficient storage and retrieval of cost time series data
    with support for multiple aggregation levels, fast queries, and
    ML-optimized data formats.
    """
    
    def __init__(self, db_path: str = "cost_timeseries.db"):
        self.db_path = db_path
        self.connection = None
        
        # Aggregation configurations
        self.aggregation_configs = {
            'hourly': AggregationConfig('1h', 'sum'),
            'daily': AggregationConfig('1d', 'sum'),
            'weekly': AggregationConfig('1w', 'sum'),
            'monthly': AggregationConfig('1m', 'sum')
        }
        
        # Query optimization settings
        self.batch_size = 10000
        self.index_columns = ['timestamp', 'account_id', 'service', 'region']
    
    async def initialize(self):
        """Initialize database and create tables"""
        logger.info("Initializing time series database", db_path=self.db_path)
        
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            self.connection.execute("PRAGMA synchronous=NORMAL")  # Balance safety and performance
            
            await self._create_tables()
            await self._create_indexes()
            
            logger.info("Time series database initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize time series database", error=str(e))
            raise
    
    async def store_cost_data(self, cost_points: List[TimeSeriesPoint]) -> int:
        """
        Store cost data points in time series format.
        
        Args:
            cost_points: List of time series data points
            
        Returns:
            Number of points stored
        """
        logger.info("Storing cost data points", count=len(cost_points))
        
        try:
            if not cost_points:
                return 0
            
            # Prepare data for batch insert
            records = []
            for point in cost_points:
                record = (
                    point.timestamp.isoformat(),
                    point.account_id,
                    point.service,
                    point.region,
                    point.metric_name,
                    point.value,
                    json.dumps(point.tags),
                    json.dumps(point.metadata)
                )
                records.append(record)
            
            # Batch insert
            cursor = self.connection.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO cost_timeseries 
                (timestamp, account_id, service, region, metric_name, value, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            self.connection.commit()
            
            logger.info("Cost data points stored successfully", stored=len(records))
            return len(records)
            
        except Exception as e:
            logger.error("Failed to store cost data points", error=str(e))
            self.connection.rollback()
            raise
    
    async def query_cost_data(self,
                            start_time: datetime,
                            end_time: datetime,
                            account_ids: Optional[List[str]] = None,
                            services: Optional[List[str]] = None,
                            regions: Optional[List[str]] = None,
                            metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query cost data for specified time range and filters.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            account_ids: Optional account ID filters
            services: Optional service filters
            regions: Optional region filters
            metric_names: Optional metric name filters
            
        Returns:
            DataFrame with cost time series data
        """
        logger.info(
            "Querying cost data",
            start_time=start_time,
            end_time=end_time,
            filters={
                'accounts': len(account_ids) if account_ids else 'all',
                'services': len(services) if services else 'all',
                'regions': len(regions) if regions else 'all',
                'metrics': len(metric_names) if metric_names else 'all'
            }
        )
        
        try:
            # Build query with filters
            query = """
                SELECT timestamp, account_id, service, region, metric_name, value, tags, metadata
                FROM cost_timeseries
                WHERE timestamp >= ? AND timestamp <= ?
            """
            params = [start_time.isoformat(), end_time.isoformat()]
            
            # Add filters
            if account_ids:
                placeholders = ','.join(['?' for _ in account_ids])
                query += f" AND account_id IN ({placeholders})"
                params.extend(account_ids)
            
            if services:
                placeholders = ','.join(['?' for _ in services])
                query += f" AND service IN ({placeholders})"
                params.extend(services)
            
            if regions:
                placeholders = ','.join(['?' for _ in regions])
                query += f" AND region IN ({placeholders})"
                params.extend(regions)
            
            if metric_names:
                placeholders = ','.join(['?' for _ in metric_names])
                query += f" AND metric_name IN ({placeholders})"
                params.extend(metric_names)
            
            query += " ORDER BY timestamp"
            
            # Execute query
            df = pd.read_sql_query(query, self.connection, params=params)
            
            # Parse JSON columns
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['tags'] = df['tags'].apply(json.loads)
                df['metadata'] = df['metadata'].apply(json.loads)
            
            logger.info("Cost data query completed", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to query cost data", error=str(e))
            raise
    
    async def aggregate_cost_data(self,
                                start_time: datetime,
                                end_time: datetime,
                                aggregation: str = 'daily',
                                group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate cost data over specified time periods.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            aggregation: Aggregation level ('hourly', 'daily', 'weekly', 'monthly')
            group_by: Optional grouping columns
            
        Returns:
            DataFrame with aggregated cost data
        """
        logger.info(
            "Aggregating cost data",
            aggregation=aggregation,
            time_range=f"{start_time} to {end_time}",
            group_by=group_by
        )
        
        try:
            # Get aggregation config
            if aggregation not in self.aggregation_configs:
                raise ValueError(f"Unsupported aggregation: {aggregation}")
            
            config = self.aggregation_configs[aggregation]
            
            # Build aggregation query
            time_format = self._get_time_format(aggregation)
            group_columns = ['time_bucket'] + (group_by or [])
            group_clause = ', '.join(group_columns)
            
            query = f"""
                SELECT 
                    strftime('{time_format}', timestamp) as time_bucket,
                    {', '.join(group_by) if group_by else ''}
                    {', ' if group_by else ''}
                    {config.aggregation_func.upper()}(value) as aggregated_value,
                    COUNT(*) as data_points
                FROM cost_timeseries
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY {group_clause}
                ORDER BY time_bucket
            """
            
            params = [start_time.isoformat(), end_time.isoformat()]
            
            # Execute aggregation query
            df = pd.read_sql_query(query, self.connection, params=params)
            
            if not df.empty:
                df['time_bucket'] = pd.to_datetime(df['time_bucket'])
            
            logger.info("Cost data aggregation completed", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to aggregate cost data", error=str(e))
            raise
    
    async def get_cost_statistics(self,
                                start_time: datetime,
                                end_time: datetime,
                                group_by: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get statistical summary of cost data.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            group_by: Optional grouping columns
            
        Returns:
            Dictionary with cost statistics
        """
        logger.info("Calculating cost statistics", time_range=f"{start_time} to {end_time}")
        
        try:
            group_clause = ', '.join(group_by) if group_by else "'total'"
            
            query = f"""
                SELECT 
                    {group_clause} as group_key,
                    COUNT(*) as data_points,
                    SUM(value) as total_cost,
                    AVG(value) as avg_cost,
                    MIN(value) as min_cost,
                    MAX(value) as max_cost,
                    (SELECT value FROM cost_timeseries 
                     WHERE timestamp >= ? AND timestamp <= ?
                     ORDER BY value LIMIT 1 OFFSET (COUNT(*) * 50 / 100)) as median_cost
                FROM cost_timeseries
                WHERE timestamp >= ? AND timestamp <= ?
                {f'GROUP BY {group_clause}' if group_by else ''}
            """
            
            params = [start_time.isoformat(), end_time.isoformat()] * 2
            
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            statistics = {}
            for row in results:
                group_key = row[0]
                statistics[group_key] = {
                    'data_points': row[1],
                    'total_cost': row[2],
                    'avg_cost': row[3],
                    'min_cost': row[4],
                    'max_cost': row[5],
                    'median_cost': row[6]
                }
            
            logger.info("Cost statistics calculated", groups=len(statistics))
            return statistics
            
        except Exception as e:
            logger.error("Failed to calculate cost statistics", error=str(e))
            raise
    
    async def get_ml_training_data(self,
                                 start_date: date,
                                 end_date: date,
                                 features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get cost data formatted for ML training.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            features: Optional list of features to include
            
        Returns:
            DataFrame optimized for ML training
        """
        logger.info("Preparing ML training data", date_range=f"{start_date} to {end_date}")
        
        try:
            # Query raw data
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date, datetime.max.time())
            
            df = await self.query_cost_data(start_time, end_time)
            
            if df.empty:
                return df
            
            # Pivot data for ML format
            ml_data = df.pivot_table(
                index='timestamp',
                columns=['service', 'region'],
                values='value',
                aggfunc='sum',
                fill_value=0
            )
            
            # Flatten column names
            ml_data.columns = [f"{service}_{region}" for service, region in ml_data.columns]
            
            # Add time-based features
            ml_data['hour'] = ml_data.index.hour
            ml_data['day_of_week'] = ml_data.index.dayofweek
            ml_data['day_of_month'] = ml_data.index.day
            ml_data['month'] = ml_data.index.month
            
            # Filter features if specified
            if features:
                available_features = [f for f in features if f in ml_data.columns]
                ml_data = ml_data[available_features]
            
            logger.info("ML training data prepared", rows=len(ml_data), features=len(ml_data.columns))
            return ml_data
            
        except Exception as e:
            logger.error("Failed to prepare ML training data", error=str(e))
            raise
    
    async def cleanup_old_data(self, retention_days: int = 365) -> int:
        """
        Clean up old time series data beyond retention period.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of records deleted
        """
        logger.info("Cleaning up old time series data", retention_days=retention_days)
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM cost_timeseries WHERE timestamp < ?",
                [cutoff_date.isoformat()]
            )
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            logger.info("Old time series data cleaned up", deleted_records=deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old time series data", error=str(e))
            self.connection.rollback()
            raise
    
    async def _create_tables(self):
        """Create time series tables"""
        cursor = self.connection.cursor()
        
        # Main time series table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                account_id TEXT NOT NULL,
                service TEXT NOT NULL,
                region TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, account_id, service, region, metric_name)
            )
        """)
        
        # Aggregated data tables for performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_timeseries_hourly (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_bucket TEXT NOT NULL,
                account_id TEXT NOT NULL,
                service TEXT NOT NULL,
                region TEXT NOT NULL,
                total_cost REAL NOT NULL,
                data_points INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(time_bucket, account_id, service, region)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_timeseries_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                account_id TEXT NOT NULL,
                service TEXT NOT NULL,
                region TEXT NOT NULL,
                total_cost REAL NOT NULL,
                data_points INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, account_id, service, region)
            )
        """)
        
        self.connection.commit()
    
    async def _create_indexes(self):
        """Create performance indexes"""
        cursor = self.connection.cursor()
        
        # Primary query indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_timeseries_timestamp ON cost_timeseries(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_account ON cost_timeseries(account_id)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_service ON cost_timeseries(service)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_region ON cost_timeseries(region)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_metric ON cost_timeseries(metric_name)",
            
            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_timeseries_time_account ON cost_timeseries(timestamp, account_id)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_time_service ON cost_timeseries(timestamp, service)",
            "CREATE INDEX IF NOT EXISTS idx_timeseries_account_service ON cost_timeseries(account_id, service)",
            
            # Aggregation table indexes
            "CREATE INDEX IF NOT EXISTS idx_hourly_time_bucket ON cost_timeseries_hourly(time_bucket)",
            "CREATE INDEX IF NOT EXISTS idx_daily_date ON cost_timeseries_daily(date)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
    
    def _get_time_format(self, aggregation: str) -> str:
        """Get SQLite time format for aggregation"""
        formats = {
            'hourly': '%Y-%m-%d %H:00:00',
            'daily': '%Y-%m-%d',
            'weekly': '%Y-%W',
            'monthly': '%Y-%m'
        }
        return formats.get(aggregation, '%Y-%m-%d')
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Time series database connection closed")


class DataPreprocessor:
    """
    Data preprocessing pipeline for cost anomaly detection.
    
    Handles data cleaning, normalization, and preparation for ML models.
    """
    
    def __init__(self):
        self.outlier_threshold = 3.0  # Z-score threshold for outlier detection
        self.missing_value_threshold = 0.1  # Maximum allowed missing value ratio
    
    def preprocess_cost_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw cost data for ML analysis.
        
        Args:
            raw_data: Raw cost data DataFrame
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        logger.info("Starting cost data preprocessing", rows=len(raw_data))
        
        try:
            if raw_data.empty:
                return raw_data
            
            # Make a copy to avoid modifying original data
            data = raw_data.copy()
            
            # Data cleaning steps
            data = self._handle_missing_values(data)
            data = self._remove_duplicates(data)
            data = self._handle_outliers(data)
            data = self._normalize_data_types(data)
            data = self._validate_data_quality(data)
            
            logger.info("Cost data preprocessing completed", final_rows=len(data))
            return data
            
        except Exception as e:
            logger.error("Cost data preprocessing failed", error=str(e))
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in cost data"""
        logger.debug("Handling missing values")
        
        # Check missing value ratio
        missing_ratio = data.isnull().sum() / len(data)
        high_missing_columns = missing_ratio[missing_ratio > self.missing_value_threshold].index
        
        if len(high_missing_columns) > 0:
            logger.warning("Columns with high missing values", columns=list(high_missing_columns))
        
        # Forward fill for time series data
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            data = data.ffill()
        
        # Fill remaining missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(0)
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        data[categorical_columns] = data[categorical_columns].fillna('unknown')
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        logger.debug("Removing duplicates")
        
        initial_count = len(data)
        
        # Define key columns for duplicate detection
        key_columns = ['timestamp', 'account_id', 'service', 'region']
        available_keys = [col for col in key_columns if col in data.columns]
        
        if available_keys:
            data = data.drop_duplicates(subset=available_keys, keep='last')
        else:
            data = data.drop_duplicates()
        
        removed_count = initial_count - len(data)
        if removed_count > 0:
            logger.info("Duplicates removed", count=removed_count)
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in cost data"""
        logger.debug("Handling outliers")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in data.columns:
                # Calculate Z-scores
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                
                # Identify outliers
                outliers = z_scores > self.outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info("Outliers detected", column=column, count=outlier_count)
                    
                    # Cap outliers at threshold percentiles
                    lower_bound = data[column].quantile(0.01)
                    upper_bound = data[column].quantile(0.99)
                    
                    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _normalize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data types for consistency"""
        logger.debug("Normalizing data types")
        
        # Convert timestamp columns
        timestamp_columns = ['timestamp', 'created_at', 'updated_at']
        for col in timestamp_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['cost_amount', 'usage_amount', 'value']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert categorical columns
        categorical_columns = ['service', 'region', 'account_id']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')
        
        return data
    
    def _validate_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and consistency"""
        logger.debug("Validating data quality")
        
        # Check for negative costs (should be rare)
        if 'cost_amount' in data.columns:
            negative_costs = data['cost_amount'] < 0
            if negative_costs.any():
                logger.warning("Negative costs detected", count=negative_costs.sum())
                data = data[~negative_costs]  # Remove negative costs
        
        # Check for future timestamps
        if 'timestamp' in data.columns:
            future_timestamps = data['timestamp'] > datetime.utcnow()
            if future_timestamps.any():
                logger.warning("Future timestamps detected", count=future_timestamps.sum())
                data = data[~future_timestamps]  # Remove future timestamps
        
        # Check data completeness
        required_columns = ['timestamp', 'service', 'cost_amount']
        missing_required = [col for col in required_columns if col not in data.columns]
        
        if missing_required:
            logger.error("Missing required columns", columns=missing_required)
            raise ValueError(f"Missing required columns: {missing_required}")
        
        return data