"""
Advanced Data Preprocessing and Normalization Pipeline

Provides comprehensive data preprocessing, normalization, feature scaling,
and data quality validation for ML workloads. Supports real-time and batch
processing with configurable preprocessing steps.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
import threading
from collections import defaultdict

logger = structlog.get_logger(__name__)


class PreprocessingStep(Enum):
    """Types of preprocessing steps"""
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_DETECTION = "outlier_detection"
    OUTLIER_TREATMENT = "outlier_treatment"
    FEATURE_SCALING = "feature_scaling"
    FEATURE_ENCODING = "feature_encoding"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    DATA_VALIDATION = "data_validation"
    TEMPORAL_FEATURES = "temporal_features"
    CUSTOM_TRANSFORMATION = "custom_transformation"


class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"          # StandardScaler
    MINMAX = "minmax"             # MinMaxScaler
    ROBUST = "robust"             # RobustScaler
    QUANTILE_UNIFORM = "quantile_uniform"  # QuantileTransformer uniform
    QUANTILE_NORMAL = "quantile_normal"    # QuantileTransformer normal
    POWER_YEO_JOHNSON = "power_yeo_johnson"  # PowerTransformer
    POWER_BOX_COX = "power_box_cox"        # PowerTransformer


class ImputationMethod(Enum):
    """Missing value imputation methods"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    KNN = "knn"
    DROP = "drop"


class OutlierMethod(Enum):
    """Outlier detection and treatment methods"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    CLIP = "clip"
    REMOVE = "remove"
    TRANSFORM = "transform"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Missing value handling
    imputation_method: ImputationMethod = ImputationMethod.MEDIAN
    imputation_constant: float = 0.0
    knn_neighbors: int = 5
    
    # Outlier handling
    outlier_detection_method: OutlierMethod = OutlierMethod.IQR
    outlier_treatment_method: OutlierMethod = OutlierMethod.CLIP
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Feature scaling
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    
    # Feature selection
    enable_feature_selection: bool = False
    feature_selection_k: int = 10
    feature_selection_method: str = "f_regression"
    
    # Dimensionality reduction
    enable_pca: bool = False
    pca_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    
    # Data validation
    enable_data_validation: bool = True
    max_missing_ratio: float = 0.3
    min_variance_threshold: float = 1e-6
    
    # Temporal features
    enable_temporal_features: bool = True
    temporal_lags: List[int] = field(default_factory=lambda: [1, 7, 30])
    temporal_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    
    # Custom transformations
    custom_transformations: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline"""
    processed_data: pd.DataFrame
    feature_names: List[str]
    preprocessing_steps: List[str]
    data_quality_report: Dict[str, Any]
    transformation_metadata: Dict[str, Any]
    processing_time_seconds: float
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_ratios: Dict[str, float]
    outliers_detected: Dict[str, int]
    data_types: Dict[str, str]
    duplicate_rows: int
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    quality_score: float
    recommendations: List[str]


class DataPreprocessingPipeline:
    """
    Advanced Data Preprocessing and Normalization Pipeline.
    
    Provides comprehensive data preprocessing including missing value imputation,
    outlier detection and treatment, feature scaling, encoding, selection,
    and dimensionality reduction for ML workloads.
    """
    
    def __init__(self, 
                 config: PreprocessingConfig = None,
                 cache_transformers: bool = True):
        self.config = config or PreprocessingConfig()
        self.cache_transformers = cache_transformers
        
        # Fitted transformers cache
        self.transformers: Dict[str, Any] = {}
        self.feature_metadata: Dict[str, Any] = {}
        
        # Processing statistics
        self.processing_stats = {
            'pipelines_executed': 0,
            'total_processing_time': 0.0,
            'last_processing_time': None,
            'average_processing_time': 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Data Preprocessing Pipeline initialized", config=self.config)
    
    async def fit_transform(self, 
                          data: pd.DataFrame,
                          target_column: Optional[str] = None,
                          feature_columns: Optional[List[str]] = None) -> PreprocessingResult:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            data: Input DataFrame to preprocess
            target_column: Optional target column name
            feature_columns: Optional list of feature columns to process
            
        Returns:
            Preprocessing result with transformed data
        """
        logger.info("Fitting and transforming data", 
                   rows=len(data), 
                   columns=len(data.columns))
        
        start_time = datetime.utcnow()
        
        try:
            # Validate input data
            if data.empty:
                raise ValueError("Input data is empty")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Create working copy
            processed_data = data.copy()
            original_shape = processed_data.shape
            
            # Initialize tracking
            preprocessing_steps = []
            transformation_metadata = {}
            
            # Step 1: Data quality assessment
            quality_report = await self._assess_data_quality(processed_data)
            preprocessing_steps.append("data_quality_assessment")
            
            # Step 2: Data validation
            if self.config.enable_data_validation:
                processed_data = await self._validate_data(processed_data, quality_report)
                preprocessing_steps.append("data_validation")
            
            # Step 3: Handle missing values
            processed_data, imputation_metadata = await self._handle_missing_values(
                processed_data, feature_columns
            )
            transformation_metadata['imputation'] = imputation_metadata
            preprocessing_steps.append("missing_value_imputation")
            
            # Step 4: Detect and treat outliers
            processed_data, outlier_metadata = await self._handle_outliers(
                processed_data, feature_columns
            )
            transformation_metadata['outliers'] = outlier_metadata
            preprocessing_steps.append("outlier_treatment")
            
            # Step 5: Feature encoding (categorical variables)
            processed_data, encoding_metadata = await self._encode_features(
                processed_data, feature_columns
            )
            transformation_metadata['encoding'] = encoding_metadata
            preprocessing_steps.append("feature_encoding")
            
            # Step 6: Generate temporal features
            if self.config.enable_temporal_features:
                processed_data, temporal_metadata = await self._generate_temporal_features(
                    processed_data
                )
                transformation_metadata['temporal'] = temporal_metadata
                preprocessing_steps.append("temporal_features")
            
            # Step 7: Feature scaling
            processed_data, scaling_metadata = await self._scale_features(
                processed_data, feature_columns
            )
            transformation_metadata['scaling'] = scaling_metadata
            preprocessing_steps.append("feature_scaling")
            
            # Step 8: Feature selection
            if self.config.enable_feature_selection and target_column:
                processed_data, selection_metadata = await self._select_features(
                    processed_data, target_column
                )
                transformation_metadata['feature_selection'] = selection_metadata
                preprocessing_steps.append("feature_selection")
            
            # Step 9: Dimensionality reduction
            if self.config.enable_pca:
                processed_data, pca_metadata = await self._apply_pca(processed_data)
                transformation_metadata['pca'] = pca_metadata
                preprocessing_steps.append("dimensionality_reduction")
            
            # Step 10: Apply custom transformations
            if self.config.custom_transformations:
                processed_data, custom_metadata = await self._apply_custom_transformations(
                    processed_data
                )
                transformation_metadata['custom'] = custom_metadata
                preprocessing_steps.append("custom_transformations")
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            with self.lock:
                self.processing_stats['pipelines_executed'] += 1
                self.processing_stats['total_processing_time'] += processing_time
                self.processing_stats['last_processing_time'] = datetime.utcnow()
                self.processing_stats['average_processing_time'] = (
                    self.processing_stats['total_processing_time'] / 
                    self.processing_stats['pipelines_executed']
                )
            
            # Create result
            result = PreprocessingResult(
                processed_data=processed_data,
                feature_names=list(processed_data.columns),
                preprocessing_steps=preprocessing_steps,
                data_quality_report=quality_report.__dict__,
                transformation_metadata=transformation_metadata,
                processing_time_seconds=processing_time,
                original_shape=original_shape,
                final_shape=processed_data.shape
            )
            
            logger.info("Data preprocessing completed", 
                       original_shape=original_shape,
                       final_shape=processed_data.shape,
                       processing_time=processing_time,
                       steps=len(preprocessing_steps))
            
            return result
            
        except Exception as e:
            logger.error("Data preprocessing failed", error=str(e))
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming new data", rows=len(data), columns=len(data.columns))
        
        try:
            if not self.transformers:
                raise ValueError("Pipeline not fitted. Call fit_transform first.")
            
            processed_data = data.copy()
            
            # Apply transformations in order
            if 'imputer' in self.transformers:
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    processed_data[numeric_columns] = self.transformers['imputer'].transform(
                        processed_data[numeric_columns]
                    )
            
            if 'scaler' in self.transformers:
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    processed_data[numeric_columns] = self.transformers['scaler'].transform(
                        processed_data[numeric_columns]
                    )
            
            if 'feature_selector' in self.transformers:
                selected_features = self.transformers['feature_selector'].get_support()
                feature_names = [col for i, col in enumerate(processed_data.columns) if selected_features[i]]
                processed_data = processed_data[feature_names]
            
            if 'pca' in self.transformers:
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    pca_features = self.transformers['pca'].transform(processed_data[numeric_columns])
                    pca_columns = [f'pca_{i}' for i in range(pca_features.shape[1])]
                    processed_data = pd.DataFrame(pca_features, columns=pca_columns, index=processed_data.index)
            
            logger.info("Data transformation completed", 
                       original_shape=data.shape,
                       final_shape=processed_data.shape)
            
            return processed_data
            
        except Exception as e:
            logger.error("Data transformation failed", error=str(e))
            raise
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Assess data quality and generate report"""
        logger.debug("Assessing data quality")
        
        try:
            # Basic statistics
            total_rows, total_columns = data.shape
            
            # Missing values
            missing_values = data.isnull().sum().to_dict()
            missing_ratios = {col: count / total_rows for col, count in missing_values.items()}
            
            # Data types
            data_types = data.dtypes.astype(str).to_dict()
            
            # Duplicate rows
            duplicate_rows = data.duplicated().sum()
            
            # Constant columns (no variance)
            constant_columns = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].nunique() <= 1:
                    constant_columns.append(col)
            
            # High cardinality columns
            high_cardinality_columns = []
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].nunique() > total_rows * 0.5:
                    high_cardinality_columns.append(col)
            
            # Outlier detection (simplified)
            outliers_detected = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outliers_detected[col] = outliers
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                missing_ratios, duplicate_rows, total_rows, constant_columns, 
                high_cardinality_columns, outliers_detected
            )
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                missing_ratios, duplicate_rows, constant_columns, 
                high_cardinality_columns, outliers_detected
            )
            
            report = DataQualityReport(
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values=missing_values,
                missing_ratios=missing_ratios,
                outliers_detected=outliers_detected,
                data_types=data_types,
                duplicate_rows=duplicate_rows,
                constant_columns=constant_columns,
                high_cardinality_columns=high_cardinality_columns,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            logger.debug("Data quality assessment completed", 
                        quality_score=quality_score,
                        recommendations=len(recommendations))
            
            return report
            
        except Exception as e:
            logger.error("Data quality assessment failed", error=str(e))
            raise
    
    async def _validate_data(self, 
                           data: pd.DataFrame, 
                           quality_report: DataQualityReport) -> pd.DataFrame:
        """Validate data and apply basic cleaning"""
        logger.debug("Validating data")
        
        try:
            validated_data = data.copy()
            
            # Remove columns with too many missing values
            high_missing_columns = [
                col for col, ratio in quality_report.missing_ratios.items()
                if ratio > self.config.max_missing_ratio
            ]
            
            if high_missing_columns:
                logger.warning("Removing columns with high missing values", 
                             columns=high_missing_columns)
                validated_data = validated_data.drop(columns=high_missing_columns)
            
            # Remove constant columns
            if quality_report.constant_columns:
                logger.warning("Removing constant columns", 
                             columns=quality_report.constant_columns)
                validated_data = validated_data.drop(columns=quality_report.constant_columns)
            
            # Remove duplicate rows
            if quality_report.duplicate_rows > 0:
                logger.warning("Removing duplicate rows", count=quality_report.duplicate_rows)
                validated_data = validated_data.drop_duplicates()
            
            logger.debug("Data validation completed", 
                        original_shape=data.shape,
                        validated_shape=validated_data.shape)
            
            return validated_data
            
        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            return data
    
    async def _handle_missing_values(self, 
                                   data: pd.DataFrame, 
                                   feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in the data"""
        logger.debug("Handling missing values", method=self.config.imputation_method.value)
        
        try:
            processed_data = data.copy()
            metadata = {'method': self.config.imputation_method.value}
            
            # Separate numeric and categorical columns
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
            
            # Filter to feature columns only
            numeric_features = [col for col in numeric_columns if col in feature_columns]
            categorical_features = [col for col in categorical_columns if col in feature_columns]
            
            # Handle numeric columns
            if numeric_features:
                if self.config.imputation_method == ImputationMethod.MEAN:
                    imputer = SimpleImputer(strategy='mean')
                elif self.config.imputation_method == ImputationMethod.MEDIAN:
                    imputer = SimpleImputer(strategy='median')
                elif self.config.imputation_method == ImputationMethod.CONSTANT:
                    imputer = SimpleImputer(strategy='constant', fill_value=self.config.imputation_constant)
                elif self.config.imputation_method == ImputationMethod.KNN:
                    imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                elif self.config.imputation_method == ImputationMethod.FORWARD_FILL:
                    processed_data[numeric_features] = processed_data[numeric_features].ffill()
                    imputer = None
                elif self.config.imputation_method == ImputationMethod.BACKWARD_FILL:
                    processed_data[numeric_features] = processed_data[numeric_features].bfill()
                    imputer = None
                elif self.config.imputation_method == ImputationMethod.INTERPOLATE:
                    processed_data[numeric_features] = processed_data[numeric_features].interpolate()
                    imputer = None
                elif self.config.imputation_method == ImputationMethod.DROP:
                    processed_data = processed_data.dropna(subset=numeric_features)
                    imputer = None
                else:
                    imputer = SimpleImputer(strategy='median')  # Default
                
                if imputer is not None:
                    processed_data[numeric_features] = imputer.fit_transform(processed_data[numeric_features])
                    if self.cache_transformers:
                        self.transformers['imputer'] = imputer
                
                metadata['numeric_features'] = numeric_features
                metadata['numeric_imputer'] = type(imputer).__name__ if imputer else 'pandas_method'
            
            # Handle categorical columns
            if categorical_features:
                # Use mode for categorical columns
                for col in categorical_features:
                    if processed_data[col].isnull().any():
                        mode_value = processed_data[col].mode()
                        if len(mode_value) > 0:
                            processed_data[col] = processed_data[col].fillna(mode_value[0])
                        else:
                            processed_data[col] = processed_data[col].fillna('unknown')
                
                metadata['categorical_features'] = categorical_features
                metadata['categorical_method'] = 'mode'
            
            # Calculate imputation statistics
            original_missing = data.isnull().sum().sum()
            final_missing = processed_data.isnull().sum().sum()
            metadata['missing_values_imputed'] = original_missing - final_missing
            
            logger.debug("Missing values handled", 
                        original_missing=original_missing,
                        final_missing=final_missing,
                        imputed=metadata['missing_values_imputed'])
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Missing value handling failed", error=str(e))
            return data, {}
    
    async def _handle_outliers(self, 
                             data: pd.DataFrame, 
                             feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and treat outliers in the data"""
        logger.debug("Handling outliers", 
                    detection_method=self.config.outlier_detection_method.value,
                    treatment_method=self.config.outlier_treatment_method.value)
        
        try:
            processed_data = data.copy()
            metadata = {
                'detection_method': self.config.outlier_detection_method.value,
                'treatment_method': self.config.outlier_treatment_method.value,
                'outliers_by_column': {}
            }
            
            # Only process numeric columns
            numeric_columns = [col for col in feature_columns 
                             if col in processed_data.select_dtypes(include=[np.number]).columns]
            
            for col in numeric_columns:
                column_data = processed_data[col].dropna()
                
                if len(column_data) == 0:
                    continue
                
                # Detect outliers
                if self.config.outlier_detection_method == OutlierMethod.Z_SCORE:
                    z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                    outlier_mask = z_scores > self.config.z_score_threshold
                elif self.config.outlier_detection_method == OutlierMethod.IQR:
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.iqr_multiplier * IQR
                    upper_bound = Q3 + self.config.iqr_multiplier * IQR
                    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
                else:
                    # Default to IQR
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.iqr_multiplier * IQR
                    upper_bound = Q3 + self.config.iqr_multiplier * IQR
                    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
                
                outlier_count = outlier_mask.sum()
                metadata['outliers_by_column'][col] = outlier_count
                
                if outlier_count > 0:
                    # Treat outliers
                    if self.config.outlier_treatment_method == OutlierMethod.CLIP:
                        # Clip to percentiles
                        lower_percentile = processed_data[col].quantile(0.01)
                        upper_percentile = processed_data[col].quantile(0.99)
                        processed_data[col] = processed_data[col].clip(
                            lower=lower_percentile, 
                            upper=upper_percentile
                        )
                    elif self.config.outlier_treatment_method == OutlierMethod.REMOVE:
                        # Remove outlier rows (be careful with this)
                        outlier_indices = column_data[outlier_mask].index
                        processed_data = processed_data.drop(outlier_indices)
                    # For TRANSFORM, we could apply log transformation or other methods
            
            total_outliers = sum(metadata['outliers_by_column'].values())
            metadata['total_outliers_detected'] = total_outliers
            
            logger.debug("Outliers handled", 
                        total_outliers=total_outliers,
                        columns_processed=len(numeric_columns))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Outlier handling failed", error=str(e))
            return data, {}
    
    async def _encode_features(self, 
                             data: pd.DataFrame, 
                             feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features"""
        logger.debug("Encoding categorical features")
        
        try:
            processed_data = data.copy()
            metadata = {'encoded_columns': [], 'encoders': {}}
            
            # Find categorical columns
            categorical_columns = [col for col in feature_columns 
                                 if col in processed_data.select_dtypes(include=['object']).columns]
            
            for col in categorical_columns:
                unique_values = processed_data[col].nunique()
                
                # Use one-hot encoding for low cardinality, label encoding for high cardinality
                if unique_values <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
                    processed_data = processed_data.drop(columns=[col])
                    
                    metadata['encoded_columns'].append(col)
                    metadata['encoders'][col] = {
                        'type': 'one_hot',
                        'categories': list(processed_data[col].unique())
                    }
                else:
                    # Label encoding
                    encoder = LabelEncoder()
                    processed_data[col] = encoder.fit_transform(processed_data[col].astype(str))
                    
                    if self.cache_transformers:
                        self.transformers[f'encoder_{col}'] = encoder
                    
                    metadata['encoded_columns'].append(col)
                    metadata['encoders'][col] = {
                        'type': 'label',
                        'classes': list(encoder.classes_)
                    }
            
            logger.debug("Feature encoding completed", 
                        encoded_columns=len(metadata['encoded_columns']))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Feature encoding failed", error=str(e))
            return data, {}
    
    async def _generate_temporal_features(self, 
                                        data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate temporal features from datetime columns"""
        logger.debug("Generating temporal features")
        
        try:
            processed_data = data.copy()
            metadata = {'temporal_features_added': [], 'datetime_columns': []}
            
            # Find datetime columns
            datetime_columns = processed_data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Also check for columns that might be datetime strings
            for col in processed_data.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(processed_data[col].dropna().iloc[:100])  # Test first 100 values
                    processed_data[col] = pd.to_datetime(processed_data[col])
                    datetime_columns.append(col)
                except:
                    continue
            
            for col in datetime_columns:
                metadata['datetime_columns'].append(col)
                
                # Extract basic temporal features
                processed_data[f'{col}_year'] = processed_data[col].dt.year
                processed_data[f'{col}_month'] = processed_data[col].dt.month
                processed_data[f'{col}_day'] = processed_data[col].dt.day
                processed_data[f'{col}_hour'] = processed_data[col].dt.hour
                processed_data[f'{col}_dayofweek'] = processed_data[col].dt.dayofweek
                processed_data[f'{col}_quarter'] = processed_data[col].dt.quarter
                
                # Cyclical encoding
                processed_data[f'{col}_month_sin'] = np.sin(2 * np.pi * processed_data[f'{col}_month'] / 12)
                processed_data[f'{col}_month_cos'] = np.cos(2 * np.pi * processed_data[f'{col}_month'] / 12)
                processed_data[f'{col}_hour_sin'] = np.sin(2 * np.pi * processed_data[f'{col}_hour'] / 24)
                processed_data[f'{col}_hour_cos'] = np.cos(2 * np.pi * processed_data[f'{col}_hour'] / 24)
                processed_data[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * processed_data[f'{col}_dayofweek'] / 7)
                processed_data[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * processed_data[f'{col}_dayofweek'] / 7)
                
                temporal_features = [
                    f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_hour', 
                    f'{col}_dayofweek', f'{col}_quarter',
                    f'{col}_month_sin', f'{col}_month_cos',
                    f'{col}_hour_sin', f'{col}_hour_cos',
                    f'{col}_dayofweek_sin', f'{col}_dayofweek_cos'
                ]
                
                metadata['temporal_features_added'].extend(temporal_features)
            
            logger.debug("Temporal features generated", 
                        datetime_columns=len(datetime_columns),
                        features_added=len(metadata['temporal_features_added']))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Temporal feature generation failed", error=str(e))
            return data, {}
    
    async def _scale_features(self, 
                            data: pd.DataFrame, 
                            feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features"""
        logger.debug("Scaling features", method=self.config.scaling_method.value)
        
        try:
            processed_data = data.copy()
            metadata = {'scaling_method': self.config.scaling_method.value}
            
            # Only scale numeric columns
            numeric_columns = [col for col in processed_data.columns 
                             if col in processed_data.select_dtypes(include=[np.number]).columns]
            
            if not numeric_columns:
                return processed_data, metadata
            
            # Choose scaler
            if self.config.scaling_method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif self.config.scaling_method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif self.config.scaling_method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
            elif self.config.scaling_method == ScalingMethod.QUANTILE_UNIFORM:
                scaler = QuantileTransformer(output_distribution='uniform')
            elif self.config.scaling_method == ScalingMethod.QUANTILE_NORMAL:
                scaler = QuantileTransformer(output_distribution='normal')
            elif self.config.scaling_method == ScalingMethod.POWER_YEO_JOHNSON:
                scaler = PowerTransformer(method='yeo-johnson')
            elif self.config.scaling_method == ScalingMethod.POWER_BOX_COX:
                scaler = PowerTransformer(method='box-cox')
            else:
                scaler = RobustScaler()  # Default
            
            # Fit and transform
            processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            
            if self.cache_transformers:
                self.transformers['scaler'] = scaler
            
            metadata['scaled_columns'] = numeric_columns
            metadata['scaler_type'] = type(scaler).__name__
            
            logger.debug("Feature scaling completed", 
                        scaled_columns=len(numeric_columns),
                        scaler=type(scaler).__name__)
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Feature scaling failed", error=str(e))
            return data, {}
    
    async def _select_features(self, 
                             data: pd.DataFrame, 
                             target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select most important features"""
        logger.debug("Selecting features", k=self.config.feature_selection_k)
        
        try:
            if target_column not in data.columns:
                logger.warning("Target column not found, skipping feature selection")
                return data, {}
            
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Only use numeric features for selection
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) <= self.config.feature_selection_k:
                logger.debug("Number of features <= k, skipping feature selection")
                return data, {}
            
            X_numeric = X[numeric_features]
            
            # Choose selection method
            if self.config.feature_selection_method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.feature_selection_k)
            else:
                selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            
            # Fit selector
            X_selected = selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Create new dataframe with selected features and target
            processed_data = pd.DataFrame(X_selected, columns=selected_features, index=data.index)
            processed_data[target_column] = y
            
            # Add non-numeric features back
            non_numeric_features = [col for col in X.columns if col not in numeric_features]
            for col in non_numeric_features:
                processed_data[col] = X[col]
            
            if self.cache_transformers:
                self.transformers['feature_selector'] = selector
            
            metadata = {
                'selection_method': self.config.feature_selection_method,
                'k': self.config.feature_selection_k,
                'selected_features': selected_features,
                'feature_scores': dict(zip(numeric_features, selector.scores_))
            }
            
            logger.debug("Feature selection completed", 
                        original_features=len(numeric_features),
                        selected_features=len(selected_features))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Feature selection failed", error=str(e))
            return data, {}
    
    async def _apply_pca(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Principal Component Analysis"""
        logger.debug("Applying PCA")
        
        try:
            # Only apply PCA to numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                logger.debug("Not enough numeric features for PCA")
                return data, {}
            
            X_numeric = data[numeric_columns]
            
            # Determine number of components
            if self.config.pca_components:
                n_components = min(self.config.pca_components, len(numeric_columns))
            else:
                # Use variance threshold
                pca_temp = PCA()
                pca_temp.fit(X_numeric)
                cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.config.pca_variance_threshold) + 1
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_numeric)
            
            # Create new dataframe with PCA components
            pca_columns = [f'pca_{i}' for i in range(n_components)]
            processed_data = pd.DataFrame(X_pca, columns=pca_columns, index=data.index)
            
            # Add non-numeric columns back
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in non_numeric_columns:
                processed_data[col] = data[col]
            
            if self.cache_transformers:
                self.transformers['pca'] = pca
            
            metadata = {
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'original_features': numeric_columns
            }
            
            logger.debug("PCA applied", 
                        original_features=len(numeric_columns),
                        pca_components=n_components,
                        variance_explained=np.sum(pca.explained_variance_ratio_))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("PCA application failed", error=str(e))
            return data, {}
    
    async def _apply_custom_transformations(self, 
                                          data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply custom transformations"""
        logger.debug("Applying custom transformations", 
                    count=len(self.config.custom_transformations))
        
        try:
            processed_data = data.copy()
            metadata = {'transformations_applied': []}
            
            for name, transformation_func in self.config.custom_transformations.items():
                try:
                    processed_data = transformation_func(processed_data)
                    metadata['transformations_applied'].append(name)
                    logger.debug("Applied custom transformation", name=name)
                except Exception as e:
                    logger.error("Custom transformation failed", 
                               name=name, 
                               error=str(e))
            
            return processed_data, metadata
            
        except Exception as e:
            logger.error("Custom transformations failed", error=str(e))
            return data, {}
    
    def _calculate_quality_score(self, 
                               missing_ratios: Dict[str, float],
                               duplicate_rows: int,
                               total_rows: int,
                               constant_columns: List[str],
                               high_cardinality_columns: List[str],
                               outliers_detected: Dict[str, int]) -> float:
        """Calculate overall data quality score"""
        try:
            score = 1.0
            
            # Penalize missing values
            avg_missing_ratio = np.mean(list(missing_ratios.values()))
            score -= avg_missing_ratio * 0.3
            
            # Penalize duplicates
            duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0
            score -= duplicate_ratio * 0.2
            
            # Penalize constant columns
            if constant_columns:
                score -= len(constant_columns) * 0.1
            
            # Penalize high cardinality columns
            if high_cardinality_columns:
                score -= len(high_cardinality_columns) * 0.05
            
            # Penalize outliers
            total_outliers = sum(outliers_detected.values())
            outlier_ratio = total_outliers / total_rows if total_rows > 0 else 0
            score -= outlier_ratio * 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _generate_quality_recommendations(self, 
                                        missing_ratios: Dict[str, float],
                                        duplicate_rows: int,
                                        constant_columns: List[str],
                                        high_cardinality_columns: List[str],
                                        outliers_detected: Dict[str, int]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Missing values
        high_missing_columns = [col for col, ratio in missing_ratios.items() if ratio > 0.3]
        if high_missing_columns:
            recommendations.append(f"Consider removing columns with high missing values: {high_missing_columns}")
        
        # Duplicates
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")
        
        # Constant columns
        if constant_columns:
            recommendations.append(f"Remove constant columns: {constant_columns}")
        
        # High cardinality
        if high_cardinality_columns:
            recommendations.append(f"Consider feature engineering for high cardinality columns: {high_cardinality_columns}")
        
        # Outliers
        high_outlier_columns = [col for col, count in outliers_detected.items() if count > 10]
        if high_outlier_columns:
            recommendations.append(f"Investigate outliers in columns: {high_outlier_columns}")
        
        return recommendations
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get preprocessing pipeline metrics"""
        with self.lock:
            return {
                'configuration': {
                    'imputation_method': self.config.imputation_method.value,
                    'scaling_method': self.config.scaling_method.value,
                    'outlier_detection_method': self.config.outlier_detection_method.value,
                    'enable_feature_selection': self.config.enable_feature_selection,
                    'enable_pca': self.config.enable_pca,
                    'enable_temporal_features': self.config.enable_temporal_features
                },
                'statistics': self.processing_stats.copy(),
                'cached_transformers': list(self.transformers.keys()),
                'feature_metadata': self.feature_metadata.copy()
            }
    
    def save_transformers(self, file_path: str):
        """Save fitted transformers to file"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'transformers': self.transformers,
                    'feature_metadata': self.feature_metadata,
                    'config': self.config
                }, f)
            
            logger.info("Transformers saved", file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to save transformers", error=str(e))
    
    def load_transformers(self, file_path: str):
        """Load fitted transformers from file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                self.transformers = data['transformers']
                self.feature_metadata = data['feature_metadata']
                self.config = data['config']
            
            logger.info("Transformers loaded", file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to load transformers", error=str(e))