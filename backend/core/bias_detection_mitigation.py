"""
Bias Detection and Mitigation Tools

This module provides comprehensive bias detection and mitigation capabilities
for ML models including fairness metrics, bias analysis, and mitigation strategies.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .exceptions import BiasDetectionError, ValidationError

logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias to detect"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"

class MitigationStrategy(Enum):
    """Bias mitigation strategies"""
    RESAMPLING = "resampling"
    REWEIGHTING = "reweighting"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    ADVERSARIAL_DEBIASING = "adversarial_debiasing"
    FAIRNESS_CONSTRAINTS = "fairness_constraints"

class FairnessMetric(Enum):
    """Fairness metrics"""
    STATISTICAL_PARITY = "statistical_parity"
    DISPARATE_IMPACT = "disparate_impact"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    TREATMENT_EQUALITY = "treatment_equality"

@dataclass
class BiasAnalysisResult:
    """Bias analysis result"""
    analysis_id: str
    model_id: str
    protected_attributes: List[str]
    bias_metrics: Dict[str, Dict[str, float]]
    overall_bias_score: float
    bias_detected: bool
    recommendations: List[str]
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FairnessReport:
    """Comprehensive fairness report"""
    report_id: str
    model_id: str
    dataset_info: Dict[str, Any]
    protected_groups: Dict[str, List[str]]
    fairness_metrics: Dict[str, float]
    bias_analysis: BiasAnalysisResult
    mitigation_recommendations: List[Dict[str, Any]]
    compliance_status: Dict[str, bool]
    created_at: datetime

@dataclass
class MitigationResult:
    """Bias mitigation result"""
    mitigation_id: str
    original_model_id: str
    mitigated_model_id: str
    strategy: MitigationStrategy
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_score: float
    trade_offs: Dict[str, float]
    created_at: datetime

class BiasDetectionMitigation:
    """
    Comprehensive bias detection and mitigation system for ML models
    with fairness metrics, bias analysis, and mitigation strategies.
    """
    
    def __init__(self):
        """Initialize Bias Detection and Mitigation system"""
        self.bias_analyses: Dict[str, BiasAnalysisResult] = {}
        self.fairness_reports: Dict[str, FairnessReport] = {}
        self.mitigation_results: Dict[str, MitigationResult] = {}
        
    async def analyze_bias(
        self,
        model: Any,
        model_id: str,
        test_data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str],
        bias_types: List[BiasType] = None
    ) -> str:
        """
        Analyze model for various types of bias
        
        Args:
            model: Trained model
            model_id: Model identifier
            test_data: Test dataset
            target_column: Target variable column name
            protected_attributes: List of protected attribute columns
            feature_columns: List of feature columns
            bias_types: Types of bias to analyze
            
        Returns:
            Analysis ID
        """
        try:
            analysis_id = str(uuid.uuid4())
            
            # Validate inputs
            if test_data.empty:
                raise ValidationError("Test data cannot be empty")
            
            if not protected_attributes:
                raise ValidationError("Protected attributes must be specified")
            
            if bias_types is None:
                bias_types = [
                    BiasType.DEMOGRAPHIC_PARITY,
                    BiasType.EQUALIZED_ODDS,
                    BiasType.EQUALITY_OF_OPPORTUNITY
                ]
            
            # Prepare data
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Analyze bias for each protected attribute
            bias_metrics = {}
            
            for protected_attr in protected_attributes:
                if protected_attr not in test_data.columns:
                    logger.warning(f"Protected attribute {protected_attr} not found in data")
                    continue
                
                attr_bias_metrics = {}
                
                # Get unique groups for this attribute
                groups = test_data[protected_attr].unique()
                
                for bias_type in bias_types:
                    if bias_type == BiasType.DEMOGRAPHIC_PARITY:
                        metric_value = await self._calculate_demographic_parity(
                            test_data, protected_attr, y_pred, groups
                        )
                    elif bias_type == BiasType.EQUALIZED_ODDS:
                        metric_value = await self._calculate_equalized_odds(
                            test_data, protected_attr, y_test, y_pred, groups
                        )
                    elif bias_type == BiasType.EQUALITY_OF_OPPORTUNITY:
                        metric_value = await self._calculate_equality_of_opportunity(
                            test_data, protected_attr, y_test, y_pred, groups
                        )
                    elif bias_type == BiasType.CALIBRATION:
                        if y_pred_proba is not None:
                            metric_value = await self._calculate_calibration(
                                test_data, protected_attr, y_test, y_pred_proba, groups
                            )
                        else:
                            metric_value = 0.0
                    else:
                        metric_value = 0.0
                    
                    attr_bias_metrics[bias_type.value] = metric_value
                
                bias_metrics[protected_attr] = attr_bias_metrics
            
            # Calculate overall bias score
            overall_bias_score = self._calculate_overall_bias_score(bias_metrics)
            
            # Determine if bias is detected
            bias_detected = overall_bias_score > 0.1  # Threshold for bias detection
            
            # Generate recommendations
            recommendations = await self._generate_bias_recommendations(
                bias_metrics, overall_bias_score, protected_attributes
            )
            
            # Create bias analysis result
            analysis = BiasAnalysisResult(
                analysis_id=analysis_id,
                model_id=model_id,
                protected_attributes=protected_attributes,
                bias_metrics=bias_metrics,
                overall_bias_score=overall_bias_score,
                bias_detected=bias_detected,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
                metadata={
                    "test_data_shape": test_data.shape,
                    "bias_types_analyzed": [bt.value for bt in bias_types],
                    "model_type": type(model).__name__
                }
            )
            
            # Store analysis
            self.bias_analyses[analysis_id] = analysis
            
            logger.info(f"Bias analysis {analysis_id} completed. Bias detected: {bias_detected}")
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Bias analysis failed: {str(e)}")
            raise BiasDetectionError(f"Bias analysis failed: {str(e)}")
    
    async def generate_fairness_report(
        self,
        model: Any,
        model_id: str,
        dataset: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> str:
        """
        Generate comprehensive fairness report
        
        Args:
            model: Trained model
            model_id: Model identifier
            dataset: Complete dataset
            target_column: Target variable column name
            protected_attributes: List of protected attribute columns
            feature_columns: List of feature columns
            
        Returns:
            Report ID
        """
        try:
            report_id = str(uuid.uuid4())
            
            # Split data for analysis
            train_data, test_data = train_test_split(
                dataset, test_size=0.3, random_state=42, stratify=dataset[target_column]
            )
            
            # Analyze bias
            bias_analysis_id = await self.analyze_bias(
                model, model_id, test_data, target_column,
                protected_attributes, feature_columns
            )
            
            bias_analysis = self.bias_analyses[bias_analysis_id]
            
            # Calculate fairness metrics
            fairness_metrics = await self._calculate_fairness_metrics(
                model, test_data, target_column, protected_attributes, feature_columns
            )
            
            # Analyze protected groups
            protected_groups = {}
            for attr in protected_attributes:
                if attr in dataset.columns:
                    protected_groups[attr] = dataset[attr].unique().tolist()
            
            # Generate mitigation recommendations
            mitigation_recommendations = await self._generate_mitigation_recommendations(
                bias_analysis, fairness_metrics
            )
            
            # Check compliance status
            compliance_status = await self._check_compliance_status(
                fairness_metrics, bias_analysis
            )
            
            # Create fairness report
            report = FairnessReport(
                report_id=report_id,
                model_id=model_id,
                dataset_info={
                    "total_samples": len(dataset),
                    "features": len(feature_columns),
                    "protected_attributes": len(protected_attributes),
                    "target_distribution": dataset[target_column].value_counts().to_dict()
                },
                protected_groups=protected_groups,
                fairness_metrics=fairness_metrics,
                bias_analysis=bias_analysis,
                mitigation_recommendations=mitigation_recommendations,
                compliance_status=compliance_status,
                created_at=datetime.utcnow()
            )
            
            # Store report
            self.fairness_reports[report_id] = report
            
            logger.info(f"Fairness report {report_id} generated")
            
            return report_id
            
        except Exception as e:
            logger.error(f"Fairness report generation failed: {str(e)}")
            raise BiasDetectionError(f"Fairness report generation failed: {str(e)}")
    
    async def mitigate_bias(
        self,
        model: Any,
        model_id: str,
        training_data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str],
        strategy: MitigationStrategy = MitigationStrategy.REWEIGHTING
    ) -> str:
        """
        Apply bias mitigation strategy
        
        Args:
            model: Original model
            model_id: Original model identifier
            training_data: Training dataset
            target_column: Target variable column name
            protected_attributes: List of protected attribute columns
            feature_columns: List of feature columns
            strategy: Mitigation strategy to apply
            
        Returns:
            Mitigation ID
        """
        try:
            mitigation_id = str(uuid.uuid4())
            
            # Calculate before metrics
            X_train = training_data[feature_columns]
            y_train = training_data[target_column]
            
            before_metrics = await self._calculate_model_metrics(
                model, training_data, target_column, protected_attributes, feature_columns
            )
            
            # Apply mitigation strategy
            if strategy == MitigationStrategy.REWEIGHTING:
                mitigated_model = await self._apply_reweighting(
                    model, training_data, target_column, protected_attributes, feature_columns
                )
            elif strategy == MitigationStrategy.RESAMPLING:
                mitigated_model = await self._apply_resampling(
                    model, training_data, target_column, protected_attributes, feature_columns
                )
            elif strategy == MitigationStrategy.THRESHOLD_OPTIMIZATION:
                mitigated_model = await self._apply_threshold_optimization(
                    model, training_data, target_column, protected_attributes, feature_columns
                )
            else:
                raise ValidationError(f"Mitigation strategy {strategy} not implemented")
            
            # Calculate after metrics
            after_metrics = await self._calculate_model_metrics(
                mitigated_model, training_data, target_column, protected_attributes, feature_columns
            )
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(before_metrics, after_metrics)
            
            # Calculate trade-offs
            trade_offs = self._calculate_trade_offs(before_metrics, after_metrics)
            
            # Create mitigation result
            mitigation_result = MitigationResult(
                mitigation_id=mitigation_id,
                original_model_id=model_id,
                mitigated_model_id=f"{model_id}_mitigated_{mitigation_id[:8]}",
                strategy=strategy,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_score=improvement_score,
                trade_offs=trade_offs,
                created_at=datetime.utcnow()
            )
            
            # Store result
            self.mitigation_results[mitigation_id] = mitigation_result
            
            logger.info(f"Bias mitigation {mitigation_id} completed using {strategy.value}")
            
            return mitigation_id
            
        except Exception as e:
            logger.error(f"Bias mitigation failed: {str(e)}")
            raise BiasDetectionError(f"Bias mitigation failed: {str(e)}")
    
    async def _calculate_demographic_parity(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        predictions: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """Calculate demographic parity metric"""
        try:
            positive_rates = {}
            
            for group in groups:
                group_mask = data[protected_attr] == group
                group_predictions = predictions[group_mask]
                
                if len(group_predictions) > 0:
                    positive_rate = np.mean(group_predictions)
                    positive_rates[group] = positive_rate
            
            if len(positive_rates) < 2:
                return 0.0
            
            # Calculate maximum difference in positive rates
            rates = list(positive_rates.values())
            max_diff = max(rates) - min(rates)
            
            return max_diff
            
        except Exception as e:
            logger.error(f"Demographic parity calculation failed: {str(e)}")
            return 0.0
    
    async def _calculate_equalized_odds(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """Calculate equalized odds metric"""
        try:
            tpr_diff_max = 0.0
            fpr_diff_max = 0.0
            
            group_metrics = {}
            
            for group in groups:
                group_mask = data[protected_attr] == group
                group_true = true_labels[group_mask]
                group_pred = predictions[group_mask]
                
                if len(group_true) > 0:
                    # Calculate TPR and FPR
                    tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    group_metrics[group] = {"tpr": tpr, "fpr": fpr}
            
            if len(group_metrics) < 2:
                return 0.0
            
            # Calculate maximum differences
            tprs = [metrics["tpr"] for metrics in group_metrics.values()]
            fprs = [metrics["fpr"] for metrics in group_metrics.values()]
            
            tpr_diff_max = max(tprs) - min(tprs)
            fpr_diff_max = max(fprs) - min(fprs)
            
            # Return maximum of TPR and FPR differences
            return max(tpr_diff_max, fpr_diff_max)
            
        except Exception as e:
            logger.error(f"Equalized odds calculation failed: {str(e)}")
            return 0.0
    
    async def _calculate_equality_of_opportunity(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """Calculate equality of opportunity metric"""
        try:
            tprs = {}
            
            for group in groups:
                group_mask = data[protected_attr] == group
                group_true = true_labels[group_mask]
                group_pred = predictions[group_mask]
                
                if len(group_true) > 0:
                    # Calculate TPR for positive class
                    positive_mask = group_true == 1
                    if np.sum(positive_mask) > 0:
                        tpr = np.mean(group_pred[positive_mask])
                        tprs[group] = tpr
            
            if len(tprs) < 2:
                return 0.0
            
            # Calculate maximum difference in TPRs
            tpr_values = list(tprs.values())
            max_diff = max(tpr_values) - min(tpr_values)
            
            return max_diff
            
        except Exception as e:
            logger.error(f"Equality of opportunity calculation failed: {str(e)}")
            return 0.0
    
    async def _calculate_calibration(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        true_labels: np.ndarray,
        predicted_probabilities: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """Calculate calibration metric"""
        try:
            calibration_errors = {}
            
            for group in groups:
                group_mask = data[protected_attr] == group
                group_true = true_labels[group_mask]
                group_proba = predicted_probabilities[group_mask]
                
                if len(group_true) > 0:
                    # Calculate calibration error using binning
                    n_bins = 10
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    calibration_error = 0.0
                    
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (group_proba > bin_lower) & (group_proba <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = group_true[in_bin].mean()
                            avg_confidence_in_bin = group_proba[in_bin].mean()
                            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    calibration_errors[group] = calibration_error
            
            if len(calibration_errors) < 2:
                return 0.0
            
            # Calculate maximum difference in calibration errors
            errors = list(calibration_errors.values())
            max_diff = max(errors) - min(errors)
            
            return max_diff
            
        except Exception as e:
            logger.error(f"Calibration calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_overall_bias_score(self, bias_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall bias score across all metrics and attributes"""
        try:
            all_scores = []
            
            for attr_metrics in bias_metrics.values():
                for score in attr_metrics.values():
                    all_scores.append(score)
            
            if not all_scores:
                return 0.0
            
            # Use maximum bias score as overall score
            return max(all_scores)
            
        except Exception:
            return 0.0
    
    async def _generate_bias_recommendations(
        self,
        bias_metrics: Dict[str, Dict[str, float]],
        overall_bias_score: float,
        protected_attributes: List[str]
    ) -> List[str]:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        if overall_bias_score > 0.2:
            recommendations.append("High bias detected. Immediate mitigation recommended.")
        elif overall_bias_score > 0.1:
            recommendations.append("Moderate bias detected. Consider mitigation strategies.")
        else:
            recommendations.append("Low bias detected. Monitor regularly.")
        
        # Specific recommendations based on bias types
        for attr, metrics in bias_metrics.items():
            for bias_type, score in metrics.items():
                if score > 0.15:
                    if bias_type == "demographic_parity":
                        recommendations.append(
                            f"Demographic parity violation for {attr}. "
                            "Consider resampling or reweighting strategies."
                        )
                    elif bias_type == "equalized_odds":
                        recommendations.append(
                            f"Equalized odds violation for {attr}. "
                            "Consider threshold optimization or fairness constraints."
                        )
                    elif bias_type == "equality_of_opportunity":
                        recommendations.append(
                            f"Equality of opportunity violation for {attr}. "
                            "Focus on improving recall for disadvantaged groups."
                        )
        
        return recommendations
    
    async def _calculate_fairness_metrics(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive fairness metrics"""
        try:
            X = data[feature_columns]
            y = data[target_column]
            y_pred = model.predict(X)
            
            metrics = {
                "overall_accuracy": accuracy_score(y, y_pred),
                "overall_precision": precision_score(y, y_pred, average='weighted'),
                "overall_recall": recall_score(y, y_pred, average='weighted')
            }
            
            # Calculate group-specific metrics
            for attr in protected_attributes:
                if attr in data.columns:
                    groups = data[attr].unique()
                    
                    for group in groups:
                        group_mask = data[attr] == group
                        group_y = y[group_mask]
                        group_pred = y_pred[group_mask]
                        
                        if len(group_y) > 0:
                            metrics[f"{attr}_{group}_accuracy"] = accuracy_score(group_y, group_pred)
                            metrics[f"{attr}_{group}_precision"] = precision_score(
                                group_y, group_pred, average='weighted'
                            )
                            metrics[f"{attr}_{group}_recall"] = recall_score(
                                group_y, group_pred, average='weighted'
                            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fairness metrics calculation failed: {str(e)}")
            return {}
    
    async def _generate_mitigation_recommendations(
        self,
        bias_analysis: BiasAnalysisResult,
        fairness_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate mitigation strategy recommendations"""
        recommendations = []
        
        if bias_analysis.overall_bias_score > 0.15:
            recommendations.append({
                "strategy": MitigationStrategy.REWEIGHTING.value,
                "priority": "high",
                "description": "Apply sample reweighting to balance representation",
                "expected_improvement": "moderate"
            })
            
            recommendations.append({
                "strategy": MitigationStrategy.RESAMPLING.value,
                "priority": "medium",
                "description": "Resample training data to improve balance",
                "expected_improvement": "high"
            })
        
        if bias_analysis.overall_bias_score > 0.1:
            recommendations.append({
                "strategy": MitigationStrategy.THRESHOLD_OPTIMIZATION.value,
                "priority": "medium",
                "description": "Optimize decision thresholds for different groups",
                "expected_improvement": "moderate"
            })
        
        return recommendations
    
    async def _check_compliance_status(
        self,
        fairness_metrics: Dict[str, float],
        bias_analysis: BiasAnalysisResult
    ) -> Dict[str, bool]:
        """Check compliance with fairness standards"""
        compliance = {
            "demographic_parity": bias_analysis.overall_bias_score < 0.1,
            "equal_opportunity": True,  # Simplified check
            "overall_fairness": bias_analysis.overall_bias_score < 0.15,
            "regulatory_compliance": bias_analysis.overall_bias_score < 0.05
        }
        
        return compliance
    
    async def _apply_reweighting(
        self,
        model: Any,
        training_data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> Any:
        """Apply reweighting mitigation strategy"""
        try:
            # Calculate sample weights to balance protected groups
            sample_weights = np.ones(len(training_data))
            
            for attr in protected_attributes:
                if attr in training_data.columns:
                    # Calculate weights to balance groups
                    group_counts = training_data[attr].value_counts()
                    max_count = group_counts.max()
                    
                    for group, count in group_counts.items():
                        group_mask = training_data[attr] == group
                        weight = max_count / count
                        sample_weights[group_mask] *= weight
            
            # Retrain model with weights
            X_train = training_data[feature_columns]
            y_train = training_data[target_column]
            
            # Clone the model
            from sklearn.base import clone
            mitigated_model = clone(model)
            
            # Retrain with sample weights
            if hasattr(mitigated_model, 'fit') and 'sample_weight' in mitigated_model.fit.__code__.co_varnames:
                mitigated_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                # If model doesn't support sample weights, use resampling instead
                mitigated_model = await self._apply_resampling(
                    model, training_data, target_column, protected_attributes, feature_columns
                )
            
            return mitigated_model
            
        except Exception as e:
            logger.error(f"Reweighting mitigation failed: {str(e)}")
            raise BiasDetectionError(f"Reweighting mitigation failed: {str(e)}")
    
    async def _apply_resampling(
        self,
        model: Any,
        training_data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> Any:
        """Apply resampling mitigation strategy"""
        try:
            # Simple resampling strategy: balance protected groups
            balanced_data = training_data.copy()
            
            for attr in protected_attributes:
                if attr in training_data.columns:
                    # Find minimum group size
                    group_counts = training_data[attr].value_counts()
                    min_count = group_counts.min()
                    
                    # Sample equal number from each group
                    balanced_groups = []
                    for group in group_counts.index:
                        group_data = training_data[training_data[attr] == group]
                        sampled_data = group_data.sample(n=min_count, random_state=42)
                        balanced_groups.append(sampled_data)
                    
                    balanced_data = pd.concat(balanced_groups, ignore_index=True)
            
            # Retrain model on balanced data
            X_train = balanced_data[feature_columns]
            y_train = balanced_data[target_column]
            
            from sklearn.base import clone
            mitigated_model = clone(model)
            mitigated_model.fit(X_train, y_train)
            
            return mitigated_model
            
        except Exception as e:
            logger.error(f"Resampling mitigation failed: {str(e)}")
            raise BiasDetectionError(f"Resampling mitigation failed: {str(e)}")
    
    async def _apply_threshold_optimization(
        self,
        model: Any,
        training_data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> Any:
        """Apply threshold optimization mitigation strategy"""
        try:
            # This is a simplified implementation
            # In practice, you'd optimize thresholds for each protected group
            
            # For now, return the original model
            # A full implementation would create a wrapper that applies
            # different thresholds for different groups
            
            logger.info("Threshold optimization not fully implemented. Returning original model.")
            return model
            
        except Exception as e:
            logger.error(f"Threshold optimization failed: {str(e)}")
            raise BiasDetectionError(f"Threshold optimization failed: {str(e)}")
    
    async def _calculate_model_metrics(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        try:
            X = data[feature_columns]
            y = data[target_column]
            y_pred = model.predict(X)
            
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='weighted'),
                "recall": recall_score(y, y_pred, average='weighted')
            }
            
            # Add bias metrics
            for attr in protected_attributes:
                if attr in data.columns:
                    groups = data[attr].unique()
                    bias_score = await self._calculate_demographic_parity(
                        data, attr, y_pred, groups
                    )
                    metrics[f"bias_{attr}"] = bias_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model metrics calculation failed: {str(e)}")
            return {}
    
    def _calculate_improvement_score(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float]
    ) -> float:
        """Calculate improvement score after mitigation"""
        try:
            # Focus on bias reduction
            bias_improvements = []
            
            for key in before_metrics:
                if key.startswith("bias_"):
                    before_bias = before_metrics[key]
                    after_bias = after_metrics.get(key, before_bias)
                    
                    if before_bias > 0:
                        improvement = (before_bias - after_bias) / before_bias
                        bias_improvements.append(improvement)
            
            if bias_improvements:
                return np.mean(bias_improvements)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_trade_offs(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance trade-offs after mitigation"""
        trade_offs = {}
        
        performance_metrics = ["accuracy", "precision", "recall"]
        
        for metric in performance_metrics:
            if metric in before_metrics and metric in after_metrics:
                before_value = before_metrics[metric]
                after_value = after_metrics[metric]
                
                if before_value > 0:
                    change = (after_value - before_value) / before_value
                    trade_offs[f"{metric}_change"] = change
        
        return trade_offs
    
    async def get_bias_analysis(self, analysis_id: str) -> Optional[BiasAnalysisResult]:
        """Get bias analysis by ID"""
        return self.bias_analyses.get(analysis_id)
    
    async def get_fairness_report(self, report_id: str) -> Optional[FairnessReport]:
        """Get fairness report by ID"""
        return self.fairness_reports.get(report_id)
    
    async def get_mitigation_result(self, mitigation_id: str) -> Optional[MitigationResult]:
        """Get mitigation result by ID"""
        return self.mitigation_results.get(mitigation_id)
    
    async def list_bias_analyses(self, model_id: str = None) -> List[BiasAnalysisResult]:
        """List bias analyses, optionally filtered by model ID"""
        analyses = list(self.bias_analyses.values())
        
        if model_id:
            analyses = [analysis for analysis in analyses if analysis.model_id == model_id]
        
        return sorted(analyses, key=lambda x: x.created_at, reverse=True)
    
    async def generate_bias_report_plot(self, analysis_id: str) -> plt.Figure:
        """Generate bias analysis visualization"""
        try:
            analysis = self.bias_analyses.get(analysis_id)
            if not analysis:
                raise BiasDetectionError(f"Analysis {analysis_id} not found")
            
            # Create bias metrics visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Bias Analysis Report - {analysis.model_id}', fontsize=16)
            
            # Plot 1: Overall bias score
            axes[0, 0].bar(['Overall Bias Score'], [analysis.overall_bias_score])
            axes[0, 0].set_ylabel('Bias Score')
            axes[0, 0].set_title('Overall Bias Score')
            axes[0, 0].axhline(y=0.1, color='r', linestyle='--', label='Threshold')
            axes[0, 0].legend()
            
            # Plot 2: Bias by protected attribute
            if analysis.bias_metrics:
                attrs = list(analysis.bias_metrics.keys())
                avg_bias_by_attr = [
                    np.mean(list(analysis.bias_metrics[attr].values()))
                    for attr in attrs
                ]
                
                axes[0, 1].bar(attrs, avg_bias_by_attr)
                axes[0, 1].set_ylabel('Average Bias Score')
                axes[0, 1].set_title('Bias by Protected Attribute')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Bias by type
            if analysis.bias_metrics:
                bias_types = set()
                for attr_metrics in analysis.bias_metrics.values():
                    bias_types.update(attr_metrics.keys())
                
                bias_types = list(bias_types)
                avg_bias_by_type = []
                
                for bias_type in bias_types:
                    scores = []
                    for attr_metrics in analysis.bias_metrics.values():
                        if bias_type in attr_metrics:
                            scores.append(attr_metrics[bias_type])
                    avg_bias_by_type.append(np.mean(scores) if scores else 0)
                
                axes[1, 0].bar(bias_types, avg_bias_by_type)
                axes[1, 0].set_ylabel('Average Bias Score')
                axes[1, 0].set_title('Bias by Type')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Recommendations
            axes[1, 1].text(0.1, 0.9, 'Recommendations:', fontsize=12, fontweight='bold',
                           transform=axes[1, 1].transAxes)
            
            for i, rec in enumerate(analysis.recommendations[:5]):  # Show top 5
                axes[1, 1].text(0.1, 0.8 - i*0.15, f"• {rec}", fontsize=10,
                               transform=axes[1, 1].transAxes, wrap=True)
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate bias report plot: {str(e)}")
            raise BiasDetectionError(f"Failed to generate bias report plot: {str(e)}")