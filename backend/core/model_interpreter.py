"""
Model Interpreter for Explainable AI

This module provides comprehensive model interpretability and explainable AI
features including SHAP values, LIME explanations, feature importance analysis,
and decision boundary visualization.
"""

import uuid
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from .exceptions import ModelInterpreterError, ValidationError

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of model explanations"""
    GLOBAL = "global"
    LOCAL = "local"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    SHAP_VALUES = "shap_values"
    LIME_EXPLANATION = "lime_explanation"

class InterpretabilityMethod(Enum):
    """Interpretability methods"""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    DECISION_TREE_RULES = "decision_tree_rules"

@dataclass
class ExplanationResult:
    """Model explanation result"""
    explanation_id: str
    model_id: str
    explanation_type: ExplanationType
    method: InterpretabilityMethod
    feature_names: List[str]
    explanation_data: Dict[str, Any]
    confidence_score: float
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FeatureImportance:
    """Feature importance result"""
    feature_name: str
    importance_score: float
    rank: int
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class LocalExplanation:
    """Local explanation for a single prediction"""
    instance_id: str
    prediction: float
    prediction_probability: Optional[float]
    feature_contributions: Dict[str, float]
    base_value: float
    explanation_method: InterpretabilityMethod

class ModelInterpreter:
    """
    Comprehensive model interpretability system providing various
    explanation methods for ML model decisions and predictions.
    """
    
    def __init__(self):
        """Initialize Model Interpreter"""
        self.explanations: Dict[str, ExplanationResult] = {}
        self.shap_explainers: Dict[str, Any] = {}
        self.lime_explainers: Dict[str, Any] = {}
        
    async def explain_model(
        self,
        model: Any,
        model_id: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        method: InterpretabilityMethod = InterpretabilityMethod.SHAP,
        explanation_type: ExplanationType = ExplanationType.GLOBAL
    ) -> str:
        """
        Generate model explanation using specified method
        
        Args:
            model: Trained model
            model_id: Model identifier
            training_data: Training dataset
            feature_names: List of feature names
            method: Interpretability method to use
            explanation_type: Type of explanation (global/local)
            
        Returns:
            Explanation ID
        """
        try:
            explanation_id = str(uuid.uuid4())
            
            # Validate inputs
            if training_data.empty:
                raise ValidationError("Training data cannot be empty")
            
            if not feature_names:
                raise ValidationError("Feature names must be provided")
            
            # Generate explanation based on method
            if method == InterpretabilityMethod.SHAP:
                explanation_data = await self._generate_shap_explanation(
                    model, model_id, training_data, feature_names, explanation_type
                )
            elif method == InterpretabilityMethod.LIME:
                explanation_data = await self._generate_lime_explanation(
                    model, model_id, training_data, feature_names, explanation_type
                )
            elif method == InterpretabilityMethod.PERMUTATION_IMPORTANCE:
                explanation_data = await self._generate_permutation_importance(
                    model, training_data, feature_names
                )
            elif method == InterpretabilityMethod.FEATURE_IMPORTANCE:
                explanation_data = await self._generate_feature_importance(
                    model, feature_names
                )
            elif method == InterpretabilityMethod.DECISION_TREE_RULES:
                explanation_data = await self._generate_decision_tree_rules(
                    model, feature_names
                )
            else:
                raise ValidationError(f"Unsupported interpretability method: {method}")
            
            # Calculate confidence score
            confidence_score = self._calculate_explanation_confidence(
                explanation_data, method
            )
            
            # Create explanation result
            explanation = ExplanationResult(
                explanation_id=explanation_id,
                model_id=model_id,
                explanation_type=explanation_type,
                method=method,
                feature_names=feature_names,
                explanation_data=explanation_data,
                confidence_score=confidence_score,
                created_at=datetime.utcnow(),
                metadata={
                    "training_data_shape": training_data.shape,
                    "model_type": type(model).__name__
                }
            )
            
            # Store explanation
            self.explanations[explanation_id] = explanation
            
            logger.info(f"Model explanation {explanation_id} generated using {method.value}")
            
            return explanation_id
            
        except Exception as e:
            logger.error(f"Failed to generate model explanation: {str(e)}")
            raise ModelInterpreterError(f"Failed to generate model explanation: {str(e)}")
    
    async def explain_prediction(
        self,
        model: Any,
        model_id: str,
        instance: pd.Series,
        training_data: pd.DataFrame,
        feature_names: List[str],
        method: InterpretabilityMethod = InterpretabilityMethod.SHAP
    ) -> LocalExplanation:
        """
        Generate explanation for a single prediction
        
        Args:
            model: Trained model
            model_id: Model identifier
            instance: Single instance to explain
            training_data: Training dataset for context
            feature_names: List of feature names
            method: Interpretability method to use
            
        Returns:
            Local explanation
        """
        try:
            instance_id = str(uuid.uuid4())
            
            # Make prediction
            prediction = model.predict([instance.values])[0]
            prediction_probability = None
            
            if hasattr(model, 'predict_proba'):
                prediction_probability = model.predict_proba([instance.values])[0].max()
            
            # Generate explanation based on method
            if method == InterpretabilityMethod.SHAP:
                feature_contributions, base_value = await self._explain_instance_shap(
                    model, model_id, instance, training_data, feature_names
                )
            elif method == InterpretabilityMethod.LIME:
                feature_contributions, base_value = await self._explain_instance_lime(
                    model, model_id, instance, training_data, feature_names
                )
            else:
                raise ValidationError(f"Method {method} not supported for local explanations")
            
            # Create local explanation
            local_explanation = LocalExplanation(
                instance_id=instance_id,
                prediction=prediction,
                prediction_probability=prediction_probability,
                feature_contributions=feature_contributions,
                base_value=base_value,
                explanation_method=method
            )
            
            return local_explanation
            
        except Exception as e:
            logger.error(f"Failed to explain prediction: {str(e)}")
            raise ModelInterpreterError(f"Failed to explain prediction: {str(e)}")
    
    async def _generate_shap_explanation(
        self,
        model: Any,
        model_id: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        explanation_type: ExplanationType
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        if not SHAP_AVAILABLE:
            raise ModelInterpreterError("SHAP not available. Install with: pip install shap")
        
        try:
            # Initialize SHAP explainer
            if model_id not in self.shap_explainers:
                X_train = training_data[feature_names]
                
                # Choose appropriate explainer based on model type
                if hasattr(model, 'tree_'):
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                elif isinstance(model, (LogisticRegression,)):
                    # Linear models
                    explainer = shap.LinearExplainer(model, X_train)
                else:
                    # General explainer
                    explainer = shap.Explainer(model, X_train)
                
                self.shap_explainers[model_id] = explainer
            else:
                explainer = self.shap_explainers[model_id]
            
            # Calculate SHAP values
            X_explain = training_data[feature_names].sample(min(1000, len(training_data)))
            shap_values = explainer.shap_values(X_explain)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            explanation_data = {
                "shap_values": shap_values.tolist(),
                "feature_importance": {
                    feature_names[i]: float(importance)
                    for i, importance in enumerate(feature_importance)
                },
                "base_value": float(explainer.expected_value),
                "explanation_type": explanation_type.value,
                "sample_size": len(X_explain)
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            raise ModelInterpreterError(f"SHAP explanation failed: {str(e)}")
    
    async def _generate_lime_explanation(
        self,
        model: Any,
        model_id: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        explanation_type: ExplanationType
    ) -> Dict[str, Any]:
        """Generate LIME-based explanation"""
        if not LIME_AVAILABLE:
            raise ModelInterpreterError("LIME not available. Install with: pip install lime")
        
        try:
            # Initialize LIME explainer
            if model_id not in self.lime_explainers:
                X_train = training_data[feature_names].values
                
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['0', '1'],  # Assuming binary classification
                    mode='classification'
                )
                
                self.lime_explainers[model_id] = explainer
            else:
                explainer = self.lime_explainers[model_id]
            
            # Generate explanations for sample instances
            X_explain = training_data[feature_names].sample(min(100, len(training_data)))
            
            feature_importance = {feature: 0.0 for feature in feature_names}
            explanation_count = 0
            
            for _, instance in X_explain.iterrows():
                try:
                    explanation = explainer.explain_instance(
                        instance.values,
                        model.predict_proba,
                        num_features=len(feature_names)
                    )
                    
                    # Aggregate feature importance
                    for feature_idx, importance in explanation.as_list():
                        if feature_idx < len(feature_names):
                            feature_importance[feature_names[feature_idx]] += abs(importance)
                    
                    explanation_count += 1
                    
                except Exception as e:
                    logger.warning(f"LIME explanation failed for instance: {str(e)}")
                    continue
            
            # Average feature importance
            if explanation_count > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= explanation_count
            
            explanation_data = {
                "feature_importance": feature_importance,
                "explanation_type": explanation_type.value,
                "sample_size": explanation_count,
                "method": "lime"
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            raise ModelInterpreterError(f"LIME explanation failed: {str(e)}")
    
    async def _generate_permutation_importance(
        self,
        model: Any,
        training_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate permutation importance explanation"""
        try:
            X = training_data[feature_names]
            y = training_data.iloc[:, -1]  # Assume last column is target
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=42
            )
            
            feature_importance = {
                feature_names[i]: {
                    "importance": float(perm_importance.importances_mean[i]),
                    "std": float(perm_importance.importances_std[i])
                }
                for i in range(len(feature_names))
            }
            
            explanation_data = {
                "feature_importance": feature_importance,
                "method": "permutation_importance",
                "n_repeats": 10
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Permutation importance failed: {str(e)}")
            raise ModelInterpreterError(f"Permutation importance failed: {str(e)}")
    
    async def _generate_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate built-in feature importance explanation"""
        try:
            if not hasattr(model, 'feature_importances_'):
                raise ModelInterpreterError("Model does not have built-in feature importance")
            
            feature_importance = {
                feature_names[i]: float(model.feature_importances_[i])
                for i in range(len(feature_names))
            }
            
            explanation_data = {
                "feature_importance": feature_importance,
                "method": "built_in_feature_importance"
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {str(e)}")
            raise ModelInterpreterError(f"Feature importance extraction failed: {str(e)}")
    
    async def _generate_decision_tree_rules(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate decision tree rules explanation"""
        try:
            if not hasattr(model, 'tree_'):
                raise ModelInterpreterError("Model is not a decision tree")
            
            # Extract tree rules as text
            tree_rules = export_text(model, feature_names=feature_names)
            
            explanation_data = {
                "tree_rules": tree_rules,
                "method": "decision_tree_rules",
                "n_nodes": model.tree_.node_count,
                "max_depth": model.tree_.max_depth
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Decision tree rules extraction failed: {str(e)}")
            raise ModelInterpreterError(f"Decision tree rules extraction failed: {str(e)}")
    
    async def _explain_instance_shap(
        self,
        model: Any,
        model_id: str,
        instance: pd.Series,
        training_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """Generate SHAP explanation for single instance"""
        if not SHAP_AVAILABLE:
            raise ModelInterpreterError("SHAP not available")
        
        try:
            # Get or create explainer
            if model_id not in self.shap_explainers:
                await self._generate_shap_explanation(
                    model, model_id, training_data, feature_names, ExplanationType.GLOBAL
                )
            
            explainer = self.shap_explainers[model_id]
            
            # Calculate SHAP values for instance
            shap_values = explainer.shap_values(instance.values.reshape(1, -1))
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1][0]  # Use positive class
            else:
                shap_values = shap_values[0]
            
            # Create feature contributions dictionary
            feature_contributions = {
                feature_names[i]: float(shap_values[i])
                for i in range(len(feature_names))
            }
            
            base_value = float(explainer.expected_value)
            
            return feature_contributions, base_value
            
        except Exception as e:
            logger.error(f"SHAP instance explanation failed: {str(e)}")
            raise ModelInterpreterError(f"SHAP instance explanation failed: {str(e)}")
    
    async def _explain_instance_lime(
        self,
        model: Any,
        model_id: str,
        instance: pd.Series,
        training_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """Generate LIME explanation for single instance"""
        if not LIME_AVAILABLE:
            raise ModelInterpreterError("LIME not available")
        
        try:
            # Get or create explainer
            if model_id not in self.lime_explainers:
                await self._generate_lime_explanation(
                    model, model_id, training_data, feature_names, ExplanationType.GLOBAL
                )
            
            explainer = self.lime_explainers[model_id]
            
            # Generate explanation
            explanation = explainer.explain_instance(
                instance.values,
                model.predict_proba,
                num_features=len(feature_names)
            )
            
            # Extract feature contributions
            feature_contributions = {}
            for feature_idx, importance in explanation.as_list():
                if feature_idx < len(feature_names):
                    feature_contributions[feature_names[feature_idx]] = importance
            
            # Fill missing features with 0
            for feature in feature_names:
                if feature not in feature_contributions:
                    feature_contributions[feature] = 0.0
            
            # LIME doesn't provide a base value, so we use the model's average prediction
            base_value = 0.0  # Could be improved by calculating actual base value
            
            return feature_contributions, base_value
            
        except Exception as e:
            logger.error(f"LIME instance explanation failed: {str(e)}")
            raise ModelInterpreterError(f"LIME instance explanation failed: {str(e)}")
    
    def _calculate_explanation_confidence(
        self,
        explanation_data: Dict[str, Any],
        method: InterpretabilityMethod
    ) -> float:
        """Calculate confidence score for explanation"""
        try:
            if method == InterpretabilityMethod.SHAP:
                # For SHAP, confidence is based on the consistency of values
                if "shap_values" in explanation_data:
                    shap_values = np.array(explanation_data["shap_values"])
                    # Calculate coefficient of variation as inverse of confidence
                    cv = np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-8)
                    confidence = max(0.1, min(1.0, 1.0 - cv))
                    return confidence
            
            elif method == InterpretabilityMethod.PERMUTATION_IMPORTANCE:
                # For permutation importance, confidence is based on standard deviation
                if "feature_importance" in explanation_data:
                    importances = [
                        data["importance"] for data in explanation_data["feature_importance"].values()
                    ]
                    stds = [
                        data["std"] for data in explanation_data["feature_importance"].values()
                    ]
                    
                    if importances and stds:
                        avg_importance = np.mean(importances)
                        avg_std = np.mean(stds)
                        confidence = max(0.1, min(1.0, 1.0 - (avg_std / (avg_importance + 1e-8))))
                        return confidence
            
            # Default confidence for other methods
            return 0.8
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    async def get_explanation(self, explanation_id: str) -> Optional[ExplanationResult]:
        """Get explanation by ID"""
        return self.explanations.get(explanation_id)
    
    async def list_explanations(self, model_id: str = None) -> List[ExplanationResult]:
        """List all explanations, optionally filtered by model ID"""
        explanations = list(self.explanations.values())
        
        if model_id:
            explanations = [exp for exp in explanations if exp.model_id == model_id]
        
        return sorted(explanations, key=lambda x: x.created_at, reverse=True)
    
    async def generate_feature_importance_plot(
        self,
        explanation_id: str,
        top_n: int = 10
    ) -> plt.Figure:
        """Generate feature importance plot"""
        try:
            explanation = self.explanations.get(explanation_id)
            if not explanation:
                raise ModelInterpreterError(f"Explanation {explanation_id} not found")
            
            feature_importance = explanation.explanation_data.get("feature_importance", {})
            
            if not feature_importance:
                raise ModelInterpreterError("No feature importance data available")
            
            # Extract importance values
            if isinstance(list(feature_importance.values())[0], dict):
                # Permutation importance format
                features = list(feature_importance.keys())
                importances = [feature_importance[f]["importance"] for f in features]
            else:
                # Simple format
                features = list(feature_importance.keys())
                importances = list(feature_importance.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[-top_n:]
            sorted_features = [features[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(sorted_features)), sorted_importances)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Feature Importance - {explanation.method.value}')
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate feature importance plot: {str(e)}")
            raise ModelInterpreterError(f"Failed to generate feature importance plot: {str(e)}")
    
    async def generate_explanation_summary(self, explanation_id: str) -> Dict[str, Any]:
        """Generate human-readable explanation summary"""
        try:
            explanation = self.explanations.get(explanation_id)
            if not explanation:
                raise ModelInterpreterError(f"Explanation {explanation_id} not found")
            
            feature_importance = explanation.explanation_data.get("feature_importance", {})
            
            # Extract top features
            if isinstance(list(feature_importance.values())[0], dict):
                # Permutation importance format
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]["importance"],
                    reverse=True
                )
                top_features = [(f, data["importance"]) for f, data in sorted_features[:5]]
            else:
                # Simple format
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_features = sorted_features[:5]
            
            summary = {
                "explanation_id": explanation_id,
                "model_id": explanation.model_id,
                "method": explanation.method.value,
                "confidence_score": explanation.confidence_score,
                "top_features": [
                    {
                        "feature": feature,
                        "importance": importance,
                        "rank": i + 1
                    }
                    for i, (feature, importance) in enumerate(top_features)
                ],
                "total_features": len(feature_importance),
                "explanation_type": explanation.explanation_type.value,
                "created_at": explanation.created_at
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate explanation summary: {str(e)}")
            raise ModelInterpreterError(f"Failed to generate explanation summary: {str(e)}")
    
    async def delete_explanation(self, explanation_id: str):
        """Delete an explanation"""
        if explanation_id in self.explanations:
            del self.explanations[explanation_id]
            logger.info(f"Explanation {explanation_id} deleted")
        else:
            raise ModelInterpreterError(f"Explanation {explanation_id} not found")