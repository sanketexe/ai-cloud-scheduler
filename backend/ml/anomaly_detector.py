"""
Real-time Anomaly Detection Engine

Orchestrates multiple ML models (Isolation Forest, LSTM, Prophet) to provide
comprehensive real-time anomaly detection for cost data. Implements business
rule validation, confidence scoring, and intelligent alert generation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from enum import Enum
import json

from .isolation_forest_detector import IsolationForestDetector, AnomalyScore
from .lstm_anomaly_detector import LSTMAnomalyDetector, LSTMPrediction
from .prophet_forecaster import ProphetForecaster, AnomalyDetection
from .ensemble_scorer import EnsembleScorer, EnsembleResult, EnsembleConfig
from .feature_engine import FeatureEngine, FeatureSet
from .data_pipeline import DataPipeline

logger = structlog.get_logger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected"""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BusinessRule:
    """Business rule for anomaly validation"""
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    threshold: float
    enabled: bool = True
    description: str = ""


@dataclass
class AnomalyAlert:
    """Comprehensive anomaly alert"""
    alert_id: str
    timestamp: datetime
    account_id: str
    service: str
    region: Optional[str]
    
    # Anomaly details
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float
    cost_impact: float
    percentage_deviation: float
    
    # Detection details
    detected_by_models: List[str]
    ensemble_score: float
    individual_scores: Dict[str, float]
    
    # Context and explanation
    explanation: str
    root_cause_analysis: Dict[str, Any]
    historical_context: Dict[str, Any]
    business_impact: Dict[str, Any]
    
    # Recommendations
    recommended_actions: List[str]
    urgency_level: str
    
    # Metadata
    detection_latency_ms: float
    data_quality_score: float
    false_positive_probability: float


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection"""
    # Model weights and thresholds
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Business rules
    business_rules: List[BusinessRule] = field(default_factory=list)
    
    # Alert thresholds
    min_cost_impact: float = 10.0  # Minimum cost impact to generate alert
    min_confidence: float = 0.7    # Minimum confidence for alert
    
    # Time windows
    real_time_window_hours: int = 24
    context_window_hours: int = 168  # 1 week
    
    # Performance settings
    max_processing_time_ms: float = 5000  # 5 seconds max
    enable_parallel_processing: bool = True
    
    # Alert filtering
    enable_alert_suppression: bool = True
    suppression_window_minutes: int = 60
    max_alerts_per_hour: int = 10


class AnomalyDetector:
    """
    Real-time anomaly detection engine that orchestrates multiple ML models.
    
    Combines Isolation Forest, LSTM, and Prophet models using ensemble methods
    to detect various types of cost anomalies. Includes business rule validation,
    confidence scoring, and intelligent alert generation.
    """
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize the anomaly detection engine.
        
        Args:
            config: Detection configuration
        """
        self.config = config or DetectionConfig()
        
        # Initialize ML models with fallback handling
        self.isolation_forest = IsolationForestDetector()
        
        # Initialize LSTM detector with fallback
        try:
            self.lstm_detector = LSTMAnomalyDetector()
        except ImportError as e:
            logger.warning("LSTM detector not available", error=str(e))
            self.lstm_detector = None
        
        # Initialize Prophet forecaster with fallback
        try:
            self.prophet_forecaster = ProphetForecaster()
        except ImportError as e:
            logger.warning("Prophet forecaster not available", error=str(e))
            self.prophet_forecaster = None
        
        # Initialize ensemble scorer
        self.ensemble_scorer = EnsembleScorer(self.config.ensemble_config)
        self.ensemble_scorer.set_models(
            isolation_forest=self.isolation_forest,
            lstm_detector=self.lstm_detector,
            prophet_forecaster=self.prophet_forecaster
        )
        
        # Initialize feature engine and data pipeline
        self.feature_engine = FeatureEngine()
        self.data_pipeline = DataPipeline()
        
        # Detection state
        self.is_initialized = False
        self.recent_alerts = []
        self.detection_metrics = {
            'total_detections': 0,
            'alerts_generated': 0,
            'false_positives': 0,
            'processing_times': [],
            'model_performance': {}
        }
        
        # Business rules engine
        self.business_rules = self.config.business_rules
        
        # Alert suppression tracking
        self.alert_suppression_cache = {}
    
    async def initialize(self):
        """Initialize the anomaly detection engine"""
        logger.info("Initializing anomaly detection engine")
        
        try:
            # Initialize data pipeline
            await self.data_pipeline.initialize()
            
            # Load or train models if needed
            await self._initialize_models()
            
            self.is_initialized = True
            logger.info("Anomaly detection engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize anomaly detection engine", error=str(e))
            raise
    
    async def detect_anomalies_real_time(self, 
                                       account_id: str,
                                       lookback_hours: int = None) -> List[AnomalyAlert]:
        """
        Perform real-time anomaly detection for an account.
        
        Args:
            account_id: AWS account ID to analyze
            lookback_hours: Hours of data to analyze (default from config)
            
        Returns:
            List of anomaly alerts
        """
        if not self.is_initialized:
            raise ValueError("Anomaly detector must be initialized before use")
        
        start_time = datetime.utcnow()
        lookback_hours = lookback_hours or self.config.real_time_window_hours
        
        logger.info(
            "Starting real-time anomaly detection",
            account_id=account_id,
            lookback_hours=lookback_hours
        )
        
        try:
            # Step 1: Get real-time data
            raw_data, feature_set = await self.data_pipeline.get_real_time_inference_data(
                lookback_hours=lookback_hours
            )
            
            # Filter for specific account
            account_data = raw_data[raw_data['account_id'] == account_id] if 'account_id' in raw_data.columns else raw_data
            
            if account_data.empty:
                logger.warning("No data found for account", account_id=account_id)
                return []
            
            # Step 2: Run ensemble prediction
            ensemble_results = self.ensemble_scorer.predict_ensemble(account_data)
            
            # Step 3: Apply business rules and filtering
            filtered_results = await self._apply_business_rules(ensemble_results, account_data)
            
            # Step 4: Generate alerts
            alerts = await self._generate_alerts(filtered_results, account_data, account_id)
            
            # Step 5: Apply alert suppression
            final_alerts = self._apply_alert_suppression(alerts)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_detection_metrics(len(ensemble_results), len(final_alerts), processing_time)
            
            logger.info(
                "Real-time anomaly detection completed",
                account_id=account_id,
                detections=len(ensemble_results),
                alerts=len(final_alerts),
                processing_time_ms=processing_time
            )
            
            return final_alerts
            
        except Exception as e:
            logger.error("Real-time anomaly detection failed", account_id=account_id, error=str(e))
            raise
    
    async def detect_anomalies_batch(self,
                                   start_date: datetime,
                                   end_date: datetime,
                                   account_ids: Optional[List[str]] = None) -> Dict[str, List[AnomalyAlert]]:
        """
        Perform batch anomaly detection for multiple accounts.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            account_ids: Optional list of account IDs to analyze
            
        Returns:
            Dictionary mapping account IDs to their anomaly alerts
        """
        logger.info(
            "Starting batch anomaly detection",
            date_range=f"{start_date} to {end_date}",
            accounts=len(account_ids) if account_ids else "all"
        )
        
        try:
            # Get batch training data
            raw_data, feature_set = await self.data_pipeline.get_ml_training_data(
                start_date.date(), end_date.date(), include_features=True
            )
            
            if raw_data.empty:
                logger.warning("No data found for batch analysis")
                return {}
            
            # Filter accounts if specified
            if account_ids:
                raw_data = raw_data[raw_data['account_id'].isin(account_ids)]
            
            # Group by account and process
            results = {}
            
            for account_id in raw_data['account_id'].unique():
                account_data = raw_data[raw_data['account_id'] == account_id]
                
                # Run ensemble prediction
                ensemble_results = self.ensemble_scorer.predict_ensemble(account_data)
                
                # Apply business rules
                filtered_results = await self._apply_business_rules(ensemble_results, account_data)
                
                # Generate alerts
                alerts = await self._generate_alerts(filtered_results, account_data, account_id)
                
                results[account_id] = alerts
            
            logger.info("Batch anomaly detection completed", accounts=len(results))
            return results
            
        except Exception as e:
            logger.error("Batch anomaly detection failed", error=str(e))
            raise
    
    async def _apply_business_rules(self, 
                                  ensemble_results: List[EnsembleResult],
                                  raw_data: pd.DataFrame) -> List[EnsembleResult]:
        """Apply business rules to filter anomaly results"""
        logger.debug("Applying business rules", rules=len(self.business_rules))
        
        if not self.business_rules:
            return ensemble_results
        
        filtered_results = []
        
        for result in ensemble_results:
            # Get corresponding raw data row
            data_row = None
            if hasattr(result, 'timestamp') and 'timestamp' in raw_data.columns:
                matching_rows = raw_data[raw_data['timestamp'] == result.timestamp]
                if not matching_rows.empty:
                    data_row = matching_rows.iloc[0]
            
            # Apply each business rule
            passes_rules = True
            
            for rule in self.business_rules:
                if not rule.enabled:
                    continue
                
                try:
                    # Create evaluation context
                    context = {
                        'result': result,
                        'data': data_row.to_dict() if data_row is not None else {},
                        'ensemble_score': result.ensemble_score,
                        'confidence': result.confidence,
                        'cost_amount': data_row.get('cost_amount', 0) if data_row is not None else 0,
                        'service': data_row.get('service', '') if data_row is not None else '',
                        'threshold': rule.threshold
                    }
                    
                    # Evaluate rule condition
                    rule_result = eval(rule.condition, {"__builtins__": {}}, context)
                    
                    if not rule_result:
                        passes_rules = False
                        logger.debug(
                            "Result filtered by business rule",
                            rule_id=rule.rule_id,
                            rule_name=rule.name
                        )
                        break
                        
                except Exception as e:
                    logger.warning(
                        "Business rule evaluation failed",
                        rule_id=rule.rule_id,
                        error=str(e)
                    )
                    # Continue with other rules on evaluation error
            
            if passes_rules:
                filtered_results.append(result)
        
        logger.debug(
            "Business rules applied",
            original_count=len(ensemble_results),
            filtered_count=len(filtered_results)
        )
        
        return filtered_results
    
    async def _generate_alerts(self,
                             ensemble_results: List[EnsembleResult],
                             raw_data: pd.DataFrame,
                             account_id: str) -> List[AnomalyAlert]:
        """Generate comprehensive anomaly alerts"""
        logger.debug("Generating anomaly alerts", detections=len(ensemble_results))
        
        alerts = []
        
        for result in ensemble_results:
            if not result.is_anomaly:
                continue
            
            # Get corresponding data
            data_row = None
            if 'timestamp' in raw_data.columns:
                matching_rows = raw_data[raw_data['timestamp'] == result.timestamp]
                if not matching_rows.empty:
                    data_row = matching_rows.iloc[0]
            
            if data_row is None:
                continue
            
            # Calculate cost impact
            cost_impact = float(data_row.get('cost_amount', 0))
            
            # Skip if below minimum cost impact
            if cost_impact < self.config.min_cost_impact:
                continue
            
            # Skip if below minimum confidence
            if result.confidence < self.config.min_confidence:
                continue
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(result, data_row)
            
            # Map severity
            severity = self._map_severity(result.severity)
            
            # Calculate percentage deviation
            percentage_deviation = self._calculate_percentage_deviation(result, data_row)
            
            # Generate explanation
            explanation = await self._generate_explanation(result, data_row, anomaly_type)
            
            # Perform root cause analysis
            root_cause_analysis = await self._perform_root_cause_analysis(result, data_row)
            
            # Get historical context
            historical_context = await self._get_historical_context(result, data_row, account_id)
            
            # Assess business impact
            business_impact = self._assess_business_impact(result, data_row, cost_impact)
            
            # Generate recommendations
            recommended_actions = self._generate_recommendations(result, data_row, anomaly_type)
            
            # Calculate urgency
            urgency_level = self._calculate_urgency(result, cost_impact, severity)
            
            # Calculate false positive probability
            fp_probability = self._calculate_false_positive_probability(result)
            
            # Create alert
            alert = AnomalyAlert(
                alert_id=f"alert_{account_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(result.ensemble_score))%10000:04d}",
                timestamp=result.timestamp,
                account_id=account_id,
                service=str(data_row.get('service', 'unknown')),
                region=str(data_row.get('region', 'unknown')),
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=result.confidence,
                cost_impact=cost_impact,
                percentage_deviation=percentage_deviation,
                detected_by_models=result.contributing_models,
                ensemble_score=result.ensemble_score,
                individual_scores={
                    'isolation_forest': result.isolation_forest_score,
                    'lstm': result.lstm_score,
                    'prophet': result.prophet_score
                },
                explanation=explanation,
                root_cause_analysis=root_cause_analysis,
                historical_context=historical_context,
                business_impact=business_impact,
                recommended_actions=recommended_actions,
                urgency_level=urgency_level,
                detection_latency_ms=0.0,  # Will be calculated later
                data_quality_score=1.0 - (pd.isna(data_row).sum() / len(data_row)),
                false_positive_probability=fp_probability
            )
            
            alerts.append(alert)
        
        logger.debug("Anomaly alerts generated", count=len(alerts))
        return alerts
    
    def _apply_alert_suppression(self, alerts: List[AnomalyAlert]) -> List[AnomalyAlert]:
        """Apply alert suppression to reduce noise"""
        if not self.config.enable_alert_suppression:
            return alerts
        
        logger.debug("Applying alert suppression", alerts=len(alerts))
        
        current_time = datetime.utcnow()
        suppression_window = timedelta(minutes=self.config.suppression_window_minutes)
        
        filtered_alerts = []
        
        for alert in alerts:
            # Create suppression key
            suppression_key = f"{alert.account_id}_{alert.service}_{alert.anomaly_type.value}"
            
            # Check if similar alert was recently sent
            if suppression_key in self.alert_suppression_cache:
                last_alert_time = self.alert_suppression_cache[suppression_key]
                if current_time - last_alert_time < suppression_window:
                    logger.debug(
                        "Alert suppressed",
                        alert_id=alert.alert_id,
                        suppression_key=suppression_key
                    )
                    continue
            
            # Update suppression cache
            self.alert_suppression_cache[suppression_key] = current_time
            filtered_alerts.append(alert)
        
        # Clean old entries from suppression cache
        self._clean_suppression_cache(current_time, suppression_window)
        
        logger.debug(
            "Alert suppression applied",
            original_count=len(alerts),
            filtered_count=len(filtered_alerts)
        )
        
        return filtered_alerts
    
    def _clean_suppression_cache(self, current_time: datetime, suppression_window: timedelta):
        """Clean old entries from suppression cache"""
        keys_to_remove = []
        
        for key, last_time in self.alert_suppression_cache.items():
            if current_time - last_time > suppression_window * 2:  # Keep for 2x window
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.alert_suppression_cache[key]
    
    def _classify_anomaly_type(self, result: EnsembleResult, data_row: pd.Series) -> AnomalyType:
        """Classify the type of anomaly detected"""
        
        # Check which models contributed to detection
        contributing_models = set(result.contributing_models)
        
        # Point anomaly: Primarily detected by Isolation Forest
        if 'isolation_forest' in contributing_models and len(contributing_models) == 1:
            return AnomalyType.POINT_ANOMALY
        
        # Trend anomaly: Primarily detected by LSTM
        if 'lstm' in contributing_models and result.lstm_score and result.lstm_score > 0.8:
            return AnomalyType.TREND_ANOMALY
        
        # Seasonal anomaly: Primarily detected by Prophet
        if 'prophet' in contributing_models and result.prophet_score and result.prophet_score > 0.8:
            return AnomalyType.SEASONAL_ANOMALY
        
        # Contextual anomaly: Multiple models with moderate scores
        if len(contributing_models) >= 2 and result.ensemble_score > 0.7:
            return AnomalyType.CONTEXTUAL_ANOMALY
        
        # Collective anomaly: High ensemble score with multiple models
        if len(contributing_models) >= 2 and result.ensemble_score > 0.9:
            return AnomalyType.COLLECTIVE_ANOMALY
        
        # Default to point anomaly
        return AnomalyType.POINT_ANOMALY
    
    def _map_severity(self, severity_str: str) -> AlertSeverity:
        """Map string severity to AlertSeverity enum"""
        severity_mapping = {
            'low': AlertSeverity.LOW,
            'medium': AlertSeverity.MEDIUM,
            'high': AlertSeverity.HIGH,
            'critical': AlertSeverity.CRITICAL
        }
        return severity_mapping.get(severity_str.lower(), AlertSeverity.MEDIUM)
    
    def _calculate_percentage_deviation(self, result: EnsembleResult, data_row: pd.Series) -> float:
        """Calculate percentage deviation from expected value"""
        
        # Try to get baseline from Prophet prediction
        if result.prophet_score and 'cost_amount' in data_row:
            actual_cost = float(data_row['cost_amount'])
            # Estimate expected cost (simplified - would use actual Prophet forecast)
            expected_cost = actual_cost / (1 + result.prophet_score)
            if expected_cost > 0:
                return ((actual_cost - expected_cost) / expected_cost) * 100
        
        # Fallback: Use ensemble score as proxy for deviation
        return result.ensemble_score * 100
    
    async def _generate_explanation(self, 
                                  result: EnsembleResult, 
                                  data_row: pd.Series,
                                  anomaly_type: AnomalyType) -> str:
        """Generate human-readable explanation for the anomaly"""
        
        service = data_row.get('service', 'unknown')
        cost = data_row.get('cost_amount', 0)
        timestamp = result.timestamp
        
        # Base explanation
        explanation = f"Anomaly detected in {service} service at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}. "
        
        # Add cost impact
        explanation += f"Cost impact: ${cost:.2f}. "
        
        # Add model-specific insights
        if 'isolation_forest' in result.contributing_models:
            explanation += "Isolation Forest detected unusual cost pattern. "
        
        if 'lstm' in result.contributing_models:
            explanation += "LSTM model identified deviation from expected time series behavior. "
        
        if 'prophet' in result.contributing_models:
            explanation += "Prophet forecaster detected deviation from seasonal patterns. "
        
        # Add confidence information
        explanation += f"Detection confidence: {result.confidence:.1%}. "
        
        # Add anomaly type context
        type_explanations = {
            AnomalyType.POINT_ANOMALY: "This appears to be a sudden, isolated cost spike.",
            AnomalyType.TREND_ANOMALY: "This indicates a concerning trend change in spending patterns.",
            AnomalyType.SEASONAL_ANOMALY: "This deviates from expected seasonal spending behavior.",
            AnomalyType.CONTEXTUAL_ANOMALY: "This cost is unusual given the current context and timing.",
            AnomalyType.COLLECTIVE_ANOMALY: "This is part of a broader pattern of unusual spending behavior."
        }
        
        explanation += type_explanations.get(anomaly_type, "")
        
        return explanation
    
    async def _perform_root_cause_analysis(self, 
                                         result: EnsembleResult, 
                                         data_row: pd.Series) -> Dict[str, Any]:
        """Perform root cause analysis for the anomaly"""
        
        root_causes = {
            'primary_indicators': [],
            'contributing_factors': [],
            'model_insights': {},
            'data_patterns': {}
        }
        
        # Analyze model contributions
        if result.isolation_forest_score and result.isolation_forest_score > 0.7:
            root_causes['primary_indicators'].append('Unusual cost magnitude detected')
            root_causes['model_insights']['isolation_forest'] = {
                'score': result.isolation_forest_score,
                'interpretation': 'Cost value is significantly different from historical patterns'
            }
        
        if result.lstm_score and result.lstm_score > 0.7:
            root_causes['primary_indicators'].append('Unexpected time series behavior')
            root_causes['model_insights']['lstm'] = {
                'score': result.lstm_score,
                'interpretation': 'Cost sequence deviates from learned temporal patterns'
            }
        
        if result.prophet_score and result.prophet_score > 0.7:
            root_causes['primary_indicators'].append('Seasonal pattern deviation')
            root_causes['model_insights']['prophet'] = {
                'score': result.prophet_score,
                'interpretation': 'Cost deviates from expected seasonal and trend patterns'
            }
        
        # Analyze data patterns
        service = data_row.get('service', 'unknown')
        region = data_row.get('region', 'unknown')
        cost = data_row.get('cost_amount', 0)
        
        root_causes['data_patterns'] = {
            'service': service,
            'region': region,
            'cost_amount': float(cost),
            'timestamp': result.timestamp.isoformat()
        }
        
        # Add potential contributing factors
        hour = result.timestamp.hour
        day_of_week = result.timestamp.weekday()
        
        if hour < 6 or hour > 22:
            root_causes['contributing_factors'].append('Unusual timing (outside business hours)')
        
        if day_of_week >= 5:  # Weekend
            root_causes['contributing_factors'].append('Weekend activity (potentially unexpected)')
        
        return root_causes
    
    async def _get_historical_context(self, 
                                    result: EnsembleResult, 
                                    data_row: pd.Series,
                                    account_id: str) -> Dict[str, Any]:
        """Get historical context for the anomaly"""
        
        context = {
            'similar_events': [],
            'baseline_statistics': {},
            'trend_analysis': {}
        }
        
        try:
            # Get historical data for context (simplified - would query actual historical data)
            service = data_row.get('service', 'unknown')
            current_cost = float(data_row.get('cost_amount', 0))
            
            # Simulate baseline statistics (would calculate from actual historical data)
            context['baseline_statistics'] = {
                'service': service,
                'average_cost_7d': current_cost * 0.8,  # Simulated
                'average_cost_30d': current_cost * 0.75,  # Simulated
                'max_cost_30d': current_cost * 1.2,  # Simulated
                'cost_volatility': 0.15  # Simulated
            }
            
            # Simulate trend analysis
            context['trend_analysis'] = {
                'trend_direction': 'increasing' if current_cost > context['baseline_statistics']['average_cost_7d'] else 'stable',
                'trend_strength': abs(current_cost - context['baseline_statistics']['average_cost_7d']) / context['baseline_statistics']['average_cost_7d'],
                'trend_duration_days': 3  # Simulated
            }
            
        except Exception as e:
            logger.warning("Failed to get historical context", error=str(e))
        
        return context
    
    def _assess_business_impact(self, 
                              result: EnsembleResult, 
                              data_row: pd.Series,
                              cost_impact: float) -> Dict[str, Any]:
        """Assess business impact of the anomaly"""
        
        impact = {
            'financial_impact': {},
            'operational_impact': {},
            'risk_assessment': {}
        }
        
        # Financial impact
        impact['financial_impact'] = {
            'immediate_cost': cost_impact,
            'projected_monthly_impact': cost_impact * 30,  # Simplified projection
            'budget_impact_percentage': min(cost_impact / 10000 * 100, 100),  # Assume $10k monthly budget
            'cost_category': 'high' if cost_impact > 1000 else 'medium' if cost_impact > 100 else 'low'
        }
        
        # Operational impact
        service = data_row.get('service', 'unknown')
        impact['operational_impact'] = {
            'affected_service': service,
            'potential_service_disruption': result.severity in ['high', 'critical'],
            'requires_immediate_attention': result.confidence > 0.9 and cost_impact > 500
        }
        
        # Risk assessment
        impact['risk_assessment'] = {
            'escalation_risk': result.severity in ['high', 'critical'],
            'recurrence_probability': 'high' if result.ensemble_score > 0.9 else 'medium',
            'mitigation_urgency': 'immediate' if cost_impact > 1000 else 'within_24h' if cost_impact > 100 else 'routine'
        }
        
        return impact
    
    def _generate_recommendations(self, 
                                result: EnsembleResult, 
                                data_row: pd.Series,
                                anomaly_type: AnomalyType) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        service = data_row.get('service', 'unknown')
        cost = data_row.get('cost_amount', 0)
        
        # General recommendations based on anomaly type
        if anomaly_type == AnomalyType.POINT_ANOMALY:
            recommendations.extend([
                f"Investigate recent changes in {service} service configuration",
                "Check for unexpected resource provisioning or scaling events",
                "Review recent deployments or infrastructure changes"
            ])
        
        elif anomaly_type == AnomalyType.TREND_ANOMALY:
            recommendations.extend([
                f"Analyze {service} usage trends over the past week",
                "Consider implementing cost controls or budget alerts",
                "Review resource utilization and optimization opportunities"
            ])
        
        elif anomaly_type == AnomalyType.SEASONAL_ANOMALY:
            recommendations.extend([
                f"Compare current {service} usage with historical seasonal patterns",
                "Check for business events or campaigns affecting usage",
                "Consider adjusting seasonal budget allocations"
            ])
        
        # Cost-based recommendations
        if cost > 1000:
            recommendations.extend([
                "Immediate cost investigation required due to high impact",
                "Consider temporary resource scaling or throttling",
                "Notify finance team and service owners"
            ])
        elif cost > 100:
            recommendations.extend([
                "Monitor closely for continued cost increases",
                "Review resource tagging and cost allocation"
            ])
        
        # Confidence-based recommendations
        if result.confidence > 0.9:
            recommendations.append("High confidence detection - prioritize investigation")
        elif result.confidence < 0.7:
            recommendations.append("Lower confidence detection - verify with additional data sources")
        
        return recommendations
    
    def _calculate_urgency(self, 
                         result: EnsembleResult, 
                         cost_impact: float,
                         severity: AlertSeverity) -> str:
        """Calculate urgency level for the alert"""
        
        # High urgency conditions
        if (severity == AlertSeverity.CRITICAL or 
            cost_impact > 1000 or 
            result.confidence > 0.95):
            return "immediate"
        
        # Medium urgency conditions
        if (severity == AlertSeverity.HIGH or 
            cost_impact > 500 or 
            result.confidence > 0.85):
            return "within_1h"
        
        # Low urgency conditions
        if (severity == AlertSeverity.MEDIUM or 
            cost_impact > 100):
            return "within_4h"
        
        # Routine
        return "within_24h"
    
    def _calculate_false_positive_probability(self, result: EnsembleResult) -> float:
        """Calculate probability that this is a false positive"""
        
        # Base false positive rate
        base_fp_rate = 0.05  # 5% base rate
        
        # Adjust based on model agreement
        agreement_factor = result.model_agreement
        fp_rate = base_fp_rate * (1 - agreement_factor)
        
        # Adjust based on confidence
        confidence_factor = result.confidence
        fp_rate = fp_rate * (1 - confidence_factor)
        
        # Adjust based on ensemble score
        score_factor = result.ensemble_score
        fp_rate = fp_rate * (1 - score_factor)
        
        return max(0.01, min(0.5, fp_rate))  # Clamp between 1% and 50%
    
    async def _initialize_models(self):
        """Initialize or load ML models"""
        logger.info("Initializing ML models")
        
        try:
            # Check if models are already trained
            if self.isolation_forest and not self.isolation_forest.is_trained:
                logger.info("Isolation Forest not trained - would need training data")
            
            if self.lstm_detector and not self.lstm_detector.is_trained:
                logger.info("LSTM detector not trained - would need training data")
            
            if self.prophet_forecaster and not self.prophet_forecaster.is_trained:
                logger.info("Prophet forecaster not trained - would need training data")
            
            # In production, would load pre-trained models or train with historical data
            logger.info("ML models initialization completed")
            
        except Exception as e:
            logger.error("Failed to initialize ML models", error=str(e))
            raise
    
    def _update_detection_metrics(self, detections: int, alerts: int, processing_time: float):
        """Update detection performance metrics"""
        self.detection_metrics['total_detections'] += detections
        self.detection_metrics['alerts_generated'] += alerts
        self.detection_metrics['processing_times'].append(processing_time)
        
        # Keep only recent processing times
        if len(self.detection_metrics['processing_times']) > 100:
            self.detection_metrics['processing_times'] = self.detection_metrics['processing_times'][-100:]
    
    def get_detection_status(self) -> Dict[str, Any]:
        """Get current detection engine status"""
        avg_processing_time = (
            np.mean(self.detection_metrics['processing_times']) 
            if self.detection_metrics['processing_times'] else 0
        )
        
        return {
            'is_initialized': self.is_initialized,
            'models_status': {
                'isolation_forest': self.isolation_forest.is_trained if self.isolation_forest else False,
                'lstm_detector': self.lstm_detector.is_trained if self.lstm_detector else False,
                'prophet_forecaster': self.prophet_forecaster.is_trained if self.prophet_forecaster else False
            },
            'metrics': {
                'total_detections': self.detection_metrics['total_detections'],
                'alerts_generated': self.detection_metrics['alerts_generated'],
                'avg_processing_time_ms': avg_processing_time,
                'recent_alerts_count': len(self.recent_alerts)
            },
            'config': {
                'min_cost_impact': self.config.min_cost_impact,
                'min_confidence': self.config.min_confidence,
                'real_time_window_hours': self.config.real_time_window_hours,
                'business_rules_count': len(self.business_rules)
            }
        }
    
    async def shutdown(self):
        """Shutdown the anomaly detection engine"""
        logger.info("Shutting down anomaly detection engine")
        
        try:
            # Shutdown data pipeline
            await self.data_pipeline.shutdown()
            
            # Clear caches
            self.alert_suppression_cache.clear()
            self.recent_alerts.clear()
            
            self.is_initialized = False
            logger.info("Anomaly detection engine shutdown completed")
            
        except Exception as e:
            logger.error("Error during anomaly detection engine shutdown", error=str(e))