"""
A/B Testing Framework for ML Model Comparison

This module provides comprehensive A/B testing capabilities for comparing
ML models in production environments with statistical significance testing,
traffic splitting, and automated decision making.
"""

import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

from .database import get_db_session
from .exceptions import ABTestingError, ValidationError

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TestType(Enum):
    """Type of A/B test"""
    MODEL_COMPARISON = "model_comparison"
    FEATURE_COMPARISON = "feature_comparison"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    HYPERPARAMETER_COMPARISON = "hyperparameter_comparison"

class TrafficSplitStrategy(Enum):
    """Traffic splitting strategy"""
    RANDOM = "random"
    WEIGHTED = "weighted"
    SEQUENTIAL = "sequential"
    GEOGRAPHIC = "geographic"

@dataclass
class TestVariant:
    """A/B test variant configuration"""
    variant_id: str
    name: str
    description: str
    model_id: str
    traffic_percentage: float
    configuration: Dict[str, Any]
    
    def __post_init__(self):
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")

@dataclass
class TestMetrics:
    """A/B test metrics"""
    variant_id: str
    sample_size: int
    conversion_rate: float
    average_response_time: float
    error_rate: float
    custom_metrics: Dict[str, float]
    confidence_interval: Tuple[float, float]
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

@dataclass
class TestConfiguration:
    """A/B test configuration"""
    test_name: str
    description: str
    test_type: TestType
    variants: List[TestVariant]
    primary_metric: str
    secondary_metrics: List[str]
    minimum_sample_size: int
    significance_level: float
    power: float
    traffic_split_strategy: TrafficSplitStrategy
    duration_days: int
    auto_conclude: bool = True
    
    def __post_init__(self):
        if len(self.variants) < 2:
            raise ValueError("At least 2 variants required for A/B testing")
        
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

class ABTestingFramework:
    """
    Comprehensive A/B testing framework for ML model comparison
    with statistical significance testing and automated decision making.
    """
    
    def __init__(self):
        """Initialize A/B Testing Framework"""
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.traffic_router = TrafficRouter()
        
    async def create_test(
        self,
        config: TestConfiguration,
        account_id: str = None
    ) -> str:
        """
        Create a new A/B test
        
        Args:
            config: Test configuration
            account_id: Account identifier
            
        Returns:
            Test ID
        """
        try:
            test_id = str(uuid.uuid4())
            
            # Validate configuration
            await self._validate_test_config(config)
            
            # Initialize test
            test_info = {
                "test_id": test_id,
                "config": config,
                "account_id": account_id or "default",
                "status": TestStatus.DRAFT,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "ended_at": None,
                "results": {},
                "metrics_history": [],
                "traffic_assignments": {}
            }
            
            self.active_tests[test_id] = test_info
            
            logger.info(f"A/B test {test_id} created: {config.test_name}")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {str(e)}")
            raise ABTestingError(f"Failed to create A/B test: {str(e)}")
    
    async def start_test(self, test_id: str) -> Dict[str, Any]:
        """
        Start an A/B test
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test start results
        """
        try:
            if test_id not in self.active_tests:
                raise ABTestingError(f"Test {test_id} not found")
            
            test_info = self.active_tests[test_id]
            
            if test_info["status"] != TestStatus.DRAFT:
                raise ABTestingError(f"Test {test_id} cannot be started. Current status: {test_info['status']}")
            
            # Initialize traffic routing
            await self.traffic_router.configure_test(test_id, test_info["config"])
            
            # Update test status
            test_info["status"] = TestStatus.RUNNING
            test_info["started_at"] = datetime.utcnow()
            
            # Initialize metrics tracking
            for variant in test_info["config"].variants:
                test_info["results"][variant.variant_id] = {
                    "sample_size": 0,
                    "conversions": 0,
                    "response_times": [],
                    "errors": 0,
                    "custom_metrics": {}
                }
            
            logger.info(f"A/B test {test_id} started")
            
            return {
                "test_id": test_id,
                "status": TestStatus.RUNNING.value,
                "started_at": test_info["started_at"],
                "expected_end_date": test_info["started_at"] + timedelta(days=test_info["config"].duration_days)
            }
            
        except Exception as e:
            logger.error(f"Failed to start A/B test {test_id}: {str(e)}")
            raise ABTestingError(f"Failed to start A/B test: {str(e)}")
    
    async def record_interaction(
        self,
        test_id: str,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record user interaction for A/B test
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            interaction_data: Interaction metrics and data
            
        Returns:
            Assigned variant and recording results
        """
        try:
            if test_id not in self.active_tests:
                raise ABTestingError(f"Test {test_id} not found")
            
            test_info = self.active_tests[test_id]
            
            if test_info["status"] != TestStatus.RUNNING:
                raise ABTestingError(f"Test {test_id} is not running")
            
            # Get variant assignment
            variant_id = await self.traffic_router.assign_variant(test_id, user_id)
            
            # Record interaction
            variant_results = test_info["results"][variant_id]
            variant_results["sample_size"] += 1
            
            # Record conversion if applicable
            if interaction_data.get("converted", False):
                variant_results["conversions"] += 1
            
            # Record response time
            if "response_time" in interaction_data:
                variant_results["response_times"].append(interaction_data["response_time"])
            
            # Record errors
            if interaction_data.get("error", False):
                variant_results["errors"] += 1
            
            # Record custom metrics
            for metric_name, value in interaction_data.get("custom_metrics", {}).items():
                if metric_name not in variant_results["custom_metrics"]:
                    variant_results["custom_metrics"][metric_name] = []
                variant_results["custom_metrics"][metric_name].append(value)
            
            # Check if test should be concluded
            if test_info["config"].auto_conclude:
                await self._check_test_conclusion(test_id)
            
            return {
                "test_id": test_id,
                "variant_id": variant_id,
                "recorded": True
            }
            
        except Exception as e:
            logger.error(f"Failed to record interaction for test {test_id}: {str(e)}")
            raise ABTestingError(f"Failed to record interaction: {str(e)}")
    
    async def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results with statistical significance testing
        
        Args:
            test_id: Test identifier
            
        Returns:
            Comprehensive test analysis results
        """
        try:
            if test_id not in self.active_tests:
                raise ABTestingError(f"Test {test_id} not found")
            
            test_info = self.active_tests[test_id]
            config = test_info["config"]
            results = test_info["results"]
            
            analysis = {
                "test_id": test_id,
                "test_name": config.test_name,
                "status": test_info["status"].value,
                "started_at": test_info["started_at"],
                "analysis_date": datetime.utcnow(),
                "variants": {},
                "statistical_tests": {},
                "recommendations": []
            }
            
            # Calculate metrics for each variant
            for variant in config.variants:
                variant_id = variant.variant_id
                variant_results = results[variant_id]
                
                # Calculate conversion rate
                conversion_rate = 0.0
                if variant_results["sample_size"] > 0:
                    conversion_rate = variant_results["conversions"] / variant_results["sample_size"]
                
                # Calculate average response time
                avg_response_time = 0.0
                if variant_results["response_times"]:
                    avg_response_time = np.mean(variant_results["response_times"])
                
                # Calculate error rate
                error_rate = 0.0
                if variant_results["sample_size"] > 0:
                    error_rate = variant_results["errors"] / variant_results["sample_size"]
                
                # Calculate confidence interval for conversion rate
                confidence_interval = self._calculate_confidence_interval(
                    variant_results["conversions"],
                    variant_results["sample_size"],
                    config.significance_level
                )
                
                # Calculate custom metrics
                custom_metrics = {}
                for metric_name, values in variant_results["custom_metrics"].items():
                    if values:
                        custom_metrics[metric_name] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "count": len(values)
                        }
                
                analysis["variants"][variant_id] = {
                    "name": variant.name,
                    "sample_size": variant_results["sample_size"],
                    "conversion_rate": conversion_rate,
                    "average_response_time": avg_response_time,
                    "error_rate": error_rate,
                    "confidence_interval": confidence_interval,
                    "custom_metrics": custom_metrics
                }
            
            # Perform statistical significance tests
            analysis["statistical_tests"] = await self._perform_statistical_tests(
                config, results, analysis["variants"]
            )
            
            # Generate recommendations
            analysis["recommendations"] = await self._generate_recommendations(
                config, analysis["variants"], analysis["statistical_tests"]
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze test results for {test_id}: {str(e)}")
            raise ABTestingError(f"Failed to analyze test results: {str(e)}")
    
    async def conclude_test(
        self,
        test_id: str,
        winning_variant_id: str = None,
        reason: str = None
    ) -> Dict[str, Any]:
        """
        Conclude an A/B test and implement the winning variant
        
        Args:
            test_id: Test identifier
            winning_variant_id: ID of winning variant (if any)
            reason: Reason for conclusion
            
        Returns:
            Test conclusion results
        """
        try:
            if test_id not in self.active_tests:
                raise ABTestingError(f"Test {test_id} not found")
            
            test_info = self.active_tests[test_id]
            
            if test_info["status"] not in [TestStatus.RUNNING, TestStatus.PAUSED]:
                raise ABTestingError(f"Test {test_id} cannot be concluded. Current status: {test_info['status']}")
            
            # Get final analysis
            final_analysis = await self.analyze_test_results(test_id)
            
            # Update test status
            test_info["status"] = TestStatus.COMPLETED
            test_info["ended_at"] = datetime.utcnow()
            test_info["final_analysis"] = final_analysis
            test_info["winning_variant_id"] = winning_variant_id
            test_info["conclusion_reason"] = reason
            
            # Stop traffic routing
            await self.traffic_router.stop_test(test_id)
            
            # Store results
            self.test_results[test_id] = test_info
            
            conclusion_results = {
                "test_id": test_id,
                "status": TestStatus.COMPLETED.value,
                "ended_at": test_info["ended_at"],
                "winning_variant_id": winning_variant_id,
                "reason": reason,
                "final_analysis": final_analysis
            }
            
            logger.info(f"A/B test {test_id} concluded. Winner: {winning_variant_id}")
            
            return conclusion_results
            
        except Exception as e:
            logger.error(f"Failed to conclude test {test_id}: {str(e)}")
            raise ABTestingError(f"Failed to conclude test: {str(e)}")
    
    async def _validate_test_config(self, config: TestConfiguration):
        """Validate A/B test configuration"""
        # Check minimum sample size
        if config.minimum_sample_size < 100:
            raise ValidationError("Minimum sample size should be at least 100")
        
        # Check significance level
        if not 0.01 <= config.significance_level <= 0.1:
            raise ValidationError("Significance level should be between 0.01 and 0.1")
        
        # Check power
        if not 0.7 <= config.power <= 0.95:
            raise ValidationError("Statistical power should be between 0.7 and 0.95")
        
        # Validate variant configurations
        for variant in config.variants:
            if not variant.model_id:
                raise ValidationError(f"Model ID required for variant {variant.variant_id}")
    
    async def _check_test_conclusion(self, test_id: str):
        """Check if test should be automatically concluded"""
        test_info = self.active_tests[test_id]
        config = test_info["config"]
        
        # Check duration
        if test_info["started_at"]:
            duration = datetime.utcnow() - test_info["started_at"]
            if duration.days >= config.duration_days:
                await self.conclude_test(test_id, reason="Duration limit reached")
                return
        
        # Check minimum sample size
        total_samples = sum(
            variant_results["sample_size"]
            for variant_results in test_info["results"].values()
        )
        
        if total_samples >= config.minimum_sample_size:
            # Perform significance test
            analysis = await self.analyze_test_results(test_id)
            
            # Check if we have a statistically significant winner
            if analysis["statistical_tests"].get("significant_difference", False):
                # Find the best performing variant
                best_variant = max(
                    analysis["variants"].items(),
                    key=lambda x: x[1]["conversion_rate"]
                )
                
                await self.conclude_test(
                    test_id,
                    winning_variant_id=best_variant[0],
                    reason="Statistically significant difference detected"
                )
    
    def _calculate_confidence_interval(
        self,
        successes: int,
        trials: int,
        alpha: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for conversion rate"""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * np.sqrt(p * (1 - p) / trials)
        
        return (max(0, p - margin), min(1, p + margin))
    
    async def _perform_statistical_tests(
        self,
        config: TestConfiguration,
        results: Dict[str, Any],
        variant_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        statistical_tests = {}
        
        # Get variant IDs
        variant_ids = list(variant_metrics.keys())
        
        if len(variant_ids) == 2:
            # Two-sample tests
            variant_a_id, variant_b_id = variant_ids
            
            # Conversion rate test (Chi-square test)
            a_conversions = results[variant_a_id]["conversions"]
            a_samples = results[variant_a_id]["sample_size"]
            b_conversions = results[variant_b_id]["conversions"]
            b_samples = results[variant_b_id]["sample_size"]
            
            if a_samples > 0 and b_samples > 0:
                # Create contingency table
                contingency_table = np.array([
                    [a_conversions, a_samples - a_conversions],
                    [b_conversions, b_samples - b_conversions]
                ])
                
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                statistical_tests["conversion_rate_test"] = {
                    "test_type": "chi_square",
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "significant": p_value < config.significance_level
                }
            
            # Response time test (t-test)
            a_response_times = results[variant_a_id]["response_times"]
            b_response_times = results[variant_b_id]["response_times"]
            
            if len(a_response_times) > 1 and len(b_response_times) > 1:
                t_stat, p_value = ttest_ind(a_response_times, b_response_times)
                
                statistical_tests["response_time_test"] = {
                    "test_type": "t_test",
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < config.significance_level
                }
        
        # Overall significance
        significant_tests = [
            test for test in statistical_tests.values()
            if test.get("significant", False)
        ]
        
        statistical_tests["significant_difference"] = len(significant_tests) > 0
        statistical_tests["significant_tests_count"] = len(significant_tests)
        
        return statistical_tests
    
    async def _generate_recommendations(
        self,
        config: TestConfiguration,
        variant_metrics: Dict[str, Any],
        statistical_tests: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not statistical_tests.get("significant_difference", False):
            recommendations.append(
                "No statistically significant difference found between variants. "
                "Consider running the test longer or increasing sample size."
            )
            return recommendations
        
        # Find best performing variant
        best_variant = max(
            variant_metrics.items(),
            key=lambda x: x[1]["conversion_rate"]
        )
        
        recommendations.append(
            f"Variant {best_variant[0]} shows the best performance with "
            f"{best_variant[1]['conversion_rate']:.2%} conversion rate."
        )
        
        # Check sample sizes
        min_sample_size = min(v["sample_size"] for v in variant_metrics.values())
        if min_sample_size < config.minimum_sample_size:
            recommendations.append(
                f"Some variants have low sample sizes (minimum: {min_sample_size}). "
                "Consider collecting more data for robust conclusions."
            )
        
        # Check error rates
        high_error_variants = [
            variant_id for variant_id, metrics in variant_metrics.items()
            if metrics["error_rate"] > 0.05  # 5% error rate threshold
        ]
        
        if high_error_variants:
            recommendations.append(
                f"Variants {', '.join(high_error_variants)} have high error rates. "
                "Investigate potential issues before deployment."
            )
        
        return recommendations
    
    async def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an A/B test"""
        if test_id in self.active_tests:
            test_info = self.active_tests[test_id]
            return {
                "test_id": test_id,
                "test_name": test_info["config"].test_name,
                "status": test_info["status"].value,
                "created_at": test_info["created_at"],
                "started_at": test_info["started_at"],
                "ended_at": test_info["ended_at"],
                "total_samples": sum(
                    variant_results["sample_size"]
                    for variant_results in test_info["results"].values()
                )
            }
        
        if test_id in self.test_results:
            test_info = self.test_results[test_id]
            return {
                "test_id": test_id,
                "test_name": test_info["config"].test_name,
                "status": test_info["status"].value,
                "created_at": test_info["created_at"],
                "started_at": test_info["started_at"],
                "ended_at": test_info["ended_at"],
                "winning_variant_id": test_info.get("winning_variant_id")
            }
        
        return None
    
    async def list_tests(self, account_id: str = None) -> List[Dict[str, Any]]:
        """List all A/B tests for an account"""
        tests = []
        
        # Active tests
        for test_id, test_info in self.active_tests.items():
            if account_id is None or test_info["account_id"] == account_id:
                tests.append({
                    "test_id": test_id,
                    "test_name": test_info["config"].test_name,
                    "status": test_info["status"].value,
                    "created_at": test_info["created_at"],
                    "started_at": test_info["started_at"]
                })
        
        # Completed tests
        for test_id, test_info in self.test_results.items():
            if account_id is None or test_info["account_id"] == account_id:
                tests.append({
                    "test_id": test_id,
                    "test_name": test_info["config"].test_name,
                    "status": test_info["status"].value,
                    "created_at": test_info["created_at"],
                    "started_at": test_info["started_at"],
                    "ended_at": test_info["ended_at"]
                })
        
        return sorted(tests, key=lambda x: x["created_at"], reverse=True)


class TrafficRouter:
    """Traffic routing for A/B tests"""
    
    def __init__(self):
        """Initialize traffic router"""
        self.test_configurations: Dict[str, TestConfiguration] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # test_id -> user_id -> variant_id
    
    async def configure_test(self, test_id: str, config: TestConfiguration):
        """Configure traffic routing for a test"""
        self.test_configurations[test_id] = config
        self.user_assignments[test_id] = {}
    
    async def assign_variant(self, test_id: str, user_id: str) -> str:
        """Assign a variant to a user"""
        if test_id not in self.test_configurations:
            raise ABTestingError(f"Test {test_id} not configured")
        
        # Check if user already assigned
        if user_id in self.user_assignments[test_id]:
            return self.user_assignments[test_id][user_id]
        
        config = self.test_configurations[test_id]
        
        # Assign based on traffic split strategy
        if config.traffic_split_strategy == TrafficSplitStrategy.RANDOM:
            variant = self._random_assignment(config.variants)
        elif config.traffic_split_strategy == TrafficSplitStrategy.WEIGHTED:
            variant = self._weighted_assignment(config.variants)
        else:
            # Default to random
            variant = self._random_assignment(config.variants)
        
        # Store assignment
        self.user_assignments[test_id][user_id] = variant.variant_id
        
        return variant.variant_id
    
    def _random_assignment(self, variants: List[TestVariant]) -> TestVariant:
        """Random variant assignment"""
        return np.random.choice(variants)
    
    def _weighted_assignment(self, variants: List[TestVariant]) -> TestVariant:
        """Weighted variant assignment based on traffic percentages"""
        weights = [v.traffic_percentage for v in variants]
        return np.random.choice(variants, p=np.array(weights) / 100.0)
    
    async def stop_test(self, test_id: str):
        """Stop traffic routing for a test"""
        if test_id in self.test_configurations:
            del self.test_configurations[test_id]
        if test_id in self.user_assignments:
            del self.user_assignments[test_id]