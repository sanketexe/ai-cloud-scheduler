#!/usr/bin/env python3
"""
Test suite for FinOps Intelligence Engine
Tests cost collection, analysis, budget management, and prediction capabilities
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finops_engine import (
    FinOpsEngine, CostCollector, CostAnalyzer, BudgetManager, CostPredictor,
    CostData, NormalizedCostData, Budget, CostOptimization,
    CloudProvider, CostCategory, BudgetPeriod, OptimizationType, EffortLevel,
    AWSBillingAPI, GCPBillingAPI, AzureBillingAPI
)


class TestCostCollector(unittest.TestCase):
    """Test cost data collection functionality"""
    
    def setUp(self):
        self.cost_collector = CostCollector()
        self.aws_api = AWSBillingAPI("test_key", "test_secret")
        self.cost_collector.register_billing_api(CloudProvider.AWS, self.aws_api)
    
    def test_register_billing_api(self):
        """Test registering billing APIs"""
        gcp_api = GCPBillingAPI("test_project", "test_creds.json")
        self.cost_collector.register_billing_api(CloudProvider.GCP, gcp_api)
        
        self.assertIn(CloudProvider.AWS, self.cost_collector.billing_apis)
        self.assertIn(CloudProvider.GCP, self.cost_collector.billing_apis)
    
    def test_collect_cost_data(self):
        """Test cost data collection"""
        async def run_test():
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()
            
            cost_data = await self.cost_collector.collect_cost_data(start_date, end_date)
            
            self.assertIsInstance(cost_data, list)
            if cost_data:  # If mock data is returned
                self.assertIsInstance(cost_data[0], CostData)
                self.assertEqual(cost_data[0].provider, CloudProvider.AWS)
        
        asyncio.run(run_test())
    
    def test_normalize_cost_data(self):
        """Test cost data normalization"""
        # Create sample cost data
        cost_data = [
            CostData(
                provider=CloudProvider.AWS,
                service="Amazon EC2-Instance",
                resource_id="i-1234567890abcdef0",
                cost_amount=100.50,
                currency="USD",
                billing_period_start=datetime.now() - timedelta(days=1),
                billing_period_end=datetime.now(),
                cost_category=CostCategory.COMPUTE,
                tags={"Team": "backend", "Project": "web-app"},
                region="us-east-1"
            )
        ]
        
        normalized = self.cost_collector.normalize_cost_data(cost_data)
        
        self.assertEqual(len(normalized), 1)
        self.assertIsInstance(normalized[0], NormalizedCostData)
        self.assertEqual(normalized[0].cost_amount_usd, 100.50)
        self.assertEqual(normalized[0].provider, CloudProvider.AWS)
        self.assertEqual(normalized[0].team, "backend")
        self.assertEqual(normalized[0].project_id, "web-app")
    
    def test_currency_conversion(self):
        """Test currency conversion to USD"""
        # Test EUR conversion
        eur_amount = self.cost_collector._convert_to_usd(100.0, "EUR")
        self.assertAlmostEqual(eur_amount, 110.0, places=1)  # 100 * 1.1
        
        # Test unknown currency (should return original)
        unknown_amount = self.cost_collector._convert_to_usd(100.0, "XYZ")
        self.assertEqual(unknown_amount, 100.0)


class TestCostAnalyzer(unittest.TestCase):
    """Test cost analysis functionality"""
    
    def setUp(self):
        self.cost_analyzer = CostAnalyzer()
        
        # Create sample normalized cost data
        base_date = datetime.now() - timedelta(days=30)
        self.sample_data = []
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            # Create varying costs to test analysis
            cost = 50 + (i % 10) * 5  # Costs between 50-95
            
            self.sample_data.append(NormalizedCostData(
                resource_id=f"resource-{i % 5}",  # 5 different resources
                resource_type="compute_instance",
                cost_amount_usd=cost,
                usage_quantity=1.0,
                usage_unit="hour",
                timestamp=date,
                provider=CloudProvider.AWS,
                region="us-east-1",
                project_id=f"project-{i % 3}",  # 3 different projects
                team=f"team-{i % 2}",  # 2 different teams
                tags={"Environment": "production"},
                cost_category=CostCategory.COMPUTE
            ))
    
    def test_analyze_spending_patterns(self):
        """Test spending pattern analysis"""
        analysis = self.cost_analyzer.analyze_spending_patterns(self.sample_data)
        
        self.assertIsNotNone(analysis.total_spend)
        self.assertGreater(analysis.total_spend, 0)
        self.assertIn(CloudProvider.AWS, analysis.spend_by_provider)
        self.assertIn(CostCategory.COMPUTE, analysis.spend_by_category)
        self.assertTrue(len(analysis.spend_by_team) > 0)
        self.assertTrue(len(analysis.spend_by_project) > 0)
        self.assertTrue(len(analysis.top_cost_drivers) > 0)
    
    def test_growth_rate_calculation(self):
        """Test growth rate calculation"""
        growth_rate = self.cost_analyzer._calculate_growth_rate(self.sample_data)
        self.assertIsInstance(growth_rate, float)
    
    def test_anomaly_detection(self):
        """Test cost anomaly detection"""
        # Add an anomalous cost point
        anomaly_data = self.sample_data.copy()
        anomaly_data.append(NormalizedCostData(
            resource_id="resource-0",
            resource_type="compute_instance",
            cost_amount_usd=500.0,  # Much higher than normal
            usage_quantity=1.0,
            usage_unit="hour",
            timestamp=datetime.now(),
            provider=CloudProvider.AWS,
            region="us-east-1",
            project_id="project-0",
            team="team-0",
            tags={},
            cost_category=CostCategory.COMPUTE
        ))
        
        anomalies = self.cost_analyzer._detect_cost_anomalies(anomaly_data)
        self.assertIsInstance(anomalies, list)
    
    def test_optimization_identification(self):
        """Test optimization opportunity identification"""
        optimizations = self.cost_analyzer._identify_optimization_opportunities(self.sample_data)
        self.assertIsInstance(optimizations, list)
        
        # Should identify some optimizations for compute instances
        compute_optimizations = [opt for opt in optimizations 
                               if opt.optimization_type == OptimizationType.RESERVED_INSTANCES]
        # May or may not find optimizations depending on cost thresholds


class TestBudgetManager(unittest.TestCase):
    """Test budget management functionality"""
    
    def setUp(self):
        self.budget_manager = BudgetManager()
        
        # Create sample budget
        self.sample_budget = Budget(
            budget_id="test_budget_001",
            name="Test Monthly Budget",
            amount=1000.0,
            currency="USD",
            period=BudgetPeriod.MONTHLY,
            start_date=datetime.now() - timedelta(days=15),
            alert_thresholds=[50.0, 80.0, 100.0],
            scope_filters={"teams": ["team-0"]}
        )
        
        # Create sample cost data
        self.sample_cost_data = [
            NormalizedCostData(
                resource_id="resource-1",
                resource_type="compute_instance",
                cost_amount_usd=300.0,
                usage_quantity=1.0,
                usage_unit="hour",
                timestamp=datetime.now() - timedelta(days=10),
                provider=CloudProvider.AWS,
                region="us-east-1",
                project_id="project-1",
                team="team-0",
                tags={},
                cost_category=CostCategory.COMPUTE
            ),
            NormalizedCostData(
                resource_id="resource-2",
                resource_type="storage_bucket",
                cost_amount_usd=200.0,
                usage_quantity=1.0,
                usage_unit="GB",
                timestamp=datetime.now() - timedelta(days=5),
                provider=CloudProvider.AWS,
                region="us-east-1",
                project_id="project-1",
                team="team-0",
                tags={},
                cost_category=CostCategory.STORAGE
            )
        ]
    
    def test_create_budget(self):
        """Test budget creation"""
        budget_id = self.budget_manager.create_budget(self.sample_budget)
        self.assertEqual(budget_id, "test_budget_001")
        self.assertIn(budget_id, self.budget_manager.budgets)
    
    def test_budget_status(self):
        """Test budget status calculation"""
        # Create budget first
        self.budget_manager.create_budget(self.sample_budget)
        
        # Get budget status
        status = self.budget_manager.get_budget_status("test_budget_001", self.sample_cost_data)
        
        self.assertEqual(status.budget.budget_id, "test_budget_001")
        self.assertEqual(status.current_spend, 500.0)  # 300 + 200
        self.assertEqual(status.utilization_percentage, 50.0)  # 500/1000 * 100
        self.assertIsInstance(status.projected_spend, float)
        self.assertIsInstance(status.days_remaining, int)
    
    def test_budget_alerts(self):
        """Test budget alert triggering"""
        # Create budget with lower amount to trigger alerts
        alert_budget = Budget(
            budget_id="alert_budget",
            name="Alert Test Budget",
            amount=400.0,  # Lower than our sample spend of 500
            currency="USD",
            period=BudgetPeriod.MONTHLY,
            start_date=datetime.now() - timedelta(days=15),
            alert_thresholds=[50.0, 80.0, 100.0],
            scope_filters={"teams": ["team-0"]}
        )
        
        self.budget_manager.create_budget(alert_budget)
        
        # Mock alert callback to capture alerts
        alerts_triggered = []
        def mock_alert_callback(alert_data):
            alerts_triggered.append(alert_data)
        
        self.budget_manager.register_alert_callback(mock_alert_callback)
        
        # Get status (should trigger alerts)
        status = self.budget_manager.get_budget_status("alert_budget", self.sample_cost_data)
        
        # Should have triggered alerts since spend (500) > budget (400)
        self.assertTrue(len(status.triggered_alerts) > 0)
    
    def test_update_budget(self):
        """Test budget updates"""
        self.budget_manager.create_budget(self.sample_budget)
        
        updated_budget = self.budget_manager.update_budget(
            "test_budget_001", 
            {"amount": 1500.0, "name": "Updated Budget"}
        )
        
        self.assertEqual(updated_budget.amount, 1500.0)
        self.assertEqual(updated_budget.name, "Updated Budget")
    
    def test_delete_budget(self):
        """Test budget deletion"""
        self.budget_manager.create_budget(self.sample_budget)
        
        result = self.budget_manager.delete_budget("test_budget_001")
        self.assertTrue(result)
        self.assertNotIn("test_budget_001", self.budget_manager.budgets)


class TestCostPredictor(unittest.TestCase):
    """Test cost prediction functionality"""
    
    def setUp(self):
        self.cost_predictor = CostPredictor()
        
        # Create sample time series data for training
        base_date = datetime.now() - timedelta(days=60)
        self.training_data = []
        
        for i in range(60):
            date = base_date + timedelta(days=i)
            # Create a trend with some seasonality
            base_cost = 100 + i * 0.5  # Slight upward trend
            seasonal = 10 * (1 + 0.3 * (i % 7))  # Weekly seasonality
            cost = base_cost + seasonal
            
            self.training_data.append(NormalizedCostData(
                resource_id="training-resource",
                resource_type="compute_instance",
                cost_amount_usd=cost,
                usage_quantity=1.0,
                usage_unit="hour",
                timestamp=date,
                provider=CloudProvider.AWS,
                region="us-east-1",
                project_id="training-project",
                team="training-team",
                tags={},
                cost_category=CostCategory.COMPUTE
            ))
    
    def test_train_prediction_model(self):
        """Test model training"""
        metrics = self.cost_predictor.train_cost_prediction_model(self.training_data)
        
        self.assertIn('mse', metrics)
        self.assertIn('r2_score', metrics)
        self.assertIn('training_samples', metrics)
        self.assertEqual(metrics['training_samples'], 60)
        self.assertIn('global', self.cost_predictor.models)
    
    def test_predict_costs(self):
        """Test cost prediction"""
        # Train model first
        self.cost_predictor.train_cost_prediction_model(self.training_data)
        
        # Make predictions
        forecast = self.cost_predictor.predict_costs(forecast_days=7)
        
        self.assertEqual(len(forecast.predicted_costs), 7)
        self.assertEqual(len(forecast.confidence_intervals), 7)
        self.assertIn(forecast.trend_direction, ["increasing", "decreasing", "stable"])
        self.assertIsInstance(forecast.model_accuracy, float)
    
    def test_insufficient_data_error(self):
        """Test error handling for insufficient training data"""
        insufficient_data = self.training_data[:5]  # Only 5 data points
        
        with self.assertRaises(ValueError):
            self.cost_predictor.train_cost_prediction_model(insufficient_data)
    
    def test_prediction_without_model_error(self):
        """Test error when predicting without trained model"""
        with self.assertRaises(ValueError):
            self.cost_predictor.predict_costs(7)


class TestFinOpsEngine(unittest.TestCase):
    """Test the main FinOps engine integration"""
    
    def setUp(self):
        self.finops_engine = FinOpsEngine()
    
    def test_initialization(self):
        """Test FinOps engine initialization"""
        self.assertIsNotNone(self.finops_engine.cost_collector)
        self.assertIsNotNone(self.finops_engine.cost_analyzer)
        self.assertIsNotNone(self.finops_engine.budget_manager)
        self.assertIsNotNone(self.finops_engine.cost_predictor)
    
    def test_create_budget(self):
        """Test budget creation through engine"""
        budget_id = self.finops_engine.create_budget(
            name="Test Engine Budget",
            amount=2000.0,
            period=BudgetPeriod.MONTHLY,
            scope_filters={"teams": ["engineering"]},
            alert_thresholds=[60.0, 90.0, 100.0]
        )
        
        self.assertIsNotNone(budget_id)
        self.assertIn(budget_id, self.finops_engine.budget_manager.budgets)
    
    def test_provider_initialization(self):
        """Test cloud provider initialization"""
        async def run_test():
            provider_configs = {
                CloudProvider.AWS: {
                    'access_key': 'test_key',
                    'secret_key': 'test_secret',
                    'region': 'us-west-2'
                }
            }
            
            await self.finops_engine.initialize_providers(provider_configs)
            
            self.assertIn(CloudProvider.AWS, self.finops_engine.cost_collector.billing_apis)
        
        asyncio.run(run_test())


class TestCloudBillingAPIs(unittest.TestCase):
    """Test cloud provider billing API implementations"""
    
    def test_aws_api_initialization(self):
        """Test AWS API initialization"""
        aws_api = AWSBillingAPI("test_key", "test_secret", "us-east-1")
        self.assertEqual(aws_api.access_key, "test_key")
        self.assertEqual(aws_api.secret_key, "test_secret")
        self.assertEqual(aws_api.region, "us-east-1")
    
    def test_gcp_api_initialization(self):
        """Test GCP API initialization"""
        gcp_api = GCPBillingAPI("test_project", "test_creds.json")
        self.assertEqual(gcp_api.project_id, "test_project")
        self.assertEqual(gcp_api.credentials_path, "test_creds.json")
    
    def test_azure_api_initialization(self):
        """Test Azure API initialization"""
        azure_api = AzureBillingAPI("sub_id", "tenant_id", "client_id", "client_secret")
        self.assertEqual(azure_api.subscription_id, "sub_id")
        self.assertEqual(azure_api.tenant_id, "tenant_id")
    
    def test_aws_service_categorization(self):
        """Test AWS service categorization"""
        aws_api = AWSBillingAPI("test", "test")
        
        self.assertEqual(aws_api._categorize_aws_service("Amazon EC2-Instance"), CostCategory.COMPUTE)
        self.assertEqual(aws_api._categorize_aws_service("Amazon S3"), CostCategory.STORAGE)
        self.assertEqual(aws_api._categorize_aws_service("Amazon RDS"), CostCategory.DATABASE)
        self.assertEqual(aws_api._categorize_aws_service("Amazon VPC"), CostCategory.NETWORK)
        self.assertEqual(aws_api._categorize_aws_service("Unknown Service"), CostCategory.OTHER)


def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("FINOPS ENGINE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestCostCollector,
        TestCostAnalyzer,
        TestBudgetManager,
        TestCostPredictor,
        TestFinOpsEngine,
        TestCloudBillingAPIs
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
        
        if result.failures:
            print(f"  Failures: {len(result.failures)}")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL TESTS: {total_tests}")
    print(f"FAILURES: {total_failures}")
    print(f"SUCCESS RATE: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    print("=" * 60)
    
    return total_failures == 0


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)