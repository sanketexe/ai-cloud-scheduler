"""
Unit tests for enhanced data models
Tests data model validation, serialization, and business logic
"""
import unittest
from datetime import datetime, timedelta
from enhanced_models import (
    CostConstraints, PerformanceRequirements, ComplianceRequirements,
    CostDataPoint, PerformanceMetrics, UtilizationTrends,
    EnhancedWorkload, EnhancedVirtualMachine,
    WorkloadPriority, CostOptimizationLevel, SeverityLevel, HealthStatus
)


class TestCostConstraints(unittest.TestCase):
    """Test CostConstraints data model"""
    
    def test_valid_cost_constraints(self):
        """Test creation of valid cost constraints"""
        constraints = CostConstraints(
            max_hourly_cost=5.0,
            max_monthly_budget=1000.0,
            cost_optimization_preference=CostOptimizationLevel.MODERATE,
            currency="USD"
        )
        
        self.assertEqual(constraints.max_hourly_cost, 5.0)
        self.assertEqual(constraints.max_monthly_budget, 1000.0)
        self.assertEqual(constraints.cost_optimization_preference, CostOptimizationLevel.MODERATE)
        self.assertEqual(constraints.currency, "USD")
    
    def test_invalid_hourly_cost(self):
        """Test validation of negative hourly cost"""
        with self.assertRaises(ValueError) as context:
            CostConstraints(max_hourly_cost=-1.0)
        
        self.assertIn("max_hourly_cost must be positive", str(context.exception))
    
    def test_zero_hourly_cost(self):
        """Test validation of zero hourly cost"""
        with self.assertRaises(ValueError):
            CostConstraints(max_hourly_cost=0.0)
    
    def test_default_values(self):
        """Test default values for optional fields"""
        constraints = CostConstraints(max_hourly_cost=2.5)
        
        self.assertIsNone(constraints.max_monthly_budget)
        self.assertEqual(constraints.cost_optimization_preference, CostOptimizationLevel.MODERATE)
        self.assertEqual(constraints.currency, "USD")


class TestPerformanceRequirements(unittest.TestCase):
    """Test PerformanceRequirements data model"""
    
    def test_valid_performance_requirements(self):
        """Test creation of valid performance requirements"""
        requirements = PerformanceRequirements(
            min_cpu_performance=80.0,
            max_latency_ms=100,
            availability_requirement=0.99,
            throughput_requirement=1000,
            min_memory_bandwidth=50.0
        )
        
        self.assertEqual(requirements.min_cpu_performance, 80.0)
        self.assertEqual(requirements.max_latency_ms, 100)
        self.assertEqual(requirements.availability_requirement, 0.99)
        self.assertEqual(requirements.throughput_requirement, 1000)
        self.assertEqual(requirements.min_memory_bandwidth, 50.0)    

    def test_invalid_cpu_performance_range(self):
        """Test validation of CPU performance range"""
        # Test negative value
        with self.assertRaises(ValueError) as context:
            PerformanceRequirements(
                min_cpu_performance=-10.0,
                max_latency_ms=100,
                availability_requirement=0.99
            )
        self.assertIn("min_cpu_performance must be between 0 and 100", str(context.exception))
        
        # Test value over 100
        with self.assertRaises(ValueError):
            PerformanceRequirements(
                min_cpu_performance=150.0,
                max_latency_ms=100,
                availability_requirement=0.99
            )
    
    def test_invalid_availability_range(self):
        """Test validation of availability requirement range"""
        # Test negative value
        with self.assertRaises(ValueError) as context:
            PerformanceRequirements(
                min_cpu_performance=80.0,
                max_latency_ms=100,
                availability_requirement=-0.1
            )
        self.assertIn("availability_requirement must be between 0.0 and 1.0", str(context.exception))
        
        # Test value over 1.0
        with self.assertRaises(ValueError):
            PerformanceRequirements(
                min_cpu_performance=80.0,
                max_latency_ms=100,
                availability_requirement=1.5
            )
    
    def test_invalid_latency(self):
        """Test validation of latency requirement"""
        with self.assertRaises(ValueError) as context:
            PerformanceRequirements(
                min_cpu_performance=80.0,
                max_latency_ms=-50,
                availability_requirement=0.99
            )
        self.assertIn("max_latency_ms must be positive", str(context.exception))


class TestComplianceRequirements(unittest.TestCase):
    """Test ComplianceRequirements data model"""
    
    def test_default_compliance_requirements(self):
        """Test default compliance requirements"""
        requirements = ComplianceRequirements()
        
        self.assertEqual(requirements.data_residency_regions, [])
        self.assertEqual(requirements.compliance_standards, [])
        self.assertTrue(requirements.encryption_required)
        self.assertTrue(requirements.audit_logging_required)
        self.assertFalse(requirements.network_isolation_required)
    
    def test_custom_compliance_requirements(self):
        """Test custom compliance requirements"""
        requirements = ComplianceRequirements(
            data_residency_regions=["us-east-1", "us-west-2"],
            compliance_standards=["SOC2", "GDPR"],
            encryption_required=True,
            audit_logging_required=True,
            network_isolation_required=True
        )
        
        self.assertEqual(requirements.data_residency_regions, ["us-east-1", "us-west-2"])
        self.assertEqual(requirements.compliance_standards, ["SOC2", "GDPR"])
        self.assertTrue(requirements.encryption_required)
        self.assertTrue(requirements.audit_logging_required)
        self.assertTrue(requirements.network_isolation_required)
    
    def test_is_compliant_region(self):
        """Test region compliance checking"""
        # No restrictions - all regions compliant
        requirements = ComplianceRequirements()
        self.assertTrue(requirements.is_compliant_region("us-east-1"))
        self.assertTrue(requirements.is_compliant_region("eu-west-1"))
        
        # With restrictions
        requirements = ComplianceRequirements(
            data_residency_regions=["us-east-1", "us-west-2"]
        )
        self.assertTrue(requirements.is_compliant_region("us-east-1"))
        self.assertTrue(requirements.is_compliant_region("us-west-2"))
        self.assertFalse(requirements.is_compliant_region("eu-west-1"))


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics data model"""
    
    def test_valid_performance_metrics(self):
        """Test creation of valid performance metrics"""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            cpu_utilization=75.5,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=25.5,
            throughput=500.0,
            timestamp=timestamp
        )
        
        self.assertEqual(metrics.cpu_utilization, 75.5)
        self.assertEqual(metrics.memory_utilization, 60.0)
        self.assertEqual(metrics.network_io_mbps, 100.0)
        self.assertEqual(metrics.disk_io_mbps, 50.0)
        self.assertEqual(metrics.response_time_ms, 25.5)
        self.assertEqual(metrics.throughput, 500.0)
        self.assertEqual(metrics.timestamp, timestamp)
    
    def test_invalid_cpu_utilization(self):
        """Test validation of CPU utilization range"""
        timestamp = datetime.now()
        
        # Test negative value
        with self.assertRaises(ValueError) as context:
            PerformanceMetrics(
                cpu_utilization=-10.0,
                memory_utilization=60.0,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
                response_time_ms=25.5,
                throughput=500.0,
                timestamp=timestamp
            )
        self.assertIn("cpu_utilization must be between 0 and 100", str(context.exception))
        
        # Test value over 100
        with self.assertRaises(ValueError):
            PerformanceMetrics(
                cpu_utilization=150.0,
                memory_utilization=60.0,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
                response_time_ms=25.5,
                throughput=500.0,
                timestamp=timestamp
            )
    
    def test_invalid_memory_utilization(self):
        """Test validation of memory utilization range"""
        timestamp = datetime.now()
        
        with self.assertRaises(ValueError) as context:
            PerformanceMetrics(
                cpu_utilization=75.0,
                memory_utilization=-5.0,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
                response_time_ms=25.5,
                throughput=500.0,
                timestamp=timestamp
            )
        self.assertIn("memory_utilization must be between 0 and 100", str(context.exception))


if __name__ == '__main__':
    unittest.main()