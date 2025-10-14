"""
Unit tests for EnhancedWorkload and EnhancedVirtualMachine classes
"""
import unittest
from datetime import datetime, timedelta
from enhanced_models import (
    CostConstraints, PerformanceRequirements, ComplianceRequirements,
    CostDataPoint, PerformanceMetrics, UtilizationTrends,
    EnhancedWorkload, EnhancedVirtualMachine,
    WorkloadPriority, CostOptimizationLevel, HealthStatus
)


class TestEnhancedWorkload(unittest.TestCase):
    """Test EnhancedWorkload data model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cost_constraints = CostConstraints(max_hourly_cost=5.0)
        self.performance_requirements = PerformanceRequirements(
            min_cpu_performance=80.0,
            max_latency_ms=100,
            availability_requirement=0.99
        )
        self.compliance_requirements = ComplianceRequirements(
            data_residency_regions=["us-east-1"]
        )
    
    def test_basic_enhanced_workload(self):
        """Test creation of basic enhanced workload"""
        workload = EnhancedWorkload(
            workload_id=1,
            cpu_required=2,
            memory_required_gb=4
        )
        
        self.assertEqual(workload.id, 1)
        self.assertEqual(workload.cpu_required, 2)
        self.assertEqual(workload.memory_required_gb, 4)
        self.assertIsNone(workload.cost_constraints)
        self.assertIsNone(workload.performance_requirements)
        self.assertIsNotNone(workload.compliance_requirements)  # Default created
        self.assertEqual(workload.priority, WorkloadPriority.NORMAL)
        self.assertEqual(workload.tags, {})
        self.assertIsNotNone(workload.created_at)
    
    def test_full_enhanced_workload(self):
        """Test creation of fully configured enhanced workload"""
        workload = EnhancedWorkload(
            workload_id=2,
            cpu_required=4,
            memory_required_gb=8,
            cost_constraints=self.cost_constraints,
            performance_requirements=self.performance_requirements,
            compliance_requirements=self.compliance_requirements,
            priority=WorkloadPriority.HIGH,
            tags={"project": "test", "team": "engineering"}
        )
        
        self.assertEqual(workload.id, 2)
        self.assertEqual(workload.cpu_required, 4)
        self.assertEqual(workload.memory_required_gb, 8)
        self.assertEqual(workload.cost_constraints, self.cost_constraints)
        self.assertEqual(workload.performance_requirements, self.performance_requirements)
        self.assertEqual(workload.compliance_requirements, self.compliance_requirements)
        self.assertEqual(workload.priority, WorkloadPriority.HIGH)
        self.assertEqual(workload.tags["project"], "test")
        self.assertEqual(workload.tags["team"], "engineering")
    
    def test_workload_helper_methods(self):
        """Test workload helper methods"""
        # Workload without constraints
        basic_workload = EnhancedWorkload(1, 2, 4)
        self.assertFalse(basic_workload.has_cost_constraints())
        self.assertFalse(basic_workload.has_performance_requirements())
        self.assertIsNone(basic_workload.get_max_acceptable_cost())
        self.assertFalse(basic_workload.is_high_priority())
        
        # Workload with constraints
        enhanced_workload = EnhancedWorkload(
            2, 4, 8,
            cost_constraints=self.cost_constraints,
            performance_requirements=self.performance_requirements,
            priority=WorkloadPriority.CRITICAL
        )
        self.assertTrue(enhanced_workload.has_cost_constraints())
        self.assertTrue(enhanced_workload.has_performance_requirements())
        self.assertEqual(enhanced_workload.get_max_acceptable_cost(), 5.0)
        self.assertTrue(enhanced_workload.is_high_priority())


class TestEnhancedVirtualMachine(unittest.TestCase):
    """Test EnhancedVirtualMachine data model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = datetime.now()
        self.performance_metrics = PerformanceMetrics(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=25.0,
            throughput=500.0,
            timestamp=self.timestamp
        )
        self.cost_data_point = CostDataPoint(
            timestamp=self.timestamp,
            hourly_cost=3.5,
            cumulative_cost=84.0,
            cost_breakdown={"compute": 2.5, "storage": 1.0}
        )
    
    def test_basic_enhanced_vm(self):
        """Test creation of basic enhanced VM"""
        vm = EnhancedVirtualMachine(
            vm_id=1,
            cpu_capacity=4,
            memory_capacity_gb=16,
            provider="aws"
        )
        
        self.assertEqual(vm.vm_id, 1)
        self.assertEqual(vm.cpu_capacity, 4)
        self.assertEqual(vm.memory_capacity_gb, 16)
        self.assertEqual(vm.provider, "aws")
        self.assertEqual(vm.region, "us-east-1")
        self.assertEqual(vm.availability_zone, "us-east-1a")
        self.assertIsNone(vm.current_performance_metrics)
        self.assertEqual(vm.cost_history, [])
        self.assertEqual(vm.health_status, HealthStatus.HEALTHY)
        self.assertIsNone(vm.utilization_trends)
        self.assertEqual(vm.compliance_certifications, [])
        self.assertEqual(vm.performance_score, 85.0)
    
    def test_vm_with_custom_region(self):
        """Test VM creation with custom region and AZ"""
        vm = EnhancedVirtualMachine(
            vm_id=2,
            cpu_capacity=8,
            memory_capacity_gb=32,
            provider="gcp",
            region="us-west-2",
            availability_zone="us-west-2b"
        )
        
        self.assertEqual(vm.region, "us-west-2")
        self.assertEqual(vm.availability_zone, "us-west-2b")
        self.assertEqual(vm.provider, "gcp")
    
    def test_update_performance_metrics(self):
        """Test updating performance metrics and health status"""
        vm = EnhancedVirtualMachine(1, 4, 16, "aws")
        
        # Update with normal metrics
        vm.update_performance_metrics(self.performance_metrics)
        self.assertEqual(vm.current_performance_metrics, self.performance_metrics)
        self.assertEqual(vm.health_status, HealthStatus.HEALTHY)
        
        # Update with high utilization (warning)
        high_util_metrics = PerformanceMetrics(
            cpu_utilization=92.0,
            memory_utilization=88.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=25.0,
            throughput=500.0,
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(high_util_metrics)
        self.assertEqual(vm.health_status, HealthStatus.WARNING)
        
        # Update with very high utilization (unhealthy)
        very_high_util_metrics = PerformanceMetrics(
            cpu_utilization=97.0,
            memory_utilization=96.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=25.0,
            throughput=500.0,
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(very_high_util_metrics)
        self.assertEqual(vm.health_status, HealthStatus.UNHEALTHY)
    
    def test_cost_data_management(self):
        """Test cost data point management"""
        vm = EnhancedVirtualMachine(1, 4, 16, "aws")
        
        # Add cost data points
        for i in range(5):
            cost_point = CostDataPoint(
                timestamp=datetime.now() - timedelta(hours=i),
                hourly_cost=3.0 + i * 0.1,
                cumulative_cost=50.0 + i * 3.0
            )
            vm.add_cost_data_point(cost_point)
        
        self.assertEqual(len(vm.cost_history), 5)
        
        # Test cost history limit (simulate adding 800 points)
        for i in range(800):
            cost_point = CostDataPoint(
                timestamp=datetime.now() - timedelta(hours=i),
                hourly_cost=3.0,
                cumulative_cost=50.0
            )
            vm.add_cost_data_point(cost_point)
        
        # Should be limited to 720 points (30 days)
        self.assertEqual(len(vm.cost_history), 720)
    
    def test_get_current_hourly_cost(self):
        """Test current hourly cost calculation"""
        from environment import CloudProvider
        provider = CloudProvider("aws", 0.5, 0.2)  # $0.5/vCPU, $0.2/GB
        vm = EnhancedVirtualMachine(1, 4, 16, provider)
        # Base cost should be 4 * 0.5 + 16 * 0.2 = 2.0 + 3.2 = 5.2
        
        # Without performance metrics - should return base cost
        self.assertEqual(vm.get_current_hourly_cost(), 5.2)
        
        # With performance metrics - should adjust based on utilization
        vm.update_performance_metrics(self.performance_metrics)
        # Average utilization: (75 + 60) / 2 = 67.5%
        # Expected cost: 5.2 * (0.5 + 0.5 * 0.675) = 5.2 * 0.8375 = 4.355
        expected_cost = 5.2 * (0.5 + 0.5 * 0.675)
        self.assertAlmostEqual(vm.get_current_hourly_cost(), expected_cost, places=2)
    
    def test_get_average_cost_24h(self):
        """Test 24-hour average cost calculation"""
        vm = EnhancedVirtualMachine(1, 4, 16, "aws")
        
        # No cost history
        self.assertIsNone(vm.get_average_cost_24h())
        
        # Add recent cost data (within 24 hours)
        now = datetime.now()
        for i in range(12):  # 12 hours of data
            cost_point = CostDataPoint(
                timestamp=now - timedelta(hours=i),
                hourly_cost=3.0 + i * 0.1,
                cumulative_cost=50.0
            )
            vm.add_cost_data_point(cost_point)
        
        # Add old cost data (beyond 24 hours)
        for i in range(5):
            cost_point = CostDataPoint(
                timestamp=now - timedelta(hours=25 + i),
                hourly_cost=10.0,  # Much higher cost
                cumulative_cost=50.0
            )
            vm.add_cost_data_point(cost_point)
        
        # Should only consider recent data
        avg_cost = vm.get_average_cost_24h()
        self.assertIsNotNone(avg_cost)
        # Should be around 3.55 (average of 3.0 to 4.1)
        self.assertGreater(avg_cost, 3.0)
        self.assertLess(avg_cost, 5.0)
    
    def test_meets_performance_requirements(self):
        """Test performance requirements checking"""
        vm = EnhancedVirtualMachine(1, 4, 16, "aws")
        vm.performance_score = 85.0
        
        requirements = PerformanceRequirements(
            min_cpu_performance=80.0,
            max_latency_ms=50,
            availability_requirement=0.99,
            throughput_requirement=400
        )
        
        # Without metrics - should return False
        self.assertFalse(vm.meets_performance_requirements(requirements))
        
        # With good metrics - should return True
        good_metrics = PerformanceMetrics(
            cpu_utilization=70.0,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=30.0,  # Below 50ms requirement
            throughput=500.0,  # Above 400 requirement
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(good_metrics)
        self.assertTrue(vm.meets_performance_requirements(requirements))
        
        # With poor latency - should return False
        poor_latency_metrics = PerformanceMetrics(
            cpu_utilization=70.0,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=100.0,  # Above 50ms requirement
            throughput=500.0,
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(poor_latency_metrics)
        self.assertFalse(vm.meets_performance_requirements(requirements))
        
        # With low throughput - should return False
        low_throughput_metrics = PerformanceMetrics(
            cpu_utilization=70.0,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=30.0,
            throughput=300.0,  # Below 400 requirement
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(low_throughput_metrics)
        self.assertFalse(vm.meets_performance_requirements(requirements))
    
    def test_meets_compliance_requirements(self):
        """Test compliance requirements checking"""
        vm = EnhancedVirtualMachine(1, 4, 16, "aws", region="us-east-1")
        vm.compliance_certifications = ["SOC2", "GDPR"]
        
        # Requirements met
        good_requirements = ComplianceRequirements(
            data_residency_regions=["us-east-1", "us-west-2"],
            compliance_standards=["SOC2"]
        )
        self.assertTrue(vm.meets_compliance_requirements(good_requirements))
        
        # Wrong region
        wrong_region_requirements = ComplianceRequirements(
            data_residency_regions=["eu-west-1"]
        )
        self.assertFalse(vm.meets_compliance_requirements(wrong_region_requirements))
        
        # Missing compliance standard
        missing_standard_requirements = ComplianceRequirements(
            compliance_standards=["HIPAA"]
        )
        self.assertFalse(vm.meets_compliance_requirements(missing_standard_requirements))
    
    def test_can_accommodate_enhanced(self):
        """Test enhanced accommodation checking"""
        from environment import CloudProvider
        provider = CloudProvider("aws", 0.5, 0.2)  # $0.5/vCPU, $0.2/GB
        vm = EnhancedVirtualMachine(1, 4, 16, provider, region="us-east-1")
        vm.performance_score = 85.0
        vm.compliance_certifications = ["SOC2"]
        
        # Update with good performance metrics
        good_metrics = PerformanceMetrics(
            cpu_utilization=70.0,
            memory_utilization=60.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=30.0,
            throughput=500.0,
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(good_metrics)
        
        # Create workload that should be accommodated
        workload = EnhancedWorkload(
            workload_id=1,
            cpu_required=2,  # Within capacity
            memory_required_gb=8,  # Within capacity
            cost_constraints=CostConstraints(max_hourly_cost=6.0),  # Within budget (base cost is 5.2)
            performance_requirements=PerformanceRequirements(
                min_cpu_performance=80.0,
                max_latency_ms=50,
                availability_requirement=0.99
            ),
            compliance_requirements=ComplianceRequirements(
                data_residency_regions=["us-east-1"],
                compliance_standards=["SOC2"]
            )
        )
        
        self.assertTrue(vm.can_accommodate_enhanced(workload))
        
        # Test with workload exceeding capacity
        large_workload = EnhancedWorkload(
            workload_id=2,
            cpu_required=8,  # Exceeds capacity
            memory_required_gb=32  # Exceeds capacity
        )
        self.assertFalse(vm.can_accommodate_enhanced(large_workload))
        
        # Test with workload exceeding cost budget
        expensive_workload = EnhancedWorkload(
            workload_id=3,
            cpu_required=2,
            memory_required_gb=8,
            cost_constraints=CostConstraints(max_hourly_cost=3.0)  # Below current cost (5.2)
        )
        self.assertFalse(vm.can_accommodate_enhanced(expensive_workload))


if __name__ == '__main__':
    unittest.main()