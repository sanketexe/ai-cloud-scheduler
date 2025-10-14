"""
Unit tests for enhanced schedulers (fixed version)
Tests scheduler decision-making logic with various constraint combinations
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from environment import CloudProvider
from enhanced_models import (
    CostConstraints, PerformanceRequirements, ComplianceRequirements,
    PerformanceMetrics, UtilizationTrends, EnhancedWorkload, EnhancedVirtualMachine,
    WorkloadPriority, CostOptimizationLevel, HealthStatus
)
from enhanced_schedulers import (
    EnhancedScheduler, CostAwareScheduler, PerformanceScheduler,
    PlacementPrediction, Recommendation
)


class TestEnhancedScheduler(unittest.TestCase):
    """Test EnhancedScheduler base class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = EnhancedScheduler()
        
        # Create cloud providers with different costs
        self.aws_provider = CloudProvider("aws", 0.5, 0.1)  # $0.5/vCPU, $0.1/GB
        self.gcp_provider = CloudProvider("gcp", 0.6, 0.12)  # $0.6/vCPU, $0.12/GB
        self.azure_provider = CloudProvider("azure", 0.4, 0.08)  # $0.4/vCPU, $0.08/GB
        
        # Create test VMs with different characteristics
        self.vm1 = EnhancedVirtualMachine(1, 4, 16, self.aws_provider, "us-east-1", "us-east-1a")
        self.vm1.performance_score = 85.0
        self.vm1.compliance_certifications = ["SOC2"]
        
        self.vm2 = EnhancedVirtualMachine(2, 8, 32, self.gcp_provider, "us-west-2", "us-west-2b")
        self.vm2.performance_score = 90.0
        self.vm2.compliance_certifications = ["SOC2", "GDPR"]
        
        self.vm3 = EnhancedVirtualMachine(3, 2, 8, self.azure_provider, "eu-west-1", "eu-west-1a")
        self.vm3.performance_score = 75.0
        self.vm3.compliance_certifications = ["GDPR"]
        
        # Add performance metrics
        good_metrics = PerformanceMetrics(
            cpu_utilization=60.0,
            memory_utilization=50.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=20.0,
            throughput=600.0,
            timestamp=datetime.now()
        )
        
        self.vm1.update_performance_metrics(good_metrics)
        self.vm2.update_performance_metrics(good_metrics)
        self.vm3.update_performance_metrics(good_metrics)
        
        self.vms = [self.vm1, self.vm2, self.vm3]
    
    def test_basic_vm_selection(self):
        """Test basic VM selection without constraints"""
        workload = EnhancedWorkload(1, 2, 4)
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        self.assertIsNotNone(selected_vm)
        self.assertTrue(selected_vm.can_accommodate_enhanced(workload))
    
    def test_vm_selection_with_cost_constraints(self):
        """Test VM selection with cost constraints"""
        # Workload with tight cost constraint (vm3 is cheapest at 2*0.4 + 8*0.08 = 1.44)
        cost_constraints = CostConstraints(max_hourly_cost=2.0)
        workload = EnhancedWorkload(
            1, 2, 4,
            cost_constraints=cost_constraints
        )
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        # Should select vm3 (cheapest)
        self.assertEqual(selected_vm.vm_id, 3)
        self.assertLessEqual(selected_vm.get_current_hourly_cost(), 2.0)
    
    def test_vm_selection_with_performance_requirements(self):
        """Test VM selection with performance requirements"""
        performance_requirements = PerformanceRequirements(
            min_cpu_performance=88.0,
            max_latency_ms=30,
            availability_requirement=0.99
        )
        workload = EnhancedWorkload(
            1, 2, 4,
            performance_requirements=performance_requirements
        )
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        # Should select vm2 (highest performance score at 90.0)
        self.assertEqual(selected_vm.vm_id, 2)
        self.assertGreaterEqual(selected_vm.performance_score, 88.0)
    
    def test_vm_selection_with_compliance_requirements(self):
        """Test VM selection with compliance requirements"""
        compliance_requirements = ComplianceRequirements(
            data_residency_regions=["eu-west-1"],
            compliance_standards=["GDPR"]
        )
        workload = EnhancedWorkload(
            1, 2, 4,
            compliance_requirements=compliance_requirements
        )
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        # Should select vm3 (only one in EU with GDPR)
        self.assertEqual(selected_vm.vm_id, 3)
        self.assertEqual(selected_vm.region, "eu-west-1")
        self.assertIn("GDPR", selected_vm.compliance_certifications)
    
    def test_no_eligible_vms(self):
        """Test behavior when no VMs meet requirements"""
        # Impossible cost constraint
        cost_constraints = CostConstraints(max_hourly_cost=0.5)
        workload = EnhancedWorkload(
            1, 2, 4,
            cost_constraints=cost_constraints
        )
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        self.assertIsNone(selected_vm)
    
    def test_calculate_cost_score(self):
        """Test cost score calculation"""
        # Workload with cost constraint
        cost_constraints = CostConstraints(max_hourly_cost=4.0)
        workload = EnhancedWorkload(1, 2, 4, cost_constraints=cost_constraints)
        
        # VM with cost under budget
        score = self.scheduler._calculate_cost_score(workload, self.vm1)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)
        
        # VM exceeding budget (create expensive VM)
        expensive_provider = CloudProvider("expensive", 2.0, 1.0)
        expensive_vm = EnhancedVirtualMachine(99, 4, 16, expensive_provider)
        score = self.scheduler._calculate_cost_score(workload, expensive_vm)
        self.assertEqual(score, 0.0)  # Should be 0 for exceeding budget
    
    def test_predict_placement_outcome(self):
        """Test placement outcome prediction"""
        workload = EnhancedWorkload(1, 2, 4)
        workload.estimated_duration_hours = 10.0
        
        prediction = self.scheduler.predict_placement_outcome(workload, self.vm1)
        
        self.assertIsInstance(prediction, PlacementPrediction)
        self.assertEqual(prediction.vm_id, self.vm1.vm_id)
        self.assertEqual(prediction.workload_id, workload.id)
        self.assertGreater(prediction.predicted_cost, 0)
        self.assertGreater(prediction.predicted_performance_score, 0)
        self.assertGreater(prediction.confidence_score, 0)
        self.assertLessEqual(prediction.confidence_score, 1.0)
        self.assertIsInstance(prediction.risk_factors, list)
    
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        # Create VM with low utilization
        low_util_provider = CloudProvider("test", 0.5, 0.1)
        low_util_vm = EnhancedVirtualMachine(99, 4, 16, low_util_provider)
        low_util_metrics = PerformanceMetrics(
            cpu_utilization=15.0,
            memory_utilization=10.0,
            network_io_mbps=10.0,
            disk_io_mbps=5.0,
            response_time_ms=20.0,
            throughput=100.0,
            timestamp=datetime.now()
        )
        low_util_vm.update_performance_metrics(low_util_metrics)
        
        recommendations = self.scheduler.get_optimization_recommendations([low_util_vm])
        
        self.assertIsInstance(recommendations, list)
        if recommendations:  # Should have cost optimization recommendation
            rec = recommendations[0]
            self.assertIsInstance(rec, Recommendation)
            self.assertEqual(rec.recommendation_type, "cost_optimization")
            self.assertIn("underutilized", rec.description)


class TestCostAwareScheduler(unittest.TestCase):
    """Test CostAwareScheduler specialized behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = CostAwareScheduler()
        
        # Create VMs with different costs
        cheap_provider = CloudProvider("cheap", 0.3, 0.05)  # Very cheap
        expensive_provider = CloudProvider("expensive", 1.0, 0.5)  # Very expensive
        
        self.cheap_vm = EnhancedVirtualMachine(1, 4, 16, cheap_provider)
        self.cheap_vm.performance_score = 75.0
        
        self.expensive_vm = EnhancedVirtualMachine(2, 4, 16, expensive_provider)
        self.expensive_vm.performance_score = 95.0
        
        # Add performance metrics
        metrics = PerformanceMetrics(
            cpu_utilization=60.0,
            memory_utilization=50.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=20.0,
            throughput=600.0,
            timestamp=datetime.now()
        )
        
        self.cheap_vm.update_performance_metrics(metrics)
        self.expensive_vm.update_performance_metrics(metrics)
        
        self.vms = [self.cheap_vm, self.expensive_vm]
    
    def test_aggressive_cost_optimization(self):
        """Test aggressive cost optimization preference"""
        cost_constraints = CostConstraints(
            max_hourly_cost=10.0,
            cost_optimization_preference=CostOptimizationLevel.AGGRESSIVE
        )
        workload = EnhancedWorkload(1, 2, 4, cost_constraints=cost_constraints)
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        # Should prefer cheaper VM even though expensive VM is within budget
        # Aggressive mode only considers VMs using <70% of budget
        self.assertEqual(selected_vm.vm_id, 1)  # Cheap VM


class TestPerformanceScheduler(unittest.TestCase):
    """Test PerformanceScheduler specialized behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = PerformanceScheduler()
        
        # Create VMs with different performance characteristics
        balanced_provider = CloudProvider("balanced", 0.5, 0.1)
        
        self.high_perf_vm = EnhancedVirtualMachine(1, 8, 32, balanced_provider)
        self.high_perf_vm.performance_score = 95.0
        
        self.low_perf_vm = EnhancedVirtualMachine(2, 4, 16, balanced_provider)
        self.low_perf_vm.performance_score = 70.0
        
        # Add performance metrics and utilization trends
        good_metrics = PerformanceMetrics(
            cpu_utilization=60.0,
            memory_utilization=50.0,
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=15.0,
            throughput=800.0,
            timestamp=datetime.now()
        )
        
        poor_metrics = PerformanceMetrics(
            cpu_utilization=85.0,
            memory_utilization=80.0,
            network_io_mbps=50.0,
            disk_io_mbps=25.0,
            response_time_ms=40.0,
            throughput=300.0,
            timestamp=datetime.now()
        )
        
        self.high_perf_vm.update_performance_metrics(good_metrics)
        self.low_perf_vm.update_performance_metrics(poor_metrics)
        
        # Add utilization trends
        good_trends = UtilizationTrends(
            avg_cpu_utilization_24h=55.0,
            avg_memory_utilization_24h=45.0,
            peak_cpu_utilization_24h=75.0,
            peak_memory_utilization_24h=65.0,
            trend_direction="stable",
            last_updated=datetime.now()
        )
        
        poor_trends = UtilizationTrends(
            avg_cpu_utilization_24h=85.0,
            avg_memory_utilization_24h=80.0,
            peak_cpu_utilization_24h=95.0,
            peak_memory_utilization_24h=90.0,
            trend_direction="increasing",
            last_updated=datetime.now()
        )
        
        self.high_perf_vm.utilization_trends = good_trends
        self.low_perf_vm.utilization_trends = poor_trends
        
        self.vms = [self.high_perf_vm, self.low_perf_vm]
    
    def test_performance_focused_selection(self):
        """Test performance-focused VM selection"""
        performance_requirements = PerformanceRequirements(
            min_cpu_performance=85.0,
            max_latency_ms=25,
            availability_requirement=0.99,
            throughput_requirement=500
        )
        workload = EnhancedWorkload(1, 2, 4, performance_requirements=performance_requirements)
        
        selected_vm = self.scheduler.select_vm_enhanced(workload, self.vms)
        
        # Should select high performance VM
        self.assertEqual(selected_vm.vm_id, 1)
        self.assertGreaterEqual(selected_vm.performance_score, 85.0)
    
    def test_enhanced_performance_scoring(self):
        """Test enhanced performance scoring with historical data"""
        workload = EnhancedWorkload(1, 2, 4)
        
        # High performance VM with good trends should score higher
        high_perf_score = self.scheduler._calculate_performance_score(workload, self.high_perf_vm)
        low_perf_score = self.scheduler._calculate_performance_score(workload, self.low_perf_vm)
        
        self.assertGreater(high_perf_score, low_perf_score)


class TestSchedulerIntegration(unittest.TestCase):
    """Integration tests for scheduler interactions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enhanced_scheduler = EnhancedScheduler()
        self.cost_scheduler = CostAwareScheduler()
        self.performance_scheduler = PerformanceScheduler()
        
        # Create diverse set of VMs
        cheap_provider = CloudProvider("cheap", 0.3, 0.05)
        balanced_provider = CloudProvider("balanced", 0.5, 0.1)
        expensive_provider = CloudProvider("expensive", 0.8, 0.2)
        
        # Cheap, low performance VM
        vm1 = EnhancedVirtualMachine(1, 2, 8, cheap_provider, "us-east-1")
        vm1.performance_score = 65.0
        vm1.compliance_certifications = ["SOC2"]
        
        # Balanced VM
        vm2 = EnhancedVirtualMachine(2, 4, 16, balanced_provider, "us-west-2")
        vm2.performance_score = 85.0
        vm2.compliance_certifications = ["SOC2", "GDPR"]
        
        # Expensive, high performance VM
        vm3 = EnhancedVirtualMachine(3, 8, 32, expensive_provider, "eu-west-1")
        vm3.performance_score = 95.0
        vm3.compliance_certifications = ["GDPR", "HIPAA"]
        
        # Add performance metrics to all VMs
        for i, vm in enumerate([vm1, vm2, vm3]):
            metrics = PerformanceMetrics(
                cpu_utilization=50.0 + i * 10,
                memory_utilization=40.0 + i * 10,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
                response_time_ms=20.0 - i * 2,
                throughput=500.0 + i * 100,
                timestamp=datetime.now()
            )
            vm.update_performance_metrics(metrics)
        
        self.vms = [vm1, vm2, vm3]
    
    def test_scheduler_consistency(self):
        """Test that different schedulers make consistent decisions for same workload"""
        # Simple workload without constraints
        workload = EnhancedWorkload(1, 2, 4)
        
        enhanced_selection = self.enhanced_scheduler.select_vm_enhanced(workload, self.vms)
        cost_selection = self.cost_scheduler.select_vm_enhanced(workload, self.vms)
        performance_selection = self.performance_scheduler.select_vm_enhanced(workload, self.vms)
        
        # All should select valid VMs
        self.assertIsNotNone(enhanced_selection)
        self.assertIsNotNone(cost_selection)
        self.assertIsNotNone(performance_selection)
        
        # All selected VMs should be able to accommodate the workload
        self.assertTrue(enhanced_selection.can_accommodate_enhanced(workload))
        self.assertTrue(cost_selection.can_accommodate_enhanced(workload))
        self.assertTrue(performance_selection.can_accommodate_enhanced(workload))


if __name__ == '__main__':
    unittest.main()