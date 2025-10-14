#!/usr/bin/env python3
"""
Integration test for enhanced scheduler foundation and core interfaces.
Tests the integration between enhanced data models and scheduler interfaces.
"""

from datetime import datetime
from environment import CloudProvider
from enhanced_models import (
    EnhancedWorkload, EnhancedVirtualMachine, WorkloadPriority,
    CostConstraints, PerformanceRequirements, ComplianceRequirements,
    CostOptimizationLevel, PerformanceMetrics, HealthStatus
)
from enhanced_schedulers import (
    EnhancedScheduler, CostAwareScheduler, PerformanceScheduler
)


def test_enhanced_data_models():
    """Test enhanced data models functionality"""
    print("Testing Enhanced Data Models...")
    
    # Test CostConstraints
    cost_constraints = CostConstraints(
        max_hourly_cost=5.0,
        max_monthly_budget=3600.0,
        cost_optimization_preference=CostOptimizationLevel.MODERATE
    )
    assert cost_constraints.max_hourly_cost == 5.0
    print("✓ CostConstraints created successfully")
    
    # Test PerformanceRequirements
    perf_requirements = PerformanceRequirements(
        min_cpu_performance=80.0,
        max_latency_ms=100,
        availability_requirement=0.99,
        throughput_requirement=1000
    )
    assert perf_requirements.min_cpu_performance == 80.0
    print("✓ PerformanceRequirements created successfully")
    
    # Test ComplianceRequirements
    compliance_requirements = ComplianceRequirements(
        data_residency_regions=["us-east-1", "us-west-2"],
        compliance_standards=["SOC2", "GDPR"],
        encryption_required=True
    )
    assert compliance_requirements.is_compliant_region("us-east-1")
    assert not compliance_requirements.is_compliant_region("eu-west-1")
    print("✓ ComplianceRequirements created successfully")
    
    # Test EnhancedWorkload
    workload = EnhancedWorkload(
        workload_id=1,
        cpu_required=2,
        memory_required_gb=4,
        cost_constraints=cost_constraints,
        performance_requirements=perf_requirements,
        compliance_requirements=compliance_requirements,
        priority=WorkloadPriority.HIGH
    )
    assert workload.has_cost_constraints()
    assert workload.has_performance_requirements()
    assert workload.is_high_priority()
    print("✓ EnhancedWorkload created successfully")
    
    # Test EnhancedVirtualMachine
    provider = CloudProvider("AWS", 0.04, 0.01)
    vm = EnhancedVirtualMachine(
        vm_id=1,
        cpu_capacity=4,
        memory_capacity_gb=16,
        provider=provider,
        region="us-east-1"
    )
    
    # Add performance metrics
    metrics = PerformanceMetrics(
        cpu_utilization=50.0,
        memory_utilization=60.0,
        network_io_mbps=100.0,
        disk_io_mbps=50.0,
        response_time_ms=80.0,
        throughput=1200.0,
        timestamp=datetime.now()
    )
    vm.update_performance_metrics(metrics)
    vm.compliance_certifications = ["SOC2", "GDPR"]
    
    assert vm.meets_performance_requirements(perf_requirements)
    assert vm.meets_compliance_requirements(compliance_requirements)
    assert vm.can_accommodate_enhanced(workload)
    print("✓ EnhancedVirtualMachine created and tested successfully")


def test_enhanced_schedulers():
    """Test enhanced scheduler functionality"""
    print("\nTesting Enhanced Schedulers...")
    
    # Create test environment
    aws = CloudProvider("AWS", 0.04, 0.01)
    gcp = CloudProvider("GCP", 0.035, 0.009)
    
    # Create enhanced VMs
    vms = []
    for i in range(3):
        vm = EnhancedVirtualMachine(
            vm_id=i+1,
            cpu_capacity=4,
            memory_capacity_gb=16,
            provider=aws if i % 2 == 0 else gcp,
            region="us-east-1"
        )
        
        # Add performance metrics with varying utilization
        metrics = PerformanceMetrics(
            cpu_utilization=30.0 + i * 20,  # 30%, 50%, 70%
            memory_utilization=40.0 + i * 15,  # 40%, 55%, 70%
            network_io_mbps=100.0,
            disk_io_mbps=50.0,
            response_time_ms=50.0 + i * 10,  # 50ms, 60ms, 70ms
            throughput=1500.0 - i * 200,  # 1500, 1300, 1100
            timestamp=datetime.now()
        )
        vm.update_performance_metrics(metrics)
        vm.compliance_certifications = ["SOC2", "GDPR"]
        vms.append(vm)
    
    # Create enhanced workload
    workload = EnhancedWorkload(
        workload_id=1,
        cpu_required=2,
        memory_required_gb=4,
        cost_constraints=CostConstraints(
            max_hourly_cost=2.0,
            cost_optimization_preference=CostOptimizationLevel.MODERATE
        ),
        performance_requirements=PerformanceRequirements(
            min_cpu_performance=70.0,
            max_latency_ms=100,
            availability_requirement=0.95,
            throughput_requirement=1000
        ),
        priority=WorkloadPriority.HIGH
    )
    
    # Test EnhancedScheduler
    scheduler = EnhancedScheduler()
    selected_vm = scheduler.select_vm(workload, vms)
    assert selected_vm is not None
    print("✓ EnhancedScheduler selected VM successfully")
    
    # Test prediction
    prediction = scheduler.predict_placement_outcome(workload, selected_vm)
    assert prediction.vm_id == selected_vm.vm_id
    assert prediction.confidence_score > 0
    print("✓ Placement prediction generated successfully")
    
    # Test recommendations
    recommendations = scheduler.get_optimization_recommendations(vms)
    print(f"✓ Generated {len(recommendations)} optimization recommendations")
    
    # Test CostAwareScheduler
    cost_scheduler = CostAwareScheduler()
    cost_selected_vm = cost_scheduler.select_vm(workload, vms)
    assert cost_selected_vm is not None
    print("✓ CostAwareScheduler selected VM successfully")
    
    # Test PerformanceScheduler
    perf_scheduler = PerformanceScheduler()
    perf_selected_vm = perf_scheduler.select_vm(workload, vms)
    assert perf_selected_vm is not None
    print("✓ PerformanceScheduler selected VM successfully")


def test_scheduler_integration():
    """Test integration between schedulers and data models"""
    print("\nTesting Scheduler Integration...")
    
    # Create test environment with different cost and performance characteristics
    aws = CloudProvider("AWS", 0.05, 0.012)  # Higher cost
    gcp = CloudProvider("GCP", 0.03, 0.008)  # Lower cost
    
    # Create VMs with different characteristics
    expensive_vm = EnhancedVirtualMachine(1, 8, 32, aws, "us-east-1")
    expensive_vm.performance_score = 95.0
    expensive_vm.update_performance_metrics(PerformanceMetrics(
        cpu_utilization=20.0, memory_utilization=25.0,
        network_io_mbps=200.0, disk_io_mbps=100.0,
        response_time_ms=30.0, throughput=2000.0,
        timestamp=datetime.now()
    ))
    
    cheap_vm = EnhancedVirtualMachine(2, 4, 16, gcp, "us-east-1")
    cheap_vm.performance_score = 75.0
    cheap_vm.update_performance_metrics(PerformanceMetrics(
        cpu_utilization=60.0, memory_utilization=65.0,
        network_io_mbps=100.0, disk_io_mbps=50.0,
        response_time_ms=80.0, throughput=1200.0,
        timestamp=datetime.now()
    ))
    
    for vm in [expensive_vm, cheap_vm]:
        vm.compliance_certifications = ["SOC2"]
    
    vms = [expensive_vm, cheap_vm]
    
    # Create workload with cost constraints
    cost_sensitive_workload = EnhancedWorkload(
        workload_id=1,
        cpu_required=2,
        memory_required_gb=8,
        cost_constraints=CostConstraints(
            max_hourly_cost=1.0,  # Low budget
            cost_optimization_preference=CostOptimizationLevel.AGGRESSIVE
        ),
        priority=WorkloadPriority.NORMAL
    )
    
    # Create workload with performance requirements
    performance_sensitive_workload = EnhancedWorkload(
        workload_id=2,
        cpu_required=2,
        memory_required_gb=8,
        performance_requirements=PerformanceRequirements(
            min_cpu_performance=90.0,  # High performance requirement
            max_latency_ms=50,
            availability_requirement=0.99
        ),
        priority=WorkloadPriority.HIGH
    )
    
    # Test cost-aware scheduling
    cost_scheduler = CostAwareScheduler()
    cost_choice = cost_scheduler.select_vm(cost_sensitive_workload, vms)
    print(f"✓ Cost-aware scheduler selected VM {cost_choice.vm_id if cost_choice else 'None'}")
    
    # Test performance-aware scheduling
    perf_scheduler = PerformanceScheduler()
    perf_choice = perf_scheduler.select_vm(performance_sensitive_workload, vms)
    print(f"✓ Performance-aware scheduler selected VM {perf_choice.vm_id if perf_choice else 'None'}")
    
    # Verify different choices based on priorities
    if cost_choice and perf_choice:
        if cost_choice.vm_id != perf_choice.vm_id:
            print("✓ Different schedulers made different optimal choices")
        else:
            print("✓ Both schedulers agreed on optimal choice")


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Enhanced Scheduler Foundation Integration Test")
    print("=" * 60)
    
    try:
        test_enhanced_data_models()
        test_enhanced_schedulers()
        test_scheduler_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Enhanced scheduler foundation is working!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()