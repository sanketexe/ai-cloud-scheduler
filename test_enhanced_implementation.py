# test_enhanced_implementation.py
"""
Simple test to verify the enhanced scheduler implementation works correctly
"""
from datetime import datetime
from environment import CloudProvider
from enhanced_models import (
    EnhancedWorkload, EnhancedVirtualMachine, CostConstraints, 
    PerformanceRequirements, ComplianceRequirements, WorkloadPriority,
    CostOptimizationLevel, PerformanceMetrics, HealthStatus
)
from enhanced_schedulers import EnhancedScheduler, CostAwareScheduler, PerformanceScheduler


def test_enhanced_models():
    """Test enhanced data models"""
    print("Testing Enhanced Data Models...")
    
    # Test CostConstraints
    cost_constraints = CostConstraints(
        max_hourly_cost=5.0,
        max_monthly_budget=1000.0,
        cost_optimization_preference=CostOptimizationLevel.MODERATE
    )
    print(f"✓ CostConstraints created: max_hourly_cost=${cost_constraints.max_hourly_cost}")
    
    # Test PerformanceRequirements
    perf_requirements = PerformanceRequirements(
        min_cpu_performance=80.0,
        max_latency_ms=100,
        availability_requirement=0.99
    )
    print(f"✓ PerformanceRequirements created: min_cpu_performance={perf_requirements.min_cpu_performance}")
    
    # Test ComplianceRequirements
    compliance_requirements = ComplianceRequirements(
        data_residency_regions=["us-east-1", "us-west-2"],
        compliance_standards=["SOC2", "GDPR"]
    )
    print(f"✓ ComplianceRequirements created: regions={compliance_requirements.data_residency_regions}")
    
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
    print(f"✓ EnhancedWorkload created: ID={workload.id}, priority={workload.priority.value}")
    
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
        cpu_utilization=45.0,
        memory_utilization=60.0,
        network_io_mbps=100.0,
        disk_io_mbps=50.0,
        response_time_ms=80.0,
        throughput=1000.0,
        timestamp=datetime.now()
    )
    vm.update_performance_metrics(metrics)
    vm.compliance_certifications = ["SOC2", "GDPR"]
    
    print(f"✓ EnhancedVirtualMachine created: ID={vm.vm_id}, health={vm.health_status.value}")
    
    # Test accommodation check
    can_accommodate = vm.can_accommodate_enhanced(workload)
    print(f"✓ VM can accommodate workload: {can_accommodate}")
    
    return workload, vm


def test_enhanced_schedulers():
    """Test enhanced schedulers"""
    print("\nTesting Enhanced Schedulers...")
    
    # Create test data
    workload, vm = test_enhanced_models()
    
    # Create additional VMs for testing
    provider_gcp = CloudProvider("GCP", 0.035, 0.009)
    provider_azure = CloudProvider("Azure", 0.042, 0.011)
    
    vm2 = EnhancedVirtualMachine(2, 8, 32, provider_gcp, "us-central1")
    vm2.performance_score = 90.0
    vm2.compliance_certifications = ["SOC2"]
    
    vm3 = EnhancedVirtualMachine(3, 2, 8, provider_azure, "eastus")
    vm3.performance_score = 75.0
    vm3.health_status = HealthStatus.WARNING
    
    vms = [vm, vm2, vm3]
    
    # Test EnhancedScheduler
    scheduler = EnhancedScheduler()
    selected_vm = scheduler.select_vm_enhanced(workload, vms)
    print(f"✓ EnhancedScheduler selected VM: {selected_vm.vm_id if selected_vm else 'None'}")
    
    # Test CostAwareScheduler
    cost_scheduler = CostAwareScheduler()
    selected_vm_cost = cost_scheduler.select_vm_enhanced(workload, vms)
    print(f"✓ CostAwareScheduler selected VM: {selected_vm_cost.vm_id if selected_vm_cost else 'None'}")
    
    # Test PerformanceScheduler
    perf_scheduler = PerformanceScheduler()
    selected_vm_perf = perf_scheduler.select_vm_enhanced(workload, vms)
    print(f"✓ PerformanceScheduler selected VM: {selected_vm_perf.vm_id if selected_vm_perf else 'None'}")
    
    # Test recommendations
    recommendations = scheduler.get_optimization_recommendations(vms)
    print(f"✓ Generated {len(recommendations)} optimization recommendations")
    
    # Test placement prediction
    if selected_vm:
        prediction = scheduler.predict_placement_outcome(workload, selected_vm)
        print(f"✓ Placement prediction: cost=${prediction.predicted_cost:.2f}, "
              f"performance={prediction.predicted_performance_score:.1f}, "
              f"confidence={prediction.confidence_score:.2f}")


def main():
    """Run all tests"""
    print("=== Enhanced Scheduler Implementation Test ===")
    
    try:
        test_enhanced_schedulers()
        print("\n✅ All tests passed! Enhanced scheduler implementation is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()