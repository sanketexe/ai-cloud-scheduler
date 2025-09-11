# main.py
from environment import CloudProvider, VirtualMachine, Workload
from schedulers import RandomScheduler, LowestCostScheduler, RoundRobinScheduler

def run_simulation(scheduler, vms, workloads):
    """Runs the simulation and prints scheduling decisions."""
    print(f"\n--- Running Simulation with {scheduler.__class__.__name__} ---")
    total_cost = 0

    for workload in workloads:
        selected_vm = scheduler.select_vm(workload, vms)

        if selected_vm:
            selected_vm.assign_workload(workload)
            cost = (workload.cpu_required * selected_vm.provider.cpu_cost) + \
                   (workload.memory_required_gb * selected_vm.provider.memory_cost_gb)
            total_cost += cost

            print(f"SUCCESS: Workload {workload.id} ({workload.cpu_required} vCPU, {workload.memory_required_gb}GB RAM) -> "
                  f"VM {selected_vm.id} on {selected_vm.provider.name}. "
                  f"Cost for this workload: ${cost:.4f}")
        else:
            print(f"FAILURE: Could not schedule Workload {workload.id} "
                  f"({workload.cpu_required} vCPU, {workload.memory_required_gb}GB RAM). No suitable VM found.")

    print(f"--- Simulation Complete ---")
    print(f"Total simulated cost for workloads: ${total_cost:.2f}\n")

if __name__ == "__main__":
    # 1. Define Cloud Providers
    aws_provider = CloudProvider(name="AWS", cpu_cost=0.04, memory_cost_gb=0.01)
    gcp_provider = CloudProvider(name="GCP", cpu_cost=0.035, memory_cost_gb=0.009)
    azure_provider = CloudProvider(name="Azure", cpu_cost=0.042, memory_cost_gb=0.011) # More expensive

    # 2. Setup Virtual Machines across providers
    vms = [
        VirtualMachine(vm_id=1, cpu_capacity=4, memory_capacity_gb=16, provider=aws_provider),
        VirtualMachine(vm_id=2, cpu_capacity=8, memory_capacity_gb=32, provider=gcp_provider),
        VirtualMachine(vm_id=3, cpu_capacity=4, memory_capacity_gb=16, provider=azure_provider),
        VirtualMachine(vm_id=4, cpu_capacity=2, memory_capacity_gb=8, provider=gcp_provider) # Small cheap VM
    ]

    # 3. Create a list of workloads to be scheduled (for now, it's hardcoded)
    workloads = [
        Workload(workload_id=101, cpu_required=2, memory_required_gb=4),
        Workload(workload_id=102, cpu_required=4, memory_required_gb=16),
        Workload(workload_id=103, cpu_required=1, memory_required_gb=2),
        Workload(workload_id=104, cpu_required=3, memory_required_gb=8),
        Workload(workload_id=105, cpu_required=8, memory_required_gb=30), # A large workload
        Workload(workload_id=106, cpu_required=2, memory_required_gb=2),
    ]

    # --- Run simulations with different schedulers ---
    # Note: We are re-using the same VM and workload lists.
    # For a proper comparison, they should be reset before each run,
    # but for this first phase, this is fine.

    run_simulation(RandomScheduler(), vms, workloads)
    run_simulation(LowestCostScheduler(), vms, workloads)
    run_simulation(RoundRobinScheduler(), vms, workloads)