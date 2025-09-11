# main.py
import pandas as pd
from environment import CloudProvider, VirtualMachine, Workload
from schedulers import RandomScheduler, LowestCostScheduler, RoundRobinScheduler



def load_workloads_from_csv(filepath):
    """Loads workload data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        workloads = []
        for index, row in df.iterrows():
            workloads.append(Workload(
                workload_id=int(row['workload_id']),
                cpu_required=int(row['cpu_required']),
                memory_required_gb=int(row['memory_required_gb'])
            ))
        print(f"Successfully loaded {len(workloads)} workloads from {filepath}")
        return workloads
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return []



def run_simulation(scheduler, vms, workloads, log_filepath):
    """Runs the simulation and logs the system state at each step."""
    print(f"\n--- Running Simulation with {scheduler.__class__.__name__} ---")

    simulation_logs = []

    for timestamp, workload in enumerate(workloads, 1):
        selected_vm = scheduler.select_vm(workload, vms)

        if selected_vm:
            selected_vm.assign_workload(workload)
            status = "SUCCESS"
        else:
            status = "FAILURE"

        # --- LOGGING ---
        # Capture the state of the system AFTER the scheduling attempt
        total_cpu_used = sum(vm.cpu_used for vm in vms)
        total_mem_used = sum(vm.memory_used_gb for vm in vms)
        total_cpu_capacity = sum(vm.cpu_capacity for vm in vms)
        total_mem_capacity = sum(vm.memory_capacity_gb for vm in vms)

        log_entry = {
            "timestamp": timestamp,
            "workload_id": workload.id,
            "status": status,
            "total_cpu_used": total_cpu_used,
            "total_cpu_capacity": total_cpu_capacity,
            "percent_cpu_used": (total_cpu_used / total_cpu_capacity) * 100,
            "total_mem_used_gb": total_mem_used,
            "total_mem_capacity_gb": total_mem_capacity,
            "percent_mem_used": (total_mem_used / total_mem_capacity) * 100,
        }
        simulation_logs.append(log_entry)

        # Optional: Print status for clarity during run
        print(f"T={timestamp}: Workload {workload.id} -> {status}")

    # --- SAVE LOGS TO FILE ---
    log_df = pd.DataFrame(simulation_logs)
    log_df.to_csv(log_filepath, index=False)

    print(f"--- Simulation Complete ---")
    print(f"Performance log saved to {log_filepath}\n")


if __name__ == "__main__":
    # Function to set up a fresh environment for each simulation run
    def setup_environment():
        aws = CloudProvider(name="AWS", cpu_cost=0.04, memory_cost_gb=0.01)
        gcp = CloudProvider(name="GCP", cpu_cost=0.035, memory_cost_gb=0.009)
        azure = CloudProvider(name="Azure", cpu_cost=0.042, memory_cost_gb=0.011)

        vms = [
            VirtualMachine(vm_id=1, cpu_capacity=4, memory_capacity_gb=16, provider=aws),
            VirtualMachine(vm_id=2, cpu_capacity=8, memory_capacity_gb=32, provider=gcp),
            VirtualMachine(vm_id=3, cpu_capacity=4, memory_capacity_gb=16, provider=azure),
            VirtualMachine(vm_id=4, cpu_capacity=2, memory_capacity_gb=8, provider=gcp)
        ]
        return vms

    # Load workloads from the CSV file
    workloads_to_schedule = load_workloads_from_csv('workload_trace.csv')

    if workloads_to_schedule:
        # --- Run for Random Scheduler ---
        vms_for_random = setup_environment()
        random_scheduler = RandomScheduler()
        run_simulation(random_scheduler, vms_for_random, workloads_to_schedule, 'log_random.csv')

        # --- Run for Lowest Cost Scheduler ---
        vms_for_lowest_cost = setup_environment()
        lowest_cost_scheduler = LowestCostScheduler()
        run_simulation(lowest_cost_scheduler, vms_for_lowest_cost, workloads_to_schedule, 'log_lowest_cost.csv')

        # --- Run for Round Robin Scheduler ---
        vms_for_round_robin = setup_environment()
        round_robin_scheduler = RoundRobinScheduler()
        run_simulation(round_robin_scheduler, vms_for_round_robin, workloads_to_schedule, 'log_round_robin.csv')