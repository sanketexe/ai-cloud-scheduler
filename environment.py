# environment.py

class CloudProvider:
    """Represents a cloud provider with specific costs."""
    def __init__(self, name, cpu_cost, memory_cost_gb):
        self.name = name
        self.cpu_cost = cpu_cost  # Cost per vCPU per hour
        self.memory_cost_gb = memory_cost_gb  # Cost per GB RAM per hour

class Workload:
    """Represents a task to be scheduled."""
    def __init__(self, workload_id, cpu_required, memory_required_gb):
        self.id = workload_id
        self.cpu_required = cpu_required
        self.memory_required_gb = memory_required_gb

class VirtualMachine:
    """Represents a VM instance on a cloud provider."""
    def __init__(self, vm_id, cpu_capacity, memory_capacity_gb, provider):
        self.id = vm_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity_gb = memory_capacity_gb
        self.provider = provider
        # Track current usage
        self.cpu_used = 0
        self.memory_used_gb = 0

    def can_accommodate(self, workload):
        """Checks if the VM has enough resources for a workload."""
        cpu_fits = (self.cpu_used + workload.cpu_required) <= self.cpu_capacity
        mem_fits = (self.memory_used_gb + workload.memory_required_gb) <= self.memory_capacity_gb
        return cpu_fits and mem_fits

    def assign_workload(self, workload):
        """Assigns a workload to this VM, updating its resource usage."""
        if self.can_accommodate(workload):
            self.cpu_used += workload.cpu_required
            self.memory_used_gb += workload.memory_required_gb
            return True
        return False

    @property
    def cost(self):
        """Calculates the current running cost of this VM based on its capacity."""
        return (self.cpu_capacity * self.provider.cpu_cost) + \
               (self.memory_capacity_gb * self.provider.memory_cost_gb)