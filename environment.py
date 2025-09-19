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
        self.vm_id = vm_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity_gb = memory_capacity_gb
        self.provider = provider
        self.cpu_used = 0
        self.memory_used_gb = 0

    def can_accommodate(self, workload):
        """Check if VM can accommodate the workload"""
        return (self.cpu_capacity - self.cpu_used >= workload.cpu_required and 
                self.memory_capacity_gb - self.memory_used_gb >= workload.memory_required_gb)

    def assign_workload(self, workload):
        """Assign workload to VM"""
        if self.can_accommodate(workload):
            self.cpu_used += workload.cpu_required
            self.memory_used_gb += workload.memory_required_gb
            return True
        return False

    @property
    def cost(self):
        """Calculate total cost for this VM"""
        return (self.cpu_capacity * self.provider.cpu_cost + 
                self.memory_capacity_gb * self.provider.memory_cost_gb)