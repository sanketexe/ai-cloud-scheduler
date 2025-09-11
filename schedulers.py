# schedulers.py
import random

class BaseScheduler:
    """Base class for all schedulers."""
    def select_vm(self, workload, vms):
        # This method must be implemented by subclasses
        raise NotImplementedError

class RandomScheduler(BaseScheduler):
    """Selects a random VM that can accommodate the workload."""
    def select_vm(self, workload, vms):
        # Filter for VMs that have enough resources
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None # No suitable VM found
        return random.choice(eligible_vms)

class LowestCostScheduler(BaseScheduler):
    """Selects the cheapest VM that can accommodate the workload."""
    def select_vm(self, workload, vms):
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None
        # Sort eligible VMs by their cost and return the first one
        eligible_vms.sort(key=lambda vm: vm.cost)
        return eligible_vms[0]

class RoundRobinScheduler(BaseScheduler):
    """Selects VMs in a cyclical order."""
    def __init__(self):
        self.last_vm_index = -1

    def select_vm(self, workload, vms):
        for i in range(len(vms)):
            # Cycle through VMs starting from the last used index
            self.last_vm_index = (self.last_vm_index + 1) % len(vms)
            vm = vms[self.last_vm_index]
            if vm.can_accommodate(workload):
                return vm
        return None # No VM in the entire list could accommodate the workload