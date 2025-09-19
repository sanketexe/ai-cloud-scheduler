# schedulers.py
import random

class BaseScheduler:
    """Base class for all schedulers."""
    def select_vm(self, workload, vms):
        raise NotImplementedError

class RandomScheduler(BaseScheduler):
    """Selects a random VM that can accommodate the workload."""
    def select_vm(self, workload, vms):
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None
        return random.choice(eligible_vms)

class LowestCostScheduler(BaseScheduler):
    """Selects the cheapest VM that can accommodate the workload."""
    def select_vm(self, workload, vms):
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None
        eligible_vms.sort(key=lambda vm: vm.cost)
        return eligible_vms[0]

class RoundRobinScheduler(BaseScheduler):
    """Selects VMs in a cyclical order."""
    def __init__(self):
        self.current_index = 0

    def select_vm(self, workload, vms):
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None
        
        # Find next eligible VM starting from current index
        for _ in range(len(vms)):
            vm = vms[self.current_index % len(vms)]
            self.current_index += 1
            if vm.can_accommodate(workload):
                return vm
        return None