# simulation_framework.py
"""
Simulation Framework for Cloud Intelligence Platform

This module provides comprehensive simulation capabilities for testing and validation:
- WorkloadGenerator: Creates realistic workload patterns and traces
- EnvironmentSimulator: Models cloud provider environments and constraints
- ScenarioEngine: Runs complex multi-variable simulation scenarios
- ValidationFramework: Compares simulation results with real-world data

Requirements addressed:
- 1.1: Workload pattern generation
- 1.4: Simulation framework for testing
- 6.1: Scenario execution and validation
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import random
import math

# Import existing modules
from environment import Workload, VirtualMachine, CloudProvider
from enhanced_models import EnhancedWorkload, CostConstraints, PerformanceRequirements


class PatternType(Enum):
    CONSTANT = "constant"
    PERIODIC = "periodic"
    BURSTY = "bursty"
    RANDOM_WALK = "random_walk"


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class SeasonalComponent:
    """Seasonal variation component for workload patterns"""
    period_hours: int  # Period in hours (24 for daily, 168 for weekly)
    amplitude: float   # Amplitude of variation (0-1)
    phase_offset: float = 0.0  # Phase offset in hours


@dataclass
class WorkloadPattern:
    """Defines patterns for generating realistic workload traces"""
    pattern_type: PatternType
    base_intensity: float  # Base number of workloads per hour
    variation_amplitude: float = 0.1  # Amplitude of variations (0-1)
    seasonal_components: List[SeasonalComponent] = field(default_factory=list)
    noise_level: float = 0.05  # Random noise level (0-1)
    
    def __post_init__(self):
        if self.base_intensity <= 0:
            raise ValueError("Base intensity must be positive")
        if not 0 <= self.variation_amplitude <= 1:
            raise ValueError("Variation amplitude must be between 0 and 1")
        if not 0 <= self.noise_level <= 1:
            raise ValueError("Noise level must be between 0 and 1")


@dataclass
class WorkloadTrace:
    """Generated sequence of workloads for simulation"""
    workloads: List[Workload]
    generation_parameters: WorkloadPattern
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_hours: int = 0
    total_workloads: int = 0
    
    def __post_init__(self):
        self.total_workloads = len(self.workloads)


@dataclass
class EnvironmentConstraints:
    """Constraints for simulated cloud environment"""
    max_vms_per_provider: Dict[str, int] = field(default_factory=dict)
    network_latency_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failure_probability: float = 0.001  # Probability of VM failure per hour
    spot_price_volatility: float = 0.1  # Volatility of spot pricing
    compliance_regions: List[str] = field(default_factory=list)


@dataclass
class SimulatedEnvironment:
    """Virtual cloud environment for simulation"""
    providers: List[CloudProvider]
    virtual_machines: List[VirtualMachine]
    constraints: EnvironmentConstraints
    current_time: datetime = field(default_factory=datetime.now)
    
    def get_available_vms(self) -> List[VirtualMachine]:
        """Get currently available VMs"""
        return [vm for vm in self.virtual_machines if hasattr(vm, 'is_available') and vm.is_available]


@dataclass
class SimulationConfig:
    """Configuration for simulation run"""
    scenario_name: str
    workload_pattern: WorkloadPattern
    duration_hours: int
    scheduler_type: str
    environment_constraints: Dict[str, Any] = field(default_factory=dict)
    validation_baseline: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.duration_hours <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class SchedulerMetrics:
    """Performance metrics for scheduler evaluation"""
    success_rate: float
    average_placement_time: float
    resource_efficiency: float
    load_balancing_score: float
    cost_optimization_score: float
    total_placements: int
    failed_placements: int


@dataclass
class SimulationResults:
    """Comprehensive results from simulation run"""
    simulation_id: str
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_workloads: int
    successful_placements: int
    failed_placements: int
    total_cost: float
    average_utilization: float
    performance_score: float
    scheduler_metrics: Optional[SchedulerMetrics] = None
    detailed_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate placement success rate"""
        if self.total_workloads == 0:
            return 0.0
        return self.successful_placements / self.total_workloads


class WorkloadGenerator:
    """Generates realistic workload patterns and traces"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WorkloadGenerator")
        self.random = random.Random()
    
    def generate_trace(self, pattern: WorkloadPattern, duration_hours: int, 
                      scale_factor: float = 1.0) -> WorkloadTrace:
        """Generate workload trace based on pattern"""
        self.logger.info(f"Generating workload trace: {pattern.pattern_type.value}, "
                        f"duration: {duration_hours}h, scale: {scale_factor}")
        
        workloads = []
        current_time = datetime.now()
        
        for hour in range(duration_hours):
            # Calculate base intensity for this hour
            intensity = self._calculate_intensity(pattern, hour)
            intensity *= scale_factor
            
            # Generate workloads for this hour
            num_workloads = max(0, int(intensity + self.random.gauss(0, intensity * pattern.noise_level)))
            
            for i in range(num_workloads):
                workload_id = len(workloads) + 1
                
                # Generate workload characteristics
                cpu_required = self._generate_cpu_requirement()
                memory_required = self._generate_memory_requirement()
                
                workload = Workload(workload_id, cpu_required, memory_required)
                workload.arrival_time = current_time + timedelta(
                    hours=hour, 
                    minutes=self.random.randint(0, 59),
                    seconds=self.random.randint(0, 59)
                )
                
                workloads.append(workload)
        
        trace = WorkloadTrace(
            workloads=workloads,
            generation_parameters=pattern,
            duration_hours=duration_hours,
            metadata={
                "scale_factor": scale_factor,
                "generated_at": datetime.now().isoformat(),
                "pattern_type": pattern.pattern_type.value
            }
        )
        
        self.logger.info(f"Generated {len(workloads)} workloads")
        return trace
    
    def _calculate_intensity(self, pattern: WorkloadPattern, hour: int) -> float:
        """Calculate workload intensity for given hour"""
        base = pattern.base_intensity
        
        if pattern.pattern_type == PatternType.CONSTANT:
            return base
        
        elif pattern.pattern_type == PatternType.PERIODIC:
            # Simple sine wave pattern
            period = 24  # 24-hour cycle
            phase = (hour % period) / period * 2 * math.pi
            variation = math.sin(phase) * pattern.variation_amplitude
            return base * (1 + variation)
        
        elif pattern.pattern_type == PatternType.BURSTY:
            # Random bursts
            if self.random.random() < 0.1:  # 10% chance of burst
                return base * (1 + pattern.variation_amplitude * 5)
            else:
                return base * 0.5
        
        elif pattern.pattern_type == PatternType.RANDOM_WALK:
            # Random walk around base intensity
            if not hasattr(self, '_current_intensity'):
                self._current_intensity = base
            
            change = self.random.gauss(0, base * pattern.variation_amplitude * 0.1)
            self._current_intensity = max(0, self._current_intensity + change)
            return self._current_intensity
        
        # Apply seasonal components
        for seasonal in pattern.seasonal_components:
            phase = ((hour + seasonal.phase_offset) % seasonal.period_hours) / seasonal.period_hours * 2 * math.pi
            seasonal_variation = math.sin(phase) * seasonal.amplitude
            base *= (1 + seasonal_variation)
        
        return base
    
    def _generate_cpu_requirement(self) -> int:
        """Generate realistic CPU requirement"""
        # Weighted distribution favoring smaller requirements
        weights = [0.4, 0.3, 0.2, 0.1]  # 1, 2, 4, 8 cores
        cpu_options = [1, 2, 4, 8]
        return self.random.choices(cpu_options, weights=weights)[0]
    
    def _generate_memory_requirement(self) -> int:
        """Generate realistic memory requirement"""
        # Weighted distribution favoring smaller requirements
        weights = [0.3, 0.3, 0.2, 0.15, 0.05]  # 1, 2, 4, 8, 16 GB
        memory_options = [1, 2, 4, 8, 16]
        return self.random.choices(memory_options, weights=weights)[0]
    
    def create_seasonal_pattern(self, base_intensity: float) -> WorkloadPattern:
        """Create pattern with daily and weekly seasonality"""
        daily_seasonal = SeasonalComponent(
            period_hours=24,
            amplitude=0.3,  # 30% variation
            phase_offset=8   # Peak at 4 PM (hour 16)
        )
        
        weekly_seasonal = SeasonalComponent(
            period_hours=168,  # 7 days * 24 hours
            amplitude=0.2,     # 20% variation
            phase_offset=0     # Peak on Monday
        )
        
        return WorkloadPattern(
            pattern_type=PatternType.PERIODIC,
            base_intensity=base_intensity,
            variation_amplitude=0.2,
            seasonal_components=[daily_seasonal, weekly_seasonal],
            noise_level=0.1
        )


class EnvironmentSimulator:
    """Simulates cloud provider environments and constraints"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnvironmentSimulator")
    
    def create_environment(self, providers: List[str], 
                          constraints: EnvironmentConstraints) -> SimulatedEnvironment:
        """Create simulated multi-cloud environment"""
        self.logger.info(f"Creating simulated environment with providers: {providers}")
        
        # Create cloud providers
        cloud_providers = []
        for provider_name in providers:
            if provider_name.lower() == "aws":
                provider = CloudProvider("AWS", cpu_cost=0.04, memory_cost_gb=0.01)
            elif provider_name.lower() == "gcp":
                provider = CloudProvider("GCP", cpu_cost=0.035, memory_cost_gb=0.009)
            elif provider_name.lower() == "azure":
                provider = CloudProvider("Azure", cpu_cost=0.042, memory_cost_gb=0.011)
            else:
                provider = CloudProvider(provider_name, cpu_cost=0.04, memory_cost_gb=0.01)
            
            cloud_providers.append(provider)
        
        # Create virtual machines
        virtual_machines = []
        vm_id = 1
        
        for provider in cloud_providers:
            max_vms = constraints.max_vms_per_provider.get(provider.name, 10)
            
            for i in range(max_vms):
                # Vary VM sizes
                if i % 3 == 0:
                    cpu_capacity, memory_capacity = 2, 8
                elif i % 3 == 1:
                    cpu_capacity, memory_capacity = 4, 16
                else:
                    cpu_capacity, memory_capacity = 8, 32
                
                vm = VirtualMachine(vm_id, cpu_capacity, memory_capacity, provider)
                vm.is_available = True
                vm.failure_probability = constraints.failure_probability
                
                virtual_machines.append(vm)
                vm_id += 1
        
        environment = SimulatedEnvironment(
            providers=cloud_providers,
            virtual_machines=virtual_machines,
            constraints=constraints
        )
        
        self.logger.info(f"Created environment with {len(virtual_machines)} VMs across {len(cloud_providers)} providers")
        return environment
    
    def simulate_failures(self, environment: SimulatedEnvironment, 
                         time_elapsed_hours: float) -> List[str]:
        """Simulate VM failures based on failure probability"""
        failed_vms = []
        
        for vm in environment.virtual_machines:
            if vm.is_available:
                # Calculate failure probability for time period
                failure_prob = 1 - (1 - vm.failure_probability) ** time_elapsed_hours
                
                if random.random() < failure_prob:
                    vm.is_available = False
                    failed_vms.append(vm.vm_id)
                    self.logger.warning(f"VM {vm.vm_id} failed")
        
        return failed_vms
    
    def simulate_recovery(self, environment: SimulatedEnvironment, 
                         recovery_time_hours: float = 1.0) -> List[str]:
        """Simulate VM recovery after failures"""
        recovered_vms = []
        
        for vm in environment.virtual_machines:
            if not vm.is_available:
                # Simple recovery model - VMs recover after recovery_time_hours
                if random.random() < 0.8:  # 80% chance of recovery
                    vm.is_available = True
                    recovered_vms.append(vm.vm_id)
                    self.logger.info(f"VM {vm.vm_id} recovered")
        
        return recovered_vms


class ScenarioEngine:
    """Runs complex multi-variable simulation scenarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ScenarioEngine")
        self.workload_generator = WorkloadGenerator()
        self.environment_simulator = EnvironmentSimulator()
    
    async def run_scenario(self, config: SimulationConfig, 
                          scheduler) -> SimulationResults:
        """Run complete simulation scenario"""
        simulation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"Starting simulation {simulation_id}: {config.scenario_name}")
        
        # Generate workload trace
        workload_trace = self.workload_generator.generate_trace(
            config.workload_pattern,
            config.duration_hours
        )
        
        # Create environment
        providers = config.environment_constraints.get('providers', ['aws', 'gcp', 'azure'])
        constraints = EnvironmentConstraints(
            max_vms_per_provider=config.environment_constraints.get('max_vms_per_provider', {}),
            failure_probability=config.environment_constraints.get('failure_probability', 0.001)
        )
        
        environment = self.environment_simulator.create_environment(providers, constraints)
        
        # Run simulation
        results = await self._execute_simulation(
            simulation_id, config.scenario_name, workload_trace, 
            environment, scheduler, start_time
        )
        
        self.logger.info(f"Completed simulation {simulation_id} in {(results.end_time - results.start_time).total_seconds():.2f}s")
        return results
    
    async def _execute_simulation(self, simulation_id: str, scenario_name: str,
                                 workload_trace: WorkloadTrace, 
                                 environment: SimulatedEnvironment,
                                 scheduler, start_time: datetime) -> SimulationResults:
        """Execute the actual simulation"""
        
        successful_placements = 0
        failed_placements = 0
        total_cost = 0.0
        detailed_logs = []
        
        # Sort workloads by arrival time
        workloads = sorted(workload_trace.workloads, 
                          key=lambda w: getattr(w, 'arrival_time', datetime.now()))
        
        # Simulate workload placement
        for workload in workloads:
            available_vms = environment.get_available_vms()
            
            if not available_vms:
                failed_placements += 1
                detailed_logs.append({
                    'workload_id': workload.id,
                    'status': 'failed',
                    'reason': 'no_available_vms',
                    'timestamp': getattr(workload, 'arrival_time', datetime.now()).isoformat()
                })
                continue
            
            # Try to place workload
            selected_vm = scheduler.select_vm(workload, available_vms)
            
            if selected_vm and selected_vm.assign_workload(workload):
                successful_placements += 1
                
                # Calculate cost
                workload_cost = (workload.cpu_required * selected_vm.provider.cpu_cost + 
                               workload.memory_required_gb * selected_vm.provider.memory_cost_gb)
                total_cost += workload_cost
                
                detailed_logs.append({
                    'workload_id': workload.id,
                    'vm_id': selected_vm.vm_id,
                    'provider': selected_vm.provider.name,
                    'status': 'success',
                    'cost': workload_cost,
                    'timestamp': getattr(workload, 'arrival_time', datetime.now()).isoformat()
                })
            else:
                failed_placements += 1
                detailed_logs.append({
                    'workload_id': workload.id,
                    'status': 'failed',
                    'reason': 'placement_failed',
                    'timestamp': getattr(workload, 'arrival_time', datetime.now()).isoformat()
                })
        
        # Calculate metrics
        total_workloads = len(workloads)
        
        # Calculate average utilization
        total_capacity = sum(vm.cpu_capacity + vm.memory_capacity_gb for vm in environment.virtual_machines)
        total_used = sum(vm.cpu_used + vm.memory_used_gb for vm in environment.virtual_machines)
        average_utilization = (total_used / total_capacity) if total_capacity > 0 else 0.0
        
        # Calculate performance score (simplified)
        performance_score = (successful_placements / total_workloads * 100) if total_workloads > 0 else 0.0
        
        # Create scheduler metrics
        scheduler_metrics = SchedulerMetrics(
            success_rate=successful_placements / total_workloads if total_workloads > 0 else 0.0,
            average_placement_time=0.1,  # Simplified
            resource_efficiency=average_utilization,
            load_balancing_score=0.8,  # Simplified
            cost_optimization_score=0.7,  # Simplified
            total_placements=successful_placements,
            failed_placements=failed_placements
        )
        
        return SimulationResults(
            simulation_id=simulation_id,
            scenario_name=scenario_name,
            start_time=start_time,
            end_time=datetime.now(),
            total_workloads=total_workloads,
            successful_placements=successful_placements,
            failed_placements=failed_placements,
            total_cost=total_cost,
            average_utilization=average_utilization,
            performance_score=performance_score,
            scheduler_metrics=scheduler_metrics,
            detailed_logs=detailed_logs
        )


class SimulationEngine:
    """Main simulation engine coordinating all simulation components"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimulationEngine")
        self.scenario_engine = ScenarioEngine()
        self.simulation_results: Dict[str, SimulationResults] = {}
    
    async def run_simulation(self, config: SimulationConfig) -> SimulationResults:
        """Run simulation with given configuration"""
        try:
            # Import scheduler dynamically based on type
            scheduler = self._get_scheduler(config.scheduler_type)
            
            # Run scenario
            results = await self.scenario_engine.run_scenario(config, scheduler)
            
            # Store results
            self.simulation_results[results.simulation_id] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running simulation: {e}")
            raise
    
    async def get_simulation_results(self, simulation_id: str) -> Optional[SimulationResults]:
        """Get simulation results by ID"""
        return self.simulation_results.get(simulation_id)
    
    async def list_simulations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List simulation runs"""
        simulations = list(self.simulation_results.values())
        
        # Sort by start time (newest first)
        simulations.sort(key=lambda x: x.start_time, reverse=True)
        
        # Apply pagination
        paginated = simulations[offset:offset + limit]
        
        return [
            {
                "simulation_id": sim.simulation_id,
                "scenario_name": sim.scenario_name,
                "start_time": sim.start_time.isoformat(),
                "end_time": sim.end_time.isoformat(),
                "total_workloads": sim.total_workloads,
                "success_rate": sim.success_rate,
                "total_cost": sim.total_cost,
                "performance_score": sim.performance_score
            }
            for sim in paginated
        ]
    
    def _get_scheduler(self, scheduler_type: str):
        """Get scheduler instance by type"""
        # Import schedulers
        try:
            from schedulers import (
                RandomScheduler, LowestCostScheduler, RoundRobinScheduler
            )
            from enhanced_schedulers import (
                IntelligentScheduler, CostAwareScheduler, PerformanceScheduler
            )
        except ImportError:
            # Fallback to basic schedulers
            from api import (
                RandomScheduler, LowestCostScheduler, RoundRobinScheduler,
                IntelligentScheduler, HybridScheduler
            )
        
        scheduler_map = {
            "random": RandomScheduler(),
            "lowest_cost": LowestCostScheduler(),
            "round_robin": RoundRobinScheduler(),
            "intelligent": IntelligentScheduler(),
            "hybrid": HybridScheduler() if 'HybridScheduler' in locals() else RandomScheduler()
        }
        
        if scheduler_type not in scheduler_map:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}, using random")
            return RandomScheduler()
        
        return scheduler_map[scheduler_type]
    
    def create_default_scenarios(self) -> List[SimulationConfig]:
        """Create default simulation scenarios for testing"""
        scenarios = []
        
        # Constant load scenario
        constant_pattern = WorkloadPattern(
            pattern_type=PatternType.CONSTANT,
            base_intensity=10.0,
            variation_amplitude=0.1,
            noise_level=0.05
        )
        
        scenarios.append(SimulationConfig(
            scenario_name="Constant Load Test",
            workload_pattern=constant_pattern,
            duration_hours=24,
            scheduler_type="intelligent",
            environment_constraints={
                'providers': ['aws', 'gcp'],
                'max_vms_per_provider': {'aws': 5, 'gcp': 5}
            }
        ))
        
        # Bursty load scenario
        bursty_pattern = WorkloadPattern(
            pattern_type=PatternType.BURSTY,
            base_intensity=5.0,
            variation_amplitude=0.5,
            noise_level=0.1
        )
        
        scenarios.append(SimulationConfig(
            scenario_name="Bursty Load Test",
            workload_pattern=bursty_pattern,
            duration_hours=12,
            scheduler_type="cost_aware",
            environment_constraints={
                'providers': ['aws', 'azure'],
                'max_vms_per_provider': {'aws': 8, 'azure': 8}
            }
        ))
        
        # Periodic load with seasonality
        generator = WorkloadGenerator()
        seasonal_pattern = generator.create_seasonal_pattern(base_intensity=15.0)
        
        scenarios.append(SimulationConfig(
            scenario_name="Seasonal Load Test",
            workload_pattern=seasonal_pattern,
            duration_hours=168,  # 1 week
            scheduler_type="performance",
            environment_constraints={
                'providers': ['aws', 'gcp', 'azure'],
                'max_vms_per_provider': {'aws': 10, 'gcp': 10, 'azure': 10}
            }
        ))
        
        return scenarios


# Example usage
async def main():
    """Example usage of the simulation framework"""
    
    # Initialize simulation engine
    engine = SimulationEngine()
    
    # Create a test scenario
    workload_pattern = WorkloadPattern(
        pattern_type=PatternType.PERIODIC,
        base_intensity=8.0,
        variation_amplitude=0.3,
        noise_level=0.1
    )
    
    config = SimulationConfig(
        scenario_name="Test Scenario",
        workload_pattern=workload_pattern,
        duration_hours=6,
        scheduler_type="intelligent",
        environment_constraints={
            'providers': ['aws', 'gcp'],
            'max_vms_per_provider': {'aws': 5, 'gcp': 5}
        }
    )
    
    # Run simulation
    results = await engine.run_simulation(config)
    
    print(f"Simulation completed: {results.simulation_id}")
    print(f"Total workloads: {results.total_workloads}")
    print(f"Success rate: {results.success_rate:.2%}")
    print(f"Total cost: ${results.total_cost:.2f}")
    print(f"Performance score: {results.performance_score:.1f}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())