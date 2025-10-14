# enhanced_models.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from environment import Workload, VirtualMachine


class WorkloadPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CostOptimizationLevel(Enum):
    NONE = "none"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CostConstraints:
    """Defines cost constraints for workload placement"""
    max_hourly_cost: float
    max_monthly_budget: Optional[float] = None
    cost_optimization_preference: CostOptimizationLevel = CostOptimizationLevel.MODERATE
    currency: str = "USD"
    
    def __post_init__(self):
        if self.max_hourly_cost <= 0:
            raise ValueError("max_hourly_cost must be positive")


@dataclass
class PerformanceRequirements:
    """Defines performance requirements for workload placement"""
    min_cpu_performance: float  # Minimum CPU performance score (0-100)
    max_latency_ms: int  # Maximum acceptable latency in milliseconds
    availability_requirement: float  # Required availability (0.0-1.0)
    throughput_requirement: Optional[int] = None  # Required throughput (requests/second)
    min_memory_bandwidth: Optional[float] = None  # Minimum memory bandwidth (GB/s)
    
    def __post_init__(self):
        if not 0 <= self.min_cpu_performance <= 100:
            raise ValueError("min_cpu_performance must be between 0 and 100")
        if not 0.0 <= self.availability_requirement <= 1.0:
            raise ValueError("availability_requirement must be between 0.0 and 1.0")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")


@dataclass
class ComplianceRequirements:
    """Defines compliance requirements for workload placement"""
    data_residency_regions: List[str] = field(default_factory=list)
    compliance_standards: List[str] = field(default_factory=list)  # e.g., ["SOC2", "GDPR", "HIPAA"]
    encryption_required: bool = True
    audit_logging_required: bool = True
    network_isolation_required: bool = False
    
    def is_compliant_region(self, region: str) -> bool:
        """Check if a region meets data residency requirements"""
        if not self.data_residency_regions:
            return True  # No restrictions
        return region in self.data_residency_regions


@dataclass
class CostDataPoint:
    """Represents a single cost data point"""
    timestamp: datetime
    hourly_cost: float
    cumulative_cost: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"


@dataclass
class PerformanceMetrics:
    """Current performance metrics for a resource"""
    cpu_utilization: float  # Percentage (0-100)
    memory_utilization: float  # Percentage (0-100)
    network_io_mbps: float  # Network I/O in Mbps
    disk_io_mbps: float  # Disk I/O in Mbps
    response_time_ms: float  # Average response time
    throughput: float  # Requests per second
    timestamp: datetime
    
    def __post_init__(self):
        if not 0 <= self.cpu_utilization <= 100:
            raise ValueError("cpu_utilization must be between 0 and 100")
        if not 0 <= self.memory_utilization <= 100:
            raise ValueError("memory_utilization must be between 0 and 100")


@dataclass
class UtilizationTrends:
    """Historical utilization trends"""
    avg_cpu_utilization_24h: float
    avg_memory_utilization_24h: float
    peak_cpu_utilization_24h: float
    peak_memory_utilization_24h: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    last_updated: datetime


class EnhancedWorkload(Workload):
    """Extended workload class with cost and performance requirements"""
    
    def __init__(self, workload_id, cpu_required, memory_required_gb,
                 cost_constraints: Optional[CostConstraints] = None,
                 performance_requirements: Optional[PerformanceRequirements] = None,
                 compliance_requirements: Optional[ComplianceRequirements] = None,
                 priority: WorkloadPriority = WorkloadPriority.NORMAL,
                 tags: Optional[Dict[str, str]] = None):
        super().__init__(workload_id, cpu_required, memory_required_gb)
        self.cost_constraints = cost_constraints
        self.performance_requirements = performance_requirements
        self.compliance_requirements = compliance_requirements or ComplianceRequirements()
        self.priority = priority
        self.tags = tags or {}
        self.created_at = datetime.now()
        self.estimated_duration_hours: Optional[float] = None
    
    def has_cost_constraints(self) -> bool:
        """Check if workload has cost constraints"""
        return self.cost_constraints is not None
    
    def has_performance_requirements(self) -> bool:
        """Check if workload has performance requirements"""
        return self.performance_requirements is not None
    
    def get_max_acceptable_cost(self) -> Optional[float]:
        """Get maximum acceptable hourly cost"""
        return self.cost_constraints.max_hourly_cost if self.cost_constraints else None
    
    def is_high_priority(self) -> bool:
        """Check if workload is high or critical priority"""
        return self.priority in [WorkloadPriority.HIGH, WorkloadPriority.CRITICAL]


class EnhancedVirtualMachine(VirtualMachine):
    """Extended VM class with performance metrics and cost history"""
    
    def __init__(self, vm_id, cpu_capacity, memory_capacity_gb, provider,
                 region: str = "us-east-1",
                 availability_zone: str = "us-east-1a"):
        super().__init__(vm_id, cpu_capacity, memory_capacity_gb, provider)
        self.region = region
        self.availability_zone = availability_zone
        self.current_performance_metrics: Optional[PerformanceMetrics] = None
        self.cost_history: List[CostDataPoint] = []
        self.health_status = HealthStatus.HEALTHY
        self.utilization_trends: Optional[UtilizationTrends] = None
        self.created_at = datetime.now()
        self.last_health_check = datetime.now()
        self.compliance_certifications: List[str] = []  # e.g., ["SOC2", "GDPR"]
        self.performance_score: float = 85.0  # Default performance score (0-100)
        
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """Update current performance metrics"""
        self.current_performance_metrics = metrics
        self.last_health_check = datetime.now()
        
        # Update health status based on metrics
        if metrics.cpu_utilization > 95 or metrics.memory_utilization > 95:
            self.health_status = HealthStatus.UNHEALTHY
        elif metrics.cpu_utilization > 90 or metrics.memory_utilization > 90:
            self.health_status = HealthStatus.WARNING
        else:
            self.health_status = HealthStatus.HEALTHY
    
    def add_cost_data_point(self, cost_point: CostDataPoint):
        """Add a cost data point to history"""
        self.cost_history.append(cost_point)
        # Keep only last 30 days of cost history
        if len(self.cost_history) > 720:  # 30 days * 24 hours
            self.cost_history = self.cost_history[-720:]
    
    def get_current_hourly_cost(self) -> float:
        """Calculate current hourly cost including utilization"""
        base_cost = super().cost
        
        # Apply utilization-based cost adjustment if metrics available
        if self.current_performance_metrics:
            utilization_factor = (
                self.current_performance_metrics.cpu_utilization + 
                self.current_performance_metrics.memory_utilization
            ) / 200.0  # Average utilization (0-1)
            return base_cost * (0.5 + 0.5 * utilization_factor)  # 50% base + 50% utilization-based
        
        return base_cost
    
    def get_average_cost_24h(self) -> Optional[float]:
        """Get average hourly cost over last 24 hours"""
        if not self.cost_history:
            return None
        
        recent_costs = [
            point.hourly_cost for point in self.cost_history
            if (datetime.now() - point.timestamp).total_seconds() <= 86400  # 24 hours
        ]
        
        return sum(recent_costs) / len(recent_costs) if recent_costs else None
    
    def meets_performance_requirements(self, requirements: PerformanceRequirements) -> bool:
        """Check if VM meets performance requirements"""
        if not self.current_performance_metrics:
            return False  # Cannot verify without metrics
        
        metrics = self.current_performance_metrics
        
        # Check CPU performance
        if self.performance_score < requirements.min_cpu_performance:
            return False
        
        # Check latency
        if metrics.response_time_ms > requirements.max_latency_ms:
            return False
        
        # Check throughput if specified
        if (requirements.throughput_requirement and 
            metrics.throughput < requirements.throughput_requirement):
            return False
        
        return True
    
    def meets_compliance_requirements(self, requirements: ComplianceRequirements) -> bool:
        """Check if VM meets compliance requirements"""
        # Check data residency
        if not requirements.is_compliant_region(self.region):
            return False
        
        # Check compliance standards
        for standard in requirements.compliance_standards:
            if standard not in self.compliance_certifications:
                return False
        
        return True
    
    def can_accommodate_enhanced(self, workload: EnhancedWorkload) -> bool:
        """Enhanced accommodation check including cost and performance"""
        # Basic capacity check
        if not super().can_accommodate(workload):
            return False
        
        # Cost constraint check
        if workload.has_cost_constraints():
            current_cost = self.get_current_hourly_cost()
            if current_cost > workload.get_max_acceptable_cost():
                return False
        
        # Performance requirements check
        if workload.has_performance_requirements():
            if not self.meets_performance_requirements(workload.performance_requirements):
                return False
        
        # Compliance requirements check
        if not self.meets_compliance_requirements(workload.compliance_requirements):
            return False
        
        return True