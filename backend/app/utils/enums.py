"""
Shared enums for the FinOps platform
"""

from enum import Enum


class SeverityLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of performance metrics"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
