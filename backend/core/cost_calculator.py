"""
Real-time Cloud Cost Calculator
Calculates estimated costs for AWS, GCP, and Azure based on user requirements
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"


@dataclass
class WorkloadRequirements:
    """User's workload requirements"""
    # Compute
    compute_cores: int = 4
    memory_gb: int = 16
    
    # Storage
    storage_tb: float = 1.0
    backup_retention_days: int = 30
    
    # Database
    database_type: str = "postgresql"
    database_size_gb: int = 100
    database_connections: int = 100
    
    # Network
    monthly_data_transfer_gb: int = 1000
    
    # High Availability
    availability_target: float = 99.9  # percentage
    multi_region: bool = False


@dataclass
class CostEstimate:
    """Cost estimate for a cloud provider"""
    provider: str
    total_monthly_cost: float
    breakdown: Dict[str, float]
    instance_recommendations: Dict[str, str]
    confidence_level: str  # "high", "medium", "low"


class CloudCostCalculator:
    """
    Calculates estimated cloud costs based on workload requirements
    Uses simplified pricing models for quick estimates
    """
    
    def __init__(self):
        # Simplified pricing (USD per month)
        # These are approximate averages as of 2024
        
        self.aws_pricing = {
            "compute_per_core": 50,  # ~t3.medium equivalent
            "memory_per_gb": 5,
            "storage_per_tb": 100,  # S3 + EBS average
            "database_per_gb": 0.50,  # RDS pricing
            "network_per_gb": 0.09,
            "backup_per_gb": 0.05,
            "ha_multiplier": 1.5,  # For multi-AZ
        }
        
        self.gcp_pricing = {
            "compute_per_core": 45,  # ~10% cheaper
            "memory_per_gb": 4.5,
            "storage_per_tb": 90,
            "database_per_gb": 0.45,
            "network_per_gb": 0.08,
            "backup_per_gb": 0.045,
            "ha_multiplier": 1.4,
        }
        
        self.azure_pricing = {
            "compute_per_core": 48,
            "memory_per_gb": 4.8,
            "storage_per_tb": 95,
            "database_per_gb": 0.48,
            "network_per_gb": 0.087,
            "backup_per_gb": 0.048,
            "ha_multiplier": 1.45,
        }
    
    def calculate_all_providers(self, requirements: WorkloadRequirements) -> Dict[str, CostEstimate]:
        """Calculate costs for all three cloud providers"""
        
        return {
            "aws": self.calculate_aws_cost(requirements),
            "gcp": self.calculate_gcp_cost(requirements),
            "azure": self.calculate_azure_cost(requirements),
        }
    
    def calculate_aws_cost(self, req: WorkloadRequirements) -> CostEstimate:
        """Calculate AWS costs"""
        
        # Compute (EC2)
        compute_cost = (req.compute_cores * self.aws_pricing["compute_per_core"] +
                       req.memory_gb * self.aws_pricing["memory_per_gb"])
        
        # Storage (S3 + EBS)
        storage_cost = req.storage_tb * self.aws_pricing["storage_per_tb"]
        
        # Database (RDS)
        database_cost = req.database_size_gb * self.aws_pricing["database_per_gb"]
        
        # Network (Data Transfer)
        network_cost = req.monthly_data_transfer_gb * self.aws_pricing["network_per_gb"]
        
        # Backup
        backup_size_gb = req.database_size_gb + (req.storage_tb * 1000 * 0.3)  # 30% of storage
        backup_cost = backup_size_gb * self.aws_pricing["backup_per_gb"]
        
        # High Availability
        if req.availability_target >= 99.9 or req.multi_region:
            compute_cost *= self.aws_pricing["ha_multiplier"]
            database_cost *= self.aws_pricing["ha_multiplier"]
        
        # Other services (monitoring, logging, etc.) - 10% of total
        subtotal = compute_cost + storage_cost + database_cost + network_cost + backup_cost
        other_cost = subtotal * 0.10
        
        total = subtotal + other_cost
        
        # Instance recommendations
        instance_type = self._recommend_aws_instance(req.compute_cores, req.memory_gb)
        db_instance = self._recommend_aws_db_instance(req.database_size_gb, req.database_connections)
        
        return CostEstimate(
            provider="aws",
            total_monthly_cost=round(total, 2),
            breakdown={
                "compute": round(compute_cost, 2),
                "storage": round(storage_cost, 2),
                "database": round(database_cost, 2),
                "network": round(network_cost, 2),
                "backup": round(backup_cost, 2),
                "other": round(other_cost, 2),
            },
            instance_recommendations={
                "compute": instance_type,
                "database": db_instance,
            },
            confidence_level="high"
        )
    
    def calculate_gcp_cost(self, req: WorkloadRequirements) -> CostEstimate:
        """Calculate GCP costs"""
        
        # Compute (Compute Engine)
        compute_cost = (req.compute_cores * self.gcp_pricing["compute_per_core"] +
                       req.memory_gb * self.gcp_pricing["memory_per_gb"])
        
        # Storage (Cloud Storage + Persistent Disk)
        storage_cost = req.storage_tb * self.gcp_pricing["storage_per_tb"]
        
        # Database (Cloud SQL)
        database_cost = req.database_size_gb * self.gcp_pricing["database_per_gb"]
        
        # Network
        network_cost = req.monthly_data_transfer_gb * self.gcp_pricing["network_per_gb"]
        
        # Backup
        backup_size_gb = req.database_size_gb + (req.storage_tb * 1000 * 0.3)
        backup_cost = backup_size_gb * self.gcp_pricing["backup_per_gb"]
        
        # High Availability
        if req.availability_target >= 99.9 or req.multi_region:
            compute_cost *= self.gcp_pricing["ha_multiplier"]
            database_cost *= self.gcp_pricing["ha_multiplier"]
        
        # Other services
        subtotal = compute_cost + storage_cost + database_cost + network_cost + backup_cost
        other_cost = subtotal * 0.10
        
        total = subtotal + other_cost
        
        # Instance recommendations
        instance_type = self._recommend_gcp_instance(req.compute_cores, req.memory_gb)
        db_instance = self._recommend_gcp_db_instance(req.database_size_gb, req.database_connections)
        
        return CostEstimate(
            provider="gcp",
            total_monthly_cost=round(total, 2),
            breakdown={
                "compute": round(compute_cost, 2),
                "storage": round(storage_cost, 2),
                "database": round(database_cost, 2),
                "network": round(network_cost, 2),
                "backup": round(backup_cost, 2),
                "other": round(other_cost, 2),
            },
            instance_recommendations={
                "compute": instance_typ