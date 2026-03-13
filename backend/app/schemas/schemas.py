from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

# --- EC2Instance Schemas ---
class EC2InstanceBase(BaseModel):
    instance_id: str
    instance_type: str
    region: str
    state: str
    launch_time: Optional[datetime] = None
    cpu_cores: int
    memory_gb: float
    cost_per_hour: float

class EC2InstanceCreate(EC2InstanceBase):
    pass

class EC2Instance(EC2InstanceBase):
    id: int

    class Config:
        from_attributes = True

# --- EBSVolume Schemas ---
class EBSVolumeBase(BaseModel):
    volume_id: str
    size_gb: int
    volume_type: str
    state: str
    iops: Optional[int] = None
    region: str
    attached_instance_id: Optional[str] = None

class EBSVolumeCreate(EBSVolumeBase):
    pass

class EBSVolume(EBSVolumeBase):
    id: int

    class Config:
        from_attributes = True

# --- OptimizationRecommendation Schemas ---
class OptimizationRecommendationBase(BaseModel):
    resource_id: str
    resource_type: str
    recommendation_type: str
    description: Optional[str] = None
    current_cost: Optional[float] = None
    projected_cost: Optional[float] = None
    potential_savings: float
    confidence_score: Optional[float] = None
    status: str = "pending"

class OptimizationRecommendationCreate(OptimizationRecommendationBase):
    ec2_instance_id: Optional[int] = None

class OptimizationRecommendation(OptimizationRecommendationBase):
    id: int
    generated_at: datetime
    applied_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- CostHistory Schemas ---
class CostHistoryBase(BaseModel):
    resource_id: str
    service_type: str
    date: datetime
    cost: float
    usage_amount: Optional[float] = None
    unit: Optional[str] = None
    currency: str = "USD"

class CostHistoryCreate(CostHistoryBase):
    ec2_instance_id: Optional[int] = None

class CostHistory(CostHistoryBase):
    id: int

    class Config:
        from_attributes = True

# --- Response Models ---
class EC2InstanceResponse(BaseModel):
    id: str
    instance_type: str
    region: str
    state: str
    
    class Config:
        from_attributes = True

class EBSVolumeResponse(BaseModel):
    id: str
    size: int
    volume_type: str
    state: str
    region: str
    
    class Config:
        from_attributes = True

class OptimizationRecommendationResponse(BaseModel):
    id: int
    resource_id: str
    resource_type: str
    recommendation_type: str
    description: Optional[str]
    potential_savings: float
    status: str
    generated_at: datetime
    
    class Config:
        from_attributes = True

# --- EC2 Instance Extended Schema ---
# To be used if we need to return instance data with recommendations
