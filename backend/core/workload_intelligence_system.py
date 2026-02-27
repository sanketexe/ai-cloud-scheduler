"""
Workload Intelligence System for Optimal Cloud Placement

This module provides intelligent workload analysis and placement recommendations
across multiple cloud providers using machine learning and multi-criteria decision analysis.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = structlog.get_logger(__name__)


class WorkloadType(Enum):
    """Types of workloads for classification"""
    WEB_APPLICATION = "web_application"
    DATABASE = "database"
    ANALYTICS = "analytics"
    MACHINE_LEARNING = "machine_learning"
    BATCH_PROCESSING = "batch_processing"
    MICROSERVICES = "microservices"
    STORAGE_INTENSIVE = "storage_intensive"
    COMPUTE_INTENSIVE = "compute_intensive"


class ComplianceRequirement(Enum):
    """Compliance requirements for workload placement"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    FEDRAMP = "fedramp"
    DATA_RESIDENCY = "data_residency"


class PlacementCriteria(Enum):
    """Criteria for placement optimization"""
    COST = "cost"
    PERFORMANCE = "performance"
    LATENCY = "latency"
    COMPLIANCE = "compliance"
    AVAILABILITY = "availability"
    SCALABILITY = "scalability"


@dataclass
class ResourceRequirements:
    """Resource requirements specification"""
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    network_bandwidth_mbps: int
    gpu_count: int = 0
    gpu_memory_gb: int = 0
    iops_required: int = 1000
    storage_type: str = "ssd"

@dataclass
class PerformanceProfile:
    """Performance characteristics of a workload"""
    cpu_utilization_avg: float
    memory_utilization_avg: float
    network_io_pattern: str  # "burst", "steady", "periodic"
    storage_io_pattern: str  # "read_heavy", "write_heavy", "balanced"
    latency_sensitivity: str  # "low", "medium", "high"
    throughput_requirements: int  # requests per second
    availability_requirement: float  # 99.9, 99.99, etc.
    scaling_pattern: str  # "predictable", "unpredictable", "seasonal"


@dataclass
class CostSensitivity:
    """Cost sensitivity and budget constraints"""
    budget_limit: Optional[Decimal] = None
    cost_priority: float = 0.5  # 0.0 = performance first, 1.0 = cost first
    reserved_instance_preference: bool = True
    spot_instance_tolerance: bool = False
    budget_flexibility: float = 0.1  # 10% flexibility


@dataclass
class WorkloadSpec:
    """Complete workload specification"""
    workload_id: str
    name: str
    workload_type: WorkloadType
    resource_requirements: ResourceRequirements
    performance_profile: PerformanceProfile
    compliance_requirements: List[ComplianceRequirement]
    cost_sensitivity: CostSensitivity
    geographic_preferences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadProfile:
    """Analyzed workload profile with ML insights"""
    workload_spec: WorkloadSpec
    complexity_score: float
    resource_efficiency_score: float
    predicted_scaling_factor: float
    performance_bottlenecks: List[str]
    optimization_opportunities: List[str]
    risk_factors: List[str]
    confidence_score: float


@dataclass
class PlacementOption:
    """Cloud placement option with scoring"""
    provider: str
    region: str
    instance_types: List[str]
    estimated_monthly_cost: Decimal
    performance_score: float
    compliance_score: float
    latency_score: float
    availability_score: float
    overall_score: float
    confidence_interval: Tuple[float, float]
    reasoning: str
    limitations: List[str]


@dataclass
class PlacementRecommendation:
    """Complete placement recommendation"""
    workload_id: str
    recommended_option: PlacementOption
    alternative_options: List[PlacementOption]
    cost_comparison: Dict[str, Decimal]
    performance_comparison: Dict[str, float]
    risk_assessment: Dict[str, float]
    migration_complexity: str
    implementation_timeline: str
    confidence_score: float
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MigrationAnalysis:
    """Migration analysis between placements"""
    current_placement: PlacementOption
    target_placement: PlacementOption
    migration_cost: Decimal
    migration_time_hours: int
    downtime_hours: float
    data_transfer_gb: int
    risk_level: str
    break_even_months: int
    roi_percentage: float
    migration_steps: List[str]


class WorkloadAnalyzer:
    """Analyzes workload characteristics and patterns"""
    
    def __init__(self):
        self.ml_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for workload analysis"""
        # Performance prediction model
        self.ml_models['performance'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Cost prediction model
        self.ml_models['cost'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Scaling factor prediction model
        self.ml_models['scaling'] = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=8
        )
        
        # Initialize scalers
        for model_name in self.ml_models.keys():
            self.scalers[model_name] = StandardScaler()
    
    async def analyze_workload(self, workload_spec: WorkloadSpec) -> WorkloadProfile:
        """
        Analyze workload characteristics and generate profile.
        
        Args:
            workload_spec: Workload specification to analyze
            
        Returns:
            Analyzed workload profile with ML insights
        """
        logger.info("Analyzing workload", workload_id=workload_spec.workload_id)
        
        try:
            # Extract features for ML analysis
            features = self._extract_workload_features(workload_spec)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(workload_spec, features)
            
            # Calculate resource efficiency score
            efficiency_score = self._calculate_efficiency_score(workload_spec, features)
            
            # Predict scaling factor
            scaling_factor = await self._predict_scaling_factor(features)
            
            # Identify performance bottlenecks
            bottlenecks = self._identify_bottlenecks(workload_spec, features)
            
            # Find optimization opportunities
            opportunities = self._find_optimization_opportunities(workload_spec, features)
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(workload_spec, features)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(features, complexity_score)
            
            profile = WorkloadProfile(
                workload_spec=workload_spec,
                complexity_score=complexity_score,
                resource_efficiency_score=efficiency_score,
                predicted_scaling_factor=scaling_factor,
                performance_bottlenecks=bottlenecks,
                optimization_opportunities=opportunities,
                risk_factors=risk_factors,
                confidence_score=confidence_score
            )
            
            logger.info(
                "Workload analysis completed",
                workload_id=workload_spec.workload_id,
                complexity_score=complexity_score,
                confidence_score=confidence_score
            )
            
            return profile
            
        except Exception as e:
            logger.error("Workload analysis failed", workload_id=workload_spec.workload_id, error=str(e))
            raise
    
    def _extract_workload_features(self, workload_spec: WorkloadSpec) -> np.ndarray:
        """Extract numerical features from workload specification"""
        features = [
            # Resource features
            workload_spec.resource_requirements.cpu_cores,
            workload_spec.resource_requirements.memory_gb,
            workload_spec.resource_requirements.storage_gb,
            workload_spec.resource_requirements.network_bandwidth_mbps,
            workload_spec.resource_requirements.gpu_count,
            workload_spec.resource_requirements.iops_required,
            
            # Performance features
            workload_spec.performance_profile.cpu_utilization_avg,
            workload_spec.performance_profile.memory_utilization_avg,
            workload_spec.performance_profile.throughput_requirements,
            workload_spec.performance_profile.availability_requirement,
            
            # Cost features
            float(workload_spec.cost_sensitivity.cost_priority),
            float(workload_spec.cost_sensitivity.budget_flexibility),
            
            # Categorical features (encoded)
            self._encode_workload_type(workload_spec.workload_type),
            self._encode_latency_sensitivity(workload_spec.performance_profile.latency_sensitivity),
            self._encode_scaling_pattern(workload_spec.performance_profile.scaling_pattern),
            
            # Compliance features
            len(workload_spec.compliance_requirements),
            len(workload_spec.geographic_preferences)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_complexity_score(self, workload_spec: WorkloadSpec, features: np.ndarray) -> float:
        """Calculate workload complexity score (0.0 to 1.0)"""
        # Base complexity from resource requirements
        resource_complexity = (
            workload_spec.resource_requirements.cpu_cores / 64.0 +
            workload_spec.resource_requirements.memory_gb / 512.0 +
            workload_spec.resource_requirements.storage_gb / 10000.0 +
            workload_spec.resource_requirements.gpu_count / 8.0
        ) / 4.0
        
        # Compliance complexity
        compliance_complexity = len(workload_spec.compliance_requirements) / 7.0
        
        # Performance complexity
        perf_complexity = (
            (100 - workload_spec.performance_profile.availability_requirement) / 1.0 +
            workload_spec.performance_profile.throughput_requirements / 10000.0
        ) / 2.0
        
        # Combine complexities
        total_complexity = (resource_complexity + compliance_complexity + perf_complexity) / 3.0
        
        return min(max(total_complexity, 0.0), 1.0)
    
    def _calculate_efficiency_score(self, workload_spec: WorkloadSpec, features: np.ndarray) -> float:
        """Calculate resource efficiency score (0.0 to 1.0)"""
        # CPU efficiency
        cpu_efficiency = workload_spec.performance_profile.cpu_utilization_avg / 100.0
        
        # Memory efficiency
        memory_efficiency = workload_spec.performance_profile.memory_utilization_avg / 100.0
        
        # Storage efficiency (based on IOPS requirements vs capacity)
        storage_efficiency = min(
            workload_spec.resource_requirements.iops_required / 
            (workload_spec.resource_requirements.storage_gb * 3.0), 1.0
        )
        
        # Network efficiency (based on bandwidth requirements)
        network_efficiency = min(
            workload_spec.resource_requirements.network_bandwidth_mbps / 1000.0, 1.0
        )
        
        # Overall efficiency
        efficiency = (cpu_efficiency + memory_efficiency + storage_efficiency + network_efficiency) / 4.0
        
        return min(max(efficiency, 0.0), 1.0)
    
    async def _predict_scaling_factor(self, features: np.ndarray) -> float:
        """Predict scaling factor using ML model"""
        try:
            # Use mock prediction for now (in production, use trained model)
            # Base scaling factor on resource requirements and utilization
            base_factor = 1.0
            
            # Adjust based on workload characteristics
            cpu_cores = features[0][0]
            memory_gb = features[0][1]
            cpu_util = features[0][6]
            memory_util = features[0][7]
            
            # Higher utilization suggests more scaling potential
            utilization_factor = (cpu_util + memory_util) / 200.0
            
            # Larger workloads typically scale more
            size_factor = min((cpu_cores + memory_gb / 8.0) / 32.0, 2.0)
            
            scaling_factor = base_factor + utilization_factor + size_factor
            
            return min(max(scaling_factor, 0.5), 5.0)
            
        except Exception as e:
            logger.warning("Scaling factor prediction failed, using default", error=str(e))
            return 1.5
    
    def _identify_bottlenecks(self, workload_spec: WorkloadSpec, features: np.ndarray) -> List[str]:
        """Identify potential performance bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        if workload_spec.performance_profile.cpu_utilization_avg > 80:
            bottlenecks.append("High CPU utilization may cause performance degradation")
        
        # Memory bottleneck
        if workload_spec.performance_profile.memory_utilization_avg > 85:
            bottlenecks.append("High memory utilization may lead to swapping")
        
        # Storage bottleneck
        if (workload_spec.resource_requirements.iops_required > 
            workload_spec.resource_requirements.storage_gb * 3):
            bottlenecks.append("IOPS requirements may exceed storage capacity")
        
        # Network bottleneck
        if workload_spec.resource_requirements.network_bandwidth_mbps > 1000:
            bottlenecks.append("High network bandwidth requirements may limit placement options")
        
        # Latency bottleneck
        if (workload_spec.performance_profile.latency_sensitivity == "high" and
            workload_spec.performance_profile.throughput_requirements > 5000):
            bottlenecks.append("High throughput with low latency requirements may be challenging")
        
        return bottlenecks
    
    def _find_optimization_opportunities(self, workload_spec: WorkloadSpec, features: np.ndarray) -> List[str]:
        """Find optimization opportunities"""
        opportunities = []
        
        # CPU optimization
        if workload_spec.performance_profile.cpu_utilization_avg < 50:
            opportunities.append("CPU resources appear over-provisioned, consider rightsizing")
        
        # Memory optimization
        if workload_spec.performance_profile.memory_utilization_avg < 60:
            opportunities.append("Memory resources may be over-allocated")
        
        # Storage optimization
        if workload_spec.resource_requirements.storage_type == "ssd" and \
           workload_spec.performance_profile.storage_io_pattern == "read_heavy":
            opportunities.append("Consider using read-optimized storage tiers")
        
        # Cost optimization
        if workload_spec.cost_sensitivity.cost_priority > 0.7:
            opportunities.append("High cost sensitivity suggests reserved instances or spot pricing")
        
        # Scaling optimization
        if workload_spec.performance_profile.scaling_pattern == "predictable":
            opportunities.append("Predictable scaling pattern enables proactive resource management")
        
        return opportunities
    
    def _assess_risk_factors(self, workload_spec: WorkloadSpec, features: np.ndarray) -> List[str]:
        """Assess risk factors for workload placement"""
        risks = []
        
        # Availability risk
        if workload_spec.performance_profile.availability_requirement > 99.95:
            risks.append("High availability requirements increase complexity and cost")
        
        # Compliance risk
        if len(workload_spec.compliance_requirements) > 3:
            risks.append("Multiple compliance requirements limit placement options")
        
        # Performance risk
        if (workload_spec.performance_profile.latency_sensitivity == "high" and
            len(workload_spec.geographic_preferences) == 0):
            risks.append("High latency sensitivity without geographic preferences")
        
        # Scaling risk
        if workload_spec.performance_profile.scaling_pattern == "unpredictable":
            risks.append("Unpredictable scaling may lead to performance issues or cost overruns")
        
        # Resource risk
        if workload_spec.resource_requirements.gpu_count > 0:
            risks.append("GPU requirements may limit availability and increase costs")
        
        return risks
    
    def _calculate_confidence_score(self, features: np.ndarray, complexity_score: float) -> float:
        """Calculate confidence score for the analysis"""
        # Base confidence
        base_confidence = 0.8
        
        # Reduce confidence for complex workloads
        complexity_penalty = complexity_score * 0.2
        
        # Reduce confidence for incomplete data (mock check)
        data_completeness = 0.9  # Assume 90% complete data
        
        confidence = base_confidence - complexity_penalty + (data_completeness - 0.5) * 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _encode_workload_type(self, workload_type: WorkloadType) -> float:
        """Encode workload type as numerical value"""
        encoding = {
            WorkloadType.WEB_APPLICATION: 1.0,
            WorkloadType.DATABASE: 2.0,
            WorkloadType.ANALYTICS: 3.0,
            WorkloadType.MACHINE_LEARNING: 4.0,
            WorkloadType.BATCH_PROCESSING: 5.0,
            WorkloadType.MICROSERVICES: 6.0,
            WorkloadType.STORAGE_INTENSIVE: 7.0,
            WorkloadType.COMPUTE_INTENSIVE: 8.0
        }
        return encoding.get(workload_type, 1.0)
    
    def _encode_latency_sensitivity(self, sensitivity: str) -> float:
        """Encode latency sensitivity as numerical value"""
        encoding = {"low": 1.0, "medium": 2.0, "high": 3.0}
        return encoding.get(sensitivity, 2.0)
    
    def _encode_scaling_pattern(self, pattern: str) -> float:
        """Encode scaling pattern as numerical value"""
        encoding = {"predictable": 1.0, "seasonal": 2.0, "unpredictable": 3.0}
        return encoding.get(pattern, 2.0)


class PlacementOptimizer:
    """Optimizes workload placement using multi-criteria decision analysis"""
    
    def __init__(self, cost_calculator=None, performance_predictor=None):
        self.cost_calculator = cost_calculator
        self.performance_predictor = performance_predictor
        self.provider_capabilities = self._initialize_provider_capabilities()
        self.region_data = self._initialize_region_data()
    
    def _initialize_provider_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize provider capabilities and constraints"""
        return {
            'aws': {
                'regions': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
                'compliance': [ComplianceRequirement.SOC2, ComplianceRequirement.HIPAA, 
                              ComplianceRequirement.PCI_DSS, ComplianceRequirement.FEDRAMP],
                'max_cpu_cores': 448,
                'max_memory_gb': 24576,
                'gpu_support': True,
                'spot_instances': True,
                'reserved_instances': True,
                'availability_zones': 3,
                'network_performance': 'high'
            },
            'gcp': {
                'regions': ['us-central1', 'europe-west1', 'asia-southeast1'],
                'compliance': [ComplianceRequirement.SOC2, ComplianceRequirement.ISO_27001, 
                              ComplianceRequirement.GDPR],
                'max_cpu_cores': 416,
                'max_memory_gb': 11776,
                'gpu_support': True,
                'spot_instances': True,
                'reserved_instances': True,
                'availability_zones': 3,
                'network_performance': 'high'
            },
            'azure': {
                'regions': ['eastus', 'westeurope', 'southeastasia'],
                'compliance': [ComplianceRequirement.SOC2, ComplianceRequirement.HIPAA, 
                              ComplianceRequirement.ISO_27001, ComplianceRequirement.GDPR],
                'max_cpu_cores': 416,
                'max_memory_gb': 11400,
                'gpu_support': True,
                'spot_instances': True,
                'reserved_instances': True,
                'availability_zones': 3,
                'network_performance': 'high'
            }
        }
    
    def _initialize_region_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize region-specific data"""
        return {
            # AWS regions
            'us-east-1': {'latency_score': 0.9, 'cost_multiplier': 1.0, 'availability': 99.99},
            'us-west-2': {'latency_score': 0.85, 'cost_multiplier': 1.05, 'availability': 99.99},
            'eu-west-1': {'latency_score': 0.8, 'cost_multiplier': 1.1, 'availability': 99.95},
            
            # GCP regions
            'us-central1': {'latency_score': 0.88, 'cost_multiplier': 0.95, 'availability': 99.95},
            'europe-west1': {'latency_score': 0.82, 'cost_multiplier': 1.08, 'availability': 99.95},
            'asia-southeast1': {'latency_score': 0.75, 'cost_multiplier': 1.15, 'availability': 99.9},
            
            # Azure regions
            'eastus': {'latency_score': 0.87, 'cost_multiplier': 1.02, 'availability': 99.95},
            'westeurope': {'latency_score': 0.83, 'cost_multiplier': 1.12, 'availability': 99.95},
            'southeastasia': {'latency_score': 0.78, 'cost_multiplier': 1.18, 'availability': 99.9}
        }
    
    async def recommend_placement(self, profile: WorkloadProfile) -> PlacementRecommendation:
        """
        Generate placement recommendations using multi-criteria decision analysis.
        
        Args:
            profile: Analyzed workload profile
            
        Returns:
            Placement recommendation with alternatives
        """
        logger.info("Generating placement recommendations", workload_id=profile.workload_spec.workload_id)
        
        try:
            # Generate placement options for all providers
            placement_options = []
            
            for provider, capabilities in self.provider_capabilities.items():
                # Check if provider can support the workload
                if not self._can_support_workload(profile.workload_spec, capabilities):
                    continue
                
                # Generate options for each region
                for region in capabilities['regions']:
                    option = await self._generate_placement_option(
                        provider, region, profile, capabilities
                    )
                    if option:
                        placement_options.append(option)
            
            if not placement_options:
                raise ValueError("No suitable placement options found")
            
            # Sort options by overall score
            placement_options.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Select best option and alternatives
            recommended_option = placement_options[0]
            alternative_options = placement_options[1:5]  # Top 4 alternatives
            
            # Generate cost and performance comparisons
            cost_comparison = {
                option.provider + '-' + option.region: option.estimated_monthly_cost
                for option in placement_options[:5]
            }
            
            performance_comparison = {
                option.provider + '-' + option.region: option.performance_score
                for option in placement_options[:5]
            }
            
            # Assess risks
            risk_assessment = self._assess_placement_risks(recommended_option, profile)
            
            # Determine migration complexity and timeline
            migration_complexity = self._assess_migration_complexity(profile)
            implementation_timeline = self._estimate_implementation_timeline(profile, migration_complexity)
            
            # Calculate overall confidence
            confidence_score = self._calculate_placement_confidence(
                recommended_option, profile, len(placement_options)
            )
            
            recommendation = PlacementRecommendation(
                workload_id=profile.workload_spec.workload_id,
                recommended_option=recommended_option,
                alternative_options=alternative_options,
                cost_comparison=cost_comparison,
                performance_comparison=performance_comparison,
                risk_assessment=risk_assessment,
                migration_complexity=migration_complexity,
                implementation_timeline=implementation_timeline,
                confidence_score=confidence_score
            )
            
            logger.info(
                "Placement recommendations generated",
                workload_id=profile.workload_spec.workload_id,
                recommended_provider=recommended_option.provider,
                recommended_region=recommended_option.region,
                confidence_score=confidence_score
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(
                "Placement recommendation failed",
                workload_id=profile.workload_spec.workload_id,
                error=str(e)
            )
            raise
    
    def _can_support_workload(self, workload_spec: WorkloadSpec, capabilities: Dict[str, Any]) -> bool:
        """Check if provider can support the workload requirements"""
        # Check resource limits
        if workload_spec.resource_requirements.cpu_cores > capabilities['max_cpu_cores']:
            return False
        
        if workload_spec.resource_requirements.memory_gb > capabilities['max_memory_gb']:
            return False
        
        if workload_spec.resource_requirements.gpu_count > 0 and not capabilities['gpu_support']:
            return False
        
        # Check compliance requirements
        provider_compliance = set(capabilities['compliance'])
        required_compliance = set(workload_spec.compliance_requirements)
        
        if not required_compliance.issubset(provider_compliance):
            return False
        
        return True
    
    async def _generate_placement_option(
        self, 
        provider: str, 
        region: str, 
        profile: WorkloadProfile, 
        capabilities: Dict[str, Any]
    ) -> Optional[PlacementOption]:
        """Generate a placement option for a specific provider and region"""
        
        try:
            workload_spec = profile.workload_spec
            region_data = self.region_data.get(region, {})
            
            # Calculate estimated cost
            estimated_cost = await self._calculate_placement_cost(
                provider, region, workload_spec, region_data
            )
            
            # Calculate performance score
            performance_score = await self._calculate_performance_score(
                provider, region, workload_spec, capabilities, region_data
            )
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(
                workload_spec, capabilities
            )
            
            # Calculate latency score
            latency_score = self._calculate_latency_score(
                workload_spec, region, region_data
            )
            
            # Calculate availability score
            availability_score = self._calculate_availability_score(
                workload_spec, capabilities, region_data
            )
            
            # Calculate overall score using weighted criteria
            overall_score = self._calculate_overall_score(
                workload_spec,
                estimated_cost,
                performance_score,
                compliance_score,
                latency_score,
                availability_score
            )
            
            # Generate instance type recommendations
            instance_types = self._recommend_instance_types(
                provider, workload_spec
            )
            
            # Generate reasoning
            reasoning = self._generate_placement_reasoning(
                provider, region, workload_spec, performance_score, estimated_cost
            )
            
            # Identify limitations
            limitations = self._identify_placement_limitations(
                provider, region, workload_spec, capabilities
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                overall_score, profile.confidence_score
            )
            
            return PlacementOption(
                provider=provider,
                region=region,
                instance_types=instance_types,
                estimated_monthly_cost=estimated_cost,
                performance_score=performance_score,
                compliance_score=compliance_score,
                latency_score=latency_score,
                availability_score=availability_score,
                overall_score=overall_score,
                confidence_interval=confidence_interval,
                reasoning=reasoning,
                limitations=limitations
            )
            
        except Exception as e:
            logger.warning(
                "Failed to generate placement option",
                provider=provider,
                region=region,
                error=str(e)
            )
            return None
    
    async def _calculate_placement_cost(
        self, 
        provider: str, 
        region: str, 
        workload_spec: WorkloadSpec, 
        region_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate estimated monthly cost for placement"""
        
        if self.cost_calculator:
            return await self.cost_calculator.calculate_workload_cost(
                provider, region, workload_spec
            )
        
        # Mock cost calculation
        base_compute_cost = Decimal(str(workload_spec.resource_requirements.cpu_cores * 0.096 * 24 * 30))
        base_memory_cost = Decimal(str(workload_spec.resource_requirements.memory_gb * 0.012 * 24 * 30))
        base_storage_cost = Decimal(str(workload_spec.resource_requirements.storage_gb * 0.023))
        base_network_cost = Decimal(str(workload_spec.resource_requirements.network_bandwidth_mbps * 0.09))
        
        # Apply provider and region multipliers
        provider_multipliers = {'aws': 1.0, 'gcp': 0.95, 'azure': 1.05}
        region_multiplier = region_data.get('cost_multiplier', 1.0)
        
        total_multiplier = Decimal(str(provider_multipliers.get(provider, 1.0) * region_multiplier))
        
        total_cost = (base_compute_cost + base_memory_cost + base_storage_cost + base_network_cost) * total_multiplier
        
        # Apply cost sensitivity adjustments
        if workload_spec.cost_sensitivity.reserved_instance_preference:
            total_cost *= Decimal('0.7')  # 30% discount for reserved instances
        
        return total_cost
    
    async def _calculate_performance_score(
        self, 
        provider: str, 
        region: str, 
        workload_spec: WorkloadSpec, 
        capabilities: Dict[str, Any], 
        region_data: Dict[str, Any]
    ) -> float:
        """Calculate performance score for placement"""
        
        if self.performance_predictor:
            return await self.performance_predictor.predict_performance(
                provider, region, workload_spec
            )
        
        # Mock performance calculation
        base_score = 0.8
        
        # Adjust for network performance
        if capabilities.get('network_performance') == 'high':
            base_score += 0.1
        
        # Adjust for availability zones
        az_count = capabilities.get('availability_zones', 1)
        base_score += min(az_count / 3.0 * 0.1, 0.1)
        
        # Adjust for region latency
        region_latency_score = region_data.get('latency_score', 0.8)
        base_score = (base_score + region_latency_score) / 2.0
        
        # Adjust for workload type
        if workload_spec.workload_type == WorkloadType.COMPUTE_INTENSIVE:
            if provider == 'aws':
                base_score += 0.05  # AWS has strong compute offerings
        elif workload_spec.workload_type == WorkloadType.MACHINE_LEARNING:
            if provider == 'gcp':
                base_score += 0.05  # GCP has strong ML offerings
        
        return min(max(base_score, 0.0), 1.0)
    
    def _calculate_compliance_score(
        self, 
        workload_spec: WorkloadSpec, 
        capabilities: Dict[str, Any]
    ) -> float:
        """Calculate compliance score for placement"""
        
        if not workload_spec.compliance_requirements:
            return 1.0
        
        provider_compliance = set(capabilities['compliance'])
        required_compliance = set(workload_spec.compliance_requirements)
        
        # Check if all requirements are met
        if required_compliance.issubset(provider_compliance):
            # Calculate score based on additional compliance capabilities
            additional_compliance = len(provider_compliance) - len(required_compliance)
            bonus_score = min(additional_compliance * 0.05, 0.2)
            return min(1.0 + bonus_score, 1.0)
        else:
            # Partial compliance
            met_requirements = len(required_compliance.intersection(provider_compliance))
            return met_requirements / len(required_compliance)
    
    def _calculate_latency_score(
        self, 
        workload_spec: WorkloadSpec, 
        region: str, 
        region_data: Dict[str, Any]
    ) -> float:
        """Calculate latency score for placement"""
        
        base_latency_score = region_data.get('latency_score', 0.8)
        
        # Adjust based on latency sensitivity
        if workload_spec.performance_profile.latency_sensitivity == "high":
            # High sensitivity requires better latency
            return base_latency_score
        elif workload_spec.performance_profile.latency_sensitivity == "medium":
            # Medium sensitivity is more forgiving
            return min(base_latency_score + 0.1, 1.0)
        else:
            # Low sensitivity doesn't penalize latency much
            return min(base_latency_score + 0.2, 1.0)
    
    def _calculate_availability_score(
        self, 
        workload_spec: WorkloadSpec, 
        capabilities: Dict[str, Any], 
        region_data: Dict[str, Any]
    ) -> float:
        """Calculate availability score for placement"""
        
        region_availability = region_data.get('availability', 99.9)
        required_availability = workload_spec.performance_profile.availability_requirement
        
        if region_availability >= required_availability:
            # Meets requirements, calculate bonus
            excess_availability = region_availability - required_availability
            bonus = min(excess_availability * 0.1, 0.1)
            return min(0.9 + bonus, 1.0)
        else:
            # Doesn't meet requirements
            return region_availability / required_availability
    
    def _calculate_overall_score(
        self,
        workload_spec: WorkloadSpec,
        estimated_cost: Decimal,
        performance_score: float,
        compliance_score: float,
        latency_score: float,
        availability_score: float
    ) -> float:
        """Calculate overall placement score using weighted criteria"""
        
        # Default weights
        weights = {
            'cost': 0.3,
            'performance': 0.25,
            'compliance': 0.2,
            'latency': 0.15,
            'availability': 0.1
        }
        
        # Adjust weights based on workload preferences
        cost_priority = workload_spec.cost_sensitivity.cost_priority
        
        # Higher cost priority increases cost weight
        weights['cost'] = 0.2 + (cost_priority * 0.3)
        weights['performance'] = 0.35 - (cost_priority * 0.15)
        
        # Adjust for compliance requirements
        if workload_spec.compliance_requirements:
            weights['compliance'] = 0.3
            weights['cost'] *= 0.85
            weights['performance'] *= 0.85
        
        # Adjust for latency sensitivity
        if workload_spec.performance_profile.latency_sensitivity == "high":
            weights['latency'] = 0.25
            weights['cost'] *= 0.9
            weights['performance'] *= 0.9
        
        # Normalize cost score (lower cost = higher score)
        max_reasonable_cost = Decimal('10000')  # $10k per month
        cost_ratio = float(estimated_cost / max_reasonable_cost)
        cost_score = max(0.0, 1.0 - cost_ratio)
        
        # Calculate weighted score
        overall_score = (
            weights['cost'] * cost_score +
            weights['performance'] * performance_score +
            weights['compliance'] * compliance_score +
            weights['latency'] * latency_score +
            weights['availability'] * availability_score
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _recommend_instance_types(self, provider: str, workload_spec: WorkloadSpec) -> List[str]:
        """Recommend instance types for the workload"""
        
        cpu_cores = workload_spec.resource_requirements.cpu_cores
        memory_gb = workload_spec.resource_requirements.memory_gb
        gpu_count = workload_spec.resource_requirements.gpu_count
        
        instance_types = []
        
        if provider == 'aws':
            if gpu_count > 0:
                if gpu_count <= 1:
                    instance_types.append('p3.2xlarge')
                elif gpu_count <= 4:
                    instance_types.append('p3.8xlarge')
                else:
                    instance_types.append('p3.16xlarge')
            elif cpu_cores <= 2:
                instance_types.append('t3.medium')
            elif cpu_cores <= 4:
                instance_types.append('t3.xlarge')
            elif cpu_cores <= 8:
                instance_types.append('c5.2xlarge')
            elif cpu_cores <= 16:
                instance_types.append('c5.4xlarge')
            else:
                instance_types.append('c5.9xlarge')
        
        elif provider == 'gcp':
            if gpu_count > 0:
                instance_types.append('n1-standard-4')  # With GPU attachment
            elif cpu_cores <= 2:
                instance_types.append('e2-medium')
            elif cpu_cores <= 4:
                instance_types.append('e2-standard-4')
            elif cpu_cores <= 8:
                instance_types.append('n2-standard-8')
            elif cpu_cores <= 16:
                instance_types.append('n2-standard-16')
            else:
                instance_types.append('n2-standard-32')
        
        elif provider == 'azure':
            if gpu_count > 0:
                instance_types.append('Standard_NC6')
            elif cpu_cores <= 2:
                instance_types.append('Standard_B2s')
            elif cpu_cores <= 4:
                instance_types.append('Standard_D4s_v3')
            elif cpu_cores <= 8:
                instance_types.append('Standard_D8s_v3')
            elif cpu_cores <= 16:
                instance_types.append('Standard_D16s_v3')
            else:
                instance_types.append('Standard_D32s_v3')
        
        return instance_types
    
    def _generate_placement_reasoning(
        self,
        provider: str,
        region: str,
        workload_spec: WorkloadSpec,
        performance_score: float,
        estimated_cost: Decimal
    ) -> str:
        """Generate human-readable reasoning for placement recommendation"""
        
        reasons = []
        
        # Cost reasoning
        if workload_spec.cost_sensitivity.cost_priority > 0.7:
            reasons.append(f"Cost-optimized choice with estimated monthly cost of ${estimated_cost}")
        
        # Performance reasoning
        if performance_score > 0.8:
            reasons.append(f"High performance score ({performance_score:.2f}) for {workload_spec.workload_type.value}")
        
        # Compliance reasoning
        if workload_spec.compliance_requirements:
            reasons.append(f"Meets all compliance requirements: {', '.join([req.value for req in workload_spec.compliance_requirements])}")
        
        # Regional reasoning
        if region in workload_spec.geographic_preferences:
            reasons.append(f"Matches geographic preference for {region}")
        
        # Provider-specific reasoning
        if provider == 'aws' and workload_spec.workload_type == WorkloadType.WEB_APPLICATION:
            reasons.append("AWS provides excellent web application hosting capabilities")
        elif provider == 'gcp' and workload_spec.workload_type == WorkloadType.MACHINE_LEARNING:
            reasons.append("GCP offers superior machine learning and AI services")
        elif provider == 'azure' and workload_spec.workload_type == WorkloadType.DATABASE:
            reasons.append("Azure provides robust database services and integration")
        
        return ". ".join(reasons) if reasons else f"Balanced choice for {workload_spec.workload_type.value} workload"
    
    def _identify_placement_limitations(
        self,
        provider: str,
        region: str,
        workload_spec: WorkloadSpec,
        capabilities: Dict[str, Any]
    ) -> List[str]:
        """Identify limitations of the placement option"""
        
        limitations = []
        
        # Resource limitations
        if workload_spec.resource_requirements.cpu_cores > capabilities['max_cpu_cores'] * 0.8:
            limitations.append("Approaching maximum CPU core limits")
        
        if workload_spec.resource_requirements.memory_gb > capabilities['max_memory_gb'] * 0.8:
            limitations.append("Approaching maximum memory limits")
        
        # Compliance limitations
        provider_compliance = set(capabilities['compliance'])
        all_compliance = {ComplianceRequirement.GDPR, ComplianceRequirement.HIPAA, 
                         ComplianceRequirement.SOC2, ComplianceRequirement.PCI_DSS,
                         ComplianceRequirement.ISO_27001, ComplianceRequirement.FEDRAMP}
        missing_compliance = all_compliance - provider_compliance
        
        if missing_compliance:
            limitations.append(f"Does not support: {', '.join([req.value for req in missing_compliance])}")
        
        # Regional limitations
        if workload_spec.performance_profile.latency_sensitivity == "high" and region not in ['us-east-1', 'us-central1', 'eastus']:
            limitations.append("May not provide optimal latency for high-sensitivity workloads")
        
        # Cost limitations
        if not workload_spec.cost_sensitivity.spot_instance_tolerance and provider in ['aws', 'gcp', 'azure']:
            limitations.append("Not utilizing spot instances for potential cost savings")
        
        return limitations
    
    def _calculate_confidence_interval(self, overall_score: float, profile_confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for the placement score"""
        
        # Base confidence interval width
        base_width = 0.1
        
        # Adjust width based on profile confidence
        width = base_width * (2.0 - profile_confidence)
        
        # Calculate bounds
        lower_bound = max(overall_score - width/2, 0.0)
        upper_bound = min(overall_score + width/2, 1.0)
        
        return (lower_bound, upper_bound)
    
    def _assess_placement_risks(self, option: PlacementOption, profile: WorkloadProfile) -> Dict[str, float]:
        """Assess risks associated with the placement"""
        
        risks = {}
        
        # Cost risk
        if option.estimated_monthly_cost > Decimal('5000'):
            risks['cost_overrun'] = 0.3
        else:
            risks['cost_overrun'] = 0.1
        
        # Performance risk
        if option.performance_score < 0.7:
            risks['performance_degradation'] = 0.4
        else:
            risks['performance_degradation'] = 0.1
        
        # Compliance risk
        if option.compliance_score < 1.0:
            risks['compliance_violation'] = 0.5
        else:
            risks['compliance_violation'] = 0.05
        
        # Availability risk
        if option.availability_score < 0.9:
            risks['availability_issues'] = 0.3
        else:
            risks['availability_issues'] = 0.05
        
        # Vendor lock-in risk
        risks['vendor_lockin'] = 0.2  # Base risk for all cloud providers
        
        return risks
    
    def _assess_migration_complexity(self, profile: WorkloadProfile) -> str:
        """Assess migration complexity"""
        
        complexity_factors = 0
        
        # Resource complexity
        if profile.complexity_score > 0.7:
            complexity_factors += 2
        elif profile.complexity_score > 0.4:
            complexity_factors += 1
        
        # Compliance complexity
        if len(profile.workload_spec.compliance_requirements) > 2:
            complexity_factors += 2
        elif len(profile.workload_spec.compliance_requirements) > 0:
            complexity_factors += 1
        
        # Performance complexity
        if profile.workload_spec.performance_profile.availability_requirement > 99.9:
            complexity_factors += 1
        
        # Data complexity
        if profile.workload_spec.resource_requirements.storage_gb > 1000:
            complexity_factors += 1
        
        if complexity_factors >= 5:
            return "high"
        elif complexity_factors >= 3:
            return "medium"
        else:
            return "low"
    
    def _estimate_implementation_timeline(self, profile: WorkloadProfile, complexity: str) -> str:
        """Estimate implementation timeline"""
        
        base_weeks = {
            "low": 2,
            "medium": 4,
            "high": 8
        }
        
        weeks = base_weeks.get(complexity, 4)
        
        # Adjust for specific factors
        if len(profile.workload_spec.compliance_requirements) > 2:
            weeks += 2
        
        if profile.workload_spec.resource_requirements.storage_gb > 5000:
            weeks += 1
        
        if profile.workload_spec.performance_profile.availability_requirement > 99.95:
            weeks += 1
        
        if weeks <= 2:
            return "1-2 weeks"
        elif weeks <= 4:
            return "2-4 weeks"
        elif weeks <= 8:
            return "1-2 months"
        else:
            return "2-3 months"
    
    def _calculate_placement_confidence(
        self, 
        option: PlacementOption, 
        profile: WorkloadProfile, 
        option_count: int
    ) -> float:
        """Calculate overall confidence in the placement recommendation"""
        
        # Base confidence from profile
        base_confidence = profile.confidence_score
        
        # Adjust for option quality
        if option.overall_score > 0.8:
            base_confidence += 0.1
        elif option.overall_score < 0.6:
            base_confidence -= 0.1
        
        # Adjust for number of options (more options = more confidence)
        if option_count > 5:
            base_confidence += 0.05
        elif option_count < 3:
            base_confidence -= 0.05
        
        # Adjust for compliance completeness
        if option.compliance_score >= 1.0:
            base_confidence += 0.05
        
        return min(max(base_confidence, 0.1), 1.0)


class PerformancePredictor:
    """ML-based performance modeling for workload placement"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for performance prediction"""
        # Response time prediction model
        self.models['response_time'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Throughput prediction model
        self.models['throughput'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Resource utilization prediction model
        self.models['utilization'] = RandomForestRegressor(
            n_estimators=80,
            random_state=42,
            max_depth=8
        )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    async def predict_performance(
        self, 
        provider: str, 
        region: str, 
        workload_spec: WorkloadSpec
    ) -> float:
        """
        Predict performance score for workload on specific provider/region.
        
        Args:
            provider: Cloud provider name
            region: Region name
            workload_spec: Workload specification
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        logger.info(
            "Predicting performance",
            provider=provider,
            region=region,
            workload_id=workload_spec.workload_id
        )
        
        try:
            # Extract features for prediction
            features = self._extract_performance_features(provider, region, workload_spec)
            
            # Predict individual performance metrics
            response_time = await self._predict_response_time(features)
            throughput = await self._predict_throughput(features)
            utilization = await self._predict_utilization(features)
            
            # Calculate composite performance score
            performance_score = self._calculate_composite_score(
                response_time, throughput, utilization, workload_spec
            )
            
            # Apply provider-specific adjustments
            adjusted_score = self._apply_provider_adjustments(
                performance_score, provider, workload_spec
            )
            
            logger.info(
                "Performance prediction completed",
                provider=provider,
                region=region,
                performance_score=adjusted_score
            )
            
            return adjusted_score
            
        except Exception as e:
            logger.error(
                "Performance prediction failed",
                provider=provider,
                region=region,
                error=str(e)
            )
            # Return conservative estimate
            return 0.7
    
    def _extract_performance_features(
        self, 
        provider: str, 
        region: str, 
        workload_spec: WorkloadSpec
    ) -> np.ndarray:
        """Extract features for performance prediction"""
        
        # Provider encoding
        provider_encoding = {'aws': 1.0, 'gcp': 2.0, 'azure': 3.0}
        
        # Region tier encoding (based on typical performance characteristics)
        region_tiers = {
            'us-east-1': 1.0, 'us-central1': 1.0, 'eastus': 1.0,  # Tier 1
            'us-west-2': 2.0, 'europe-west1': 2.0, 'westeurope': 2.0,  # Tier 2
            'eu-west-1': 3.0, 'asia-southeast1': 3.0, 'southeastasia': 3.0  # Tier 3
        }
        
        features = [
            # Provider and region features
            provider_encoding.get(provider, 1.0),
            region_tiers.get(region, 2.0),
            
            # Resource features
            workload_spec.resource_requirements.cpu_cores,
            workload_spec.resource_requirements.memory_gb,
            workload_spec.resource_requirements.storage_gb,
            workload_spec.resource_requirements.network_bandwidth_mbps,
            workload_spec.resource_requirements.gpu_count,
            workload_spec.resource_requirements.iops_required,
            
            # Performance profile features
            workload_spec.performance_profile.cpu_utilization_avg,
            workload_spec.performance_profile.memory_utilization_avg,
            workload_spec.performance_profile.throughput_requirements,
            workload_spec.performance_profile.availability_requirement,
            
            # Workload type encoding
            self._encode_workload_type_for_performance(workload_spec.workload_type),
            
            # IO pattern encoding
            self._encode_io_pattern(workload_spec.performance_profile.network_io_pattern),
            self._encode_io_pattern(workload_spec.performance_profile.storage_io_pattern),
            
            # Latency sensitivity encoding
            self._encode_latency_sensitivity_for_performance(
                workload_spec.performance_profile.latency_sensitivity
            ),
            
            # Scaling pattern encoding
            self._encode_scaling_pattern_for_performance(
                workload_spec.performance_profile.scaling_pattern
            )
        ]
        
        return np.array(features).reshape(1, -1)
    
    async def _predict_response_time(self, features: np.ndarray) -> float:
        """Predict response time in milliseconds"""
        try:
            # Mock prediction (in production, use trained model)
            base_response_time = 100.0  # 100ms base
            
            # Adjust based on CPU cores (more cores = better response time)
            cpu_cores = features[0][2]
            cpu_factor = max(0.5, 1.0 - (cpu_cores - 4) * 0.05)
            
            # Adjust based on memory (more memory = better response time)
            memory_gb = features[0][3]
            memory_factor = max(0.7, 1.0 - (memory_gb - 8) * 0.02)
            
            # Adjust based on utilization
            cpu_util = features[0][8]
            util_factor = 1.0 + (cpu_util - 50) * 0.01
            
            predicted_response_time = base_response_time * cpu_factor * memory_factor * util_factor
            
            return max(predicted_response_time, 10.0)  # Minimum 10ms
            
        except Exception as e:
            logger.warning("Response time prediction failed, using default", error=str(e))
            return 100.0
    
    async def _predict_throughput(self, features: np.ndarray) -> float:
        """Predict throughput in requests per second"""
        try:
            # Mock prediction (in production, use trained model)
            base_throughput = 1000.0  # 1000 RPS base
            
            # Adjust based on CPU cores
            cpu_cores = features[0][2]
            cpu_factor = 1.0 + (cpu_cores - 4) * 0.2
            
            # Adjust based on network bandwidth
            network_mbps = features[0][5]
            network_factor = 1.0 + min(network_mbps / 1000.0, 2.0)
            
            # Adjust based on workload type
            workload_type_encoding = features[0][12]
            if workload_type_encoding == 1.0:  # Web application
                type_factor = 1.2
            elif workload_type_encoding == 2.0:  # Database
                type_factor = 0.8
            else:
                type_factor = 1.0
            
            predicted_throughput = base_throughput * cpu_factor * network_factor * type_factor
            
            return max(predicted_throughput, 100.0)  # Minimum 100 RPS
            
        except Exception as e:
            logger.warning("Throughput prediction failed, using default", error=str(e))
            return 1000.0
    
    async def _predict_utilization(self, features: np.ndarray) -> Dict[str, float]:
        """Predict resource utilization percentages"""
        try:
            # Mock prediction (in production, use trained model)
            base_cpu_util = features[0][8]  # Current CPU utilization
            base_memory_util = features[0][9]  # Current memory utilization
            
            # Predict under load
            cpu_cores = features[0][2]
            memory_gb = features[0][3]
            
            # More resources typically lead to lower utilization under same load
            cpu_util_factor = max(0.5, 1.0 - (cpu_cores - 4) * 0.05)
            memory_util_factor = max(0.5, 1.0 - (memory_gb - 8) * 0.03)
            
            predicted_cpu_util = base_cpu_util * cpu_util_factor
            predicted_memory_util = base_memory_util * memory_util_factor
            
            return {
                'cpu': min(max(predicted_cpu_util, 10.0), 95.0),
                'memory': min(max(predicted_memory_util, 10.0), 90.0),
                'network': min(features[0][5] / 1000.0 * 50, 80.0),  # Network utilization
                'storage': min(features[0][7] / 10000.0 * 60, 70.0)  # Storage utilization
            }
            
        except Exception as e:
            logger.warning("Utilization prediction failed, using defaults", error=str(e))
            return {'cpu': 60.0, 'memory': 70.0, 'network': 40.0, 'storage': 30.0}
    
    def _calculate_composite_score(
        self, 
        response_time: float, 
        throughput: float, 
        utilization: Dict[str, float], 
        workload_spec: WorkloadSpec
    ) -> float:
        """Calculate composite performance score"""
        
        # Normalize metrics to 0-1 scale
        # Response time score (lower is better)
        response_time_score = max(0.0, 1.0 - (response_time - 50) / 500.0)
        
        # Throughput score (higher is better, up to requirements)
        required_throughput = workload_spec.performance_profile.throughput_requirements
        if required_throughput > 0:
            throughput_score = min(throughput / required_throughput, 1.0)
        else:
            throughput_score = min(throughput / 1000.0, 1.0)
        
        # Utilization score (optimal around 70-80%)
        avg_utilization = sum(utilization.values()) / len(utilization)
        if 70 <= avg_utilization <= 80:
            utilization_score = 1.0
        elif avg_utilization < 70:
            utilization_score = 0.8 + (avg_utilization - 50) / 20.0 * 0.2
        else:
            utilization_score = max(0.0, 1.0 - (avg_utilization - 80) / 20.0)
        
        # Weight the scores based on workload type
        if workload_spec.workload_type == WorkloadType.WEB_APPLICATION:
            weights = {'response_time': 0.4, 'throughput': 0.4, 'utilization': 0.2}
        elif workload_spec.workload_type == WorkloadType.DATABASE:
            weights = {'response_time': 0.3, 'throughput': 0.3, 'utilization': 0.4}
        elif workload_spec.workload_type == WorkloadType.BATCH_PROCESSING:
            weights = {'response_time': 0.2, 'throughput': 0.3, 'utilization': 0.5}
        else:
            weights = {'response_time': 0.33, 'throughput': 0.33, 'utilization': 0.34}
        
        composite_score = (
            weights['response_time'] * response_time_score +
            weights['throughput'] * throughput_score +
            weights['utilization'] * utilization_score
        )
        
        return min(max(composite_score, 0.0), 1.0)
    
    def _apply_provider_adjustments(
        self, 
        base_score: float, 
        provider: str, 
        workload_spec: WorkloadSpec
    ) -> float:
        """Apply provider-specific performance adjustments"""
        
        adjusted_score = base_score
        
        # Provider strengths
        if provider == 'aws':
            if workload_spec.workload_type == WorkloadType.WEB_APPLICATION:
                adjusted_score += 0.05  # AWS excels at web applications
            elif workload_spec.workload_type == WorkloadType.STORAGE_INTENSIVE:
                adjusted_score += 0.03  # Strong storage offerings
        
        elif provider == 'gcp':
            if workload_spec.workload_type == WorkloadType.MACHINE_LEARNING:
                adjusted_score += 0.08  # GCP excels at ML workloads
            elif workload_spec.workload_type == WorkloadType.ANALYTICS:
                adjusted_score += 0.05  # Strong analytics platform
        
        elif provider == 'azure':
            if workload_spec.workload_type == WorkloadType.DATABASE:
                adjusted_score += 0.04  # Strong database services
            elif workload_spec.workload_type == WorkloadType.MICROSERVICES:
                adjusted_score += 0.03  # Good container orchestration
        
        # Network performance adjustments
        if workload_spec.performance_profile.network_io_pattern == "burst":
            if provider == 'aws':
                adjusted_score += 0.02  # AWS handles burst traffic well
        
        # GPU workload adjustments
        if workload_spec.resource_requirements.gpu_count > 0:
            if provider == 'gcp':
                adjusted_score += 0.03  # GCP has competitive GPU offerings
        
        return min(max(adjusted_score, 0.0), 1.0)
    
    def _encode_workload_type_for_performance(self, workload_type: WorkloadType) -> float:
        """Encode workload type for performance prediction"""
        encoding = {
            WorkloadType.WEB_APPLICATION: 1.0,
            WorkloadType.DATABASE: 2.0,
            WorkloadType.ANALYTICS: 3.0,
            WorkloadType.MACHINE_LEARNING: 4.0,
            WorkloadType.BATCH_PROCESSING: 5.0,
            WorkloadType.MICROSERVICES: 6.0,
            WorkloadType.STORAGE_INTENSIVE: 7.0,
            WorkloadType.COMPUTE_INTENSIVE: 8.0
        }
        return encoding.get(workload_type, 1.0)
    
    def _encode_io_pattern(self, pattern: str) -> float:
        """Encode IO pattern for performance prediction"""
        encoding = {
            "burst": 1.0, "steady": 2.0, "periodic": 3.0,
            "read_heavy": 1.0, "write_heavy": 2.0, "balanced": 3.0
        }
        return encoding.get(pattern, 2.0)
    
    def _encode_latency_sensitivity_for_performance(self, sensitivity: str) -> float:
        """Encode latency sensitivity for performance prediction"""
        encoding = {"low": 1.0, "medium": 2.0, "high": 3.0}
        return encoding.get(sensitivity, 2.0)
    
    def _encode_scaling_pattern_for_performance(self, pattern: str) -> float:
        """Encode scaling pattern for performance prediction"""
        encoding = {"predictable": 1.0, "seasonal": 2.0, "unpredictable": 3.0}
        return encoding.get(pattern, 2.0)


class ContinuousReassessmentEngine:
    """Engine for continuous workload reassessment and migration recommendations"""
    
    def __init__(self, workload_analyzer: WorkloadAnalyzer, placement_optimizer: PlacementOptimizer):
        self.workload_analyzer = workload_analyzer
        self.placement_optimizer = placement_optimizer
        self.reassessment_history = {}
        self.migration_recommendations = {}
        self.reassessment_interval_hours = 24  # Daily reassessment by default
    
    async def schedule_continuous_reassessment(self, workload_id: str, workload_spec: WorkloadSpec):
        """Schedule continuous reassessment for a workload"""
        logger.info("Scheduling continuous reassessment", workload_id=workload_id)
        
        try:
            # Perform initial assessment
            profile = await self.workload_analyzer.analyze_workload(workload_spec)
            recommendation = await self.placement_optimizer.recommend_placement(profile)
            
            # Store initial recommendation
            self.reassessment_history[workload_id] = {
                'last_assessment': datetime.utcnow(),
                'current_recommendation': recommendation,
                'assessment_count': 1,
                'performance_history': []
            }
            
            logger.info(
                "Initial assessment completed",
                workload_id=workload_id,
                recommended_provider=recommendation.recommended_option.provider
            )
            
            return recommendation
            
        except Exception as e:
            logger.error("Failed to schedule continuous reassessment", workload_id=workload_id, error=str(e))
            raise
    
    async def perform_reassessment(self, workload_id: str, workload_spec: WorkloadSpec) -> Optional[PlacementRecommendation]:
        """Perform reassessment for a workload"""
        logger.info("Performing workload reassessment", workload_id=workload_id)
        
        try:
            if workload_id not in self.reassessment_history:
                return await self.schedule_continuous_reassessment(workload_id, workload_spec)
            
            # Analyze current workload state
            current_profile = await self.workload_analyzer.analyze_workload(workload_spec)
            new_recommendation = await self.placement_optimizer.recommend_placement(current_profile)
            
            # Compare with previous recommendation
            history = self.reassessment_history[workload_id]
            previous_recommendation = history['current_recommendation']
            
            # Check if recommendation has changed significantly
            if self._should_update_recommendation(previous_recommendation, new_recommendation):
                # Generate migration analysis if placement changed
                if (previous_recommendation.recommended_option.provider != new_recommendation.recommended_option.provider or
                    previous_recommendation.recommended_option.region != new_recommendation.recommended_option.region):
                    
                    migration_analysis = await self.analyze_migration(
                        previous_recommendation.recommended_option,
                        new_recommendation.recommended_option
                    )
                    
                    # Store migration recommendation
                    self.migration_recommendations[workload_id] = {
                        'migration_analysis': migration_analysis,
                        'generated_at': datetime.utcnow(),
                        'confidence_score': new_recommendation.confidence_score
                    }
                
                # Update history
                history['current_recommendation'] = new_recommendation
                history['last_assessment'] = datetime.utcnow()
                history['assessment_count'] += 1
                
                logger.info(
                    "Recommendation updated",
                    workload_id=workload_id,
                    new_provider=new_recommendation.recommended_option.provider,
                    previous_provider=previous_recommendation.recommended_option.provider
                )
                
                return new_recommendation
            else:
                # Update assessment timestamp but keep same recommendation
                history['last_assessment'] = datetime.utcnow()
                history['assessment_count'] += 1
                
                logger.info("No significant changes in recommendation", workload_id=workload_id)
                return None
                
        except Exception as e:
            logger.error("Reassessment failed", workload_id=workload_id, error=str(e))
            raise
    
    async def analyze_migration(self, current_placement: PlacementOption, target_placement: PlacementOption) -> MigrationAnalysis:
        """Analyze migration between two placements"""
        logger.info(
            "Analyzing migration",
            from_provider=current_placement.provider,
            to_provider=target_placement.provider
        )
        
        try:
            # Calculate migration cost
            migration_cost = self._calculate_migration_cost(current_placement, target_placement)
            
            # Estimate migration time
            migration_time_hours = self._estimate_migration_time(current_placement, target_placement)
            
            # Calculate downtime
            downtime_hours = self._estimate_downtime(current_placement, target_placement)
            
            # Estimate data transfer
            data_transfer_gb = self._estimate_data_transfer(current_placement, target_placement)
            
            # Assess risk level
            risk_level = self._assess_migration_risk(current_placement, target_placement)
            
            # Calculate break-even period
            monthly_savings = current_placement.estimated_monthly_cost - target_placement.estimated_monthly_cost
            break_even_months = int(migration_cost / max(monthly_savings, Decimal('1')))
            
            # Calculate ROI
            annual_savings = monthly_savings * 12
            roi_percentage = float((annual_savings - migration_cost) / migration_cost * 100) if migration_cost > 0 else 0.0
            
            # Generate migration steps
            migration_steps = self._generate_migration_steps(current_placement, target_placement)
            
            analysis = MigrationAnalysis(
                current_placement=current_placement,
                target_placement=target_placement,
                migration_cost=migration_cost,
                migration_time_hours=migration_time_hours,
                downtime_hours=downtime_hours,
                data_transfer_gb=data_transfer_gb,
                risk_level=risk_level,
                break_even_months=break_even_months,
                roi_percentage=roi_percentage,
                migration_steps=migration_steps
            )
            
            logger.info(
                "Migration analysis completed",
                migration_cost=migration_cost,
                break_even_months=break_even_months,
                roi_percentage=roi_percentage
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Migration analysis failed", error=str(e))
            raise
    
    def _should_update_recommendation(self, previous: PlacementRecommendation, new: PlacementRecommendation) -> bool:
        """Determine if recommendation should be updated"""
        
        # Update if provider or region changed
        if (previous.recommended_option.provider != new.recommended_option.provider or
            previous.recommended_option.region != new.recommended_option.region):
            return True
        
        # Update if cost savings are significant (>10%)
        cost_diff = abs(previous.recommended_option.estimated_monthly_cost - new.recommended_option.estimated_monthly_cost)
        cost_threshold = previous.recommended_option.estimated_monthly_cost * Decimal('0.1')
        if cost_diff > cost_threshold:
            return True
        
        # Update if performance score improved significantly (>0.1)
        perf_diff = abs(previous.recommended_option.performance_score - new.recommended_option.performance_score)
        if perf_diff > 0.1:
            return True
        
        # Update if confidence score improved significantly (>0.15)
        conf_diff = abs(previous.confidence_score - new.confidence_score)
        if conf_diff > 0.15:
            return True
        
        return False
    
    def _calculate_migration_cost(self, current: PlacementOption, target: PlacementOption) -> Decimal:
        """Calculate migration cost between placements"""
        
        base_migration_cost = Decimal('5000')  # Base migration cost
        
        # Add cost for cross-provider migration
        if current.provider != target.provider:
            base_migration_cost += Decimal('10000')
        
        # Add cost for cross-region migration
        if current.region != target.region:
            base_migration_cost += Decimal('2000')
        
        # Add cost based on complexity (estimated from instance types)
        complexity_multiplier = len(current.instance_types) * Decimal('0.5')
        base_migration_cost *= (1 + complexity_multiplier)
        
        return base_migration_cost
    
    def _estimate_migration_time(self, current: PlacementOption, target: PlacementOption) -> int:
        """Estimate migration time in hours"""
        
        base_time = 8  # 8 hours base
        
        # Add time for cross-provider migration
        if current.provider != target.provider:
            base_time += 16
        
        # Add time for cross-region migration
        if current.region != target.region:
            base_time += 4
        
        # Add time based on complexity
        complexity_hours = len(current.instance_types) * 2
        
        return base_time + complexity_hours
    
    def _estimate_downtime(self, current: PlacementOption, target: PlacementOption) -> float:
        """Estimate downtime in hours"""
        
        if current.provider == target.provider and current.region == target.region:
            return 0.5  # Same provider/region, minimal downtime
        elif current.provider == target.provider:
            return 2.0  # Same provider, different region
        else:
            return 4.0  # Different provider, more downtime
    
    def _estimate_data_transfer(self, current: PlacementOption, target: PlacementOption) -> int:
        """Estimate data transfer in GB"""
        
        # Mock estimation based on instance types (in production, would use actual data)
        base_data = 100  # 100 GB base
        
        # Estimate based on instance count and types
        instance_multiplier = len(current.instance_types) * 50
        
        return base_data + instance_multiplier
    
    def _assess_migration_risk(self, current: PlacementOption, target: PlacementOption) -> str:
        """Assess migration risk level"""
        
        risk_factors = 0
        
        # Cross-provider migration is riskier
        if current.provider != target.provider:
            risk_factors += 2
        
        # Cross-region migration adds some risk
        if current.region != target.region:
            risk_factors += 1
        
        # Complex workloads are riskier to migrate
        if len(current.instance_types) > 3:
            risk_factors += 1
        
        # Low confidence in target placement increases risk
        if target.confidence_interval[0] < 0.6:
            risk_factors += 1
        
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_migration_steps(self, current: PlacementOption, target: PlacementOption) -> List[str]:
        """Generate migration steps"""
        
        steps = [
            "1. Backup current workload data and configurations",
            "2. Set up target infrastructure in destination provider/region",
            "3. Configure networking and security groups",
            "4. Test connectivity and basic functionality"
        ]
        
        if current.provider != target.provider:
            steps.extend([
                "5. Set up data replication between providers",
                "6. Configure DNS and load balancer for traffic switching",
                "7. Perform gradual traffic migration with monitoring",
                "8. Validate all services in target environment",
                "9. Complete traffic cutover and monitor for issues",
                "10. Decommission source infrastructure after validation"
            ])
        else:
            steps.extend([
                "5. Set up data replication to target region",
                "6. Update DNS and routing configurations",
                "7. Perform traffic cutover with monitoring",
                "8. Validate services and decommission source resources"
            ])
        
        return steps
    
    async def get_migration_recommendations(self, workload_id: str) -> Optional[Dict[str, Any]]:
        """Get migration recommendations for a workload"""
        return self.migration_recommendations.get(workload_id)
    
    async def get_reassessment_history(self, workload_id: str) -> Optional[Dict[str, Any]]:
        """Get reassessment history for a workload"""
        return self.reassessment_history.get(workload_id)


class CostCalculator:
    """Accurate TCO analysis across providers for workload placement"""
    
    def __init__(self):
        self.provider_pricing = self._initialize_provider_pricing()
        self.region_multipliers = self._initialize_region_multipliers()
    
    def _initialize_provider_pricing(self) -> Dict[str, Dict[str, float]]:
        """Initialize provider pricing data"""
        return {
            'aws': {
                'compute_per_vcpu_hour': 0.048,
                'memory_per_gb_hour': 0.006,
                'storage_per_gb_month': 0.10,
                'network_per_gb': 0.09,
                'gpu_per_hour': 3.06
            },
            'gcp': {
                'compute_per_vcpu_hour': 0.045,
                'memory_per_gb_hour': 0.0055,
                'storage_per_gb_month': 0.08,
                'network_per_gb': 0.08,
                'gpu_per_hour': 2.85
            },
            'azure': {
                'compute_per_vcpu_hour': 0.050,
                'memory_per_gb_hour': 0.0065,
                'storage_per_gb_month': 0.12,
                'network_per_gb': 0.10,
                'gpu_per_hour': 3.20
            }
        }
    
    def _initialize_region_multipliers(self) -> Dict[str, float]:
        """Initialize region cost multipliers"""
        return {
            # AWS regions
            'us-east-1': 1.0,
            'us-west-2': 1.05,
            'eu-west-1': 1.15,
            
            # GCP regions
            'us-central1': 1.0,
            'europe-west1': 1.12,
            'asia-southeast1': 1.20,
            
            # Azure regions
            'eastus': 1.0,
            'westeurope': 1.18,
            'southeastasia': 1.25
        }
    
    async def calculate_workload_cost(self, provider: str, region: str, workload_spec: WorkloadSpec) -> Decimal:
        """Calculate monthly cost for workload on specific provider/region"""
        
        try:
            pricing = self.provider_pricing.get(provider, self.provider_pricing['aws'])
            region_multiplier = self.region_multipliers.get(region, 1.0)
            
            # Calculate base costs
            compute_cost = (
                workload_spec.resource_requirements.cpu_cores * 
                pricing['compute_per_vcpu_hour'] * 24 * 30
            )
            
            memory_cost = (
                workload_spec.resource_requirements.memory_gb * 
                pricing['memory_per_gb_hour'] * 24 * 30
            )
            
            storage_cost = (
                workload_spec.resource_requirements.storage_gb * 
                pricing['storage_per_gb_month']
            )
            
            network_cost = (
                workload_spec.resource_requirements.network_bandwidth_mbps * 
                pricing['network_per_gb'] * 0.1  # Approximate GB per month
            )
            
            gpu_cost = 0
            if workload_spec.resource_requirements.gpu_count > 0:
                gpu_cost = (
                    workload_spec.resource_requirements.gpu_count * 
                    pricing['gpu_per_hour'] * 24 * 30
                )
            
            # Calculate total base cost
            total_cost = compute_cost + memory_cost + storage_cost + network_cost + gpu_cost
            
            # Apply region multiplier
            total_cost *= region_multiplier
            
            # Apply reserved instance discount if preferred
            if workload_spec.cost_sensitivity.reserved_instance_preference:
                total_cost *= 0.65  # 35% discount for 1-year reserved instances
            
            # Apply spot instance discount if tolerated
            if workload_spec.cost_sensitivity.spot_instance_tolerance:
                total_cost *= 0.30  # 70% discount for spot instances
            
            return Decimal(str(round(total_cost, 2)))
            
        except Exception as e:
            logger.error(f"Cost calculation failed for {provider}/{region}: {e}")
            # Return conservative estimate
            return Decimal('1000.00')


class WorkloadIntelligenceSystem:
    """Main system orchestrating workload intelligence components"""
    
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self.performance_predictor = PerformancePredictor()
        self.workload_analyzer = WorkloadAnalyzer()
        self.placement_optimizer = PlacementOptimizer(
            cost_calculator=self.cost_calculator,
            performance_predictor=self.performance_predictor
        )
        self.reassessment_engine = ContinuousReassessmentEngine(
            workload_analyzer=self.workload_analyzer,
            placement_optimizer=self.placement_optimizer
        )
    
    async def analyze_and_recommend_placement(self, workload_spec: WorkloadSpec) -> PlacementRecommendation:
        """Complete workload analysis and placement recommendation"""
        logger.info("Starting workload intelligence analysis", workload_id=workload_spec.workload_id)
        
        try:
            # Analyze workload characteristics
            profile = await self.workload_analyzer.analyze_workload(workload_spec)
            
            # Generate placement recommendations
            recommendation = await self.placement_optimizer.recommend_placement(profile)
            
            # Schedule continuous reassessment
            await self.reassessment_engine.schedule_continuous_reassessment(
                workload_spec.workload_id, workload_spec
            )
            
            logger.info(
                "Workload intelligence analysis completed",
                workload_id=workload_spec.workload_id,
                recommended_provider=recommendation.recommended_option.provider,
                confidence_score=recommendation.confidence_score
            )
            
            return recommendation
            
        except Exception as e:
            logger.error("Workload intelligence analysis failed", workload_id=workload_spec.workload_id, error=str(e))
            raise
    
    async def reassess_workload(self, workload_id: str, workload_spec: WorkloadSpec) -> Optional[PlacementRecommendation]:
        """Reassess workload and generate new recommendations if needed"""
        return await self.reassessment_engine.perform_reassessment(workload_id, workload_spec)
    
    async def get_migration_analysis(self, workload_id: str) -> Optional[Dict[str, Any]]:
        """Get migration analysis for a workload"""
        return await self.reassessment_engine.get_migration_recommendations(workload_id)
    
    async def evaluate_migration(self, current_placement: PlacementOption, target_placement: PlacementOption) -> MigrationAnalysis:
        """Evaluate migration between two placements"""
        return await self.reassessment_engine.analyze_migration(current_placement, target_placement)
    
    async def optimize_multi_workload(self, workloads: List[WorkloadSpec]) -> Dict[str, PlacementRecommendation]:
        """Optimize placement for multiple workloads considering interdependencies"""
        logger.info("Optimizing multi-workload placement", workload_count=len(workloads))
        
        try:
            recommendations = {}
            
            # Analyze each workload individually first
            for workload in workloads:
                recommendation = await self.analyze_and_recommend_placement(workload)
                recommendations[workload.workload_id] = recommendation
            
            # TODO: Implement cross-workload optimization considering:
            # - Network latency between workloads
            # - Data transfer costs
            # - Shared resource opportunities
            # - Compliance boundary requirements
            
            logger.info("Multi-workload optimization completed", workload_count=len(workloads))
            return recommendations
            
        except Exception as e:
            logger.error("Multi-workload optimization failed", error=str(e))
            raise