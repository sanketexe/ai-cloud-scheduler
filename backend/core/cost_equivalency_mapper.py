"""
Cost Equivalency Mapper

Maps services across cloud providers and analyzes feature parity for accurate cost comparisons.
Handles compute, storage, network, and database service equivalencies.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .multi_cloud_models import WorkloadSpec, ComputeSpec, StorageSpec, NetworkSpec, DatabaseSpec
from .service_mappings import SERVICE_EQUIVALENCY_CONFIG, FEATURE_MAPPING_CONFIG

logger = logging.getLogger(__name__)


class ServiceCategory(Enum):
    """Service categories for mapping."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    ANALYTICS = "analytics"
    SECURITY = "security"


@dataclass
class EquivalentService:
    """Represents an equivalent service across providers."""
    provider: str
    service_name: str
    service_type: str
    confidence_score: float  # 0.0 to 1.0
    feature_parity_score: float  # 0.0 to 1.0
    performance_ratio: float  # Relative performance compared to reference
    cost_efficiency_score: float  # Cost per performance unit
    limitations: List[str]
    additional_features: List[str]
    migration_complexity: str  # "low", "medium", "high"


@dataclass
class FeatureParityMatrix:
    """Feature parity analysis between services."""
    reference_service: str
    equivalent_services: List[EquivalentService]
    feature_comparison: Dict[str, Dict[str, bool]]  # feature -> provider -> supported
    missing_features: Dict[str, List[str]]  # provider -> missing features
    additional_features: Dict[str, List[str]]  # provider -> additional features
    overall_parity_score: float


@dataclass
class ServiceMapping:
    """Complete service mapping for a workload component."""
    source_service: str
    category: ServiceCategory
    mappings: Dict[str, List[EquivalentService]]  # provider -> equivalent services
    recommended_mapping: Dict[str, EquivalentService]  # provider -> best match
    mapping_confidence: float


class CostEquivalencyMapper:
    """
    Maps services across cloud providers and analyzes equivalencies for cost comparison.
    
    Provides intelligent service mapping based on:
    - Feature parity analysis
    - Performance characteristics
    - Cost efficiency
    - Migration complexity
    """
    
    def __init__(self):
        """Initialize the cost equivalency mapper."""
        self.service_config = SERVICE_EQUIVALENCY_CONFIG
        self.feature_config = FEATURE_MAPPING_CONFIG
        self._load_service_mappings()
    
    def _load_service_mappings(self):
        """Load and validate service mapping configurations."""
        logger.info("Loading service mapping configurations")
        
        # Validate configuration completeness
        required_providers = {"aws", "gcp", "azure"}
        
        for category, mappings in self.service_config.items():
            for service, provider_mappings in mappings.items():
                available_providers = set(provider_mappings.keys())
                missing_providers = required_providers - available_providers
                
                if missing_providers:
                    logger.warning(
                        f"Service {service} in {category} missing mappings for: {missing_providers}"
                    )
    
    def map_compute_services(self, compute_spec: ComputeSpec) -> Dict[str, List[EquivalentService]]:
        """
        Map compute services across providers based on workload requirements.
        
        Args:
            compute_spec: Compute workload specification
            
        Returns:
            Dict mapping provider names to lists of equivalent services
        """
        logger.info(f"Mapping compute services for spec: {compute_spec}")
        
        mappings = {}
        
        # Get base service mappings
        compute_mappings = self.service_config.get("compute", {})
        
        # Determine source service type based on requirements
        source_service = self._determine_compute_service_type(compute_spec)
        
        if source_service not in compute_mappings:
            logger.warning(f"No mappings found for compute service: {source_service}")
            return mappings
        
        provider_services = compute_mappings[source_service]
        
        for provider, service_list in provider_services.items():
            equivalent_services = []
            
            for service_info in service_list:
                equivalent_service = self._create_equivalent_service(
                    provider=provider,
                    service_info=service_info,
                    workload_spec=compute_spec,
                    category=ServiceCategory.COMPUTE
                )
                equivalent_services.append(equivalent_service)
            
            # Sort by confidence score
            equivalent_services.sort(key=lambda x: x.confidence_score, reverse=True)
            mappings[provider] = equivalent_services
        
        return mappings
    
    def map_storage_services(self, storage_spec: StorageSpec) -> Dict[str, List[EquivalentService]]:
        """
        Map storage services across providers based on storage requirements.
        
        Args:
            storage_spec: Storage workload specification
            
        Returns:
            Dict mapping provider names to lists of equivalent services
        """
        logger.info(f"Mapping storage services for spec: {storage_spec}")
        
        mappings = {}
        
        # Get storage service mappings
        storage_mappings = self.service_config.get("storage", {})
        
        # Determine storage service type
        source_service = self._determine_storage_service_type(storage_spec)
        
        if source_service not in storage_mappings:
            logger.warning(f"No mappings found for storage service: {source_service}")
            return mappings
        
        provider_services = storage_mappings[source_service]
        
        for provider, service_list in provider_services.items():
            equivalent_services = []
            
            for service_info in service_list:
                equivalent_service = self._create_equivalent_service(
                    provider=provider,
                    service_info=service_info,
                    workload_spec=storage_spec,
                    category=ServiceCategory.STORAGE
                )
                equivalent_services.append(equivalent_service)
            
            equivalent_services.sort(key=lambda x: x.confidence_score, reverse=True)
            mappings[provider] = equivalent_services
        
        return mappings
    
    def map_network_services(self, network_spec: NetworkSpec) -> Dict[str, List[EquivalentService]]:
        """
        Map network services across providers based on networking requirements.
        
        Args:
            network_spec: Network workload specification
            
        Returns:
            Dict mapping provider names to lists of equivalent services
        """
        logger.info(f"Mapping network services for spec: {network_spec}")
        
        mappings = {}
        
        # Get network service mappings
        network_mappings = self.service_config.get("network", {})
        
        # Map each network component
        for component in network_spec.components:
            # Handle both Pydantic model and dict formats
            if hasattr(component, 'service_type'):
                service_type = component.service_type.value if hasattr(component.service_type, 'value') else str(component.service_type)
            else:
                service_type = component.get('service_type', 'unknown')
            
            if service_type not in network_mappings:
                logger.warning(f"No mappings found for network service: {service_type}")
                continue
            
            provider_services = network_mappings[service_type]
            
            for provider, service_list in provider_services.items():
                if provider not in mappings:
                    mappings[provider] = []
                
                for service_info in service_list:
                    equivalent_service = self._create_equivalent_service(
                        provider=provider,
                        service_info=service_info,
                        workload_spec=component,
                        category=ServiceCategory.NETWORK
                    )
                    mappings[provider].append(equivalent_service)
        
        # Sort each provider's services by confidence
        for provider in mappings:
            mappings[provider].sort(key=lambda x: x.confidence_score, reverse=True)
        
        return mappings
    
    def analyze_feature_parity(self, service_mappings: Dict[str, List[EquivalentService]]) -> FeatureParityMatrix:
        """
        Analyze feature parity across equivalent services.
        
        Args:
            service_mappings: Service mappings from map_*_services methods
            
        Returns:
            FeatureParityMatrix with detailed parity analysis
        """
        logger.info("Analyzing feature parity across service mappings")
        
        if not service_mappings:
            return FeatureParityMatrix(
                reference_service="",
                equivalent_services=[],
                feature_comparison={},
                missing_features={},
                additional_features={},
                overall_parity_score=0.0
            )
        
        # Use the first provider's first service as reference
        reference_provider = list(service_mappings.keys())[0]
        reference_service = service_mappings[reference_provider][0].service_name
        
        # Collect all equivalent services
        all_equivalent_services = []
        for provider_services in service_mappings.values():
            all_equivalent_services.extend(provider_services)
        
        # Get feature set for reference service
        reference_features = self._get_service_features(reference_service, reference_provider)
        
        # Build feature comparison matrix
        feature_comparison = {}
        missing_features = {}
        additional_features = {}
        
        for feature in reference_features:
            feature_comparison[feature] = {}
            
            for service in all_equivalent_services:
                service_features = self._get_service_features(service.service_name, service.provider)
                feature_comparison[feature][service.provider] = feature in service_features
                
                # Track missing features
                if feature not in service_features:
                    if service.provider not in missing_features:
                        missing_features[service.provider] = []
                    missing_features[service.provider].append(feature)
        
        # Find additional features not in reference
        for service in all_equivalent_services:
            service_features = self._get_service_features(service.service_name, service.provider)
            additional = set(service_features) - set(reference_features)
            
            if additional:
                additional_features[service.provider] = list(additional)
        
        # Calculate overall parity score
        total_comparisons = len(reference_features) * len(all_equivalent_services)
        matching_features = sum(
            1 for feature_providers in feature_comparison.values()
            for has_feature in feature_providers.values()
            if has_feature
        )
        
        overall_parity_score = matching_features / total_comparisons if total_comparisons > 0 else 0.0
        
        return FeatureParityMatrix(
            reference_service=reference_service,
            equivalent_services=all_equivalent_services,
            feature_comparison=feature_comparison,
            missing_features=missing_features,
            additional_features=additional_features,
            overall_parity_score=overall_parity_score
        )
    
    def get_equivalent_services(self, provider: str, service: str) -> List[EquivalentService]:
        """
        Get equivalent services for a specific provider service.
        
        Args:
            provider: Source provider name
            service: Source service name
            
        Returns:
            List of equivalent services in other providers
        """
        logger.info(f"Finding equivalents for {provider}:{service}")
        
        equivalent_services = []
        
        # Search across all categories
        for category, mappings in self.service_config.items():
            for source_service, provider_mappings in mappings.items():
                
                # Check if the service exists in the source provider
                if provider in provider_mappings:
                    provider_services = provider_mappings[provider]
                    
                    # Check if our service is in this mapping
                    service_names = [s.get("name", s) if isinstance(s, dict) else s for s in provider_services]
                    
                    if service in service_names:
                        # Found the service, now get equivalents from other providers
                        for other_provider, other_services in provider_mappings.items():
                            if other_provider != provider:
                                for service_info in other_services:
                                    equivalent_service = self._create_equivalent_service(
                                        provider=other_provider,
                                        service_info=service_info,
                                        workload_spec=None,
                                        category=ServiceCategory(category)
                                    )
                                    equivalent_services.append(equivalent_service)
        
        # Sort by confidence score
        equivalent_services.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return equivalent_services
    
    def _determine_compute_service_type(self, compute_spec: ComputeSpec) -> str:
        """Determine the appropriate compute service type based on specifications."""
        
        # Check for container requirements
        if hasattr(compute_spec, 'container_requirements') and compute_spec.container_requirements:
            if compute_spec.container_requirements.get('orchestration') == 'kubernetes':
                return "kubernetes"
            else:
                return "container_service"
        
        # Check for serverless requirements
        if hasattr(compute_spec, 'execution_model') and compute_spec.execution_model == 'serverless':
            return "serverless_compute"
        
        # Check for high-performance computing
        if compute_spec.vcpus > 32 or compute_spec.memory_gb > 128:
            return "high_performance_compute"
        
        # Check for GPU requirements
        if hasattr(compute_spec, 'gpu_requirements') and compute_spec.gpu_requirements:
            return "gpu_compute"
        
        # Default to virtual machines
        return "virtual_machines"
    
    def _determine_storage_service_type(self, storage_spec: StorageSpec) -> str:
        """Determine the appropriate storage service type based on specifications."""
        
        # Handle both string and enum storage types
        storage_type = storage_spec.storage_type
        if hasattr(storage_type, 'value'):
            storage_type = storage_type.value
        
        if storage_type == 'object':
            # Check for archival requirements
            if hasattr(storage_spec, 'access_pattern') and storage_spec.access_pattern == 'archive':
                return "object_storage_archive"
            else:
                return "object_storage"
        
        elif storage_type == 'block':
            # Check for high IOPS requirements
            if hasattr(storage_spec, 'iops_requirement') and storage_spec.iops_requirement and storage_spec.iops_requirement > 10000:
                return "high_performance_block"
            else:
                return "block_storage"
        
        elif storage_type == 'file':
            return "file_storage"
        
        elif storage_type == 'database':
            return "managed_database_storage"
        
        else:
            return "block_storage"  # Default
    
    def _create_equivalent_service(
        self,
        provider: str,
        service_info: Any,
        workload_spec: Optional[Any],
        category: ServiceCategory
    ) -> EquivalentService:
        """Create an EquivalentService object from service information."""
        
        # Handle both string and dict service info
        if isinstance(service_info, str):
            service_name = service_info
            service_type = service_info
            confidence_score = 0.8  # Default confidence
            limitations = []
            additional_features = []
            migration_complexity = "medium"
        else:
            service_name = service_info.get("name", "unknown")
            service_type = service_info.get("type", service_name)
            confidence_score = service_info.get("confidence", 0.8)
            limitations = service_info.get("limitations", [])
            additional_features = service_info.get("additional_features", [])
            migration_complexity = service_info.get("migration_complexity", "medium")
        
        # Calculate feature parity score
        feature_parity_score = self._calculate_feature_parity_score(
            service_name, provider, workload_spec
        )
        
        # Calculate performance ratio (placeholder - would use real benchmarks)
        performance_ratio = service_info.get("performance_ratio", 1.0) if isinstance(service_info, dict) else 1.0
        
        # Calculate cost efficiency score (placeholder)
        cost_efficiency_score = service_info.get("cost_efficiency", 0.7) if isinstance(service_info, dict) else 0.7
        
        return EquivalentService(
            provider=provider,
            service_name=service_name,
            service_type=service_type,
            confidence_score=confidence_score,
            feature_parity_score=feature_parity_score,
            performance_ratio=performance_ratio,
            cost_efficiency_score=cost_efficiency_score,
            limitations=limitations,
            additional_features=additional_features,
            migration_complexity=migration_complexity
        )
    
    def _calculate_feature_parity_score(
        self,
        service_name: str,
        provider: str,
        workload_spec: Optional[Any]
    ) -> float:
        """Calculate feature parity score for a service."""
        
        # Get service features
        service_features = self._get_service_features(service_name, provider)
        
        if not workload_spec:
            return 0.8  # Default score when no workload spec available
        
        # Get required features from workload spec
        required_features = self._extract_required_features(workload_spec)
        
        if not required_features:
            return 0.8  # Default score when no specific requirements
        
        # Calculate how many required features are supported
        supported_features = set(service_features) & set(required_features)
        parity_score = len(supported_features) / len(required_features)
        
        return min(parity_score, 1.0)
    
    def _get_service_features(self, service_name: str, provider: str) -> List[str]:
        """Get the list of features supported by a service."""
        
        # Look up features in configuration
        provider_features = self.feature_config.get(provider, {})
        service_features = provider_features.get(service_name, [])
        
        return service_features
    
    def _extract_required_features(self, workload_spec: Any) -> List[str]:
        """Extract required features from workload specification."""
        
        required_features = []
        
        # Extract features based on workload spec type and attributes
        if hasattr(workload_spec, 'required_features'):
            required_features.extend(workload_spec.required_features)
        
        if hasattr(workload_spec, 'performance_requirements'):
            perf_req = workload_spec.performance_requirements
            if perf_req.get('high_availability'):
                required_features.append('high_availability')
            if perf_req.get('auto_scaling'):
                required_features.append('auto_scaling')
            if perf_req.get('load_balancing'):
                required_features.append('load_balancing')
        
        if hasattr(workload_spec, 'security_requirements'):
            sec_req = workload_spec.security_requirements
            if sec_req.get('encryption_at_rest'):
                required_features.append('encryption_at_rest')
            if sec_req.get('encryption_in_transit'):
                required_features.append('encryption_in_transit')
            if sec_req.get('network_isolation'):
                required_features.append('network_isolation')
        
        return required_features
    
    def get_service_mapping_summary(self, workload_spec: WorkloadSpec) -> Dict[str, ServiceMapping]:
        """
        Get a complete service mapping summary for a workload.
        
        Args:
            workload_spec: Complete workload specification
            
        Returns:
            Dict mapping service names to ServiceMapping objects
        """
        logger.info("Generating complete service mapping summary")
        
        mappings = {}
        
        # Map compute services
        if workload_spec.compute:
            compute_mappings = self.map_compute_services(workload_spec.compute)
            if compute_mappings:
                mappings["compute"] = ServiceMapping(
                    source_service="compute",
                    category=ServiceCategory.COMPUTE,
                    mappings=compute_mappings,
                    recommended_mapping=self._get_recommended_mappings(compute_mappings),
                    mapping_confidence=self._calculate_mapping_confidence(compute_mappings)
                )
        
        # Map storage services
        if workload_spec.storage:
            for i, storage in enumerate(workload_spec.storage):
                storage_mappings = self.map_storage_services(storage)
                if storage_mappings:
                    mappings[f"storage_{i}"] = ServiceMapping(
                        source_service=f"storage_{i}",
                        category=ServiceCategory.STORAGE,
                        mappings=storage_mappings,
                        recommended_mapping=self._get_recommended_mappings(storage_mappings),
                        mapping_confidence=self._calculate_mapping_confidence(storage_mappings)
                    )
        
        # Map network services
        if workload_spec.network:
            network_mappings = self.map_network_services(workload_spec.network)
            if network_mappings:
                mappings["network"] = ServiceMapping(
                    source_service="network",
                    category=ServiceCategory.NETWORK,
                    mappings=network_mappings,
                    recommended_mapping=self._get_recommended_mappings(network_mappings),
                    mapping_confidence=self._calculate_mapping_confidence(network_mappings)
                )
        
        return mappings
    
    def _get_recommended_mappings(self, service_mappings: Dict[str, List[EquivalentService]]) -> Dict[str, EquivalentService]:
        """Get the recommended service for each provider."""
        
        recommended = {}
        
        for provider, services in service_mappings.items():
            if services:
                # Recommend the service with highest combined score
                best_service = max(
                    services,
                    key=lambda s: (s.confidence_score + s.feature_parity_score + s.cost_efficiency_score) / 3
                )
                recommended[provider] = best_service
        
        return recommended
    
    def _calculate_mapping_confidence(self, service_mappings: Dict[str, List[EquivalentService]]) -> float:
        """Calculate overall confidence in the service mappings."""
        
        if not service_mappings:
            return 0.0
        
        total_confidence = 0.0
        total_services = 0
        
        for services in service_mappings.values():
            for service in services:
                total_confidence += service.confidence_score
                total_services += 1
        
        return total_confidence / total_services if total_services > 0 else 0.0