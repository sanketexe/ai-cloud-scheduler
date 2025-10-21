# cloud_provider_config.py
"""
Cloud Provider Configuration and Setup System

This module provides the core functionality for configuring cloud providers,
managing credentials, and establishing API connections for the FinOps platform.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudProviderType(Enum):
    """Supported cloud provider types"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    CUSTOM = "custom"


class CredentialType(Enum):
    """Types of credentials supported"""
    API_KEY = "api_key"
    SERVICE_ACCOUNT = "service_account"
    ACCESS_TOKEN = "access_token"
    USERNAME_PASSWORD = "username_password"


class ConnectionStatus(Enum):
    """Connection status for API validation"""
    NOT_TESTED = "not_tested"
    CONNECTED = "connected"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ProviderCredentials:
    """Secure credential storage for cloud providers"""
    credential_type: CredentialType
    credentials: Dict[str, str]  # Will be encrypted in production
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def needs_rotation(self, rotation_days: int = 90) -> bool:
        """Check if credentials need rotation"""
        if self.last_rotated:
            return datetime.now() > self.last_rotated + timedelta(days=rotation_days)
        return datetime.now() > self.created_at + timedelta(days=rotation_days)


@dataclass
class APIEndpoints:
    """API endpoints for different cloud provider services"""
    billing: str
    resources: str
    monitoring: str
    pricing: Optional[str] = None
    
    def validate_endpoints(self) -> bool:
        """Validate that all required endpoints are provided"""
        return bool(self.billing and self.resources and self.monitoring)


@dataclass
class ProviderConfig:
    """Complete configuration for a cloud provider"""
    provider_type: CloudProviderType
    provider_name: str
    credentials: ProviderCredentials
    api_endpoints: APIEndpoints
    regions: List[str] = field(default_factory=list)
    billing_account_id: Optional[str] = None
    connection_status: ConnectionStatus = ConnectionStatus.NOT_TESTED
    last_tested: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if provider configuration is valid"""
        return (
            self.provider_name and
            self.credentials and
            not self.credentials.is_expired() and
            self.api_endpoints.validate_endpoints()
        )


@dataclass
class ValidationResult:
    """Result of API connection validation"""
    success: bool
    provider_type: CloudProviderType
    tested_at: datetime
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    response_times: Dict[str, float] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add an error to the validation result"""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result"""
        self.warnings.append(warning)


@dataclass
class ConfigurationResult:
    """Result of provider configuration process"""
    success: bool
    provider_config: Optional[ProviderConfig]
    validation_result: Optional[ValidationResult]
    message: str
    created_at: datetime = field(default_factory=datetime.now)


class CloudProviderSelector:
    """Handles cloud provider selection and initial configuration"""
    
    SUPPORTED_PROVIDERS = {
        CloudProviderType.AWS: {
            "name": "Amazon Web Services",
            "description": "AWS cloud services with comprehensive billing and resource APIs",
            "required_credentials": ["access_key_id", "secret_access_key"],
            "optional_credentials": ["session_token", "region"],
            "default_endpoints": {
                "billing": "https://ce.{region}.amazonaws.com",
                "resources": "https://ec2.{region}.amazonaws.com",
                "monitoring": "https://monitoring.{region}.amazonaws.com",
                "pricing": "https://pricing.us-east-1.amazonaws.com"
            }
        },
        CloudProviderType.GCP: {
            "name": "Google Cloud Platform",
            "description": "GCP cloud services with billing and resource management APIs",
            "required_credentials": ["service_account_key"],
            "optional_credentials": ["project_id"],
            "default_endpoints": {
                "billing": "https://cloudbilling.googleapis.com",
                "resources": "https://compute.googleapis.com",
                "monitoring": "https://monitoring.googleapis.com"
            }
        },
        CloudProviderType.AZURE: {
            "name": "Microsoft Azure",
            "description": "Azure cloud services with cost management and resource APIs",
            "required_credentials": ["client_id", "client_secret", "tenant_id"],
            "optional_credentials": ["subscription_id"],
            "default_endpoints": {
                "billing": "https://management.azure.com",
                "resources": "https://management.azure.com",
                "monitoring": "https://management.azure.com"
            }
        },
        CloudProviderType.CUSTOM: {
            "name": "Custom Provider",
            "description": "Custom cloud provider with user-defined endpoints",
            "required_credentials": [],
            "optional_credentials": [],
            "default_endpoints": {}
        }
    }
    
    def get_supported_providers(self) -> Dict[CloudProviderType, Dict[str, Any]]:
        """Get list of supported cloud providers with their details"""
        return self.SUPPORTED_PROVIDERS.copy()
    
    def get_provider_requirements(self, provider_type: CloudProviderType) -> Dict[str, Any]:
        """Get requirements for a specific provider"""
        return self.SUPPORTED_PROVIDERS.get(provider_type, {})
    
    def validate_provider_selection(self, provider_type: CloudProviderType) -> bool:
        """Validate that the selected provider is supported"""
        return provider_type in self.SUPPORTED_PROVIDERS
    
    def create_default_endpoints(self, provider_type: CloudProviderType, 
                                region: str = "us-east-1") -> APIEndpoints:
        """Create default API endpoints for a provider"""
        provider_info = self.SUPPORTED_PROVIDERS.get(provider_type, {})
        default_endpoints = provider_info.get("default_endpoints", {})
        
        # Replace region placeholders
        endpoints = {}
        for service, url in default_endpoints.items():
            endpoints[service] = url.format(region=region)
        
        return APIEndpoints(
            billing=endpoints.get("billing", ""),
            resources=endpoints.get("resources", ""),
            monitoring=endpoints.get("monitoring", ""),
            pricing=endpoints.get("pricing")
        )


class CredentialValidator:
    """Validates and manages cloud provider credentials"""
    
    def validate_credentials(self, provider_type: CloudProviderType, 
                           credentials: Dict[str, str]) -> ValidationResult:
        """Validate credentials for a specific provider"""
        result = ValidationResult(
            success=True,
            provider_type=provider_type,
            tested_at=datetime.now()
        )
        
        provider_info = CloudProviderSelector.SUPPORTED_PROVIDERS.get(provider_type, {})
        required_creds = provider_info.get("required_credentials", [])
        
        # Check required credentials
        for required_cred in required_creds:
            if required_cred not in credentials or not credentials[required_cred]:
                result.add_error(f"Missing required credential: {required_cred}")
        
        # Provider-specific validation
        if provider_type == CloudProviderType.AWS:
            self._validate_aws_credentials(credentials, result)
        elif provider_type == CloudProviderType.GCP:
            self._validate_gcp_credentials(credentials, result)
        elif provider_type == CloudProviderType.AZURE:
            self._validate_azure_credentials(credentials, result)
        
        return result
    
    def _validate_aws_credentials(self, credentials: Dict[str, str], 
                                 result: ValidationResult):
        """Validate AWS-specific credentials"""
        access_key = credentials.get("access_key_id", "")
        secret_key = credentials.get("secret_access_key", "")
        
        if access_key and not access_key.startswith("AKIA"):
            result.add_warning("AWS Access Key ID should start with 'AKIA'")
        
        if len(secret_key) != 40:
            result.add_warning("AWS Secret Access Key should be 40 characters long")
    
    def _validate_gcp_credentials(self, credentials: Dict[str, str], 
                                 result: ValidationResult):
        """Validate GCP-specific credentials"""
        service_account_key = credentials.get("service_account_key", "")
        
        if service_account_key:
            try:
                # Try to parse as JSON
                key_data = json.loads(service_account_key)
                required_fields = ["type", "project_id", "private_key", "client_email"]
                for field in required_fields:
                    if field not in key_data:
                        result.add_error(f"Missing field in service account key: {field}")
            except json.JSONDecodeError:
                result.add_error("Service account key must be valid JSON")
    
    def _validate_azure_credentials(self, credentials: Dict[str, str], 
                                   result: ValidationResult):
        """Validate Azure-specific credentials"""
        client_id = credentials.get("client_id", "")
        tenant_id = credentials.get("tenant_id", "")
        
        # Basic UUID format validation
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        
        if client_id and not uuid_pattern.match(client_id):
            result.add_warning("Azure Client ID should be a valid UUID")
        
        if tenant_id and not uuid_pattern.match(tenant_id):
            result.add_warning("Azure Tenant ID should be a valid UUID")


class ProviderConfigurationInterface:
    """Main interface for cloud provider configuration"""
    
    def __init__(self):
        self.selector = CloudProviderSelector()
        self.validator = CredentialValidator()
        self.configurations: Dict[str, ProviderConfig] = {}
    
    def get_supported_providers(self) -> Dict[CloudProviderType, Dict[str, Any]]:
        """Get list of supported providers for UI display"""
        return self.selector.get_supported_providers()
    
    def configure_provider(self, provider_type: CloudProviderType, 
                          provider_name: str,
                          credentials: Dict[str, str],
                          regions: List[str] = None,
                          custom_endpoints: Dict[str, str] = None,
                          billing_account_id: str = None) -> ConfigurationResult:
        """Configure a new cloud provider"""
        
        logger.info(f"Configuring provider: {provider_name} ({provider_type.value})")
        
        # Validate provider type
        if not self.selector.validate_provider_selection(provider_type):
            return ConfigurationResult(
                success=False,
                provider_config=None,
                validation_result=None,
                message=f"Unsupported provider type: {provider_type.value}"
            )
        
        # Validate credentials
        credential_validation = self.validator.validate_credentials(provider_type, credentials)
        if not credential_validation.success:
            return ConfigurationResult(
                success=False,
                provider_config=None,
                validation_result=credential_validation,
                message=f"Credential validation failed: {', '.join(credential_validation.errors)}"
            )
        
        # Create provider credentials
        provider_credentials = ProviderCredentials(
            credential_type=self._determine_credential_type(provider_type),
            credentials=credentials  # In production, this would be encrypted
        )
        
        # Create API endpoints
        if custom_endpoints:
            api_endpoints = APIEndpoints(
                billing=custom_endpoints.get("billing", ""),
                resources=custom_endpoints.get("resources", ""),
                monitoring=custom_endpoints.get("monitoring", ""),
                pricing=custom_endpoints.get("pricing")
            )
        else:
            default_region = regions[0] if regions else "us-east-1"
            api_endpoints = self.selector.create_default_endpoints(provider_type, default_region)
        
        # Create provider configuration
        provider_config = ProviderConfig(
            provider_type=provider_type,
            provider_name=provider_name,
            credentials=provider_credentials,
            api_endpoints=api_endpoints,
            regions=regions or [],
            billing_account_id=billing_account_id
        )
        
        # Store configuration
        self.configurations[provider_name] = provider_config
        
        logger.info(f"Successfully configured provider: {provider_name}")
        
        return ConfigurationResult(
            success=True,
            provider_config=provider_config,
            validation_result=credential_validation,
            message=f"Provider {provider_name} configured successfully"
        )
    
    def _determine_credential_type(self, provider_type: CloudProviderType) -> CredentialType:
        """Determine the credential type based on provider"""
        if provider_type == CloudProviderType.AWS:
            return CredentialType.API_KEY
        elif provider_type == CloudProviderType.GCP:
            return CredentialType.SERVICE_ACCOUNT
        elif provider_type == CloudProviderType.AZURE:
            return CredentialType.SERVICE_ACCOUNT
        else:
            return CredentialType.API_KEY
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.configurations.get(provider_name)
    
    def list_configured_providers(self) -> List[str]:
        """List all configured provider names"""
        return list(self.configurations.keys())
    
    def remove_provider_config(self, provider_name: str) -> bool:
        """Remove a provider configuration"""
        if provider_name in self.configurations:
            del self.configurations[provider_name]
            logger.info(f"Removed provider configuration: {provider_name}")
            return True
        return False
    
    def update_provider_config(self, provider_name: str, 
                              updates: Dict[str, Any]) -> ConfigurationResult:
        """Update an existing provider configuration"""
        if provider_name not in self.configurations:
            return ConfigurationResult(
                success=False,
                provider_config=None,
                validation_result=None,
                message=f"Provider {provider_name} not found"
            )
        
        config = self.configurations[provider_name]
        
        # Update configuration fields
        if "regions" in updates:
            config.regions = updates["regions"]
        if "billing_account_id" in updates:
            config.billing_account_id = updates["billing_account_id"]
        if "metadata" in updates:
            config.metadata.update(updates["metadata"])
        
        logger.info(f"Updated provider configuration: {provider_name}")
        
        return ConfigurationResult(
            success=True,
            provider_config=config,
            validation_result=None,
            message=f"Provider {provider_name} updated successfully"
        )


# Example usage and testing
if __name__ == "__main__":
    # Initialize the configuration interface
    config_interface = ProviderConfigurationInterface()
    
    # Get supported providers
    providers = config_interface.get_supported_providers()
    print("Supported Providers:")
    for provider_type, info in providers.items():
        print(f"  {provider_type.value}: {info['name']}")
    
    # Configure AWS provider
    aws_credentials = {
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    }
    
    result = config_interface.configure_provider(
        provider_type=CloudProviderType.AWS,
        provider_name="production-aws",
        credentials=aws_credentials,
        regions=["us-east-1", "us-west-2"],
        billing_account_id="123456789012"
    )
    
    print(f"\nAWS Configuration Result: {result.message}")
    if result.validation_result:
        print(f"Validation Errors: {result.validation_result.errors}")
        print(f"Validation Warnings: {result.validation_result.warnings}")
    
    # List configured providers
    print(f"\nConfigured Providers: {config_interface.list_configured_providers()}")