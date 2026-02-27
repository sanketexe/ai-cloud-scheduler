"""
Cost Factors Configuration

Defines cost multipliers, overhead percentages, and hidden cost factors
for accurate Total Cost of Ownership calculations across cloud providers.
"""

from decimal import Decimal
from typing import Dict, List, Any


class CostFactors:
    """
    Configuration class for cost calculation factors and multipliers.
    
    Contains industry-standard percentages and multipliers for:
    - Support costs
    - Operational overhead
    - Compliance costs
    - Hidden costs
    - Regional adjustments
    """
    
    # Support Cost Percentages (as percentage of base infrastructure cost)
    SUPPORT_COST_PERCENTAGES = {
        "aws": {
            "basic": Decimal("0.03"),      # 3% - Basic support
            "developer": Decimal("0.03"),   # 3% - Developer support (min $29/month)
            "business": Decimal("0.10"),    # 10% - Business support (min $100/month)
            "enterprise": Decimal("0.15"),  # 15% - Enterprise support (min $15,000/month)
            "none": Decimal("0.00")         # No support
        },
        "gcp": {
            "basic": Decimal("0.00"),       # Free basic support
            "standard": Decimal("0.04"),    # 4% - Standard support (min $150/month)
            "enhanced": Decimal("0.08"),    # 8% - Enhanced support (min $500/month)
            "premium": Decimal("0.12"),     # 12% - Premium support (min $12,500/month)
            "none": Decimal("0.00")
        },
        "azure": {
            "basic": Decimal("0.00"),       # Free basic support
            "developer": Decimal("0.029"),  # $29/month flat rate
            "standard": Decimal("0.10"),    # 10% - Standard support (min $100/month)
            "professional": Decimal("0.10"), # 10% - Professional Direct (min $1,000/month)
            "premier": Decimal("0.15"),     # 15% - Premier support (custom pricing)
            "none": Decimal("0.00")
        }
    }
    
    # Operational Overhead Factors (as percentage of base infrastructure cost)
    OPERATIONAL_OVERHEAD_FACTORS = {
        "monitoring_and_alerting": Decimal("0.05"),     # 5% - Monitoring tools and services
        "backup_and_disaster_recovery": Decimal("0.15"), # 15% - Backup storage and DR setup
        "security_and_compliance": Decimal("0.08"),     # 8% - Security tools and auditing
        "devops_tooling": Decimal("0.06"),             # 6% - CI/CD, deployment tools
        "network_and_connectivity": Decimal("0.04"),    # 4% - VPN, direct connect, etc.
        "training_and_certification": Decimal("0.03"),  # 3% - Staff training costs
        "third_party_tools": Decimal("0.07"),          # 7% - Monitoring, security, management tools
        "professional_services": Decimal("0.10"),      # 10% - Consulting and implementation
        "management_overhead": Decimal("0.12")          # 12% - Administrative and management costs
    }
    
    # Compliance Cost Factors (annual costs in USD)
    COMPLIANCE_COST_FACTORS = {
        "sox": {
            "audit_costs": Decimal("50000"),        # SOX compliance auditing
            "tooling": Decimal("25000"),            # Compliance monitoring tools
            "staff_training": Decimal("15000")      # Staff training and certification
        },
        "pci_dss": {
            "audit_costs": Decimal("30000"),        # PCI DSS assessment
            "tooling": Decimal("20000"),            # Security scanning and monitoring
            "staff_training": Decimal("10000")      # Security training
        },
        "hipaa": {
            "audit_costs": Decimal("40000"),        # HIPAA compliance assessment
            "tooling": Decimal("35000"),            # Healthcare security tools
            "staff_training": Decimal("12000")      # Healthcare compliance training
        },
        "gdpr": {
            "audit_costs": Decimal("25000"),        # GDPR compliance assessment
            "tooling": Decimal("15000"),            # Data protection tools
            "staff_training": Decimal("8000")       # Privacy training
        },
        "iso_27001": {
            "audit_costs": Decimal("35000"),        # ISO 27001 certification
            "tooling": Decimal("30000"),            # Information security tools
            "staff_training": Decimal("15000")      # Security management training
        },
        "fedramp": {
            "audit_costs": Decimal("100000"),       # FedRAMP authorization
            "tooling": Decimal("75000"),            # Government security tools
            "staff_training": Decimal("25000")      # Federal compliance training
        }
    }
    
    # Hidden Cost Multipliers (as percentage of base cost)
    HIDDEN_COST_MULTIPLIERS = {
        "data_egress": {
            "aws": Decimal("0.12"),     # 12% average data egress costs
            "gcp": Decimal("0.10"),     # 10% average data egress costs
            "azure": Decimal("0.11")    # 11% average data egress costs
        },
        "api_calls": {
            "aws": Decimal("0.02"),     # 2% API call costs
            "gcp": Decimal("0.015"),    # 1.5% API call costs
            "azure": Decimal("0.018")   # 1.8% API call costs
        },
        "storage_operations": {
            "aws": Decimal("0.03"),     # 3% storage operation costs
            "gcp": Decimal("0.025"),    # 2.5% storage operation costs
            "azure": Decimal("0.028")   # 2.8% storage operation costs
        },
        "cross_region_traffic": {
            "aws": Decimal("0.08"),     # 8% cross-region data transfer
            "gcp": Decimal("0.06"),     # 6% cross-region data transfer
            "azure": Decimal("0.07")    # 7% cross-region data transfer
        }
    }
    
    # Regional Cost Multipliers (relative to US East/Central)
    REGIONAL_COST_MULTIPLIERS = {
        "aws": {
            "us-east-1": Decimal("1.00"),      # US East (N. Virginia) - baseline
            "us-east-2": Decimal("1.00"),      # US East (Ohio)
            "us-west-1": Decimal("1.05"),      # US West (N. California)
            "us-west-2": Decimal("1.02"),      # US West (Oregon)
            "eu-west-1": Decimal("1.08"),      # Europe (Ireland)
            "eu-west-2": Decimal("1.10"),      # Europe (London)
            "eu-central-1": Decimal("1.09"),   # Europe (Frankfurt)
            "ap-southeast-1": Decimal("1.12"), # Asia Pacific (Singapore)
            "ap-southeast-2": Decimal("1.15"), # Asia Pacific (Sydney)
            "ap-northeast-1": Decimal("1.18"), # Asia Pacific (Tokyo)
            "ap-south-1": Decimal("1.08"),     # Asia Pacific (Mumbai)
            "sa-east-1": Decimal("1.25"),      # South America (São Paulo)
            "ca-central-1": Decimal("1.06")    # Canada (Central)
        },
        "gcp": {
            "us-central1": Decimal("1.00"),    # US Central - baseline
            "us-east1": Decimal("1.00"),       # US East
            "us-west1": Decimal("1.03"),       # US West
            "europe-west1": Decimal("1.08"),   # Europe (Belgium)
            "europe-west2": Decimal("1.10"),   # Europe (London)
            "europe-west3": Decimal("1.09"),   # Europe (Frankfurt)
            "asia-southeast1": Decimal("1.12"), # Asia (Singapore)
            "asia-east1": Decimal("1.15"),     # Asia (Taiwan)
            "asia-northeast1": Decimal("1.18"), # Asia (Tokyo)
            "southamerica-east1": Decimal("1.25") # South America (São Paulo)
        },
        "azure": {
            "eastus": Decimal("1.00"),         # East US - baseline
            "eastus2": Decimal("1.00"),        # East US 2
            "westus": Decimal("1.05"),         # West US
            "westus2": Decimal("1.02"),        # West US 2
            "northeurope": Decimal("1.08"),    # North Europe (Ireland)
            "westeurope": Decimal("1.09"),     # West Europe (Netherlands)
            "uksouth": Decimal("1.10"),        # UK South (London)
            "southeastasia": Decimal("1.12"),  # Southeast Asia (Singapore)
            "australiaeast": Decimal("1.15"),  # Australia East (Sydney)
            "japaneast": Decimal("1.18"),      # Japan East (Tokyo)
            "centralindia": Decimal("1.08"),   # Central India (Pune)
            "brazilsouth": Decimal("1.25")     # Brazil South (São Paulo)
        }
    }
    
    # Workload Complexity Multipliers
    WORKLOAD_COMPLEXITY_MULTIPLIERS = {
        "simple": {
            "operational_overhead": Decimal("0.15"),    # 15% overhead for simple workloads
            "management_complexity": Decimal("1.0")     # No additional complexity
        },
        "moderate": {
            "operational_overhead": Decimal("0.25"),    # 25% overhead for moderate workloads
            "management_complexity": Decimal("1.2")     # 20% management complexity increase
        },
        "complex": {
            "operational_overhead": Decimal("0.40"),    # 40% overhead for complex workloads
            "management_complexity": Decimal("1.5")     # 50% management complexity increase
        },
        "enterprise": {
            "operational_overhead": Decimal("0.60"),    # 60% overhead for enterprise workloads
            "management_complexity": Decimal("2.0")     # 100% management complexity increase
        }
    }
    
    # Data Transfer Cost Rates (per GB in USD)
    DATA_TRANSFER_RATES = {
        "aws": {
            "outbound_internet": Decimal("0.09"),       # First 10TB per month
            "outbound_internet_next_40tb": Decimal("0.085"), # Next 40TB per month
            "outbound_internet_next_100tb": Decimal("0.07"),  # Next 100TB per month
            "cross_region": Decimal("0.02"),            # Cross-region transfer
            "inbound": Decimal("0.00")                  # Inbound is free
        },
        "gcp": {
            "outbound_internet": Decimal("0.12"),       # First 1TB per month free, then $0.12/GB
            "cross_region": Decimal("0.01"),            # Cross-region transfer
            "inbound": Decimal("0.00")                  # Inbound is free
        },
        "azure": {
            "outbound_internet": Decimal("0.087"),      # First 5GB free, then $0.087/GB
            "cross_region": Decimal("0.02"),            # Cross-region transfer
            "inbound": Decimal("0.00")                  # Inbound is free
        }
    }
    
    # Reserved Instance Discount Rates
    RESERVED_INSTANCE_DISCOUNTS = {
        "aws": {
            "1_year_no_upfront": Decimal("0.20"),       # 20% discount
            "1_year_partial_upfront": Decimal("0.25"),  # 25% discount
            "1_year_all_upfront": Decimal("0.30"),      # 30% discount
            "3_year_no_upfront": Decimal("0.35"),       # 35% discount
            "3_year_partial_upfront": Decimal("0.45"),  # 45% discount
            "3_year_all_upfront": Decimal("0.55")       # 55% discount
        },
        "gcp": {
            "1_year": Decimal("0.25"),                  # 25% discount for 1-year commitment
            "3_year": Decimal("0.50")                   # 50% discount for 3-year commitment
        },
        "azure": {
            "1_year": Decimal("0.20"),                  # 20% discount for 1-year reservation
            "3_year": Decimal("0.40")                   # 40% discount for 3-year reservation
        }
    }
    
    # Growth Rate Assumptions (annual growth percentages)
    DEFAULT_GROWTH_ASSUMPTIONS = {
        "compute_growth": Decimal("0.15"),      # 15% annual compute growth
        "storage_growth": Decimal("0.25"),      # 25% annual storage growth
        "network_growth": Decimal("0.20"),      # 20% annual network growth
        "user_growth": Decimal("0.30"),         # 30% annual user growth
        "data_growth": Decimal("0.40")          # 40% annual data growth
    }
    
    # Minimum Support Costs (monthly minimums in USD)
    MINIMUM_SUPPORT_COSTS = {
        "aws": {
            "developer": Decimal("29"),
            "business": Decimal("100"),
            "enterprise": Decimal("15000")
        },
        "gcp": {
            "standard": Decimal("150"),
            "enhanced": Decimal("500"),
            "premium": Decimal("12500")
        },
        "azure": {
            "developer": Decimal("29"),
            "standard": Decimal("100"),
            "professional": Decimal("1000")
        }
    }
    
    @classmethod
    def get_support_cost_percentage(cls, provider: str, support_level: str) -> Decimal:
        """Get support cost percentage for a provider and support level."""
        return cls.SUPPORT_COST_PERCENTAGES.get(provider, {}).get(support_level, Decimal("0.10"))
    
    @classmethod
    def get_operational_overhead_total(cls, workload_complexity: str = "moderate") -> Decimal:
        """Calculate total operational overhead percentage."""
        base_overhead = sum(cls.OPERATIONAL_OVERHEAD_FACTORS.values())
        complexity_multiplier = cls.WORKLOAD_COMPLEXITY_MULTIPLIERS.get(
            workload_complexity, cls.WORKLOAD_COMPLEXITY_MULTIPLIERS["moderate"]
        )["operational_overhead"]
        
        return base_overhead * complexity_multiplier
    
    @classmethod
    def get_compliance_cost_total(cls, compliance_requirements: List[str]) -> Decimal:
        """Calculate total annual compliance costs."""
        total_cost = Decimal("0")
        
        for requirement in compliance_requirements:
            requirement_costs = cls.COMPLIANCE_COST_FACTORS.get(requirement.lower(), {})
            total_cost += sum(requirement_costs.values())
        
        return total_cost
    
    @classmethod
    def get_regional_multiplier(cls, provider: str, region: str) -> Decimal:
        """Get regional cost multiplier for a provider and region."""
        return cls.REGIONAL_COST_MULTIPLIERS.get(provider, {}).get(region, Decimal("1.10"))
    
    @classmethod
    def get_hidden_cost_multiplier(cls, provider: str, cost_type: str) -> Decimal:
        """Get hidden cost multiplier for a specific cost type."""
        return cls.HIDDEN_COST_MULTIPLIERS.get(cost_type, {}).get(provider, Decimal("0.05"))
    
    @classmethod
    def get_data_transfer_rate(cls, provider: str, transfer_type: str) -> Decimal:
        """Get data transfer rate for a provider and transfer type."""
        return cls.DATA_TRANSFER_RATES.get(provider, {}).get(transfer_type, Decimal("0.09"))
    
    @classmethod
    def get_reserved_instance_discount(cls, provider: str, term: str) -> Decimal:
        """Get reserved instance discount rate."""
        return cls.RESERVED_INSTANCE_DISCOUNTS.get(provider, {}).get(term, Decimal("0.25"))
    
    @classmethod
    def apply_growth_rate(cls, base_cost: Decimal, years: int, growth_rate: Decimal) -> Decimal:
        """Apply compound growth rate to a base cost over multiple years."""
        return base_cost * ((Decimal("1") + growth_rate) ** years)
    
    @classmethod
    def get_minimum_support_cost(cls, provider: str, support_level: str) -> Decimal:
        """Get minimum monthly support cost."""
        return cls.MINIMUM_SUPPORT_COSTS.get(provider, {}).get(support_level, Decimal("0"))