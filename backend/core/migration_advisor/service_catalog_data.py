"""
Cloud Provider Service Catalog Data

This module contains the actual service catalog data for AWS, GCP, and Azure,
including compute, storage, database, ML services, and service comparison mappings.
"""

from typing import Dict, List
from .provider_catalog import (
    CloudProvider, CloudProviderName, ServiceSpecification, ServiceCategory,
    ServiceComparison, RegionSpecification, PerformanceCapability
)


def create_aws_services() -> Dict[str, ServiceSpecification]:
    """Create AWS service catalog"""
    services = {}
    
    # Compute Services
    services["ec2"] = ServiceSpecification(
        service_id="ec2",
        service_name="Amazon EC2",
        category=ServiceCategory.COMPUTE,
        description="Scalable virtual servers in the cloud",
        features=["Auto Scaling", "Elastic Load Balancing", "Multiple instance types", "Spot instances"],
        use_cases=["Web applications", "Batch processing", "High-performance computing"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.99
    )
    
    services["lambda"] = ServiceSpecification(
        service_id="lambda",
        service_name="AWS Lambda",
        category=ServiceCategory.SERVERLESS,
        description="Run code without provisioning servers",
        features=["Event-driven", "Auto-scaling", "Pay per execution", "Multiple runtimes"],
        use_cases=["API backends", "Data processing", "Real-time file processing"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.95
    )
    
    services["ecs"] = ServiceSpecification(
        service_id="ecs",
        service_name="Amazon ECS",
        category=ServiceCategory.CONTAINERS,
        description="Container orchestration service",
        features=["Docker support", "Fargate integration", "Auto-scaling", "Load balancing"],
        use_cases=["Microservices", "Batch processing", "Machine learning"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.99
    )
    
    services["eks"] = ServiceSpecification(
        service_id="eks",
        service_name="Amazon EKS",
        category=ServiceCategory.CONTAINERS,
        description="Managed Kubernetes service",
        features=["Kubernetes compatibility", "Auto-scaling", "Multi-AZ", "IAM integration"],
        use_cases=["Kubernetes workloads", "Microservices", "Hybrid deployments"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.95
    )
    
    # Storage Services
    services["s3"] = ServiceSpecification(
        service_id="s3",
        service_name="Amazon S3",
        category=ServiceCategory.STORAGE,
        description="Object storage service",
        features=["11 9's durability", "Versioning", "Lifecycle policies", "Multiple storage classes"],
        use_cases=["Backup and archive", "Data lakes", "Static website hosting"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.9
    )
    
    services["ebs"] = ServiceSpecification(
        service_id="ebs",
        service_name="Amazon EBS",
        category=ServiceCategory.STORAGE,
        description="Block storage for EC2",
        features=["SSD and HDD options", "Snapshots", "Encryption", "High performance"],
        use_cases=["Database storage", "Boot volumes", "Enterprise applications"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.99
    )
    
    services["efs"] = ServiceSpecification(
        service_id="efs",
        service_name="Amazon EFS",
        category=ServiceCategory.STORAGE,
        description="Managed file storage for EC2",
        features=["NFS compatible", "Auto-scaling", "Multi-AZ", "Encryption"],
        use_cases=["Shared file storage", "Content management", "Web serving"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.99
    )
    
    # Database Services
    services["rds"] = ServiceSpecification(
        service_id="rds",
        service_name="Amazon RDS",
        category=ServiceCategory.DATABASE,
        description="Managed relational database service",
        features=["Multiple engines", "Automated backups", "Multi-AZ", "Read replicas"],
        use_cases=["Web applications", "E-commerce", "Mobile applications"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.95
    )
    
    services["dynamodb"] = ServiceSpecification(
        service_id="dynamodb",
        service_name="Amazon DynamoDB",
        category=ServiceCategory.DATABASE,
        description="Managed NoSQL database",
        features=["Single-digit millisecond latency", "Auto-scaling", "Global tables", "Streams"],
        use_cases=["Gaming", "IoT", "Mobile backends"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.99
    )
    
    services["aurora"] = ServiceSpecification(
        service_id="aurora",
        service_name="Amazon Aurora",
        category=ServiceCategory.DATABASE,
        description="MySQL and PostgreSQL-compatible relational database",
        features=["5x faster than MySQL", "Auto-scaling storage", "Multi-AZ", "Global database"],
        use_cases=["Enterprise applications", "SaaS applications", "Web applications"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.95
    )
    
    # Machine Learning Services
    services["sagemaker"] = ServiceSpecification(
        service_id="sagemaker",
        service_name="Amazon SageMaker",
        category=ServiceCategory.MACHINE_LEARNING,
        description="Build, train, and deploy machine learning models",
        features=["Built-in algorithms", "Jupyter notebooks", "Model training", "Model deployment"],
        use_cases=["Predictive analytics", "Computer vision", "Natural language processing"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.9
    )
    
    services["rekognition"] = ServiceSpecification(
        service_id="rekognition",
        service_name="Amazon Rekognition",
        category=ServiceCategory.MACHINE_LEARNING,
        description="Image and video analysis",
        features=["Face detection", "Object detection", "Text detection", "Content moderation"],
        use_cases=["Content moderation", "Security", "Media analysis"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.9
    )
    
    # Analytics Services
    services["redshift"] = ServiceSpecification(
        service_id="redshift",
        service_name="Amazon Redshift",
        category=ServiceCategory.ANALYTICS,
        description="Fast, scalable data warehouse",
        features=["Columnar storage", "Massively parallel processing", "SQL interface", "Data lake integration"],
        use_cases=["Business intelligence", "Data warehousing", "Analytics"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.9
    )
    
    services["emr"] = ServiceSpecification(
        service_id="emr",
        service_name="Amazon EMR",
        category=ServiceCategory.ANALYTICS,
        description="Managed Hadoop framework",
        features=["Spark", "Hadoop", "Presto", "Auto-scaling"],
        use_cases=["Big data processing", "Machine learning", "Data transformation"],
        regions_available=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        sla_percentage=99.9
    )
    
    return services


def create_gcp_services() -> Dict[str, ServiceSpecification]:
    """Create GCP service catalog"""
    services = {}
    
    # Compute Services
    services["compute_engine"] = ServiceSpecification(
        service_id="compute_engine",
        service_name="Google Compute Engine",
        category=ServiceCategory.COMPUTE,
        description="Virtual machines running in Google's data centers",
        features=["Custom machine types", "Preemptible VMs", "Live migration", "Auto-scaling"],
        use_cases=["Web applications", "Batch processing", "High-performance computing"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.99
    )
    
    services["cloud_functions"] = ServiceSpecification(
        service_id="cloud_functions",
        service_name="Google Cloud Functions",
        category=ServiceCategory.SERVERLESS,
        description="Event-driven serverless compute platform",
        features=["Event-driven", "Auto-scaling", "Multiple runtimes", "HTTP triggers"],
        use_cases=["API backends", "Data processing", "Webhooks"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.95
    )
    
    services["gke"] = ServiceSpecification(
        service_id="gke",
        service_name="Google Kubernetes Engine",
        category=ServiceCategory.CONTAINERS,
        description="Managed Kubernetes service",
        features=["Kubernetes compatibility", "Auto-scaling", "Auto-repair", "Multi-cluster"],
        use_cases=["Microservices", "Containerized applications", "Hybrid deployments"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.95
    )
    
    services["cloud_run"] = ServiceSpecification(
        service_id="cloud_run",
        service_name="Google Cloud Run",
        category=ServiceCategory.CONTAINERS,
        description="Fully managed serverless platform for containers",
        features=["Serverless", "Auto-scaling", "Pay per use", "Any language"],
        use_cases=["APIs", "Microservices", "Web applications"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.95
    )
    
    # Storage Services
    services["cloud_storage"] = ServiceSpecification(
        service_id="cloud_storage",
        service_name="Google Cloud Storage",
        category=ServiceCategory.STORAGE,
        description="Object storage for companies of all sizes",
        features=["11 9's durability", "Multiple storage classes", "Lifecycle management", "Versioning"],
        use_cases=["Backup and archive", "Data lakes", "Content delivery"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.95
    )
    
    services["persistent_disk"] = ServiceSpecification(
        service_id="persistent_disk",
        service_name="Google Persistent Disk",
        category=ServiceCategory.STORAGE,
        description="Block storage for Compute Engine",
        features=["SSD and HDD options", "Snapshots", "Encryption", "High performance"],
        use_cases=["Database storage", "Boot volumes", "Enterprise applications"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.99
    )
    
    services["filestore"] = ServiceSpecification(
        service_id="filestore",
        service_name="Google Filestore",
        category=ServiceCategory.STORAGE,
        description="Managed file storage for applications",
        features=["NFS compatible", "High performance", "Snapshots", "Encryption"],
        use_cases=["Shared file storage", "Content management", "Media processing"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.99
    )
    
    # Database Services
    services["cloud_sql"] = ServiceSpecification(
        service_id="cloud_sql",
        service_name="Google Cloud SQL",
        category=ServiceCategory.DATABASE,
        description="Managed MySQL, PostgreSQL, and SQL Server",
        features=["Automated backups", "High availability", "Read replicas", "Point-in-time recovery"],
        use_cases=["Web applications", "E-commerce", "Mobile applications"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.95
    )
    
    services["firestore"] = ServiceSpecification(
        service_id="firestore",
        service_name="Google Cloud Firestore",
        category=ServiceCategory.DATABASE,
        description="Serverless document database",
        features=["Real-time sync", "Offline support", "Auto-scaling", "ACID transactions"],
        use_cases=["Mobile apps", "Web apps", "Real-time applications"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.99
    )
    
    services["spanner"] = ServiceSpecification(
        service_id="spanner",
        service_name="Google Cloud Spanner",
        category=ServiceCategory.DATABASE,
        description="Globally distributed relational database",
        features=["Global consistency", "Horizontal scaling", "SQL interface", "99.999% availability"],
        use_cases=["Global applications", "Financial services", "Gaming"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.999
    )
    
    # Machine Learning Services
    services["vertex_ai"] = ServiceSpecification(
        service_id="vertex_ai",
        service_name="Google Vertex AI",
        category=ServiceCategory.MACHINE_LEARNING,
        description="Unified ML platform for building and deploying models",
        features=["AutoML", "Custom training", "Model deployment", "Feature store"],
        use_cases=["Predictive analytics", "Computer vision", "Natural language processing"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.9
    )
    
    services["vision_ai"] = ServiceSpecification(
        service_id="vision_ai",
        service_name="Google Vision AI",
        category=ServiceCategory.MACHINE_LEARNING,
        description="Image analysis and recognition",
        features=["Object detection", "Face detection", "OCR", "Content moderation"],
        use_cases=["Content moderation", "Document processing", "Visual search"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.9
    )
    
    # Analytics Services
    services["bigquery"] = ServiceSpecification(
        service_id="bigquery",
        service_name="Google BigQuery",
        category=ServiceCategory.ANALYTICS,
        description="Serverless, highly scalable data warehouse",
        features=["Serverless", "SQL interface", "Real-time analytics", "ML integration"],
        use_cases=["Business intelligence", "Data warehousing", "Analytics"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.99
    )
    
    services["dataproc"] = ServiceSpecification(
        service_id="dataproc",
        service_name="Google Cloud Dataproc",
        category=ServiceCategory.ANALYTICS,
        description="Managed Spark and Hadoop service",
        features=["Fast cluster creation", "Auto-scaling", "Integrated with GCP", "Cost-effective"],
        use_cases=["Big data processing", "Machine learning", "Data transformation"],
        regions_available=["us-central1", "us-west1", "europe-west1", "asia-southeast1"],
        sla_percentage=99.9
    )
    
    return services


def create_azure_services() -> Dict[str, ServiceSpecification]:
    """Create Azure service catalog"""
    services = {}
    
    # Compute Services
    services["virtual_machines"] = ServiceSpecification(
        service_id="virtual_machines",
        service_name="Azure Virtual Machines",
        category=ServiceCategory.COMPUTE,
        description="On-demand scalable computing resources",
        features=["Multiple VM sizes", "Spot VMs", "Auto-scaling", "Availability sets"],
        use_cases=["Web applications", "Batch processing", "High-performance computing"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.99
    )
    
    services["azure_functions"] = ServiceSpecification(
        service_id="azure_functions",
        service_name="Azure Functions",
        category=ServiceCategory.SERVERLESS,
        description="Event-driven serverless compute",
        features=["Event-driven", "Auto-scaling", "Multiple languages", "Durable functions"],
        use_cases=["API backends", "Data processing", "Scheduled tasks"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.95
    )
    
    services["aks"] = ServiceSpecification(
        service_id="aks",
        service_name="Azure Kubernetes Service",
        category=ServiceCategory.CONTAINERS,
        description="Managed Kubernetes service",
        features=["Kubernetes compatibility", "Auto-scaling", "Azure integration", "Virtual nodes"],
        use_cases=["Microservices", "Containerized applications", "DevOps"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.95
    )
    
    services["container_instances"] = ServiceSpecification(
        service_id="container_instances",
        service_name="Azure Container Instances",
        category=ServiceCategory.CONTAINERS,
        description="Run containers without managing servers",
        features=["Fast startup", "Per-second billing", "Custom sizes", "Virtual network"],
        use_cases=["Batch jobs", "Task automation", "Development/testing"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    # Storage Services
    services["blob_storage"] = ServiceSpecification(
        service_id="blob_storage",
        service_name="Azure Blob Storage",
        category=ServiceCategory.STORAGE,
        description="Object storage for the cloud",
        features=["Multiple access tiers", "Lifecycle management", "Versioning", "Immutable storage"],
        use_cases=["Backup and archive", "Data lakes", "Content delivery"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    services["managed_disks"] = ServiceSpecification(
        service_id="managed_disks",
        service_name="Azure Managed Disks",
        category=ServiceCategory.STORAGE,
        description="Block-level storage volumes for Azure VMs",
        features=["SSD and HDD options", "Snapshots", "Encryption", "High availability"],
        use_cases=["Database storage", "Boot volumes", "Enterprise applications"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.99
    )
    
    services["azure_files"] = ServiceSpecification(
        service_id="azure_files",
        service_name="Azure Files",
        category=ServiceCategory.STORAGE,
        description="Managed file shares in the cloud",
        features=["SMB protocol", "Azure AD integration", "Snapshots", "Encryption"],
        use_cases=["Shared file storage", "Lift and shift", "Cloud-native apps"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    # Database Services
    services["sql_database"] = ServiceSpecification(
        service_id="sql_database",
        service_name="Azure SQL Database",
        category=ServiceCategory.DATABASE,
        description="Managed relational SQL database",
        features=["Built-in intelligence", "Auto-scaling", "High availability", "Advanced security"],
        use_cases=["Web applications", "E-commerce", "SaaS applications"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.99
    )
    
    services["cosmos_db"] = ServiceSpecification(
        service_id="cosmos_db",
        service_name="Azure Cosmos DB",
        category=ServiceCategory.DATABASE,
        description="Globally distributed, multi-model database",
        features=["Global distribution", "Multi-model", "Low latency", "Multiple APIs"],
        use_cases=["IoT", "Gaming", "Mobile applications"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.999
    )
    
    services["postgresql"] = ServiceSpecification(
        service_id="postgresql",
        service_name="Azure Database for PostgreSQL",
        category=ServiceCategory.DATABASE,
        description="Managed PostgreSQL database service",
        features=["Built-in high availability", "Automated backups", "Scaling", "Security"],
        use_cases=["Web applications", "Analytics", "Geospatial applications"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.99
    )
    
    # Machine Learning Services
    services["machine_learning"] = ServiceSpecification(
        service_id="machine_learning",
        service_name="Azure Machine Learning",
        category=ServiceCategory.MACHINE_LEARNING,
        description="Enterprise-grade ML service",
        features=["AutoML", "Designer", "MLOps", "Responsible AI"],
        use_cases=["Predictive analytics", "Computer vision", "Natural language processing"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    services["cognitive_services"] = ServiceSpecification(
        service_id="cognitive_services",
        service_name="Azure Cognitive Services",
        category=ServiceCategory.MACHINE_LEARNING,
        description="AI services and cognitive APIs",
        features=["Vision", "Speech", "Language", "Decision"],
        use_cases=["Content moderation", "Chatbots", "Document processing"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    # Analytics Services
    services["synapse_analytics"] = ServiceSpecification(
        service_id="synapse_analytics",
        service_name="Azure Synapse Analytics",
        category=ServiceCategory.ANALYTICS,
        description="Limitless analytics service",
        features=["Unified experience", "Serverless and dedicated", "Data integration", "Power BI integration"],
        use_cases=["Business intelligence", "Data warehousing", "Big data analytics"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.9
    )
    
    services["databricks"] = ServiceSpecification(
        service_id="databricks",
        service_name="Azure Databricks",
        category=ServiceCategory.ANALYTICS,
        description="Apache Spark-based analytics platform",
        features=["Collaborative notebooks", "Auto-scaling", "Delta Lake", "MLflow integration"],
        use_cases=["Big data processing", "Machine learning", "Data engineering"],
        regions_available=["eastus", "westus", "westeurope", "southeastasia"],
        sla_percentage=99.95
    )
    
    return services


def create_service_comparisons() -> List[ServiceComparison]:
    """Create service comparison mappings across providers"""
    comparisons = []
    
    # Compute - Virtual Machines
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.COMPUTE,
        service_purpose="Virtual Machines",
        feature_comparison={
            "instance_types": {"aws": "100+", "gcp": "50+", "azure": "100+"},
            "spot_instances": {"aws": True, "gcp": True, "azure": True},
            "auto_scaling": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    # Compute - Serverless Functions
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.SERVERLESS,
        service_purpose="Serverless Functions",
        feature_comparison={
            "max_timeout": {"aws": "15 min", "gcp": "60 min", "azure": "Unlimited"},
            "languages": {"aws": "Multiple", "gcp": "Multiple", "azure": "Multiple"},
            "cold_start": {"aws": "Low", "gcp": "Medium", "azure": "Low"}
        }
    ))
    
    # Containers - Kubernetes
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.CONTAINERS,
        service_purpose="Managed Kubernetes",
        feature_comparison={
            "kubernetes_version": {"aws": "Latest", "gcp": "Latest", "azure": "Latest"},
            "auto_upgrade": {"aws": True, "gcp": True, "azure": True},
            "serverless_nodes": {"aws": "Fargate", "gcp": "Autopilot", "azure": "Virtual nodes"}
        }
    ))
    
    # Storage - Object Storage
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.STORAGE,
        service_purpose="Object Storage",
        feature_comparison={
            "durability": {"aws": "99.999999999%", "gcp": "99.999999999%", "azure": "99.999999999%"},
            "storage_classes": {"aws": "6", "gcp": "4", "azure": "4"},
            "lifecycle_management": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    # Database - Relational
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.DATABASE,
        service_purpose="Managed Relational Database",
        feature_comparison={
            "engines": {"aws": "6", "gcp": "3", "azure": "4"},
            "auto_scaling": {"aws": True, "gcp": True, "azure": True},
            "read_replicas": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    # Database - NoSQL
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.DATABASE,
        service_purpose="NoSQL Database",
        feature_comparison={
            "consistency_models": {"aws": "Eventually consistent", "gcp": "Strong", "azure": "Multiple"},
            "global_distribution": {"aws": True, "gcp": True, "azure": True},
            "auto_scaling": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    # Machine Learning - Platform
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.MACHINE_LEARNING,
        service_purpose="ML Platform",
        feature_comparison={
            "automl": {"aws": True, "gcp": True, "azure": True},
            "built_in_algorithms": {"aws": "18+", "gcp": "Multiple", "azure": "Multiple"},
            "model_deployment": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    # Analytics - Data Warehouse
    comparisons.append(ServiceComparison(
        service_category=ServiceCategory.ANALYTICS,
        service_purpose="Data Warehouse",
        feature_comparison={
            "architecture": {"aws": "Cluster-based", "gcp": "Serverless", "azure": "Hybrid"},
            "sql_interface": {"aws": True, "gcp": True, "azure": True},
            "ml_integration": {"aws": True, "gcp": True, "azure": True}
        }
    ))
    
    return comparisons



def build_aws_provider() -> CloudProvider:
    """Build complete AWS provider catalog"""
    services = create_aws_services()
    
    # Organize services by category
    service_categories = {}
    for service_id, service in services.items():
        category = service.category
        if category not in service_categories:
            service_categories[category] = []
        service_categories[category].append(service_id)
    
    # Define AWS regions
    regions = [
        RegionSpecification(
            region_id="us-east-1",
            region_name="US East (N. Virginia)",
            geographic_location="North America",
            availability_zones=6,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="us-west-2",
            region_name="US West (Oregon)",
            geographic_location="North America",
            availability_zones=4,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="eu-west-1",
            region_name="Europe (Ireland)",
            geographic_location="Europe",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["GDPR", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="ap-southeast-1",
            region_name="Asia Pacific (Singapore)",
            geographic_location="Asia Pacific",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        )
    ]
    
    # Performance capabilities
    performance = PerformanceCapability(
        max_compute_instances=None,  # No hard limit
        max_storage_capacity_tb=None,  # No hard limit
        max_network_bandwidth_gbps=100,
        gpu_availability=True,
        gpu_types=["NVIDIA A100", "NVIDIA V100", "NVIDIA T4"],
        specialized_compute=["Graviton", "Inferentia", "Trainium"],
        auto_scaling_capabilities=True,
        load_balancing_options=["ALB", "NLB", "CLB", "GWLB"]
    )
    
    return CloudProvider(
        provider_name=CloudProviderName.AWS,
        display_name="Amazon Web Services",
        description="Comprehensive cloud platform with 200+ services",
        headquarters="Seattle, WA, USA",
        founded_year=2006,
        services=services,
        service_categories=service_categories,
        regions=regions,
        total_regions=33,
        total_availability_zones=105,
        performance_capabilities=performance,
        market_share_percentage=32.0,
        enterprise_customers=1000000
    )


def build_gcp_provider() -> CloudProvider:
    """Build complete GCP provider catalog"""
    services = create_gcp_services()
    
    # Organize services by category
    service_categories = {}
    for service_id, service in services.items():
        category = service.category
        if category not in service_categories:
            service_categories[category] = []
        service_categories[category].append(service_id)
    
    # Define GCP regions
    regions = [
        RegionSpecification(
            region_id="us-central1",
            region_name="Iowa",
            geographic_location="North America",
            availability_zones=4,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="us-west1",
            region_name="Oregon",
            geographic_location="North America",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="europe-west1",
            region_name="Belgium",
            geographic_location="Europe",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["GDPR", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="asia-southeast1",
            region_name="Singapore",
            geographic_location="Asia Pacific",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        )
    ]
    
    # Performance capabilities
    performance = PerformanceCapability(
        max_compute_instances=None,  # No hard limit
        max_storage_capacity_tb=None,  # No hard limit
        max_network_bandwidth_gbps=100,
        gpu_availability=True,
        gpu_types=["NVIDIA A100", "NVIDIA V100", "NVIDIA T4", "NVIDIA P4"],
        specialized_compute=["TPU", "Custom machine types"],
        auto_scaling_capabilities=True,
        load_balancing_options=["Global LB", "Regional LB", "Internal LB"]
    )
    
    return CloudProvider(
        provider_name=CloudProviderName.GCP,
        display_name="Google Cloud Platform",
        description="Cloud platform with strong data analytics and ML capabilities",
        headquarters="Mountain View, CA, USA",
        founded_year=2008,
        services=services,
        service_categories=service_categories,
        regions=regions,
        total_regions=40,
        total_availability_zones=121,
        performance_capabilities=performance,
        market_share_percentage=11.0,
        enterprise_customers=150000
    )


def build_azure_provider() -> CloudProvider:
    """Build complete Azure provider catalog"""
    services = create_azure_services()
    
    # Organize services by category
    service_categories = {}
    for service_id, service in services.items():
        category = service.category
        if category not in service_categories:
            service_categories[category] = []
        service_categories[category].append(service_id)
    
    # Define Azure regions
    regions = [
        RegionSpecification(
            region_id="eastus",
            region_name="East US",
            geographic_location="North America",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="westus",
            region_name="West US",
            geographic_location="North America",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["HIPAA", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="westeurope",
            region_name="West Europe",
            geographic_location="Europe",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["GDPR", "PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        ),
        RegionSpecification(
            region_id="southeastasia",
            region_name="Southeast Asia",
            geographic_location="Asia Pacific",
            availability_zones=3,
            services_available=list(services.keys()),
            compliance_certifications=["PCI DSS", "SOC 2", "ISO 27001"],
            data_residency_compliant=True
        )
    ]
    
    # Performance capabilities
    performance = PerformanceCapability(
        max_compute_instances=None,  # No hard limit
        max_storage_capacity_tb=None,  # No hard limit
        max_network_bandwidth_gbps=100,
        gpu_availability=True,
        gpu_types=["NVIDIA A100", "NVIDIA V100", "AMD MI25"],
        specialized_compute=["FPGA", "HPC"],
        auto_scaling_capabilities=True,
        load_balancing_options=["Application Gateway", "Load Balancer", "Traffic Manager"]
    )
    
    return CloudProvider(
        provider_name=CloudProviderName.AZURE,
        display_name="Microsoft Azure",
        description="Enterprise cloud platform with strong hybrid capabilities",
        headquarters="Redmond, WA, USA",
        founded_year=2010,
        services=services,
        service_categories=service_categories,
        regions=regions,
        total_regions=60,
        total_availability_zones=180,
        performance_capabilities=performance,
        market_share_percentage=23.0,
        enterprise_customers=500000
    )


def initialize_provider_catalog() -> 'ProviderCatalog':
    """Initialize the complete provider catalog with all providers and comparisons"""
    from .provider_catalog import ProviderCatalog
    
    catalog = ProviderCatalog()
    
    # Add providers
    catalog.add_provider(build_aws_provider())
    catalog.add_provider(build_gcp_provider())
    catalog.add_provider(build_azure_provider())
    
    # Add service comparisons
    catalog.service_comparisons = create_service_comparisons()
    
    # Set AWS service references in comparisons
    aws_services = create_aws_services()
    gcp_services = create_gcp_services()
    azure_services = create_azure_services()
    
    for comparison in catalog.service_comparisons:
        if comparison.service_purpose == "Virtual Machines":
            comparison.aws_service = aws_services.get("ec2")
            comparison.gcp_service = gcp_services.get("compute_engine")
            comparison.azure_service = azure_services.get("virtual_machines")
        elif comparison.service_purpose == "Serverless Functions":
            comparison.aws_service = aws_services.get("lambda")
            comparison.gcp_service = gcp_services.get("cloud_functions")
            comparison.azure_service = azure_services.get("azure_functions")
        elif comparison.service_purpose == "Managed Kubernetes":
            comparison.aws_service = aws_services.get("eks")
            comparison.gcp_service = gcp_services.get("gke")
            comparison.azure_service = azure_services.get("aks")
        elif comparison.service_purpose == "Object Storage":
            comparison.aws_service = aws_services.get("s3")
            comparison.gcp_service = gcp_services.get("cloud_storage")
            comparison.azure_service = azure_services.get("blob_storage")
        elif comparison.service_purpose == "Managed Relational Database":
            comparison.aws_service = aws_services.get("rds")
            comparison.gcp_service = gcp_services.get("cloud_sql")
            comparison.azure_service = azure_services.get("sql_database")
        elif comparison.service_purpose == "NoSQL Database":
            comparison.aws_service = aws_services.get("dynamodb")
            comparison.gcp_service = gcp_services.get("firestore")
            comparison.azure_service = azure_services.get("cosmos_db")
        elif comparison.service_purpose == "ML Platform":
            comparison.aws_service = aws_services.get("sagemaker")
            comparison.gcp_service = gcp_services.get("vertex_ai")
            comparison.azure_service = azure_services.get("machine_learning")
        elif comparison.service_purpose == "Data Warehouse":
            comparison.aws_service = aws_services.get("redshift")
            comparison.gcp_service = gcp_services.get("bigquery")
            comparison.azure_service = azure_services.get("synapse_analytics")
    
    return catalog


# Global catalog instance
_catalog_instance = None


def get_provider_catalog() -> 'ProviderCatalog':
    """Get or create the global provider catalog instance"""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = initialize_provider_catalog()
    return _catalog_instance
