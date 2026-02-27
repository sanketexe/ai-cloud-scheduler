"""
Service Equivalency Configuration

Comprehensive mapping of services across AWS, GCP, and Azure for cost comparison.
Includes feature parity information and migration complexity assessments.
"""

# Compute Service Mappings
COMPUTE_MAPPINGS = {
    "virtual_machines": {
        "aws": [
            {"name": "EC2", "type": "virtual_machine", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "Lightsail", "type": "simple_vm", "confidence": 0.8, "performance_ratio": 0.9}
        ],
        "gcp": [
            {"name": "Compute Engine", "type": "virtual_machine", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "Preemptible VMs", "type": "spot_vm", "confidence": 0.85, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Virtual Machines", "type": "virtual_machine", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "Spot VMs", "type": "spot_vm", "confidence": 0.85, "performance_ratio": 1.0}
        ]
    },
    
    "container_service": {
        "aws": [
            {"name": "ECS", "type": "container_orchestration", "confidence": 0.9, "performance_ratio": 1.0},
            {"name": "Fargate", "type": "serverless_container", "confidence": 0.85, "performance_ratio": 0.95}
        ],
        "gcp": [
            {"name": "Cloud Run", "type": "serverless_container", "confidence": 0.9, "performance_ratio": 1.0},
            {"name": "GKE Autopilot", "type": "managed_kubernetes", "confidence": 0.85, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Container Instances", "type": "serverless_container", "confidence": 0.85, "performance_ratio": 0.9},
            {"name": "Container Apps", "type": "serverless_container", "confidence": 0.8, "performance_ratio": 0.95}
        ]
    },
    
    "kubernetes": {
        "aws": [
            {"name": "EKS", "type": "managed_kubernetes", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "GKE", "type": "managed_kubernetes", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "AKS", "type": "managed_kubernetes", "confidence": 0.95, "performance_ratio": 1.0}
        ]
    },
    
    "serverless_compute": {
        "aws": [
            {"name": "Lambda", "type": "function_as_service", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "Cloud Functions", "type": "function_as_service", "confidence": 0.9, "performance_ratio": 0.95}
        ],
        "azure": [
            {"name": "Functions", "type": "function_as_service", "confidence": 0.9, "performance_ratio": 0.9}
        ]
    }
}

# Storage Service Mappings  
STORAGE_MAPPINGS = {
    "object_storage": {
        "aws": [
            {"name": "S3", "type": "object_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "Cloud Storage", "type": "object_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Blob Storage", "type": "object_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ]
    },
    
    "block_storage": {
        "aws": [
            {"name": "EBS", "type": "block_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "Persistent Disk", "type": "block_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Managed Disks", "type": "block_storage", "confidence": 0.95, "performance_ratio": 1.0}
        ]
    },
    
    "file_storage": {
        "aws": [
            {"name": "EFS", "type": "managed_file_storage", "confidence": 0.9, "performance_ratio": 1.0},
            {"name": "FSx", "type": "high_performance_file", "confidence": 0.85, "performance_ratio": 1.2}
        ],
        "gcp": [
            {"name": "Filestore", "type": "managed_file_storage", "confidence": 0.9, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Files", "type": "managed_file_storage", "confidence": 0.9, "performance_ratio": 1.0}
        ]
    },
    
    "object_storage_archive": {
        "aws": [
            {"name": "S3 Glacier", "type": "archive_storage", "confidence": 0.95, "performance_ratio": 0.1},
            {"name": "S3 Deep Archive", "type": "deep_archive", "confidence": 0.9, "performance_ratio": 0.05}
        ],
        "gcp": [
            {"name": "Cloud Storage Archive", "type": "archive_storage", "confidence": 0.9, "performance_ratio": 0.1},
            {"name": "Cloud Storage Coldline", "type": "cold_storage", "confidence": 0.85, "performance_ratio": 0.3}
        ],
        "azure": [
            {"name": "Blob Archive", "type": "archive_storage", "confidence": 0.9, "performance_ratio": 0.1},
            {"name": "Blob Cool", "type": "cool_storage", "confidence": 0.85, "performance_ratio": 0.5}
        ]
    }
}

# Network Service Mappings
NETWORK_MAPPINGS = {
    "load_balancer": {
        "aws": [
            {"name": "ALB", "type": "application_load_balancer", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "NLB", "type": "network_load_balancer", "confidence": 0.9, "performance_ratio": 1.1},
            {"name": "CLB", "type": "classic_load_balancer", "confidence": 0.7, "performance_ratio": 0.8}
        ],
        "gcp": [
            {"name": "Cloud Load Balancing", "type": "global_load_balancer", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "azure": [
            {"name": "Load Balancer", "type": "network_load_balancer", "confidence": 0.9, "performance_ratio": 1.0},
            {"name": "Application Gateway", "type": "application_load_balancer", "confidence": 0.85, "performance_ratio": 0.95}
        ]
    },
    
    "cdn": {
        "aws": [
            {"name": "CloudFront", "type": "content_delivery_network", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "Cloud CDN", "type": "content_delivery_network", "confidence": 0.9, "performance_ratio": 0.95}
        ],
        "azure": [
            {"name": "CDN", "type": "content_delivery_network", "confidence": 0.9, "performance_ratio": 0.95}
        ]
    },
    
    "api_gateway": {
        "aws": [
            {"name": "API Gateway", "type": "api_management", "confidence": 0.95, "performance_ratio": 1.0}
        ],
        "gcp": [
            {"name": "Cloud Endpoints", "type": "api_management", "confidence": 0.85, "performance_ratio": 0.9},
            {"name": "Apigee", "type": "enterprise_api_management", "confidence": 0.9, "performance_ratio": 1.1}
        ],
        "azure": [
            {"name": "API Management", "type": "api_management", "confidence": 0.9, "performance_ratio": 1.0}
        ]
    }
}

# Database Service Mappings
DATABASE_MAPPINGS = {
    "relational_database": {
        "aws": [
            {"name": "RDS", "type": "managed_relational_db", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "Aurora", "type": "cloud_native_db", "confidence": 0.9, "performance_ratio": 1.2}
        ],
        "gcp": [
            {"name": "Cloud SQL", "type": "managed_relational_db", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "Spanner", "type": "globally_distributed_db", "confidence": 0.85, "performance_ratio": 1.3}
        ],
        "azure": [
            {"name": "SQL Database", "type": "managed_relational_db", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "SQL Managed Instance", "type": "managed_sql_server", "confidence": 0.9, "performance_ratio": 1.0}
        ]
    },
    
    "nosql_database": {
        "aws": [
            {"name": "DynamoDB", "type": "document_db", "confidence": 0.95, "performance_ratio": 1.0},
            {"name": "DocumentDB", "type": "mongodb_compatible", "confidence": 0.85, "performance_ratio": 0.9}
        ],
        "gcp": [
            {"name": "Firestore", "type": "document_db", "confidence": 0.9, "performance_ratio": 0.95},
            {"name": "Bigtable", "type": "wide_column_db", "confidence": 0.85, "performance_ratio": 1.1}
        ],
        "azure": [
            {"name": "Cosmos DB", "type": "multi_model_db", "confidence": 0.9, "performance_ratio": 1.0}
        ]
    }
}

# Complete service equivalency configuration
SERVICE_EQUIVALENCY_CONFIG = {
    "compute": COMPUTE_MAPPINGS,
    "storage": STORAGE_MAPPINGS,
    "network": NETWORK_MAPPINGS,
    "database": DATABASE_MAPPINGS
}

# Feature mapping configuration
FEATURE_MAPPING_CONFIG = {
    "aws": {
        "EC2": [
            "auto_scaling", "load_balancing", "high_availability", "spot_instances",
            "reserved_instances", "encryption_at_rest", "encryption_in_transit",
            "network_isolation", "custom_networking", "gpu_support", "bare_metal"
        ],
        "S3": [
            "versioning", "lifecycle_management", "cross_region_replication",
            "encryption_at_rest", "encryption_in_transit", "access_logging",
            "event_notifications", "transfer_acceleration", "multipart_upload"
        ],
        "RDS": [
            "automated_backups", "point_in_time_recovery", "multi_az_deployment",
            "read_replicas", "encryption_at_rest", "encryption_in_transit",
            "performance_insights", "automated_patching"
        ]
    },
    "gcp": {
        "Compute Engine": [
            "auto_scaling", "load_balancing", "high_availability", "preemptible_instances",
            "committed_use_discounts", "encryption_at_rest", "encryption_in_transit",
            "network_isolation", "custom_networking", "gpu_support", "sole_tenancy"
        ],
        "Cloud Storage": [
            "versioning", "lifecycle_management", "cross_region_replication",
            "encryption_at_rest", "encryption_in_transit", "access_logging",
            "event_notifications", "transfer_service", "resumable_uploads"
        ],
        "Cloud SQL": [
            "automated_backups", "point_in_time_recovery", "high_availability",
            "read_replicas", "encryption_at_rest", "encryption_in_transit",
            "query_insights", "automated_patching"
        ]
    },
    "azure": {
        "Virtual Machines": [
            "auto_scaling", "load_balancing", "high_availability", "spot_instances",
            "reserved_instances", "encryption_at_rest", "encryption_in_transit",
            "network_isolation", "custom_networking", "gpu_support", "dedicated_hosts"
        ],
        "Blob Storage": [
            "versioning", "lifecycle_management", "geo_replication",
            "encryption_at_rest", "encryption_in_transit", "access_logging",
            "event_grid_integration", "data_lake_storage", "hierarchical_namespace"
        ],
        "SQL Database": [
            "automated_backups", "point_in_time_recovery", "geo_replication",
            "read_replicas", "encryption_at_rest", "encryption_in_transit",
            "query_performance_insights", "automated_tuning"
        ]
    }
}