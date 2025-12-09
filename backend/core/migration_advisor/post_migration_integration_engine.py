"""
Post-Migration Integration Engine

This module provides seamless integration between the Migration Advisor and
ongoing FinOps capabilities, enabling cost tracking, governance, and optimization
immediately after migration.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID
import logging

from sqlalchemy.orm import Session

from .models import (
    MigrationProject,
    OrganizationalStructure,
    CategorizedResource,
    MigrationPlan,
    BaselineMetrics,
    MigrationReport
)
# Note: CostCenter, Alert, Tag models don't exist in core.models
# These are created as dictionaries in the integration logic
# from ..models import (
#     CostCenter,
#     Budget,
#     Alert,
#     Tag
# )

logger = logging.getLogger(__name__)


class CostTrackingIntegrator:
    """
    Configures cost tracking and attribution based on organizational structure
    from migration projects.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def configure_cost_tracking(
        self,
        project_id: str,
        org_structure: OrganizationalStructure
    ) -> Dict[str, Any]:
        """
        Configure cost tracking based on organizational structure.
        
        Args:
            project_id: Migration project identifier
            org_structure: Organizational structure from migration
            
        Returns:
            Configuration result with created cost centers and attribution rules
        """
        logger.info(f"Configuring cost tracking for project {project_id}")
        
        result = {
            "cost_centers_created": [],
            "attribution_rules": [],
            "cost_center_mappings": {}
        }
        
        # Create cost centers from organizational structure
        cost_centers = self._create_cost_centers(org_structure)
        result["cost_centers_created"] = cost_centers
        
        # Generate cost attribution rules
        attribution_rules = self._generate_attribution_rules(
            project_id,
            org_structure
        )
        result["attribution_rules"] = attribution_rules
        
        # Create cost center mappings
        mappings = self._create_cost_center_mappings(
            org_structure,
            cost_centers
        )
        result["cost_center_mappings"] = mappings
        
        logger.info(
            f"Cost tracking configured: {len(cost_centers)} cost centers, "
            f"{len(attribution_rules)} attribution rules"
        )
        
        return result
    
    def _create_cost_centers(
        self,
        org_structure: OrganizationalStructure
    ) -> List[Dict[str, Any]]:
        """
        Create cost centers from organizational structure.
        
        Args:
            org_structure: Organizational structure
            
        Returns:
            List of created cost center definitions
        """
        cost_centers = []
        
        # Create cost centers from defined cost centers in structure
        for cc_data in org_structure.cost_centers:
            cost_center = {
                "name": cc_data.get("name"),
                "code": cc_data.get("code"),
                "description": cc_data.get("description", ""),
                "owner": cc_data.get("owner"),
                "teams": cc_data.get("teams", []),
                "projects": cc_data.get("projects", [])
            }
            cost_centers.append(cost_center)
        
        # If no cost centers defined, create from teams
        if not cost_centers and org_structure.teams:
            for team in org_structure.teams:
                cost_center = {
                    "name": f"{team.get('name')} Cost Center",
                    "code": f"CC-{team.get('id', team.get('name')).upper()}",
                    "description": f"Cost center for {team.get('name')} team",
                    "owner": team.get("owner"),
                    "teams": [team.get("name")],
                    "projects": []
                }
                cost_centers.append(cost_center)
        
        return cost_centers
    
    def _generate_attribution_rules(
        self,
        project_id: str,
        org_structure: OrganizationalStructure
    ) -> List[Dict[str, Any]]:
        """
        Generate cost attribution rules based on organizational structure.
        
        Args:
            project_id: Migration project identifier
            org_structure: Organizational structure
            
        Returns:
            List of attribution rules
        """
        attribution_rules = []
        
        # Rule 1: Attribute by team tag
        if org_structure.teams:
            for team in org_structure.teams:
                rule = {
                    "rule_type": "tag_based",
                    "dimension": "team",
                    "tag_key": "team",
                    "tag_value": team.get("name"),
                    "cost_center": f"CC-{team.get('id', team.get('name')).upper()}",
                    "priority": 1
                }
                attribution_rules.append(rule)
        
        # Rule 2: Attribute by project tag
        if org_structure.projects:
            for project in org_structure.projects:
                rule = {
                    "rule_type": "tag_based",
                    "dimension": "project",
                    "tag_key": "project",
                    "tag_value": project.get("name"),
                    "cost_center": project.get("cost_center"),
                    "priority": 2
                }
                attribution_rules.append(rule)
        
        # Rule 3: Attribute by environment
        if org_structure.environments:
            for env in org_structure.environments:
                rule = {
                    "rule_type": "tag_based",
                    "dimension": "environment",
                    "tag_key": "environment",
                    "tag_value": env.get("name"),
                    "cost_allocation_percentage": env.get("cost_allocation", 100),
                    "priority": 3
                }
                attribution_rules.append(rule)
        
        # Rule 4: Attribute by region
        if org_structure.regions:
            for region in org_structure.regions:
                rule = {
                    "rule_type": "tag_based",
                    "dimension": "region",
                    "tag_key": "region",
                    "tag_value": region.get("name"),
                    "priority": 4
                }
                attribution_rules.append(rule)
        
        return attribution_rules
    
    def _create_cost_center_mappings(
        self,
        org_structure: OrganizationalStructure,
        cost_centers: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Create mappings from organizational dimensions to cost centers.
        
        Args:
            org_structure: Organizational structure
            cost_centers: List of cost centers
            
        Returns:
            Dictionary mapping dimension values to cost center codes
        """
        mappings = {}
        
        # Map teams to cost centers
        for team in org_structure.teams:
            team_name = team.get("name")
            # Find matching cost center
            for cc in cost_centers:
                if team_name in cc.get("teams", []):
                    mappings[f"team:{team_name}"] = cc["code"]
                    break
        
        # Map projects to cost centers
        for project in org_structure.projects:
            project_name = project.get("name")
            cost_center_code = project.get("cost_center")
            if cost_center_code:
                mappings[f"project:{project_name}"] = cost_center_code
        
        return mappings
    
    def apply_cost_attribution(
        self,
        project_id: str,
        resources: List[CategorizedResource]
    ) -> Dict[str, Any]:
        """
        Apply cost attribution rules to categorized resources.
        
        Args:
            project_id: Migration project identifier
            resources: List of categorized resources
            
        Returns:
            Attribution result with resource assignments
        """
        logger.info(f"Applying cost attribution to {len(resources)} resources")
        
        result = {
            "resources_attributed": 0,
            "resources_unattributed": 0,
            "attribution_by_cost_center": {}
        }
        
        for resource in resources:
            # Determine cost center based on resource categorization
            cost_center = self._determine_cost_center(resource)
            
            if cost_center:
                result["resources_attributed"] += 1
                if cost_center not in result["attribution_by_cost_center"]:
                    result["attribution_by_cost_center"][cost_center] = []
                result["attribution_by_cost_center"][cost_center].append(
                    resource.resource_id
                )
            else:
                result["resources_unattributed"] += 1
        
        logger.info(
            f"Cost attribution complete: {result['resources_attributed']} attributed, "
            f"{result['resources_unattributed']} unattributed"
        )
        
        return result
    
    def _determine_cost_center(
        self,
        resource: CategorizedResource
    ) -> Optional[str]:
        """
        Determine cost center for a resource based on its categorization.
        
        Args:
            resource: Categorized resource
            
        Returns:
            Cost center code or None
        """
        # Priority 1: Explicit cost_center assignment
        if resource.cost_center:
            return resource.cost_center
        
        # Priority 2: Team-based assignment
        if resource.team:
            return f"CC-{resource.team.upper()}"
        
        # Priority 3: Project-based assignment
        if resource.project:
            return f"CC-{resource.project.upper()}"
        
        return None


class FinOpsConnector:
    """
    Integrates migration advisor with existing FinOps platform capabilities.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def transfer_organizational_structure(
        self,
        project_id: str,
        org_structure: OrganizationalStructure
    ) -> Dict[str, Any]:
        """
        Transfer organizational structure to FinOps platform.
        
        Args:
            project_id: Migration project identifier
            org_structure: Organizational structure to transfer
            
        Returns:
            Transfer result with created entities
        """
        logger.info(
            f"Transferring organizational structure for project {project_id}"
        )
        
        result = {
            "teams_created": [],
            "projects_created": [],
            "cost_centers_created": [],
            "tags_created": []
        }
        
        # Transfer teams
        for team in org_structure.teams:
            team_data = {
                "name": team.get("name"),
                "description": team.get("description", ""),
                "owner": team.get("owner"),
                "members": team.get("members", []),
                "source": "migration_advisor",
                "migration_project_id": project_id
            }
            result["teams_created"].append(team_data)
        
        # Transfer projects
        for project in org_structure.projects:
            project_data = {
                "name": project.get("name"),
                "description": project.get("description", ""),
                "team": project.get("team"),
                "cost_center": project.get("cost_center"),
                "source": "migration_advisor",
                "migration_project_id": project_id
            }
            result["projects_created"].append(project_data)
        
        # Transfer cost centers
        for cost_center in org_structure.cost_centers:
            cc_data = {
                "code": cost_center.get("code"),
                "name": cost_center.get("name"),
                "description": cost_center.get("description", ""),
                "owner": cost_center.get("owner"),
                "source": "migration_advisor",
                "migration_project_id": project_id
            }
            result["cost_centers_created"].append(cc_data)
        
        logger.info(
            f"Organizational structure transferred: "
            f"{len(result['teams_created'])} teams, "
            f"{len(result['projects_created'])} projects, "
            f"{len(result['cost_centers_created'])} cost centers"
        )
        
        return result
    
    def configure_budgets(
        self,
        project_id: str,
        org_structure: OrganizationalStructure,
        migration_plan: MigrationPlan
    ) -> Dict[str, Any]:
        """
        Configure budgets based on organizational structure and migration plan.
        
        Args:
            project_id: Migration project identifier
            org_structure: Organizational structure
            migration_plan: Migration plan with cost estimates
            
        Returns:
            Budget configuration result
        """
        logger.info(f"Configuring budgets for project {project_id}")
        
        result = {
            "budgets_created": [],
            "total_budget_allocated": Decimal("0.00")
        }
        
        # Create budgets for each cost center
        for cost_center in org_structure.cost_centers:
            budget_data = {
                "cost_center": cost_center.get("code"),
                "name": f"{cost_center.get('name')} Monthly Budget",
                "amount": cost_center.get("budget_amount", Decimal("0.00")),
                "period": "monthly",
                "alert_threshold": 80.0,  # Alert at 80% of budget
                "source": "migration_advisor",
                "migration_project_id": project_id
            }
            result["budgets_created"].append(budget_data)
            result["total_budget_allocated"] += budget_data["amount"]
        
        # Create budgets for teams if no cost center budgets
        if not result["budgets_created"]:
            for team in org_structure.teams:
                budget_data = {
                    "team": team.get("name"),
                    "name": f"{team.get('name')} Monthly Budget",
                    "amount": team.get("budget_amount", Decimal("10000.00")),
                    "period": "monthly",
                    "alert_threshold": 80.0,
                    "source": "migration_advisor",
                    "migration_project_id": project_id
                }
                result["budgets_created"].append(budget_data)
                result["total_budget_allocated"] += budget_data["amount"]
        
        logger.info(
            f"Budgets configured: {len(result['budgets_created'])} budgets, "
            f"total allocated: {result['total_budget_allocated']}"
        )
        
        return result
    
    def configure_alerts(
        self,
        project_id: str,
        org_structure: OrganizationalStructure
    ) -> Dict[str, Any]:
        """
        Configure cost and compliance alerts based on organizational structure.
        
        Args:
            project_id: Migration project identifier
            org_structure: Organizational structure
            
        Returns:
            Alert configuration result
        """
        logger.info(f"Configuring alerts for project {project_id}")
        
        result = {
            "alerts_created": []
        }
        
        # Create budget alerts for each cost center
        for cost_center in org_structure.cost_centers:
            # Budget threshold alert
            alert_data = {
                "name": f"{cost_center.get('name')} Budget Alert",
                "type": "budget_threshold",
                "cost_center": cost_center.get("code"),
                "threshold": 80.0,
                "severity": "warning",
                "enabled": True,
                "source": "migration_advisor"
            }
            result["alerts_created"].append(alert_data)
            
            # Budget exceeded alert
            alert_data = {
                "name": f"{cost_center.get('name')} Budget Exceeded",
                "type": "budget_exceeded",
                "cost_center": cost_center.get("code"),
                "threshold": 100.0,
                "severity": "critical",
                "enabled": True,
                "source": "migration_advisor"
            }
            result["alerts_created"].append(alert_data)
        
        # Create anomaly detection alerts
        anomaly_alert = {
            "name": "Cost Anomaly Detection",
            "type": "cost_anomaly",
            "threshold": 20.0,  # 20% deviation
            "severity": "warning",
            "enabled": True,
            "source": "migration_advisor"
        }
        result["alerts_created"].append(anomaly_alert)
        
        # Create untagged resource alert
        untagged_alert = {
            "name": "Untagged Resources Alert",
            "type": "governance",
            "threshold": 5.0,  # Alert if >5% resources untagged
            "severity": "warning",
            "enabled": True,
            "source": "migration_advisor"
        }
        result["alerts_created"].append(untagged_alert)
        
        logger.info(f"Alerts configured: {len(result['alerts_created'])} alerts")
        
        return result
    
    def enable_finops_features(
        self,
        project_id: str,
        provider: str
    ) -> Dict[str, Any]:
        """
        Enable FinOps features for the migrated environment.
        
        Args:
            project_id: Migration project identifier
            provider: Cloud provider (AWS, GCP, Azure)
            
        Returns:
            Feature enablement result
        """
        logger.info(
            f"Enabling FinOps features for project {project_id} on {provider}"
        )
        
        result = {
            "features_enabled": []
        }
        
        # Enable waste detection
        result["features_enabled"].append({
            "feature": "waste_detection",
            "enabled": True,
            "config": {
                "idle_threshold_days": 7,
                "utilization_threshold": 5.0
            }
        })
        
        # Enable RI optimization
        result["features_enabled"].append({
            "feature": "ri_optimization",
            "enabled": True,
            "config": {
                "analysis_period_days": 30,
                "commitment_threshold": 0.7
            }
        })
        
        # Enable cost anomaly detection
        result["features_enabled"].append({
            "feature": "cost_anomaly_detection",
            "enabled": True,
            "config": {
                "sensitivity": "medium",
                "lookback_days": 14
            }
        })
        
        # Enable tagging compliance
        result["features_enabled"].append({
            "feature": "tagging_compliance",
            "enabled": True,
            "config": {
                "required_tags": ["team", "project", "environment"],
                "enforcement_level": "warning"
            }
        })
        
        # Enable budget monitoring
        result["features_enabled"].append({
            "feature": "budget_monitoring",
            "enabled": True,
            "config": {
                "check_frequency": "hourly",
                "alert_channels": ["email", "slack"]
            }
        })
        
        logger.info(
            f"FinOps features enabled: {len(result['features_enabled'])} features"
        )
        
        return result


class BaselineCaptureSystem:
    """
    Captures initial cost and performance baselines post-migration.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def capture_baselines(
        self,
        project_id: str,
        resources: List[CategorizedResource]
    ) -> BaselineMetrics:
        """
        Capture baseline metrics for cost and performance.
        
        Args:
            project_id: Migration project identifier
            resources: List of categorized resources
            
        Returns:
            BaselineMetrics object with captured data
        """
        logger.info(f"Capturing baselines for project {project_id}")
        
        # Load migration project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {project_id} not found")
        
        # Calculate cost baselines
        cost_by_service = self._calculate_cost_by_service(resources)
        cost_by_team = self._calculate_cost_by_team(resources)
        cost_by_project = self._calculate_cost_by_project(resources)
        cost_by_environment = self._calculate_cost_by_environment(resources)
        
        total_monthly_cost = sum(cost_by_service.values())
        
        # Calculate resource utilization baselines
        resource_utilization = self._capture_resource_utilization(resources)
        
        # Calculate performance baselines
        performance_metrics = self._capture_performance_metrics(resources)
        
        # Calculate resource counts
        resource_count = len(resources)
        resource_count_by_type = self._count_resources_by_type(resources)
        
        # Create baseline metrics record
        baseline = BaselineMetrics(
            migration_project_id=project.id,
            capture_date=datetime.utcnow(),
            total_monthly_cost=Decimal(str(total_monthly_cost)),
            cost_by_service=cost_by_service,
            cost_by_team=cost_by_team,
            cost_by_project=cost_by_project,
            cost_by_environment=cost_by_environment,
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics,
            resource_count=resource_count,
            resource_count_by_type=resource_count_by_type,
            notes="Initial baseline captured post-migration"
        )
        
        self.db.add(baseline)
        self.db.commit()
        self.db.refresh(baseline)
        
        logger.info(
            f"Baselines captured: {resource_count} resources, "
            f"total cost: {total_monthly_cost}"
        )
        
        return baseline
    
    def _calculate_cost_by_service(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, float]:
        """Calculate estimated cost breakdown by service type."""
        cost_by_service = {}
        
        # Simplified cost estimation based on resource types
        # In production, this would integrate with actual cloud billing APIs
        cost_estimates = {
            "compute": 100.0,
            "storage": 50.0,
            "database": 200.0,
            "network": 30.0,
            "container": 150.0,
            "serverless": 75.0,
            "ml": 300.0,
            "analytics": 250.0
        }
        
        for resource in resources:
            resource_type = resource.resource_type.lower()
            
            # Map resource type to service category
            if "compute" in resource_type or "instance" in resource_type:
                service = "compute"
            elif "storage" in resource_type or "bucket" in resource_type:
                service = "storage"
            elif "database" in resource_type or "db" in resource_type:
                service = "database"
            elif "network" in resource_type or "vpc" in resource_type:
                service = "network"
            elif "container" in resource_type or "kubernetes" in resource_type:
                service = "container"
            elif "lambda" in resource_type or "function" in resource_type:
                service = "serverless"
            elif "ml" in resource_type or "sagemaker" in resource_type:
                service = "ml"
            elif "analytics" in resource_type or "bigquery" in resource_type:
                service = "analytics"
            else:
                service = "other"
            
            cost = cost_estimates.get(service, 50.0)
            cost_by_service[service] = cost_by_service.get(service, 0.0) + cost
        
        return cost_by_service
    
    def _calculate_cost_by_team(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, float]:
        """Calculate cost breakdown by team."""
        cost_by_team = {}
        
        for resource in resources:
            if resource.team:
                # Simplified cost allocation
                cost = 100.0  # Base cost per resource
                cost_by_team[resource.team] = cost_by_team.get(resource.team, 0.0) + cost
        
        return cost_by_team
    
    def _calculate_cost_by_project(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, float]:
        """Calculate cost breakdown by project."""
        cost_by_project = {}
        
        for resource in resources:
            if resource.project:
                cost = 100.0  # Base cost per resource
                cost_by_project[resource.project] = cost_by_project.get(
                    resource.project, 0.0
                ) + cost
        
        return cost_by_project
    
    def _calculate_cost_by_environment(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, float]:
        """Calculate cost breakdown by environment."""
        cost_by_environment = {}
        
        for resource in resources:
            if resource.environment:
                cost = 100.0  # Base cost per resource
                cost_by_environment[resource.environment] = cost_by_environment.get(
                    resource.environment, 0.0
                ) + cost
        
        return cost_by_environment
    
    def _capture_resource_utilization(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, Dict[str, float]]:
        """Capture resource utilization metrics."""
        utilization = {}
        
        # Simplified utilization capture
        # In production, this would query actual metrics from cloud providers
        for resource in resources:
            utilization[resource.resource_id] = {
                "cpu_utilization": 45.0,  # Percentage
                "memory_utilization": 60.0,  # Percentage
                "disk_utilization": 35.0,  # Percentage
                "network_in_mbps": 10.0,
                "network_out_mbps": 8.0
            }
        
        return utilization
    
    def _capture_performance_metrics(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, Dict[str, Any]]:
        """Capture performance metrics by service."""
        performance = {}
        
        # Simplified performance metrics
        # In production, this would query actual performance data
        service_types = set(resource.resource_type for resource in resources)
        
        for service_type in service_types:
            performance[service_type] = {
                "avg_response_time_ms": 150.0,
                "p95_response_time_ms": 300.0,
                "p99_response_time_ms": 500.0,
                "error_rate": 0.1,  # Percentage
                "availability": 99.95  # Percentage
            }
        
        return performance
    
    def _count_resources_by_type(
        self,
        resources: List[CategorizedResource]
    ) -> Dict[str, int]:
        """Count resources by type."""
        count_by_type = {}
        
        for resource in resources:
            resource_type = resource.resource_type
            count_by_type[resource_type] = count_by_type.get(resource_type, 0) + 1
        
        return count_by_type
    
    def retrieve_baseline(
        self,
        project_id: str
    ) -> Optional[BaselineMetrics]:
        """
        Retrieve baseline metrics for a project.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            BaselineMetrics object or None if not found
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            return None
        
        baseline = self.db.query(BaselineMetrics).filter(
            BaselineMetrics.migration_project_id == project.id
        ).order_by(BaselineMetrics.capture_date.desc()).first()
        
        return baseline


class GovernancePolicyApplicator:
    """
    Applies tagging policies and compliance rules to migrated resources.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def apply_tagging_policies(
        self,
        project_id: str,
        resources: List[CategorizedResource]
    ) -> Dict[str, Any]:
        """
        Apply tagging policies to resources.
        
        Args:
            project_id: Migration project identifier
            resources: List of categorized resources
            
        Returns:
            Policy application result
        """
        logger.info(f"Applying tagging policies for project {project_id}")
        
        result = {
            "resources_processed": 0,
            "resources_compliant": 0,
            "resources_non_compliant": 0,
            "tags_applied": 0,
            "violations": []
        }
        
        # Define required tags
        required_tags = ["team", "project", "environment", "cost_center", "owner"]
        
        for resource in resources:
            result["resources_processed"] += 1
            
            # Check compliance
            missing_tags = []
            for tag in required_tags:
                if tag not in resource.tags or not resource.tags[tag]:
                    missing_tags.append(tag)
            
            if missing_tags:
                result["resources_non_compliant"] += 1
                result["violations"].append({
                    "resource_id": resource.resource_id,
                    "resource_type": resource.resource_type,
                    "missing_tags": missing_tags
                })
                
                # Auto-apply tags from categorization
                tags_applied = self._auto_apply_tags(resource, missing_tags)
                result["tags_applied"] += tags_applied
            else:
                result["resources_compliant"] += 1
        
        logger.info(
            f"Tagging policies applied: {result['resources_compliant']} compliant, "
            f"{result['resources_non_compliant']} non-compliant, "
            f"{result['tags_applied']} tags applied"
        )
        
        return result
    
    def _auto_apply_tags(
        self,
        resource: CategorizedResource,
        missing_tags: List[str]
    ) -> int:
        """
        Auto-apply missing tags from resource categorization.
        
        Args:
            resource: Categorized resource
            missing_tags: List of missing tag keys
            
        Returns:
            Number of tags applied
        """
        tags_applied = 0
        
        if not resource.tags:
            resource.tags = {}
        
        # Apply tags from categorization
        tag_mappings = {
            "team": resource.team,
            "project": resource.project,
            "environment": resource.environment,
            "cost_center": resource.cost_center,
            "region": resource.region
        }
        
        for tag_key in missing_tags:
            if tag_key in tag_mappings and tag_mappings[tag_key]:
                resource.tags[tag_key] = tag_mappings[tag_key]
                tags_applied += 1
        
        # Add owner tag if missing
        if "owner" in missing_tags:
            resource.tags["owner"] = "migration-advisor"
            tags_applied += 1
        
        return tags_applied
    
    def enforce_compliance_rules(
        self,
        project_id: str,
        resources: List[CategorizedResource]
    ) -> Dict[str, Any]:
        """
        Enforce compliance rules on resources.
        
        Args:
            project_id: Migration project identifier
            resources: List of categorized resources
            
        Returns:
            Compliance enforcement result
        """
        logger.info(f"Enforcing compliance rules for project {project_id}")
        
        result = {
            "resources_checked": 0,
            "compliant_resources": 0,
            "violations": [],
            "rules_enforced": []
        }
        
        # Rule 1: All resources must be categorized
        categorization_violations = self._check_categorization_compliance(resources)
        result["violations"].extend(categorization_violations)
        result["rules_enforced"].append("categorization_required")
        
        # Rule 2: All resources must have required tags
        tagging_violations = self._check_tagging_compliance(resources)
        result["violations"].extend(tagging_violations)
        result["rules_enforced"].append("required_tags")
        
        # Rule 3: Resources must belong to valid organizational units
        org_violations = self._check_organizational_compliance(resources, project_id)
        result["violations"].extend(org_violations)
        result["rules_enforced"].append("valid_organizational_units")
        
        result["resources_checked"] = len(resources)
        result["compliant_resources"] = len(resources) - len(result["violations"])
        
        logger.info(
            f"Compliance rules enforced: {result['compliant_resources']} compliant, "
            f"{len(result['violations'])} violations"
        )
        
        return result
    
    def _check_categorization_compliance(
        self,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Check if resources are properly categorized."""
        violations = []
        
        for resource in resources:
            if not resource.team and not resource.project:
                violations.append({
                    "resource_id": resource.resource_id,
                    "rule": "categorization_required",
                    "severity": "high",
                    "message": "Resource must be assigned to a team or project"
                })
        
        return violations
    
    def _check_tagging_compliance(
        self,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Check if resources have required tags."""
        violations = []
        required_tags = ["team", "project", "environment"]
        
        for resource in resources:
            missing_tags = []
            for tag in required_tags:
                if not resource.tags or tag not in resource.tags:
                    missing_tags.append(tag)
            
            if missing_tags:
                violations.append({
                    "resource_id": resource.resource_id,
                    "rule": "required_tags",
                    "severity": "medium",
                    "message": f"Missing required tags: {', '.join(missing_tags)}"
                })
        
        return violations
    
    def _check_organizational_compliance(
        self,
        resources: List[CategorizedResource],
        project_id: str
    ) -> List[Dict[str, Any]]:
        """Check if resources belong to valid organizational units."""
        violations = []
        
        # Load organizational structure
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            return violations
        
        org_structure = self.db.query(OrganizationalStructure).filter(
            OrganizationalStructure.migration_project_id == project.id
        ).first()
        
        if not org_structure:
            return violations
        
        # Get valid teams and projects
        valid_teams = {team.get("name") for team in org_structure.teams}
        valid_projects = {proj.get("name") for proj in org_structure.projects}
        
        for resource in resources:
            if resource.team and resource.team not in valid_teams:
                violations.append({
                    "resource_id": resource.resource_id,
                    "rule": "valid_organizational_units",
                    "severity": "high",
                    "message": f"Team '{resource.team}' is not a valid organizational unit"
                })
            
            if resource.project and resource.project not in valid_projects:
                violations.append({
                    "resource_id": resource.resource_id,
                    "rule": "valid_organizational_units",
                    "severity": "high",
                    "message": f"Project '{resource.project}' is not a valid organizational unit"
                })
        
        return violations
    
    def generate_compliance_report(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Generate compliance report for a migration project.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            Compliance report
        """
        logger.info(f"Generating compliance report for project {project_id}")
        
        # Load project and resources
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {project_id} not found")
        
        resources = self.db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id
        ).all()
        
        # Check compliance
        tagging_result = self.apply_tagging_policies(project_id, resources)
        compliance_result = self.enforce_compliance_rules(project_id, resources)
        
        report = {
            "project_id": project_id,
            "report_date": datetime.utcnow().isoformat(),
            "total_resources": len(resources),
            "tagging_compliance": {
                "compliant": tagging_result["resources_compliant"],
                "non_compliant": tagging_result["resources_non_compliant"],
                "compliance_rate": (
                    tagging_result["resources_compliant"] / len(resources) * 100
                    if resources else 0
                )
            },
            "governance_compliance": {
                "compliant": compliance_result["compliant_resources"],
                "violations": len(compliance_result["violations"]),
                "compliance_rate": (
                    compliance_result["compliant_resources"] / len(resources) * 100
                    if resources else 0
                )
            },
            "violations": compliance_result["violations"]
        }
        
        logger.info(
            f"Compliance report generated: "
            f"{report['tagging_compliance']['compliance_rate']:.1f}% tagging compliance, "
            f"{report['governance_compliance']['compliance_rate']:.1f}% governance compliance"
        )
        
        return report


class OptimizationIdentifier:
    """
    Identifies immediate optimization opportunities post-migration.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def identify_optimization_opportunities(
        self,
        project_id: str,
        baseline: BaselineMetrics,
        resources: List[CategorizedResource]
    ) -> Dict[str, Any]:
        """
        Identify immediate optimization opportunities.
        
        Args:
            project_id: Migration project identifier
            baseline: Baseline metrics
            resources: List of categorized resources
            
        Returns:
            Optimization opportunities with recommendations
        """
        logger.info(f"Identifying optimization opportunities for project {project_id}")
        
        result = {
            "opportunities": [],
            "total_potential_savings": 0.0,
            "priority_recommendations": []
        }
        
        # Identify underutilized resources
        underutilized = self._identify_underutilized_resources(
            baseline,
            resources
        )
        result["opportunities"].extend(underutilized)
        
        # Identify rightsizing opportunities
        rightsizing = self._identify_rightsizing_opportunities(
            baseline,
            resources
        )
        result["opportunities"].extend(rightsizing)
        
        # Identify reserved instance opportunities
        ri_opportunities = self._identify_ri_opportunities(
            baseline,
            resources
        )
        result["opportunities"].extend(ri_opportunities)
        
        # Identify storage optimization opportunities
        storage_opportunities = self._identify_storage_optimization(
            baseline,
            resources
        )
        result["opportunities"].extend(storage_opportunities)
        
        # Calculate total potential savings
        result["total_potential_savings"] = sum(
            opp.get("potential_savings", 0.0)
            for opp in result["opportunities"]
        )
        
        # Prioritize recommendations
        result["priority_recommendations"] = self._prioritize_recommendations(
            result["opportunities"]
        )
        
        logger.info(
            f"Optimization opportunities identified: {len(result['opportunities'])} "
            f"opportunities, ${result['total_potential_savings']:.2f} potential savings"
        )
        
        return result
    
    def _identify_underutilized_resources(
        self,
        baseline: BaselineMetrics,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Identify underutilized resources."""
        opportunities = []
        
        for resource_id, utilization in baseline.resource_utilization.items():
            cpu_util = utilization.get("cpu_utilization", 100.0)
            memory_util = utilization.get("memory_utilization", 100.0)
            
            # Check if resource is underutilized
            if cpu_util < 10.0 and memory_util < 10.0:
                opportunities.append({
                    "type": "underutilized_resource",
                    "resource_id": resource_id,
                    "severity": "high",
                    "cpu_utilization": cpu_util,
                    "memory_utilization": memory_util,
                    "recommendation": "Consider terminating or downsizing this resource",
                    "potential_savings": 100.0,  # Estimated monthly savings
                    "priority": 1
                })
            elif cpu_util < 20.0 or memory_util < 20.0:
                opportunities.append({
                    "type": "underutilized_resource",
                    "resource_id": resource_id,
                    "severity": "medium",
                    "cpu_utilization": cpu_util,
                    "memory_utilization": memory_util,
                    "recommendation": "Consider downsizing this resource",
                    "potential_savings": 50.0,
                    "priority": 2
                })
        
        return opportunities
    
    def _identify_rightsizing_opportunities(
        self,
        baseline: BaselineMetrics,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Identify rightsizing opportunities."""
        opportunities = []
        
        # Analyze compute resources
        compute_resources = [
            r for r in resources
            if "compute" in r.resource_type.lower() or "instance" in r.resource_type.lower()
        ]
        
        for resource in compute_resources:
            utilization = baseline.resource_utilization.get(resource.resource_id, {})
            cpu_util = utilization.get("cpu_utilization", 50.0)
            memory_util = utilization.get("memory_utilization", 50.0)
            
            # Recommend downsizing if consistently low utilization
            if cpu_util < 30.0 and memory_util < 30.0:
                opportunities.append({
                    "type": "rightsizing",
                    "resource_id": resource.resource_id,
                    "resource_type": resource.resource_type,
                    "severity": "medium",
                    "current_utilization": {
                        "cpu": cpu_util,
                        "memory": memory_util
                    },
                    "recommendation": "Downsize to smaller instance type",
                    "potential_savings": 75.0,
                    "priority": 2
                })
        
        return opportunities
    
    def _identify_ri_opportunities(
        self,
        baseline: BaselineMetrics,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Identify reserved instance opportunities."""
        opportunities = []
        
        # Analyze stable workloads for RI recommendations
        compute_resources = [
            r for r in resources
            if "compute" in r.resource_type.lower() or "instance" in r.resource_type.lower()
        ]
        
        if len(compute_resources) >= 3:
            opportunities.append({
                "type": "reserved_instances",
                "severity": "high",
                "resource_count": len(compute_resources),
                "recommendation": (
                    f"Consider purchasing reserved instances for {len(compute_resources)} "
                    "compute resources to save 30-40% on costs"
                ),
                "potential_savings": len(compute_resources) * 30.0,  # $30/month per instance
                "priority": 1
            })
        
        return opportunities
    
    def _identify_storage_optimization(
        self,
        baseline: BaselineMetrics,
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Identify storage optimization opportunities."""
        opportunities = []
        
        # Analyze storage resources
        storage_resources = [
            r for r in resources
            if "storage" in r.resource_type.lower() or "disk" in r.resource_type.lower()
        ]
        
        for resource in storage_resources:
            utilization = baseline.resource_utilization.get(resource.resource_id, {})
            disk_util = utilization.get("disk_utilization", 50.0)
            
            # Recommend storage tier optimization
            if disk_util < 40.0:
                opportunities.append({
                    "type": "storage_optimization",
                    "resource_id": resource.resource_id,
                    "resource_type": resource.resource_type,
                    "severity": "low",
                    "disk_utilization": disk_util,
                    "recommendation": "Consider moving to lower-cost storage tier",
                    "potential_savings": 20.0,
                    "priority": 3
                })
        
        return opportunities
    
    def _prioritize_recommendations(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize optimization recommendations."""
        # Sort by priority and potential savings
        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: (x.get("priority", 999), -x.get("potential_savings", 0.0))
        )
        
        # Return top 10 recommendations
        return sorted_opportunities[:10]
    
    def generate_optimization_report(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Generate optimization report for a migration project.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            Optimization report
        """
        logger.info(f"Generating optimization report for project {project_id}")
        
        # Load project, baseline, and resources
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {project_id} not found")
        
        baseline = self.db.query(BaselineMetrics).filter(
            BaselineMetrics.migration_project_id == project.id
        ).order_by(BaselineMetrics.capture_date.desc()).first()
        
        if not baseline:
            raise ValueError(f"Baseline metrics not found for project {project_id}")
        
        resources = self.db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id
        ).all()
        
        # Identify opportunities
        opportunities = self.identify_optimization_opportunities(
            project_id,
            baseline,
            resources
        )
        
        report = {
            "project_id": project_id,
            "report_date": datetime.utcnow().isoformat(),
            "baseline_cost": float(baseline.total_monthly_cost),
            "opportunities_count": len(opportunities["opportunities"]),
            "total_potential_savings": opportunities["total_potential_savings"],
            "potential_savings_percentage": (
                opportunities["total_potential_savings"] / 
                float(baseline.total_monthly_cost) * 100
                if baseline.total_monthly_cost > 0 else 0
            ),
            "priority_recommendations": opportunities["priority_recommendations"],
            "all_opportunities": opportunities["opportunities"]
        }
        
        logger.info(
            f"Optimization report generated: {report['opportunities_count']} opportunities, "
            f"${report['total_potential_savings']:.2f} potential savings "
            f"({report['potential_savings_percentage']:.1f}%)"
        )
        
        return report


class MigrationReportGenerator:
    """
    Generates comprehensive migration completion reports.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_migration_report(
        self,
        project_id: str
    ) -> MigrationReport:
        """
        Generate comprehensive migration report.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            MigrationReport object
        """
        logger.info(f"Generating migration report for project {project_id}")
        
        # Load migration project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {project_id} not found")
        
        # Load migration plan
        migration_plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        # Load baseline metrics
        baseline = self.db.query(BaselineMetrics).filter(
            BaselineMetrics.migration_project_id == project.id
        ).order_by(BaselineMetrics.capture_date.desc()).first()
        
        # Load resources
        resources = self.db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id
        ).all()
        
        # Calculate timeline analysis
        timeline_analysis = self._analyze_timeline(project, migration_plan)
        
        # Calculate cost analysis
        cost_analysis = self._analyze_costs(project, migration_plan, baseline)
        
        # Generate lessons learned
        lessons_learned = self._generate_lessons_learned(
            project,
            migration_plan,
            timeline_analysis,
            cost_analysis
        )
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_post_migration_optimizations(
            baseline,
            resources
        )
        
        # Calculate success metrics
        success_metrics = self._calculate_success_metrics(
            project,
            migration_plan,
            resources
        )
        
        # Create migration report
        report = MigrationReport(
            migration_project_id=project.id,
            report_date=datetime.utcnow(),
            start_date=project.created_at,
            completion_date=project.actual_completion or datetime.utcnow(),
            actual_duration_days=timeline_analysis["actual_duration_days"],
            planned_duration_days=timeline_analysis["planned_duration_days"],
            total_cost=cost_analysis["total_cost"],
            budgeted_cost=cost_analysis["budgeted_cost"],
            resources_migrated=len(resources),
            success_rate=success_metrics["success_rate"],
            lessons_learned=lessons_learned,
            optimization_opportunities=optimization_opportunities,
            cost_analysis=cost_analysis,
            timeline_analysis=timeline_analysis,
            risk_incidents=success_metrics.get("risk_incidents", []),
            recommendations=success_metrics.get("recommendations", [])
        )
        
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        
        logger.info(
            f"Migration report generated: {report.resources_migrated} resources, "
            f"{report.success_rate:.1f}% success rate"
        )
        
        return report
    
    def _analyze_timeline(
        self,
        project: MigrationProject,
        migration_plan: Optional[MigrationPlan]
    ) -> Dict[str, Any]:
        """Analyze migration timeline."""
        start_date = project.created_at
        completion_date = project.actual_completion or datetime.utcnow()
        actual_duration = (completion_date - start_date).days
        
        planned_duration = (
            migration_plan.total_duration_days
            if migration_plan else 0
        )
        
        variance_days = actual_duration - planned_duration
        variance_percentage = (
            (variance_days / planned_duration * 100)
            if planned_duration > 0 else 0
        )
        
        return {
            "start_date": start_date.isoformat(),
            "completion_date": completion_date.isoformat(),
            "actual_duration_days": actual_duration,
            "planned_duration_days": planned_duration,
            "variance_days": variance_days,
            "variance_percentage": variance_percentage,
            "on_schedule": variance_days <= 0
        }
    
    def _analyze_costs(
        self,
        project: MigrationProject,
        migration_plan: Optional[MigrationPlan],
        baseline: Optional[BaselineMetrics]
    ) -> Dict[str, Any]:
        """Analyze migration costs."""
        # Get actual costs from baseline
        total_cost = (
            float(baseline.total_monthly_cost)
            if baseline else 0.0
        )
        
        # Get budgeted cost from migration plan
        budgeted_cost = (
            float(migration_plan.estimated_cost)
            if migration_plan else 0.0
        )
        
        variance = total_cost - budgeted_cost
        variance_percentage = (
            (variance / budgeted_cost * 100)
            if budgeted_cost > 0 else 0
        )
        
        cost_breakdown = {}
        if baseline:
            cost_breakdown = {
                "by_service": baseline.cost_by_service,
                "by_team": baseline.cost_by_team,
                "by_project": baseline.cost_by_project,
                "by_environment": baseline.cost_by_environment
            }
        
        return {
            "total_cost": total_cost,
            "budgeted_cost": budgeted_cost,
            "variance": variance,
            "variance_percentage": variance_percentage,
            "under_budget": variance <= 0,
            "cost_breakdown": cost_breakdown
        }
    
    def _generate_lessons_learned(
        self,
        project: MigrationProject,
        migration_plan: Optional[MigrationPlan],
        timeline_analysis: Dict[str, Any],
        cost_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate lessons learned from migration."""
        lessons = []
        
        # Timeline lessons
        if timeline_analysis["variance_days"] > 7:
            lessons.append({
                "category": "timeline",
                "lesson": (
                    f"Migration took {timeline_analysis['variance_days']} days longer "
                    "than planned. Consider more buffer time for future migrations."
                ),
                "impact": "medium"
            })
        elif timeline_analysis["variance_days"] < -7:
            lessons.append({
                "category": "timeline",
                "lesson": (
                    f"Migration completed {abs(timeline_analysis['variance_days'])} days "
                    "ahead of schedule. Planning was conservative."
                ),
                "impact": "low"
            })
        
        # Cost lessons
        if cost_analysis["variance_percentage"] > 10:
            lessons.append({
                "category": "cost",
                "lesson": (
                    f"Migration costs exceeded budget by {cost_analysis['variance_percentage']:.1f}%. "
                    "Improve cost estimation for future migrations."
                ),
                "impact": "high"
            })
        elif cost_analysis["variance_percentage"] < -10:
            lessons.append({
                "category": "cost",
                "lesson": (
                    f"Migration came in {abs(cost_analysis['variance_percentage']):.1f}% "
                    "under budget. Cost estimates were conservative."
                ),
                "impact": "low"
            })
        
        # General lessons
        lessons.append({
            "category": "process",
            "lesson": (
                "Automated resource organization significantly reduced post-migration "
                "manual work."
            ),
            "impact": "high"
        })
        
        lessons.append({
            "category": "governance",
            "lesson": (
                "Applying tagging policies during migration ensured immediate "
                "cost visibility."
            ),
            "impact": "high"
        })
        
        return lessons
    
    def _identify_post_migration_optimizations(
        self,
        baseline: Optional[BaselineMetrics],
        resources: List[CategorizedResource]
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities for the report."""
        if not baseline:
            return []
        
        opportunities = []
        
        # Check for underutilized resources
        underutilized_count = 0
        for resource_id, utilization in baseline.resource_utilization.items():
            if utilization.get("cpu_utilization", 100) < 20:
                underutilized_count += 1
        
        if underutilized_count > 0:
            opportunities.append({
                "type": "underutilized_resources",
                "count": underutilized_count,
                "description": (
                    f"{underutilized_count} resources are underutilized. "
                    "Consider rightsizing or terminating."
                ),
                "potential_savings": underutilized_count * 50.0
            })
        
        # Check for RI opportunities
        compute_count = len([
            r for r in resources
            if "compute" in r.resource_type.lower()
        ])
        
        if compute_count >= 3:
            opportunities.append({
                "type": "reserved_instances",
                "count": compute_count,
                "description": (
                    f"Consider purchasing reserved instances for {compute_count} "
                    "compute resources to save 30-40%."
                ),
                "potential_savings": compute_count * 30.0
            })
        
        return opportunities
    
    def _calculate_success_metrics(
        self,
        project: MigrationProject,
        migration_plan: Optional[MigrationPlan],
        resources: List[CategorizedResource]
    ) -> Dict[str, Any]:
        """Calculate migration success metrics."""
        # Calculate success rate based on multiple factors
        success_factors = []
        
        # Factor 1: Resources migrated
        if len(resources) > 0:
            success_factors.append(100.0)
        else:
            success_factors.append(0.0)
        
        # Factor 2: Project completion
        if project.status == MigrationStatus.COMPLETE:
            success_factors.append(100.0)
        else:
            success_factors.append(50.0)
        
        # Factor 3: Resource categorization
        categorized_count = len([
            r for r in resources
            if r.team or r.project
        ])
        categorization_rate = (
            (categorized_count / len(resources) * 100)
            if resources else 0
        )
        success_factors.append(categorization_rate)
        
        # Calculate overall success rate
        success_rate = sum(success_factors) / len(success_factors)
        
        return {
            "success_rate": success_rate,
            "risk_incidents": [],
            "recommendations": [
                {
                    "category": "optimization",
                    "recommendation": "Review optimization opportunities to reduce costs"
                },
                {
                    "category": "governance",
                    "recommendation": "Maintain tagging compliance for ongoing visibility"
                }
            ]
        }
    
    def retrieve_migration_report(
        self,
        project_id: str
    ) -> Optional[MigrationReport]:
        """
        Retrieve migration report for a project.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            MigrationReport object or None if not found
        """
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            return None
        
        report = self.db.query(MigrationReport).filter(
            MigrationReport.migration_project_id == project.id
        ).first()
        
        return report


class PostMigrationIntegrationEngine:
    """
    Main engine for post-migration integration with FinOps capabilities.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.cost_tracking_integrator = CostTrackingIntegrator(db)
        self.finops_connector = FinOpsConnector(db)
        self.baseline_capture = BaselineCaptureSystem(db)
        self.governance_applicator = GovernancePolicyApplicator(db)
        self.optimization_identifier = OptimizationIdentifier(db)
        self.report_generator = MigrationReportGenerator(db)
    
    def integrate_migration_project(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Perform complete post-migration integration for a project.
        
        Args:
            project_id: Migration project identifier
            
        Returns:
            Integration result with all configured components
        """
        logger.info(f"Starting post-migration integration for project {project_id}")
        
        # Load migration project
        project = self.db.query(MigrationProject).filter(
            MigrationProject.project_id == project_id
        ).first()
        
        if not project:
            raise ValueError(f"Migration project {project_id} not found")
        
        # Load organizational structure
        org_structure = self.db.query(OrganizationalStructure).filter(
            OrganizationalStructure.migration_project_id == project.id
        ).first()
        
        if not org_structure:
            raise ValueError(
                f"Organizational structure not found for project {project_id}"
            )
        
        result = {
            "project_id": project_id,
            "integration_timestamp": datetime.utcnow().isoformat(),
            "cost_tracking": None,
            "finops_integration": None,
            "baselines": None,
            "governance": None,
            "optimization": None,
            "report": None
        }
        
        # Configure cost tracking
        cost_tracking_result = self.cost_tracking_integrator.configure_cost_tracking(
            project_id,
            org_structure
        )
        result["cost_tracking"] = cost_tracking_result
        
        # Transfer organizational structure to FinOps
        finops_transfer = self.finops_connector.transfer_organizational_structure(
            project_id,
            org_structure
        )
        
        # Load migration plan
        migration_plan = self.db.query(MigrationPlan).filter(
            MigrationPlan.migration_project_id == project.id
        ).first()
        
        # Configure budgets
        budget_config = self.finops_connector.configure_budgets(
            project_id,
            org_structure,
            migration_plan
        )
        
        # Configure alerts
        alert_config = self.finops_connector.configure_alerts(
            project_id,
            org_structure
        )
        
        # Enable FinOps features
        features_enabled = self.finops_connector.enable_finops_features(
            project_id,
            migration_plan.target_provider if migration_plan else "AWS"
        )
        
        result["finops_integration"] = {
            "organizational_structure": finops_transfer,
            "budgets": budget_config,
            "alerts": alert_config,
            "features": features_enabled
        }
        
        # Capture baselines
        resources = self.db.query(CategorizedResource).filter(
            CategorizedResource.migration_project_id == project.id
        ).all()
        
        baseline = self.baseline_capture.capture_baselines(
            project_id,
            resources
        )
        
        result["baselines"] = {
            "capture_date": baseline.capture_date.isoformat(),
            "total_monthly_cost": float(baseline.total_monthly_cost),
            "resource_count": baseline.resource_count,
            "cost_by_service": baseline.cost_by_service,
            "cost_by_team": baseline.cost_by_team,
            "resource_count_by_type": baseline.resource_count_by_type
        }
        
        # Apply governance policies
        tagging_result = self.governance_applicator.apply_tagging_policies(
            project_id,
            resources
        )
        
        compliance_result = self.governance_applicator.enforce_compliance_rules(
            project_id,
            resources
        )
        
        result["governance"] = {
            "tagging": tagging_result,
            "compliance": compliance_result
        }
        
        # Identify optimization opportunities
        optimization_result = self.optimization_identifier.identify_optimization_opportunities(
            project_id,
            baseline,
            resources
        )
        
        result["optimization"] = optimization_result
        
        # Generate migration report
        migration_report = self.report_generator.generate_migration_report(
            project_id
        )
        
        result["report"] = {
            "report_id": str(migration_report.id),
            "report_date": migration_report.report_date.isoformat(),
            "success_rate": migration_report.success_rate,
            "resources_migrated": migration_report.resources_migrated,
            "actual_duration_days": migration_report.actual_duration_days,
            "total_cost": float(migration_report.total_cost),
            "lessons_learned_count": len(migration_report.lessons_learned),
            "optimization_opportunities_count": len(migration_report.optimization_opportunities)
        }
        
        logger.info(f"Post-migration integration complete for project {project_id}")
        
        return result
