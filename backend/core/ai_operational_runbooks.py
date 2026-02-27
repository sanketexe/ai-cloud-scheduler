"""
AI System Operational Runbooks

This module contains comprehensive operational runbooks for managing
AI/ML systems in production, including incident response procedures,
troubleshooting guides, and maintenance protocols.
"""

from datetime import datetime
from typing import Dict, Any, List
from enum import Enum


class RunbookSeverity(Enum):
    """Runbook severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunbookCategory(Enum):
    """Runbook categories"""
    INCIDENT_RESPONSE = "incident_response"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"
    TROUBLESHOOTING = "troubleshooting"
    DEPLOYMENT = "deployment"


class AIOperationalRunbooks:
    """
    Comprehensive collection of operational runbooks for AI/ML systems.
    
    Provides structured procedures for incident response, maintenance,
    optimization, and troubleshooting of AI systems in production.
    """
    
    def __init__(self):
        self.runbooks = self._initialize_runbooks()
    
    def _initialize_runbooks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all operational runbooks"""
        return {
            "ai_system_health_degradation": {
                "title": "AI System Health Degradation Response",
                "category": RunbookCategory.INCIDENT_RESPONSE.value,
                "severity": RunbookSeverity.MEDIUM.value,
                "description": "Comprehensive response procedure for AI system health degradation",
                "estimated_time": "15-30 minutes",
                "prerequisites": [
                    "Access to monitoring dashboard",
                    "Admin privileges for AI systems",
                    "Knowledge of system architecture"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Initial Assessment",
                        "description": "Assess the scope and severity of the health degradation",
                        "actions": [
                            "Check AI system monitoring dashboard",
                            "Identify affected AI systems and components",
                            "Review current alert status and severity",
                            "Determine if issue is isolated or system-wide"
                        ],
                        "expected_time": "2-3 minutes",
                        "success_criteria": "Clear understanding of affected systems"
                    },
                    {
                        "step": 2,
                        "title": "Metrics Analysis",
                        "description": "Analyze system metrics to identify root cause",
                        "actions": [
                            "Review response time trends over last 24 hours",
                            "Check error rate patterns and spikes",
                            "Analyze resource utilization (CPU, memory, GPU)",
                            "Examine throughput and accuracy metrics"
                        ],
                        "expected_time": "3-5 minutes",
                        "success_criteria": "Identification of performance bottlenecks"
                    },
                    {
                        "step": 3,
                        "title": "Log Investigation",
                        "description": "Investigate system logs for error patterns",
                        "actions": [
                            "Review application logs for error messages",
                            "Check system logs for resource constraints",
                            "Analyze AI model inference logs",
                            "Look for correlation with recent deployments"
                        ],
                        "expected_time": "5-7 minutes",
                        "success_criteria": "Root cause hypothesis identified"
                    },
                    {
                        "step": 4,
                        "title": "Immediate Mitigation",
                        "description": "Apply immediate fixes to restore service",
                        "actions": [
                            "Restart affected AI services if needed",
                            "Scale up resources if utilization is high",
                            "Switch to backup models if available",
                            "Implement circuit breakers if cascading failures"
                        ],
                        "expected_time": "5-10 minutes",
                        "success_criteria": "Service restored to acceptable levels"
                    },
                    {
                        "step": 5,
                        "title": "Monitoring and Validation",
                        "description": "Monitor system recovery and validate fixes",
                        "actions": [
                            "Monitor metrics for improvement trends",
                            "Validate that alerts are clearing",
                            "Test critical AI system functionality",
                            "Confirm user-facing services are working"
                        ],
                        "expected_time": "5-10 minutes",
                        "success_criteria": "Sustained improvement in metrics"
                    }
                ],
                "escalation_criteria": [
                    "Response time > 5 seconds for more than 10 minutes",
                    "Accuracy drop > 10% from baseline",
                    "Error rate > 15%",
                    "System unavailable for > 5 minutes",
                    "Multiple AI systems affected simultaneously"
                ],
                "rollback_procedures": [
                    "Revert to previous AI model versions",
                    "Scale back resource changes",
                    "Restore previous configuration settings",
                    "Re-enable disabled features or services"
                ],
                "post_incident_actions": [
                    "Document incident timeline and actions taken",
                    "Update monitoring thresholds if needed",
                    "Schedule post-mortem meeting",
                    "Implement preventive measures"
                ]
            },
            
            "model_performance_degradation": {
                "title": "ML Model Performance Degradation",
                "category": RunbookCategory.INCIDENT_RESPONSE.value,
                "severity": RunbookSeverity.HIGH.value,
                "description": "Response procedure for ML model accuracy or performance issues",
                "estimated_time": "20-45 minutes",
                "prerequisites": [
                    "Access to ML model management system",
                    "Understanding of model deployment pipeline",
                    "Access to training data and model artifacts"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Performance Assessment",
                        "description": "Assess the extent of model performance degradation",
                        "actions": [
                            "Check model performance dashboard",
                            "Compare current metrics to baseline performance",
                            "Identify which models are affected",
                            "Determine business impact of degradation"
                        ],
                        "expected_time": "3-5 minutes",
                        "success_criteria": "Clear understanding of performance impact"
                    },
                    {
                        "step": 2,
                        "title": "Data Drift Analysis",
                        "description": "Analyze input data for distribution changes",
                        "actions": [
                            "Check data drift detection scores",
                            "Compare recent input distributions to training data",
                            "Analyze feature importance changes",
                            "Identify potential data quality issues"
                        ],
                        "expected_time": "5-8 minutes",
                        "success_criteria": "Understanding of data distribution changes"
                    },
                    {
                        "step": 3,
                        "title": "Model Diagnostics",
                        "description": "Run comprehensive model diagnostics",
                        "actions": [
                            "Check model inference latency trends",
                            "Analyze prediction confidence distributions",
                            "Review model resource usage patterns",
                            "Test model with known good inputs"
                        ],
                        "expected_time": "5-10 minutes",
                        "success_criteria": "Model health status determined"
                    },
                    {
                        "step": 4,
                        "title": "Immediate Response",
                        "description": "Apply immediate fixes to restore performance",
                        "actions": [
                            "Rollback to previous model version if available",
                            "Implement input data validation and filtering",
                            "Adjust model confidence thresholds",
                            "Enable fallback to rule-based systems if needed"
                        ],
                        "expected_time": "5-15 minutes",
                        "success_criteria": "Model performance stabilized"
                    },
                    {
                        "step": 5,
                        "title": "Long-term Resolution",
                        "description": "Plan and execute long-term fixes",
                        "actions": [
                            "Collect new representative training data",
                            "Trigger model retraining pipeline",
                            "Implement A/B testing for new model",
                            "Update data preprocessing pipelines"
                        ],
                        "expected_time": "10-20 minutes setup",
                        "success_criteria": "Retraining pipeline initiated"
                    }
                ],
                "escalation_criteria": [
                    "Accuracy drop > 15% from baseline",
                    "Data drift score > 50%",
                    "Model rollback fails",
                    "Business metrics significantly impacted",
                    "Multiple models affected simultaneously"
                ],
                "rollback_procedures": [
                    "Deploy previous model version",
                    "Restore previous data preprocessing",
                    "Revert confidence threshold changes",
                    "Re-enable human oversight if disabled"
                ],
                "post_incident_actions": [
                    "Analyze root cause of performance degradation",
                    "Improve data drift monitoring",
                    "Update model validation procedures",
                    "Enhance automated rollback capabilities"
                ]
            },
            
            "resource_optimization": {
                "title": "AI System Resource Optimization",
                "category": RunbookCategory.OPTIMIZATION.value,
                "severity": RunbookSeverity.LOW.value,
                "description": "Systematic approach to optimize AI system resource usage and costs",
                "estimated_time": "30-60 minutes",
                "prerequisites": [
                    "Resource monitoring access",
                    "Cost analysis tools",
                    "Change management approval for resource modifications"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Resource Analysis",
                        "description": "Analyze current resource utilization patterns",
                        "actions": [
                            "Review resource optimization recommendations",
                            "Analyze CPU, memory, and GPU utilization trends",
                            "Identify under-utilized and over-provisioned systems",
                            "Calculate current resource costs"
                        ],
                        "expected_time": "10-15 minutes",
                        "success_criteria": "Clear picture of resource utilization"
                    },
                    {
                        "step": 2,
                        "title": "Optimization Planning",
                        "description": "Plan resource allocation changes",
                        "actions": [
                            "Prioritize optimization opportunities by impact",
                            "Plan implementation during low-traffic periods",
                            "Prepare rollback procedures",
                            "Estimate cost savings and performance impact"
                        ],
                        "expected_time": "10-15 minutes",
                        "success_criteria": "Detailed optimization plan created"
                    },
                    {
                        "step": 3,
                        "title": "Gradual Implementation",
                        "description": "Implement resource changes gradually",
                        "actions": [
                            "Start with lowest-risk optimizations",
                            "Implement changes in small increments",
                            "Monitor performance impact after each change",
                            "Validate that SLA requirements are maintained"
                        ],
                        "expected_time": "15-30 minutes",
                        "success_criteria": "Optimizations implemented without issues"
                    },
                    {
                        "step": 4,
                        "title": "Validation and Documentation",
                        "description": "Validate optimizations and document results",
                        "actions": [
                            "Monitor system performance for 24-48 hours",
                            "Calculate actual cost savings achieved",
                            "Document optimization results and lessons learned",
                            "Update resource allocation baselines"
                        ],
                        "expected_time": "Ongoing monitoring",
                        "success_criteria": "Optimizations validated and documented"
                    }
                ],
                "escalation_criteria": [
                    "Resource costs exceed budget by > 20%",
                    "Performance degradation after optimization",
                    "System instability after resource changes",
                    "SLA violations due to resource constraints"
                ],
                "rollback_procedures": [
                    "Restore previous resource allocations",
                    "Revert scaling policies",
                    "Re-enable disabled optimization features",
                    "Restore backup configurations"
                ],
                "post_incident_actions": [
                    "Update resource optimization algorithms",
                    "Improve cost monitoring and alerting",
                    "Refine optimization recommendation logic",
                    "Schedule regular optimization reviews"
                ]
            },
            
            "alert_storm_management": {
                "title": "AI System Alert Storm Management",
                "category": RunbookCategory.INCIDENT_RESPONSE.value,
                "severity": RunbookSeverity.CRITICAL.value,
                "description": "Procedure for handling multiple simultaneous AI system alerts",
                "estimated_time": "10-20 minutes",
                "prerequisites": [
                    "Alert management system access",
                    "Incident coordination tools",
                    "Team communication channels"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Alert Triage",
                        "description": "Quickly triage and prioritize alerts",
                        "actions": [
                            "Identify root cause of alert storm",
                            "Group related alerts by system or cause",
                            "Prioritize critical system alerts first",
                            "Suppress duplicate or secondary alerts"
                        ],
                        "expected_time": "2-3 minutes",
                        "success_criteria": "Alert priorities established"
                    },
                    {
                        "step": 2,
                        "title": "Immediate Stabilization",
                        "description": "Stabilize critical systems first",
                        "actions": [
                            "Focus on system-wide issues before individual components",
                            "Implement emergency scaling if resource-related",
                            "Enable circuit breakers to prevent cascading failures",
                            "Temporarily increase alert thresholds if needed"
                        ],
                        "expected_time": "5-10 minutes",
                        "success_criteria": "Critical systems stabilized"
                    },
                    {
                        "step": 3,
                        "title": "Team Coordination",
                        "description": "Coordinate team response to avoid conflicts",
                        "actions": [
                            "Establish incident commander role",
                            "Assign team members to specific systems",
                            "Set up communication channels",
                            "Avoid duplicate remediation efforts"
                        ],
                        "expected_time": "2-5 minutes",
                        "success_criteria": "Team coordination established"
                    },
                    {
                        "step": 4,
                        "title": "Systematic Resolution",
                        "description": "Systematically resolve remaining issues",
                        "actions": [
                            "Address alerts in priority order",
                            "Document actions taken for each alert",
                            "Monitor for new alerts or escalations",
                            "Communicate status updates regularly"
                        ],
                        "expected_time": "Ongoing",
                        "success_criteria": "Alert count decreasing steadily"
                    }
                ],
                "escalation_criteria": [
                    "> 10 critical alerts in 5 minutes",
                    "Multiple AI systems affected simultaneously",
                    "Customer-facing services impacted",
                    "Alert storm continues for > 30 minutes"
                ],
                "rollback_procedures": [
                    "Revert recent deployments or changes",
                    "Restore previous system configurations",
                    "Re-enable disabled services",
                    "Reset alert thresholds to normal levels"
                ],
                "post_incident_actions": [
                    "Conduct thorough post-mortem analysis",
                    "Improve alert correlation and grouping",
                    "Update alert storm detection logic",
                    "Enhance automated response capabilities"
                ]
            },
            
            "data_drift_response": {
                "title": "Data Drift Detection Response",
                "category": RunbookCategory.TROUBLESHOOTING.value,
                "severity": RunbookSeverity.MEDIUM.value,
                "description": "Comprehensive response to detected data drift in ML models",
                "estimated_time": "45-90 minutes",
                "prerequisites": [
                    "Data drift monitoring access",
                    "Model retraining capabilities",
                    "Data pipeline management access"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Drift Validation",
                        "description": "Confirm and analyze data drift detection",
                        "actions": [
                            "Validate data drift detection accuracy",
                            "Analyze drift patterns and affected features",
                            "Compare current data to training distribution",
                            "Assess statistical significance of drift"
                        ],
                        "expected_time": "10-15 minutes",
                        "success_criteria": "Drift confirmed and characterized"
                    },
                    {
                        "step": 2,
                        "title": "Root Cause Investigation",
                        "description": "Investigate the source of data changes",
                        "actions": [
                            "Review recent changes to data sources",
                            "Check for upstream system modifications",
                            "Analyze seasonal or temporal patterns",
                            "Investigate data quality issues"
                        ],
                        "expected_time": "15-20 minutes",
                        "success_criteria": "Root cause identified"
                    },
                    {
                        "step": 3,
                        "title": "Business Impact Assessment",
                        "description": "Assess the business impact of detected drift",
                        "actions": [
                            "Evaluate model performance degradation",
                            "Assess impact on business metrics",
                            "Determine urgency of response needed",
                            "Communicate findings to stakeholders"
                        ],
                        "expected_time": "10-15 minutes",
                        "success_criteria": "Impact assessment completed"
                    },
                    {
                        "step": 4,
                        "title": "Data Collection and Retraining",
                        "description": "Collect new data and retrain models",
                        "actions": [
                            "Collect new representative training data",
                            "Update data preprocessing pipelines",
                            "Retrain model with updated data distribution",
                            "Validate new model performance"
                        ],
                        "expected_time": "20-40 minutes",
                        "success_criteria": "New model trained and validated"
                    },
                    {
                        "step": 5,
                        "title": "Deployment and Monitoring",
                        "description": "Deploy updated model and enhance monitoring",
                        "actions": [
                            "Deploy new model using A/B testing",
                            "Implement enhanced drift monitoring",
                            "Update drift detection thresholds",
                            "Monitor model performance post-deployment"
                        ],
                        "expected_time": "10-15 minutes",
                        "success_criteria": "Updated model deployed successfully"
                    }
                ],
                "escalation_criteria": [
                    "Drift score > 70%",
                    "Multiple models affected simultaneously",
                    "Prediction accuracy drops > 20%",
                    "Business metrics significantly impacted",
                    "Unable to identify root cause within 2 hours"
                ],
                "rollback_procedures": [
                    "Revert to previous model version",
                    "Restore previous data preprocessing",
                    "Re-enable human validation workflows",
                    "Implement temporary data filtering"
                ],
                "post_incident_actions": [
                    "Improve drift detection sensitivity",
                    "Enhance automated retraining pipelines",
                    "Update data quality monitoring",
                    "Document drift patterns for future reference"
                ]
            },
            
            "ai_system_deployment": {
                "title": "AI System Deployment Procedures",
                "category": RunbookCategory.DEPLOYMENT.value,
                "severity": RunbookSeverity.MEDIUM.value,
                "description": "Safe deployment procedures for AI/ML systems",
                "estimated_time": "60-120 minutes",
                "prerequisites": [
                    "Deployment pipeline access",
                    "Testing environment availability",
                    "Rollback procedures prepared"
                ],
                "steps": [
                    {
                        "step": 1,
                        "title": "Pre-deployment Validation",
                        "description": "Validate system readiness for deployment",
                        "actions": [
                            "Run comprehensive test suite",
                            "Validate model performance metrics",
                            "Check resource requirements and availability",
                            "Verify configuration and dependencies"
                        ],
                        "expected_time": "15-20 minutes",
                        "success_criteria": "All validation checks pass"
                    },
                    {
                        "step": 2,
                        "title": "Staged Deployment",
                        "description": "Deploy using staged rollout approach",
                        "actions": [
                            "Deploy to staging environment first",
                            "Run integration tests in staging",
                            "Deploy to production with traffic splitting",
                            "Gradually increase traffic to new version"
                        ],
                        "expected_time": "30-45 minutes",
                        "success_criteria": "Successful staged deployment"
                    },
                    {
                        "step": 3,
                        "title": "Monitoring and Validation",
                        "description": "Monitor deployment and validate functionality",
                        "actions": [
                            "Monitor key performance indicators",
                            "Validate AI system functionality",
                            "Check for errors or performance degradation",
                            "Confirm business metrics remain stable"
                        ],
                        "expected_time": "15-30 minutes",
                        "success_criteria": "System performing as expected"
                    },
                    {
                        "step": 4,
                        "title": "Full Rollout or Rollback",
                        "description": "Complete rollout or initiate rollback",
                        "actions": [
                            "Complete traffic migration if successful",
                            "Initiate rollback if issues detected",
                            "Update monitoring and alerting",
                            "Document deployment results"
                        ],
                        "expected_time": "15-25 minutes",
                        "success_criteria": "Deployment completed or rolled back"
                    }
                ],
                "escalation_criteria": [
                    "Deployment validation failures",
                    "Performance degradation > 20%",
                    "Error rate increase > 10%",
                    "Rollback procedures fail",
                    "Business impact detected"
                ],
                "rollback_procedures": [
                    "Immediate traffic routing to previous version",
                    "Database rollback if schema changes made",
                    "Configuration rollback",
                    "Dependency version rollback"
                ],
                "post_incident_actions": [
                    "Update deployment procedures based on lessons learned",
                    "Improve automated testing coverage",
                    "Enhance rollback automation",
                    "Document deployment best practices"
                ]
            }
        }
    
    def get_runbook(self, runbook_id: str) -> Dict[str, Any]:
        """Get a specific runbook by ID"""
        if runbook_id not in self.runbooks:
            raise ValueError(f"Runbook {runbook_id} not found")
        
        runbook = self.runbooks[runbook_id].copy()
        runbook["id"] = runbook_id
        runbook["last_updated"] = datetime.utcnow().isoformat()
        
        return runbook
    
    def get_runbooks_by_category(self, category: RunbookCategory) -> List[Dict[str, Any]]:
        """Get all runbooks in a specific category"""
        return [
            {**runbook, "id": runbook_id}
            for runbook_id, runbook in self.runbooks.items()
            if runbook["category"] == category.value
        ]
    
    def get_runbooks_by_severity(self, severity: RunbookSeverity) -> List[Dict[str, Any]]:
        """Get all runbooks with a specific severity level"""
        return [
            {**runbook, "id": runbook_id}
            for runbook_id, runbook in self.runbooks.items()
            if runbook["severity"] == severity.value
        ]
    
    def get_all_runbooks(self) -> Dict[str, Dict[str, Any]]:
        """Get all available runbooks"""
        return {
            runbook_id: {**runbook, "id": runbook_id}
            for runbook_id, runbook in self.runbooks.items()
        }
    
    def search_runbooks(self, query: str) -> List[Dict[str, Any]]:
        """Search runbooks by title, description, or content"""
        query_lower = query.lower()
        results = []
        
        for runbook_id, runbook in self.runbooks.items():
            # Search in title and description
            if (query_lower in runbook["title"].lower() or 
                query_lower in runbook["description"].lower()):
                results.append({**runbook, "id": runbook_id})
                continue
            
            # Search in steps
            for step in runbook.get("steps", []):
                if (query_lower in step.get("title", "").lower() or
                    query_lower in step.get("description", "").lower()):
                    results.append({**runbook, "id": runbook_id})
                    break
        
        return results


# Global runbooks instance
_operational_runbooks = None

def get_operational_runbooks() -> AIOperationalRunbooks:
    """Get global operational runbooks instance"""
    global _operational_runbooks
    if _operational_runbooks is None:
        _operational_runbooks = AIOperationalRunbooks()
    return _operational_runbooks