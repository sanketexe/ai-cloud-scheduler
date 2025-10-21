"""
Integrated Tagging Compliance and Governance System

This module integrates all tagging compliance components into a unified system
providing comprehensive policy management, monitoring, enforcement, and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from tagging_policy_manager import TagPolicyManager, TaggingPolicy
from compliance_monitor import ComplianceMonitor, CloudResource, ComplianceViolation
from tag_suggestion_engine import TagSuggestionEngine, ResourceContext, BulkTaggingJob
from governance_enforcer import GovernanceEnforcer, EnforcementAction, ApprovalRequest
from compliance_reporting import ComplianceReportingSystem, ReportConfiguration, ReportType, ReportFormat

logger = logging.getLogger(__name__)


@dataclass
class SystemConfiguration:
    """Configuration for the tagging compliance system"""
    auto_enforcement_enabled: bool = True
    auto_tagging_threshold: float = 0.8
    scan_interval_minutes: int = 60
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    quarantine_enabled: bool = True
    approval_required_for_deletion: bool = True
    executive_reporting_enabled: bool = True
    
    
class TaggingComplianceSystem:
    """
    Integrated tagging compliance and governance system that provides
    comprehensive policy management, monitoring, enforcement, and reporting.
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # Initialize core components
        self.policy_manager = TagPolicyManager()
        self.tag_suggestion_engine = TagSuggestionEngine()
        self.compliance_monitor = ComplianceMonitor(self.policy_manager)
        self.governance_enforcer = GovernanceEnforcer(self.tag_suggestion_engine)
        self.reporting_system = ComplianceReportingSystem(
            self.compliance_monitor,
            self.governance_enforcer,
            self.policy_manager
        )
        
        # System state
        self.is_running = False
        self.scan_task: Optional[asyncio.Task] = None
        
        # Setup callbacks and integrations
        self._setup_integrations()
        
        logger.info("Tagging Compliance System initialized")
    
    def _setup_integrations(self):
        """Setup integrations between components"""
        
        # Setup compliance monitor callbacks
        self.compliance_monitor.add_violation_callback(self._handle_violation_detected)
        self.compliance_monitor.add_compliance_callback(self._handle_compliance_update)
        
        # Setup governance enforcer callbacks
        self.governance_enforcer.add_notification_callback(self._handle_enforcement_notification)
        self.governance_enforcer.add_approval_callback(self._handle_approval_request)
    
    async def start_system(self):
        """Start the compliance monitoring and enforcement system"""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        self.is_running = True
        
        # Start continuous compliance monitoring
        self.scan_task = asyncio.create_task(self._continuous_monitoring())
        
        logger.info("Tagging Compliance System started")
    
    async def stop_system(self):
        """Stop the compliance monitoring and enforcement system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Tagging Compliance System stopped")
    
    async def _continuous_monitoring(self):
        """Continuous compliance monitoring loop"""
        while self.is_running:
            try:
                # Perform compliance scan
                await self._perform_compliance_scan()
                
                # Process scheduled enforcement actions
                await self.governance_enforcer.process_scheduled_actions()
                
                # Update KPIs and metrics
                self.reporting_system.update_kpis()
                
                # Wait for next scan interval
                await asyncio.sleep(self.config.scan_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_compliance_scan(self):
        """Perform a compliance scan of all resources"""
        try:
            # In a real implementation, this would fetch resources from cloud providers
            # For now, we'll use a simulated resource list
            resources = self._get_all_resources()
            
            if resources:
                compliance_score = await self.compliance_monitor.scan_resources(resources)
                logger.info(f"Compliance scan completed. Score: {compliance_score.overall_score:.2f}")
            
        except Exception as e:
            logger.error(f"Compliance scan failed: {str(e)}")
    
    def _get_all_resources(self) -> List[CloudResource]:
        """Get all cloud resources for scanning (simulated)"""
        # In a real implementation, this would integrate with cloud provider APIs
        # to discover and fetch all resources
        
        sample_resources = [
            CloudResource(
                resource_id="i-1234567890abcdef0",
                resource_type="ec2_instance",
                provider="aws",
                region="us-east-1",
                tags={"Name": "web-server-prod", "Environment": "prod"},
                attributes={"instance_type": "t3.medium"},
                created_at=datetime.now() - timedelta(days=30),
                last_modified=datetime.now()
            ),
            CloudResource(
                resource_id="vol-0987654321fedcba0",
                resource_type="ebs_volume",
                provider="aws",
                region="us-east-1",
                tags={},  # Untagged resource
                attributes={"size": "100GB", "volume_type": "gp3"},
                created_at=datetime.now() - timedelta(days=15),
                last_modified=datetime.now()
            ),
            CloudResource(
                resource_id="sg-abcdef1234567890",
                resource_type="security_group",
                provider="aws",
                region="us-east-1",
                tags={"Environment": "invalid_env"},  # Invalid tag value
                attributes={"vpc_id": "vpc-12345678"},
                created_at=datetime.now() - timedelta(days=7),
                last_modified=datetime.now()
            )
        ]
        
        return sample_resources
    
    def _handle_violation_detected(self, violation: ComplianceViolation):
        """Handle newly detected compliance violations"""
        logger.info(f"Violation detected: {violation.violation_id} - {violation.description}")
        
        if self.config.auto_enforcement_enabled:
            # Create resource context for enforcement
            resource_context = self._create_resource_context(violation.resource_id)
            
            # Schedule async processing
            asyncio.create_task(self._process_violation_async(violation, resource_context))
    
    async def _process_violation_async(self, violation: ComplianceViolation, resource_context: Optional[ResourceContext]):
        """Async processing of violations"""
        try:
            # Process violation through governance enforcer
            action_ids = await self.governance_enforcer.process_violation(violation, resource_context)
            
            if action_ids:
                logger.info(f"Created {len(action_ids)} enforcement actions for violation {violation.violation_id}")
        except Exception as e:
            logger.error(f"Failed to process violation {violation.violation_id}: {str(e)}")
    
    def _handle_compliance_update(self, compliance_score):
        """Handle compliance score updates"""
        logger.info(f"Compliance score updated: {compliance_score.overall_score:.2f}")
        
        # Trigger alerts for low compliance scores
        if compliance_score.overall_score < 70:
            self._send_compliance_alert(compliance_score)
    
    def _handle_enforcement_notification(self, notification_data: Dict[str, Any]):
        """Handle enforcement notifications"""
        logger.info(f"Enforcement notification: {notification_data.get('type')} - {notification_data.get('message')}")
        
        # In a real implementation, this would send notifications via configured channels
        # (email, Slack, Teams, etc.)
    
    def _handle_approval_request(self, approval_request: ApprovalRequest):
        """Handle approval requests"""
        logger.info(f"Approval request created: {approval_request.request_id}")
        
        # In a real implementation, this would notify approvers via configured channels
    
    def _create_resource_context(self, resource_id: str) -> Optional[ResourceContext]:
        """Create resource context for a given resource ID"""
        # In a real implementation, this would fetch resource details from cloud provider
        # For now, return a simulated context
        
        return ResourceContext(
            resource_id=resource_id,
            resource_type="ec2_instance",
            resource_name=f"resource-{resource_id}",
            provider="aws",
            region="us-east-1",
            account_id="123456789012",
            existing_tags={},
            resource_attributes={}
        )
    
    def _send_compliance_alert(self, compliance_score):
        """Send compliance alert for low scores"""
        alert_data = {
            "type": "compliance_alert",
            "severity": "high" if compliance_score.overall_score < 50 else "medium",
            "message": f"Compliance score is {compliance_score.overall_score:.2f}% - below acceptable threshold",
            "score": compliance_score.overall_score,
            "violations": compliance_score.violation_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send via configured notification channels
        self._handle_enforcement_notification(alert_data)
    
    # Public API methods
    
    def create_tagging_policy(self, policy: TaggingPolicy) -> bool:
        """Create a new tagging policy"""
        return self.policy_manager.create_policy(policy)
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        current_score = (self.compliance_monitor.compliance_history[-1] 
                        if self.compliance_monitor.compliance_history else None)
        
        violations = self.compliance_monitor.get_violations(resolved=False)
        enforcement_stats = self.governance_enforcer.get_enforcement_statistics()
        
        return {
            "compliance_score": current_score.overall_score if current_score else 0,
            "total_resources": current_score.total_resources if current_score else 0,
            "active_violations": len(violations),
            "critical_violations": len([v for v in violations if v.severity.value == "critical"]),
            "enforcement_actions": enforcement_stats.get("pending_actions", {}),
            "last_scan": self.compliance_monitor.last_scan_time.isoformat() if self.compliance_monitor.last_scan_time else None
        }
    
    def suggest_tags_for_resource(self, resource_context: ResourceContext) -> List[Dict[str, Any]]:
        """Get tag suggestions for a resource"""
        suggestions = self.tag_suggestion_engine.suggest_tags(resource_context)
        
        return [
            {
                "tag_key": s.tag_key,
                "suggested_value": s.suggested_value,
                "confidence": s.confidence.value,
                "confidence_score": s.confidence_score,
                "reasoning": s.reasoning
            }
            for s in suggestions
        ]
    
    def create_bulk_tagging_job(self, resource_contexts: List[ResourceContext]) -> str:
        """Create a bulk tagging job"""
        job = self.tag_suggestion_engine.create_bulk_tagging_job(
            resource_contexts, 
            auto_apply_threshold=self.config.auto_tagging_threshold
        )
        return job.job_id
    
    def execute_bulk_tagging_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a bulk tagging job"""
        return self.tag_suggestion_engine.execute_bulk_tagging_job(job_id)
    
    def approve_enforcement_action(self, request_id: str, approver: str, comments: str = "") -> bool:
        """Approve an enforcement action"""
        return self.governance_enforcer.approve_action(request_id, approver, comments)
    
    def reject_enforcement_action(self, request_id: str, approver: str, reason: str) -> bool:
        """Reject an enforcement action"""
        return self.governance_enforcer.reject_action(request_id, approver, reason)
    
    def create_governance_exception(self, resource_id: str, policy_id: str, 
                                  justification: str, approved_by: str, 
                                  duration_days: int = 30) -> str:
        """Create a governance exception"""
        return self.governance_enforcer.create_governance_exception(
            resource_id, policy_id, None, justification, approved_by, duration_days
        )
    
    def generate_compliance_report(self, report_type: ReportType, 
                                 days_back: int = 30) -> Dict[str, Any]:
        """Generate a compliance report"""
        config = ReportConfiguration(
            report_type=report_type,
            format=ReportFormat.JSON,
            time_period={
                "start": datetime.now() - timedelta(days=days_back),
                "end": datetime.now()
            }
        )
        
        return self.reporting_system.generate_report(config)
    
    def get_dashboard_data(self, dashboard_id: str = "executive_overview") -> Dict[str, Any]:
        """Get dashboard data"""
        return self.reporting_system.get_dashboard_data(dashboard_id)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "policy_manager": {
                "total_policies": len(self.policy_manager.policies),
                "active_policies": len(self.policy_manager.list_policies(active_only=True)),
                "policy_conflicts": len(self.policy_manager.get_policy_conflicts())
            },
            "compliance_monitor": self.compliance_monitor.get_scan_statistics(),
            "governance_enforcer": self.governance_enforcer.get_enforcement_statistics(),
            "tag_suggestion_engine": self.tag_suggestion_engine.get_suggestion_statistics(),
            "system_config": {
                "auto_enforcement_enabled": self.config.auto_enforcement_enabled,
                "scan_interval_minutes": self.config.scan_interval_minutes,
                "auto_tagging_threshold": self.config.auto_tagging_threshold
            }
        }


# Example usage and integration test
async def main():
    """Example usage of the integrated tagging compliance system"""
    
    # Initialize system
    config = SystemConfiguration(
        auto_enforcement_enabled=True,
        auto_tagging_threshold=0.7,
        scan_interval_minutes=30
    )
    
    system = TaggingComplianceSystem(config)
    
    # Create a sample policy using template
    financial_policy = system.policy_manager.create_policy_from_template(
        template_id="financial_governance",
        policy_id="prod_financial_policy",
        name="Production Financial Governance",
        scope_filter={"environment": "prod"}
    )
    
    if financial_policy:
        print(f"Created policy: {financial_policy.policy_id}")
    
    # Start the system
    await system.start_system()
    
    # Let it run for a short time to demonstrate functionality
    await asyncio.sleep(5)
    
    # Get compliance status
    status = system.get_compliance_status()
    print(f"Compliance Status: {status}")
    
    # Generate executive report
    report = system.generate_compliance_report(ReportType.EXECUTIVE_SUMMARY)
    print(f"Generated report: {report['report_id']}")
    
    # Get system statistics
    stats = system.get_system_statistics()
    print(f"System Statistics: {stats}")
    
    # Stop the system
    await system.stop_system()
    print("System stopped")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())