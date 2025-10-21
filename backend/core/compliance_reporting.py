"""
Compliance Reporting and Metrics System

This module provides comprehensive compliance dashboards, reports,
trend analysis, and executive reporting for governance program effectiveness.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import statistics
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from compliance_monitor import ComplianceMonitor, ComplianceScore, ComplianceViolation, ViolationType, ViolationSeverity
from governance_enforcer import GovernanceEnforcer, EnforcementAction, QuarantineRecord
from tagging_policy_manager import TagPolicyManager

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of compliance reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_COMPLIANCE = "detailed_compliance"
    VIOLATION_ANALYSIS = "violation_analysis"
    TREND_ANALYSIS = "trend_analysis"
    POLICY_EFFECTIVENESS = "policy_effectiveness"
    ENFORCEMENT_SUMMARY = "enforcement_summary"
    RESOURCE_INVENTORY = "resource_inventory"
    COST_IMPACT = "cost_impact"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    EXCEL = "xlsx"


class MetricType(Enum):
    """Types of compliance metrics"""
    COMPLIANCE_SCORE = "compliance_score"
    VIOLATION_COUNT = "violation_count"
    RESOLUTION_TIME = "resolution_time"
    POLICY_COVERAGE = "policy_coverage"
    ENFORCEMENT_RATE = "enforcement_rate"
    COST_SAVINGS = "cost_savings"


@dataclass
class ComplianceMetric:
    """Represents a compliance metric"""
    metric_id: str
    metric_type: MetricType
    name: str
    description: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    time_period: Dict[str, datetime]
    filters: Dict[str, Any] = field(default_factory=dict)
    include_charts: bool = True
    include_recommendations: bool = True
    recipients: List[str] = field(default_factory=list)
    schedule: Optional[str] = None  # cron expression for scheduled reports


@dataclass
class Dashboard:
    """Represents a compliance dashboard"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    permissions: Dict[str, List[str]]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceKPI:
    """Key Performance Indicator for compliance"""
    kpi_id: str
    name: str
    description: str
    target_value: float
    current_value: float
    unit: str
    trend: str  # "improving", "declining", "stable"
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def achievement_rate(self) -> float:
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)
    
    @property
    def status(self) -> str:
        rate = self.achievement_rate
        if rate >= 95:
            return "excellent"
        elif rate >= 80:
            return "good"
        elif rate >= 60:
            return "warning"
        else:
            return "critical"


class ComplianceReportingSystem:
    """
    Comprehensive compliance reporting and metrics system that provides
    dashboards, reports, trend analysis, and executive reporting.
    """
    
    def __init__(self, compliance_monitor: ComplianceMonitor, 
                 governance_enforcer: GovernanceEnforcer,
                 policy_manager: TagPolicyManager):
        self.compliance_monitor = compliance_monitor
        self.governance_enforcer = governance_enforcer
        self.policy_manager = policy_manager
        
        self.metrics_history: List[ComplianceMetric] = []
        self.dashboards: Dict[str, Dashboard] = {}
        self.report_templates: Dict[str, ReportConfiguration] = {}
        self.kpis: Dict[str, ComplianceKPI] = {}
        
        # Initialize default dashboards and KPIs
        self._initialize_default_dashboards()
        self._initialize_default_kpis()
    
    def _initialize_default_dashboards(self):
        """Initialize default compliance dashboards"""
        
        # Executive Dashboard
        executive_dashboard = Dashboard(
            dashboard_id="executive_overview",
            name="Executive Compliance Overview",
            description="High-level compliance metrics for executives",
            widgets=[
                {
                    "type": "kpi_card",
                    "title": "Overall Compliance Score",
                    "metric": "compliance_score",
                    "size": "large"
                },
                {
                    "type": "trend_chart",
                    "title": "Compliance Trend (30 days)",
                    "metric": "compliance_score",
                    "period": "30d"
                },
                {
                    "type": "violation_breakdown",
                    "title": "Violations by Severity",
                    "chart_type": "pie"
                },
                {
                    "type": "policy_effectiveness",
                    "title": "Policy Effectiveness",
                    "chart_type": "bar"
                }
            ],
            layout={"columns": 2, "responsive": True},
            permissions={"view": ["executives", "compliance_team"]}
        )
        
        # Operational Dashboard
        operational_dashboard = Dashboard(
            dashboard_id="operational_details",
            name="Operational Compliance Dashboard",
            description="Detailed compliance metrics for operations teams",
            widgets=[
                {
                    "type": "violation_list",
                    "title": "Recent Violations",
                    "limit": 20,
                    "filters": {"resolved": False}
                },
                {
                    "type": "enforcement_actions",
                    "title": "Active Enforcement Actions",
                    "status": "pending"
                },
                {
                    "type": "quarantine_status",
                    "title": "Quarantined Resources",
                    "chart_type": "table"
                },
                {
                    "type": "resource_compliance",
                    "title": "Resource Compliance by Type",
                    "chart_type": "horizontal_bar"
                }
            ],
            layout={"columns": 2, "responsive": True},
            permissions={"view": ["operations", "compliance_team", "administrators"]}
        )
        
        self.dashboards = {
            "executive_overview": executive_dashboard,
            "operational_details": operational_dashboard
        }
    
    def _initialize_default_kpis(self):
        """Initialize default compliance KPIs"""
        
        default_kpis = [
            ComplianceKPI(
                kpi_id="overall_compliance_score",
                name="Overall Compliance Score",
                description="Organization-wide compliance score",
                target_value=85.0,
                current_value=0.0,
                unit="percentage",
                trend="stable"
            ),
            ComplianceKPI(
                kpi_id="critical_violations",
                name="Critical Violations",
                description="Number of critical compliance violations",
                target_value=0.0,
                current_value=0.0,
                unit="count",
                trend="stable"
            ),
            ComplianceKPI(
                kpi_id="resolution_time",
                name="Average Resolution Time",
                description="Average time to resolve violations",
                target_value=24.0,
                current_value=0.0,
                unit="hours",
                trend="stable"
            ),
            ComplianceKPI(
                kpi_id="policy_coverage",
                name="Policy Coverage",
                description="Percentage of resources covered by policies",
                target_value=95.0,
                current_value=0.0,
                unit="percentage",
                trend="stable"
            ),
            ComplianceKPI(
                kpi_id="auto_remediation_rate",
                name="Auto-Remediation Rate",
                description="Percentage of violations auto-remediated",
                target_value=70.0,
                current_value=0.0,
                unit="percentage",
                trend="stable"
            )
        ]
        
        for kpi in default_kpis:
            self.kpis[kpi.kpi_id] = kpi
    
    def generate_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate a compliance report based on configuration"""
        
        try:
            report_data = {
                "report_id": f"{config.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "report_type": config.report_type.value,
                "generated_at": datetime.now().isoformat(),
                "time_period": {
                    "start": config.time_period["start"].isoformat(),
                    "end": config.time_period["end"].isoformat()
                },
                "filters": config.filters
            }
            
            # Generate report content based on type
            if config.report_type == ReportType.EXECUTIVE_SUMMARY:
                report_data.update(self._generate_executive_summary(config))
            
            elif config.report_type == ReportType.DETAILED_COMPLIANCE:
                report_data.update(self._generate_detailed_compliance_report(config))
            
            elif config.report_type == ReportType.VIOLATION_ANALYSIS:
                report_data.update(self._generate_violation_analysis(config))
            
            elif config.report_type == ReportType.TREND_ANALYSIS:
                report_data.update(self._generate_trend_analysis(config))
            
            elif config.report_type == ReportType.POLICY_EFFECTIVENESS:
                report_data.update(self._generate_policy_effectiveness_report(config))
            
            elif config.report_type == ReportType.ENFORCEMENT_SUMMARY:
                report_data.update(self._generate_enforcement_summary(config))
            
            else:
                raise ValueError(f"Unsupported report type: {config.report_type}")
            
            # Add recommendations if requested
            if config.include_recommendations:
                report_data["recommendations"] = self._generate_recommendations(report_data)
            
            # Generate charts if requested
            if config.include_charts:
                report_data["charts"] = self._generate_charts(report_data, config)
            
            logger.info(f"Generated {config.report_type.value} report: {report_data['report_id']}")
            return report_data
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _generate_executive_summary(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate executive summary report"""
        
        # Get current compliance score
        current_score = self.compliance_monitor.compliance_history[-1] if self.compliance_monitor.compliance_history else None
        
        # Get trend data
        trend_data = self.compliance_monitor.get_compliance_trends(days=30)
        
        # Get violation summary
        violations = self.compliance_monitor.get_violations(resolved=False)
        violation_summary = self._summarize_violations(violations)
        
        # Get enforcement statistics
        enforcement_stats = self.governance_enforcer.get_enforcement_statistics()
        
        # Calculate key metrics
        total_resources = current_score.total_resources if current_score else 0
        compliant_resources = current_score.compliant_resources if current_score else 0
        compliance_rate = (compliant_resources / total_resources * 100) if total_resources > 0 else 0
        
        return {
            "summary": {
                "compliance_score": current_score.overall_score if current_score else 0,
                "compliance_rate": compliance_rate,
                "total_resources": total_resources,
                "active_violations": len(violations),
                "critical_violations": violation_summary.get("critical", 0),
                "trend_direction": trend_data.get("trend_direction", "unknown")
            },
            "key_metrics": {
                "policy_coverage": self._calculate_policy_coverage(),
                "resolution_efficiency": self._calculate_resolution_efficiency(),
                "enforcement_effectiveness": self._calculate_enforcement_effectiveness(),
                "cost_impact": self._estimate_cost_impact(violations)
            },
            "trend_analysis": trend_data,
            "violation_breakdown": violation_summary,
            "enforcement_summary": {
                "active_actions": enforcement_stats.get("pending_actions", {}),
                "quarantined_resources": enforcement_stats.get("quarantine_statistics", {}).get("active_quarantines", 0)
            }
        }
    
    def _generate_detailed_compliance_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate detailed compliance report"""
        
        # Get violations in time period
        violations = self._get_violations_in_period(config.time_period)
        
        # Get compliance scores over time
        compliance_history = self._get_compliance_history(config.time_period)
        
        # Get policy analysis
        policy_analysis = self._analyze_policy_performance(violations)
        
        # Get resource analysis
        resource_analysis = self._analyze_resource_compliance(violations)
        
        return {
            "compliance_overview": {
                "total_violations": len(violations),
                "violation_types": self._count_by_attribute(violations, "violation_type"),
                "severity_distribution": self._count_by_attribute(violations, "severity"),
                "policy_distribution": self._count_by_attribute(violations, "policy_id")
            },
            "compliance_history": compliance_history,
            "policy_analysis": policy_analysis,
            "resource_analysis": resource_analysis,
            "violations": [self._violation_to_dict(v) for v in violations[:100]]  # Limit for report size
        }
    
    def _generate_violation_analysis(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate violation analysis report"""
        
        violations = self._get_violations_in_period(config.time_period)
        
        # Analyze violation patterns
        patterns = self._analyze_violation_patterns(violations)
        
        # Analyze resolution times
        resolution_analysis = self._analyze_resolution_times(violations)
        
        # Analyze root causes
        root_cause_analysis = self._analyze_root_causes(violations)
        
        return {
            "violation_statistics": {
                "total_violations": len(violations),
                "resolved_violations": len([v for v in violations if v.is_resolved]),
                "average_resolution_time": resolution_analysis.get("average_hours", 0),
                "most_common_type": patterns.get("most_common_type"),
                "most_affected_policy": patterns.get("most_affected_policy")
            },
            "pattern_analysis": patterns,
            "resolution_analysis": resolution_analysis,
            "root_cause_analysis": root_cause_analysis,
            "recommendations": self._generate_violation_recommendations(violations)
        }
    
    def _generate_trend_analysis(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate trend analysis report"""
        
        # Get historical data
        compliance_trends = self.compliance_monitor.get_compliance_trends(days=90)
        
        # Analyze violation trends
        violation_trends = self._analyze_violation_trends(config.time_period)
        
        # Analyze enforcement trends
        enforcement_trends = self._analyze_enforcement_trends(config.time_period)
        
        # Predict future trends
        predictions = self._predict_compliance_trends()
        
        return {
            "compliance_trends": compliance_trends,
            "violation_trends": violation_trends,
            "enforcement_trends": enforcement_trends,
            "predictions": predictions,
            "seasonal_patterns": self._identify_seasonal_patterns(),
            "improvement_opportunities": self._identify_improvement_opportunities()
        }
    
    def _generate_policy_effectiveness_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate policy effectiveness report"""
        
        policies = self.policy_manager.list_policies()
        violations = self._get_violations_in_period(config.time_period)
        
        policy_metrics = {}
        
        for policy in policies:
            policy_violations = [v for v in violations if v.policy_id == policy.policy_id]
            
            policy_metrics[policy.policy_id] = {
                "policy_name": policy.name,
                "total_violations": len(policy_violations),
                "violation_rate": len(policy_violations) / max(1, len(violations)) * 100,
                "severity_breakdown": self._count_by_attribute(policy_violations, "severity"),
                "resolution_rate": len([v for v in policy_violations if v.is_resolved]) / max(1, len(policy_violations)) * 100,
                "effectiveness_score": self._calculate_policy_effectiveness(policy, policy_violations)
            }
        
        return {
            "policy_metrics": policy_metrics,
            "most_effective_policies": sorted(
                policy_metrics.items(),
                key=lambda x: x[1]["effectiveness_score"],
                reverse=True
            )[:5],
            "least_effective_policies": sorted(
                policy_metrics.items(),
                key=lambda x: x[1]["effectiveness_score"]
            )[:5],
            "policy_recommendations": self._generate_policy_recommendations(policy_metrics)
        }
    
    def _generate_enforcement_summary(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate enforcement summary report"""
        
        enforcement_stats = self.governance_enforcer.get_enforcement_statistics()
        
        # Get enforcement actions in period
        actions = self._get_enforcement_actions_in_period(config.time_period)
        
        # Analyze enforcement effectiveness
        effectiveness = self._analyze_enforcement_effectiveness(actions)
        
        return {
            "enforcement_statistics": enforcement_stats,
            "action_analysis": {
                "total_actions": len(actions),
                "successful_actions": len([a for a in actions if a.status == "completed"]),
                "failed_actions": len([a for a in actions if a.status == "failed"]),
                "pending_actions": len([a for a in actions if a.status == "pending"])
            },
            "effectiveness_analysis": effectiveness,
            "quarantine_summary": self._summarize_quarantine_activity(config.time_period),
            "approval_summary": self._summarize_approval_activity(config.time_period)
        }
    
    def update_kpis(self):
        """Update all KPI values with current data"""
        
        try:
            # Get current compliance score
            current_score = self.compliance_monitor.compliance_history[-1] if self.compliance_monitor.compliance_history else None
            
            if current_score:
                # Update compliance score KPI
                self.kpis["overall_compliance_score"].current_value = current_score.overall_score
                self.kpis["critical_violations"].current_value = current_score.critical_violations
                
                # Update policy coverage
                self.kpis["policy_coverage"].current_value = self._calculate_policy_coverage()
                
                # Update resolution time
                self.kpis["resolution_time"].current_value = self._calculate_average_resolution_time()
                
                # Update auto-remediation rate
                self.kpis["auto_remediation_rate"].current_value = self._calculate_auto_remediation_rate()
                
                # Update trends
                for kpi in self.kpis.values():
                    kpi.trend = self._calculate_kpi_trend(kpi)
                    kpi.last_updated = datetime.now()
            
            logger.info("Updated all KPIs")
            
        except Exception as e:
            logger.error(f"Failed to update KPIs: {str(e)}")
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for a specific dashboard"""
        
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "last_updated": datetime.now().isoformat(),
            "widgets": []
        }
        
        # Generate data for each widget
        for widget_config in dashboard.widgets:
            widget_data = self._generate_widget_data(widget_config)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    def _generate_widget_data(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for a dashboard widget"""
        
        widget_type = widget_config["type"]
        
        if widget_type == "kpi_card":
            return self._generate_kpi_widget(widget_config)
        
        elif widget_type == "trend_chart":
            return self._generate_trend_widget(widget_config)
        
        elif widget_type == "violation_breakdown":
            return self._generate_violation_breakdown_widget(widget_config)
        
        elif widget_type == "violation_list":
            return self._generate_violation_list_widget(widget_config)
        
        elif widget_type == "enforcement_actions":
            return self._generate_enforcement_widget(widget_config)
        
        elif widget_type == "quarantine_status":
            return self._generate_quarantine_widget(widget_config)
        
        elif widget_type == "resource_compliance":
            return self._generate_resource_compliance_widget(widget_config)
        
        else:
            return {"type": widget_type, "error": "Unknown widget type"}
    
    def export_report(self, report_data: Dict[str, Any], format: ReportFormat, 
                     output_path: Optional[str] = None) -> str:
        """Export report to specified format"""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"compliance_report_{timestamp}.{format.value}"
        
        try:
            if format == ReportFormat.JSON:
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            elif format == ReportFormat.HTML:
                html_content = self._generate_html_report(report_data)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            
            elif format == ReportFormat.CSV:
                # Convert key metrics to CSV
                df = pd.DataFrame([report_data.get("summary", {})])
                df.to_csv(output_path, index=False)
            
            elif format == ReportFormat.EXCEL:
                # Create multi-sheet Excel report
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([report_data.get("summary", {})])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Violations sheet
                    if "violations" in report_data:
                        violations_df = pd.DataFrame(report_data["violations"])
                        violations_df.to_excel(writer, sheet_name='Violations', index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported report to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export report: {str(e)}")
            raise
    
    def schedule_report(self, config: ReportConfiguration, cron_expression: str):
        """Schedule a report for automatic generation"""
        # In a real implementation, this would integrate with a job scheduler
        logger.info(f"Scheduled {config.report_type.value} report with cron: {cron_expression}")
    
    # Helper methods for data analysis
    
    def _get_violations_in_period(self, time_period: Dict[str, datetime]) -> List[ComplianceViolation]:
        """Get violations within a time period"""
        all_violations = self.compliance_monitor.get_violations()
        return [
            v for v in all_violations
            if time_period["start"] <= v.detected_at <= time_period["end"]
        ]
    
    def _summarize_violations(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Summarize violations by severity"""
        summary = defaultdict(int)
        for violation in violations:
            summary[violation.severity.value] += 1
        return dict(summary)
    
    def _count_by_attribute(self, violations: List[ComplianceViolation], attribute: str) -> Dict[str, int]:
        """Count violations by a specific attribute"""
        counts = defaultdict(int)
        for violation in violations:
            value = getattr(violation, attribute)
            if hasattr(value, 'value'):  # Handle enums
                value = value.value
            counts[str(value)] += 1
        return dict(counts)
    
    def _calculate_policy_coverage(self) -> float:
        """Calculate percentage of resources covered by policies"""
        # Simplified calculation - in practice would analyze actual resource coverage
        total_policies = len(self.policy_manager.list_policies())
        active_policies = len(self.policy_manager.list_policies(active_only=True))
        return (active_policies / max(1, total_policies)) * 100
    
    def _calculate_resolution_efficiency(self) -> float:
        """Calculate violation resolution efficiency"""
        violations = self.compliance_monitor.get_violations()
        if not violations:
            return 100.0
        
        resolved_count = len([v for v in violations if v.is_resolved])
        return (resolved_count / len(violations)) * 100
    
    def _calculate_enforcement_effectiveness(self) -> float:
        """Calculate enforcement action effectiveness"""
        stats = self.governance_enforcer.get_enforcement_statistics()
        total_actions = sum(stats.get("pending_actions", {}).values())
        
        if total_actions == 0:
            return 100.0
        
        completed_actions = stats.get("pending_actions", {}).get("completed", 0)
        return (completed_actions / total_actions) * 100
    
    def _estimate_cost_impact(self, violations: List[ComplianceViolation]) -> Dict[str, float]:
        """Estimate cost impact of compliance violations"""
        # Simplified cost estimation
        cost_per_violation = {
            ViolationSeverity.CRITICAL: 1000.0,
            ViolationSeverity.HIGH: 500.0,
            ViolationSeverity.MEDIUM: 100.0,
            ViolationSeverity.LOW: 25.0
        }
        
        total_cost = 0.0
        breakdown = defaultdict(float)
        
        for violation in violations:
            cost = cost_per_violation.get(violation.severity, 0.0)
            total_cost += cost
            breakdown[violation.severity.value] += cost
        
        return {
            "total_estimated_cost": total_cost,
            "cost_breakdown": dict(breakdown),
            "currency": "USD"
        }
    
    def _violation_to_dict(self, violation: ComplianceViolation) -> Dict[str, Any]:
        """Convert violation object to dictionary"""
        return {
            "violation_id": violation.violation_id,
            "resource_id": violation.resource_id,
            "violation_type": violation.violation_type.value,
            "severity": violation.severity.value,
            "policy_id": violation.policy_id,
            "tag_key": violation.tag_key,
            "tag_value": violation.tag_value,
            "description": violation.description,
            "detected_at": violation.detected_at.isoformat(),
            "resolved_at": violation.resolved_at.isoformat() if violation.resolved_at else None,
            "is_resolved": violation.is_resolved
        }
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .critical { color: red; }
                .high { color: orange; }
                .medium { color: yellow; }
                .low { color: green; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Report</h1>
                <p>Generated: {generated_at}</p>
                <p>Report ID: {report_id}</p>
            </div>
            
            <h2>Summary</h2>
            <div class="metrics">
                {summary_metrics}
            </div>
            
            <h2>Details</h2>
            <pre>{details}</pre>
        </body>
        </html>
        """
        
        # Generate summary metrics HTML
        summary = report_data.get("summary", {})
        summary_html = ""
        for key, value in summary.items():
            summary_html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        
        # Format details
        details = json.dumps(report_data, indent=2, default=str)
        
        return html_template.format(
            generated_at=report_data.get("generated_at", ""),
            report_id=report_data.get("report_id", ""),
            summary_metrics=summary_html,
            details=details
        )
    
    def _calculate_average_resolution_time(self) -> float:
        """Calculate average resolution time for violations"""
        violations = self.compliance_monitor.get_violations(resolved=True)
        
        if not violations:
            return 0.0
        
        total_time = 0.0
        count = 0
        
        for violation in violations:
            if violation.resolved_at and violation.detected_at:
                resolution_time = (violation.resolved_at - violation.detected_at).total_seconds() / 3600  # hours
                total_time += resolution_time
                count += 1
        
        return total_time / count if count > 0 else 0.0
    
    def _calculate_auto_remediation_rate(self) -> float:
        """Calculate auto-remediation rate"""
        stats = self.governance_enforcer.get_enforcement_statistics()
        total_actions = sum(stats.get("pending_actions", {}).values())
        
        if total_actions == 0:
            return 100.0
        
        completed_actions = stats.get("pending_actions", {}).get("completed", 0)
        return (completed_actions / total_actions) * 100
    
    def _calculate_kpi_trend(self, kpi: ComplianceKPI) -> str:
        """Calculate KPI trend based on historical data"""
        # Simplified trend calculation
        return "stable"
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on report data"""
        recommendations = []
        
        summary = report_data.get("summary", {})
        compliance_score = summary.get("compliance_score", 100)
        
        if compliance_score < 70:
            recommendations.append("Compliance score is below acceptable threshold. Immediate action required.")
        
        if summary.get("critical_violations", 0) > 0:
            recommendations.append("Address critical violations immediately to improve compliance.")
        
        if summary.get("active_violations", 0) > 10:
            recommendations.append("High number of active violations. Consider reviewing tagging policies.")
        
        if not recommendations:
            recommendations.append("Compliance is good. Continue monitoring and maintain current practices.")
        
        return recommendations
    
    def _generate_charts(self, report_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Generate charts for the report"""
        charts = {}
        
        # Generate compliance trend chart
        if "trend_analysis" in report_data:
            charts["compliance_trend"] = {
                "type": "line_chart",
                "title": "Compliance Score Trend",
                "data": report_data["trend_analysis"]
            }
        
        # Generate violation breakdown chart
        if "violation_breakdown" in report_data:
            charts["violation_breakdown"] = {
                "type": "pie_chart", 
                "title": "Violations by Severity",
                "data": report_data["violation_breakdown"]
            }
        
        return charts
    
    # Additional helper methods would be implemented here for:
    # - _analyze_violation_patterns
    # - _analyze_resolution_times
    # - _analyze_root_causes
    # - _predict_compliance_trends
    # - _generate_widget_data methods
    # - etc.


# Example usage and testing
if __name__ == "__main__":
    from compliance_monitor import ComplianceMonitor
    from governance_enforcer import GovernanceEnforcer
    from tagging_policy_manager import TagPolicyManager
    from tag_suggestion_engine import TagSuggestionEngine
    
    # Initialize components
    policy_manager = TagPolicyManager()
    tag_engine = TagSuggestionEngine()
    monitor = ComplianceMonitor(policy_manager)
    enforcer = GovernanceEnforcer(tag_engine)
    reporting = ComplianceReportingSystem(monitor, enforcer, policy_manager)
    
    # Generate executive summary report
    config = ReportConfiguration(
        report_type=ReportType.EXECUTIVE_SUMMARY,
        format=ReportFormat.JSON,
        time_period={
            "start": datetime.now() - timedelta(days=30),
            "end": datetime.now()
        }
    )
    
    report = reporting.generate_report(config)
    print(f"Generated report: {report['report_id']}")
    
    # Update KPIs
    reporting.update_kpis()
    
    # Get dashboard data
    dashboard_data = reporting.get_dashboard_data("executive_overview")
    print(f"Dashboard widgets: {len(dashboard_data['widgets'])}")
    
    # Export report
    output_file = reporting.export_report(report, ReportFormat.JSON)
    print(f"Exported report to: {output_file}")