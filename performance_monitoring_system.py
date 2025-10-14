# performance_monitoring_system.py
"""
Comprehensive Performance Monitoring and Health Management System

This module integrates all performance monitoring components:
- MetricsCollector: Gathers performance data from cloud resources
- AnomalyDetector: ML-based anomaly detection for performance issues
- AlertManager: Intelligent alerting system with noise reduction
- HealthChecker: Resource health and availability monitoring
- PerformanceAnalyzer: Trend analysis and capacity planning

Requirements addressed:
- 3.1: Performance metrics collection system
- 3.2: Anomaly detection and alerting
- 3.3: Health monitoring and trend analysis
- 3.4: Resource health and availability monitoring
- 3.5: Trend analysis and capacity planning
- 3.6: Scaling recommendation engine
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from performance_monitor import (
    MetricsCollector, CloudResource, MetricType, MetricsData,
    AWSCloudWatchAPI, GCPMonitoringAPI, AzureMonitorAPI, CloudProvider
)
from anomaly_detector import AnomalyDetector, Anomaly
from alert_manager import AlertManager, Alert, AlertRule
from health_checker import HealthChecker, ResourceHealth, HealthCheck
from performance_analyzer import (
    PerformanceAnalyzer, PerformanceTrends, CapacityForecast, 
    ScalingRecommendation, ResourceCapacity
)
from enhanced_models import SeverityLevel


@dataclass
class MonitoringConfig:
    """Configuration for the performance monitoring system"""
    metrics_collection_interval_minutes: int = 5
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = 0.1
    health_check_interval_minutes: int = 5
    trend_analysis_period_days: int = 30
    capacity_forecast_days: int = 90
    alert_cooldown_minutes: int = 15
    enable_auto_scaling_recommendations: bool = True


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report"""
    report_id: str
    generated_at: datetime
    time_period_start: datetime
    time_period_end: datetime
    
    # Metrics summary
    total_resources_monitored: int
    total_metrics_collected: int
    
    # Health summary
    healthy_resources: int
    warning_resources: int
    unhealthy_resources: int
    average_health_score: float
    
    # Anomalies and alerts
    total_anomalies_detected: int
    total_alerts_triggered: int
    critical_alerts: int
    
    # Trends and forecasts
    resources_with_increasing_trends: int
    resources_needing_scaling: int
    capacity_exhaustion_warnings: int
    
    # Recommendations
    scaling_recommendations: List[ScalingRecommendation] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    # Detailed data
    resource_health_details: Dict[str, ResourceHealth] = field(default_factory=dict)
    performance_trends: Dict[str, PerformanceTrends] = field(default_factory=dict)
    capacity_forecasts: Dict[str, CapacityForecast] = field(default_factory=dict)


class PerformanceMonitoringSystem:
    """Comprehensive performance monitoring and health management system"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitoringSystem")
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector(sensitivity=self.config.anomaly_sensitivity)
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # System state
        self.monitored_resources: Dict[str, CloudResource] = {}
        self.resource_capacities: Dict[str, ResourceCapacity] = {}
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Performance Monitoring System initialized")
    
    def setup_cloud_providers(self, provider_configs: Dict[CloudProvider, Dict[str, Any]]):
        """Setup cloud provider integrations"""
        for provider, config in provider_configs.items():
            if provider == CloudProvider.AWS:
                api = AWSCloudWatchAPI(
                    access_key=config['access_key'],
                    secret_key=config['secret_key'],
                    region=config.get('region', 'us-east-1')
                )
                self.metrics_collector.register_monitoring_api(provider, api)
            
            elif provider == CloudProvider.GCP:
                api = GCPMonitoringAPI(
                    project_id=config['project_id'],
                    credentials_path=config['credentials_path']
                )
                self.metrics_collector.register_monitoring_api(provider, api)
            
            elif provider == CloudProvider.AZURE:
                api = AzureMonitorAPI(
                    subscription_id=config['subscription_id'],
                    tenant_id=config['tenant_id'],
                    client_id=config['client_id'],
                    client_secret=config['client_secret']
                )
                self.metrics_collector.register_monitoring_api(provider, api)
        
        self.logger.info(f"Configured {len(provider_configs)} cloud providers")
    
    def add_resources_to_monitor(self, resources: List[CloudResource], 
                               capacities: Optional[Dict[str, ResourceCapacity]] = None):
        """Add resources to monitoring"""
        for resource in resources:
            self.monitored_resources[resource.resource_id] = resource
            
            # Add default capacity if not provided
            if capacities and resource.resource_id in capacities:
                self.resource_capacities[resource.resource_id] = capacities[resource.resource_id]
            else:
                # Default capacity for demonstration
                self.resource_capacities[resource.resource_id] = ResourceCapacity(
                    cpu_cores=2,
                    memory_gb=8.0,
                    disk_gb=100.0,
                    network_bandwidth_mbps=1000.0,
                    instance_type="default"
                )
        
        self.logger.info(f"Added {len(resources)} resources to monitoring")
    
    def setup_default_monitoring(self):
        """Setup default monitoring configuration"""
        # Create default health checks
        self.health_checker.create_default_health_checks()
        
        # Create default alert rules
        self.alert_manager.create_default_alert_rules()
        
        # Setup notification channels (examples)
        self.alert_manager.register_notification_channel("email_ops", {
            'type': 'email',
            'recipients': ['ops-team@company.com'],
            'smtp_server': 'smtp.company.com'
        })
        
        self.alert_manager.register_notification_channel("slack_alerts", {
            'type': 'slack',
            'webhook_url': 'https://hooks.slack.com/services/...',
            'channel': '#alerts'
        })
        
        self.logger.info("Default monitoring configuration applied")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            self.logger.warning("Monitoring system is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                await self._execute_monitoring_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.metrics_collection_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _execute_monitoring_cycle(self):
        """Execute one complete monitoring cycle"""
        cycle_start = datetime.now()
        self.logger.info("Starting monitoring cycle")
        
        if not self.monitored_resources:
            self.logger.warning("No resources to monitor")
            return
        
        # 1. Collect metrics
        resources = list(self.monitored_resources.values())
        metrics_data = await self.metrics_collector.collect_metrics(
            resources=resources,
            metric_types=[
                MetricType.CPU_UTILIZATION,
                MetricType.MEMORY_UTILIZATION,
                MetricType.DISK_IO,
                MetricType.NETWORK_IO,
                MetricType.RESPONSE_TIME,
                MetricType.THROUGHPUT,
                MetricType.ERROR_RATE,
                MetricType.AVAILABILITY
            ]
        )
        
        # 2. Normalize and store metrics
        normalized_metrics = self.metrics_collector.normalize_metrics(metrics_data)
        self.metrics_collector.store_metrics(normalized_metrics)
        
        # 3. Detect anomalies (if enabled)
        anomalies = []
        if self.config.anomaly_detection_enabled:
            anomalies = self.anomaly_detector.detect_anomalies(normalized_metrics)
        
        # 4. Execute health checks
        resource_health = await self.health_checker.execute_health_checks(
            resources, normalized_metrics
        )
        
        # 5. Evaluate alerts
        triggered_alerts = self.alert_manager.evaluate_alerts(
            normalized_metrics, anomalies
        )
        
        # 6. Analyze performance trends (less frequent)
        if cycle_start.minute % 30 == 0:  # Every 30 minutes
            performance_trends = self.performance_analyzer.analyze_performance_trends(
                normalized_metrics, self.config.trend_analysis_period_days
            )
            
            # Generate scaling recommendations
            if self.config.enable_auto_scaling_recommendations:
                scaling_recommendations = self.performance_analyzer.generate_scaling_recommendations(
                    performance_trends, self.resource_capacities
                )
                
                # Log recommendations
                for rec in scaling_recommendations:
                    self.logger.info(f"Scaling recommendation for {rec.resource_id}: "
                                   f"{rec.scaling_direction.value} - {rec.rationale}")
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        self.logger.info(f"Monitoring cycle completed in {cycle_duration:.2f} seconds - "
                        f"Metrics: {len(normalized_metrics)}, "
                        f"Anomalies: {len(anomalies)}, "
                        f"Alerts: {len(triggered_alerts)}")
    
    async def generate_monitoring_report(self, 
                                       time_period_hours: int = 24) -> MonitoringReport:
        """Generate comprehensive monitoring report"""
        report_end = datetime.now()
        report_start = report_end - timedelta(hours=time_period_hours)
        
        # Collect current metrics for analysis
        resources = list(self.monitored_resources.values())
        current_metrics = await self.metrics_collector.collect_metrics(
            resources=resources,
            metric_types=[MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION],
            start_time=report_start,
            end_time=report_end
        )
        
        # Get health summary
        health_summary = self.health_checker.get_health_summary()
        
        # Get anomaly summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary(time_period_hours)
        
        # Get alert statistics
        alert_stats = self.alert_manager.get_alert_statistics(time_period_hours)
        
        # Analyze trends
        performance_trends = self.performance_analyzer.analyze_performance_trends(
            current_metrics, min(time_period_hours // 24, 30)
        )
        
        # Generate capacity forecasts
        capacity_forecasts = {}
        for resource_id, trends in performance_trends.items():
            forecast = self.performance_analyzer.create_capacity_forecast(
                resource_id, trends, self.config.capacity_forecast_days
            )
            capacity_forecasts[resource_id] = forecast
        
        # Generate scaling recommendations
        scaling_recommendations = self.performance_analyzer.generate_scaling_recommendations(
            performance_trends, self.resource_capacities
        )
        
        # Count resources with increasing trends
        increasing_trends = len([
            t for t in performance_trends.values()
            if t.cpu_trend.value == "increasing" or t.memory_trend.value == "increasing"
        ])
        
        # Count capacity exhaustion warnings
        capacity_warnings = len([
            f for f in capacity_forecasts.values()
            if any(date and date < datetime.now() + timedelta(days=60) 
                  for date in f.capacity_exhaustion_dates.values())
        ])
        
        # Generate optimization opportunities
        optimization_opportunities = []
        for rec in scaling_recommendations:
            if rec.scaling_direction.value == "down":
                optimization_opportunities.append(
                    f"Resource {rec.resource_id}: Potential cost savings through downsizing"
                )
            elif rec.urgency.value in ["high", "critical"]:
                optimization_opportunities.append(
                    f"Resource {rec.resource_id}: Urgent scaling required to prevent performance issues"
                )
        
        report = MonitoringReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=report_end,
            time_period_start=report_start,
            time_period_end=report_end,
            
            # Metrics summary
            total_resources_monitored=len(self.monitored_resources),
            total_metrics_collected=sum(len(data.metrics) for data in current_metrics.values()),
            
            # Health summary
            healthy_resources=health_summary.get('by_status', {}).get('healthy', 0),
            warning_resources=health_summary.get('by_status', {}).get('warning', 0),
            unhealthy_resources=health_summary.get('by_status', {}).get('unhealthy', 0),
            average_health_score=health_summary.get('average_health_score', 0.0),
            
            # Anomalies and alerts
            total_anomalies_detected=anomaly_summary.get('total_anomalies', 0),
            total_alerts_triggered=alert_stats.get('total_alerts', 0),
            critical_alerts=alert_stats.get('by_severity', {}).get('critical', 0),
            
            # Trends and forecasts
            resources_with_increasing_trends=increasing_trends,
            resources_needing_scaling=len(scaling_recommendations),
            capacity_exhaustion_warnings=capacity_warnings,
            
            # Recommendations
            scaling_recommendations=scaling_recommendations,
            optimization_opportunities=optimization_opportunities,
            
            # Detailed data
            resource_health_details=self.health_checker.resource_health,
            performance_trends=performance_trends,
            capacity_forecasts=capacity_forecasts
        )
        
        self.logger.info(f"Generated monitoring report: {report.report_id}")
        return report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'monitored_resources': len(self.monitored_resources),
            'configured_providers': len(self.metrics_collector.monitoring_apis),
            'active_health_checks': len(self.health_checker.health_checks),
            'active_alert_rules': len(self.alert_manager.alert_rules),
            'active_alerts': len(self.alert_manager.active_alerts),
            'anomaly_detection_enabled': self.config.anomaly_detection_enabled,
            'last_monitoring_cycle': datetime.now().isoformat() if self.is_running else None
        }
    
    async def run_manual_check(self, resource_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run manual monitoring check for specified resources"""
        if resource_ids:
            resources = [
                self.monitored_resources[rid] for rid in resource_ids 
                if rid in self.monitored_resources
            ]
        else:
            resources = list(self.monitored_resources.values())
        
        if not resources:
            return {'error': 'No valid resources specified'}
        
        self.logger.info(f"Running manual check for {len(resources)} resources")
        
        # Collect current metrics
        metrics_data = await self.metrics_collector.collect_metrics(
            resources=resources,
            metric_types=[MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION, 
                         MetricType.RESPONSE_TIME, MetricType.ERROR_RATE]
        )
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(metrics_data)
        
        # Check health
        resource_health = await self.health_checker.execute_health_checks(
            resources, metrics_data
        )
        
        return {
            'resources_checked': len(resources),
            'metrics_collected': len(metrics_data),
            'anomalies_detected': len(anomalies),
            'health_status': {
                rid: health.overall_status.value 
                for rid, health in resource_health.items()
            },
            'check_timestamp': datetime.now().isoformat()
        }


# Example usage and integration
async def main():
    """Example usage of the Performance Monitoring System"""
    
    # Initialize system
    config = MonitoringConfig(
        metrics_collection_interval_minutes=5,
        anomaly_detection_enabled=True,
        health_check_interval_minutes=5,
        trend_analysis_period_days=30
    )
    
    monitoring_system = PerformanceMonitoringSystem(config)
    
    # Setup cloud providers
    provider_configs = {
        CloudProvider.AWS: {
            'access_key': 'your_aws_access_key',
            'secret_key': 'your_aws_secret_key',
            'region': 'us-east-1'
        }
    }
    monitoring_system.setup_cloud_providers(provider_configs)
    
    # Add resources to monitor
    resources = [
        CloudResource(
            resource_id="i-1234567890abcdef0",
            resource_type="ec2_instance",
            provider=CloudProvider.AWS,
            region="us-east-1",
            tags={"Environment": "production", "Team": "backend"}
        )
    ]
    
    capacities = {
        "i-1234567890abcdef0": ResourceCapacity(
            cpu_cores=4,
            memory_gb=16.0,
            disk_gb=100.0,
            network_bandwidth_mbps=1000.0,
            instance_type="m5.xlarge"
        )
    }
    
    monitoring_system.add_resources_to_monitor(resources, capacities)
    
    # Setup default monitoring
    monitoring_system.setup_default_monitoring()
    
    # Start monitoring
    await monitoring_system.start_monitoring()
    
    # Let it run for a while
    await asyncio.sleep(300)  # 5 minutes
    
    # Generate report
    report = await monitoring_system.generate_monitoring_report(time_period_hours=1)
    print(f"Generated report: {report.report_id}")
    print(f"Resources monitored: {report.total_resources_monitored}")
    print(f"Anomalies detected: {report.total_anomalies_detected}")
    print(f"Scaling recommendations: {len(report.scaling_recommendations)}")
    
    # Stop monitoring
    await monitoring_system.stop_monitoring()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())