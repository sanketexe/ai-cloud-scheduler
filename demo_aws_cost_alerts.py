#!/usr/bin/env python3
"""
Demo script for AWS Cost Alerts & Monitoring

This script demonstrates the real AWS cost monitoring and alerting functionality
with realistic scenarios and sample data.
"""

import json
from datetime import datetime, timedelta
from backend.core.aws_cost_monitor import (
    AWSCostMonitor, 
    BudgetThreshold, 
    CostAlert,
    CostAnomaly,
    DailyCostSummary
)

def create_demo_monitor():
    """Create a demo monitor with mock data for testing"""
    
    class DemoAWSCostMonitor(AWSCostMonitor):
        """Demo version that doesn't require real AWS credentials"""
        
        def __init__(self):
            # Skip AWS initialization for demo
            self.active_alerts = []
            self.budget_thresholds = []
        
        def _get_current_month_costs(self):
            """Return demo current month costs"""
            return {
                'Amazon Elastic Compute Cloud - Compute': 1245.67,
                'Amazon Elastic Block Store': 387.45,
                'Amazon Relational Database Service': 298.33,
                'Amazon Simple Storage Service': 156.78,
                'Amazon CloudFront': 89.45,
                'Elastic Load Balancing': 67.23,
                'Amazon Route 53': 34.56,
                'AWS Lambda': 23.45
            }
        
        def _get_daily_costs(self, date):
            """Return demo daily costs"""
            base_costs = {
                'Amazon Elastic Compute Cloud - Compute': 42.50,
                'Amazon Elastic Block Store': 12.80,
                'Amazon Relational Database Service': 9.95,
                'Amazon Simple Storage Service': 5.20,
                'Amazon CloudFront': 2.95,
                'Elastic Load Balancing': 2.25,
                'Amazon Route 53': 1.15,
                'AWS Lambda': 0.78
            }
            
            # Add some variation based on date
            import random
            random.seed(date.day)
            
            varied_costs = {}
            for service, cost in base_costs.items():
                # Add ±20% variation
                variation = random.uniform(0.8, 1.2)
                varied_costs[service] = cost * variation
            
            return varied_costs
        
        def detect_cost_anomalies(self, days_back=14):
            """Return demo anomalies"""
            return [
                CostAnomaly(
                    service='Amazon Elastic Compute Cloud - Compute',
                    current_cost=89.50,
                    expected_cost=42.50,
                    deviation_percentage=110.6,
                    confidence_score=0.95,
                    detection_date=datetime.now()
                ),
                CostAnomaly(
                    service='Amazon Simple Storage Service',
                    current_cost=12.40,
                    expected_cost=5.20,
                    confidence_score=0.87,
                    deviation_percentage=138.5,
                    detection_date=datetime.now()
                )
            ]
        
        def detect_cost_spikes(self, spike_threshold=50.0):
            """Return demo cost spikes"""
            alerts = []
            
            # Simulate a cost spike in EC2
            alert = CostAlert(
                alert_id=f"spike_ec2_{datetime.now().strftime('%Y%m%d')}",
                alert_type='spike',
                severity='high',
                title="📈 Cost Spike Alert: Amazon EC2",
                description="Daily cost increased by 78.3% ($42.50 → $75.75)",
                current_cost=75.75,
                threshold_cost=42.50,
                percentage_change=78.3,
                service_affected='Amazon Elastic Compute Cloud - Compute',
                recommended_actions=[
                    "Investigate what changed in EC2 today",
                    "Check for new instance launches",
                    "Review usage patterns and scaling events",
                    "Consider immediate cost controls if unplanned"
                ],
                created_at=datetime.now()
            )
            alerts.append(alert)
            
            return alerts
    
    return DemoAWSCostMonitor()

def demo_budget_monitoring():
    """Demo budget threshold monitoring"""
    print("🎯 BUDGET THRESHOLD MONITORING")
    print("=" * 50)
    
    monitor = create_demo_monitor()
    
    # Add budget thresholds
    thresholds = [
        BudgetThreshold(
            name="Monthly AWS Budget",
            monthly_budget=2000.0,
            warning_threshold=80.0,
            critical_threshold=95.0,
            services=[],
            enabled=True
        ),
        BudgetThreshold(
            name="EC2 Budget",
            monthly_budget=1000.0,
            warning_threshold=75.0,
            critical_threshold=90.0,
            services=['Amazon Elastic Compute Cloud - Compute'],
            enabled=True
        ),
        BudgetThreshold(
            name="Storage Budget",
            monthly_budget=300.0,
            warning_threshold=85.0,
            critical_threshold=95.0,
            services=['Amazon Elastic Block Store', 'Amazon Simple Storage Service'],
            enabled=True
        )
    ]
    
    for threshold in thresholds:
        monitor.add_budget_threshold(threshold)
        print(f"✅ Added budget: {threshold.name} - ${threshold.monthly_budget}")
    
    # Check budget thresholds
    print(f"\n📊 Checking Budget Thresholds...")
    budget_alerts = monitor.check_budget_thresholds()
    
    if budget_alerts:
        print(f"⚠️  Found {len(budget_alerts)} budget alerts:")
        for alert in budget_alerts:
            print(f"   • {alert.title}")
            print(f"     {alert.description}")
            print(f"     Severity: {alert.severity.upper()}")
            print()
    else:
        print("✅ All budgets within thresholds")
    
    return budget_alerts

def demo_anomaly_detection():
    """Demo cost anomaly detection"""
    print("\n🔍 COST ANOMALY DETECTION")
    print("=" * 50)
    
    monitor = create_demo_monitor()
    
    # Detect anomalies
    anomalies = monitor.detect_cost_anomalies()
    
    if anomalies:
        print(f"🚨 Detected {len(anomalies)} cost anomalies:")
        for anomaly in anomalies:
            print(f"   • Service: {anomaly.service}")
            print(f"     Current: ${anomaly.current_cost:.2f} | Expected: ${anomaly.expected_cost:.2f}")
            print(f"     Deviation: {anomaly.deviation_percentage:.1f}%")
            print(f"     Confidence: {anomaly.confidence_score:.1%}")
            print()
        
        # Generate alerts from anomalies
        anomaly_alerts = monitor.generate_anomaly_alerts(anomalies)
        print(f"📢 Generated {len(anomaly_alerts)} anomaly alerts")
        
        return anomaly_alerts
    else:
        print("✅ No cost anomalies detected")
        return []

def demo_spike_detection():
    """Demo cost spike detection"""
    print("\n📈 COST SPIKE DETECTION")
    print("=" * 50)
    
    monitor = create_demo_monitor()
    
    # Detect cost spikes
    spike_alerts = monitor.detect_cost_spikes()
    
    if spike_alerts:
        print(f"⚡ Detected {len(spike_alerts)} cost spikes:")
        for alert in spike_alerts:
            print(f"   • {alert.title}")
            print(f"     {alert.description}")
            print(f"     Service: {alert.service_affected}")
            print(f"     Change: {alert.percentage_change:.1f}%")
            print()
        
        return spike_alerts
    else:
        print("✅ No significant cost spikes detected")
        return []

def demo_daily_summary():
    """Demo daily cost summary"""
    print("\n📋 DAILY COST SUMMARY")
    print("=" * 50)
    
    monitor = create_demo_monitor()
    summary = monitor.generate_daily_summary()
    
    print(f"📅 Date: {summary.date}")
    print(f"💰 Total Cost: ${summary.total_cost:.2f}")
    print(f"📊 Change: {'+' if summary.cost_change >= 0 else ''}${summary.cost_change:.2f} ({summary.cost_change_percentage:+.1f}%)")
    print(f"🚨 Active Alerts: {summary.alerts_count}")
    print(f"💡 Optimization Opportunities: {summary.optimization_opportunities}")
    
    print(f"\n🏆 Top Services Today:")
    for i, service in enumerate(summary.top_services[:5], 1):
        print(f"   {i}. {service['service']}")
        print(f"      ${service['cost']:.2f} ({service['percentage']:.1f}%)")
    
    return summary

def demo_complete_monitoring_cycle():
    """Demo complete monitoring cycle"""
    print("\n🔄 COMPLETE MONITORING CYCLE")
    print("=" * 50)
    
    monitor = create_demo_monitor()
    
    # Add budget thresholds
    monitor.add_budget_threshold(BudgetThreshold(
        name="Demo Budget",
        monthly_budget=2000.0,
        warning_threshold=80.0,
        critical_threshold=95.0,
        services=[],
        enabled=True
    ))
    
    # Run complete monitoring cycle
    results = monitor.run_monitoring_cycle()
    
    print(f"📊 Monitoring Results:")
    print(f"   • New Alerts: {results['new_alerts_count']}")
    print(f"   • Total Active: {results['total_active_alerts']}")
    print(f"   • Budget Alerts: {results['budget_alerts']}")
    print(f"   • Anomaly Alerts: {results['anomaly_alerts']}")
    print(f"   • Spike Alerts: {results['spike_alerts']}")
    
    # Show alert summary
    alert_summary = monitor.get_alert_summary()
    print(f"\n📈 Alert Summary:")
    print(f"   • Total Active: {alert_summary['total_active']}")
    print(f"   • By Severity: {alert_summary['by_severity']}")
    print(f"   • By Type: {alert_summary['by_type']}")
    
    return results

def demo_notification_scenarios():
    """Demo different notification scenarios"""
    print("\n📧 NOTIFICATION SCENARIOS")
    print("=" * 50)
    
    # Critical budget alert
    critical_alert = CostAlert(
        alert_id="demo_critical_budget",
        alert_type='budget_threshold',
        severity='critical',
        title="🚨 Critical Budget Alert: Monthly AWS Budget",
        description="Budget usage at 97.5% ($1,950.00 of $2,000.00)",
        current_cost=1950.00,
        threshold_cost=2000.00,
        percentage_change=97.5,
        service_affected='All Services',
        recommended_actions=[
            "Review and pause non-essential resources immediately",
            "Check for cost optimization opportunities",
            "Consider increasing budget or implementing cost controls"
        ],
        created_at=datetime.now()
    )
    
    # Anomaly alert
    anomaly_alert = CostAlert(
        alert_id="demo_anomaly_ec2",
        alert_type='anomaly',
        severity='high',
        title="🔍 Cost Anomaly Detected: Amazon EC2",
        description="Unusual spending detected - 110.6% above normal ($89.50 vs expected $42.50)",
        current_cost=89.50,
        threshold_cost=42.50,
        percentage_change=110.6,
        service_affected='Amazon Elastic Compute Cloud - Compute',
        recommended_actions=[
            "Investigate recent changes in EC2",
            "Check for new resources or increased usage",
            "Review CloudTrail logs for unusual activity"
        ],
        created_at=datetime.now()
    )
    
    print("📧 Sample Email Notifications:")
    print(f"   1. {critical_alert.title}")
    print(f"      Severity: {critical_alert.severity.upper()}")
    print(f"      Action: Immediate attention required")
    print()
    print(f"   2. {anomaly_alert.title}")
    print(f"      Severity: {anomaly_alert.severity.upper()}")
    print(f"      Action: Investigation recommended")
    
    print(f"\n💬 Sample Slack Notifications:")
    print(f"   🔥 Critical: {critical_alert.description}")
    print(f"   ⚠️  High: {anomaly_alert.description}")

def main():
    """Run the complete AWS Cost Alerts demo"""
    print("🚀 AWS Cost Alerts & Monitoring Demo")
    print("=" * 60)
    print("Demonstrating proactive cost monitoring with real-time alerts")
    print("=" * 60)
    
    # Run all demo scenarios
    budget_alerts = demo_budget_monitoring()
    anomaly_alerts = demo_anomaly_detection()
    spike_alerts = demo_spike_detection()
    daily_summary = demo_daily_summary()
    monitoring_results = demo_complete_monitoring_cycle()
    demo_notification_scenarios()
    
    # Final summary
    total_alerts = len(budget_alerts) + len(anomaly_alerts) + len(spike_alerts)
    
    print(f"\n🎉 DEMO SUMMARY")
    print("=" * 50)
    print(f"✅ Monitoring Features Demonstrated:")
    print(f"   • Budget threshold monitoring")
    print(f"   • Cost anomaly detection")
    print(f"   • Cost spike detection")
    print(f"   • Daily cost summaries")
    print(f"   • Real-time alerting")
    print(f"   • Notification systems")
    
    print(f"\n📊 Demo Results:")
    print(f"   • Total Alerts Generated: {total_alerts}")
    print(f"   • Budget Alerts: {len(budget_alerts)}")
    print(f"   • Anomaly Alerts: {len(anomaly_alerts)}")
    print(f"   • Spike Alerts: {len(spike_alerts)}")
    print(f"   • Daily Cost: ${daily_summary.total_cost:.2f}")
    print(f"   • Cost Change: {daily_summary.cost_change_percentage:+.1f}%")
    
    print(f"\n💡 Key Benefits:")
    print(f"   • Proactive cost monitoring prevents surprises")
    print(f"   • Real-time alerts enable immediate action")
    print(f"   • Anomaly detection catches unusual patterns")
    print(f"   • Budget thresholds provide early warnings")
    print(f"   • Daily summaries keep you informed")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Connect your AWS account for real monitoring")
    print(f"   • Set up budget thresholds for your needs")
    print(f"   • Configure email/Slack notifications")
    print(f"   • Enable automated cost optimization")
    
    print(f"\n✅ AWS Cost Alerts Demo completed successfully!")

if __name__ == "__main__":
    main()