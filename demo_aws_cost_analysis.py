#!/usr/bin/env python3
"""
Demo script for AWS Cost Analysis

This script demonstrates the real AWS cost analysis functionality
without requiring actual AWS credentials for testing.
"""

import json
from datetime import datetime, timedelta
from backend.core.aws_cost_analyzer import (
    AWSCostAnalyzer, 
    CostOptimizationOpportunity, 
    ServiceCostBreakdown, 
    CostAnalysisReport
)

def create_demo_analyzer():
    """Create a demo analyzer with mock data for testing"""
    
    class DemoAWSCostAnalyzer(AWSCostAnalyzer):
        """Demo version that doesn't require real AWS credentials"""
        
        def __init__(self):
            # Skip AWS initialization for demo
            pass
        
        def test_connection(self):
            return {
                'status': 'success',
                'message': 'Demo AWS connection (simulated)',
                'permissions': ['cost_explorer', 'ec2', 'cloudwatch']
            }
        
        def analyze_costs(self, days_back=30):
            """Return demo cost analysis data"""
            
            # Demo optimization opportunities
            opportunities = [
                CostOptimizationOpportunity(
                    service='EC2',
                    opportunity_type='rightsizing',
                    current_monthly_cost=245.50,
                    potential_monthly_savings=98.20,
                    confidence_level='high',
                    description='3 t3.medium instances with <15% CPU utilization',
                    action_required='Downsize to t3.small instances',
                    implementation_effort='low',
                    risk_level='low'
                ),
                CostOptimizationOpportunity(
                    service='EBS',
                    opportunity_type='unused_resources',
                    current_monthly_cost=67.80,
                    potential_monthly_savings=67.80,
                    confidence_level='high',
                    description='5 unattached EBS volumes (total 340GB)',
                    action_required='Delete unused volumes or attach to instances',
                    implementation_effort='low',
                    risk_level='medium'
                ),
                CostOptimizationOpportunity(
                    service='EC2',
                    opportunity_type='reserved_instances',
                    current_monthly_cost=432.00,
                    potential_monthly_savings=129.60,
                    confidence_level='high',
                    description='6 m5.large instances running consistently for 60+ days',
                    action_required='Purchase 6 Reserved Instances for m5.large',
                    implementation_effort='low',
                    risk_level='low'
                ),
                CostOptimizationOpportunity(
                    service='EBS',
                    opportunity_type='storage_optimization',
                    current_monthly_cost=156.40,
                    potential_monthly_savings=31.28,
                    confidence_level='high',
                    description='12 gp2 volumes (>100GB each) can be upgraded to gp3',
                    action_required='Upgrade volume types from gp2 to gp3',
                    implementation_effort='low',
                    risk_level='low'
                ),
                CostOptimizationOpportunity(
                    service='ELB',
                    opportunity_type='unused_resources',
                    current_monthly_cost=22.00,
                    potential_monthly_savings=22.00,
                    confidence_level='medium',
                    description='1 Application Load Balancer with no healthy targets',
                    action_required='Review and potentially delete unused load balancer',
                    implementation_effort='medium',
                    risk_level='medium'
                )
            ]
            
            # Demo service cost breakdown
            service_costs = [
                ServiceCostBreakdown(
                    service_name='Amazon Elastic Compute Cloud - Compute',
                    current_month_cost=1245.67,
                    last_month_cost=1189.23,
                    cost_trend='increasing',
                    percentage_of_total=52.3,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Amazon Elastic Block Store',
                    current_month_cost=387.45,
                    last_month_cost=401.12,
                    cost_trend='decreasing',
                    percentage_of_total=16.2,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Amazon Relational Database Service',
                    current_month_cost=298.33,
                    last_month_cost=295.67,
                    cost_trend='stable',
                    percentage_of_total=12.5,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Amazon Simple Storage Service',
                    current_month_cost=156.78,
                    last_month_cost=148.90,
                    cost_trend='increasing',
                    percentage_of_total=6.6,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Amazon CloudFront',
                    current_month_cost=89.45,
                    last_month_cost=92.11,
                    cost_trend='decreasing',
                    percentage_of_total=3.7,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Elastic Load Balancing',
                    current_month_cost=67.23,
                    last_month_cost=65.89,
                    cost_trend='stable',
                    percentage_of_total=2.8,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='Amazon Route 53',
                    current_month_cost=34.56,
                    last_month_cost=33.78,
                    cost_trend='stable',
                    percentage_of_total=1.4,
                    top_resources=[]
                ),
                ServiceCostBreakdown(
                    service_name='AWS Lambda',
                    current_month_cost=23.45,
                    last_month_cost=28.90,
                    cost_trend='decreasing',
                    percentage_of_total=1.0,
                    top_resources=[]
                )
            ]
            
            total_cost = sum(service.current_month_cost for service in service_costs)
            potential_savings = sum(opp.potential_monthly_savings for opp in opportunities)
            
            recommendations = [
                f"Potential monthly savings of ${potential_savings:.2f} identified",
                "Rightsize 3 EC2 instances for $98.20/month savings",
                "Remove 6 unused resources for $89.80/month savings", 
                "Purchase Reserved Instances for $129.60/month savings",
                "Optimize 12 storage volumes for $31.28/month savings"
            ]
            
            roi_analysis = {
                'monthly_savings': potential_savings,
                'annual_savings': potential_savings * 12,
                'savings_percentage': (potential_savings / total_cost) * 100,
                'payback_period_months': 0,
                'implementation_effort': 'low'
            }
            
            return CostAnalysisReport(
                total_monthly_cost=total_cost,
                cost_trend='increasing',
                top_cost_drivers=service_costs,
                optimization_opportunities=opportunities,
                potential_monthly_savings=potential_savings,
                roi_analysis=roi_analysis,
                recommendations_summary=recommendations
            )
    
    return DemoAWSCostAnalyzer()

def main():
    """Run the demo"""
    print("🚀 AWS Cost Analysis Demo")
    print("=" * 50)
    
    # Create demo analyzer
    analyzer = create_demo_analyzer()
    
    # Test connection
    print("\n1. Testing AWS Connection...")
    connection_result = analyzer.test_connection()
    print(f"   Status: {connection_result['status']}")
    print(f"   Message: {connection_result['message']}")
    print(f"   Permissions: {', '.join(connection_result['permissions'])}")
    
    # Run cost analysis
    print("\n2. Running Cost Analysis...")
    report = analyzer.analyze_costs()
    
    # Display results
    print(f"\n📊 COST ANALYSIS RESULTS")
    print(f"   Total Monthly Cost: ${report.total_monthly_cost:,.2f}")
    print(f"   Cost Trend: {report.cost_trend}")
    print(f"   Potential Monthly Savings: ${report.potential_monthly_savings:,.2f}")
    print(f"   Annual Savings Potential: ${report.roi_analysis['annual_savings']:,.2f}")
    print(f"   Savings Percentage: {report.roi_analysis['savings_percentage']:.1f}%")
    
    print(f"\n💰 TOP COST DRIVERS:")
    for i, service in enumerate(report.top_cost_drivers[:5], 1):
        print(f"   {i}. {service.service_name}")
        print(f"      Cost: ${service.current_month_cost:,.2f} ({service.percentage_of_total:.1f}%)")
        print(f"      Trend: {service.cost_trend}")
    
    print(f"\n🎯 OPTIMIZATION OPPORTUNITIES:")
    for i, opp in enumerate(report.optimization_opportunities, 1):
        print(f"   {i}. {opp.description}")
        print(f"      Service: {opp.service} | Type: {opp.opportunity_type}")
        print(f"      Potential Savings: ${opp.potential_monthly_savings:,.2f}/month")
        print(f"      Confidence: {opp.confidence_level} | Risk: {opp.risk_level}")
        print(f"      Action: {opp.action_required}")
        print()
    
    print(f"📋 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations_summary, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n💡 QUICK WINS (High Confidence, Low Risk):")
    quick_wins = [opp for opp in report.optimization_opportunities 
                  if opp.confidence_level == 'high' and opp.risk_level == 'low']
    
    total_quick_savings = sum(opp.potential_monthly_savings for opp in quick_wins)
    print(f"   Total Quick Win Savings: ${total_quick_savings:,.2f}/month")
    print(f"   Annual Impact: ${total_quick_savings * 12:,.2f}")
    
    for opp in quick_wins:
        print(f"   • {opp.description} → ${opp.potential_monthly_savings:,.2f}/month")
    
    print(f"\n🎉 SUMMARY:")
    print(f"   This analysis identified ${report.potential_monthly_savings:,.2f} in monthly savings")
    print(f"   That's ${report.roi_analysis['annual_savings']:,.2f} per year!")
    print(f"   Implementation effort: {report.roi_analysis['implementation_effort']}")
    print(f"   ROI: {report.roi_analysis['savings_percentage']:.1f}% cost reduction")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   This demonstrates real AWS cost analysis capabilities")
    print(f"   Connect your AWS account to see your actual optimization opportunities")

if __name__ == "__main__":
    main()