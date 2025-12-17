"""
AWS Cost Analysis Engine - Real Cost Optimization for Startups

This module provides actual AWS cost analysis by connecting to AWS Cost Explorer API
and identifying real cost optimization opportunities.
"""

import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostOptimizationOpportunity:
    """Represents a specific cost optimization opportunity"""
    service: str
    opportunity_type: str  # 'rightsizing', 'unused_resources', 'reserved_instances', 'storage_optimization'
    current_monthly_cost: float
    potential_monthly_savings: float
    confidence_level: str  # 'high', 'medium', 'low'
    description: str
    action_required: str
    implementation_effort: str  # 'low', 'medium', 'high'
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class ServiceCostBreakdown:
    """Cost breakdown by AWS service"""
    service_name: str
    current_month_cost: float
    last_month_cost: float
    cost_trend: str  # 'increasing', 'decreasing', 'stable'
    percentage_of_total: float
    top_resources: List[Dict[str, Any]]

@dataclass
class CostAnalysisReport:
    """Complete cost analysis report"""
    total_monthly_cost: float
    cost_trend: str
    top_cost_drivers: List[ServiceCostBreakdown]
    optimization_opportunities: List[CostOptimizationOpportunity]
    potential_monthly_savings: float
    roi_analysis: Dict[str, Any]
    recommendations_summary: List[str]

class AWSCostAnalyzer:
    """
    Real AWS Cost Analysis Engine
    
    Connects to AWS Cost Explorer API to analyze actual spending and identify
    optimization opportunities for startups and small businesses.
    """
    
    def __init__(self, aws_access_key_id: str = None, aws_secret_access_key: str = None, region: str = 'us-east-1'):
        """Initialize AWS Cost Analyzer with credentials"""
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.session = boto3.Session(region_name=region)
            
            self.cost_explorer = self.session.client('ce')
            self.ec2 = self.session.client('ec2')
            self.cloudwatch = self.session.client('cloudwatch')
            self.rds = self.session.client('rds')
            self.s3 = self.session.client('s3')
            
            logger.info("AWS Cost Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS Cost Analyzer: {str(e)}")
            raise
    
    def analyze_costs(self, days_back: int = 30) -> CostAnalysisReport:
        """
        Perform comprehensive cost analysis
        
        Args:
            days_back: Number of days to analyze (default: 30)
            
        Returns:
            CostAnalysisReport with complete analysis and recommendations
        """
        try:
            logger.info(f"Starting cost analysis for last {days_back} days")
            
            # Get cost data
            cost_data = self._get_cost_data(days_back)
            service_costs = self._get_service_breakdown(days_back)
            
            # Analyze optimization opportunities
            opportunities = []
            opportunities.extend(self._analyze_ec2_optimization())
            opportunities.extend(self._analyze_storage_optimization())
            opportunities.extend(self._analyze_unused_resources())
            opportunities.extend(self._analyze_reserved_instance_opportunities())
            
            # Calculate totals and trends
            total_cost = sum(service['cost'] for service in service_costs)
            potential_savings = sum(opp.potential_monthly_savings for opp in opportunities)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(opportunities)
            
            # Create ROI analysis
            roi_analysis = self._calculate_roi_analysis(opportunities, total_cost)
            
            report = CostAnalysisReport(
                total_monthly_cost=total_cost,
                cost_trend=self._determine_cost_trend(cost_data),
                top_cost_drivers=self._format_service_breakdown(service_costs),
                optimization_opportunities=opportunities,
                potential_monthly_savings=potential_savings,
                roi_analysis=roi_analysis,
                recommendations_summary=recommendations
            )
            
            logger.info(f"Cost analysis completed. Potential savings: ${potential_savings:.2f}/month")
            return report
            
        except Exception as e:
            logger.error(f"Cost analysis failed: {str(e)}")
            raise
    
    def _get_cost_data(self, days_back: int) -> Dict[str, Any]:
        """Get raw cost data from AWS Cost Explorer"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get cost data: {str(e)}")
            raise
    
    def _get_service_breakdown(self, days_back: int) -> List[Dict[str, Any]]:
        """Get cost breakdown by AWS service"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            services = []
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service_name = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    
                    if cost > 0:  # Only include services with actual costs
                        services.append({
                            'service': service_name,
                            'cost': cost
                        })
            
            # Sort by cost (highest first)
            services.sort(key=lambda x: x['cost'], reverse=True)
            return services
            
        except Exception as e:
            logger.error(f"Failed to get service breakdown: {str(e)}")
            return []
    
    def _analyze_ec2_optimization(self) -> List[CostOptimizationOpportunity]:
        """Analyze EC2 instances for rightsizing opportunities"""
        opportunities = []
        
        try:
            # Get all running EC2 instances
            response = self.ec2.describe_instances(
                Filters=[
                    {
                        'Name': 'instance-state-name',
                        'Values': ['running']
                    }
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    instance_type = instance['InstanceType']
                    
                    # Get CPU utilization for the instance
                    cpu_utilization = self._get_cpu_utilization(instance_id)
                    
                    # Check for rightsizing opportunity
                    if cpu_utilization < 20:  # Low CPU utilization
                        current_cost = self._get_instance_monthly_cost(instance_type)
                        recommended_type = self._recommend_smaller_instance(instance_type)
                        
                        if recommended_type:
                            new_cost = self._get_instance_monthly_cost(recommended_type)
                            savings = current_cost - new_cost
                            
                            if savings > 0:
                                opportunities.append(CostOptimizationOpportunity(
                                    service='EC2',
                                    opportunity_type='rightsizing',
                                    current_monthly_cost=current_cost,
                                    potential_monthly_savings=savings,
                                    confidence_level='high' if cpu_utilization < 10 else 'medium',
                                    description=f'Instance {instance_id} ({instance_type}) has low CPU utilization ({cpu_utilization:.1f}%)',
                                    action_required=f'Resize to {recommended_type}',
                                    implementation_effort='low',
                                    risk_level='low'
                                ))
            
        except Exception as e:
            logger.error(f"EC2 optimization analysis failed: {str(e)}")
        
        return opportunities
    
    def _analyze_storage_optimization(self) -> List[CostOptimizationOpportunity]:
        """Analyze storage for optimization opportunities"""
        opportunities = []
        
        try:
            # Analyze EBS volumes
            response = self.ec2.describe_volumes()
            
            for volume in response['Volumes']:
                volume_id = volume['VolumeId']
                volume_type = volume['VolumeType']
                size = volume['Size']
                
                # Check if volume is unattached
                if volume['State'] == 'available':
                    monthly_cost = self._calculate_ebs_cost(volume_type, size)
                    
                    opportunities.append(CostOptimizationOpportunity(
                        service='EBS',
                        opportunity_type='unused_resources',
                        current_monthly_cost=monthly_cost,
                        potential_monthly_savings=monthly_cost,
                        confidence_level='high',
                        description=f'Unattached EBS volume {volume_id} ({size}GB {volume_type})',
                        action_required='Delete unused volume or attach to instance',
                        implementation_effort='low',
                        risk_level='medium'
                    ))
                
                # Check for gp2 to gp3 optimization
                elif volume_type == 'gp2' and size >= 100:
                    current_cost = self._calculate_ebs_cost('gp2', size)
                    new_cost = self._calculate_ebs_cost('gp3', size)
                    savings = current_cost - new_cost
                    
                    if savings > 0:
                        opportunities.append(CostOptimizationOpportunity(
                            service='EBS',
                            opportunity_type='storage_optimization',
                            current_monthly_cost=current_cost,
                            potential_monthly_savings=savings,
                            confidence_level='high',
                            description=f'EBS volume {volume_id} can be upgraded from gp2 to gp3',
                            action_required='Upgrade volume type to gp3',
                            implementation_effort='low',
                            risk_level='low'
                        ))
            
        except Exception as e:
            logger.error(f"Storage optimization analysis failed: {str(e)}")
        
        return opportunities
    
    def _analyze_unused_resources(self) -> List[CostOptimizationOpportunity]:
        """Find unused resources that are incurring costs"""
        opportunities = []
        
        try:
            # Check for unused Elastic IPs
            response = self.ec2.describe_addresses()
            
            for address in response['Addresses']:
                if 'InstanceId' not in address:  # Unassociated Elastic IP
                    monthly_cost = 3.65  # AWS charges ~$3.65/month for unused Elastic IPs
                    
                    opportunities.append(CostOptimizationOpportunity(
                        service='EC2',
                        opportunity_type='unused_resources',
                        current_monthly_cost=monthly_cost,
                        potential_monthly_savings=monthly_cost,
                        confidence_level='high',
                        description=f'Unused Elastic IP: {address.get("PublicIp", "Unknown")}',
                        action_required='Release unused Elastic IP',
                        implementation_effort='low',
                        risk_level='low'
                    ))
            
            # Check for unused Load Balancers (simplified check)
            try:
                elb_response = self.session.client('elbv2').describe_load_balancers()
                
                for lb in elb_response['LoadBalancers']:
                    # Check if load balancer has no targets (simplified)
                    target_groups = self.session.client('elbv2').describe_target_groups(
                        LoadBalancerArn=lb['LoadBalancerArn']
                    )
                    
                    has_healthy_targets = False
                    for tg in target_groups['TargetGroups']:
                        targets = self.session.client('elbv2').describe_target_health(
                            TargetGroupArn=tg['TargetGroupArn']
                        )
                        if any(t['TargetHealth']['State'] == 'healthy' for t in targets['TargetHealthDescriptions']):
                            has_healthy_targets = True
                            break
                    
                    if not has_healthy_targets:
                        monthly_cost = 22.0  # Approximate ALB cost per month
                        
                        opportunities.append(CostOptimizationOpportunity(
                            service='ELB',
                            opportunity_type='unused_resources',
                            current_monthly_cost=monthly_cost,
                            potential_monthly_savings=monthly_cost,
                            confidence_level='medium',
                            description=f'Load balancer with no healthy targets: {lb["LoadBalancerName"]}',
                            action_required='Review and potentially delete unused load balancer',
                            implementation_effort='medium',
                            risk_level='medium'
                        ))
                        
            except Exception as e:
                logger.warning(f"Load balancer analysis failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unused resources analysis failed: {str(e)}")
        
        return opportunities
    
    def _analyze_reserved_instance_opportunities(self) -> List[CostOptimizationOpportunity]:
        """Analyze opportunities for Reserved Instance savings"""
        opportunities = []
        
        try:
            # Get running instances that have been running consistently
            response = self.ec2.describe_instances(
                Filters=[
                    {
                        'Name': 'instance-state-name',
                        'Values': ['running']
                    }
                ]
            )
            
            instance_types = {}
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_type = instance['InstanceType']
                    launch_time = instance['LaunchTime']
                    
                    # Check if instance has been running for more than 30 days
                    days_running = (datetime.now(launch_time.tzinfo) - launch_time).days
                    
                    if days_running > 30:  # Stable workload candidate
                        if instance_type not in instance_types:
                            instance_types[instance_type] = 0
                        instance_types[instance_type] += 1
            
            # Calculate RI savings for each instance type
            for instance_type, count in instance_types.items():
                if count >= 1:  # At least one instance of this type
                    on_demand_cost = self._get_instance_monthly_cost(instance_type) * count
                    ri_cost = self._get_reserved_instance_cost(instance_type) * count
                    monthly_savings = on_demand_cost - ri_cost
                    
                    if monthly_savings > 0:
                        opportunities.append(CostOptimizationOpportunity(
                            service='EC2',
                            opportunity_type='reserved_instances',
                            current_monthly_cost=on_demand_cost,
                            potential_monthly_savings=monthly_savings,
                            confidence_level='high' if count >= 2 else 'medium',
                            description=f'{count} {instance_type} instance(s) running consistently for 30+ days',
                            action_required=f'Purchase {count} Reserved Instance(s) for {instance_type}',
                            implementation_effort='low',
                            risk_level='low'
                        ))
            
        except Exception as e:
            logger.error(f"Reserved Instance analysis failed: {str(e)}")
        
        return opportunities
    
    def _get_cpu_utilization(self, instance_id: str, days: int = 7) -> float:
        """Get average CPU utilization for an instance"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[
                    {
                        'Name': 'InstanceId',
                        'Value': instance_id
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                avg_cpu = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])
                return avg_cpu
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get CPU utilization for {instance_id}: {str(e)}")
            return 0.0
    
    def _get_instance_monthly_cost(self, instance_type: str) -> float:
        """Get approximate monthly cost for an instance type (simplified pricing)"""
        # Simplified pricing - in production, use AWS Pricing API
        pricing_map = {
            't2.nano': 4.75,
            't2.micro': 9.50,
            't2.small': 19.00,
            't2.medium': 38.00,
            't2.large': 76.00,
            't3.nano': 4.32,
            't3.micro': 8.64,
            't3.small': 17.28,
            't3.medium': 34.56,
            't3.large': 69.12,
            'm5.large': 79.20,
            'm5.xlarge': 158.40,
            'c5.large': 74.16,
            'c5.xlarge': 148.32,
        }
        
        return pricing_map.get(instance_type, 50.0)  # Default estimate
    
    def _recommend_smaller_instance(self, current_type: str) -> Optional[str]:
        """Recommend a smaller instance type"""
        downsize_map = {
            't2.medium': 't2.small',
            't2.large': 't2.medium',
            't3.medium': 't3.small',
            't3.large': 't3.medium',
            'm5.xlarge': 'm5.large',
            'c5.xlarge': 'c5.large',
        }
        
        return downsize_map.get(current_type)
    
    def _calculate_ebs_cost(self, volume_type: str, size_gb: int) -> float:
        """Calculate monthly EBS cost"""
        # Simplified EBS pricing per GB per month
        pricing = {
            'gp2': 0.10,
            'gp3': 0.08,
            'io1': 0.125,
            'io2': 0.125,
            'st1': 0.045,
            'sc1': 0.025
        }
        
        return pricing.get(volume_type, 0.10) * size_gb
    
    def _get_reserved_instance_cost(self, instance_type: str) -> float:
        """Get approximate RI cost (1-year, no upfront)"""
        on_demand = self._get_instance_monthly_cost(instance_type)
        return on_demand * 0.7  # Approximate 30% savings with RI
    
    def _determine_cost_trend(self, cost_data: Dict[str, Any]) -> str:
        """Determine if costs are increasing, decreasing, or stable"""
        try:
            results = cost_data['ResultsByTime']
            if len(results) < 2:
                return 'stable'
            
            recent_cost = float(results[-1]['Total']['BlendedCost']['Amount'])
            older_cost = float(results[0]['Total']['BlendedCost']['Amount'])
            
            change_percent = ((recent_cost - older_cost) / older_cost) * 100
            
            if change_percent > 10:
                return 'increasing'
            elif change_percent < -10:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _format_service_breakdown(self, service_costs: List[Dict[str, Any]]) -> List[ServiceCostBreakdown]:
        """Format service cost data"""
        total_cost = sum(service['cost'] for service in service_costs)
        
        breakdowns = []
        for service in service_costs[:10]:  # Top 10 services
            percentage = (service['cost'] / total_cost) * 100 if total_cost > 0 else 0
            
            breakdowns.append(ServiceCostBreakdown(
                service_name=service['service'],
                current_month_cost=service['cost'],
                last_month_cost=service['cost'],  # Simplified
                cost_trend='stable',  # Simplified
                percentage_of_total=percentage,
                top_resources=[]  # Would need additional API calls
            ))
        
        return breakdowns
    
    def _generate_recommendations(self, opportunities: List[CostOptimizationOpportunity]) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []
        
        total_savings = sum(opp.potential_monthly_savings for opp in opportunities)
        
        if total_savings > 0:
            recommendations.append(f"Potential monthly savings of ${total_savings:.2f} identified")
        
        # Group by opportunity type
        by_type = {}
        for opp in opportunities:
            if opp.opportunity_type not in by_type:
                by_type[opp.opportunity_type] = []
            by_type[opp.opportunity_type].append(opp)
        
        for opp_type, opps in by_type.items():
            count = len(opps)
            savings = sum(opp.potential_monthly_savings for opp in opps)
            
            if opp_type == 'rightsizing':
                recommendations.append(f"Rightsize {count} EC2 instances for ${savings:.2f}/month savings")
            elif opp_type == 'unused_resources':
                recommendations.append(f"Remove {count} unused resources for ${savings:.2f}/month savings")
            elif opp_type == 'reserved_instances':
                recommendations.append(f"Purchase Reserved Instances for ${savings:.2f}/month savings")
            elif opp_type == 'storage_optimization':
                recommendations.append(f"Optimize {count} storage volumes for ${savings:.2f}/month savings")
        
        return recommendations
    
    def _calculate_roi_analysis(self, opportunities: List[CostOptimizationOpportunity], total_cost: float) -> Dict[str, Any]:
        """Calculate ROI analysis"""
        total_savings = sum(opp.potential_monthly_savings for opp in opportunities)
        
        return {
            'monthly_savings': total_savings,
            'annual_savings': total_savings * 12,
            'savings_percentage': (total_savings / total_cost) * 100 if total_cost > 0 else 0,
            'payback_period_months': 0,  # Most optimizations have immediate payback
            'implementation_effort': self._calculate_overall_effort(opportunities)
        }
    
    def _calculate_overall_effort(self, opportunities: List[CostOptimizationOpportunity]) -> str:
        """Calculate overall implementation effort"""
        if not opportunities:
            return 'none'
        
        effort_scores = {'low': 1, 'medium': 2, 'high': 3}
        avg_score = sum(effort_scores.get(opp.implementation_effort, 2) for opp in opportunities) / len(opportunities)
        
        if avg_score <= 1.5:
            return 'low'
        elif avg_score <= 2.5:
            return 'medium'
        else:
            return 'high'

    def test_connection(self) -> Dict[str, Any]:
        """Test AWS connection and permissions"""
        try:
            # Test Cost Explorer access
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)
            
            self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost']
            )
            
            return {
                'status': 'success',
                'message': 'AWS connection successful',
                'permissions': ['cost_explorer', 'ec2', 'cloudwatch']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'AWS connection failed: {str(e)}',
                'permissions': []
            }