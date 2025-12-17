"""
Automated Cost Optimization Savings Calculator

This module provides comprehensive cost tracking and savings calculation for
automated cost optimization actions. It tracks real-time savings, handles
rollback impact calculations, and generates detailed reporting.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from .automation_models import (
    OptimizationAction, ActionStatus, ActionType, AutomationAuditLog
)
# from .aws_pricing_service import AWSPricingService

logger = logging.getLogger(__name__)


class SavingsCategory(Enum):
    """Categories of cost savings"""
    COMPUTE_OPTIMIZATION = "compute_optimization"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    UNUSED_RESOURCES = "unused_resources"


@dataclass
class SavingsMetrics:
    """Comprehensive savings metrics for an action"""
    action_id: str
    estimated_monthly_savings: Decimal
    actual_monthly_savings: Optional[Decimal]
    total_savings_to_date: Decimal
    cost_before_action: Decimal
    cost_after_action: Optional[Decimal]
    savings_percentage: Optional[float]
    payback_period_days: Optional[int]
    roi_percentage: Optional[float]


@dataclass
class SavingsReport:
    """Detailed savings report"""
    period_start: datetime
    period_end: datetime
    total_estimated_savings: Decimal
    total_actual_savings: Decimal
    savings_by_category: Dict[SavingsCategory, Decimal]
    savings_by_action_type: Dict[ActionType, Decimal]
    top_performing_actions: List[SavingsMetrics]
    rollback_impact: Decimal
    net_savings: Decimal
    actions_count: int
    success_rate: float


class SavingsCalculator:
    """
    Comprehensive savings calculation engine for automated cost optimization.
    
    Provides accurate cost modeling, real-time savings tracking, rollback impact
    calculation, and detailed reporting capabilities.
    """
    
    def __init__(self, db: Session, pricing_service: Optional[Any] = None):
        self.db = db
        self.pricing_service = pricing_service
        self.logger = logging.getLogger(__name__ + ".SavingsCalculator")
    
    def calculate_estimated_savings(self, action: OptimizationAction) -> SavingsMetrics:
        """
        Calculate estimated savings for an optimization action.
        
        Args:
            action: The optimization action to calculate savings for
            
        Returns:
            SavingsMetrics with estimated savings information
        """
        self.logger.info(f"Calculating estimated savings for action {action.id}")
        
        try:
            # Get current resource cost
            cost_before = self._get_resource_cost(action.resource_id, action.resource_type)
            
            # Calculate estimated cost after action
            cost_after = self._calculate_post_action_cost(action, cost_before)
            
            # Calculate monthly savings
            monthly_savings = cost_before - cost_after
            
            # Calculate metrics
            savings_percentage = (monthly_savings / cost_before * 100) if cost_before > 0 else 0
            payback_period = self._calculate_payback_period(action, monthly_savings)
            roi_percentage = self._calculate_roi(action, monthly_savings)
            
            return SavingsMetrics(
                action_id=str(action.id),
                estimated_monthly_savings=monthly_savings,
                actual_monthly_savings=None,
                total_savings_to_date=Decimal('0'),
                cost_before_action=cost_before,
                cost_after_action=cost_after,
                savings_percentage=float(savings_percentage),
                payback_period_days=payback_period,
                roi_percentage=roi_percentage
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating estimated savings for action {action.id}: {str(e)}")
            raise
    
    def track_actual_savings(self, action: OptimizationAction) -> SavingsMetrics:
        """
        Track actual savings after action execution.
        
        Args:
            action: The completed optimization action
            
        Returns:
            SavingsMetrics with actual savings information
        """
        self.logger.info(f"Tracking actual savings for action {action.id}")
        
        if action.execution_status != ActionStatus.COMPLETED:
            raise ValueError(f"Action {action.id} is not completed, cannot track actual savings")
        
        try:
            # Get pre-action cost from metadata
            cost_before = Decimal(str(action.resource_metadata.get('cost_before_action', 0)))
            
            # Get current cost
            cost_after = self._get_resource_cost(action.resource_id, action.resource_type)
            
            # Calculate actual monthly savings
            actual_monthly_savings = cost_before - cost_after
            
            # Calculate total savings to date
            days_since_execution = (datetime.utcnow() - action.execution_completed_at).days
            total_savings = actual_monthly_savings * (Decimal(str(days_since_execution)) / Decimal('30.0'))
            
            # Update action with actual savings
            action.actual_savings = actual_monthly_savings
            self.db.commit()
            
            # Calculate updated metrics
            savings_percentage = (actual_monthly_savings / cost_before * 100) if cost_before > 0 else 0
            roi_percentage = self._calculate_roi(action, actual_monthly_savings)
            
            return SavingsMetrics(
                action_id=str(action.id),
                estimated_monthly_savings=action.estimated_monthly_savings,
                actual_monthly_savings=actual_monthly_savings,
                total_savings_to_date=total_savings,
                cost_before_action=cost_before,
                cost_after_action=cost_after,
                savings_percentage=float(savings_percentage),
                payback_period_days=self._calculate_payback_period(action, actual_monthly_savings),
                roi_percentage=roi_percentage
            )
            
        except Exception as e:
            self.logger.error(f"Error tracking actual savings for action {action.id}: {str(e)}")
            raise
    
    def calculate_rollback_impact(self, action: OptimizationAction) -> Decimal:
        """
        Calculate the cost impact of rolling back an action.
        
        Args:
            action: The action to calculate rollback impact for
            
        Returns:
            Decimal representing the cost impact of rollback
        """
        self.logger.info(f"Calculating rollback impact for action {action.id}")
        
        try:
            if action.execution_status != ActionStatus.COMPLETED:
                return Decimal('0')
            
            # Get actual savings achieved
            actual_savings = action.actual_savings or Decimal('0')
            
            # Calculate days the action was in effect
            if action.execution_completed_at:
                days_active = (datetime.utcnow() - action.execution_completed_at).days
                
                # Calculate total savings lost due to rollback
                rollback_impact = actual_savings * (Decimal(str(days_active)) / Decimal('30.0'))
                
                # Add any rollback costs (e.g., re-creating resources)
                rollback_costs = self._calculate_rollback_costs(action)
                
                total_impact = rollback_impact + rollback_costs
                
                self.logger.info(f"Rollback impact for action {action.id}: ${total_impact}")
                return total_impact
            
            return Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error calculating rollback impact for action {action.id}: {str(e)}")
            raise
    
    def generate_savings_report(
        self, 
        start_date: datetime, 
        end_date: datetime,
        include_estimates: bool = True
    ) -> SavingsReport:
        """
        Generate comprehensive savings report for a time period.
        
        Args:
            start_date: Start of reporting period
            end_date: End of reporting period
            include_estimates: Whether to include estimated savings for pending actions
            
        Returns:
            SavingsReport with detailed savings analysis
        """
        self.logger.info(f"Generating savings report from {start_date} to {end_date}")
        
        try:
            # Query actions in the time period
            actions_query = self.db.query(OptimizationAction).filter(
                and_(
                    OptimizationAction.created_at >= start_date,
                    OptimizationAction.created_at <= end_date
                )
            )
            
            if not include_estimates:
                actions_query = actions_query.filter(
                    OptimizationAction.execution_status == ActionStatus.COMPLETED
                )
            
            actions = actions_query.all()
            
            # Calculate totals
            total_estimated = sum(action.estimated_monthly_savings for action in actions)
            total_actual = sum(action.actual_savings or Decimal('0') for action in actions)
            
            # Calculate savings by category
            savings_by_category = self._calculate_savings_by_category(actions)
            
            # Calculate savings by action type
            savings_by_action_type = self._calculate_savings_by_action_type(actions)
            
            # Get top performing actions
            top_actions = self._get_top_performing_actions(actions, limit=10)
            
            # Calculate rollback impact
            rollback_impact = sum(
                self.calculate_rollback_impact(action) 
                for action in actions 
                if action.execution_status == ActionStatus.ROLLED_BACK
            )
            
            # Calculate net savings
            net_savings = total_actual - rollback_impact
            
            # Calculate success rate
            completed_actions = [a for a in actions if a.execution_status == ActionStatus.COMPLETED]
            success_rate = len(completed_actions) / len(actions) * 100 if actions else 0
            
            return SavingsReport(
                period_start=start_date,
                period_end=end_date,
                total_estimated_savings=total_estimated,
                total_actual_savings=total_actual,
                savings_by_category=savings_by_category,
                savings_by_action_type=savings_by_action_type,
                top_performing_actions=top_actions,
                rollback_impact=rollback_impact,
                net_savings=net_savings,
                actions_count=len(actions),
                success_rate=success_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error generating savings report: {str(e)}")
            raise
    
    def get_monthly_savings_trend(self, months_back: int = 12) -> List[Dict[str, Any]]:
        """
        Get monthly savings trend over time.
        
        Args:
            months_back: Number of months to include in trend
            
        Returns:
            List of monthly savings data
        """
        self.logger.info(f"Getting monthly savings trend for {months_back} months")
        
        try:
            trend_data = []
            end_date = datetime.utcnow()
            
            for i in range(months_back):
                # Calculate month boundaries
                month_end = end_date.replace(day=1) - timedelta(days=i*30)
                month_start = month_end.replace(day=1)
                month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                # Get actions for this month
                monthly_actions = self.db.query(OptimizationAction).filter(
                    and_(
                        OptimizationAction.execution_completed_at >= month_start,
                        OptimizationAction.execution_completed_at <= month_end,
                        OptimizationAction.execution_status == ActionStatus.COMPLETED
                    )
                ).all()
                
                # Calculate monthly metrics
                monthly_savings = sum(action.actual_savings or Decimal('0') for action in monthly_actions)
                actions_count = len(monthly_actions)
                
                trend_data.append({
                    'month': month_start.strftime('%Y-%m'),
                    'savings': float(monthly_savings),
                    'actions_count': actions_count,
                    'average_savings_per_action': float(monthly_savings / actions_count) if actions_count > 0 else 0
                })
            
            return list(reversed(trend_data))  # Return chronological order
            
        except Exception as e:
            self.logger.error(f"Error getting monthly savings trend: {str(e)}")
            raise
    
    def get_historical_savings_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive historical savings summary.
        
        Returns:
            Dictionary with historical savings metrics
        """
        self.logger.info("Getting historical savings summary")
        
        try:
            # Get all completed actions
            completed_actions = self.db.query(OptimizationAction).filter(
                OptimizationAction.execution_status == ActionStatus.COMPLETED
            ).all()
            
            if not completed_actions:
                return {
                    'total_actions': 0,
                    'total_savings': 0,
                    'average_monthly_savings': 0,
                    'best_performing_action_type': None,
                    'total_rollback_impact': 0
                }
            
            # Calculate total savings
            total_savings = sum(
                self._calculate_total_savings_for_action(action) 
                for action in completed_actions
            )
            
            # Calculate average monthly savings
            avg_monthly_savings = sum(action.actual_savings or Decimal('0') for action in completed_actions)
            
            # Find best performing action type
            action_type_savings = {}
            for action in completed_actions:
                action_type = action.action_type
                if action_type not in action_type_savings:
                    action_type_savings[action_type] = Decimal('0')
                action_type_savings[action_type] += action.actual_savings or Decimal('0')
            
            best_action_type = max(action_type_savings.items(), key=lambda x: x[1])[0] if action_type_savings else None
            
            # Calculate total rollback impact
            rolled_back_actions = self.db.query(OptimizationAction).filter(
                OptimizationAction.execution_status == ActionStatus.ROLLED_BACK
            ).all()
            
            total_rollback_impact = sum(
                self.calculate_rollback_impact(action) 
                for action in rolled_back_actions
            )
            
            return {
                'total_actions': len(completed_actions),
                'total_savings': float(total_savings),
                'average_monthly_savings': float(avg_monthly_savings),
                'best_performing_action_type': best_action_type.value if best_action_type else None,
                'total_rollback_impact': float(total_rollback_impact)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical savings summary: {str(e)}")
            raise
    
    # Private helper methods
    
    def _get_resource_cost(self, resource_id: str, resource_type: str) -> Decimal:
        """Get current monthly cost for a resource"""
        try:
            if resource_type.lower() == 'ec2':
                return self._get_ec2_cost(resource_id)
            elif resource_type.lower() == 'ebs':
                return self._get_ebs_cost(resource_id)
            elif resource_type.lower() == 'eip':
                return self._get_eip_cost(resource_id)
            elif resource_type.lower() == 'elb':
                return self._get_elb_cost(resource_id)
            else:
                self.logger.warning(f"Unknown resource type: {resource_type}")
                return Decimal('0')
        except Exception as e:
            self.logger.error(f"Error getting cost for resource {resource_id}: {str(e)}")
            return Decimal('0')
    
    def _get_ec2_cost(self, instance_id: str) -> Decimal:
        """Get EC2 instance monthly cost"""
        # This would integrate with AWS Cost Explorer or pricing API
        # For now, using simplified calculation based on instance type
        try:
            if self.pricing_service:
                # Get instance details from pricing service
                instance_cost = self.pricing_service.get_ec2_instance_cost(instance_id)
                return Decimal(str(instance_cost))
        except Exception:
            pass
        # Fallback to estimated cost
        return Decimal('50.00')  # Default estimate
    
    def _get_ebs_cost(self, volume_id: str) -> Decimal:
        """Get EBS volume monthly cost"""
        try:
            if self.pricing_service:
                volume_cost = self.pricing_service.get_ebs_volume_cost(volume_id)
                return Decimal(str(volume_cost))
        except Exception:
            pass
        return Decimal('10.00')  # Default estimate
    
    def _get_eip_cost(self, allocation_id: str) -> Decimal:
        """Get Elastic IP monthly cost"""
        # Elastic IPs cost $3.65/month when not associated
        return Decimal('3.65')
    
    def _get_elb_cost(self, load_balancer_arn: str) -> Decimal:
        """Get Load Balancer monthly cost"""
        try:
            if self.pricing_service:
                elb_cost = self.pricing_service.get_elb_cost(load_balancer_arn)
                return Decimal(str(elb_cost))
        except Exception:
            pass
        return Decimal('25.00')  # Default estimate
    
    def _calculate_post_action_cost(self, action: OptimizationAction, current_cost: Decimal) -> Decimal:
        """Calculate expected cost after action execution"""
        if action.action_type == ActionType.STOP_INSTANCE:
            # Stopped instances only pay for EBS storage
            return current_cost * Decimal('0.1')  # ~10% for storage
        elif action.action_type == ActionType.TERMINATE_INSTANCE:
            return Decimal('0')
        elif action.action_type == ActionType.DELETE_VOLUME:
            return Decimal('0')
        elif action.action_type == ActionType.RELEASE_ELASTIC_IP:
            return Decimal('0')
        elif action.action_type == ActionType.UPGRADE_STORAGE:
            # gp3 is ~20% cheaper than gp2
            return current_cost * Decimal('0.8')
        elif action.action_type == ActionType.RESIZE_INSTANCE:
            # Estimate 30% savings from rightsizing
            return current_cost * Decimal('0.7')
        else:
            return current_cost
    
    def _calculate_payback_period(self, action: OptimizationAction, monthly_savings: Decimal) -> Optional[int]:
        """Calculate payback period in days"""
        # Most optimization actions have no upfront cost, so payback is immediate
        if monthly_savings > 0:
            return 0  # Immediate payback
        return None
    
    def _calculate_roi(self, action: OptimizationAction, monthly_savings: Decimal) -> Optional[float]:
        """Calculate ROI percentage"""
        # Since most actions have no cost, ROI is effectively infinite for positive savings
        if monthly_savings > 0:
            return 100.0  # 100% ROI (no investment required)
        return None
    
    def _calculate_rollback_costs(self, action: OptimizationAction) -> Decimal:
        """Calculate costs associated with rolling back an action"""
        if action.action_type == ActionType.TERMINATE_INSTANCE:
            # Cost to recreate instance
            return Decimal('10.00')  # Estimated setup cost
        elif action.action_type == ActionType.DELETE_VOLUME:
            # Cost to restore from snapshot
            return Decimal('5.00')
        else:
            return Decimal('0')
    
    def _calculate_savings_by_category(self, actions: List[OptimizationAction]) -> Dict[SavingsCategory, Decimal]:
        """Calculate savings grouped by category"""
        category_savings = {category: Decimal('0') for category in SavingsCategory}
        
        for action in actions:
            savings = action.actual_savings or action.estimated_monthly_savings
            category = self._get_action_category(action.action_type)
            category_savings[category] += savings
        
        return category_savings
    
    def _calculate_savings_by_action_type(self, actions: List[OptimizationAction]) -> Dict[ActionType, Decimal]:
        """Calculate savings grouped by action type"""
        type_savings = {}
        
        for action in actions:
            savings = action.actual_savings or action.estimated_monthly_savings
            if action.action_type not in type_savings:
                type_savings[action.action_type] = Decimal('0')
            type_savings[action.action_type] += savings
        
        return type_savings
    
    def _get_action_category(self, action_type: ActionType) -> SavingsCategory:
        """Map action type to savings category"""
        if action_type in [ActionType.STOP_INSTANCE, ActionType.TERMINATE_INSTANCE, ActionType.RESIZE_INSTANCE]:
            return SavingsCategory.COMPUTE_OPTIMIZATION
        elif action_type in [ActionType.DELETE_VOLUME, ActionType.UPGRADE_STORAGE]:
            return SavingsCategory.STORAGE_OPTIMIZATION
        elif action_type in [ActionType.RELEASE_ELASTIC_IP, ActionType.DELETE_LOAD_BALANCER, ActionType.CLEANUP_SECURITY_GROUP]:
            return SavingsCategory.NETWORK_OPTIMIZATION
        else:
            return SavingsCategory.UNUSED_RESOURCES
    
    def _get_top_performing_actions(self, actions: List[OptimizationAction], limit: int = 10) -> List[SavingsMetrics]:
        """Get top performing actions by savings amount"""
        action_metrics = []
        
        for action in actions:
            if action.execution_status == ActionStatus.COMPLETED and action.actual_savings:
                metrics = SavingsMetrics(
                    action_id=str(action.id),
                    estimated_monthly_savings=action.estimated_monthly_savings,
                    actual_monthly_savings=action.actual_savings,
                    total_savings_to_date=self._calculate_total_savings_for_action(action),
                    cost_before_action=Decimal(str(action.resource_metadata.get('cost_before_action', 0))),
                    cost_after_action=Decimal(str(action.resource_metadata.get('cost_after_action', 0))),
                    savings_percentage=float(action.actual_savings / action.estimated_monthly_savings * 100) if action.estimated_monthly_savings > 0 else 0,
                    payback_period_days=0,
                    roi_percentage=100.0
                )
                action_metrics.append(metrics)
        
        # Sort by actual savings and return top performers
        return sorted(action_metrics, key=lambda x: x.actual_monthly_savings or Decimal('0'), reverse=True)[:limit]
    
    def _calculate_total_savings_for_action(self, action: OptimizationAction) -> Decimal:
        """Calculate total savings achieved by an action since execution"""
        if not action.execution_completed_at or not action.actual_savings:
            return Decimal('0')
        
        days_active = (datetime.utcnow() - action.execution_completed_at).days
        return action.actual_savings * (Decimal(str(days_active)) / Decimal('30.0'))