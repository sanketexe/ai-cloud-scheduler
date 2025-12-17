"""
Unit tests for SavingsCalculator

Tests the cost tracking and savings calculation functionality for
automated cost optimization actions.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from backend.core.savings_calculator import (
    SavingsCalculator, SavingsMetrics, SavingsReport, SavingsCategory
)
from backend.core.automation_models import (
    OptimizationAction, ActionType, ActionStatus, RiskLevel, ApprovalStatus
)
# from backend.core.aws_pricing_service import AWSPricingService


class TestSavingsCalculator:
    """Test SavingsCalculator functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        return Mock()
    
    @pytest.fixture
    def mock_pricing_service(self):
        """Mock AWS pricing service"""
        service = Mock()
        service.get_ec2_instance_cost.return_value = 50.0
        service.get_ebs_volume_cost.return_value = 10.0
        service.get_elb_cost.return_value = 25.0
        return service
    
    @pytest.fixture
    def calculator(self, mock_db, mock_pricing_service):
        """SavingsCalculator instance with mocked dependencies"""
        return SavingsCalculator(mock_db, mock_pricing_service)
    
    @pytest.fixture
    def sample_action(self):
        """Sample optimization action for testing"""
        action = Mock(spec=OptimizationAction)
        action.id = "test-action-123"
        action.action_type = ActionType.STOP_INSTANCE
        action.resource_id = "i-1234567890abcdef0"
        action.resource_type = "ec2"
        action.estimated_monthly_savings = Decimal('40.00')
        action.actual_savings = None
        action.execution_status = ActionStatus.PENDING
        action.resource_metadata = {'cost_before_action': 50.0}
        action.execution_completed_at = None
        return action
    
    def test_calculate_estimated_savings_ec2_stop(self, calculator, sample_action):
        """Test estimated savings calculation for EC2 instance stop"""
        # Test stopping an EC2 instance
        sample_action.action_type = ActionType.STOP_INSTANCE
        
        with patch.object(calculator, '_get_resource_cost', return_value=Decimal('50.00')):
            metrics = calculator.calculate_estimated_savings(sample_action)
        
        assert metrics.action_id == "test-action-123"
        assert metrics.cost_before_action == Decimal('50.00')
        assert metrics.cost_after_action == Decimal('5.00')  # 10% for storage
        assert metrics.estimated_monthly_savings == Decimal('45.00')
        assert metrics.savings_percentage == 90.0
        assert metrics.payback_period_days == 0  # Immediate payback
        assert metrics.roi_percentage == 100.0
    
    def test_calculate_estimated_savings_volume_delete(self, calculator, sample_action):
        """Test estimated savings calculation for volume deletion"""
        sample_action.action_type = ActionType.DELETE_VOLUME
        sample_action.resource_type = "ebs"
        
        with patch.object(calculator, '_get_resource_cost', return_value=Decimal('10.00')):
            metrics = calculator.calculate_estimated_savings(sample_action)
        
        assert metrics.cost_before_action == Decimal('10.00')
        assert metrics.cost_after_action == Decimal('0.00')
        assert metrics.estimated_monthly_savings == Decimal('10.00')
        assert metrics.savings_percentage == 100.0
    
    def test_calculate_estimated_savings_storage_upgrade(self, calculator, sample_action):
        """Test estimated savings calculation for storage upgrade"""
        sample_action.action_type = ActionType.UPGRADE_STORAGE
        sample_action.resource_type = "ebs"
        
        with patch.object(calculator, '_get_resource_cost', return_value=Decimal('20.00')):
            metrics = calculator.calculate_estimated_savings(sample_action)
        
        assert metrics.cost_before_action == Decimal('20.00')
        assert metrics.cost_after_action == Decimal('16.00')  # 20% savings
        assert metrics.estimated_monthly_savings == Decimal('4.00')
        assert metrics.savings_percentage == 20.0
    
    def test_track_actual_savings_completed_action(self, calculator, sample_action):
        """Test tracking actual savings for completed action"""
        # Set up completed action
        sample_action.execution_status = ActionStatus.COMPLETED
        sample_action.execution_completed_at = datetime.utcnow() - timedelta(days=30)
        sample_action.resource_metadata = {'cost_before_action': 50.0}
        
        with patch.object(calculator, '_get_resource_cost', return_value=Decimal('5.00')):
            metrics = calculator.track_actual_savings(sample_action)
        
        assert metrics.actual_monthly_savings == Decimal('45.00')
        assert metrics.total_savings_to_date == Decimal('45.00')  # 30 days = 1 month
        assert sample_action.actual_savings == Decimal('45.00')
    
    def test_track_actual_savings_not_completed(self, calculator, sample_action):
        """Test tracking actual savings fails for non-completed action"""
        sample_action.execution_status = ActionStatus.PENDING
        
        with pytest.raises(ValueError, match="Action .* is not completed"):
            calculator.track_actual_savings(sample_action)
    
    def test_calculate_rollback_impact_completed_action(self, calculator, sample_action):
        """Test rollback impact calculation for completed action"""
        sample_action.execution_status = ActionStatus.COMPLETED
        sample_action.actual_savings = Decimal('30.00')
        sample_action.execution_completed_at = datetime.utcnow() - timedelta(days=15)
        sample_action.action_type = ActionType.STOP_INSTANCE
        
        impact = calculator.calculate_rollback_impact(sample_action)
        
        # 15 days = 0.5 months, so impact = 30 * 0.5 = 15
        assert impact == Decimal('15.00')
    
    def test_calculate_rollback_impact_with_costs(self, calculator, sample_action):
        """Test rollback impact calculation including rollback costs"""
        sample_action.execution_status = ActionStatus.COMPLETED
        sample_action.actual_savings = Decimal('20.00')
        sample_action.execution_completed_at = datetime.utcnow() - timedelta(days=30)
        sample_action.action_type = ActionType.TERMINATE_INSTANCE  # Has rollback costs
        
        impact = calculator.calculate_rollback_impact(sample_action)
        
        # 30 days = 1 month, so savings lost = 20 * 1 = 20
        # Plus rollback cost of 10 for terminate instance = 30 total
        assert impact == Decimal('30.00')
    
    def test_calculate_rollback_impact_not_completed(self, calculator, sample_action):
        """Test rollback impact is zero for non-completed actions"""
        sample_action.execution_status = ActionStatus.PENDING
        
        impact = calculator.calculate_rollback_impact(sample_action)
        
        assert impact == Decimal('0')
    
    def test_generate_savings_report(self, calculator, mock_db):
        """Test comprehensive savings report generation"""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        # Mock actions
        action1 = Mock(spec=OptimizationAction)
        action1.estimated_monthly_savings = Decimal('50.00')
        action1.actual_savings = Decimal('45.00')
        action1.execution_status = ActionStatus.COMPLETED
        action1.action_type = ActionType.STOP_INSTANCE
        
        action2 = Mock(spec=OptimizationAction)
        action2.estimated_monthly_savings = Decimal('20.00')
        action2.actual_savings = Decimal('18.00')
        action2.execution_status = ActionStatus.COMPLETED
        action2.action_type = ActionType.DELETE_VOLUME
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [action1, action2]
        mock_db.query.return_value = mock_query
        
        with patch.object(calculator, '_calculate_savings_by_category') as mock_category, \
             patch.object(calculator, '_calculate_savings_by_action_type') as mock_action_type, \
             patch.object(calculator, '_get_top_performing_actions') as mock_top_actions:
            
            mock_category.return_value = {SavingsCategory.COMPUTE_OPTIMIZATION: Decimal('45.00')}
            mock_action_type.return_value = {ActionType.STOP_INSTANCE: Decimal('45.00')}
            mock_top_actions.return_value = []
            
            report = calculator.generate_savings_report(start_date, end_date)
        
        assert report.period_start == start_date
        assert report.period_end == end_date
        assert report.total_estimated_savings == Decimal('70.00')
        assert report.total_actual_savings == Decimal('63.00')
        assert report.actions_count == 2
        assert report.success_rate == 100.0
    
    def test_get_monthly_savings_trend(self, calculator, mock_db):
        """Test monthly savings trend calculation"""
        # Mock actions for different months
        action1 = Mock(spec=OptimizationAction)
        action1.actual_savings = Decimal('100.00')
        action1.execution_status = ActionStatus.COMPLETED
        
        action2 = Mock(spec=OptimizationAction)
        action2.actual_savings = Decimal('50.00')
        action2.execution_status = ActionStatus.COMPLETED
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [action1, action2]
        mock_db.query.return_value = mock_query
        
        trend = calculator.get_monthly_savings_trend(months_back=2)
        
        assert len(trend) == 2
        assert all('month' in month_data for month_data in trend)
        assert all('savings' in month_data for month_data in trend)
        assert all('actions_count' in month_data for month_data in trend)
    
    def test_get_historical_savings_summary(self, calculator, mock_db):
        """Test historical savings summary"""
        # Mock completed actions
        action1 = Mock(spec=OptimizationAction)
        action1.actual_savings = Decimal('100.00')
        action1.action_type = ActionType.STOP_INSTANCE
        action1.execution_completed_at = datetime.utcnow() - timedelta(days=30)
        
        action2 = Mock(spec=OptimizationAction)
        action2.actual_savings = Decimal('50.00')
        action2.action_type = ActionType.DELETE_VOLUME
        action2.execution_completed_at = datetime.utcnow() - timedelta(days=15)
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [action1, action2]
        mock_db.query.return_value = mock_query
        
        with patch.object(calculator, '_calculate_total_savings_for_action') as mock_total:
            mock_total.side_effect = [Decimal('100.00'), Decimal('25.00')]
            
            summary = calculator.get_historical_savings_summary()
        
        assert summary['total_actions'] == 2
        assert summary['total_savings'] == 125.0
        assert summary['average_monthly_savings'] == 150.0
        assert summary['best_performing_action_type'] == ActionType.STOP_INSTANCE.value
    
    def test_get_historical_savings_summary_no_actions(self, calculator, mock_db):
        """Test historical savings summary with no actions"""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query
        
        summary = calculator.get_historical_savings_summary()
        
        assert summary['total_actions'] == 0
        assert summary['total_savings'] == 0
        assert summary['average_monthly_savings'] == 0
        assert summary['best_performing_action_type'] is None
    
    def test_get_resource_cost_ec2(self, calculator, mock_pricing_service):
        """Test getting EC2 resource cost"""
        cost = calculator._get_resource_cost("i-1234567890abcdef0", "ec2")
        
        assert cost == Decimal('50.0')
        mock_pricing_service.get_ec2_instance_cost.assert_called_once_with("i-1234567890abcdef0")
    
    def test_get_resource_cost_ebs(self, calculator, mock_pricing_service):
        """Test getting EBS resource cost"""
        cost = calculator._get_resource_cost("vol-1234567890abcdef0", "ebs")
        
        assert cost == Decimal('10.0')
        mock_pricing_service.get_ebs_volume_cost.assert_called_once_with("vol-1234567890abcdef0")
    
    def test_get_resource_cost_eip(self, calculator):
        """Test getting Elastic IP cost"""
        cost = calculator._get_resource_cost("eipalloc-12345678", "eip")
        
        assert cost == Decimal('3.65')
    
    def test_get_resource_cost_unknown_type(self, calculator):
        """Test getting cost for unknown resource type"""
        cost = calculator._get_resource_cost("unknown-resource", "unknown")
        
        assert cost == Decimal('0')
    
    def test_calculate_post_action_cost_scenarios(self, calculator, sample_action):
        """Test post-action cost calculation for different action types"""
        current_cost = Decimal('100.00')
        
        # Test stop instance
        sample_action.action_type = ActionType.STOP_INSTANCE
        cost = calculator._calculate_post_action_cost(sample_action, current_cost)
        assert cost == Decimal('10.00')  # 10% for storage
        
        # Test terminate instance
        sample_action.action_type = ActionType.TERMINATE_INSTANCE
        cost = calculator._calculate_post_action_cost(sample_action, current_cost)
        assert cost == Decimal('0')
        
        # Test delete volume
        sample_action.action_type = ActionType.DELETE_VOLUME
        cost = calculator._calculate_post_action_cost(sample_action, current_cost)
        assert cost == Decimal('0')
        
        # Test upgrade storage
        sample_action.action_type = ActionType.UPGRADE_STORAGE
        cost = calculator._calculate_post_action_cost(sample_action, current_cost)
        assert cost == Decimal('80.00')  # 20% savings
        
        # Test resize instance
        sample_action.action_type = ActionType.RESIZE_INSTANCE
        cost = calculator._calculate_post_action_cost(sample_action, current_cost)
        assert cost == Decimal('70.00')  # 30% savings
    
    def test_calculate_rollback_costs(self, calculator, sample_action):
        """Test rollback cost calculation for different action types"""
        # Test terminate instance rollback
        sample_action.action_type = ActionType.TERMINATE_INSTANCE
        cost = calculator._calculate_rollback_costs(sample_action)
        assert cost == Decimal('10.00')
        
        # Test delete volume rollback
        sample_action.action_type = ActionType.DELETE_VOLUME
        cost = calculator._calculate_rollback_costs(sample_action)
        assert cost == Decimal('5.00')
        
        # Test stop instance rollback (no cost)
        sample_action.action_type = ActionType.STOP_INSTANCE
        cost = calculator._calculate_rollback_costs(sample_action)
        assert cost == Decimal('0')
    
    def test_get_action_category_mapping(self, calculator):
        """Test action type to category mapping"""
        # Compute optimization
        assert calculator._get_action_category(ActionType.STOP_INSTANCE) == SavingsCategory.COMPUTE_OPTIMIZATION
        assert calculator._get_action_category(ActionType.TERMINATE_INSTANCE) == SavingsCategory.COMPUTE_OPTIMIZATION
        assert calculator._get_action_category(ActionType.RESIZE_INSTANCE) == SavingsCategory.COMPUTE_OPTIMIZATION
        
        # Storage optimization
        assert calculator._get_action_category(ActionType.DELETE_VOLUME) == SavingsCategory.STORAGE_OPTIMIZATION
        assert calculator._get_action_category(ActionType.UPGRADE_STORAGE) == SavingsCategory.STORAGE_OPTIMIZATION
        
        # Network optimization
        assert calculator._get_action_category(ActionType.RELEASE_ELASTIC_IP) == SavingsCategory.NETWORK_OPTIMIZATION
        assert calculator._get_action_category(ActionType.DELETE_LOAD_BALANCER) == SavingsCategory.NETWORK_OPTIMIZATION
        assert calculator._get_action_category(ActionType.CLEANUP_SECURITY_GROUP) == SavingsCategory.NETWORK_OPTIMIZATION
    
    def test_calculate_total_savings_for_action(self, calculator, sample_action):
        """Test total savings calculation for an action"""
        sample_action.execution_completed_at = datetime.utcnow() - timedelta(days=60)
        sample_action.actual_savings = Decimal('30.00')
        
        total_savings = calculator._calculate_total_savings_for_action(sample_action)
        
        # 60 days = 2 months, so total = 30 * 2 = 60
        assert total_savings == Decimal('60.00')
    
    def test_calculate_total_savings_for_action_no_completion(self, calculator, sample_action):
        """Test total savings calculation for action without completion date"""
        sample_action.execution_completed_at = None
        sample_action.actual_savings = Decimal('30.00')
        
        total_savings = calculator._calculate_total_savings_for_action(sample_action)
        
        assert total_savings == Decimal('0')
    
    def test_calculate_total_savings_for_action_no_actual_savings(self, calculator, sample_action):
        """Test total savings calculation for action without actual savings"""
        sample_action.execution_completed_at = datetime.utcnow() - timedelta(days=30)
        sample_action.actual_savings = None
        
        total_savings = calculator._calculate_total_savings_for_action(sample_action)
        
        assert total_savings == Decimal('0')


if __name__ == "__main__":
    pytest.main([__file__])