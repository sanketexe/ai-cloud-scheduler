"""Add performance optimization indexes

Revision ID: 012_add_performance_indexes
Revises: 011_multi_cloud_schema
Create Date: 2024-12-31 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Index

# revision identifiers, used by Alembic.
revision = '012_add_performance_indexes'
down_revision = '011_multi_cloud_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Add performance optimization indexes"""
    
    # Workload Specifications indexes
    op.create_index(
        'idx_workload_spec_user_created',
        'workload_specifications',
        ['created_by', 'created_at'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_workload_spec_name_search',
        'workload_specifications',
        ['name'],
        postgresql_using='gin',
        postgresql_ops={'name': 'gin_trgm_ops'}
    )
    
    # Cost Comparisons indexes
    op.create_index(
        'idx_cost_comparison_workload_date',
        'multi_cloud_cost_comparisons',
        ['workload_id', 'comparison_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_cost_comparison_date_range',
        'multi_cloud_cost_comparisons',
        ['comparison_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_cost_comparison_providers',
        'multi_cloud_cost_comparisons',
        ['aws_monthly_cost', 'gcp_monthly_cost', 'azure_monthly_cost'],
        postgresql_using='btree'
    )
    
    # TCO Analysis indexes
    op.create_index(
        'idx_tco_analysis_workload_date',
        'tco_analyses',
        ['workload_id', 'analysis_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_tco_analysis_time_horizon',
        'tco_analyses',
        ['time_horizon_years', 'analysis_date'],
        postgresql_using='btree'
    )
    
    # Migration Analysis indexes
    op.create_index(
        'idx_migration_analysis_workload_date',
        'migration_analyses',
        ['workload_id', 'analysis_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_migration_analysis_providers',
        'migration_analyses',
        ['source_provider', 'target_provider'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_migration_analysis_roi',
        'migration_analyses',
        ['break_even_months', 'migration_cost'],
        postgresql_using='btree'
    )
    
    # Provider Pricing indexes
    op.create_index(
        'idx_provider_pricing_lookup',
        'provider_pricing',
        ['provider', 'service_name', 'region', 'effective_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_provider_pricing_current',
        'provider_pricing',
        ['provider', 'region', 'effective_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_provider_pricing_history',
        'provider_pricing',
        ['provider', 'service_name', 'effective_date'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_provider_pricing_category',
        'provider_pricing',
        ['service_category', 'provider', 'region'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_provider_pricing_cleanup',
        'provider_pricing',
        ['last_updated'],
        postgresql_using='btree'
    )
    
    # Service Equivalency indexes
    op.create_index(
        'idx_service_equivalency_source',
        'service_equivalencies',
        ['source_provider', 'source_service'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_service_equivalency_target',
        'service_equivalencies',
        ['target_provider', 'target_service'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_service_equivalency_category',
        'service_equivalencies',
        ['service_category', 'confidence_score'],
        postgresql_using='btree'
    )
    
    # Feature Parity Analysis indexes
    op.create_index(
        'idx_feature_parity_reference',
        'feature_parity_analyses',
        ['reference_provider', 'reference_service'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_feature_parity_comparison',
        'feature_parity_analyses',
        ['comparison_provider', 'comparison_service'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_feature_parity_date',
        'feature_parity_analyses',
        ['analysis_date'],
        postgresql_using='btree'
    )
    
    # Composite indexes for common query patterns
    op.create_index(
        'idx_workload_user_name_date',
        'workload_specifications',
        ['created_by', 'name', 'created_at'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_comparison_workload_providers_date',
        'multi_cloud_cost_comparisons',
        ['workload_id', 'comparison_date', 'aws_monthly_cost', 'gcp_monthly_cost'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_pricing_provider_service_region_date',
        'provider_pricing',
        ['provider', 'service_name', 'region', 'effective_date'],
        postgresql_using='btree'
    )


def downgrade():
    """Remove performance optimization indexes"""
    
    # Drop composite indexes
    op.drop_index('idx_pricing_provider_service_region_date', table_name='provider_pricing')
    op.drop_index('idx_comparison_workload_providers_date', table_name='multi_cloud_cost_comparisons')
    op.drop_index('idx_workload_user_name_date', table_name='workload_specifications')
    
    # Drop Feature Parity Analysis indexes
    op.drop_index('idx_feature_parity_date', table_name='feature_parity_analyses')
    op.drop_index('idx_feature_parity_comparison', table_name='feature_parity_analyses')
    op.drop_index('idx_feature_parity_reference', table_name='feature_parity_analyses')
    
    # Drop Service Equivalency indexes
    op.drop_index('idx_service_equivalency_category', table_name='service_equivalencies')
    op.drop_index('idx_service_equivalency_target', table_name='service_equivalencies')
    op.drop_index('idx_service_equivalency_source', table_name='service_equivalencies')
    
    # Drop Provider Pricing indexes
    op.drop_index('idx_provider_pricing_cleanup', table_name='provider_pricing')
    op.drop_index('idx_provider_pricing_category', table_name='provider_pricing')
    op.drop_index('idx_provider_pricing_history', table_name='provider_pricing')
    op.drop_index('idx_provider_pricing_current', table_name='provider_pricing')
    op.drop_index('idx_provider_pricing_lookup', table_name='provider_pricing')
    
    # Drop Migration Analysis indexes
    op.drop_index('idx_migration_analysis_roi', table_name='migration_analyses')
    op.drop_index('idx_migration_analysis_providers', table_name='migration_analyses')
    op.drop_index('idx_migration_analysis_workload_date', table_name='migration_analyses')
    
    # Drop TCO Analysis indexes
    op.drop_index('idx_tco_analysis_time_horizon', table_name='tco_analyses')
    op.drop_index('idx_tco_analysis_workload_date', table_name='tco_analyses')
    
    # Drop Cost Comparisons indexes
    op.drop_index('idx_cost_comparison_providers', table_name='multi_cloud_cost_comparisons')
    op.drop_index('idx_cost_comparison_date_range', table_name='multi_cloud_cost_comparisons')
    op.drop_index('idx_cost_comparison_workload_date', table_name='multi_cloud_cost_comparisons')
    
    # Drop Workload Specifications indexes
    op.drop_index('idx_workload_spec_name_search', table_name='workload_specifications')
    op.drop_index('idx_workload_spec_user_created', table_name='workload_specifications')