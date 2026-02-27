"""Add multi-cloud cost comparison tables

Revision ID: 005_add_multi_cloud_tables
Revises: 004
Create Date: 2024-12-29 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005_add_multi_cloud_tables'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    """Create multi-cloud cost comparison tables"""
    
    # Create workload_specifications table
    op.create_table('workload_specifications',
        sa.Column('id', sa.String(36), nullable=False),  # UUID as string for SQLite
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('compute_spec', sa.Text(), nullable=False),  # JSON as text for SQLite
        sa.Column('storage_spec', sa.Text(), nullable=False),
        sa.Column('network_spec', sa.Text(), nullable=False),
        sa.Column('database_spec', sa.Text(), nullable=True),
        sa.Column('additional_services', sa.Text(), nullable=True),
        sa.Column('usage_patterns', sa.Text(), nullable=False),
        sa.Column('compliance_requirements', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for workload_specifications
    op.create_index('ix_workload_specs_created_by', 'workload_specifications', ['created_by'])
    op.create_index('ix_workload_specs_name', 'workload_specifications', ['name'])
    
    # Create multi_cloud_cost_comparisons table
    op.create_table('multi_cloud_cost_comparisons',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('workload_id', sa.String(36), nullable=False),
        sa.Column('comparison_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('aws_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('gcp_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('azure_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('aws_annual_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('gcp_annual_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('azure_annual_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('cost_breakdown', sa.Text(), nullable=False),
        sa.Column('recommendations', sa.Text(), nullable=True),
        sa.Column('pricing_data_version', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['workload_id'], ['workload_specifications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for multi_cloud_cost_comparisons
    op.create_index('ix_cost_comparisons_workload_date', 'multi_cloud_cost_comparisons', ['workload_id', 'comparison_date'])
    
    # Create tco_analyses table
    op.create_table('tco_analyses',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('workload_id', sa.String(36), nullable=False),
        sa.Column('analysis_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('time_horizon_years', sa.Integer(), nullable=False),
        sa.Column('aws_tco', sa.Text(), nullable=False),
        sa.Column('gcp_tco', sa.Text(), nullable=False),
        sa.Column('azure_tco', sa.Text(), nullable=False),
        sa.Column('hidden_costs', sa.Text(), nullable=False),
        sa.Column('operational_costs', sa.Text(), nullable=False),
        sa.Column('cost_projections', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['workload_id'], ['workload_specifications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for tco_analyses
    op.create_index('ix_tco_analyses_workload_date', 'tco_analyses', ['workload_id', 'analysis_date'])
    
    # Create migration_analyses table
    op.create_table('migration_analyses',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('workload_id', sa.String(36), nullable=False),
        sa.Column('source_provider', sa.String(length=20), nullable=False),
        sa.Column('target_provider', sa.String(length=20), nullable=False),
        sa.Column('analysis_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('migration_cost', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('migration_timeline_days', sa.Integer(), nullable=False),
        sa.Column('break_even_months', sa.Integer(), nullable=True),
        sa.Column('risk_assessment', sa.Text(), nullable=False),
        sa.Column('cost_breakdown', sa.Text(), nullable=False),
        sa.Column('recommendations', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['workload_id'], ['workload_specifications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for migration_analyses
    op.create_index('ix_migration_analyses_workload_providers', 'migration_analyses', ['workload_id', 'source_provider', 'target_provider'])
    op.create_index('ix_migration_analyses_date', 'migration_analyses', ['analysis_date'])
    
    # Create provider_pricing table
    op.create_table('provider_pricing',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('provider', sa.String(length=20), nullable=False),
        sa.Column('service_name', sa.String(length=100), nullable=False),
        sa.Column('service_category', sa.String(length=50), nullable=False),
        sa.Column('region', sa.String(length=50), nullable=False),
        sa.Column('pricing_unit', sa.String(length=20), nullable=False),
        sa.Column('price_per_unit', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False),
        sa.Column('effective_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False),
        sa.Column('pricing_details', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for provider_pricing
    op.create_index('ix_provider_pricing_provider_service', 'provider_pricing', ['provider', 'service_name'])
    op.create_index('ix_provider_pricing_region_date', 'provider_pricing', ['region', 'effective_date'])
    op.create_index('ix_provider_pricing_category', 'provider_pricing', ['service_category'])
    
    # Create service_equivalencies table
    op.create_table('service_equivalencies',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('source_provider', sa.String(length=20), nullable=False),
        sa.Column('source_service', sa.String(length=100), nullable=False),
        sa.Column('target_provider', sa.String(length=20), nullable=False),
        sa.Column('target_service', sa.String(length=100), nullable=False),
        sa.Column('service_category', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('feature_parity_score', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('performance_ratio', sa.Numeric(precision=4, scale=2), nullable=False),
        sa.Column('cost_efficiency_score', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('migration_complexity', sa.String(length=20), nullable=False),
        sa.Column('limitations', sa.Text(), nullable=True),
        sa.Column('additional_features', sa.Text(), nullable=True),
        sa.Column('mapping_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_provider', 'source_service', 'target_provider', 'target_service', name='uq_service_equivalency')
    )
    
    # Create indexes for service_equivalencies
    op.create_index('ix_service_equiv_source', 'service_equivalencies', ['source_provider', 'source_service'])
    op.create_index('ix_service_equiv_target', 'service_equivalencies', ['target_provider', 'target_service'])
    op.create_index('ix_service_equiv_category', 'service_equivalencies', ['service_category'])
    
    # Create feature_parity_analyses table
    op.create_table('feature_parity_analyses',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('analysis_id', sa.String(length=255), nullable=False),
        sa.Column('reference_service', sa.String(length=100), nullable=False),
        sa.Column('reference_provider', sa.String(length=20), nullable=False),
        sa.Column('comparison_services', sa.Text(), nullable=False),
        sa.Column('feature_matrix', sa.Text(), nullable=False),
        sa.Column('missing_features', sa.Text(), nullable=False),
        sa.Column('additional_features', sa.Text(), nullable=False),
        sa.Column('overall_parity_score', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('analysis_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('workload_context', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('analysis_id', name='uq_feature_parity_analysis_id')
    )
    
    # Create indexes for feature_parity_analyses
    op.create_index('ix_feature_parity_reference', 'feature_parity_analyses', ['reference_provider', 'reference_service'])
    op.create_index('ix_feature_parity_date', 'feature_parity_analyses', ['analysis_date'])


def downgrade():
    """Drop multi-cloud cost comparison tables"""
    
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('feature_parity_analyses')
    op.drop_table('service_equivalencies')
    op.drop_table('provider_pricing')
    op.drop_table('migration_analyses')
    op.drop_table('tco_analyses')
    op.drop_table('multi_cloud_cost_comparisons')
    op.drop_table('workload_specifications')