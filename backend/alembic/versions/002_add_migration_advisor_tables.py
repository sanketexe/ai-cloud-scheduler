"""Add migration advisor tables

Revision ID: 002
Revises: 001
Create Date: 2024-11-16 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enums for migration advisor
    op.execute("""
        CREATE TYPE migrationstatus AS ENUM (
            'assessment', 'analysis', 'recommendation', 'planning', 
            'execution', 'complete', 'cancelled'
        )
    """)
    
    op.execute("""
        CREATE TYPE companysize AS ENUM ('small', 'medium', 'large', 'enterprise')
    """)
    
    op.execute("""
        CREATE TYPE infrastructuretype AS ENUM (
            'on_premises', 'cloud', 'hybrid', 'multi_cloud'
        )
    """)
    
    op.execute("""
        CREATE TYPE experiencelevel AS ENUM ('none', 'beginner', 'intermediate', 'advanced')
    """)
    
    op.execute("""
        CREATE TYPE phasestatus AS ENUM (
            'not_started', 'in_progress', 'completed', 'failed', 'rolled_back'
        )
    """)
    
    op.execute("""
        CREATE TYPE ownershipstatus AS ENUM ('assigned', 'unassigned', 'pending')
    """)
    
    op.execute("""
        CREATE TYPE migrationrisklevel AS ENUM ('low', 'medium', 'high', 'critical')
    """)
    
    # Create migration_projects table
    op.create_table('migration_projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.String(length=100), nullable=False),
        sa.Column('organization_name', sa.String(length=255), nullable=False),
        sa.Column('status', sa.Enum(
            'assessment', 'analysis', 'recommendation', 'planning', 
            'execution', 'complete', 'cancelled',
            name='migrationstatus'
        ), nullable=False),
        sa.Column('current_phase', sa.String(length=100), nullable=True),
        sa.Column('estimated_completion', sa.DateTime(timezone=True), nullable=True),
        sa.Column('actual_completion', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_migration_projects_project_id'), 'migration_projects', ['project_id'], unique=True)
    op.create_index('ix_migration_projects_status', 'migration_projects', ['status'], unique=False)
    op.create_index('ix_migration_projects_created_at', 'migration_projects', ['created_at'], unique=False)
    
    # Create organization_profiles table
    op.create_table('organization_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('company_size', sa.Enum('small', 'medium', 'large', 'enterprise', name='companysize'), nullable=False),
        sa.Column('industry', sa.String(length=100), nullable=False),
        sa.Column('current_infrastructure', sa.Enum(
            'on_premises', 'cloud', 'hybrid', 'multi_cloud',
            name='infrastructuretype'
        ), nullable=False),
        sa.Column('geographic_presence', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('it_team_size', sa.Integer(), nullable=False),
        sa.Column('cloud_experience_level', sa.Enum(
            'none', 'beginner', 'intermediate', 'advanced',
            name='experiencelevel'
        ), nullable=False),
        sa.Column('additional_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create workload_profiles table
    op.create_table('workload_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workload_name', sa.String(length=255), nullable=False),
        sa.Column('application_type', sa.String(length=100), nullable=False),
        sa.Column('total_compute_cores', sa.Integer(), nullable=True),
        sa.Column('total_memory_gb', sa.Integer(), nullable=True),
        sa.Column('total_storage_tb', sa.Float(), nullable=True),
        sa.Column('database_types', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('data_volume_tb', sa.Float(), nullable=True),
        sa.Column('peak_transaction_rate', sa.Integer(), nullable=True),
        sa.Column('workload_patterns', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('dependencies', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workload_profiles_migration_project_id'), 'workload_profiles', ['migration_project_id'], unique=False)
    
    # Create performance_requirements table
    op.create_table('performance_requirements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('latency_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('availability_target', sa.Float(), nullable=False),
        sa.Column('disaster_recovery_rto', sa.Integer(), nullable=True),
        sa.Column('disaster_recovery_rpo', sa.Integer(), nullable=True),
        sa.Column('geographic_distribution', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('peak_load_multiplier', sa.Float(), nullable=True),
        sa.Column('additional_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create compliance_requirements table
    op.create_table('compliance_requirements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('regulatory_frameworks', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('data_residency_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('industry_certifications', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('security_standards', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('audit_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('additional_compliance', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create budget_constraints table
    op.create_table('budget_constraints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('current_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('migration_budget', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('target_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('cost_optimization_priority', sa.String(length=20), nullable=True),
        sa.Column('acceptable_cost_variance', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=True),
        sa.Column('additional_constraints', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create technical_requirements table
    op.create_table('technical_requirements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('required_services', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ml_ai_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('analytics_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('container_orchestration', sa.Boolean(), nullable=True),
        sa.Column('serverless_requirements', sa.Boolean(), nullable=True),
        sa.Column('specialized_compute', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('integration_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('additional_technical', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create provider_evaluations table
    op.create_table('provider_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider_name', sa.String(length=50), nullable=False),
        sa.Column('service_availability_score', sa.Float(), nullable=False),
        sa.Column('pricing_score', sa.Float(), nullable=False),
        sa.Column('compliance_score', sa.Float(), nullable=False),
        sa.Column('technical_fit_score', sa.Float(), nullable=False),
        sa.Column('migration_complexity_score', sa.Float(), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('strengths', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('weaknesses', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('detailed_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_provider_evaluations_migration_project_id'), 'provider_evaluations', ['migration_project_id'], unique=False)
    op.create_index('ix_provider_evaluations_project_score', 'provider_evaluations', ['migration_project_id', 'overall_score'], unique=False)
    
    # Create recommendation_reports table
    op.create_table('recommendation_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('primary_recommendation', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('key_differentiators', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_comparison', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('risk_assessment', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('justification', sa.Text(), nullable=False),
        sa.Column('scoring_weights', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('alternative_recommendations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    
    # Create migration_plans table
    op.create_table('migration_plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('plan_id', sa.String(length=100), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_provider', sa.String(length=50), nullable=False),
        sa.Column('total_duration_days', sa.Integer(), nullable=False),
        sa.Column('estimated_cost', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('risk_level', sa.Enum('low', 'medium', 'high', 'critical', name='migrationrisklevel'), nullable=False),
        sa.Column('dependencies_graph', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('migration_waves', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success_criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('rollback_strategy', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )
    op.create_index(op.f('ix_migration_plans_plan_id'), 'migration_plans', ['plan_id'], unique=True)
    
    # Create migration_phases table
    op.create_table('migration_phases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('phase_id', sa.String(length=100), nullable=False),
        sa.Column('migration_plan_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('phase_name', sa.String(length=255), nullable=False),
        sa.Column('phase_order', sa.Integer(), nullable=False),
        sa.Column('workloads', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('actual_start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('actual_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Enum(
            'not_started', 'in_progress', 'completed', 'failed', 'rolled_back',
            name='phasestatus'
        ), nullable=False),
        sa.Column('prerequisites', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success_criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('rollback_plan', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_plan_id'], ['migration_plans.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_migration_phases_phase_id'), 'migration_phases', ['phase_id'], unique=False)
    op.create_index(op.f('ix_migration_phases_migration_plan_id'), 'migration_phases', ['migration_plan_id'], unique=False)
    op.create_index('ix_migration_phases_plan_order', 'migration_phases', ['migration_plan_id', 'phase_order'], unique=False)
    op.create_index('ix_migration_phases_status', 'migration_phases', ['status'], unique=False)
    
    # Create organizational_structures table
    op.create_table('organizational_structures',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('structure_name', sa.String(length=255), nullable=False),
        sa.Column('teams', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('projects', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('environments', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('regions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_centers', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('custom_dimensions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_organizational_structures_migration_project_id'), 'organizational_structures', ['migration_project_id'], unique=False)
    
    # Create categorized_resources table
    op.create_table('categorized_resources',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=False),
        sa.Column('resource_name', sa.String(length=255), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('team', sa.String(length=100), nullable=True),
        sa.Column('project', sa.String(length=100), nullable=True),
        sa.Column('environment', sa.String(length=50), nullable=True),
        sa.Column('region', sa.String(length=50), nullable=True),
        sa.Column('cost_center', sa.String(length=100), nullable=True),
        sa.Column('custom_attributes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ownership_status', sa.Enum('assigned', 'unassigned', 'pending', name='ownershipstatus'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_categorized_resources_migration_project_id'), 'categorized_resources', ['migration_project_id'], unique=False)
    op.create_index(op.f('ix_categorized_resources_resource_id'), 'categorized_resources', ['resource_id'], unique=False)
    op.create_index('ix_categorized_resources_project_resource', 'categorized_resources', ['migration_project_id', 'resource_id'], unique=False)
    op.create_index('ix_categorized_resources_team', 'categorized_resources', ['team'], unique=False)
    op.create_index('ix_categorized_resources_project', 'categorized_resources', ['project'], unique=False)
    op.create_index('ix_categorized_resources_ownership', 'categorized_resources', ['ownership_status'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('ix_categorized_resources_ownership', table_name='categorized_resources')
    op.drop_index('ix_categorized_resources_project', table_name='categorized_resources')
    op.drop_index('ix_categorized_resources_team', table_name='categorized_resources')
    op.drop_index('ix_categorized_resources_project_resource', table_name='categorized_resources')
    op.drop_index(op.f('ix_categorized_resources_resource_id'), table_name='categorized_resources')
    op.drop_index(op.f('ix_categorized_resources_migration_project_id'), table_name='categorized_resources')
    op.drop_table('categorized_resources')
    
    op.drop_index(op.f('ix_organizational_structures_migration_project_id'), table_name='organizational_structures')
    op.drop_table('organizational_structures')
    
    op.drop_index('ix_migration_phases_status', table_name='migration_phases')
    op.drop_index('ix_migration_phases_plan_order', table_name='migration_phases')
    op.drop_index(op.f('ix_migration_phases_migration_plan_id'), table_name='migration_phases')
    op.drop_index(op.f('ix_migration_phases_phase_id'), table_name='migration_phases')
    op.drop_table('migration_phases')
    
    op.drop_index(op.f('ix_migration_plans_plan_id'), table_name='migration_plans')
    op.drop_table('migration_plans')
    
    op.drop_table('recommendation_reports')
    
    op.drop_index('ix_provider_evaluations_project_score', table_name='provider_evaluations')
    op.drop_index(op.f('ix_provider_evaluations_migration_project_id'), table_name='provider_evaluations')
    op.drop_table('provider_evaluations')
    
    op.drop_table('technical_requirements')
    op.drop_table('budget_constraints')
    op.drop_table('compliance_requirements')
    op.drop_table('performance_requirements')
    
    op.drop_index(op.f('ix_workload_profiles_migration_project_id'), table_name='workload_profiles')
    op.drop_table('workload_profiles')
    
    op.drop_table('organization_profiles')
    
    op.drop_index('ix_migration_projects_created_at', table_name='migration_projects')
    op.drop_index('ix_migration_projects_status', table_name='migration_projects')
    op.drop_index(op.f('ix_migration_projects_project_id'), table_name='migration_projects')
    op.drop_table('migration_projects')
    
    # Drop enums
    op.execute('DROP TYPE migrationrisklevel')
    op.execute('DROP TYPE ownershipstatus')
    op.execute('DROP TYPE phasestatus')
    op.execute('DROP TYPE experiencelevel')
    op.execute('DROP TYPE infrastructuretype')
    op.execute('DROP TYPE companysize')
    op.execute('DROP TYPE migrationstatus')
