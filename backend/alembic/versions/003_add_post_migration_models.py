"""add post migration models

Revision ID: 003
Revises: 002
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    # Create baseline_metrics table
    op.create_table(
        'baseline_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('capture_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('total_monthly_cost', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('cost_by_service', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_by_team', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_by_project', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_by_environment', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('resource_utilization', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('resource_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('resource_count_by_type', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_baseline_metrics_project_date', 'baseline_metrics', ['migration_project_id', 'capture_date'], unique=False)

    # Create migration_reports table
    op.create_table(
        'migration_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('migration_project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('report_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completion_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('actual_duration_days', sa.Integer(), nullable=False),
        sa.Column('planned_duration_days', sa.Integer(), nullable=False),
        sa.Column('total_cost', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('budgeted_cost', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('resources_migrated', sa.Integer(), nullable=False),
        sa.Column('success_rate', sa.Float(), nullable=False),
        sa.Column('lessons_learned', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('optimization_opportunities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timeline_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('risk_incidents', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('recommendations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['migration_project_id'], ['migration_projects.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('migration_project_id')
    )


def downgrade():
    op.drop_table('migration_reports')
    op.drop_index('ix_baseline_metrics_project_date', table_name='baseline_metrics')
    op.drop_table('baseline_metrics')
