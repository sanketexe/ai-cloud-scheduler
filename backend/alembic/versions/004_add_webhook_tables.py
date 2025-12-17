"""Add webhook tables

Revision ID: 004
Revises: 003
Create Date: 2025-12-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    # Create webhook_endpoints table
    op.create_table('webhook_endpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('url', sa.String(length=500), nullable=False),
        sa.Column('event_types', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('security_type', sa.String(length=50), nullable=False),
        sa.Column('security_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('headers', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('timeout', sa.Integer(), nullable=False),
        sa.Column('retry_attempts', sa.Integer(), nullable=False),
        sa.Column('retry_delay', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('last_success', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_failure', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failure_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create webhook_deliveries table
    op.create_table('webhook_deliveries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('endpoint_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('response_status', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('response_headers', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('delivery_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('attempt_number', sa.Integer(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_webhook_deliveries_endpoint_id', 'webhook_deliveries', ['endpoint_id'])
    op.create_index('ix_webhook_deliveries_event_id', 'webhook_deliveries', ['event_id'])
    op.create_index('ix_webhook_deliveries_event_type', 'webhook_deliveries', ['event_type'])
    op.create_index('ix_webhook_deliveries_created_at', 'webhook_deliveries', ['created_at'])
    
    # Set default values for existing columns
    op.execute("UPDATE webhook_endpoints SET security_type = 'none' WHERE security_type IS NULL")
    op.execute("UPDATE webhook_endpoints SET security_config = '{}' WHERE security_config IS NULL")
    op.execute("UPDATE webhook_endpoints SET headers = '{}' WHERE headers IS NULL")
    op.execute("UPDATE webhook_endpoints SET event_types = '[]' WHERE event_types IS NULL")
    op.execute("UPDATE webhook_endpoints SET timeout = 30 WHERE timeout IS NULL")
    op.execute("UPDATE webhook_endpoints SET retry_attempts = 3 WHERE retry_attempts IS NULL")
    op.execute("UPDATE webhook_endpoints SET retry_delay = 5 WHERE retry_delay IS NULL")
    op.execute("UPDATE webhook_endpoints SET is_active = true WHERE is_active IS NULL")
    op.execute("UPDATE webhook_endpoints SET status = 'active' WHERE status IS NULL")
    op.execute("UPDATE webhook_endpoints SET failure_count = 0 WHERE failure_count IS NULL")


def downgrade():
    # Drop indexes
    op.drop_index('ix_webhook_deliveries_created_at', table_name='webhook_deliveries')
    op.drop_index('ix_webhook_deliveries_event_type', table_name='webhook_deliveries')
    op.drop_index('ix_webhook_deliveries_event_id', table_name='webhook_deliveries')
    op.drop_index('ix_webhook_deliveries_endpoint_id', table_name='webhook_deliveries')
    
    # Drop tables
    op.drop_table('webhook_deliveries')
    op.drop_table('webhook_endpoints')