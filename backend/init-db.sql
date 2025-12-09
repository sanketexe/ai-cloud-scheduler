-- FinOps Platform Database Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database if it doesn't exist (this won't work in init script, but kept for reference)
-- CREATE DATABASE finops_db;

-- Set timezone
SET timezone = 'UTC';

-- Create custom types for better performance
DO $$
BEGIN
    -- User roles enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('admin', 'finance_manager', 'analyst', 'viewer');
    END IF;
    
    -- Provider types enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'provider_type') THEN
        CREATE TYPE provider_type AS ENUM ('aws', 'gcp', 'azure', 'other');
    END IF;
    
    -- Budget types enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'budget_type') THEN
        CREATE TYPE budget_type AS ENUM ('monthly', 'quarterly', 'annual', 'project');
    END IF;
    
    -- Alert types enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_type') THEN
        CREATE TYPE alert_type AS ENUM ('threshold', 'forecast', 'anomaly');
    END IF;
    
    -- Recommendation types enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'recommendation_type') THEN
        CREATE TYPE recommendation_type AS ENUM ('rightsizing', 'reserved_instance', 'unused_resource', 'underutilized');
    END IF;
    
    -- Recommendation status enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'recommendation_status') THEN
        CREATE TYPE recommendation_status AS ENUM ('pending', 'approved', 'rejected', 'implemented');
    END IF;
    
    -- Risk level enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'risk_level') THEN
        CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high');
    END IF;
END
$$;

-- Create a function to update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create indexes for better performance (these will be created by Alembic migrations)
-- But we can create some additional performance indexes here

-- Function to create partition tables for cost_data (for better performance with large datasets)
CREATE OR REPLACE FUNCTION create_cost_data_partition(start_date DATE, end_date DATE)
RETURNS VOID AS $$
DECLARE
    table_name TEXT;
BEGIN
    table_name := 'cost_data_' || to_char(start_date, 'YYYY_MM');
    
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF cost_data
        FOR VALUES FROM (%L) TO (%L)
    ', table_name, start_date, end_date);
    
    -- Create indexes on partition
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (provider_id, cost_date)', 
                   table_name || '_provider_date_idx', table_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (resource_id, cost_date)', 
                   table_name || '_resource_date_idx', table_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING GIN (tags)', 
                   table_name || '_tags_idx', table_name);
END;
$$ LANGUAGE plpgsql;

-- Create initial admin user (password: Admin123!)
-- This will be handled by the application, but kept for reference
/*
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at, updated_at)
VALUES (
    uuid_generate_v4(),
    'admin@finops.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e', -- Admin123!
    'System',
    'Administrator',
    'admin',
    true,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT (email) DO NOTHING;
*/

-- Create system configurations
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default system settings
INSERT INTO system_settings (key, value, description) VALUES
    ('cost_sync_frequency_hours', '24', 'How often to sync cost data from cloud providers'),
    ('budget_alert_frequency_hours', '6', 'How often to check budgets and send alerts'),
    ('optimization_analysis_frequency_hours', '168', 'How often to run optimization analysis (weekly)'),
    ('data_retention_days', '2555', 'How long to keep historical data (7 years)'),
    ('max_api_requests_per_minute', '1000', 'Rate limiting for API requests'),
    ('enable_email_notifications', 'true', 'Whether to send email notifications'),
    ('default_currency', '"USD"', 'Default currency for cost calculations'),
    ('cost_anomaly_threshold', '0.2', 'Threshold for detecting cost anomalies (20% increase)')
ON CONFLICT (key) DO NOTHING;

-- Create materialized views for better query performance
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_cost_summary AS
SELECT 
    provider_id,
    service_name,
    cost_date,
    SUM(cost_amount) as total_cost,
    COUNT(*) as resource_count,
    currency
FROM cost_data 
GROUP BY provider_id, service_name, cost_date, currency;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS daily_cost_summary_unique_idx 
ON daily_cost_summary (provider_id, service_name, cost_date, currency);

-- Create monthly cost summary view
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_cost_summary AS
SELECT 
    provider_id,
    service_name,
    DATE_TRUNC('month', cost_date) as month,
    SUM(cost_amount) as total_cost,
    COUNT(*) as resource_count,
    currency
FROM cost_data 
GROUP BY provider_id, service_name, DATE_TRUNC('month', cost_date), currency;

-- Create index on monthly summary
CREATE UNIQUE INDEX IF NOT EXISTS monthly_cost_summary_unique_idx 
ON monthly_cost_summary (provider_id, service_name, month, currency);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_cost_summaries()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_cost_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_cost_summary;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO finops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO finops;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO finops;

-- Create a job to refresh materialized views daily (requires pg_cron extension)
-- SELECT cron.schedule('refresh-cost-summaries', '0 2 * * *', 'SELECT refresh_cost_summaries();');

COMMIT;