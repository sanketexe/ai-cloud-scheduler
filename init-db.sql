-- Initialize Cloud Intelligence Platform Database
-- This script sets up the initial database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS workloads;
CREATE SCHEMA IF NOT EXISTS costs;
CREATE SCHEMA IF NOT EXISTS performance;
CREATE SCHEMA IF NOT EXISTS security;

-- Workloads schema tables
CREATE TABLE IF NOT EXISTS workloads.workload_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workload_id INTEGER NOT NULL,
    cpu_required INTEGER NOT NULL,
    memory_required_gb INTEGER NOT NULL,
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    vm_id INTEGER,
    provider VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    cost DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workloads.vm_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id INTEGER NOT NULL,
    provider VARCHAR(50) NOT NULL,
    cpu_capacity INTEGER NOT NULL,
    memory_capacity_gb INTEGER NOT NULL,
    cpu_used INTEGER DEFAULT 0,
    memory_used_gb INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Costs schema tables
CREATE TABLE IF NOT EXISTS costs.cost_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider VARCHAR(50) NOT NULL,
    service VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    cost_amount DECIMAL(12,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    cost_category VARCHAR(50),
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS costs.budgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    period VARCHAR(20) NOT NULL, -- monthly, quarterly, yearly
    alert_thresholds DECIMAL[] DEFAULT ARRAY[0.8, 0.9, 1.0],
    scope JSONB, -- filters for budget scope
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance schema tables
CREATE TABLE IF NOT EXISTS performance.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance.anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    description TEXT,
    suggested_actions TEXT[],
    status VARCHAR(20) DEFAULT 'open',
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Security schema tables
CREATE TABLE IF NOT EXISTS security.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'success'
);

CREATE TABLE IF NOT EXISTS security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_workload_history_scheduled_at ON workloads.workload_history(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_workload_history_provider ON workloads.workload_history(provider);
CREATE INDEX IF NOT EXISTS idx_vm_status_provider ON workloads.vm_status(provider);
CREATE INDEX IF NOT EXISTS idx_cost_data_provider_service ON costs.cost_data(provider, service);
CREATE INDEX IF NOT EXISTS idx_cost_data_billing_period ON costs.cost_data(billing_period_start, billing_period_end);
CREATE INDEX IF NOT EXISTS idx_metrics_resource_timestamp ON performance.metrics(resource_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON performance.metrics(metric_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_anomalies_detected_at ON performance.anomalies(detected_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON security.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_action ON security.audit_logs(user_id, action);

-- Create views for common queries
CREATE OR REPLACE VIEW workloads.current_vm_utilization AS
SELECT 
    vm_id,
    provider,
    cpu_capacity,
    memory_capacity_gb,
    cpu_used,
    memory_used_gb,
    ROUND((cpu_used::DECIMAL / cpu_capacity) * 100, 2) as cpu_utilization_percent,
    ROUND((memory_used_gb::DECIMAL / memory_capacity_gb) * 100, 2) as memory_utilization_percent,
    last_updated
FROM workloads.vm_status
WHERE status = 'active';

CREATE OR REPLACE VIEW costs.monthly_cost_summary AS
SELECT 
    provider,
    service,
    DATE_TRUNC('month', billing_period_start) as month,
    SUM(cost_amount) as total_cost,
    COUNT(*) as resource_count
FROM costs.cost_data
GROUP BY provider, service, DATE_TRUNC('month', billing_period_start)
ORDER BY month DESC, total_cost DESC;

-- Insert sample data
INSERT INTO workloads.vm_status (vm_id, provider, cpu_capacity, memory_capacity_gb) VALUES
(1, 'AWS', 4, 16),
(2, 'GCP', 8, 32),
(3, 'Azure', 4, 16),
(4, 'GCP', 2, 8)
ON CONFLICT DO NOTHING;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION workloads.update_vm_utilization(
    p_vm_id INTEGER,
    p_cpu_delta INTEGER,
    p_memory_delta INTEGER
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE workloads.vm_status 
    SET 
        cpu_used = cpu_used + p_cpu_delta,
        memory_used_gb = memory_used_gb + p_memory_delta,
        last_updated = NOW()
    WHERE vm_id = p_vm_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA workloads TO postgres;
GRANT USAGE ON SCHEMA costs TO postgres;
GRANT USAGE ON SCHEMA performance TO postgres;
GRANT USAGE ON SCHEMA security TO postgres;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA workloads TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA costs TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA performance TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA security TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA workloads TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA costs TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA performance TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA security TO postgres;