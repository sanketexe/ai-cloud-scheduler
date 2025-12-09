# FinOps Platform Monitoring and Health Checks

This directory contains monitoring, logging, and health check configurations for the FinOps Platform.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **ELK Stack**: Log aggregation and analysis (Elasticsearch, Logstash, Kibana)
- **Filebeat**: Log shipping from containers
- **cAdvisor**: Container resource monitoring
- **Node Exporter**: System metrics collection

## Health Check Endpoints

The platform provides comprehensive health check endpoints:

### Basic Health Checks
- `GET /health` - Basic health status
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe

### Detailed Health Checks
- `GET /health/detailed` - Comprehensive system health
- `GET /health/dependencies` - External dependency status
- `GET /health/metrics` - Prometheus-format metrics

### Component-Specific Health Checks
- `GET /api/v1/health/cache` - Redis cache health
- `GET /api/v1/health/system` - System resource health
- `GET /api/v1/health/performance` - Performance metrics

## Running with Monitoring

### Development Environment

By default, monitoring services are disabled in development. To enable them:

```bash
# Start with monitoring services
docker-compose --profile monitoring --profile logging up -d

# Or start specific monitoring components
docker-compose --profile monitoring up -d
```

### Production Environment

Use the production configuration that enables all monitoring:

```bash
# Production deployment with full monitoring
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Accessing Monitoring Services

Once running, access the monitoring services at:

- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **cAdvisor**: http://localhost:8080

## Health Check Scripts

### Python Health Checker

Use the comprehensive Python health checker:

```bash
# Basic health check
python scripts/health-check.py

# Check specific URL
python scripts/health-check.py --url http://localhost:8000

# JSON output for automation
python scripts/health-check.py --format json

# Exit with error code if unhealthy (useful for CI/CD)
python scripts/health-check.py --exit-code
```

### Docker Health Check Script

The Docker health check script is used internally by containers:

```bash
# Manual health check (inside container)
./scripts/docker-health-check.sh

# Check specific service type
SERVICE_TYPE=worker ./scripts/docker-health-check.sh
SERVICE_TYPE=scheduler ./scripts/docker-health-check.sh
```

## Container Health Checks

Each service has specific health check configurations:

### API Service
- **Check**: HTTP GET to `/health/ready`
- **Interval**: 30s
- **Timeout**: 10s
- **Retries**: 3
- **Start Period**: 60s

### Database (PostgreSQL)
- **Check**: `pg_isready -U finops -d finops_db`
- **Interval**: 10s
- **Timeout**: 5s
- **Retries**: 5
- **Start Period**: 30s

### Cache (Redis)
- **Check**: `redis-cli -a redis_password ping`
- **Interval**: 10s
- **Timeout**: 3s
- **Retries**: 5
- **Start Period**: 10s

### Worker Services
- **Check**: Process existence check
- **Interval**: 60s
- **Timeout**: 30s
- **Retries**: 3
- **Start Period**: 120s

## Log Aggregation

### Log Collection
- **Filebeat** collects logs from all Docker containers
- **Logstash** processes and enriches log data
- **Elasticsearch** stores processed logs
- **Kibana** provides log search and visualization

### Log Format
All services use structured JSON logging with:
- Timestamp
- Log level
- Service name
- Correlation ID (for request tracing)
- Contextual metadata

### Log Retention
- Container logs: 10MB max size, 3-5 files retained
- Elasticsearch: Configurable retention policy
- Log rotation handled automatically

## Metrics Collection

### Application Metrics
- API response times
- Database query performance
- Cache hit/miss ratios
- Error rates and counts
- Business metrics (cost data processing, etc.)

### System Metrics
- CPU usage
- Memory consumption
- Disk I/O
- Network traffic
- Container resource usage

### Custom Metrics
The platform exposes custom metrics at `/health/metrics`:
- `finops_database_health` - Database connectivity (1=healthy, 0=unhealthy)
- `finops_redis_health` - Redis connectivity
- `finops_cpu_usage_percent` - CPU utilization
- `finops_memory_usage_percent` - Memory utilization
- `finops_database_response_time_ms` - Database response time
- `finops_redis_response_time_ms` - Redis response time

## Alerting

### Grafana Alerts
Configure alerts in Grafana for:
- Service health status
- High resource usage (CPU > 80%, Memory > 85%)
- Slow response times (> 2 seconds)
- Low cache hit ratios (< 70%)

### Prometheus Alerting
Set up Prometheus alerting rules for:
- Service downtime
- Database connectivity issues
- Redis connectivity issues
- Container resource limits

## Troubleshooting

### Common Issues

1. **Health checks failing**
   ```bash
   # Check container logs
   docker-compose logs api
   
   # Run manual health check
   python scripts/health-check.py --url http://localhost:8000
   ```

2. **Monitoring services not starting**
   ```bash
   # Check if profiles are enabled
   docker-compose --profile monitoring ps
   
   # Check resource availability
   docker system df
   ```

3. **Log aggregation issues**
   ```bash
   # Check Elasticsearch health
   curl http://localhost:9200/_cluster/health
   
   # Check Logstash pipeline
   curl http://localhost:9600/_node/stats
   ```

### Performance Tuning

1. **Elasticsearch**
   - Adjust heap size: `ES_JAVA_OPTS=-Xms1g -Xmx1g`
   - Configure index lifecycle management
   - Set up index templates

2. **Prometheus**
   - Adjust retention time: `--storage.tsdb.retention.time=200h`
   - Configure recording rules for expensive queries
   - Set up federation for multi-instance deployments

3. **Container Resources**
   - Monitor resource usage with cAdvisor
   - Adjust memory and CPU limits based on usage patterns
   - Use horizontal scaling for high-load services

## Security Considerations

1. **Monitoring Access**
   - Change default Grafana password
   - Set up proper authentication for monitoring services
   - Use network policies to restrict access

2. **Log Security**
   - Ensure sensitive data is not logged
   - Set up log encryption in transit
   - Configure proper access controls for log data

3. **Metrics Security**
   - Protect metrics endpoints with authentication
   - Use TLS for metrics collection
   - Sanitize metric labels to prevent information leakage

## Configuration Files

- `prometheus.yml` - Prometheus configuration
- `grafana/provisioning/` - Grafana datasources and dashboards
- `logstash/` - Logstash pipeline configuration
- `filebeat/filebeat.yml` - Filebeat log shipping configuration
- `../docker-compose.yml` - Main service definitions with health checks
- `../docker-compose.prod.yml` - Production overrides
- `../docker-compose.override.yml` - Development overrides