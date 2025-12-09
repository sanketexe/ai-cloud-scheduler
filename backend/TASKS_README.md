# FinOps Platform - Background Task System

This document describes the background task system implemented for the FinOps Platform using Celery and Redis.

## Overview

The background task system handles automated data synchronization, monitoring, and processing tasks that are essential for the FinOps Platform's operation. It includes:

- **Cost Data Synchronization**: Automated collection of cost data from cloud providers
- **Resource Discovery**: Periodic discovery of cloud resources and their metadata
- **Budget Monitoring**: Real-time monitoring of budget thresholds and alerts
- **Optimization Analysis**: Analysis of cost optimization opportunities
- **Anomaly Detection**: Detection of unusual spending patterns
- **Data Maintenance**: Cleanup of old data and maintenance tasks

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Celery Worker  │    │  Celery Beat    │
│                 │    │                 │    │   Scheduler     │
│  - Task APIs    │    │  - Execute      │    │                 │
│  - Monitoring   │    │    Tasks        │    │  - Periodic     │
│  - Management   │    │  - Process      │    │    Tasks        │
│                 │    │    Data         │    │  - Scheduling   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │      Redis      │
                    │                 │
                    │  - Message      │
                    │    Broker       │
                    │  - Result       │
                    │    Backend      │
                    └─────────────────┘
```

## Task Types

### 1. Cost Data Synchronization
- **Task**: `sync_provider_cost_data`
- **Schedule**: Every 6 hours
- **Purpose**: Collect cost and usage data from cloud provider APIs
- **Queue**: `cost_sync`

### 2. Resource Discovery
- **Task**: `discover_provider_resources`
- **Schedule**: Daily at 2 AM UTC
- **Purpose**: Discover and inventory cloud resources
- **Queue**: `resource_discovery`

### 3. Budget Monitoring
- **Task**: `monitor_budget`
- **Schedule**: Every hour
- **Purpose**: Check budget thresholds and trigger alerts
- **Queue**: `monitoring`

### 4. Optimization Analysis
- **Task**: `analyze_optimization_opportunities`
- **Schedule**: Daily at 3 AM UTC
- **Purpose**: Generate cost optimization recommendations
- **Queue**: `analysis`

### 5. Anomaly Detection
- **Task**: `detect_cost_anomalies`
- **Schedule**: Every 4 hours
- **Purpose**: Detect unusual spending patterns
- **Queue**: `analysis`

### 6. Data Cleanup
- **Task**: `cleanup_old_data`
- **Schedule**: Weekly on Sunday at 1 AM UTC
- **Purpose**: Clean up old data and maintain database performance
- **Queue**: `maintenance`

## Development Setup

### Prerequisites
- Redis server
- PostgreSQL database
- Python 3.9+

### Quick Start

1. **Start the task system for development**:
   ```bash
   python start_dev_tasks.py
   ```

2. **Or start components individually**:
   ```bash
   # Start Redis (if not running)
   redis-server
   
   # Start Celery worker
   cd backend
   python celery_worker.py
   
   # Start Celery beat scheduler (in another terminal)
   cd backend
   python celery_beat.py
   ```

### Monitoring Tasks

Use the monitoring script to check task status:

```bash
# Show active tasks
python backend/monitor_tasks.py active

# Show worker statistics
python backend/monitor_tasks.py workers

# Check specific task status
python backend/monitor_tasks.py status <task_id>

# Watch tasks in real-time
python backend/monitor_tasks.py watch

# Cancel a task
python backend/monitor_tasks.py cancel <task_id>
```

## Production Deployment

### Docker Compose

The system includes Docker services for production deployment:

```yaml
services:
  redis:      # Message broker and result backend
  worker:     # Celery worker processes
  scheduler:  # Celery beat scheduler
```

### Environment Variables

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis
CELERY_BROKER_URL=redis://host:port/db
CELERY_RESULT_BACKEND=redis://host:port/db

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-32-char-encryption-key
```

### Scaling

- **Workers**: Scale horizontally by adding more worker containers
- **Queues**: Tasks are distributed across specialized queues
- **Redis**: Use Redis Cluster for high availability
- **Monitoring**: Use Flower for web-based monitoring

## API Endpoints

The task system provides REST API endpoints for management:

### Trigger Tasks
- `POST /api/v1/tasks/sync/cost-data` - Trigger cost data sync
- `POST /api/v1/tasks/sync/resources` - Trigger resource discovery
- `POST /api/v1/tasks/monitor/budgets` - Trigger budget monitoring
- `POST /api/v1/tasks/analyze/optimization` - Trigger optimization analysis
- `POST /api/v1/tasks/analyze/anomalies` - Trigger anomaly detection

### Monitor Tasks
- `GET /api/v1/tasks/status/{task_id}` - Get task status
- `GET /api/v1/tasks/active` - List active tasks
- `GET /api/v1/tasks/workers/stats` - Get worker statistics
- `DELETE /api/v1/tasks/cancel/{task_id}` - Cancel a task

### Health Check
- `GET /api/v1/tasks/health` - Task system health check

## Task Configuration

### Retry Logic
- **Max Retries**: 3 attempts for most tasks
- **Backoff**: Exponential backoff (1, 2, 4 minutes)
- **Timeout**: 30 minutes hard limit, 25 minutes soft limit

### Queue Configuration
- **default**: General tasks
- **cost_sync**: Cost data synchronization
- **resource_discovery**: Resource discovery tasks
- **monitoring**: Budget and alert monitoring
- **analysis**: Optimization and anomaly analysis
- **maintenance**: Data cleanup and maintenance

### Error Handling
- Structured logging with correlation IDs
- Automatic retry with exponential backoff
- Dead letter queue for failed tasks
- Alert notifications for critical failures

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Check Redis logs
   redis-cli monitor
   ```

2. **Worker Not Processing Tasks**
   ```bash
   # Check worker logs
   python backend/monitor_tasks.py workers
   
   # Restart worker
   docker-compose restart worker
   ```

3. **Tasks Stuck in PENDING**
   ```bash
   # Check if workers are running
   python backend/monitor_tasks.py active
   
   # Check Redis connection
   redis-cli info
   ```

4. **High Memory Usage**
   ```bash
   # Check worker memory limits
   python backend/monitor_tasks.py workers
   
   # Restart workers to clear memory
   docker-compose restart worker
   ```

### Monitoring

- **Logs**: Structured JSON logs with correlation IDs
- **Metrics**: Task execution times, success/failure rates
- **Alerts**: Failed task notifications
- **Health Checks**: Automated health monitoring

## Security Considerations

- **Credentials**: Cloud provider credentials are encrypted at rest
- **Network**: Redis should not be exposed to public internet
- **Authentication**: Task API endpoints require authentication
- **Audit**: All task executions are logged for audit trails

## Performance Optimization

- **Connection Pooling**: Database and Redis connection pooling
- **Batch Processing**: Bulk operations for large datasets
- **Caching**: Redis caching for frequently accessed data
- **Compression**: Gzip compression for task payloads
- **Resource Limits**: Memory and CPU limits for workers