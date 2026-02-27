# Cloud Migration Advisor Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Cloud Migration Advisor platform to production using Kubernetes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Scaling](#scaling)
8. [Backup and Recovery](#backup-and-recovery)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- **Kubernetes Cluster**: v1.24 or higher
- **kubectl**: v1.24 or higher
- **Docker**: v20.10 or higher
- **Helm**: v3.10 or higher (optional)
- **Git**: v2.30 or higher

### Cluster Requirements

**Minimum Resources**:
- 3 worker nodes
- 8 CPU cores per node
- 16 GB RAM per node
- 100 GB storage per node

**Recommended Resources**:
- 5 worker nodes
- 16 CPU cores per node
- 32 GB RAM per node
- 500 GB storage per node

### Access Requirements

- Kubernetes cluster admin access
- Container registry access (GitHub Container Registry, Docker Hub, etc.)
- DNS management access
- SSL certificate management access

## Infrastructure Setup

### 1. Create Kubernetes Cluster

#### AWS EKS

```bash
eksctl create cluster \
  --name migration-advisor \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

#### GCP GKE

```bash
gcloud container clusters create migration-advisor \
  --region us-central1 \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

#### Azure AKS

```bash
az aks create \
  --resource-group migration-advisor-rg \
  --name migration-advisor \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

### 2. Configure kubectl

```bash
# AWS
aws eks update-kubeconfig --name migration-advisor --region us-east-1

# GCP
gcloud container clusters get-credentials migration-advisor --region us-central1

# Azure
az aks get-credentials --resource-group migration-advisor-rg --name migration-advisor
```

### 3. Install Ingress Controller

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Verify installation
kubectl get pods -n ingress-nginx
```

### 4. Install Cert-Manager (for SSL)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@cloudmigration.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Configuration

### 1. Create Namespace

```bash
kubectl create namespace migration-advisor
```

### 2. Configure Secrets

Create a secrets file (do not commit to version control):

```bash
# Create secrets.yaml
cat <<EOF > secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: migration-advisor-secrets
  namespace: migration-advisor
type: Opaque
stringData:
  DATABASE_USER: "migration_user"
  DATABASE_PASSWORD: "$(openssl rand -base64 32)"
  JWT_SECRET_KEY: "$(openssl rand -base64 64)"
  REDIS_PASSWORD: "$(openssl rand -base64 32)"
  GRAFANA_PASSWORD: "$(openssl rand -base64 16)"
EOF

# Apply secrets
kubectl apply -f secrets.yaml

# Delete the file
rm secrets.yaml
```

### 3. Configure ConfigMap

Update the ConfigMap in `k8s/migration-advisor-deployment.yaml` with your values:

```yaml
data:
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "migration_advisor"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
```

### 4. Configure Ingress

Update the Ingress host in `k8s/migration-advisor-deployment.yaml`:

```yaml
spec:
  tls:
  - hosts:
    - api.yourdomain.com  # Change this
    secretName: migration-advisor-tls
  rules:
  - host: api.yourdomain.com  # Change this
```

## Deployment

### 1. Build Docker Image

```bash
# Build image
docker build -t migration-advisor/api:latest .

# Tag for registry
docker tag migration-advisor/api:latest ghcr.io/your-org/migration-advisor:latest

# Push to registry
docker push ghcr.io/your-org/migration-advisor:latest
```

### 2. Deploy Database

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n migration-advisor --timeout=300s

# Verify
kubectl get pods -n migration-advisor -l app=postgres
```

### 3. Deploy Redis

```bash
# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis -n migration-advisor --timeout=300s

# Verify
kubectl get pods -n migration-advisor -l app=redis
```

### 4. Run Database Migrations

```bash
# Create a migration job
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: migration-advisor
spec:
  template:
    spec:
      containers:
      - name: migration
        image: ghcr.io/your-org/migration-advisor:latest
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - configMapRef:
            name: migration-advisor-config
        - secretRef:
            name: migration-advisor-secrets
      restartPolicy: Never
  backoffLimit: 3
EOF

# Wait for migration to complete
kubectl wait --for=condition=complete job/db-migration -n migration-advisor --timeout=300s

# Check logs
kubectl logs job/db-migration -n migration-advisor
```

### 5. Deploy Application

```bash
# Deploy application
kubectl apply -f k8s/migration-advisor-deployment.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/migration-advisor-api -n migration-advisor --timeout=300s

# Verify
kubectl get pods -n migration-advisor -l app=migration-advisor
kubectl get services -n migration-advisor
kubectl get ingress -n migration-advisor
```

### 6. Verify Deployment

```bash
# Check all resources
kubectl get all -n migration-advisor

# Check pod logs
kubectl logs -f deployment/migration-advisor-api -n migration-advisor

# Test health endpoint
kubectl port-forward -n migration-advisor svc/migration-advisor-api 8000:80
curl http://localhost:8000/health
```

## Monitoring and Logging

### 1. Deploy Monitoring Stack

```bash
# Deploy Prometheus and Grafana
kubectl apply -f k8s/monitoring.yaml

# Wait for deployments
kubectl wait --for=condition=available deployment/prometheus -n migration-advisor --timeout=300s
kubectl wait --for=condition=available deployment/grafana -n migration-advisor --timeout=300s

# Access Grafana
kubectl port-forward -n migration-advisor svc/grafana 3000:3000
# Open http://localhost:3000
# Login with admin / <GRAFANA_PASSWORD from secrets>
```

### 2. Configure Alerts

Create alert rules in Prometheus:

```yaml
# alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
  namespace: migration-advisor
data:
  alerts.yml: |
    groups:
    - name: migration-advisor
      rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total{namespace="migration-advisor"}[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
```

### 3. Deploy Logging

```bash
# Deploy Fluentd
kubectl apply -f k8s/monitoring.yaml

# Verify
kubectl get daemonset -n migration-advisor fluentd
```

## CI/CD Pipeline

### 1. Configure GitHub Actions

The CI/CD pipeline is defined in `.github/workflows/deploy.yml`.

**Required Secrets**:
- `KUBE_CONFIG_STAGING`: Kubeconfig for staging cluster
- `KUBE_CONFIG_PRODUCTION`: Kubeconfig for production cluster
- `SLACK_WEBHOOK`: Slack webhook for notifications (optional)

Add secrets in GitHub:
1. Go to repository Settings > Secrets and variables > Actions
2. Add each secret

### 2. Deployment Workflow

**On Push to `develop` branch**:
1. Run tests
2. Build Docker image
3. Deploy to staging environment

**On Push to `main` branch**:
1. Run tests
2. Build Docker image
3. Deploy to production environment
4. Run smoke tests
5. Send notification

### 3. Manual Deployment

To deploy manually:

```bash
# Build and push image
docker build -t ghcr.io/your-org/migration-advisor:v1.0.0 .
docker push ghcr.io/your-org/migration-advisor:v1.0.0

# Update deployment
kubectl set image deployment/migration-advisor-api \
  api=ghcr.io/your-org/migration-advisor:v1.0.0 \
  -n migration-advisor

# Monitor rollout
kubectl rollout status deployment/migration-advisor-api -n migration-advisor
```

## Scaling

### 1. Horizontal Pod Autoscaling

HPA is configured in `k8s/migration-advisor-deployment.yaml`:

```yaml
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Monitor autoscaling:

```bash
kubectl get hpa -n migration-advisor
kubectl describe hpa migration-advisor-api-hpa -n migration-advisor
```

### 2. Vertical Pod Autoscaling

Install VPA:

```bash
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh
```

Create VPA for migration-advisor:

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: migration-advisor-api-vpa
  namespace: migration-advisor
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: migration-advisor-api
  updatePolicy:
    updateMode: "Auto"
```

### 3. Cluster Autoscaling

Cluster autoscaling is configured during cluster creation. Monitor:

```bash
# AWS
kubectl get nodes
kubectl describe configmap cluster-autoscaler-status -n kube-system

# GCP
gcloud container clusters describe migration-advisor --region us-central1

# Azure
az aks show --resource-group migration-advisor-rg --name migration-advisor
```

## Backup and Recovery

### 1. Database Backups

Create a CronJob for automated backups:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: migration-advisor
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h postgres-service -U $DATABASE_USER -d $DATABASE_NAME | \
              gzip > /backups/backup-$(date +%Y%m%d-%H%M%S).sql.gz
            envFrom:
            - secretRef:
                name: migration-advisor-secrets
            volumeMounts:
            - name: backups
              mountPath: /backups
          volumes:
          - name: backups
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### 2. Restore from Backup

```bash
# Copy backup file to pod
kubectl cp backup-20231116-020000.sql.gz \
  migration-advisor/postgres-0:/tmp/backup.sql.gz

# Restore
kubectl exec -it -n migration-advisor postgres-0 -- \
  bash -c "gunzip < /tmp/backup.sql.gz | psql -U migration_user -d migration_advisor"
```

### 3. Disaster Recovery

**Full Cluster Backup**:

```bash
# Backup all resources
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml

# Backup specific namespace
kubectl get all -n migration-advisor -o yaml > migration-advisor-backup.yaml
```

**Restore**:

```bash
kubectl apply -f migration-advisor-backup.yaml
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n migration-advisor

# Describe pod
kubectl describe pod <pod-name> -n migration-advisor

# Check logs
kubectl logs <pod-name> -n migration-advisor

# Common causes:
# - Image pull errors: Check image name and registry access
# - Resource limits: Check if node has enough resources
# - Configuration errors: Check ConfigMap and Secrets
```

#### 2. Database Connection Errors

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -n migration-advisor -- \
  psql -h postgres-service -U migration_user -d migration_advisor

# Check database pod
kubectl logs -n migration-advisor postgres-0

# Check service
kubectl get svc -n migration-advisor postgres-service
```

#### 3. High Memory Usage

```bash
# Check resource usage
kubectl top pods -n migration-advisor

# Increase memory limits
kubectl set resources deployment/migration-advisor-api \
  --limits=memory=4Gi \
  -n migration-advisor
```

#### 4. SSL Certificate Issues

```bash
# Check certificate
kubectl get certificate -n migration-advisor

# Describe certificate
kubectl describe certificate migration-advisor-tls -n migration-advisor

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager
```

### Health Checks

```bash
# API health
curl https://api.yourdomain.com/health

# Database health
kubectl exec -n migration-advisor postgres-0 -- pg_isready

# Redis health
kubectl exec -n migration-advisor deployment/redis -- redis-cli ping
```

### Performance Tuning

**Database**:
```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check connections
SELECT count(*) FROM pg_stat_activity;
```

**Application**:
```bash
# Check request latency
kubectl exec -n migration-advisor deployment/migration-advisor-api -- \
  curl localhost:8000/metrics | grep http_request_duration

# Check memory usage
kubectl exec -n migration-advisor deployment/migration-advisor-api -- \
  ps aux | grep python
```

## Security Checklist

- [ ] Secrets are not committed to version control
- [ ] TLS/SSL certificates are configured
- [ ] Network policies are in place
- [ ] RBAC is configured
- [ ] Pod security policies are enabled
- [ ] Container images are scanned for vulnerabilities
- [ ] Database credentials are rotated regularly
- [ ] Audit logging is enabled
- [ ] Backup encryption is enabled
- [ ] Monitoring and alerting are configured

## Support

For deployment issues:
- **Documentation**: https://docs.cloudmigration.example.com/deployment
- **Support Email**: devops@cloudmigration.example.com
- **Slack Channel**: #migration-advisor-ops

---

**Version**: 1.0.0  
**Last Updated**: November 2023
