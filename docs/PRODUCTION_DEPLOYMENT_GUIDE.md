# FinOps Cost Optimization Platform - Production Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying the FinOps Cost Optimization Platform to production environments. It covers infrastructure setup, security configuration, monitoring deployment, and operational procedures.

## Prerequisites

### Infrastructure Requirements
- **Kubernetes Cluster**: EKS 1.24+ with at least 3 worker nodes
- **Node Specifications**: 
  - Minimum: 4 vCPU, 16GB RAM per node
  - Recommended: 8 vCPU, 32GB RAM per node
- **Storage**: EBS CSI driver with gp3 storage class
- **Networking**: VPC with private subnets, NAT Gateway, Application Load Balancer
- **DNS**: Route53 hosted zone for domain management

### AWS Services Required
- **EKS**: Kubernetes cluster management
- **RDS**: PostgreSQL 15+ (Multi-AZ for production)
- **ElastiCache**: Redis 7+ (Cluster mode enabled)
- **S3**: Backup storage and audit logs
- **IAM**: Service roles and policies
- **CloudWatch**: Logging and basic monitoring
- **Route53**: DNS management
- **Certificate Manager**: SSL/TLS certificates

### Tools and Access
- `kubectl` v1.24+
- `helm` v3.8+
- `aws` CLI v2.0+
- `docker` v20.0+
- AWS Console access with administrative privileges
- Domain name and DNS management access

## Pre-Deployment Setup

### 1. AWS Infrastructure Setup

#### Create EKS Cluster
```bash
# Create EKS cluster using eksctl
eksctl create cluster \
  --name finops-production \
  --version 1.24 \
  --region us-east-1 \
  --nodegroup-name finops-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed \
  --enable-ssm \
  --asg-access \
  --external-dns-access \
  --full-ecr-access \
  --appmesh-access \
  --alb-ingress-access

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name finops-production
```

#### Create RDS Instance
```bash
# Create RDS subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name finops-db-subnet-group \
  --db-subnet-group-description "FinOps Database Subnet Group" \
  --subnet-ids subnet-12345678 subnet-87654321

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier finops-production-db \
  --db-instance-class db.r5.2xlarge \
  --engine postgres \
  --engine-version 15.4 \
  --master-username finops \
  --master-user-password "CHANGE_ME_IN_PRODUCTION" \
  --allocated-storage 500 \
  --storage-type gp3 \
  --storage-encrypted \
  --multi-az \
  --db-subnet-group-name finops-db-subnet-group \
  --vpc-security-group-ids sg-12345678 \
  --backup-retention-period 30 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "sun:04:00-sun:05:00" \
  --deletion-protection \
  --enable-performance-insights \
  --performance-insights-retention-period 7
```

#### Create ElastiCache Cluster
```bash
# Create ElastiCache subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name finops-redis-subnet-group \
  --cache-subnet-group-description "FinOps Redis Subnet Group" \
  --subnet-ids subnet-12345678 subnet-87654321

# Create ElastiCache replication group
aws elasticache create-replication-group \
  --replication-group-id finops-production-redis \
  --description "FinOps Production Redis Cluster" \
  --node-type cache.r6g.2xlarge \
  --cache-parameter-group-name default.redis7 \
  --port 6379 \
  --num-cache-clusters 3 \
  --cache-subnet-group-name finops-redis-subnet-group \
  --security-group-ids sg-87654321 \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token "CHANGE_ME_IN_PRODUCTION" \
  --automatic-failover-enabled \
  --multi-az-enabled \
  --snapshot-retention-limit 7 \
  --snapshot-window "03:00-05:00"
```

### 2. Security Configuration

#### Create IAM Roles and Policies
```bash
# Run the deployment script to set up IAM
./scripts/deploy-production.sh iam-only
```

#### Configure Secrets
```bash
# Create namespace
kubectl create namespace finops-automation

# Create secrets
kubectl create secret generic finops-secrets \
  --namespace=finops-automation \
  --from-literal=DATABASE_USER=finops \
  --from-literal=DATABASE_PASSWORD="YOUR_SECURE_DB_PASSWORD" \
  --from-literal=JWT_SECRET_KEY="YOUR_32_CHAR_JWT_SECRET_KEY_HERE" \
  --from-literal=ENCRYPTION_KEY="YOUR_32_CHAR_ENCRYPTION_KEY_HERE" \
  --from-literal=REDIS_PASSWORD="YOUR_SECURE_REDIS_PASSWORD" \
  --from-literal=SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
  --from-literal=EMAIL_SMTP_PASSWORD="YOUR_SMTP_PASSWORD"

# Verify secrets
kubectl get secrets -n finops-automation
```

### 3. Storage Configuration

#### Create Storage Classes
```bash
# Create EBS CSI storage classes
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-sc
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-utils
  fileSystemId: fs-12345678  # Replace with your EFS ID
  directoryPerms: "0755"
volumeBindingMode: Immediate
EOF
```

## Deployment Process

### 1. Build and Push Images
```bash
# Set environment variables
export ECR_REGISTRY="123456789012.dkr.ecr.us-east-1.amazonaws.com"
export IMAGE_TAG="v1.0.0"

# Build and push images
./scripts/deploy-production.sh build-only
```

### 2. Deploy Application
```bash
# Update deployment configuration
# Edit k8s/finops-production-deployment.yaml and replace:
# - ACCOUNT_ID with your AWS account ID
# - Domain names with your actual domains
# - Image references with your ECR registry

# Deploy to Kubernetes
./scripts/deploy-production.sh k8s-only
```

### 3. Setup Monitoring
```bash
# Deploy monitoring stack
./scripts/deploy-production.sh monitoring-only
```

### 4. Configure Ingress and SSL
```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb"
```

### 5. Database Migration
```bash
# Run database migrations
kubectl exec -it deployment/finops-api -n finops-automation -- \
  alembic upgrade head

# Verify database schema
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.database import engine; print('Database connection successful')"
```

## Post-Deployment Configuration

### 1. Initial System Configuration
```bash
# Create default automation policies
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/setup-default-policies.py

# Configure business hours
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/configure-business-hours.py \
    --timezone "America/New_York" \
    --start-hour 9 \
    --end-hour 17 \
    --days "monday,tuesday,wednesday,thursday,friday"

# Set up notification channels
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/configure-notifications.py \
    --email "finops-team@example.com" \
    --slack-webhook "$SLACK_WEBHOOK_URL"
```

### 2. AWS Integration Setup
```bash
# Configure AWS credentials and regions
kubectl patch configmap finops-config -n finops-automation --patch '
data:
  AWS_DEFAULT_REGION: "us-east-1"
  COST_OPTIMIZATION_ROLE_ARN: "arn:aws:iam::123456789012:role/FinOpsCostOptimizationRole"
  AUTOMATION_ENABLED: "true"
  SAFETY_CHECKS_ENABLED: "true"
'

# Restart pods to pick up new configuration
kubectl rollout restart deployment/finops-api -n finops-automation
kubectl rollout restart deployment/finops-worker -n finops-automation
```

### 3. Monitoring and Alerting Setup
```bash
# Configure Grafana admin password
kubectl patch secret prometheus-grafana -n monitoring --patch '
data:
  admin-password: '$(echo -n "YOUR_SECURE_GRAFANA_PASSWORD" | base64)

# Import FinOps dashboards
kubectl exec -it deployment/prometheus-grafana -n monitoring -- \
  grafana-cli admin reset-admin-password "YOUR_SECURE_GRAFANA_PASSWORD"

# Configure alert manager
kubectl patch secret alertmanager-prometheus-kube-prometheus-alertmanager -n monitoring --patch '
data:
  alertmanager.yml: '$(cat k8s/alertmanager-config.yml | base64 -w 0)
```

## Validation and Testing

### 1. Health Checks
```bash
# Run comprehensive health check
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/health-check.py --comprehensive

# Check all endpoints
curl -f https://api.finops.example.com/health
curl -f https://api.finops.example.com/health/ready
curl -f https://api.finops.example.com/health/live
```

### 2. Functional Testing
```bash
# Test automation system (dry run)
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/test-automation-system.py --dry-run

# Test safety systems
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/test-safety-system.py --comprehensive

# Test notification system
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/test-notifications.py --all-channels
```

### 3. Load Testing
```bash
# Install k6 for load testing
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: k6-load-test
  namespace: finops-automation
data:
  load-test.js: |
    import http from 'k6/http';
    import { check } from 'k6';
    
    export let options = {
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 0 },
      ],
    };
    
    export default function() {
      let response = http.get('https://api.finops.example.com/health');
      check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
      });
    }
---
apiVersion: batch/v1
kind: Job
metadata:
  name: k6-load-test
  namespace: finops-automation
spec:
  template:
    spec:
      containers:
      - name: k6
        image: grafana/k6:latest
        command: ["k6", "run", "/scripts/load-test.js"]
        volumeMounts:
        - name: k6-script
          mountPath: /scripts
      volumes:
      - name: k6-script
        configMap:
          name: k6-load-test
      restartPolicy: Never
EOF
```

## Backup and Disaster Recovery Setup

### 1. Configure Automated Backups
```bash
# Set up backup infrastructure
./scripts/backup-restore.sh setup

# Configure automated daily backups
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: finops-daily-backup
  namespace: finops-automation
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: finops/api:latest
            command: ["/bin/bash", "-c"]
            args:
            - |
              /app/scripts/backup-restore.sh backup-all
              /app/scripts/backup-restore.sh cleanup
          restartPolicy: OnFailure
EOF
```

### 2. Test Disaster Recovery
```bash
# Test backup system
./scripts/backup-restore.sh health-check

# Perform test restore (in non-production environment)
./scripts/backup-restore.sh list
./scripts/backup-restore.sh restore-db TIMESTAMP
```

## Security Hardening

### 1. Network Security
```bash
# Apply network policies
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: finops-network-policy
  namespace: finops-automation
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
EOF
```

### 2. Pod Security Standards
```bash
# Apply pod security standards
kubectl label namespace finops-automation \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### 3. RBAC Configuration
```bash
# Create service account with minimal permissions
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: finops-service-account
  namespace: finops-automation
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/FinOpsCostOptimizationRole
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: finops-automation
  name: finops-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: finops-role-binding
  namespace: finops-automation
subjects:
- kind: ServiceAccount
  name: finops-service-account
  namespace: finops-automation
roleRef:
  kind: Role
  name: finops-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

## Monitoring and Alerting Configuration

### 1. Custom Metrics
```bash
# Deploy custom metrics
kubectl apply -f k8s/monitoring-config.yaml
```

### 2. Alert Configuration
```bash
# Configure alert routing
kubectl patch secret alertmanager-prometheus-kube-prometheus-alertmanager -n monitoring --patch "
data:
  alertmanager.yml: $(cat <<EOF | base64 -w 0
global:
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'finops-alerts@example.com'
  
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'finops-team'
  routes:
  - match:
      severity: critical
    receiver: 'finops-critical'
    
receivers:
- name: 'finops-team'
  email_configs:
  - to: 'finops-team@example.com'
    subject: '[FinOps] {{ .GroupLabels.alertname }}'
    
- name: 'finops-critical'
  email_configs:
  - to: 'finops-oncall@example.com'
    subject: '[CRITICAL] FinOps Alert'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#finops-critical'
EOF
)"
```

## Operational Procedures

### 1. Daily Operations Checklist
- [ ] Check system health dashboard
- [ ] Review overnight alerts and incidents
- [ ] Verify backup completion
- [ ] Monitor cost optimization metrics
- [ ] Review audit logs for anomalies

### 2. Weekly Operations Checklist
- [ ] Review performance trends
- [ ] Update security patches
- [ ] Analyze cost savings reports
- [ ] Review and update documentation
- [ ] Test disaster recovery procedures

### 3. Monthly Operations Checklist
- [ ] Conduct security audit
- [ ] Review capacity planning
- [ ] Update runbooks and procedures
- [ ] Perform chaos engineering tests
- [ ] Review and optimize costs

## Troubleshooting

### Common Issues and Solutions

#### Pods Not Starting
```bash
# Check pod status and events
kubectl describe pod -n finops-automation
kubectl get events -n finops-automation --sort-by='.lastTimestamp'

# Check resource constraints
kubectl top nodes
kubectl top pods -n finops-automation
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/finops-api -n finops-automation -- \
  pg_isready -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER

# Check database logs
aws rds describe-db-log-files --db-instance-identifier finops-production-db
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n finops-automation
kubectl describe hpa -n finops-automation

# Check application metrics
curl -s https://api.finops.example.com/metrics | grep -E "(request_duration|error_rate)"
```

## Support and Escalation

### Support Contacts
- **Platform Team**: finops-team@example.com
- **On-Call Engineer**: finops-oncall@example.com
- **Security Team**: security@example.com
- **Infrastructure Team**: infrastructure@example.com

### Escalation Procedures
1. **P0 (Critical)**: Immediate notification to on-call engineer
2. **P1 (High)**: Notification within 1 hour to platform team
3. **P2 (Medium)**: Notification within 4 hours to platform team
4. **P3 (Low)**: Next business day notification

### Documentation and Resources
- **Runbooks**: `/docs/runbooks/`
- **API Documentation**: `https://api.finops.example.com/docs`
- **Monitoring**: `https://monitoring.finops.example.com`
- **Status Page**: `https://status.finops.example.com`

## Conclusion

This deployment guide provides a comprehensive approach to deploying the FinOps Cost Optimization Platform in production. Follow all steps carefully, test thoroughly, and maintain proper documentation of any customizations made for your specific environment.

For questions or issues not covered in this guide, refer to the runbooks in `/docs/runbooks/` or contact the platform team.