# FinOps Cost Optimization Platform - Operational Runbooks

This directory contains operational runbooks and troubleshooting guides for the FinOps Cost Optimization Platform. These runbooks provide step-by-step procedures for common operational tasks, incident response, and system maintenance.

## Quick Reference

### Emergency Contacts
- **On-Call Engineer**: finops-oncall@example.com
- **Platform Team**: finops-team@example.com  
- **Security Team**: security@example.com
- **Infrastructure Team**: infrastructure@example.com

### Critical System URLs
- **Production API**: https://api.finops.example.com
- **Monitoring Dashboard**: https://monitoring.finops.example.com
- **Log Analysis**: https://logs.finops.example.com
- **Status Page**: https://status.finops.example.com

### Emergency Procedures
1. **System Down**: [System Outage Response](./system-outage-response.md)
2. **Security Incident**: [Security Incident Response](./security-incident-response.md)
3. **Data Loss**: [Data Recovery Procedures](./data-recovery-procedures.md)
4. **Cost Optimization Failure**: [Automation Failure Response](./automation-failure-response.md)

## Runbook Categories

### 🚨 Incident Response
- [System Outage Response](./system-outage-response.md)
- [Security Incident Response](./security-incident-response.md)
- [Performance Degradation](./performance-degradation.md)
- [Automation Failure Response](./automation-failure-response.md)

### 🔧 Maintenance Procedures
- [Deployment Procedures](./deployment-procedures.md)
- [Database Maintenance](./database-maintenance.md)
- [Backup and Recovery](./backup-recovery.md)
- [Certificate Renewal](./certificate-renewal.md)

### 📊 Monitoring and Alerting
- [Alert Investigation Guide](./alert-investigation.md)
- [Metrics Troubleshooting](./metrics-troubleshooting.md)
- [Log Analysis Guide](./log-analysis.md)
- [Health Check Procedures](./health-check-procedures.md)

### 💰 Cost Optimization Specific
- [Safety Check Failures](./safety-check-failures.md)
- [Automation Engine Troubleshooting](./automation-engine-troubleshooting.md)
- [Policy Configuration Issues](./policy-configuration-issues.md)
- [Rollback Procedures](./rollback-procedures.md)

### 🔐 Security Operations
- [Access Management](./access-management.md)
- [Audit Log Investigation](./audit-log-investigation.md)
- [Compliance Reporting](./compliance-reporting.md)
- [Incident Forensics](./incident-forensics.md)

## Runbook Standards

### Format
Each runbook follows a standard format:
1. **Overview** - Brief description of the issue/procedure
2. **Prerequisites** - Required access, tools, and knowledge
3. **Symptoms** - How to identify the issue
4. **Investigation Steps** - Diagnostic procedures
5. **Resolution Steps** - Step-by-step fix procedures
6. **Verification** - How to confirm the fix worked
7. **Prevention** - How to prevent recurrence
8. **Escalation** - When and how to escalate

### Severity Levels
- **P0 (Critical)**: System down, data loss, security breach
- **P1 (High)**: Major functionality impaired, performance severely degraded
- **P2 (Medium)**: Minor functionality impaired, workaround available
- **P3 (Low)**: Cosmetic issues, enhancement requests

### Response Times
- **P0**: 15 minutes
- **P1**: 1 hour
- **P2**: 4 hours
- **P3**: Next business day

## Tools and Access

### Required Tools
- `kubectl` - Kubernetes CLI
- `aws` - AWS CLI
- `docker` - Container management
- `helm` - Kubernetes package manager
- `psql` - PostgreSQL client
- `redis-cli` - Redis client

### Access Requirements
- AWS Console access with appropriate IAM roles
- Kubernetes cluster access (RBAC configured)
- Monitoring system access (Grafana, Prometheus)
- Log aggregation system access (ELK stack)
- Source code repository access

### Useful Commands

#### Kubernetes
```bash
# Check pod status
kubectl get pods -n finops-automation

# View pod logs
kubectl logs -f deployment/finops-api -n finops-automation

# Execute into pod
kubectl exec -it deployment/finops-api -n finops-automation -- /bin/bash

# Check resource usage
kubectl top pods -n finops-automation
```

#### Database
```bash
# Connect to database
kubectl exec -it deployment/postgres -n finops-automation -- psql -U finops -d finops_db

# Check database size
kubectl exec -it deployment/postgres -n finops-automation -- psql -U finops -d finops_db -c "SELECT pg_size_pretty(pg_database_size('finops_db'));"
```

#### Monitoring
```bash
# Check Prometheus targets
curl -s http://prometheus.monitoring.svc.cluster.local:9090/api/v1/targets

# Query metrics
curl -s "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=up"
```

## Maintenance Schedule

### Daily
- [ ] Check system health dashboard
- [ ] Review overnight alerts
- [ ] Verify backup completion
- [ ] Monitor cost optimization metrics

### Weekly
- [ ] Review performance trends
- [ ] Update security patches
- [ ] Analyze cost savings reports
- [ ] Review and update documentation

### Monthly
- [ ] Disaster recovery testing
- [ ] Security audit review
- [ ] Capacity planning review
- [ ] Runbook accuracy review

## Contact Information

For questions about these runbooks or to report issues:
- **Documentation Team**: docs@example.com
- **Platform Team**: finops-team@example.com
- **Emergency Hotline**: +1-555-FINOPS-1

## Contributing

To update or add runbooks:
1. Follow the standard format
2. Test all procedures in a non-production environment
3. Get review from at least two team members
4. Update the index and cross-references
5. Notify the team of changes