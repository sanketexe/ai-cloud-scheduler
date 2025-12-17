# Automation Failure Response Runbook

## Overview
This runbook provides procedures for responding to failures in the FinOps automated cost optimization system. It covers detection, investigation, and resolution of automation engine failures, safety check violations, and policy enforcement issues.

## Prerequisites
- Access to Kubernetes cluster and monitoring systems
- Understanding of FinOps cost optimization workflows
- AWS console access for resource investigation
- Familiarity with safety check mechanisms

## Severity Classification

### P0 (Critical) - Immediate Response Required
- Safety checks failing and production resources at risk
- Unauthorized actions being executed
- Mass resource deletion or modification
- Security policy violations

### P1 (High) - Response within 1 hour
- Automation engine completely down
- High rate of action failures (>50%)
- Rollback system failures
- Audit logging failures

### P2 (Medium) - Response within 4 hours
- Individual optimization actions failing
- Policy configuration issues
- Notification system failures
- Performance degradation

## Common Symptoms

### Automation Engine Down
```
Alert: FinOpsAutomationEngineDown
Symptoms:
- No optimization actions being executed
- Engine status metrics showing 0
- Worker processes not responding
- Queue backlog building up
```

### Safety Check Failures
```
Alert: SafetyCheckFailures
Symptoms:
- Production resources being targeted
- Safety validation errors in logs
- Blocked actions due to safety violations
- Emergency stop triggers activated
```

### High Action Failure Rate
```
Alert: HighFailedOptimizationActions
Symptoms:
- Multiple AWS API errors
- Permission denied errors
- Resource state conflicts
- Timeout errors
```

## Investigation Steps

### Step 1: Assess Immediate Risk
```bash
# Check if any critical resources are at risk
kubectl logs -f deployment/finops-api -n finops-automation | grep -i "CRITICAL\|ERROR\|SAFETY"

# Verify safety systems are active
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.safety_checker import SafetyChecker; print(SafetyChecker().system_status())"

# Check for any ongoing dangerous actions
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.action_engine import ActionEngine; print(ActionEngine().get_active_actions())"
```

### Step 2: Check System Health
```bash
# Check pod status
kubectl get pods -n finops-automation

# Check resource usage
kubectl top pods -n finops-automation

# Check recent events
kubectl get events -n finops-automation --sort-by='.lastTimestamp' | tail -20
```

### Step 3: Examine Logs
```bash
# API logs
kubectl logs deployment/finops-api -n finops-automation --tail=100

# Worker logs
kubectl logs deployment/finops-worker -n finops-automation --tail=100

# Scheduler logs
kubectl logs deployment/finops-scheduler -n finops-automation --tail=100

# Filter for errors
kubectl logs deployment/finops-api -n finops-automation | grep -E "(ERROR|CRITICAL|EXCEPTION)" | tail -20
```

### Step 4: Check Database Connectivity
```bash
# Test database connection
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.database import get_db; print('DB OK' if get_db() else 'DB FAIL')"

# Check for database locks
kubectl exec -it deployment/postgres -n finops-automation -- \
  psql -U finops -d finops_db -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

### Step 5: Verify AWS Permissions
```bash
# Test AWS connectivity and permissions
kubectl exec -it deployment/finops-api -n finops-automation -- \
  aws sts get-caller-identity

# Test specific permissions
kubectl exec -it deployment/finops-api -n finops-automation -- \
  aws ec2 describe-instances --max-items 1

# Check IAM role assumptions
kubectl exec -it deployment/finops-api -n finops-automation -- \
  aws sts assume-role --role-arn "$COST_OPTIMIZATION_ROLE_ARN" --role-session-name "test-session"
```

## Resolution Steps

### For P0 (Critical) Issues

#### Immediate Actions
1. **Emergency Stop All Automation**
```bash
# Disable automation immediately
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.auto_remediation_engine import AutoRemediationEngine; AutoRemediationEngine().emergency_stop()"

# Scale down worker pods to prevent new actions
kubectl scale deployment finops-worker --replicas=0 -n finops-automation
```

2. **Assess Damage**
```bash
# Check recent actions in audit logs
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.automation_audit_logger import AuditLogger; AuditLogger().get_recent_actions(hours=1)"

# Identify any resources that need immediate attention
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/emergency-resource-check.py
```

3. **Notify Stakeholders**
```bash
# Send emergency notification
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.notification_service import NotificationService; NotificationService().send_emergency_alert('Automation system emergency stop activated')"
```

### For P1 (High) Issues

#### Automation Engine Recovery
1. **Restart Core Services**
```bash
# Restart API pods
kubectl rollout restart deployment/finops-api -n finops-automation

# Restart worker pods
kubectl rollout restart deployment/finops-worker -n finops-automation

# Restart scheduler
kubectl rollout restart deployment/finops-scheduler -n finops-automation

# Wait for pods to be ready
kubectl wait --for=condition=available deployment/finops-api -n finops-automation --timeout=300s
```

2. **Clear Stuck Tasks**
```bash
# Clear Redis queues if corrupted
kubectl exec -it deployment/redis -n finops-automation -- redis-cli FLUSHDB

# Reset Celery beat schedule
kubectl exec -it deployment/finops-scheduler -n finops-automation -- \
  rm -f /tmp/celerybeat-schedule
```

3. **Verify System Recovery**
```bash
# Check system health
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/health-check.py --exit-code

# Test a safe action
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.auto_remediation_engine import AutoRemediationEngine; AutoRemediationEngine().test_safe_action()"
```

### For P2 (Medium) Issues

#### Individual Action Failures
1. **Identify Failed Actions**
```bash
# Get failed actions from last 24 hours
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.repositories import ActionRepository; ActionRepository().get_failed_actions(hours=24)"
```

2. **Retry Failed Actions**
```bash
# Retry specific failed action
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.auto_remediation_engine import AutoRemediationEngine; AutoRemediationEngine().retry_action('ACTION_ID')"
```

3. **Update Policies if Needed**
```bash
# Check policy configuration
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.policy_manager import PolicyManager; PolicyManager().validate_all_policies()"
```

## Verification Steps

### System Health Verification
```bash
# 1. Check all pods are running
kubectl get pods -n finops-automation

# 2. Verify API health
curl -f https://api.finops.example.com/health/ready

# 3. Check automation status
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.auto_remediation_engine import AutoRemediationEngine; print(AutoRemediationEngine().get_status())"

# 4. Verify safety systems
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.safety_checker import SafetyChecker; print(SafetyChecker().run_full_check())"

# 5. Test notification system
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.notification_service import NotificationService; NotificationService().test_all_channels()"
```

### Functional Verification
```bash
# 1. Execute a test optimization action
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/test-optimization-action.py --dry-run

# 2. Verify audit logging
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "from backend.core.automation_audit_logger import AuditLogger; print(AuditLogger().test_logging())"

# 3. Check metrics collection
curl -s https://monitoring.finops.example.com/api/v1/query?query=finops_automation_engine_status
```

## Prevention Measures

### Monitoring Improvements
1. **Enhanced Health Checks**
   - Add more granular health check endpoints
   - Implement circuit breakers for external dependencies
   - Add predictive failure detection

2. **Better Alerting**
   - Set up alerts for leading indicators
   - Implement alert fatigue reduction
   - Add context-aware alert routing

### System Resilience
1. **Graceful Degradation**
   - Implement fallback mechanisms
   - Add retry logic with exponential backoff
   - Create manual override capabilities

2. **Testing Improvements**
   - Regular chaos engineering exercises
   - Automated integration testing
   - Load testing for peak scenarios

### Process Improvements
1. **Change Management**
   - Implement blue-green deployments
   - Add automated rollback triggers
   - Require approval for high-risk changes

2. **Documentation**
   - Keep runbooks updated
   - Document all configuration changes
   - Maintain incident post-mortems

## Escalation Procedures

### When to Escalate
- Unable to resolve within SLA timeframe
- Issue requires architectural changes
- Security implications identified
- Customer data at risk

### Escalation Contacts
1. **Platform Team Lead**: platform-lead@example.com
2. **Engineering Manager**: eng-manager@example.com
3. **CTO**: cto@example.com
4. **Security Team**: security@example.com

### Escalation Information to Provide
- Incident severity and impact
- Steps already taken
- Current system status
- Estimated time to resolution
- Resources needed

## Post-Incident Actions

### Immediate (Within 24 hours)
- [ ] Document incident timeline
- [ ] Identify root cause
- [ ] Implement temporary fixes
- [ ] Communicate status to stakeholders

### Short-term (Within 1 week)
- [ ] Implement permanent fixes
- [ ] Update monitoring and alerting
- [ ] Review and update runbooks
- [ ] Conduct team retrospective

### Long-term (Within 1 month)
- [ ] Implement prevention measures
- [ ] Update system architecture if needed
- [ ] Provide team training
- [ ] Update disaster recovery procedures

## Related Runbooks
- [Safety Check Failures](./safety-check-failures.md)
- [System Outage Response](./system-outage-response.md)
- [Rollback Procedures](./rollback-procedures.md)
- [Performance Degradation](./performance-degradation.md)