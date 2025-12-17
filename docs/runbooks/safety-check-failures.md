# Safety Check Failures Runbook

## Overview
This runbook addresses failures in the FinOps cost optimization safety check system. Safety checks are critical for preventing automation from impacting production resources or violating business rules.

## Prerequisites
- Understanding of FinOps safety check mechanisms
- Access to Kubernetes cluster and AWS console
- Knowledge of production resource tagging standards
- Familiarity with business hours and maintenance windows

## Severity Classification

### P0 (Critical) - Immediate Response Required
- Production resources being targeted for optimization
- Safety system completely bypassed
- Mass safety check failures (>10 per minute)
- Unauthorized resource modifications detected

### P1 (High) - Response within 1 hour
- Safety check system partially down
- High rate of safety violations (>5 per minute)
- Business hours protection not working
- Critical resource protection failing

### P2 (Medium) - Response within 4 hours
- Individual safety check failures
- Tag-based protection issues
- Policy configuration problems
- False positive safety alerts

## Common Safety Check Failure Scenarios

### Production Resource Protection Failure
```
Alert: ProductionResourcesAtRisk
Symptoms:
- Resources with "Environment=production" being targeted
- Critical resources in optimization queue
- Safety tags being ignored
- Production workloads affected
```

### Business Hours Protection Failure
```
Alert: BusinessHoursViolation
Symptoms:
- Actions executing during business hours
- Maintenance window logic not working
- Timezone configuration issues
- Emergency override misuse
```

### Tag-Based Safety Failure
```
Alert: TagBasedSafetyFailure
Symptoms:
- Resources with safety tags being processed
- Tag validation not working
- Inconsistent tag enforcement
- Missing required tags
```

## Investigation Steps

### Step 1: Immediate Safety Assessment
```bash
# Check current safety system status
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
print('Safety System Status:', checker.get_system_status())
print('Active Safety Rules:', checker.get_active_rules())
print('Recent Violations:', checker.get_recent_violations())
"

# Check for any active dangerous actions
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.action_engine import ActionEngine
engine = ActionEngine()
active_actions = engine.get_active_actions()
for action in active_actions:
    if action.risk_level == 'high':
        print(f'HIGH RISK ACTION ACTIVE: {action.action_id} - {action.resource_id}')
"
```

### Step 2: Analyze Safety Check Logs
```bash
# Get recent safety check failures
kubectl logs deployment/finops-api -n finops-automation | grep -i "safety.*fail" | tail -20

# Check safety checker specific logs
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
failures = checker.get_failure_log(hours=1)
for failure in failures:
    print(f'FAILURE: {failure.timestamp} - {failure.resource_id} - {failure.reason}')
"

# Look for safety system errors
kubectl logs deployment/finops-api -n finops-automation | grep -E "(SafetyChecker|safety_check)" | tail -50
```

### Step 3: Verify Resource Protection
```bash
# Check production resource tagging
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
production_resources = checker.get_production_resources()
print(f'Found {len(production_resources)} production resources')
for resource in production_resources[:10]:  # Show first 10
    print(f'  {resource.resource_id}: {resource.tags}')
"

# Verify tag validation logic
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
test_resource = {'ResourceId': 'i-test123', 'Tags': [{'Key': 'Environment', 'Value': 'production'}]}
result = checker.check_production_tags(test_resource)
print(f'Production tag check result: {result}')
"
```

### Step 4: Check Business Hours Logic
```bash
# Verify current time and business hours configuration
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
from datetime import datetime
checker = SafetyChecker()
now = datetime.utcnow()
is_business_hours = checker.is_business_hours(now)
print(f'Current UTC time: {now}')
print(f'Is business hours: {is_business_hours}')
print(f'Business hours config: {checker.get_business_hours_config()}')
"

# Check timezone handling
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
import os
from datetime import datetime
import pytz
print(f'System timezone: {os.environ.get(\"TZ\", \"Not set\")}')
print(f'UTC time: {datetime.utcnow()}')
print(f'Local time: {datetime.now()}')
"
```

### Step 5: Validate Safety Rules Configuration
```bash
# Check safety rules configuration
kubectl get configmap finops-config -n finops-automation -o yaml | grep -A 20 -B 5 -i safety

# Verify safety rules in database
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.policy_manager import PolicyManager
manager = PolicyManager()
safety_policies = manager.get_safety_policies()
print(f'Active safety policies: {len(safety_policies)}')
for policy in safety_policies:
    print(f'  {policy.name}: {policy.enabled}')
"
```

## Resolution Steps

### For P0 (Critical) Issues

#### Immediate Emergency Actions
1. **Emergency Stop All Automation**
```bash
# Immediately halt all automation
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.auto_remediation_engine import AutoRemediationEngine
engine = AutoRemediationEngine()
engine.emergency_stop('Safety system failure - manual intervention required')
print('EMERGENCY STOP ACTIVATED')
"

# Scale down workers to prevent new actions
kubectl scale deployment finops-worker --replicas=0 -n finops-automation
```

2. **Assess and Protect Critical Resources**
```bash
# Get list of all resources currently being processed
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.action_engine import ActionEngine
engine = ActionEngine()
active_actions = engine.get_all_active_actions()
print('ACTIVE ACTIONS REQUIRING REVIEW:')
for action in active_actions:
    print(f'  {action.action_id}: {action.resource_id} - {action.action_type} - Risk: {action.risk_level}')
"

# Check AWS resources for any recent modifications
aws ec2 describe-instances --filters "Name=tag:FinOpsManaged,Values=true" --query 'Reservations[*].Instances[?LaunchTime>=`2024-01-01`].[InstanceId,State.Name,Tags[?Key==`Environment`].Value|[0]]' --output table
```

3. **Immediate Notification**
```bash
# Send critical alert to all stakeholders
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.notification_service import NotificationService
service = NotificationService()
service.send_critical_alert(
    'CRITICAL: Safety system failure detected. All automation halted.',
    channels=['email', 'slack', 'pagerduty']
)
"
```

### For P1 (High) Issues

#### Safety System Recovery
1. **Restart Safety Components**
```bash
# Restart API pods to reload safety configurations
kubectl rollout restart deployment/finops-api -n finops-automation

# Wait for pods to be ready
kubectl wait --for=condition=available deployment/finops-api -n finops-automation --timeout=300s

# Verify safety system is operational
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
status = checker.run_full_system_check()
print(f'Safety system status: {status}')
"
```

2. **Reload Safety Configuration**
```bash
# Reload configuration from ConfigMap
kubectl rollout restart deployment/finops-api -n finops-automation

# Verify configuration is loaded correctly
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
config = checker.get_current_config()
print('Safety Configuration:')
print(f'  Business Hours Protection: {config.get(\"business_hours_protection\")}')
print(f'  Production Tag Protection: {config.get(\"production_tag_protection\")}')
print(f'  Safety Check Timeout: {config.get(\"safety_check_timeout\")}')
"
```

3. **Test Safety System**
```bash
# Run comprehensive safety system test
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/test-safety-system.py --comprehensive

# Test specific safety scenarios
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()

# Test production resource protection
test_prod_resource = {
    'ResourceId': 'i-test-prod',
    'Tags': [{'Key': 'Environment', 'Value': 'production'}]
}
result = checker.check_production_tags(test_prod_resource)
print(f'Production resource test: {\"PASS\" if not result.safe_to_modify else \"FAIL\"}')

# Test business hours protection
result = checker.verify_business_hours()
print(f'Business hours test: {\"PASS\" if result.compliant else \"FAIL\"}')
"
```

### For P2 (Medium) Issues

#### Configuration Fixes
1. **Update Safety Rules**
```bash
# Update safety configuration if needed
kubectl patch configmap finops-config -n finops-automation --patch '
data:
  SAFETY_CHECKS_ENABLED: "true"
  BUSINESS_HOURS_PROTECTION: "true"
  PRODUCTION_TAG_PROTECTION: "true"
  SAFETY_CHECK_TIMEOUT: "30"
'

# Restart pods to pick up new configuration
kubectl rollout restart deployment/finops-api -n finops-automation
```

2. **Fix Tag-Based Protection**
```bash
# Update tag validation rules
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.policy_manager import PolicyManager
manager = PolicyManager()

# Add or update safety tags
safety_tags = [
    {'key': 'Environment', 'values': ['production', 'prod']},
    {'key': 'Critical', 'values': ['true', 'yes']},
    {'key': 'DoNotModify', 'values': ['true', 'yes']},
    {'key': 'FinOpsExclude', 'values': ['true', 'yes']}
]

for tag in safety_tags:
    manager.add_safety_tag_rule(tag['key'], tag['values'])
    print(f'Added safety rule for tag: {tag[\"key\"]}')
"
```

3. **Update Business Hours Configuration**
```bash
# Update business hours configuration
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.policy_manager import PolicyManager
manager = PolicyManager()

# Update business hours (example: 9 AM to 5 PM EST, Monday-Friday)
business_hours_config = {
    'timezone': 'America/New_York',
    'start_hour': 9,
    'end_hour': 17,
    'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'maintenance_windows': [
        {'start': '02:00', 'end': '04:00', 'days': ['sunday']},
        {'start': '01:00', 'end': '03:00', 'days': ['saturday']}
    ]
}

manager.update_business_hours_config(business_hours_config)
print('Business hours configuration updated')
"
```

## Verification Steps

### Safety System Verification
```bash
# 1. Verify safety system is active
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
print('Safety System Active:', checker.is_active())
print('Last Health Check:', checker.get_last_health_check())
"

# 2. Test production resource protection
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
checker = SafetyChecker()
test_cases = [
    {'ResourceId': 'i-prod-test', 'Tags': [{'Key': 'Environment', 'Value': 'production'}]},
    {'ResourceId': 'i-dev-test', 'Tags': [{'Key': 'Environment', 'Value': 'development'}]},
    {'ResourceId': 'i-critical-test', 'Tags': [{'Key': 'Critical', 'Value': 'true'}]}
]
for test in test_cases:
    result = checker.check_production_tags(test)
    expected = 'BLOCKED' if any(tag['Value'] in ['production', 'true'] for tag in test['Tags']) else 'ALLOWED'
    actual = 'BLOCKED' if not result.safe_to_modify else 'ALLOWED'
    status = 'PASS' if expected == actual else 'FAIL'
    print(f'{test[\"ResourceId\"]}: {status} (Expected: {expected}, Actual: {actual})')
"

# 3. Test business hours protection
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.safety_checker import SafetyChecker
from datetime import datetime, timedelta
checker = SafetyChecker()

# Test current time
current_result = checker.verify_business_hours()
print(f'Current time check: {\"PASS\" if current_result.compliant else \"FAIL\"} - {current_result.reason}')

# Test known business hours (simulate)
test_times = [
    datetime(2024, 1, 15, 14, 0),  # Monday 2 PM (should be blocked)
    datetime(2024, 1, 15, 2, 0),   # Monday 2 AM (should be allowed)
    datetime(2024, 1, 13, 14, 0),  # Saturday 2 PM (should be allowed)
]

for test_time in test_times:
    result = checker.is_business_hours(test_time)
    print(f'{test_time}: {\"BLOCKED\" if result else \"ALLOWED\"}')
"

# 4. Verify audit logging of safety checks
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python -c "
from backend.core.automation_audit_logger import AuditLogger
logger = AuditLogger()
recent_safety_events = logger.get_safety_check_events(hours=1)
print(f'Recent safety check events: {len(recent_safety_events)}')
for event in recent_safety_events[-5:]:  # Show last 5
    print(f'  {event.timestamp}: {event.event_type} - {event.resource_id}')
"
```

### End-to-End Safety Test
```bash
# Run comprehensive safety test
kubectl exec -it deployment/finops-api -n finops-automation -- \
  python scripts/comprehensive-safety-test.py

# Expected output should show all safety mechanisms working
```

## Prevention Measures

### Enhanced Safety Monitoring
1. **Proactive Safety Alerts**
```bash
# Add alerts for safety system health
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: finops-safety-alerts
  namespace: monitoring
spec:
  groups:
  - name: finops.safety
    rules:
    - alert: SafetySystemDown
      expr: finops_safety_system_status != 1
      for: 30s
      labels:
        severity: critical
      annotations:
        summary: "Safety system is down"
        description: "The FinOps safety system is not operational"
    
    - alert: SafetyCheckLatency
      expr: finops_safety_check_duration_seconds > 10
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "Safety checks are slow"
        description: "Safety checks taking longer than 10 seconds"
EOF
```

2. **Safety Metrics Dashboard**
   - Create Grafana dashboard for safety metrics
   - Monitor safety check success rates
   - Track safety rule effectiveness

### System Improvements
1. **Redundant Safety Checks**
   - Implement multiple independent safety validators
   - Add cross-validation between safety systems
   - Create safety check circuit breakers

2. **Better Configuration Management**
   - Version control safety configurations
   - Implement configuration validation
   - Add configuration rollback capabilities

### Process Improvements
1. **Safety Testing**
   - Regular safety system testing
   - Chaos engineering for safety systems
   - Automated safety regression testing

2. **Documentation and Training**
   - Keep safety documentation updated
   - Train team on safety procedures
   - Regular safety system reviews

## Escalation Procedures

### When to Escalate
- Safety system cannot be restored within 1 hour
- Production resources have been impacted
- Security implications identified
- Pattern of recurring safety failures

### Escalation Contacts
1. **Security Team**: security@example.com (for production impacts)
2. **Platform Team Lead**: platform-lead@example.com
3. **Engineering Manager**: eng-manager@example.com
4. **CTO**: cto@example.com (for critical business impact)

## Related Runbooks
- [Automation Failure Response](./automation-failure-response.md)
- [Security Incident Response](./security-incident-response.md)
- [Policy Configuration Issues](./policy-configuration-issues.md)
- [System Outage Response](./system-outage-response.md)