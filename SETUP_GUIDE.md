# FinOps Platform - Enterprise Setup Guide

This guide will help you set up the FinOps platform for multi-account, multi-team AWS environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Organization Setup](#aws-organization-setup)
3. [IAM Roles Configuration](#iam-roles-configuration)
4. [Account Tagging Strategy](#account-tagging-strategy)
5. [Notification Channels](#notification-channels)
6. [Cost Centers & Teams](#cost-centers--teams)
7. [Budget Configuration](#budget-configuration)
8. [Testing the Setup](#testing-the-setup)

---

## Prerequisites

### Required AWS Permissions
The master/management account needs:
- `organizations:*` - Full Organizations access
- `ce:*` - Cost Explorer access
- `sts:AssumeRole` - Cross-account access
- `iam:*` - IAM management

### Python Dependencies
```bash
pip install -r backend/requirements.txt
```

### Environment Variables
Create a `.env` file:
```bash
# Database
DATABASE_URL=postgresql://finops:finops_password@localhost:5432/finops_db

# AWS Master Account Credentials
AWS_MASTER_ACCESS_KEY_ID=AKIA...
AWS_MASTER_SECRET_ACCESS_KEY=...
AWS_MASTER_REGION=us-east-1

# Email Notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@company.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=finops@company.com

# Slack Notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Microsoft Teams (optional)
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR/WEBHOOK/URL
```

---

## AWS Organization Setup

### Step 1: Enable AWS Organizations

1. Log into your AWS master/management account
2. Navigate to AWS Organizations
3. Click "Create organization"
4. Choose "All features" (not just consolidated billing)

### Step 2: Enable Cost Explorer

```bash
# Enable Cost Explorer in master account
aws ce enable-cost-explorer --region us-east-1
```

### Step 3: Enable Cost Allocation Tags

1. Go to AWS Billing Console → Cost Allocation Tags
2. Activate these tags:
   - `Team`
   - `Project`
   - `Environment`
   - `CostCenter`
   - `Owner`

---

## IAM Roles Configuration

### Step 1: Create Cross-Account Role in Each Member Account

Deploy this CloudFormation template in **each member account**:

```yaml
# Save as: finops-cross-account-role.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'FinOps Cross-Account Access Role'

Parameters:
  MasterAccountId:
    Type: String
    Description: 'Master/Management Account ID'
    Default: '123456789012'  # Replace with your master account ID
  
  ExternalId:
    Type: String
    Description: 'External ID for additional security'
    Default: 'finops-external-id-2024'  # Change this!

Resources:
  FinOpsRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: FinOpsAccessRole
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${MasterAccountId}:root'
            Action: 'sts:AssumeRole'
            Condition:
              StringEquals:
                'sts:ExternalId': !Ref ExternalId
      ManagedPolicyArns:
        - 'arn:aws:iam::aws:policy/ReadOnlyAccess'
      Policies:
        - PolicyName: FinOpsCostExplorerAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - 'ce:*'
                  - 'cur:*'
                  - 'pricing:*'
                  - 'budgets:*'
                  - 'organizations:Describe*'
                  - 'organizations:List*'
                Resource: '*'
      Tags:
        - Key: Purpose
          Value: FinOps
        - Key: ManagedBy
          Value: FinOpsPlatform

Outputs:
  RoleArn:
    Description: 'ARN of the FinOps access role'
    Value: !GetAtt FinOpsRole.Arn
    Export:
      Name: FinOpsRoleArn
```

Deploy using AWS CLI:
```bash
# Deploy to each member account
aws cloudformation create-stack \
  --stack-name finops-access-role \
  --template-body file://finops-cross-account-role.yaml \
  --parameters ParameterKey=MasterAccountId,ParameterValue=123456789012 \
               ParameterKey=ExternalId,ParameterValue=finops-external-id-2024 \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

### Step 2: Configure Platform with Role ARNs

```python
# In your application code or configuration
from backend.core.multi_account_manager import MultiAccountManager, CrossAccountRole

# Initialize manager
manager = MultiAccountManager({
    'access_key_id': os.getenv('AWS_MASTER_ACCESS_KEY_ID'),
    'secret_access_key': os.getenv('AWS_MASTER_SECRET_ACCESS_KEY'),
    'region': 'us-east-1'
})

# Discover all accounts
accounts = await manager.discover_accounts()

# Configure cross-account roles
for account in accounts:
    role = CrossAccountRole(
        role_arn=f"arn:aws:iam::{account.account_id}:role/FinOpsAccessRole",
        external_id="finops-external-id-2024",
        session_name="FinOpsSession"
    )
    await manager.assume_role(account.account_id, role)
```

---

## Account Tagging Strategy

### Recommended Tag Structure

| Tag Key | Purpose | Example Values |
|---------|---------|----------------|
| `Team` | Team ownership | Engineering, DataScience, DevOps |
| `Project` | Project assignment | WebApp, MobileApp, DataPipeline |
| `Environment` | Environment type | Production, Staging, Development |
| `CostCenter` | Cost center code | CC-ENG, CC-DS, CC-OPS |
| `Owner` | Technical owner | john@company.com |
| `Application` | Application name | CustomerPortal, Analytics |

### Apply Tags to Accounts

```python
# Tag accounts programmatically
await manager.tag_account('123456789012', {
    'Team': 'Engineering',
    'CostCenter': 'CC-ENG',
    'Environment': 'Production'
})

# Or use AWS CLI
aws organizations tag-resource \
  --resource-id 123456789012 \
  --tags Key=Team,Value=Engineering Key=CostCenter,Value=CC-ENG
```

### Enforce Tagging Policy

Create an AWS Organizations Service Control Policy (SCP):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RequireTagsOnResources",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances",
        "rds:CreateDBInstance",
        "s3:CreateBucket"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotLike": {
          "aws:RequestTag/Team": "*",
          "aws:RequestTag/Project": "*",
          "aws:RequestTag/Environment": "*"
        }
      }
    }
  ]
}
```

---

## Notification Channels

### Email Configuration

```python
from backend.core.notification_service import (
    get_notification_service, 
    EmailConfig, 
    NotificationMessage,
    NotificationPriority
)

# Get service
notif_service = get_notification_service()

# Register email channel
email_config = EmailConfig(
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    smtp_username='finops@company.com',
    smtp_password='your-app-password',
    from_address='finops@company.com',
    use_tls=True
)
notif_service.register_email_channel('email-eng-team', email_config)

# Send test notification
message = NotificationMessage(
    title='Budget Alert: Engineering Team',
    message='Your team has reached 80% of monthly budget ($40,000 of $50,000)',
    priority=NotificationPriority.HIGH,
    metadata={
        'to_address': 'eng-leads@company.com',
        'team': 'Engineering',
        'budget_used': 40000,
        'budget_total': 50000
    }
)

await notif_service.send_notification(['email-eng-team'], message)
```

### Slack Configuration

```python
from backend.core.notification_service import SlackConfig

# Register Slack channel
slack_config = SlackConfig(
    webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
    channel='#finops-alerts',
    username='FinOps Bot',
    icon_emoji=':moneybag:'
)
notif_service.register_slack_channel('slack-finops', slack_config)

# Send to Slack
await notif_service.send_notification(['slack-finops'], message)
```

### Microsoft Teams Configuration

```python
from backend.core.notification_service import TeamsConfig

# Register Teams channel
teams_config = TeamsConfig(
    webhook_url='https://outlook.office.com/webhook/YOUR/WEBHOOK/URL'
)
notif_service.register_teams_channel('teams-finops', teams_config)
```

---

## Cost Centers & Teams

### Define Organizational Structure

```python
# Define your organization structure
org_structure = {
    'teams': [
        {
            'name': 'Engineering',
            'owner': 'john@company.com',
            'members': ['john@company.com', 'jane@company.com'],
            'budget_amount': 50000  # Monthly budget in USD
        },
        {
            'name': 'Data Science',
            'owner': 'alice@company.com',
            'members': ['alice@company.com', 'bob@company.com'],
            'budget_amount': 30000
        },
        {
            'name': 'DevOps',
            'owner': 'charlie@company.com',
            'members': ['charlie@company.com'],
            'budget_amount': 20000
        }
    ],
    'cost_centers': [
        {
            'code': 'CC-ENG',
            'name': 'Engineering Cost Center',
            'teams': ['Engineering'],
            'owner': 'john@company.com',
            'budget_amount': 50000
        },
        {
            'code': 'CC-DS',
            'name': 'Data Science Cost Center',
            'teams': ['Data Science'],
            'owner': 'alice@company.com',
            'budget_amount': 30000
        },
        {
            'code': 'CC-OPS',
            'name': 'Operations Cost Center',
            'teams': ['DevOps'],
            'owner': 'charlie@company.com',
            'budget_amount': 20000
        }
    ],
    'projects': [
        {
            'name': 'WebApp',
            'team': 'Engineering',
            'cost_center': 'CC-ENG',
            'environment': 'Production'
        },
        {
            'name': 'DataPipeline',
            'team': 'Data Science',
            'cost_center': 'CC-DS',
            'environment': 'Production'
        }
    ],
    'environments': [
        {'name': 'Production', 'cost_allocation': 70},
        {'name': 'Staging', 'cost_allocation': 20},
        {'name': 'Development', 'cost_allocation': 10}
    ]
}
```

### Configure Cost Tracking

```python
from backend.core.post_migration_integration_engine import CostTrackingIntegrator

# Initialize integrator
integrator = CostTrackingIntegrator(db_session)

# Configure cost tracking
result = integrator.configure_cost_tracking(
    project_id='main-migration',
    org_structure=org_structure
)

print(f"Created {len(result['cost_centers_created'])} cost centers")
print(f"Created {len(result['attribution_rules'])} attribution rules")
```

---

## Budget Configuration

### Create Team Budgets

```python
from backend.core.models import Budget, BudgetType
from decimal import Decimal

# Create budget for Engineering team
eng_budget = Budget(
    name='Engineering Monthly Budget',
    amount=Decimal('50000.00'),
    budget_type=BudgetType.MONTHLY,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    alert_threshold=80.0,  # Alert at 80%
    tags={'team': 'Engineering', 'cost_center': 'CC-ENG'},
    created_by=current_user.id
)
db.add(eng_budget)
db.commit()
```

### Configure Budget Alerts

```python
from backend.core.alert_manager import AlertManager, AlertRule

alert_manager = AlertManager()

# Budget threshold alert
budget_alert = AlertRule(
    rule_id='budget-eng-80',
    name='Engineering Budget 80% Alert',
    condition_type='budget_threshold',
    threshold_value=80.0,
    resource_filters={'team': ['Engineering']},
    notification_channels=['email-eng-team', 'slack-finops'],
    cooldown_minutes=60
)

alert_manager.add_alert_rule(budget_alert)

# Cost anomaly alert
anomaly_alert = AlertRule(
    rule_id='anomaly-all-teams',
    name='Cost Anomaly Detection',
    condition_type='anomaly',
    threshold_value=20.0,  # 20% deviation
    notification_channels=['email-finops-team', 'slack-finops'],
    cooldown_minutes=240  # 4 hours
)

alert_manager.add_alert_rule(anomaly_alert)
```

---

## Testing the Setup

### 1. Test AWS Connectivity

```python
# Test master account access
from backend.core.multi_account_manager import create_multi_account_manager

manager = create_multi_account_manager({
    'access_key_id': os.getenv('AWS_MASTER_ACCESS_KEY_ID'),
    'secret_access_key': os.getenv('AWS_MASTER_SECRET_ACCESS_KEY'),
    'region': 'us-east-1'
})

# Discover accounts
accounts = await manager.discover_accounts()
print(f"Discovered {len(accounts)} accounts")

for account in accounts:
    print(f"  - {account.account_name} ({account.account_id})")
    print(f"    Team: {account.team}, Cost Center: {account.cost_center}")
```

### 2. Test Cost Data Retrieval

```python
# Fetch cost data for last 7 days
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

cost_data = await manager.get_all_accounts_cost_data(start_date, end_date)

for account_id, data in cost_data.items():
    total_cost = sum(item['cost'] for item in data)
    print(f"Account {account_id}: ${total_cost:,.2f}")
```

### 3. Test Notifications

```python
# Send test notification
test_message = NotificationMessage(
    title='Test Alert: System Check',
    message='This is a test notification from the FinOps platform.',
    priority=NotificationPriority.LOW,
    metadata={'test': True}
)

results = await notif_service.send_notification(
    ['email-eng-team', 'slack-finops'],
    test_message
)

print(f"Notification results: {results}")
```

### 4. Test Pricing API

```python
from backend.core.aws_pricing_service import get_pricing_service

pricing_service = get_pricing_service()

# Get EC2 pricing
price = await pricing_service.get_ec2_on_demand_pricing(
    instance_type='m5.large',
    region='us-east-1'
)

print(f"m5.large on-demand price: ${price}/hour")

# Get RI pricing
ri_price = await pricing_service.get_ec2_reserved_pricing(
    instance_type='m5.large',
    region='us-east-1',
    term_years=1,
    payment_option='No Upfront'
)

print(f"m5.large 1-year RI: ${ri_price['upfront']} upfront, ${ri_price['hourly']}/hour")
```

---

## Production Checklist

- [ ] AWS Organizations enabled
- [ ] Cost Explorer enabled in master account
- [ ] Cost allocation tags activated
- [ ] Cross-account IAM roles deployed to all member accounts
- [ ] Accounts tagged with Team, CostCenter, Environment
- [ ] Email notification channel configured and tested
- [ ] Slack/Teams webhooks configured (if using)
- [ ] Cost centers defined in platform
- [ ] Team budgets created
- [ ] Budget alerts configured
- [ ] Celery workers running for scheduled tasks
- [ ] Database migrations applied
- [ ] Monitoring dashboards configured

---

## Troubleshooting

### Issue: Cannot discover accounts
**Solution**: Ensure master account has `organizations:ListAccounts` permission

### Issue: Cannot assume cross-account role
**Solution**: 
1. Verify role ARN is correct
2. Check external ID matches
3. Ensure trust policy allows master account

### Issue: No cost data appearing
**Solution**:
1. Verify Cost Explorer is enabled
2. Check that accounts have had activity in the date range
3. Ensure IAM role has `ce:GetCostAndUsage` permission

### Issue: Notifications not sending
**Solution**:
1. Check SMTP credentials for email
2. Verify webhook URLs for Slack/Teams
3. Check notification service logs

---

## Next Steps

1. **Set up automated reports**: Configure daily/weekly cost reports
2. **Enable RI recommendations**: Run RI optimization analysis
3. **Configure waste detection**: Set up idle resource detection
4. **Create custom dashboards**: Build team-specific cost dashboards
5. **Implement chargeback**: Set up automated cost allocation reports

For support, contact: finops-support@company.com
