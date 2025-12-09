# FinOps Platform - Quick Start Guide

Get your FinOps platform up and running in 15 minutes!

## Prerequisites

- Docker & Docker Compose installed
- AWS account with Organizations enabled
- Python 3.10+ (for setup script)

---

## Step 1: Clone & Configure (5 minutes)

```bash
# Clone the repository (if not already done)
cd TS_AI_CLOUD_SCHEDULER

# Copy example configuration
cp config.example.json config.json

# Edit configuration
nano config.json
```

### Minimum Required Configuration:

```json
{
  "aws": {
    "master_account": {
      "account_id": "YOUR_MASTER_ACCOUNT_ID",
      "access_key_id": "AKIA...",
      "secret_access_key": "YOUR_SECRET_KEY",
      "region": "us-east-1"
    }
  },
  "notifications": {
    "email": {
      "smtp_host": "smtp.gmail.com",
      "smtp_port": 587,
      "smtp_username": "your-email@company.com",
      "smtp_password": "your-app-password",
      "from_address": "finops@company.com"
    }
  }
}
```

---

## Step 2: Deploy IAM Roles (5 minutes)

Deploy this CloudFormation stack to **each AWS member account**:

```bash
# Save this as finops-role.yaml
cat > finops-role.yaml << 'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Parameters:
  MasterAccountId:
    Type: String
    Default: 'YOUR_MASTER_ACCOUNT_ID'
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
      ManagedPolicyArns:
        - 'arn:aws:iam::aws:policy/ReadOnlyAccess'
      Policies:
        - PolicyName: FinOpsCostAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action: ['ce:*', 'cur:*', 'pricing:*', 'budgets:*']
                Resource: '*'
Outputs:
  RoleArn:
    Value: !GetAtt FinOpsRole.Arn
EOF

# Deploy to each account
aws cloudformation create-stack \
  --stack-name finops-access-role \
  --template-body file://finops-role.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

---

## Step 3: Run Setup Wizard (3 minutes)

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run setup wizard
python scripts/setup_finops.py --config config.json
```

The wizard will:
- ✅ Discover all AWS accounts
- ✅ Configure notification channels
- ✅ Set up alerts
- ✅ Send test notifications

---

## Step 4: Start the Platform (2 minutes)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

---

## Step 5: Access the Platform

### Frontend
```
http://localhost:3000
```

### API Documentation
```
http://localhost:8000/docs
```

### Monitoring
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

---

## Verify Everything Works

### 1. Check Cost Data Sync

```bash
# View worker logs
docker-compose logs -f worker

# Should see:
# "Cost data sync completed successfully"
```

### 2. Test Notifications

```python
# Run this in Python
import asyncio
from backend.core.notification_service import *

async def test():
    service = get_notification_service()
    
    # Register email
    service.register_email_channel('test', EmailConfig(
        smtp_host='smtp.gmail.com',
        smtp_port=587,
        smtp_username='your-email@company.com',
        smtp_password='your-password',
        from_address='finops@company.com'
    ))
    
    # Send test
    msg = NotificationMessage(
        title='Test Alert',
        message='Platform is working!',
        priority=NotificationPriority.LOW
    )
    
    result = await service.send_notification(['test'], msg)
    print(f"Result: {result}")

asyncio.run(test())
```

### 3. Check Account Discovery

```python
# Run this in Python
import asyncio
from backend.core.multi_account_manager import *

async def test():
    manager = create_multi_account_manager({
        'access_key_id': 'YOUR_KEY',
        'secret_access_key': 'YOUR_SECRET',
        'region': 'us-east-1'
    })
    
    accounts = await manager.discover_accounts()
    print(f"Found {len(accounts)} accounts:")
    for acc in accounts:
        print(f"  - {acc.account_name} ({acc.account_id})")

asyncio.run(test())
```

---

## Common Issues & Solutions

### Issue: "Cannot discover accounts"
**Solution**: Ensure master account has `organizations:ListAccounts` permission

### Issue: "Cost data not appearing"
**Solution**: 
1. Enable Cost Explorer in AWS Console
2. Wait 24 hours for initial data
3. Check Celery worker logs

### Issue: "Notifications not sending"
**Solution**:
1. For Gmail: Use App Password, not regular password
2. For Slack: Verify webhook URL is correct
3. Check notification service logs

### Issue: "Cannot assume cross-account role"
**Solution**:
1. Verify role ARN format: `arn:aws:iam::ACCOUNT_ID:role/FinOpsAccessRole`
2. Check trust policy allows master account
3. Ensure role is deployed in target account

---

## Default Scheduled Tasks

| Task | Frequency | Description |
|------|-----------|-------------|
| Cost Data Sync | Every 6 hours | Fetches cost data from AWS |
| Budget Monitoring | Every hour | Checks budget thresholds |
| Anomaly Detection | Every 4 hours | Detects cost anomalies |
| RI Recommendations | Weekly (Monday 2 AM) | Generates RI recommendations |
| Waste Detection | Daily (3 AM) | Identifies idle resources |

---

## Quick Commands

```bash
# View all logs
docker-compose logs -f

# Restart a service
docker-compose restart api

# Stop everything
docker-compose down

# Start fresh (WARNING: Deletes data)
docker-compose down -v
docker-compose up -d

# Run database migrations
docker-compose exec api alembic upgrade head

# Access database
docker-compose exec postgres psql -U finops -d finops_db

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL
```

---

## Next Steps

1. **Configure Teams**: Edit `config.json` to add your teams
2. **Set Budgets**: Create monthly budgets for each team
3. **Enable Tagging**: Enforce tagging policy on AWS resources
4. **Create Dashboards**: Build custom cost dashboards
5. **Train Users**: Share access with team leads

---

## Getting Help

- **Documentation**: See `SETUP_GUIDE.md` for detailed instructions
- **API Docs**: http://localhost:8000/docs
- **Logs**: `docker-compose logs -f [service]`
- **Issues**: Check GitHub issues or create a new one

---

## Success Checklist

- [ ] Docker containers running
- [ ] AWS accounts discovered
- [ ] Cost data syncing
- [ ] Notifications working
- [ ] Alerts configured
- [ ] Frontend accessible
- [ ] API responding
- [ ] Celery tasks running

---

**Congratulations! Your FinOps platform is ready! 🎉**

Access it at: http://localhost:3000
