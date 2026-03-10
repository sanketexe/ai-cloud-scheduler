# AWS Setup Guide for FinOps Platform

This guide walks you through setting up AWS integration for the FinOps Platform, from creating an IAM user to verifying the connection.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Create IAM User](#step-1-create-iam-user)
3. [Step 2: Attach IAM Policy](#step-2-attach-iam-policy)
4. [Step 3: Generate Access Keys](#step-3-generate-access-keys)
5. [Step 4: Configure Platform](#step-4-configure-platform)
6. [Step 5: Enable Cost Explorer](#step-5-enable-cost-explorer)
7. [Step 6: Verify Connection](#step-6-verify-connection)
8. [Optional: Advanced Configuration](#optional-advanced-configuration)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- AWS account with administrative access
- AWS Console access
- Platform installed locally (see main README.md)
- Python 3.10+ and pip installed

---

## Step 1: Create IAM User

### Option A: Using AWS Console (Recommended for Beginners)

1. Log into AWS Console: https://console.aws.amazon.com
2. Navigate to **IAM** service
3. Click **Users** in the left sidebar
4. Click **Add users** button
5. Configure user:
   - **User name**: `finops-platform`
   - **Access type**: Select **Programmatic access**
   - Click **Next: Permissions**

### Option B: Using AWS CLI

```bash
aws iam create-user --user-name finops-platform
```

---

## Step 2: Attach IAM Policy

### Create Custom Policy

1. In IAM Console, click **Policies** in left sidebar
2. Click **Create policy**
3. Click **JSON** tab
4. Paste the following policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CostExplorerFullAccess",
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "ce:GetDimensionValues",
        "ce:GetTags",
        "ce:GetReservationUtilization",
        "ce:GetSavingsPlansUtilization"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EC2ReadOnlyAccess",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeVolumes",
        "ec2:DescribeSnapshots",
        "ec2:DescribeRegions",
        "ec2:DescribeAvailabilityZones",
        "ec2:DescribeTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchReadAccess",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:GetMetricData",
        "cloudwatch:ListMetrics",
        "cloudwatch:DescribeAlarms"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BudgetsReadAccess",
      "Effect": "Allow",
      "Action": [
        "budgets:ViewBudget",
        "budgets:DescribeBudgets",
        "budgets:DescribeBudgetPerformanceHistory"
      ],
      "Resource": "*"
    },
    {
      "Sid": "OrganizationsReadAccess",
      "Effect": "Allow",
      "Action": [
        "organizations:DescribeOrganization",
        "organizations:ListAccounts",
        "organizations:DescribeAccount"
      ],
      "Resource": "*"
    },
    {
      "Sid": "STSAccess",
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PricingAccess",
      "Effect": "Allow",
      "Action": [
        "pricing:GetProducts",
        "pricing:DescribeServices"
      ],
      "Resource": "*"
    }
  ]
}
```

5. Click **Next: Tags** (optional)
6. Click **Next: Review**
7. **Name**: `FinOpsPlatformPolicy`
8. **Description**: `Policy for FinOps Platform cost analysis and optimization`
9. Click **Create policy**

### Attach Policy to User

1. Go back to **Users** → **finops-platform**
2. Click **Add permissions** → **Attach policies directly**
3. Search for `FinOpsPlatformPolicy`
4. Check the box next to it
5. Click **Next: Review** → **Add permissions**

### Using AWS CLI

```bash
# Create policy
aws iam create-policy \
  --policy-name FinOpsPlatformPolicy \
  --policy-document file://finops-policy.json

# Attach to user
aws iam attach-user-policy \
  --user-name finops-platform \
  --policy-arn arn:aws:iam::YOUR_ACCOUNT_ID:policy/FinOpsPlatformPolicy
```

---

## Step 3: Generate Access Keys

### Using AWS Console

1. In IAM Console, go to **Users** → **finops-platform**
2. Click **Security credentials** tab
3. Scroll to **Access keys** section
4. Click **Create access key**
5. Select **Application running outside AWS**
6. Click **Next**
7. (Optional) Add description: "FinOps Platform Integration"
8. Click **Create access key**
9. **IMPORTANT**: Copy both:
   - **Access key ID** (starts with `AKIA...`)
   - **Secret access key** (shown only once!)
10. Click **Download .csv file** for backup
11. Click **Done**

### Using AWS CLI

```bash
aws iam create-access-key --user-name finops-platform
```

**Security Best Practice**: Store these credentials securely. Never commit them to git or share them publicly.

---

## Step 4: Configure Platform

### Create .env File

In the root directory of the platform, create a `.env` file:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# Application Configuration
DEMO_MODE=false
LOG_LEVEL=INFO

# Optional: Database (if using PostgreSQL)
# DATABASE_URL=postgresql://user:password@localhost:5432/finops_db

# Optional: Redis Cache
# REDIS_URL=redis://localhost:6379/0

# Optional: Notifications
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your-email@example.com
# SMTP_PASSWORD=your-app-password
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Replace**:
- `AWS_ACCESS_KEY_ID`: Your actual access key ID
- `AWS_SECRET_ACCESS_KEY`: Your actual secret access key
- `AWS_REGION`: Your preferred AWS region (e.g., `us-east-1`, `eu-west-1`)
- `AWS_ACCOUNT_ID`: Your 12-digit AWS account ID

### Find Your AWS Account ID

**Method 1: AWS Console**
- Click your username in top-right corner
- Account ID shown in dropdown

**Method 2: AWS CLI**
```bash
aws sts get-caller-identity --query Account --output text
```

**Method 3: Platform will auto-detect**
- Leave blank, platform will fetch it automatically

---

## Step 5: Enable Cost Explorer

Cost Explorer must be enabled to retrieve cost data. It's free but has a 24-hour activation period.

### Using AWS Console

1. Go to **AWS Billing Console**: https://console.aws.amazon.com/billing
2. Click **Cost Explorer** in left sidebar
3. If not enabled, click **Enable Cost Explorer**
4. Wait 24 hours for data to populate

### Using AWS CLI

```bash
aws ce enable-cost-explorer
```

### Important Notes

- **24-Hour Delay**: Cost Explorer data has a 24-hour delay. Today's costs won't appear until tomorrow.
- **First Activation**: After enabling, wait 24 hours before data appears.
- **No Cost**: Cost Explorer is free for AWS accounts.

---

## Step 6: Verify Connection

### Run Integration Test

The platform includes a test script to verify AWS connectivity:

```bash
python test_aws_integration.py
```

**Expected Output**:

```
=== AWS Integration Test Report ===
Generated: 2024-03-08 14:30:00

1. AWS Credentials Configuration
   Status: ✓ PASS
   Details: Credentials found in environment variables

2. AWS Account Identity Verification
   Status: ✓ PASS
   Account ID: 123456789012
   User ARN: arn:aws:iam::123456789012:user/finops-platform

3. Cost Explorer API Access
   Status: ✓ PASS
   Retrieved cost data for last 7 days
   Total cost: $123.45

4. EC2 API Access
   Status: ✓ PASS
   Found 5 EC2 instances across 2 regions

5. CloudWatch API Access
   Status: ✓ PASS
   Retrieved metrics for instance i-1234567890abcdef0

=== Summary ===
Total Tests: 5
Passed: 5
Failed: 0

✓ AWS integration is working correctly!
```

### Troubleshooting Test Failures

**If Test 1 Fails** (Credentials):
- Verify `.env` file exists in root directory
- Check no extra spaces in credentials
- Ensure file is named exactly `.env` (not `.env.txt`)

**If Test 2 Fails** (Identity):
- Verify access key ID and secret are correct
- Check credentials haven't been deleted in AWS Console
- Try creating new access keys

**If Test 3 Fails** (Cost Explorer):
- Verify Cost Explorer is enabled (see Step 5)
- Wait 24 hours after enabling
- Check IAM policy includes `ce:GetCostAndUsage`

**If Test 4 Fails** (EC2):
- Check IAM policy includes `ec2:DescribeInstances`
- Verify you have EC2 instances in your account
- Try a different region

**If Test 5 Fails** (CloudWatch):
- Check IAM policy includes `cloudwatch:GetMetricStatistics`
- Verify EC2 instances exist to query metrics for

### Start the Platform

If all tests pass, start the platform:

```bash
# Terminal 1: Backend
python start_backend.py

# Terminal 2: Frontend
cd frontend
npm install
npm start
```

Open http://localhost:3000 and verify:
- Dashboard loads without errors
- Cost data displays (if you have recent AWS usage)
- Account ID shows correctly in UI

---

## Optional: Advanced Configuration

### Multi-Account Setup

For organizations with multiple AWS accounts:

1. **Enable AWS Organizations** in master account
2. **Create cross-account IAM roles** in each member account
3. **Configure platform** with role ARNs

See `docs/SETUP_GUIDE.md` for detailed multi-account setup.

### Automation Features

To enable automated cost optimization actions:

1. **Add automation permissions** to IAM policy:

```json
{
  "Sid": "EC2AutomationAccess",
  "Effect": "Allow",
  "Action": [
    "ec2:StopInstances",
    "ec2:StartInstances",
    "ec2:ModifyInstanceAttribute"
  ],
  "Resource": "*",
  "Condition": {
    "StringEquals": {
      "ec2:ResourceTag/Environment": ["Development", "Staging"]
    }
  }
}
```

2. **Configure safety tags** on production resources:
   - Add `Environment=Production` tag to protect resources
   - Automation will skip resources with this tag

### Budget Creation

To create AWS Budgets via the platform:

1. **Add budget permissions** to IAM policy:

```json
{
  "Sid": "BudgetsWriteAccess",
  "Effect": "Allow",
  "Action": [
    "budgets:CreateBudget",
    "budgets:ModifyBudget",
    "budgets:DeleteBudget"
  ],
  "Resource": "*"
}
```

### Notification Setup

Configure email/Slack notifications for alerts:

1. **Email (SMTP)**:
```bash
# Add to .env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=finops@example.com
```

2. **Slack**:
```bash
# Add to .env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

3. **Microsoft Teams**:
```bash
# Add to .env
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR/WEBHOOK/URL
```

---

## Troubleshooting

### "No cost data available"

**Cause**: Cost Explorer not enabled or no recent usage

**Solutions**:
1. Enable Cost Explorer (see Step 5)
2. Wait 24 hours after enabling
3. Ensure you have AWS usage in the selected date range
4. Check IAM permissions include `ce:GetCostAndUsage`

### "AccessDeniedException" errors

**Cause**: IAM policy missing required permissions

**Solutions**:
1. Review IAM policy in Step 2
2. Ensure all required actions are included
3. Wait 5-10 minutes for IAM changes to propagate
4. Try creating new access keys

### "InvalidClientTokenId" error

**Cause**: Access key ID is incorrect or deleted

**Solutions**:
1. Verify access key ID in `.env` matches AWS Console
2. Check access key hasn't been deleted
3. Create new access keys if needed

### "SignatureDoesNotMatch" error

**Cause**: Secret access key is incorrect

**Solutions**:
1. Verify secret access key in `.env` is correct
2. Check for extra spaces or newlines
3. Create new access keys if needed

### Platform shows $0.00 costs

**Cause**: No AWS usage or Cost Explorer delay

**Solutions**:
1. Verify you have actual AWS usage (running EC2, S3 storage, etc.)
2. Check date range - Cost Explorer has 24-hour delay
3. Try querying older dates (7-30 days ago)
4. Run `python test_aws_integration.py` to verify data retrieval

### Slow dashboard loading

**Cause**: AWS API calls are slow or not cached

**Solutions**:
1. Enable Redis caching (see main README.md)
2. Reduce date range for queries
3. Check AWS service health: https://status.aws.amazon.com
4. Consider using CloudFront CDN for frontend

---

## Security Best Practices

### Credential Management

- **Never commit** `.env` file to git (already in `.gitignore`)
- **Rotate access keys** every 90 days
- **Use IAM roles** on EC2 instead of access keys when possible
- **Enable MFA** on IAM user for console access
- **Monitor CloudTrail** for unauthorized API calls

### Least Privilege

- **Start with read-only** permissions
- **Add automation permissions** only when needed
- **Use conditions** to restrict actions (e.g., tag-based)
- **Separate roles** for different functions

### Monitoring

- **Enable CloudTrail** to log all API calls
- **Set up CloudWatch alarms** for unusual activity
- **Review IAM Access Advisor** to remove unused permissions
- **Use AWS Config** to track configuration changes

---

## Next Steps

After successful setup:

1. **Explore the Dashboard**: Familiarize yourself with cost visualizations
2. **Create Budgets**: Set up budget tracking for teams/projects
3. **Review Recommendations**: Check optimization suggestions
4. **Configure Alerts**: Set up notifications for cost anomalies
5. **Enable Automation**: Configure automated cost-saving actions (optional)

For more information:
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Troubleshooting**: See main `README.md`
- **API Documentation**: http://localhost:8000/docs

---

## Support

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting section](#troubleshooting)
2. Run `python test_aws_integration.py` for diagnostics
3. Review backend logs for specific errors
4. Check AWS CloudTrail for API call failures
5. Open an issue on GitHub with test results and error logs

---

**Congratulations!** Your AWS FinOps Platform is now configured and ready to use. 🎉
