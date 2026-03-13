# ☁️ AWS FinOps Platform — Cost Intelligence & Cloud Migration Advisor

**An intelligent platform for AWS cost optimization and cloud migration planning with support for 5 major cloud providers (AWS, Azure, GCP, IBM Cloud, Oracle Cloud).**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB?logo=react)](frontend/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](backend/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](requirements.txt)

---

## 📌 What is this?

A comprehensive platform that combines:

### 1. AWS FinOps & Cost Intelligence
- **Real-Time Cost Analysis** — Live AWS cost monitoring with Cost Explorer API
- **AI-Powered Optimization** — ML-based cost forecasting and anomaly detection
- **Budget Management** — Automated budget tracking with alerts
- **Resource Optimization** — EC2 rightsizing and idle resource detection
- **Automation Engine** — Policy-based auto-remediation

### 2. Cloud Migration Advisor
- **5 Cloud Providers** — AWS, Azure, GCP, IBM Cloud, Oracle Cloud
- **Intelligent Scoring** — Weighted algorithm across 12 dimensions
- **Real-Time Preview** — Live scoring as you complete assessment
- **Evidence-Based** — Transparent recommendations with detailed breakdowns
- **Complexity Assessment** — Automatic migration timeline estimation

---

## 🏗️ Architecture

```
Frontend (React 18) → Backend (FastAPI) → AWS APIs / ML Models
     ↓                      ↓
  Dashboard          Cost Analysis
  Migration Wizard   Optimization
  Budget Mgmt        Automation
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, TypeScript, Material-UI |
| **Backend** | Python 3.10+, FastAPI |
| **ML/AI** | Scikit-learn, Prophet, PyTorch |
| **Database** | SQLite (dev), PostgreSQL (prod) |

---

## 📂 Project Structure

```
TS_AI_CLOUD_SCHEDULER/
├── frontend/                    # React application
│   ├── src/
│   │   ├── pages/              # Dashboard, MigrationWizard, etc.
│   │   ├── components/         # Reusable UI components
│   │   └── services/           # API clients
│   └── package.json
│
├── backend/                     # FastAPI backend
│   ├── main.py                 # App entry point
│   ├── api/                    # REST API endpoints
│   ├── core/                   # Business logic
│   │   ├── migration_advisor/  # Migration advisor engine
│   │   ├── finops_engine.py   # Cost analysis
│   │   └── ...
│   ├── ml/                     # ML models
│   └── requirements.txt
│
├── .env                        # Environment variables
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 16+ and **npm**
- **Python** 3.10+
- **AWS Account** with appropriate IAM permissions (see [IAM Setup](#iam-permissions-required))
- **pip** (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/sanketexe/ai-cloud-scheduler.git
cd ai-cloud-scheduler
```

### 2. Configure AWS Credentials

Create a `.env` file in the root directory:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your_account_id

# Optional: Demo Mode (uses sample data)
DEMO_MODE=false
```

**Important**: See [IAM Permissions Required](#iam-permissions-required) for the minimum IAM policy needed.

### 3. Install Dependencies

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 4. Start the Application

**Terminal 1 — Backend API:**
```bash
python start_py
```
> Backend runs at **http://localhost:8000**
> API docs available at **http://localhost:8000/docs**

**Terminal 2 — Frontend:**
```bash
cd frontend
npm start
```
> Frontend runs at **http://localhost:3000**

### 5. Access the Application

- **Dashboard**: http://localhost:3000
- **Migration Wizard**: http://localhost:3000/migration-wizard
- **API Docs**: http://localhost:8000/docs

**Note**: Cost Explorer data has a 24-hour delay.

---

## 📋 Key Features

### AWS FinOps
- Real-time cost analysis with Cost Explorer API
- ML-powered anomaly detection
- Budget management with alerts
- EC2 rightsizing recommendations
- Automated cost optimization

### Cloud Migration Advisor
- **5 Providers**: AWS ☁️, Azure 🔷, GCP 🌐, IBM 🔷, Oracle 🔴
- **Intelligent Scoring**: 12 weighted dimensions
- **Hard Eliminators**: FedRAMP, HIPAA, budget, data residency
- **Real-Time Preview**: Live scoring as you answer
- **Evidence-Based**: Transparent recommendation breakdown
- **Complexity Assessment**: Automatic timeline estimation

---

## 🧪 API Endpoints

### FinOps
- `GET /api/cost-analysis` - AWS cost breakdown
- `GET /api/dashboard` - Dashboard metrics
- `GET /api/budgets` - Budget tracking
- `GET /api/optimization/recommendations` - Cost savings

### Migration Advisor
- `POST /api/migration-advisor/projects` - Create assessment
- `GET /api/migration-advisor/projects/{id}/score-preview` - Real-time scores
- `GET /api/migration-advisor/projects/{id}/enhanced-recommendation` - Full recommendation

Full API docs: **http://localhost:8000/docs**

---

## 🔐 IAM Permissions Required

The AWS user/role needs the following permissions to use all platform features:

### Minimum Required Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CostExplorerAccess",
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "ce:GetDimensionValues",
        "ce:GetTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EC2ReadAccess",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeVolumes",
        "ec2:DescribeSnapshots",
        "ec2:DescribeRegions"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchReadAccess",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BudgetsAccess",
      "Effect": "Allow",
      "Action": [
        "budgets:ViewBudget",
        "budgets:DescribeBudgets"
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
    }
  ]
}
```

### Optional: Automation Features

For automated remediation and optimization actions, add:

```json
{
  "Sid": "EC2ManagementAccess",
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

**Security Note**: The automation policy includes a condition to only affect non-production resources. Adjust the condition based on your tagging strategy.

---

## 🔧 Troubleshooting Guide

### Issue: "AWS credentials not found"

**Symptoms**: Backend fails to start or dashboard shows credential errors

**Solutions**:
1. Verify `.env` file exists in the root directory
2. Check that `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set
3. Ensure no extra spaces or quotes around the values
4. Test credentials with: `python test_aws_integration.py`

### Issue: "No cost data available"

**Symptoms**: Dashboard loads but shows no cost information

**Solutions**:
1. **Cost Explorer Delay**: AWS Cost Explorer has a 24-hour data delay. Recent costs won't appear immediately.
2. **Enable Cost Explorer**: Go to AWS Console → Billing → Cost Explorer and enable it (takes 24 hours to activate)
3. **Check Date Range**: Ensure you're looking at dates with actual AWS usage
4. **Verify Permissions**: Run `python test_aws_integration.py` to check IAM permissions

### Issue: "AccessDeniedException" errors

**Symptoms**: API calls fail with permission denied errors

**Solutions**:
1. Review the [IAM Permissions Required](#iam-permissions-required) section
2. Attach the minimum required policy to your IAM user/role
3. Wait 5-10 minutes for IAM changes to propagate
4. Test with: `aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity DAILY --metrics BlendedCost`

### Issue: "ThrottlingException" errors

**Symptoms**: API calls fail with rate limit errors

**Solutions**:
1. The platform implements automatic retry with exponential backoff
2. Reduce the frequency of dashboard refreshes
3. Consider implementing Redis caching (see `CACHING_IMPLEMENTATION.md`)
4. For high-volume usage, request AWS API limit increases

### Issue: Frontend won't start

**Symptoms**: `npm start` fails with errors

**Solutions**:
1. Delete `node_modules` and `package-lock.json`: `rm -rf node_modules package-lock.json`
2. Reinstall dependencies: `npm install`
3. Check Node.js version: `node --version` (requires 16+)
4. Clear npm cache: `npm cache clean --force`

### Issue: Backend won't start

**Symptoms**: `python start_py` fails

**Solutions**:
1. Check Python version: `python --version` (requires 3.10+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check for port conflicts: `lsof -i :8000` (kill conflicting processes)
4. Review error logs for specific issues

### Issue: "Connection refused" when frontend calls backend

**Symptoms**: Frontend loads but API calls fail

**Solutions**:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check CORS configuration in `backend/main.py`
3. Ensure frontend is configured to call `http://localhost:8000`
4. Check browser console for specific error messages

### Issue: Slow dashboard loading

**Symptoms**: Dashboard takes >10 seconds to load

**Solutions**:
1. Implement caching (see `CACHING_IMPLEMENTATION.md`)
2. Reduce date range for cost queries
3. Enable Redis for response caching
4. Check AWS API response times with `test_aws_integration.py`

### Getting Help

If you encounter issues not covered here:

1. Check the logs: Backend logs appear in the terminal running `start_py`
2. Run the integration test: `python test_aws_integration.py` for detailed diagnostics
3. Review AWS CloudTrail for API call failures
4. Check the [GitHub Issues](https://github.com/sanketexe/ai-cloud-scheduler/issues) for similar problems

---

## 🔧 Troubleshooting

### Common Issues

**"AWS credentials not found"**
- Verify `.env` file exists with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Run: `python test_aws_integration.py`

**"No cost data available"**
- Cost Explorer has 24-hour delay
- Enable Cost Explorer in AWS Console → Billing
- Check IAM permissions

**"AccessDeniedException"**
- Review IAM permissions (see above)
- Wait 5-10 minutes for IAM changes to propagate

**Frontend/Backend won't start**
- Check port conflicts (8000, 3000)
- Reinstall dependencies
- Check Node.js (16+) and Python (3.10+) versions

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when running)
- **AWS Setup**: See IAM Permissions section above
- **Troubleshooting**: See section above

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📧 Support

- **GitHub Issues**: https://github.com/sanketexe/ai-cloud-scheduler/issues
- **Email**: support@example.com

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file