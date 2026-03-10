# Troubleshooting Guide - AWS FinOps Platform

This guide covers common issues, their causes, and solutions for the AWS FinOps Platform.

## Table of Contents

1. [AWS Connection Issues](#aws-connection-issues)
2. [Cost Data Issues](#cost-data-issues)
3. [Application Startup Issues](#application-startup-issues)
4. [Performance Issues](#performance-issues)
5. [API Errors](#api-errors)
6. [Frontend Issues](#frontend-issues)
7. [Docker Issues](#docker-issues)
8. [Data Accuracy Issues](#data-accuracy-issues)

---

## AWS Connection Issues

### Issue: "AWS credentials not found"

**Symptoms**:
- Backend fails to start
- Error message: `NoCredentialsError: Unable to locate credentials`
- Dashboard shows "Configuration Error"

**Causes**:
- `.env` file missing or in wrong location
- Environment variables not set
- Incorrect variable names

**Solutions**:

1. **Verify .env file exists**:
   ```bash
   ls -la .env
   ```
   Should show `.env` file in root directory

2. **Check .env contents**:
   ```bash
   cat .env
   ```
   Should contain:
   ```
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=us-east-1
   ```

3. **Verify no extra spaces**:
   ```bash
   # Bad (has spaces)
   AWS_ACCESS_KEY_ID = AKIA...
   
   # Good (no spaces)
   AWS_ACCESS_KEY_ID=AKIA...
   ```

4. **Test credentials**:
   ```bash
   python test_aws_integration.py
   ```

### Issue: "InvalidClientTokenId" error

**Symptoms**:
- API calls fail with `InvalidClientTokenId`
- Test script fails on identity verification
- Error: "The security token included in the request is invalid"

**Causes**:
- Access key ID is incorrect
- Access key has been deleted in AWS Console
- Typo in access key ID

**Solutions**:

1. **Verify access key in AWS Console**:
   - Go to IAM → Users → [your-user] → Security credentials
   - Check access key ID matches `.env` file
   - Look for "Active" status

2. **Create new access keys**:
   ```bash
   aws iam create-access-key --user-name finops-platform
   ```

3. **Update .env with new credentials**

4. **Test again**:
   ```bash
   python test_aws_integration.py
   ```

### Issue: "SignatureDoesNotMatch" error

**Symptoms**:
- API calls fail with `SignatureDoesNotMatch`
- Error: "The request signature we calculated does not match"

**Causes**:
- Secret access key is incorrect
- Extra spaces or newlines in secret key
- Copy-paste error

**Solutions**:

1. **Regenerate access keys**:
   - Cannot retrieve existing secret key
   - Must create new access key pair

2. **Careful copy-paste**:
   - Copy secret key immediately after creation
   - Avoid extra spaces or newlines
   - Use "Download .csv" option for accuracy

3. **Update .env file**:
   ```bash
   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   ```
   (No quotes, no spaces)

### Issue: "AccessDeniedException" errors

**Symptoms**:
- API calls fail with `AccessDeniedException`
- Error: "User is not authorized to perform: ce:GetCostAndUsage"
- Test script shows permission failures

**Causes**:
- IAM policy missing required permissions
- Policy not attached to user
- IAM changes not yet propagated

**Solutions**:

1. **Verify IAM policy attached**:
   ```bash
   aws iam list-attached-user-policies --user-name finops-platform
   ```

2. **Check policy permissions**:
   - See `docs/AWS_SETUP_GUIDE.md` for required policy
   - Ensure all actions are included:
     - `ce:GetCostAndUsage`
     - `ec2:DescribeInstances`
     - `cloudwatch:GetMetricStatistics`
     - `sts:GetCallerIdentity`

3. **Wait for propagation**:
   - IAM changes take 5-10 minutes to propagate
   - Wait and retry

4. **Test specific permission**:
   ```bash
   aws ce get-cost-and-usage \
     --time-period Start=2024-01-01,End=2024-01-02 \
     --granularity DAILY \
     --metrics BlendedCost
   ```

---

## Cost Data Issues

### Issue: "No cost data available"

**Symptoms**:
- Dashboard loads but shows $0.00 or "No data"
- Charts are empty
- Message: "No cost data found for this period"

**Causes**:
- Cost Explorer not enabled
- 24-hour data delay
- No AWS usage in selected period
- Date range too recent

**Solutions**:

1. **Enable Cost Explorer**:
   - Go to AWS Console → Billing → Cost Explorer
   - Click "Enable Cost Explorer"
   - Wait 24 hours for activation

2. **Check date range**:
   - Cost Explorer has 24-hour delay
   - Today's costs won't appear until tomorrow
   - Try querying 2-7 days ago

3. **Verify AWS usage**:
   - Check you have running resources (EC2, S3, etc.)
   - View AWS Billing Console to confirm charges exist

4. **Test data retrieval**:
   ```bash
   python test_aws_integration.py
   ```
   Should show cost data in test results

### Issue: Cost data seems incorrect

**Symptoms**:
- Costs don't match AWS Billing Console
- Numbers seem too high or too low
- Service breakdown doesn't add up

**Causes**:
- Different date ranges
- Different cost types (blended vs unblended)
- Tax and credits included/excluded
- Currency conversion

**Solutions**:

1. **Match date ranges**:
   - Ensure same start/end dates as AWS Console
   - Check timezone differences

2. **Check cost type**:
   - Platform uses "BlendedCost" by default
   - AWS Console may show "UnblendedCost"
   - Blended includes RI/SP discounts

3. **Verify filters**:
   - Check if filtering by service/region
   - Ensure no filters applied in AWS Console

4. **Compare API response**:
   ```bash
   aws ce get-cost-and-usage \
     --time-period Start=2024-03-01,End=2024-03-08 \
     --granularity DAILY \
     --metrics BlendedCost
   ```

### Issue: "Cost Explorer is not enabled"

**Symptoms**:
- Error: "Cost Explorer is not enabled for this account"
- API calls fail with specific error code

**Causes**:
- Cost Explorer never enabled
- Recently enabled (< 24 hours)
- Wrong AWS account

**Solutions**:

1. **Enable Cost Explorer**:
   ```bash
   aws ce enable-cost-explorer
   ```

2. **Wait 24 hours**:
   - Cost Explorer takes 24 hours to activate
   - Data won't appear immediately

3. **Verify account**:
   ```bash
   aws sts get-caller-identity
   ```
   Ensure correct account ID

---

## Application Startup Issues

### Issue: Backend won't start

**Symptoms**:
- `python start_backend.py` fails
- Import errors
- Port already in use

**Causes**:
- Python version too old
- Missing dependencies
- Port 8000 already in use
- Environment variable issues

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version
   # Should be 3.10 or higher
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check port availability**:
   ```bash
   # Linux/Mac
   lsof -i :8000
   
   # Windows
   netstat -ano | findstr :8000
   ```
   Kill process if port in use

4. **Check for errors**:
   ```bash
   python start_backend.py
   ```
   Read error messages carefully

### Issue: Frontend won't start

**Symptoms**:
- `npm start` fails
- Dependency errors
- Port 3000 already in use

**Causes**:
- Node.js version too old
- Corrupted node_modules
- Port conflict
- Missing dependencies

**Solutions**:

1. **Check Node.js version**:
   ```bash
   node --version
   # Should be 16 or higher
   ```

2. **Clean install**:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Clear npm cache**:
   ```bash
   npm cache clean --force
   npm install
   ```

4. **Use different port**:
   ```bash
   PORT=3001 npm start
   ```

### Issue: "Connection refused" between frontend and backend

**Symptoms**:
- Frontend loads but API calls fail
- Console error: "Network Error"
- Error: "connect ECONNREFUSED 127.0.0.1:8000"

**Causes**:
- Backend not running
- Wrong API URL in frontend
- CORS issues
- Firewall blocking

**Solutions**:

1. **Verify backend running**:
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy"}
   ```

2. **Check frontend API configuration**:
   - File: `frontend/src/services/api.ts`
   - Should have: `baseURL: 'http://localhost:8000'`

3. **Check CORS settings**:
   - File: `backend/main.py`
   - Should allow `http://localhost:3000`

4. **Check firewall**:
   ```bash
   # Temporarily disable firewall to test
   # Linux: sudo ufw disable
   # Mac: System Preferences → Security → Firewall
   # Windows: Control Panel → Windows Defender Firewall
   ```

---

## Performance Issues

### Issue: Dashboard loads very slowly

**Symptoms**:
- Dashboard takes >10 seconds to load
- API calls timeout
- Browser becomes unresponsive

**Causes**:
- No caching enabled
- Large date ranges
- Slow AWS API responses
- Too many concurrent requests

**Solutions**:

1. **Enable Redis caching**:
   ```bash
   # Install Redis
   # Linux: sudo apt install redis-server
   # Mac: brew install redis
   
   # Add to .env
   REDIS_URL=redis://localhost:6379/0
   
   # Start Redis
   redis-server
   ```

2. **Reduce date range**:
   - Query 7-30 days instead of 90+ days
   - Use monthly granularity for long periods

3. **Check AWS API performance**:
   ```bash
   time aws ce get-cost-and-usage \
     --time-period Start=2024-03-01,End=2024-03-08 \
     --granularity DAILY \
     --metrics BlendedCost
   ```

4. **Enable parallel requests**:
   - Already implemented in frontend
   - Check browser network tab for concurrent calls

### Issue: "ThrottlingException" errors

**Symptoms**:
- API calls fail intermittently
- Error: "Rate exceeded"
- Error: "ThrottlingException"

**Causes**:
- Too many AWS API calls
- AWS service limits reached
- No retry logic
- Concurrent requests

**Solutions**:

1. **Platform has automatic retry**:
   - Exponential backoff implemented
   - Should retry automatically

2. **Enable caching**:
   - Reduces AWS API calls
   - See Redis setup above

3. **Reduce request frequency**:
   - Don't refresh dashboard too often
   - Use longer cache TTL

4. **Request limit increase**:
   - Contact AWS Support
   - Request higher API limits

---

## API Errors

### Issue: 500 Internal Server Error

**Symptoms**:
- API returns 500 status code
- Generic error message
- Backend logs show exception

**Causes**:
- Unhandled exception in backend
- Database connection failure
- AWS API error not caught

**Solutions**:

1. **Check backend logs**:
   - Look at terminal running `start_backend.py`
   - Find stack trace and error message

2. **Common fixes**:
   ```bash
   # Restart backend
   # Ctrl+C to stop
   python start_backend.py
   ```

3. **Check database connection**:
   ```bash
   # If using PostgreSQL
   psql -U finops -d finops_db -c "SELECT 1"
   ```

4. **Enable debug mode**:
   ```bash
   # Add to .env
   LOG_LEVEL=DEBUG
   ```

### Issue: 404 Not Found

**Symptoms**:
- API returns 404 status code
- Error: "Not Found"

**Causes**:
- Wrong API endpoint URL
- Route not registered
- Typo in URL

**Solutions**:

1. **Check API documentation**:
   - Open http://localhost:8000/docs
   - Verify endpoint exists

2. **Check frontend API calls**:
   - Look in browser console
   - Verify URL is correct

3. **Check backend routes**:
   - File: `backend/main.py`
   - Ensure route is registered

---

## Frontend Issues

### Issue: Charts not rendering

**Symptoms**:
- Dashboard loads but charts are blank
- Console errors about Recharts
- Data loads but visualization fails

**Causes**:
- Invalid data format
- Missing chart dependencies
- Browser compatibility

**Solutions**:

1. **Check browser console**:
   - Look for JavaScript errors
   - Check data format

2. **Verify data structure**:
   ```javascript
   // Expected format for line chart
   [
     { date: '2024-03-01', cost: 123.45 },
     { date: '2024-03-02', cost: 134.56 }
   ]
   ```

3. **Update dependencies**:
   ```bash
   cd frontend
   npm update recharts
   ```

4. **Try different browser**:
   - Test in Chrome, Firefox, Safari
   - Check browser version

### Issue: "Module not found" errors

**Symptoms**:
- Frontend fails to compile
- Error: "Module not found: Can't resolve..."

**Causes**:
- Missing dependency
- Incorrect import path
- Case sensitivity issue

**Solutions**:

1. **Install missing dependency**:
   ```bash
   cd frontend
   npm install [missing-package]
   ```

2. **Check import paths**:
   ```typescript
   // Bad (wrong case)
   import { Button } from './Components/button'
   
   // Good (correct case)
   import { Button } from './components/Button'
   ```

3. **Reinstall dependencies**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

---

## Docker Issues

### Issue: Docker containers won't start

**Symptoms**:
- `docker-compose up` fails
- Containers exit immediately
- Port binding errors

**Causes**:
- Ports already in use
- Missing .env file
- Docker daemon not running

**Solutions**:

1. **Check Docker running**:
   ```bash
   docker ps
   ```

2. **Check port conflicts**:
   ```bash
   docker-compose down
   # Kill processes on ports 3000, 8000, 6379, 5432
   docker-compose up -d
   ```

3. **View container logs**:
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```

4. **Rebuild containers**:
   ```bash
   docker-compose down -v
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Issue: Docker containers running but not accessible

**Symptoms**:
- Containers show as "Up"
- Can't access http://localhost:3000
- Connection refused

**Causes**:
- Port mapping incorrect
- Container networking issue
- Service not listening on 0.0.0.0

**Solutions**:

1. **Check port mappings**:
   ```bash
   docker-compose ps
   # Should show: 0.0.0.0:3000->3000/tcp
   ```

2. **Check container logs**:
   ```bash
   docker-compose logs -f backend
   ```

3. **Test from inside container**:
   ```bash
   docker-compose exec backend curl http://localhost:8000/health
   ```

---

## Data Accuracy Issues

### Issue: Budget tracking shows incorrect utilization

**Symptoms**:
- Budget percentage doesn't match manual calculation
- Utilization seems off

**Causes**:
- Different date ranges
- Budget period mismatch
- Timezone differences

**Solutions**:

1. **Verify budget period**:
   - Check if budget is monthly, quarterly, or annual
   - Ensure comparing same period

2. **Check date range**:
   - Budget tracking uses current month by default
   - Verify dates match your expectation

3. **Manual verification**:
   ```python
   # In Python console
   from datetime import datetime
   from backend.core.budget_management_system import BudgetManager
   
   manager = BudgetManager()
   utilization = manager.get_budget_utilization(budget_id)
   print(f"Spent: ${utilization['spent']}")
   print(f"Budget: ${utilization['budget']}")
   print(f"Percentage: {utilization['percentage']}%")
   ```

### Issue: Optimization recommendations seem wrong

**Symptoms**:
- Recommends downsizing busy instances
- Suggests stopping production resources
- Savings estimates don't make sense

**Causes**:
- Insufficient CloudWatch data
- Wrong time period analyzed
- Safety tags not configured

**Solutions**:

1. **Check CloudWatch metrics period**:
   - Platform analyzes last 7-14 days
   - Ensure representative workload during that time

2. **Verify safety tags**:
   - Add `Environment=Production` to production resources
   - Platform will skip these in automation

3. **Review recommendation logic**:
   ```python
   # Check CPU utilization threshold
   # File: backend/core/ec2_instance_optimizer.py
   # Default: < 10% CPU = idle
   ```

4. **Adjust thresholds**:
   - Modify idle detection thresholds
   - Customize for your workload patterns

---

## Getting Additional Help

If your issue isn't covered here:

1. **Run diagnostics**:
   ```bash
   python test_aws_integration.py > diagnostics.txt
   ```

2. **Collect logs**:
   ```bash
   # Backend logs
   python start_backend.py > backend.log 2>&1
   
   # Frontend logs
   cd frontend
   npm start > frontend.log 2>&1
   ```

3. **Check AWS CloudTrail**:
   - Review API calls made by platform
   - Look for errors or denied requests

4. **Review documentation**:
   - `README.md` - Main documentation
   - `docs/AWS_SETUP_GUIDE.md` - AWS configuration
   - `docs/ARCHITECTURE.md` - System design

5. **Open GitHub issue**:
   - Include diagnostics output
   - Include relevant log excerpts
   - Describe steps to reproduce

---

## Preventive Measures

### Regular Maintenance

1. **Rotate AWS credentials** every 90 days
2. **Update dependencies** monthly:
   ```bash
   pip install -r requirements.txt --upgrade
   cd frontend && npm update
   ```
3. **Monitor AWS CloudTrail** for unusual activity
4. **Review IAM permissions** quarterly
5. **Test backup/restore** procedures

### Monitoring

1. **Set up CloudWatch alarms** for API errors
2. **Monitor application logs** for warnings
3. **Track API call counts** to avoid throttling
4. **Monitor cache hit rates** for performance

### Best Practices

1. **Use IAM roles** instead of access keys when possible
2. **Enable MFA** on AWS accounts
3. **Tag all resources** for better cost tracking
4. **Document custom configurations**
5. **Keep .env.example** updated with required variables

---

**Last Updated**: March 2024

For the latest troubleshooting tips, check the GitHub repository issues and discussions.
