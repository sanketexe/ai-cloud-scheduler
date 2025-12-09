# 📊 What Data is Shown in the Dashboard?

## 🎯 Current State: Demo/Mock Data

The dashboard is currently showing **MOCK/DEMO DATA** - not real company data.

### Why Mock Data?

The platform is designed to work with **YOUR cloud accounts**, but until you connect them, it shows sample data so you can:
- ✅ See how the interface looks
- ✅ Understand the features
- ✅ Test the functionality
- ✅ Explore without risk

---

## 📋 What Mock Data is Being Shown?

### 1. **Dashboard (FinOps Overview)**
**File:** `frontend/src/pages/Dashboard.tsx`

**Mock Data Includes:**
- **Cost Trends:** Sample monthly costs ($45K-$52K)
- **Budget Data:** Fictional budget limits
- **Forecast Data:** Simulated predictions
- **Company Name:** "Your Company" (placeholder)

**Example:**
```typescript
costTrendData = [
  { date: '2024-01-01', cost: 45230, budget: 50000 },
  { date: '2024-02-01', cost: 48500, budget: 50000 },
  // ... more sample data
]
```

---

### 2. **Cost Analysis Page**
**File:** `frontend/src/pages/CostAnalysis.tsx`

**Mock Data Includes:**
- **Service Costs:** Compute, Storage, Network, Database
- **Monthly Breakdown:** Sample cost distribution
- **Trends:** Fictional growth patterns

**Example:**
```typescript
{ 
  date: '2024-01-01', 
  compute: 1200, 
  storage: 800, 
  network: 300, 
  database: 500 
}
```

---

### 3. **Budget Management**
**File:** `frontend/src/pages/BudgetManagement.tsx`

**Mock Data Includes:**
- **Team Budgets:** Engineering, Marketing, Sales
- **Spending:** Fictional utilization percentages
- **Alerts:** Sample budget warnings

---

### 4. **Optimization Recommendations**
**File:** `frontend/src/pages/Optimization.tsx`

**Mock Data Includes:**
- **Idle Resources:** Sample EC2 instances
- **Savings Opportunities:** Fictional cost reductions
- **Right-sizing Suggestions:** Example recommendations

---

### 5. **Migration Results**
**File:** `frontend/src/pages/MigrationResults.tsx`

**Mock Data Includes:**
- **Provider Comparison:** AWS vs GCP vs Azure
- **Cost Estimates:** Sample pricing
- **Migration Timeline:** Fictional phases

**Example:**
```typescript
mockResults = {
  organization_name: 'Your Company',
  providers: [
    { name: 'AWS', monthly_cost: 12500, score: 85 },
    { name: 'GCP', monthly_cost: 11800, score: 82 },
    { name: 'Azure', monthly_cost: 13200, score: 78 }
  ]
}
```

---

### 6. **Compliance Dashboard**
**File:** `frontend/src/pages/Compliance.tsx`

**Mock Data Includes:**
- **Compliance Score:** 87% (fictional)
- **Violations:** Sample tagging issues
- **Policies:** Example governance rules

---

### 7. **Alerts**
**File:** `frontend/src/pages/Alerts.tsx`

**Mock Data Includes:**
- **Budget Alerts:** Sample warnings
- **Cost Anomalies:** Fictional spikes
- **Compliance Issues:** Example violations

---

## 🔄 How to Show YOUR Company's Data

### Step 1: Connect Your Cloud Account

#### For AWS:
```
1. Go to Settings → Cloud Providers
2. Click "Add AWS Account"
3. Enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Account Name
4. Click "Connect"
```

#### For GCP:
```
1. Go to Settings → Cloud Providers
2. Click "Add GCP Account"
3. Enter:
   - Project ID
   - Service Account JSON
4. Click "Connect"
```

#### For Azure:
```
1. Go to Settings → Cloud Providers
2. Click "Add Azure Account"
3. Enter:
   - Subscription ID
   - Tenant ID
   - Client ID
   - Client Secret
4. Click "Connect"
```

---

### Step 2: Sync Cost Data

Once connected, the platform will:
1. **Discover Resources** - Find all your cloud resources
2. **Fetch Cost Data** - Pull billing information
3. **Analyze Spending** - Calculate trends and patterns
4. **Generate Insights** - Create recommendations

**Time to First Data:** 5-15 minutes (depending on account size)

---

### Step 3: Configure Organization

Set up your company structure:
1. **Teams:** Engineering, Marketing, Sales, etc.
2. **Projects:** Product A, Product B, etc.
3. **Cost Centers:** Department budgets
4. **Tags:** Resource categorization

---

## 🎨 Customizing the Mock Data

If you want to customize the demo data before connecting real accounts:

### Option 1: Edit Frontend Mock Data

**File:** `frontend/src/pages/Dashboard.tsx`

```typescript
// Change company name
const mockResults = {
  organization_name: 'Acme Corporation', // Your company name
  // ... rest of data
}

// Change cost values
const costTrendData = [
  { date: '2024-01-01', cost: 75000, budget: 80000 }, // Your numbers
  // ...
]
```

### Option 2: Use Backend Sample Data Generator

**File:** `backend/core/waste_detection_engine.py`

```python
def create_sample_data():
    """Customize sample data here"""
    resources = [
        CloudResource(
            resource_id="your-resource-id",
            resource_type="your-type",
            # ... customize
        )
    ]
```

---

## 🔍 How to Tell if Data is Real or Mock

### Mock Data Indicators:
- ✅ Company name is "Your Company"
- ✅ Resource IDs like "i-1234567890abcdef0"
- ✅ Round numbers (e.g., exactly $50,000)
- ✅ Perfect patterns (no real-world noise)
- ✅ No cloud provider logo/badge

### Real Data Indicators:
- ✅ Your actual company name
- ✅ Real AWS/GCP/Azure resource IDs
- ✅ Irregular numbers (e.g., $47,832.47)
- ✅ Real-world variations
- ✅ Cloud provider badges shown

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────┐
│  BEFORE CONNECTING CLOUD ACCOUNTS               │
│                                                  │
│  Frontend → Mock Data (hardcoded in files)      │
│  Dashboard shows: "Your Company" + sample data  │
└─────────────────────────────────────────────────┘

                      ↓ Connect AWS/GCP/Azure

┌─────────────────────────────────────────────────┐
│  AFTER CONNECTING CLOUD ACCOUNTS                │
│                                                  │
│  Cloud Provider → Backend API → Database        │
│  Frontend → Real Data from your accounts        │
│  Dashboard shows: Your company + actual costs   │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Summary

### Current State:
- **Data Source:** Mock/Demo data (hardcoded)
- **Company:** "Your Company" (placeholder)
- **Purpose:** Demonstration and testing

### To Get Real Data:
1. Connect your AWS/GCP/Azure account
2. Wait for initial sync (5-15 minutes)
3. Configure your organization structure
4. Start seeing YOUR actual costs and resources

### Why This Approach?
- ✅ Safe to explore without connecting accounts
- ✅ See full functionality before committing
- ✅ No risk to your cloud infrastructure
- ✅ Easy to test and demo

---

## 🚀 Next Steps

1. **Explore the mock data** - Get familiar with features
2. **Read the documentation** - Understand capabilities
3. **Connect your cloud account** - See real data
4. **Configure settings** - Customize for your needs
5. **Start optimizing** - Save money on cloud costs!

---

## ❓ FAQ

**Q: Is the mock data based on real companies?**
A: No, it's completely fictional sample data for demonstration.

**Q: Will my data be mixed with mock data?**
A: No, once you connect your account, all mock data is replaced with your real data.

**Q: Can I use the platform without connecting cloud accounts?**
A: Yes, you can explore all features with mock data, but you won't get real insights.

**Q: How do I know when real data is loaded?**
A: You'll see your actual company name, real resource IDs, and a "Last Synced" timestamp.

**Q: Is my cloud data secure?**
A: Yes, credentials are encrypted, and the platform uses read-only access to your cloud accounts.

---

**Ready to see YOUR company's data? Connect your cloud account in Settings!** 🎉
