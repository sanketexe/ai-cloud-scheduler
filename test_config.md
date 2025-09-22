# 🧪 Manual Testing Guide - AI Cloud Scheduler

> Complete step-by-step testing procedures for the AI Cloud Scheduler using the web interface

![Testing](https://img.shields.io/badge/Testing-Manual-blue?style=for-the-badge)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge)
![Guide](https://img.shields.io/badge/Guide-Beginner--Friendly-green?style=for-the-badge)

## 📋 Table of Contents

- [Pre-Testing Setup](#-pre-testing-setup)
- [Test Environment Verification](#-test-environment-verification)
- [Core Feature Testing](#-core-feature-testing)
- [Advanced Feature Testing](#-advanced-feature-testing)
- [Error Scenarios Testing](#-error-scenarios-testing)
- [Performance Testing](#-performance-testing)
- [Test Results Documentation](#-test-results-documentation)

## 🚀 Pre-Testing Setup

### Step 1: Start the Application

#### 1.1 Start the API Server
```bash
# Open Terminal 1
cd /path/to/ai-cloud-scheduler
python api.py
```

**Expected Output:**
```
🚀 API Server initialized and ready!
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 1.2 Start the Web Interface
```bash
# Open Terminal 2
cd /path/to/ai-cloud-scheduler
streamlit run streamlit_app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

#### 1.3 Open Browser
- Navigate to: `http://localhost:8501`
- You should see the AI Cloud Scheduler interface

---

## ✅ Test Environment Verification

### Test 1: Initial Interface Load
**Objective:** Verify the application loads correctly

**Steps:**
1. Open browser to `http://localhost:8501`
2. Wait for page to fully load
3. Check page title shows "AI Cloud Scheduler"
4. Verify sidebar navigation is visible

**Expected Results:**
- ✅ Page loads without errors
- ✅ Title displays: "☁️ AI-Powered Cloud Scheduler"
- ✅ Sidebar shows navigation options: Dashboard, Configuration, Simulation, Results, ML Predictions
- ✅ API status in sidebar shows "🟢 API Connected"

**If API Status Shows Red:**
- Check Terminal 1 to ensure API is running
- Verify no error messages in browser console (F12)
- Restart API server if needed

### Test 2: Navigation Functionality
**Objective:** Verify all navigation links work

**Steps:**
1. Click each sidebar navigation item:
   - Dashboard
   - Configuration  
   - Simulation
   - Results
   - ML Predictions

**Expected Results:**
- ✅ Each page loads without errors
- ✅ Page content changes appropriately
- ✅ No broken layouts or missing elements
- ✅ Navigation remains functional

---

## 🧩 Core Feature Testing

### Test 3: Dashboard Overview
**Objective:** Verify dashboard displays system status correctly

**Steps:**
1. Navigate to "Dashboard" page
2. Observe the metrics displayed
3. Check Quick Actions section
4. Note the metric values

**Expected Results:**
- ✅ Shows 4 metrics: Total VMs, Schedulers Available, Loaded Workloads, Last Simulation
- ✅ Total VMs shows "4"
- ✅ Schedulers Available shows "3"
- ✅ Loaded Workloads initially shows "0"
- ✅ Last Simulation shows "None"
- ✅ Three Quick Action buttons are visible and clickable

### Test 4: Load Sample Data
**Objective:** Test loading sample workloads

**Steps:**
1. On Dashboard, click "📈 Load Sample Data" button
2. Wait for processing
3. Observe the success message
4. Check if "Loaded Workloads" metric updates

**Expected Results:**
- ✅ Success message appears: "✅ Loaded X sample workloads"
- ✅ "Loaded Workloads" metric updates to show count > 0
- ✅ No error messages displayed
- ✅ Button click is responsive

### Test 5: Configuration - Cloud Providers
**Objective:** Verify provider configuration displays and works

**Steps:**
1. Navigate to "Configuration" tab
2. Click on "Cloud Providers" tab
3. Expand each provider configuration
4. Note the default values
5. Try modifying a CPU cost value
6. Change it back to original

**Expected Results:**
- ✅ Shows 3 providers: AWS, GCP, Azure
- ✅ Each provider has expandable configuration
- ✅ AWS CPU Cost: 0.04, Memory Cost: 0.01
- ✅ GCP CPU Cost: 0.035, Memory Cost: 0.009  
- ✅ Azure CPU Cost: 0.042, Memory Cost: 0.011
- ✅ Values can be modified using number inputs
- ✅ Changes are reflected immediately

### Test 6: Configuration - Virtual Machines
**Objective:** Test VM configuration display

**Steps:**
1. Stay in Configuration tab
2. Click on "Virtual Machines" tab
3. Expand each VM configuration
4. Note the VM specifications
5. Try changing a CPU capacity value

**Expected Results:**
- ✅ Shows 4 VMs with different configurations
- ✅ VM 1: 4 CPU, 16GB Memory, AWS
- ✅ VM 2: 8 CPU, 32GB Memory, GCP
- ✅ VM 3: 4 CPU, 16GB Memory, Azure
- ✅ VM 4: 2 CPU, 8GB Memory, GCP
- ✅ Dropdown for provider selection works
- ✅ Number inputs are functional

### Test 7: Workload Configuration - Sample Data
**Objective:** Test different workload input methods

**Steps:**
1. Go to Configuration → Workloads tab
2. Select "Use Sample Data" radio button
3. Click "Load Sample Workloads" button
4. Check if workloads appear in the table

**Expected Results:**
- ✅ Radio button selection works
- ✅ Button click loads data successfully
- ✅ Success message appears
- ✅ Data table shows workloads with columns: workload_id, cpu_required, memory_required_gb
- ✅ Table displays approximately 8 sample workloads

### Test 8: Workload Configuration - Manual Entry
**Objective:** Test manual workload creation

**Steps:**
1. Still in Workloads tab
2. Select "Manual Entry" radio button
3. Fill in the form:
   - Workload ID: 999
   - CPU Required: 3
   - Memory Required: 6
4. Click "Add Workload" button
5. Check if workload appears in the table

**Expected Results:**
- ✅ Form appears with three input fields
- ✅ All fields accept numerical input
- ✅ Form submission works without errors
- ✅ Success message: "Workload added!"
- ✅ New workload appears in the data table
- ✅ Page refreshes to show updated data

### Test 9: Workload Configuration - Random Generation
**Objective:** Test random workload generator

**Steps:**
1. Select "Generate Random" radio button
2. Set parameters:
   - Number of Workloads: 5
   - Min CPU: 1
   - Max CPU: 4
   - Min Memory: 2
   - Max Memory: 8
3. Click "Generate Random Workloads"
4. Verify generated workloads

**Expected Results:**
- ✅ Form shows all parameter inputs
- ✅ Number inputs accept values within specified ranges
- ✅ Generation completes successfully
- ✅ Success message shows count of generated workloads
- ✅ Generated workloads appear in table
- ✅ Values fall within specified ranges

### Test 10: Clear Workloads Function
**Objective:** Test workload clearing functionality

**Steps:**
1. Ensure workloads are loaded from previous tests
2. Click "Clear All Workloads" button
3. Confirm the action if prompted
4. Check the workload table

**Expected Results:**
- ✅ Button is visible when workloads exist
- ✅ Button click clears all workloads
- ✅ Table becomes empty
- ✅ Dashboard metric "Loaded Workloads" resets to 0
- ✅ Page updates immediately

---

## 🎯 Advanced Feature Testing

### Test 11: Basic Simulation Run
**Objective:** Test simulation execution with default settings

**Steps:**
1. Ensure workloads are loaded (use sample data if empty)
2. Navigate to "Simulation" tab
3. Keep all three scheduler checkboxes selected:
   - Random Scheduler
   - Lowest Cost Scheduler  
   - Round Robin Scheduler
4. Click "🚀 Start Simulation" button
5. Wait for processing
6. Navigate to "Results" tab

**Expected Results:**
- ✅ Simulation page loads correctly
- ✅ Three scheduler options are visible and selected
- ✅ Button click triggers simulation
- ✅ Progress indicators or loading messages appear
- ✅ Success message: "All simulations completed successfully!"
- ✅ Balloons animation appears
- ✅ Results tab shows simulation data

### Test 12: Results Analysis
**Objective:** Verify simulation results display correctly

**Steps:**
1. In Results tab (after running simulation)
2. Check the Summary section metrics
3. Examine the Performance Comparison charts
4. Expand each scheduler's detailed results
5. Look for assignment logs and pie charts

**Expected Results:**
- ✅ Summary shows metrics for all three schedulers
- ✅ Metrics include success rate percentages
- ✅ Three comparison charts display: Success Rate, Total Cost, Successful Workloads
- ✅ Each scheduler has expandable detail section
- ✅ Detail sections show 4 metrics: Success Rate, Successful, Total Workloads, Total Cost
- ✅ Assignment tables show workload assignments
- ✅ Pie charts display success/failure ratios

### Test 13: Single Scheduler Simulation
**Objective:** Test running simulation with only one scheduler

**Steps:**
1. Go back to Simulation tab
2. Uncheck "Random Scheduler" and "Round Robin Scheduler"
3. Keep only "Lowest Cost Scheduler" checked
4. Click "🚀 Start Simulation"
5. Check results

**Expected Results:**
- ✅ Checkboxes work independently
- ✅ Simulation runs with only selected scheduler
- ✅ Results show data for only one scheduler
- ✅ Charts and metrics display correctly for single scheduler

### Test 14: System Configuration Overview
**Objective:** Test configuration management interface

**Steps:**
1. Navigate to Configuration → System Config tab
2. Review the configuration overview
3. Check different configuration categories
4. Note the metrics and settings displayed

**Expected Results:**
- ✅ Configuration overview displays system status
- ✅ Shows total categories and last updated time
- ✅ Multiple expandable categories are visible (API, Scheduler, Providers, etc.)
- ✅ Each category shows setting count and key configurations
- ✅ Edit buttons are present (may show "coming soon" message)

---

## 🤖 Advanced Feature Testing - ML Predictions

### Test 15: ML Model Training
**Objective:** Test machine learning model training interface

**Steps:**
1. Navigate to "ML Predictions" tab
2. Check the model status indicator
3. Go to "Train Model" sub-tab
4. Click "🚀 Train Model" button (without uploading data)
5. Wait for training completion
6. Check status update

**Expected Results:**
- ✅ Model status initially shows "⚠️ ML Model not trained yet"
- ✅ Train Model tab is accessible
- ✅ Training button is functional
- ✅ Training completes quickly (mock training)
- ✅ Success message: "✅ Model trained successfully"
- ✅ Status updates to "✅ ML Model is trained and ready!"
- ✅ Balloons animation appears

### Test 16: Single CPU Usage Prediction
**Objective:** Test individual prediction functionality

**Steps:**
1. After model training, go to "Make Predictions" sub-tab
2. In the "Single Step Prediction" section
3. Use the default sequence or enter: `45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7`
4. Click "Predict Next Value" button
5. Examine the results and chart

**Expected Results:**
- ✅ Text area accepts comma-separated values
- ✅ Prediction button is functional
- ✅ Success message shows predicted value (e.g., "🎯 Next CPU Usage Prediction: **XX.XX%**")
- ✅ Chart displays with historical data in blue
- ✅ Prediction point appears in red
- ✅ Chart has proper labels and title

### Test 17: Multi-Step Prediction
**Objective:** Test multiple prediction functionality

**Steps:**
1. Still in Make Predictions sub-tab
2. In "Multi-Step Prediction" section
3. Set "Number of future steps" to 7
4. Click "Predict Multiple Steps" button
5. Review the results

**Expected Results:**
- ✅ Number input accepts values between 1-20
- ✅ Prediction executes successfully
- ✅ Shows success message with step count
- ✅ Lists individual predictions (Step 1: XX.XX%, Step 2: XX.XX%, etc.)
- ✅ Chart shows historical data and dotted line for predictions
- ✅ Predictions extend beyond historical data

### Test 18: Model Information
**Objective:** Verify model information display

**Steps:**
1. Click on "Model Info" sub-tab
2. Review the displayed information
3. Check model architecture details
4. Look for use cases section

**Expected Results:**
- ✅ Model architecture information is displayed
- ✅ Shows LSTM configuration details
- ✅ Training parameters are listed
- ✅ Use cases section lists practical applications
- ✅ Information is well-formatted and readable

---

## ❌ Error Scenarios Testing

### Test 19: Empty Workload Simulation
**Objective:** Test behavior when no workloads are loaded

**Steps:**
1. Clear all workloads (Configuration → Workloads → Clear All Workloads)
2. Navigate to Simulation tab
3. Try to start simulation
4. Check the response

**Expected Results:**
- ✅ Warning message appears: "Please load workloads first!"
- ✅ Simulation does not proceed
- ✅ No error crashes or broken functionality

### Test 20: No Scheduler Selection
**Objective:** Test simulation with no schedulers selected

**Steps:**
1. Load workloads
2. Go to Simulation tab
3. Uncheck all scheduler options
4. Click "🚀 Start Simulation"

**Expected Results:**
- ✅ Warning message: "Please select at least one scheduler!"
- ✅ Simulation button remains functional
- ✅ No system errors occur

### Test 21: Invalid ML Input
**Objective:** Test ML predictions with invalid data

**Steps:**
1. Ensure ML model is trained
2. Go to ML Predictions → Make Predictions
3. Enter invalid sequence: `45, 52, 48` (only 3 values instead of 12)
4. Click "Predict Next Value"

**Expected Results:**
- ✅ Error message: "Please provide exactly 12 values"
- ✅ No prediction is made
- ✅ System remains stable

### Test 22: API Connection Loss Simulation
**Objective:** Test frontend behavior when API is unavailable

**Steps:**
1. Stop the API server (Ctrl+C in Terminal 1)
2. Refresh the Streamlit page
3. Try to perform any API-dependent action
4. Check the API status indicator

**Expected Results:**
- ✅ API status shows "🔴 API Offline"
- ✅ Error messages appear for API-dependent actions
- ✅ Frontend remains functional for basic navigation
- ✅ Helpful error messages guide user to restart API

---

## 📊 Performance Testing

### Test 23: Large Workload Handling
**Objective:** Test system with maximum workloads

**Steps:**
1. Go to Configuration → Workloads
2. Select "Generate Random"
3. Set "Number of Workloads" to 100 (maximum)
4. Generate workloads
5. Run simulation with all schedulers
6. Check performance and results

**Expected Results:**
- ✅ System handles 100 workloads without crashing
- ✅ Generation completes within reasonable time (< 30 seconds)
- ✅ Simulation executes successfully
- ✅ Results display correctly
- ✅ UI remains responsive

### Test 24: Rapid Navigation Testing
**Objective:** Test UI responsiveness with quick navigation

**Steps:**
1. Rapidly click between different tabs:
   - Dashboard → Configuration → Simulation → Results → ML Predictions
2. Repeat this cycle 5 times quickly
3. Check for any broken states or errors

**Expected Results:**
- ✅ All tabs load consistently
- ✅ No loading errors or broken layouts
- ✅ Data persists correctly across navigation
- ✅ UI remains stable and responsive

---

## 📝 Test Results Documentation

### Test Completion Checklist

Copy this checklist and mark completed tests:

#### Environment Setup
- [ ] API server started successfully
- [ ] Streamlit frontend launched
- [ ] Browser can access application
- [ ] API connection status shows green

#### Core Functionality  
- [ ] Dashboard displays correctly
- [ ] Sample data loads successfully
- [ ] Provider configuration works
- [ ] VM configuration displays properly
- [ ] Manual workload entry functions
- [ ] Random workload generation works
- [ ] Workload clearing works

#### Simulation Features
- [ ] Multi-scheduler simulation runs
- [ ] Single scheduler simulation works  
- [ ] Results display correctly
- [ ] Charts and metrics accurate
- [ ] Detailed logs accessible

#### ML Features
- [ ] Model training completes
- [ ] Single predictions work
- [ ] Multi-step predictions function
- [ ] Model information displays

#### Error Handling
- [ ] Empty workload warnings work
- [ ] Invalid input handling proper
- [ ] API offline handling graceful
- [ ] No unexpected crashes occur

#### Performance
- [ ] Large workload handling adequate
- [ ] Navigation remains responsive
- [ ] Memory usage reasonable
- [ ] No significant delays

### Issues Encountered

**Format for documenting issues:**

```
Issue #1: [Brief Description]
Steps to Reproduce: 
1. [Step 1]
2. [Step 2] 
3. [Result]

Expected Behavior: [What should happen]
Actual Behavior: [What actually happened]
Severity: [High/Medium/Low]
Browser: [Chrome/Firefox/Safari/Edge]
Status: [Open/Resolved]
```

### Overall Test Summary

**Total Tests Performed:** ___/24
**Passed:** ___
**Failed:** ___
**Issues Found:** ___
**Test Duration:** ___ minutes
**Tester:** _______________
**Date:** _______________

### Recommendations

After completing all tests, note:
1. **Critical Issues:** Any problems that prevent core functionality
2. **Usability Improvements:** Suggestions for better user experience  
3. **Performance Notes:** Any slowness or responsiveness issues
4. **Feature Requests:** Additional functionality that would be helpful

### Next Steps

Based on test results:
- [ ] Report critical bugs to development team
- [ ] Document all issues in issue tracking system
- [ ] Schedule follow-up testing after fixes
- [ ] Plan user acceptance testing
- [ ] Prepare production deployment checklist

---

## 🎯 Quick Test Scenarios

### 5-Minute Quick Test
For rapid verification:
1. Load sample workloads (30 seconds)
2. Run simulation with all schedulers (2 minutes)
3. Check results display (1 minute)
4. Train ML model and make one prediction (1.5 minutes)

### 15-Minute Comprehensive Test  
For thorough verification:
1. Complete environment setup verification
2. Test all workload input methods
3. Run multiple simulation scenarios
4. Test all ML prediction features
5. Verify error handling scenarios

### 30-Minute Full Test Suite
For complete system validation:
- Execute all 24 test cases
- Document results thoroughly
- Test edge cases and performance limits
- Generate comprehensive test report

---

**Happy Testing! 🚀**

*This guide ensures your AI Cloud Scheduler works perfectly for end users.*