# AI Cloud Scheduler - Complete Testing Guide

## 📖 Overview

This comprehensive guide covers all testing procedures for the AI Cloud Scheduler backend system. Whether you're a developer, tester, or system administrator, this guide will help you understand, run, and troubleshoot all aspects of the testing framework.

---

## 🎯 What We're Testing

The AI Cloud Scheduler is a backend system that:
- **Manages cloud providers** (AWS, GCP, Azure)
- **Schedules workloads** across virtual machines
- **Uses ML algorithms** for intelligent predictions
- **Provides REST API** for integration with frontends

### Core Components Tested

GitHub Copilot
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │ Cloud │ │ Workload │ │ Machine │ │ Providers │◄──►│ Scheduler │◄──►│ Learning │ │ (AWS/GCP/ │ │ (Algorithms) │ │ (Predictions) │ │ Azure) │ │ │ │ │ └─────────────────┘ └─────────────────┘ └─────────────────┘ ▲ ▲ ▲ │ │ │ └────────────────────────┼────────────────────────┘ │ ┌─────────────────┐ │ REST API │ │ (FastAPI) │ │ │ └─────────────────┘

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Environment Check
```bash
# Check your location
pwd
# Should show: /e/clg/TS_AI_CLOUD_SCHEDULER or similar

# Check Python version
python --version
# Should be: Python 3.10.x or higher

# Check virtual environment
where python
# Should point to: .venv\Scripts\python.exe

# Terminal 1: Start backend server
python api.py

# Wait for this message:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# 🚀 API Server initialized and ready!

# Terminal 2: Run basic test
python quick_test.py

# Expected output:
🔥 Quick Smoke Test for AI Cloud Scheduler
--------------------------------------------------
✅ API server is ready
✅ 1/6 - API server responding
✅ 2/6 - API documentation accessible
✅ 3/6 - Provider endpoints working
✅ 4/6 - VM endpoints working
✅ 5/6 - Workload endpoints working
✅ 6/6 - Basic simulation working

🏁 Quick Test Results: 6/6 passed
🎉 All essential features are working!
```

---

## 📂 Project Structure

```
TS_AI_CLOUD_SCHEDULER/
├── 📄 api.py                       # Main backend server
├── 📄 quick_test.py                # Quick health check (START HERE)
├── 📄 run_tests.py                 # Comprehensive test runner
├── 📄 fix_tests.py                 # Automated problem solver
├── 📄 simple_test.py               # All-in-one test file
├── 📄 README_TESTING.md            # This guide
├── 
├── 📁 tests/                       # Main test directory
│   ├── 📄 test_config.py          # Test settings
│   ├── 📄 test_utils.py           # Common test functions
│   ├── 📄 test_basic_api.py       # API endpoint tests
│   ├── 📄 test_workloads.py       # Workload management tests
│   ├── 📄 test_simulation.py      # Scheduling algorithm tests
│   ├── 📄 test_ml.py              # Machine learning tests
│   ├── 📄 test_performance.py     # Speed and load tests
│   ├── 📄 test_configuration.py   # Configuration management tests
│   └── 📁 test_data/              # Sample data files
│       ├── 📄 sample_workloads.csv
│       └── 📄 sample_cpu_data.csv
└──
```

### 8. Configuration Tests (`test_configuration.py`)
**Purpose**: Test system configuration management and settings

**Tests performed**:
- ✅ View all system configurations
- ✅ Show configuration overview with `show_configuration`
- ✅ Get specific configuration categories
- ✅ Update configuration settings
- ✅ Export/import configurations
- ✅ Validate configuration data
- ✅ Handle invalid categories and data

**Configuration Categories**:
- 🔧 **API**: Server settings, ports, debugging
- 📊 **Scheduler**: Algorithm settings, limits, timeouts
- ☁️ **Providers**: Cloud provider costs and settings
- 🤖 **ML**: Machine learning model configuration
- ⚡ **Performance**: Caching, rate limiting, concurrency
- 📝 **Logging**: Log levels, file settings
- 🔒 **Security**: API keys, HTTPS, CORS settings

**Endpoints Tested**:
- `GET /api/config` - Get all configurations
- `GET /api/config/show` - Show configuration overview
- `GET /api/config/{category}` - Get specific category
- `POST /api/config/{category}` - Update configuration
- `GET /api/config/export` - Export all configurations
- `POST /api/config/import` - Import configurations


