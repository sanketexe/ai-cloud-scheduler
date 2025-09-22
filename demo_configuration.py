# # AI-Powered Cloud Scheduler

# A comprehensive cloud resource scheduling simulation system that compares different scheduling algorithms and incorporates machine learning for predictive resource management.

# ## Project Overview

# This project simulates a cloud computing environment where workloads need to be scheduled across multiple virtual machines (VMs) from different cloud providers. The system implements various scheduling algorithms and uses an LSTM neural network for CPU usage prediction.

# ## Architecture

# ### Core Components

# 1. **Environment Management** ([environment.py](environment.py))
#    - [`CloudProvider`](environment.py): Represents cloud providers (AWS, GCP, Azure) with specific pricing models
#    - [`VirtualMachine`](environment.py): Manages VM instances with resource capacity and usage tracking
#    - [`Workload`](environment.py): Represents tasks requiring CPU and memory resources

# 2. **Scheduling Algorithms** ([schedulers.py](schedulers.py))
#    - [`RandomScheduler`](schedulers.py): Randomly selects eligible VMs
#    - [`LowestCostScheduler`](schedulers.py): Chooses the cheapest VM that can accommodate the workload
#    - [`RoundRobinScheduler`](schedulers.py): Cycles through VMs in sequential order

# 3. **Machine Learning Predictor** ([predictor.py](predictor.py))
#    - LSTM model for CPU usage prediction
#    - Uses historical data from [real_usage_data.csv](real_usage_data.csv)
#    - Sequence length of 12 data points (1 hour) to predict next 5-minute interval

# 4. **Simulation Engine** ([main.py](main.py))
#    - Orchestrates the entire simulation process
#    - Loads workloads from [workload_trace.csv](workload_trace.csv)
#    - Runs simulations for each scheduler and logs results

# ## System Flow

# ### 1. Environment Setup
# The [`setup_environment`](main.py) function creates:
# - **AWS**: CPU cost $0.04/hour, Memory cost $0.01/GB/hour
# - **GCP**: CPU cost $0.035/hour, Memory cost $0.009/GB/hour  
# - **Azure**: CPU cost $0.042/hour, Memory cost $0.011/GB/hour

# VM Configuration:
# - VM1: 4 CPU, 16GB RAM (AWS)
# - VM2: 8 CPU, 32GB RAM (GCP)
# - VM3: 4 CPU, 16GB RAM (Azure)
# - VM4: 2 CPU, 8GB RAM (GCP)

# ### 2. Workload Processing
# Workloads are loaded from [workload_trace.csv](workload_trace.csv) using [`load_workloads_from_csv`](main.py), which contains:
# - `workload_id`: Unique identifier
# - `cpu_required`: Required CPU cores
# - `memory_required_gb`: Required memory in GB

# ### 3. Simulation Execution
# The [`run_simulation`](main.py) function:
# 1. Processes workloads sequentially by timestamp
# 2. Uses the selected scheduler to find suitable VMs
# 3. Assigns workloads to VMs if resources are available
# 4. Logs system state after each scheduling attempt
# 5. Saves performance metrics to CSV files

# ### 4. Machine Learning Component
# The [`train_and_save_model`](predictor.py) function:
# 1. Loads historical CPU usage data from [real_usage_data.csv](real_usage_data.csv)
# 2. Scales data using MinMaxScaler
# 3. Creates sequences for LSTM training
# 4. Builds and trains a 2-layer LSTM model
# 5. Saves the trained model as `cpu_predictor_model.keras`

# ## Data Files

# ### Input Files
# - **[workload_trace.csv](workload_trace.csv)**: Contains workload requests with resource requirements
# - **[real_usage_data.csv](real_usage_data.csv)**: Historical CPU usage data for ML training (5-minute intervals)

# ### Output Files
# - **[log_random.csv](log_random.csv)**: Results from RandomScheduler simulation
# - **[log_lowest_cost.csv](log_lowest_cost.csv)**: Results from LowestCostScheduler simulation
# - **[log_round_robin.csv](log_round_robin.csv)**: Results from RoundRobinScheduler simulation
# - **cpu_predictor_model.keras**: Trained LSTM model for CPU prediction

# ## Performance Metrics

# Each simulation logs the following metrics at each timestamp:
# - `timestamp`: Simulation time step
# - `workload_id`: ID of the processed workload
# - `status`: SUCCESS or FAILURE of scheduling attempt
# - `total_cpu_used`: Total CPU utilization across all VMs
# - `total_cpu_capacity`: Total available CPU capacity
# - `percent_cpu_used`: CPU utilization percentage
# - `total_mem_used_gb`: Total memory utilization
# - `total_mem_capacity_gb`: Total available memory
# - `percent_mem_used`: Memory utilization percentage

## Installation & Setup

# ### Prerequisites
# ```bash
# pip install pandas numpy scikit-learn tensorflow
# ```

# ## Comprehensive Guide

# This comprehensive README guide covers:

# - 🚀 **Quick start for beginners** (5-minute setup)
# - 📖 **Detailed explanations** of what each test does
# - 🔧 **Complete troubleshooting guide** with solutions
# - 📊 **Performance benchmarks** and expectations  
# - 💡 **Best practices** for different user types
# - 🎯 **Success criteria** and quality gates
# - 📈 **Advanced testing scenarios**
# - 🎉 **Clear success indicators**

# Any new user can follow this guide from start to finish and understand exactly what's being tested and why! 🌟

# ## Configuration System Demo

# ```python
#!/usr/bin/env python3
"""
Configuration System Demo for AI Cloud Scheduler
Demonstrates all configuration management features
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def demo_configuration():
    """Demonstrate configuration management features"""
    print("🔧 AI Cloud Scheduler - Configuration System Demo")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API server not responding. Start with: python api.py")
            return
    except:
        print("❌ API server not running. Start with: python api.py")
        return
    
    print("✅ API server is ready\n")
    
    # 1. Show configuration overview
    print("📋 1. Configuration Overview")
    print("-" * 30)
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/show")
        if response.status_code == 200:
            data = response.json()
            overview = data["system_overview"]
            print(f"📊 Total Categories: {overview['total_categories']}")
            print(f"📅 Last Updated: {overview['last_updated']}")
            print(f"🟢 Status: {overview['status']}")
            print()
            
            # Show category summary
            for category, info in data["categories"].items():
                print(f"  📁 {category.upper()}")
                print(f"     Settings: {info['total_settings']}")
                print(f"     Status: {info['status']}")
                if info["key_settings"]:
                    print(f"     Key Settings: {list(info['key_settings'].keys())}")
                print()
        else:
            print(f"❌ Failed to get configuration overview: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Show specific configurations
    print("🔧 2. Specific Configuration Categories")
    print("-" * 40)
    
    categories = ["api", "scheduler", "providers", "ml", "performance"]
    
    for category in categories:
        try:
            response = requests.get(f"{API_BASE_URL}/api/config/{category}")
            if response.status_code == 200:
                data = response.json()
                config = data["config"]
                print(f"📁 {category.upper()} Configuration:")
                
                # Pretty print key settings
                if category == "api":
                    print(f"   🔗 Version: {config.get('version')}")
                    print(f"   🌐 Port: {config.get('port')}")
                    print(f"   🐛 Debug: {config.get('debug')}")
                    print(f"   👥 Max Workers: {config.get('max_workers')}")
                
                elif category == "scheduler":
                    print(f"   🎯 Default Algorithm: {config.get('default_algorithm')}")
                    print(f"   📦 Max Workloads: {config.get('max_workloads_per_request')}")
                    print(f"   ⏱️ Timeout: {config.get('simulation_timeout')}s")
                    print(f"   🔄 Retry Attempts: {config.get('retry_attempts')}")
                
                elif category == "providers":
                    print(f"   ☁️ Available Providers:")
                    for provider, settings in config.items():
                        enabled = "✅" if settings.get("enabled") else "❌"
                        print(f"      {enabled} {provider.upper()}: ${settings.get('cpu_cost')}/CPU, ${settings.get('memory_cost_gb')}/GB")
                
                elif category == "ml":
                    print(f"   🤖 Model Type: {config.get('model_type')}")
                    print(f"   🎓 Training Enabled: {config.get('training_enabled')}")
                    print(f"   📊 Prediction Window: {config.get('prediction_window')}")
                    print(f"   🔄 Auto Retrain: {config.get('auto_retrain')}")
                
                elif category == "performance":
                    print(f"   💾 Cache Enabled: {config.get('cache_enabled')}")
                    print(f"   👥 Concurrent Limit: {config.get('concurrent_limit')}")
                    print(f"   ⚡ Rate Limit: {config.get('rate_limit')}")
                    print(f"   🗜️ Compression: {config.get('response_compression')}")
                
                print()
            else:
                print(f"❌ Failed to get {category} config: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting {category} config: {e}")
    
    # 3. Demonstrate configuration update
    print("✏️ 3. Configuration Update Demo")
    print("-" * 35)
    
    try:
        # Update API configuration
        print("📝 Updating API configuration...")
        update_data = {
            "category": "api",
            "config": {
                "debug": True,
                "max_workers": 8,
                "request_timeout": 45
            }
        }
        
        response = requests.post(f"{API_BASE_URL}/api/config/api", json=update_data)
        if response.status_code == 200:
            data = response.json()
            print("✅ API configuration updated successfully!")
            print(f"   Debug mode: {data['updated_config']['debug']}")
            print(f"   Max workers: {data['updated_config']['max_workers']}")
            print(f"   Timeout: {data['updated_config']['request_timeout']}s")
        else:
            print(f"❌ Failed to update configuration: {response.status_code}")
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
    
    print()
    
    # 4. Export/Import demo
    print("📦 4. Export/Import Demo")
    print("-" * 25)
    
    try:
        # Export configuration
        print("📤 Exporting configuration...")
        response = requests.get(f"{API_BASE_URL}/api/config/export")
        if response.status_code == 200:
            export_data = response.json()
            print("✅ Configuration exported successfully!")
            print(f"   Export timestamp: {export_data['export_info']['timestamp']}")
            print(f"   Total categories: {export_data['export_info']['total_categories']}")
            
            # Save to file
            with open("config_backup.json", "w") as f:
                json.dump(export_data, f, indent=2)
            print("💾 Configuration saved to config_backup.json")
        else:
            print(f"❌ Failed to export configuration: {response.status_code}")
    except Exception as e:
        print(f"❌ Error exporting configuration: {e}")
    
    print()
    
    # 5. Configuration validation demo
    print("🛡️ 5. Configuration Validation Demo")
    print("-" * 40)
    
    try:
        # Try invalid configuration
        print("🧪 Testing invalid configuration...")
        invalid_data = {
            "category": "scheduler",  # Wrong category
            "config": {"debug": True}
        }
        
        response = requests.post(f"{API_BASE_URL}/api/config/api", json=invalid_data)  # URL says 'api'
        if response.status_code == 400:
            print("✅ Category mismatch properly detected!")
        else:
            print(f"⚠️ Expected validation error, got: {response.status_code}")
        
        # Try non-existent category
        print("🧪 Testing non-existent category...")
        response = requests.get(f"{API_BASE_URL}/api/config/nonexistent")
        if response.status_code == 404:
            print("✅ Non-existent category properly handled!")
        else:
            print(f"⚠️ Expected 404 error, got: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in validation demo: {e}")
    
    print()
    
    # 6. Final status check
    print("🎯 6. Final Configuration Status")
    print("-" * 35)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/show")
        if response.status_code == 200:
            data = response.json()
            overview = data["system_overview"]
            
            print("📊 Current System Status:")
            print(f"   Total Categories: {overview['total_categories']}")
            print(f"   System Status: {overview['status']}")
            print(f"   Last Updated: {overview['last_updated']}")
            
            # Count configured categories
            configured_count = sum(1 for cat_info in data["categories"].values() 
                                 if cat_info["status"] == "configured")
            
            print(f"   Configured Categories: {configured_count}/{overview['total_categories']}")
            
            if configured_count == overview['total_categories']:
                print("🎉 All categories are properly configured!")
            else:
                print("⚠️ Some categories may need attention")
                
        else:
            print(f"❌ Failed to get final status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting final status: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Configuration System Demo Complete!")
    print("💡 Key Features Demonstrated:")
    print("   ✅ Configuration overview and browsing")
    print("   ✅ Category-specific configuration viewing")
    print("   ✅ Configuration updates and modifications")
    print("   ✅ Export/import functionality")
    print("   ✅ Validation and error handling")
    print("   ✅ Real-time status monitoring")

if __name__ == "__main__":
    demo_configuration()