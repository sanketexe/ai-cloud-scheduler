import requests
import time
import csv
import os
from test_config import API_BASE_URL, TEST_DATA_DIR, REQUEST_TIMEOUT

class APITestClient:
    """Utility class for API testing"""
    
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get(self, endpoint, **kwargs):
        """Make GET request"""
        url = f"{self.base_url}{endpoint}"
        # Remove any duplicate timeout parameters
        if 'timeout' not in kwargs:
            kwargs['timeout'] = REQUEST_TIMEOUT
        return self.session.get(url, **kwargs)
    
    def post(self, endpoint, **kwargs):
        """Make POST request"""
        url = f"{self.base_url}{endpoint}"
        # Remove any duplicate timeout parameters
        if 'timeout' not in kwargs:
            kwargs['timeout'] = REQUEST_TIMEOUT
        return self.session.post(url, **kwargs)
    
    def is_server_running(self):
        """Check if API server is running"""
        try:
            response = self.get("/")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

def create_test_workload_csv():
    """Create test workload CSV file"""
    test_file = os.path.join(TEST_DATA_DIR, "test_workloads.csv")
    
    workloads = [
        {"workload_id": 1, "cpu_required": 2, "memory_required_gb": 4},
        {"workload_id": 2, "cpu_required": 1, "memory_required_gb": 2},
        {"workload_id": 3, "cpu_required": 4, "memory_required_gb": 8},
        {"workload_id": 4, "cpu_required": 3, "memory_required_gb": 6},
        {"workload_id": 5, "cpu_required": 2, "memory_required_gb": 4},
        {"workload_id": 6, "cpu_required": 1, "memory_required_gb": 1},
        {"workload_id": 7, "cpu_required": 2, "memory_required_gb": 3},
        {"workload_id": 8, "cpu_required": 1, "memory_required_gb": 2},
    ]
    
    with open(test_file, 'w', newline='') as csvfile:
        fieldnames = ['workload_id', 'cpu_required', 'memory_required_gb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(workloads)
    
    return test_file

def create_test_ml_data():
    """Create test ML training data CSV file"""
    test_file = os.path.join(TEST_DATA_DIR, "test_cpu_data.csv")
    
    # Generate sample CPU usage data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create 100 data points with realistic CPU usage patterns
    timestamps = []
    cpu_usage = []
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    base_usage = 50
    
    for i in range(100):
        timestamps.append(start_time + timedelta(minutes=5*i))
        # Add some realistic variation with daily patterns
        hour_factor = np.sin(2 * np.pi * i / 288) * 10  # Daily cycle
        noise = np.random.normal(0, 5)  # Random variation
        usage = max(10, min(90, base_usage + hour_factor + noise))
        cpu_usage.append(round(usage, 1))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage
    })
    
    df.to_csv(test_file, index=False)
    return test_file

def wait_for_server(max_attempts=30, delay=1):
    """Wait for API server to be ready"""
    client = APITestClient()
    
    for attempt in range(max_attempts):
        if client.is_server_running():
            print("✅ API server is ready")
            return True
        print(f"⏳ Waiting for API server... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(delay)
    
    print("❌ API server is not responding")
    return False

class TestResult:
    """Class to store test results"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%" if total > 0 else "No tests run")
        
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        
        return self.failed == 0