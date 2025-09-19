#!/usr/bin/env python3
"""
Quick fix script for test issues
"""

import os
import re

def fix_run_tests_typo():
    """Fix the typo in run_tests.py"""
    file_path = "run_tests.py"
    
    if os.path.exists(file_path):
        try:
            # Try UTF-8 encoding first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Try with UTF-8 and ignore errors
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                # Fallback to latin-1 which can read any byte sequence
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        # Fix the typo
        content = content.replace('test_basic_endpoinnts', 'test_basic_endpoints')
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed typo in run_tests.py")
    else:
        print("❌ run_tests.py not found")

def check_and_fix_test_files():
    """Check and fix test file locations"""
    tests_dir = "tests"
    root_test_files = []
    
    # Find test files in root directory
    for file in os.listdir("."):
        if file.startswith("test_") and file.endswith(".py"):
            root_test_files.append(file)
    
    if root_test_files:
        print(f"⚠️  Found test files in root directory: {root_test_files}")
        
        # Create tests directory if it doesn't exist
        if not os.path.exists(tests_dir):
            os.makedirs(tests_dir)
            print(f"✅ Created {tests_dir} directory")
        
        # Move test files to tests directory
        for file in root_test_files:
            if not os.path.exists(os.path.join(tests_dir, file)):
                try:
                    # Read with proper encoding
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Write to tests directory
                    with open(os.path.join(tests_dir, file), 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Moved {file} to tests/ directory")
                except Exception as e:
                    print(f"❌ Error moving {file}: {e}")
            else:
                print(f"ℹ️  {file} already exists in tests/ directory")

def create_basic_test_file():
    """Create basic test file if it doesn't exist"""
    test_file = os.path.join("tests", "test_basic_api.py")
    
    if not os.path.exists(test_file):
        content = '''"""
Basic API tests for AI Cloud Scheduler
"""

import requests
from test_utils import APITestClient, TestResult

def test_basic_endpoints():
    """Test basic API endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Basic API Endpoints")
    print("-" * 40)
    
    # Test 1: Root endpoint
    try:
        response = client.get("/")
        if response.status_code == 200:
            data = response.json()
            if "message" in data:
                result.add_pass("Root endpoint (/)")
            else:
                result.add_fail("Root endpoint (/)", "Missing message field")
        else:
            result.add_fail("Root endpoint (/)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Root endpoint (/)", str(e))
    
    # Test 2: Health check
    try:
        response = client.get("/health")
        if response.status_code == 200:
            result.add_pass("Health check (/health)")
        else:
            result.add_fail("Health check (/health)", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Health check (/health)", str(e))
    
    return result

def test_provider_endpoints():
    """Test provider endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("\\n🧪 Testing Provider Endpoints")
    print("-" * 40)
    
    # Test 1: Get default providers
    try:
        response = client.get("/api/providers/default")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 3:
                result.add_pass("Get default providers")
            else:
                result.add_fail("Get default providers", "Invalid response format")
        else:
            result.add_fail("Get default providers", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get default providers", str(e))
    
    return result

def test_vm_endpoints():
    """Test VM endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("\\n🧪 Testing VM Endpoints")
    print("-" * 40)
    
    # Test 1: Get default VMs
    try:
        response = client.get("/api/vms/default")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) >= 4:
                result.add_pass("Get default VMs")
            else:
                result.add_fail("Get default VMs", "Invalid response format")
        else:
            result.add_fail("Get default VMs", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get default VMs", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        exit(1)
    
    # Run tests
    results = []
    results.append(test_basic_endpoints())
    results.append(test_provider_endpoints())
    results.append(test_vm_endpoints())
    
    # Summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    print(f"\\n{'='*50}")
    print(f"BASIC API TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
'''
        
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Created test_basic_api.py")
        except Exception as e:
            print(f"❌ Error creating test_basic_api.py: {e}")

def create_optimized_api():
    """Create optimized API file"""
    print("✅ API optimizations ready to apply")
    print("   Please update your api.py file with the optimizations from the previous message")

if __name__ == "__main__":
    print("🔧 Applying test fixes...")
    
    # Fix encoding and file structure issues
    check_and_fix_test_files()
    
    # Create basic test file if missing
    create_basic_test_file()
    
    # Fix the typo in run_tests.py
    fix_run_tests_typo()
    
    # API optimization reminder
    create_optimized_api()
    
    print("\n🚀 Fixes applied! Now run:")
    print("   1. Check if your API server is running: python api.py")
    print("   2. Run quick test: python quick_test.py")
    print("   3. Run full tests: python run_tests.py")