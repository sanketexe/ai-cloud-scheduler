import time
from test_utils import APITestClient, TestResult, create_test_workload_csv
from test_config import SIMULATION_TIMEOUT

def test_simulation_endpoints():
    """Test simulation endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Simulation Endpoints")
    print("-" * 40)
    
    # Setup: Get sample workloads
    workloads_response = client.get("/api/workloads/sample")
    if workloads_response.status_code != 200:
        result.add_fail("Setup", "Cannot get sample workloads")
        return result
    
    workloads = workloads_response.json()
    
    # Test 1: Random Scheduler Simulation
    try:
        simulation_data = {
            "scheduler_type": "random",
            "workloads": workloads[:5]  # Use first 5 workloads
        }
        response = client.post("/api/simulation/run", json=simulation_data, timeout=SIMULATION_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["logs", "summary", "scheduler_type"]
            if all(field in data for field in required_fields):
                if len(data["logs"]) == 5:  # Should have 5 log entries
                    result.add_pass("Random scheduler simulation")
                else:
                    result.add_fail("Random scheduler simulation", "Incorrect number of log entries")
            else:
                result.add_fail("Random scheduler simulation", "Missing required response fields")
        else:
            result.add_fail("Random scheduler simulation", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Random scheduler simulation", str(e))
    
    # Test 2: Lowest Cost Scheduler Simulation
    try:
        simulation_data = {
            "scheduler_type": "lowest_cost",
            "workloads": workloads[:3]
        }
        response = client.post("/api/simulation/run", json=simulation_data, timeout=SIMULATION_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("scheduler_type") == "lowest_cost":
                result.add_pass("Lowest cost scheduler simulation")
            else:
                result.add_fail("Lowest cost scheduler simulation", "Incorrect scheduler type in response")
        else:
            result.add_fail("Lowest cost scheduler simulation", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Lowest cost scheduler simulation", str(e))
    
    # Test 3: Round Robin Scheduler Simulation
    try:
        simulation_data = {
            "scheduler_type": "round_robin",
            "workloads": workloads[:4]
        }
        response = client.post("/api/simulation/run", json=simulation_data, timeout=SIMULATION_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("scheduler_type") == "round_robin":
                result.add_pass("Round robin scheduler simulation")
            else:
                result.add_fail("Round robin scheduler simulation", "Incorrect scheduler type in response")
        else:
            result.add_fail("Round robin scheduler simulation", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Round robin scheduler simulation", str(e))
    
    # Test 4: Compare multiple schedulers
    try:
        comparison_data = {
            "scheduler_types": ["random", "lowest_cost", "round_robin"],
            "workloads": workloads[:6]
        }
        response = client.post("/api/simulation/compare", json=comparison_data, timeout=SIMULATION_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) == 3:
                result.add_pass("Multiple scheduler comparison")
            else:
                result.add_fail("Multiple scheduler comparison", "Incorrect comparison results")
        else:
            result.add_fail("Multiple scheduler comparison", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Multiple scheduler comparison", str(e))
    
    # Test 5: Invalid scheduler type
    try:
        invalid_data = {
            "scheduler_type": "invalid_scheduler",
            "workloads": workloads[:2]
        }
        response = client.post("/api/simulation/run", json=invalid_data)
        
        if response.status_code == 400:  # Bad request expected
            result.add_pass("Reject invalid scheduler type")
        else:
            result.add_fail("Reject invalid scheduler type", "Should have rejected invalid scheduler")
    except Exception as e:
        result.add_fail("Reject invalid scheduler type", str(e))
    
    # Test 6: Empty workloads list
    try:
        empty_data = {
            "scheduler_type": "random",
            "workloads": []
        }
        response = client.post("/api/simulation/run", json=empty_data)
        
        if response.status_code == 400:  # Bad request expected
            result.add_pass("Reject empty workloads")
        else:
            result.add_fail("Reject empty workloads", "Should have rejected empty workloads")
    except Exception as e:
        result.add_fail("Reject empty workloads", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        exit(1)
    
    result = test_simulation_endpoints()
    result.summary()