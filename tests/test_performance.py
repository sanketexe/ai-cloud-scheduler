import time
import concurrent.futures
from test_utils import APITestClient, TestResult

def test_concurrent_requests():
    """Test API performance under concurrent load"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing Performance & Concurrency")
    print("-" * 40)
    
    # Test 1: Response time for basic endpoints
    try:
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        if response_time < 1.0:  # Should respond within 1 second
            result.add_pass(f"Response time ({response_time:.3f}s)")
        else:
            result.add_fail("Response time", f"Too slow: {response_time:.3f}s")
    except Exception as e:
        result.add_fail("Response time", str(e))
    
    # Test 2: Concurrent requests
    try:
        def make_request():
            return client.get("/api/providers/default").status_code == 200
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_count = sum(results)
        if success_count == 10:
            result.add_pass("Concurrent requests (10 threads)")
        else:
            result.add_fail("Concurrent requests", f"Only {success_count}/10 succeeded")
    except Exception as e:
        result.add_fail("Concurrent requests", str(e))
    
    # Test 3: Large workload simulation
    try:
        # Generate a large number of workloads
        large_workloads = []
        for i in range(50):
            large_workloads.append({
                "id": i + 1000,
                "cpu_required": (i % 4) + 1,
                "memory_required_gb": (i % 8) + 1
            })
        
        start_time = time.time()
        simulation_data = {
            "scheduler_type": "random",
            "workloads": large_workloads
        }
        response = client.post("/api/simulation/run", json=simulation_data, timeout=60)
        end_time = time.time()
        
        simulation_time = end_time - start_time
        if response.status_code == 200 and simulation_time < 30:  # Should complete within 30 seconds
            result.add_pass(f"Large simulation (50 workloads, {simulation_time:.1f}s)")
        else:
            result.add_fail("Large simulation", f"Failed or too slow: {simulation_time:.1f}s")
    except Exception as e:
        result.add_fail("Large simulation", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        exit(1)
    
    result = test_concurrent_requests()
    result.summary()