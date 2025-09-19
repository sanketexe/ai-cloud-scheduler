import time
import os
from test_utils import APITestClient, TestResult, create_test_ml_data
from test_config import ML_TRAINING_TIMEOUT

def test_ml_endpoints():
    """Test ML prediction endpoints"""
    client = APITestClient()
    result = TestResult()
    
    print("🧪 Testing ML Endpoints")
    print("-" * 40)
    
    # Test 1: Check model status (initially should be not trained)
    try:
        response = client.get("/api/ml/model-status")
        if response.status_code == 200:
            data = response.json()
            if "model_trained" in data:
                result.add_pass("Get model status")
            else:
                result.add_fail("Get model status", "Missing model_trained field")
        else:
            result.add_fail("Get model status", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Get model status", str(e))
    
    # Test 2: Upload training data
    try:
        csv_file = create_test_ml_data()
        with open(csv_file, 'rb') as f:
            files = {'file': ('test_cpu_data.csv', f, 'text/csv')}
            response = client.post("/api/ml/upload-training-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "rows" in data and data["rows"] == 100:
                result.add_pass("Upload ML training data")
            else:
                result.add_fail("Upload ML training data", "Incorrect number of rows uploaded")
        else:
            result.add_fail("Upload ML training data", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Upload ML training data", str(e))
    
    # Test 3: Train model (this takes time)
    try:
        print("   ⏳ Training ML model (this may take a few minutes)...")
        response = client.post("/api/ml/train", timeout=ML_TRAINING_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                result.add_pass("Train ML model")
            else:
                result.add_fail("Train ML model", "Training did not complete successfully")
        else:
            result.add_fail("Train ML model", f"HTTP {response.status_code}")
    except Exception as e:
        result.add_fail("Train ML model", str(e))
    
    # Test 4: Single prediction (only if model is trained)
    try:
        # Check if model is trained first
        status_response = client.get("/api/ml/model-status")
        if status_response.status_code == 200 and status_response.json().get("model_trained"):
            # Test prediction with 12 values
            sequence_data = [45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7]
            response = client.post("/api/ml/predict", json=sequence_data)
            
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data and isinstance(data["prediction"], (int, float)):
                    result.add_pass("Single ML prediction")
                else:
                    result.add_fail("Single ML prediction", "Invalid prediction response")
            else:
                result.add_fail("Single ML prediction", f"HTTP {response.status_code}")
        else:
            result.add_fail("Single ML prediction", "Model not trained for prediction test")
    except Exception as e:
        result.add_fail("Single ML prediction", str(e))
    
    # Test 5: Multiple predictions
    try:
        status_response = client.get("/api/ml/model-status")
        if status_response.status_code == 200 and status_response.json().get("model_trained"):
            sequence_data = [45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7]
            response = client.post("/api/ml/predict-multiple", json=sequence_data, params={"steps": 5})
            
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == 5:
                    result.add_pass("Multiple ML predictions")
                else:
                    result.add_fail("Multiple ML predictions", "Incorrect number of predictions")
            else:
                result.add_fail("Multiple ML predictions", f"HTTP {response.status_code}")
        else:
            result.add_fail("Multiple ML predictions", "Model not trained for prediction test")
    except Exception as e:
        result.add_fail("Multiple ML predictions", str(e))
    
    # Test 6: Invalid prediction input
    try:
        # Test with wrong number of sequence values (should be 12)
        invalid_sequence = [45.2, 52.3, 48.1]  # Only 3 values instead of 12
        response = client.post("/api/ml/predict", json=invalid_sequence)
        
        if response.status_code == 400:  # Bad request expected
            result.add_pass("Reject invalid prediction input")
        else:
            result.add_fail("Reject invalid prediction input", "Should have rejected invalid input")
    except Exception as e:
        result.add_fail("Reject invalid prediction input", str(e))
    
    return result

if __name__ == "__main__":
    from test_utils import wait_for_server
    
    if not wait_for_server():
        print("❌ Cannot run tests - API server is not running")
        exit(1)
    
    result = test_ml_endpoints()
    result.summary()