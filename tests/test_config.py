import os
import sys

# Add the parent directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# Create test data directory if it doesn't exist
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Test timeouts
REQUEST_TIMEOUT = 30
SIMULATION_TIMEOUT = 60
ML_TRAINING_TIMEOUT = 300