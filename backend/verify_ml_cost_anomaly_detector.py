import sys
import os

# Add the current directory to sys.path so we can import 'app'
sys.path.append(os.getcwd())

print("Testing imports for ml_cost_anomaly_detector...")
try:
    from app.ml.ml_cost_anomaly_detector import CostAnomalyDetector
    print("Success: ml_cost_anomaly_detector imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
