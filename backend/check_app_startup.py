import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Attempting to import app from main.py...")
try:
    from main import app
    print("Successfully imported app from main.py")
except ImportError as e:
    print(f"ImportError during startup check: {e}")
except Exception as e:
    print(f"Error during startup check: {e}")
    import traceback
    traceback.print_exc()
