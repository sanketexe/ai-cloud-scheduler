import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    print("Attempting to import app.services.startup_migration...")
    import app.services.startup_migration
    print("Success: app.services.startup_migration imported.")
except ImportError as e:
    print(f"Error importing app.services.startup_migration: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
