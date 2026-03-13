import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    print("Attempting to import backend.app.services.startup_migration...")
    import backend.app.services.startup_migration
    print("Success: backend.app.services.startup_migration imported.")
except ImportError as e:
    print(f"Error importing backend.app.services.startup_migration: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
