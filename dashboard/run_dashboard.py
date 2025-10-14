#!/usr/bin/env python3
"""
Dashboard startup script
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description="Cloud Intelligence Platform Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", 
                       help="Backend API URL")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["API_BASE_URL"] = args.api_url
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent
    app_path = dashboard_dir / "app.py"
    
    # Check if app.py exists
    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"Starting Cloud Intelligence Platform Dashboard...")
    print(f"Dashboard URL: http://{args.host}:{args.port}")
    print(f"Backend API: {args.api_url}")
    print(f"Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, cwd=dashboard_dir)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()