"""
FinOps Platform - Development Server Starter
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        import pandas
        import numpy
        print("✅ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Install dependencies with: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting FinOps Platform Backend...")
    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        sys.exit(1)

def main():
    """Main function to start the development environment"""
    print("🔧 FinOps Platform Development Starter")
    print("=" * 50)
    
    # Check Python dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🎯 Starting the platform...")
    print("\n📱 Frontend: http://localhost:3000 (start with: cd frontend && npm start)")
    print("🔗 API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    # Start the backend
    start_backend()

if __name__ == "__main__":
    main()