#!/usr/bin/env python3
"""
Development Task System Startup Script
Starts Redis, Celery worker, and Celery beat for local development
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_redis():
    """Check if Redis is running"""
    try:
        result = subprocess.run(['redis-cli', 'ping'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and 'PONG' in result.stdout
    except:
        return False

def start_redis():
    """Start Redis server"""
    print("Starting Redis server...")
    try:
        # Try to start Redis
        redis_process = subprocess.Popen(['redis-server', '--daemonize', 'yes'])
        time.sleep(2)  # Give Redis time to start
        
        if check_redis():
            print("✓ Redis server started successfully")
            return True
        else:
            print("✗ Failed to start Redis server")
            return False
    except FileNotFoundError:
        print("✗ Redis not found. Please install Redis:")
        print("  - macOS: brew install redis")
        print("  - Ubuntu: sudo apt-get install redis-server")
        print("  - Windows: Download from https://redis.io/download")
        return False

def start_celery_worker():
    """Start Celery worker"""
    print("Starting Celery worker...")
    backend_dir = Path(__file__).parent / "backend"
    
    env = os.environ.copy()
    env['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    env['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    env['PYTHONPATH'] = str(backend_dir)
    
    worker_process = subprocess.Popen([
        sys.executable, 'celery_worker.py'
    ], cwd=backend_dir, env=env)
    
    return worker_process

def start_celery_beat():
    """Start Celery beat scheduler"""
    print("Starting Celery beat scheduler...")
    backend_dir = Path(__file__).parent / "backend"
    
    env = os.environ.copy()
    env['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    env['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    env['PYTHONPATH'] = str(backend_dir)
    
    beat_process = subprocess.Popen([
        sys.executable, 'celery_beat.py'
    ], cwd=backend_dir, env=env)
    
    return beat_process

def main():
    """Main function"""
    print("FinOps Platform - Development Task System Startup")
    print("=" * 50)
    
    # Check if Redis is already running
    if not check_redis():
        if not start_redis():
            print("Cannot start task system without Redis")
            return 1
    else:
        print("✓ Redis is already running")
    
    # Start Celery processes
    processes = []
    
    try:
        # Start worker
        worker_process = start_celery_worker()
        processes.append(('worker', worker_process))
        time.sleep(2)
        
        # Start beat scheduler
        beat_process = start_celery_beat()
        processes.append(('beat', beat_process))
        time.sleep(2)
        
        print("\n✓ Task system started successfully!")
        print("\nRunning processes:")
        for name, process in processes:
            print(f"  - Celery {name}: PID {process.pid}")
        
        print("\nTask system is ready. Press Ctrl+C to stop all processes.")
        print("\nYou can now:")
        print("  - Monitor tasks: python backend/monitor_tasks.py active")
        print("  - Check workers: python backend/monitor_tasks.py workers")
        print("  - View task status: python backend/monitor_tasks.py status <task_id>")
        
        # Wait for interrupt
        try:
            while True:
                # Check if processes are still running
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"\n✗ Celery {name} process stopped unexpectedly")
                        return 1
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down task system...")
            
            # Terminate processes gracefully
            for name, process in processes:
                print(f"Stopping Celery {name}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    print(f"✓ Celery {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"Force killing Celery {name}...")
                    process.kill()
                    process.wait()
            
            print("✓ Task system stopped")
            return 0
    
    except Exception as e:
        print(f"\n✗ Error starting task system: {e}")
        
        # Clean up any started processes
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())