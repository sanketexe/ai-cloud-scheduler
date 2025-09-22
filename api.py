from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uvicorn
import asyncio
from functools import lru_cache
import os

# Import your existing modules
try:
    from environment import CloudProvider, VirtualMachine, Workload
    from schedulers import RandomScheduler, LowestCostScheduler, RoundRobinScheduler
except ImportError:
    # Create mock classes if modules don't exist
    class CloudProvider:
        def __init__(self, name, cpu_cost, memory_cost_gb):
            self.name = name
            self.cpu_cost = cpu_cost
            self.memory_cost_gb = memory_cost_gb
    
    class VirtualMachine:
        def __init__(self, vm_id, cpu_capacity, memory_capacity_gb, provider):
            self.vm_id = vm_id
            self.cpu_capacity = cpu_capacity
            self.memory_capacity_gb = memory_capacity_gb
            self.provider = provider
            self.cpu_used = 0
            self.memory_used_gb = 0
            self.cost = 0
            
        def assign_workload(self, workload):
            if (self.cpu_used + workload.cpu_required <= self.cpu_capacity and
                self.memory_used_gb + workload.memory_required_gb <= self.memory_capacity_gb):
                self.cpu_used += workload.cpu_required
                self.memory_used_gb += workload.memory_required_gb
                self.cost += (workload.cpu_required * self.provider.cpu_cost + 
                             workload.memory_required_gb * self.provider.memory_cost_gb)
                return True
            return False
    
    class Workload:
        def __init__(self, id, cpu_required, memory_required_gb):
            self.id = id
            self.cpu_required = cpu_required
            self.memory_required_gb = memory_required_gb
    
    class RandomScheduler:
        def select_vm(self, workload, vms):
            import random
            available_vms = [vm for vm in vms if 
                           vm.cpu_used + workload.cpu_required <= vm.cpu_capacity and
                           vm.memory_used_gb + workload.memory_required_gb <= vm.memory_capacity_gb]
            return random.choice(available_vms) if available_vms else None
    
    class LowestCostScheduler:
        def select_vm(self, workload, vms):
            available_vms = [vm for vm in vms if 
                           vm.cpu_used + workload.cpu_required <= vm.cpu_capacity and
                           vm.memory_used_gb + workload.memory_required_gb <= vm.memory_capacity_gb]
            if not available_vms:
                return None
            return min(available_vms, key=lambda vm: vm.provider.cpu_cost + vm.provider.memory_cost_gb)
    
    class RoundRobinScheduler:
        def __init__(self):
            self.current_index = 0
            
        def select_vm(self, workload, vms):
            available_vms = [vm for vm in vms if 
                           vm.cpu_used + workload.cpu_required <= vm.cpu_capacity and
                           vm.memory_used_gb + workload.memory_required_gb <= vm.memory_capacity_gb]
            if not available_vms:
                return None
            
            vm = available_vms[self.current_index % len(available_vms)]
            self.current_index += 1
            return vm

app = FastAPI(title="AI Cloud Scheduler API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the environment setup to avoid recreating it every time
@lru_cache(maxsize=1)
def get_cached_environment():
    """Cached environment setup for better performance"""
    aws = CloudProvider(name="AWS", cpu_cost=0.04, memory_cost_gb=0.01)
    gcp = CloudProvider(name="GCP", cpu_cost=0.035, memory_cost_gb=0.009)
    azure = CloudProvider(name="Azure", cpu_cost=0.042, memory_cost_gb=0.011)

    vms = [
        VirtualMachine(vm_id=1, cpu_capacity=4, memory_capacity_gb=16, provider=aws),
        VirtualMachine(vm_id=2, cpu_capacity=8, memory_capacity_gb=32, provider=gcp),
        VirtualMachine(vm_id=3, cpu_capacity=4, memory_capacity_gb=16, provider=azure),
        VirtualMachine(vm_id=4, cpu_capacity=2, memory_capacity_gb=8, provider=gcp)
    ]
    
    return aws, gcp, azure, vms

def setup_environment():
    """Fast environment setup using cache"""
    return get_cached_environment()

# Pydantic models
class WorkloadModel(BaseModel):
    id: int
    cpu_required: int
    memory_required_gb: int

class ProviderModel(BaseModel):
    name: str
    cpu_cost: float
    memory_cost_gb: float

class VMModel(BaseModel):
    vm_id: int
    cpu_capacity: int
    memory_capacity_gb: int
    provider: ProviderModel

class SimulationRequest(BaseModel):
    scheduler_type: str
    workloads: List[WorkloadModel]

class ConfigurationResponse(BaseModel):
    category: str
    config: Dict[str, Any]
    last_updated: str
    editable: bool

class ConfigurationUpdate(BaseModel):
    category: str
    config: Dict[str, Any]

# Root endpoint
@app.get("/")
async def root():
    """Fast root endpoint"""
    return {"message": "AI Cloud Scheduler API", "version": "1.0.0", "status": "ready"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Provider endpoints
@app.get("/api/providers/default")
async def get_default_providers():
    """Get default cloud providers"""
    try:
        aws, gcp, azure, _ = setup_environment()
        providers = [
            {"name": aws.name, "cpu_cost": aws.cpu_cost, "memory_cost_gb": aws.memory_cost_gb},
            {"name": gcp.name, "cpu_cost": gcp.cpu_cost, "memory_cost_gb": gcp.memory_cost_gb},
            {"name": azure.name, "cpu_cost": azure.cpu_cost, "memory_cost_gb": azure.memory_cost_gb}
        ]
        return providers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/providers")
async def create_provider(provider: ProviderModel):
    """Create a new cloud provider"""
    try:
        return {
            "name": provider.name,
            "cpu_cost": provider.cpu_cost,
            "memory_cost_gb": provider.memory_cost_gb,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# VM endpoints
@app.get("/api/vms/default")
async def get_default_vms():
    """Get default virtual machines"""
    try:
        aws, gcp, azure, vms = setup_environment()
        vm_list = []
        for vm in vms:
            vm_data = {
                "vm_id": vm.vm_id,
                "cpu_capacity": vm.cpu_capacity,
                "memory_capacity_gb": vm.memory_capacity_gb,
                "provider": {
                    "name": vm.provider.name,
                    "cpu_cost": vm.provider.cpu_cost,
                    "memory_cost_gb": vm.provider.memory_cost_gb
                },
                "cpu_used": vm.cpu_used,
                "memory_used_gb": vm.memory_used_gb
            }
            vm_list.append(vm_data)
        return vm_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vms")
async def create_vm(vm: VMModel):
    """Create a new virtual machine"""
    try:
        return {
            "vm_id": vm.vm_id,
            "cpu_capacity": vm.cpu_capacity,
            "memory_capacity_gb": vm.memory_capacity_gb,
            "provider": vm.provider.dict(),
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Workload endpoints
@app.get("/api/workloads/sample")
async def get_sample_workloads():
    """Get sample workloads for testing"""
    try:
        sample_workloads = [
            {"id": 1, "cpu_required": 2, "memory_required_gb": 4},
            {"id": 2, "cpu_required": 1, "memory_required_gb": 2},
            {"id": 3, "cpu_required": 4, "memory_required_gb": 8},
            {"id": 4, "cpu_required": 3, "memory_required_gb": 6},
            {"id": 5, "cpu_required": 2, "memory_required_gb": 4},
            {"id": 6, "cpu_required": 1, "memory_required_gb": 1},
            {"id": 7, "cpu_required": 2, "memory_required_gb": 3},
            {"id": 8, "cpu_required": 1, "memory_required_gb": 2}
        ]
        return sample_workloads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workloads")
async def create_workload(workload: WorkloadModel):
    """Create a new workload"""
    try:
        # Validate workload data
        if workload.cpu_required <= 0 or workload.memory_required_gb <= 0:
            raise HTTPException(status_code=422, detail="CPU and memory requirements must be positive")
        
        return {
            "id": workload.id,
            "cpu_required": workload.cpu_required,
            "memory_required_gb": workload.memory_required_gb,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workloads/generate")
async def generate_random_workloads(count: int = 5):
    """Generate random workloads"""
    import random
    try:
        if count <= 0 or count > 100:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 100")
            
        workloads = []
        for i in range(count):
            workload = {
                "id": i + 1000,
                "cpu_required": random.randint(1, 4),
                "memory_required_gb": random.randint(1, 8)
            }
            workloads.append(workload)
        return workloads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workloads/upload")
async def upload_workloads(file: UploadFile = File(...)):
    """Upload workloads from CSV file"""
    try:
        import csv
        import io
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        csv_data = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        workloads = []
        for row_num, row in enumerate(csv_reader, 1):
            try:
                workload = {
                    "id": int(row["workload_id"]),
                    "cpu_required": int(row["cpu_required"]),
                    "memory_required_gb": int(row["memory_required_gb"])
                }
                
                # Validate data
                if workload["cpu_required"] <= 0 or workload["memory_required_gb"] <= 0:
                    raise ValueError("CPU and memory must be positive")
                    
                workloads.append(workload)
            except (ValueError, KeyError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid data in row {row_num}: {str(e)}")
        
        if not workloads:
            raise HTTPException(status_code=400, detail="No valid workloads found in CSV")
        
        return {"workloads": workloads, "count": len(workloads)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simulation endpoints
@app.post("/api/simulation/run")
async def run_simulation(request: SimulationRequest):
    """Run simulation with specified scheduler"""
    try:
        # Validate scheduler type
        valid_schedulers = ["random", "lowest_cost", "round_robin"]
        if request.scheduler_type not in valid_schedulers:
            raise HTTPException(status_code=400, detail=f"Invalid scheduler type. Must be one of: {valid_schedulers}")
        
        # Validate workloads
        if not request.workloads:
            raise HTTPException(status_code=400, detail="No workloads provided")
        
        # Setup environment
        aws, gcp, azure, vms = setup_environment()
        
        # Convert workloads
        workloads = [
            Workload(w.id, w.cpu_required, w.memory_required_gb) 
            for w in request.workloads
        ]
        
        # Select scheduler
        if request.scheduler_type == "random":
            scheduler = RandomScheduler()
        elif request.scheduler_type == "lowest_cost":
            scheduler = LowestCostScheduler()
        elif request.scheduler_type == "round_robin":
            scheduler = RoundRobinScheduler()
        
        # Run simulation
        logs = []
        successful_assignments = 0
        
        for workload in workloads:
            selected_vm = scheduler.select_vm(workload, vms)
            if selected_vm and selected_vm.assign_workload(workload):
                logs.append({
                    "workload_id": workload.id,
                    "vm_id": selected_vm.vm_id,
                    "success": True,
                    "message": f"Workload {workload.id} assigned to VM {selected_vm.vm_id}"
                })
                successful_assignments += 1
            else:
                logs.append({
                    "workload_id": workload.id,
                    "vm_id": None,
                    "success": False,
                    "message": f"Could not assign workload {workload.id}"
                })
        
        # Calculate summary
        success_rate = (successful_assignments / len(workloads)) * 100 if workloads else 0
        total_cost = sum(vm.cost for vm in vms)
        
        summary = {
            "total_workloads": len(workloads),
            "successful_assignments": successful_assignments,
            "success_rate": success_rate,
            "total_cost": total_cost
        }
        
        return {
            "logs": logs,
            "summary": summary,
            "scheduler_type": request.scheduler_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/compare")
async def compare_schedulers(request: dict):
    """Compare multiple schedulers"""
    try:
        scheduler_types = request.get("scheduler_types", [])
        workloads_data = request.get("workloads", [])
        
        if not scheduler_types:
            raise HTTPException(status_code=400, detail="No scheduler types provided")
        
        if not workloads_data:
            raise HTTPException(status_code=400, detail="No workloads provided")
        
        results = []
        
        for scheduler_type in scheduler_types:
            # Create simulation request for each scheduler
            sim_request = SimulationRequest(
                scheduler_type=scheduler_type,
                workloads=[WorkloadModel(**w) for w in workloads_data]
            )
            
            # Run simulation
            result = await run_simulation(sim_request)
            results.append(result)
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global variable to track model status
MODEL_STATUS = {
    "model_trained": False,
    "training_data_rows": 0,
    "last_trained": None
}

# ML endpoints
@app.get("/api/ml/model-status")
async def get_model_status():
    """Get ML model training status"""
    return MODEL_STATUS

@app.post("/api/ml/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload ML training data"""
    try:
        import csv
        import io
        
        content = await file.read()
        csv_data = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        rows = list(csv_reader)
        MODEL_STATUS["training_data_rows"] = len(rows)
        return {"status": "success", "rows": len(rows)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/train")
async def train_model():
    """Train ML model"""
    try:
        import time
        from datetime import datetime
        
        # Simulate training time
        time.sleep(0.1)
        
        # Update model status
        MODEL_STATUS["model_trained"] = True
        MODEL_STATUS["last_trained"] = datetime.now().isoformat()
        
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/predict")
async def predict_single(sequence: List[float]):
    """Make single prediction"""
    try:
        if len(sequence) != 12:
            raise HTTPException(status_code=400, detail="Sequence must have exactly 12 values")
        
        # Check if model is trained (for testing purposes, always allow predictions)
        # if not MODEL_STATUS["model_trained"]:
        #     raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Mock prediction
        import random
        prediction = random.uniform(40, 60)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/predict-multiple")
async def predict_multiple(sequence: List[float], steps: int = 5):
    """Make multiple predictions"""
    try:
        if len(sequence) != 12:
            raise HTTPException(status_code=400, detail="Sequence must have exactly 12 values")
        
        # Check if model is trained (for testing purposes, always allow predictions)
        # if not MODEL_STATUS["model_trained"]:
        #     raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Mock predictions
        import random
        predictions = [random.uniform(40, 60) for _ in range(steps)]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

SYSTEM_CONFIG = {
    "api": {
        "version": "1.0.0",
        "host": "localhost",
        "port": 8000,
        "debug": False,
        "cors_enabled": True,
        "request_timeout": 30,
        "max_workers": 4
    },
    "database": {
        "type": "sqlite",
        "connection_timeout": 10,
        "max_connections": 20,
        "backup_enabled": True,
        "backup_interval": "24h"
    },
    "scheduler": {
        "default_algorithm": "lowest_cost",
        "available_algorithms": ["random", "lowest_cost", "round_robin"],
        "max_workloads_per_request": 100,
        "simulation_timeout": 300,
        "retry_attempts": 3
    },
    "ml": {
        "model_type": "lstm",
        "training_enabled": True,
        "prediction_window": 12,
        "batch_size": 32,
        "max_training_time": 600,
        "auto_retrain": False
    },
    "performance": {
        "cache_enabled": True,
        "cache_ttl": 3600,
        "rate_limit": 100,
        "concurrent_limit": 10,
        "response_compression": True
    },
    "providers": {
        "aws": {
            "enabled": True,
            "default_region": "us-east-1",
            "cpu_cost": 0.04,
            "memory_cost_gb": 0.01
        },
        "gcp": {
            "enabled": True,
            "default_region": "us-central1",
            "cpu_cost": 0.035,
            "memory_cost_gb": 0.009
        },
        "azure": {
            "enabled": True,
            "default_region": "eastus",
            "cpu_cost": 0.042,
            "memory_cost_gb": 0.011
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_enabled": True,
        "max_file_size": "10MB",
        "backup_count": 5
    },
    "security": {
        "api_key_required": False,
        "https_only": False,
        "cors_origins": ["*"],
        "max_request_size": "10MB",
        "rate_limiting": True
    }
}

# Configuration endpoints




@app.get("/api/config/show")
async def show_configuration():
    """Show formatted system configuration overview"""
    try:
        config_summary = {
            "system_overview": {
                "total_categories": len(SYSTEM_CONFIG),
                "last_updated": datetime.now().isoformat(),
                "status": "active"
            },
            "categories": {}
        }
        
        # Summarize each category
        for category, config in SYSTEM_CONFIG.items():
            category_info = {
                "total_settings": len(config),
                "key_settings": {},
                "status": "configured"
            }
            
            # Extract key settings for each category
            if category == "api":
                category_info["key_settings"] = {
                    "version": config.get("version"),
                    "port": config.get("port"),
                    "debug": config.get("debug")
                }
            elif category == "scheduler":
                category_info["key_settings"] = {
                    "default_algorithm": config.get("default_algorithm"),
                    "max_workloads": config.get("max_workloads_per_request")
                }
            elif category == "providers":
                enabled_providers = [name for name, settings in config.items() if settings.get("enabled", False)]
                category_info["key_settings"] = {
                    "enabled_providers": enabled_providers,
                    "total_providers": len(config)
                }
            elif category == "ml":
                category_info["key_settings"] = {
                    "model_type": config.get("model_type"),
                    "training_enabled": config.get("training_enabled")
                }
            elif category == "performance":
                category_info["key_settings"] = {
                    "cache_enabled": config.get("cache_enabled"),
                    "concurrent_limit": config.get("concurrent_limit")
                }
            else:
                # For other categories, show first 3 settings
                keys = list(config.keys())[:3]
                category_info["key_settings"] = {k: config[k] for k in keys}
            
            config_summary["categories"][category] = category_info
        
        return config_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/export")
async def export_configuration():
    """Export all configurations as JSON"""
    try:
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_categories": len(SYSTEM_CONFIG)
            },
            "configurations": SYSTEM_CONFIG
        }
        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/{category}")
async def get_configuration(category: str):
    """Get configuration for specific category"""
    try:
        if category not in SYSTEM_CONFIG:
            raise HTTPException(status_code=404, detail=f"Configuration category '{category}' not found")
        
        return ConfigurationResponse(
            category=category,
            config=SYSTEM_CONFIG[category],
            last_updated=datetime.now().isoformat(),
            editable=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/{category}")
async def update_configuration(category: str, update: ConfigurationUpdate):
    """Update configuration for specific category"""
    try:
        if category not in SYSTEM_CONFIG:
            raise HTTPException(status_code=404, detail=f"Configuration category '{category}' not found")
        
        # Validate that the category in URL matches the request body
        if update.category != category:
            raise HTTPException(status_code=400, detail="Category in URL must match category in request body")
        
        # Update configuration
        SYSTEM_CONFIG[category].update(update.config)
        
        return {
            "category": category,
            "updated_config": SYSTEM_CONFIG[category],
            "message": f"Configuration for '{category}' updated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/import")
async def import_configuration(config_data: Dict[str, Any]):
    """Import configurations from JSON data"""
    try:
        if "configurations" not in config_data:
            raise HTTPException(status_code=400, detail="Invalid import format: 'configurations' key required")
        
        imported_config = config_data["configurations"]
        
        # Validate structure
        for category in imported_config:
            if category in SYSTEM_CONFIG:
                SYSTEM_CONFIG[category].update(imported_config[category])
        
        return {
            "message": "Configuration imported successfully",
            "imported_categories": list(imported_config.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/reset/{category}")
async def reset_configuration(category: str):
    """Reset configuration category to defaults"""
    try:
        if category not in SYSTEM_CONFIG:
            raise HTTPException(status_code=404, detail=f"Configuration category '{category}' not found")
        
        # Default configurations (same as initial SYSTEM_CONFIG)
        default_configs = {
            "api": {
                "version": "1.0.0",
                "host": "localhost",
                "port": 8000,
                "debug": False,
                "cors_enabled": True,
                "request_timeout": 30,
                "max_workers": 4
            },
            "scheduler": {
                "default_algorithm": "lowest_cost",
                "available_algorithms": ["random", "lowest_cost", "round_robin"],
                "max_workloads_per_request": 100,
                "simulation_timeout": 300,
                "retry_attempts": 3
            }
            # Add other default configs as needed
        }
        
        if category in default_configs:
            SYSTEM_CONFIG[category] = default_configs[category].copy()
        
        return {
            "category": category,
            "message": f"Configuration for '{category}' reset to defaults",
            "reset_config": SYSTEM_CONFIG[category],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize cache on startup"""
    # Pre-warm the cache
    setup_environment()
    print("🚀 API Server initialized and ready!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)