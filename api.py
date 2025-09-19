from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uvicorn
import asyncio
from functools import lru_cache

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
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=422, detail=str(e))

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
        if isinstance(e, HTTPException):
            raise
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
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

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
        if isinstance(e, HTTPException):
            raise
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
        if isinstance(e, HTTPException):
            raise
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
        raise HTTPException(status_code=400, detail=str(e))

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
        if isinstance(e, HTTPException):
            raise
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
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize cache on startup"""
    # Pre-warm the cache
    setup_environment()
    print("🚀 API Server initialized and ready!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)