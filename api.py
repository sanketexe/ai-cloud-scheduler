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

    # Add new intelligent scheduler classes
    class IntelligentScheduler:
        def select_vm(self, workload, vms):
            available_vms = [vm for vm in vms if 
                           vm.cpu_used + workload.cpu_required <= vm.cpu_capacity and
                           vm.memory_used_gb + workload.memory_required_gb <= vm.memory_capacity_gb]
            if not available_vms:
                return None
            
            # Score VMs based on resource utilization and cost
            def vm_score(vm):
                cpu_util = (vm.cpu_used + workload.cpu_required) / vm.cpu_capacity
                mem_util = (vm.memory_used_gb + workload.memory_required_gb) / vm.memory_capacity_gb
                cost = workload.cpu_required * vm.provider.cpu_cost + workload.memory_required_gb * vm.provider.memory_cost_gb
                
                # Prefer balanced utilization and lower cost
                utilization_score = min(cpu_util, mem_util) * 0.7  # Favor balanced utilization
                cost_score = (1.0 / (cost + 0.001)) * 0.3  # Favor lower cost
                
                return utilization_score + cost_score
            
            return max(available_vms, key=vm_score)
    
    class HybridScheduler:
        def __init__(self):
            self.schedulers = [
                RandomScheduler(),
                LowestCostScheduler(),
                RoundRobinScheduler()
            ]
            self.current_scheduler_index = 0
        
        def select_vm(self, workload, vms):
            # Rotate between different scheduling strategies
            scheduler = self.schedulers[self.current_scheduler_index % len(self.schedulers)]
            self.current_scheduler_index += 1
            return scheduler.select_vm(workload, vms)

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
    """Upload workloads from CSV file - accepts any column format"""
    try:
        import csv
        import io
        import re
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        csv_data = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        # Get column headers
        fieldnames = csv_reader.fieldnames
        if not fieldnames:
            raise HTTPException(status_code=400, detail="CSV file appears to be empty or malformed")
        
        # Function to find matching column by pattern
        def find_column(patterns, columns):
            """Find column that matches any of the given patterns (case-insensitive)"""
            for pattern in patterns:
                for col in columns:
                    if re.search(pattern, col.lower().strip()):
                        return col
            return None
        
        # Define flexible patterns for each required field
        id_patterns = [
            r'.*id.*',           # anything with 'id'
            r'.*workload.*',     # anything with 'workload'
            r'.*task.*',         # anything with 'task'
            r'.*job.*',          # anything with 'job'
            r'.*number.*',       # anything with 'number'
            r'^id$',             # exact 'id'
            r'^#.*'              # starts with #
        ]
        
        cpu_patterns = [
            r'.*cpu.*',          # anything with 'cpu'
            r'.*core.*',         # anything with 'core'
            r'.*processor.*',    # anything with 'processor'
            r'.*vcpu.*',         # anything with 'vcpu'
            r'.*compute.*'       # anything with 'compute'
        ]
        
        memory_patterns = [
            r'.*memory.*',       # anything with 'memory'
            r'.*mem.*',          # anything with 'mem'
            r'.*ram.*',          # anything with 'ram'
            r'.*gb.*',           # anything with 'gb'
            r'.*storage.*'       # anything with 'storage'
        ]
        
        # Find matching columns
        id_column = find_column(id_patterns, fieldnames)
        cpu_column = find_column(cpu_patterns, fieldnames)
        memory_column = find_column(memory_patterns, fieldnames)
        
        # If automatic detection fails, use positional mapping
        if not id_column and len(fieldnames) >= 1:
            id_column = fieldnames[0]
            
        if not cpu_column and len(fieldnames) >= 2:
            cpu_column = fieldnames[1]
            
        if not memory_column and len(fieldnames) >= 3:
            memory_column = fieldnames[2]
        
        # Validation message for user
        column_mapping = {
            "ID Column": id_column,
            "CPU Column": cpu_column, 
            "Memory Column": memory_column
        }
        
        missing_columns = [key for key, value in column_mapping.items() if not value]
        if missing_columns:
            available_columns = ", ".join(fieldnames)
            raise HTTPException(
                status_code=400, 
                detail=f"Could not identify columns for: {', '.join(missing_columns)}. "
                       f"Available columns: {available_columns}. "
                       f"Please ensure your CSV has columns for workload ID, CPU requirements, and memory requirements."
            )
        
        workloads = []
        errors = []
        
        for row_num, row in enumerate(csv_reader, 1):
            try:
                # Extract values with flexible parsing
                id_value = row.get(id_column, '').strip()
                cpu_value = row.get(cpu_column, '').strip()
                memory_value = row.get(memory_column, '').strip()
                
                # Skip empty rows
                if not any([id_value, cpu_value, memory_value]):
                    continue
                
                # Parse ID - handle various formats
                try:
                    workload_id = int(float(id_value))  # Handle decimals like "1.0"
                except (ValueError, TypeError):
                    # If ID is not numeric, generate one
                    workload_id = 1000 + len(workloads)
                
                # Parse CPU - extract numbers from strings
                try:
                    cpu_required = int(float(re.findall(r'[\d.]+', cpu_value)[0]))
                    if cpu_required <= 0:
                        raise ValueError("CPU must be positive")
                except (ValueError, IndexError, TypeError):
                    errors.append(f"Row {row_num}: Invalid CPU value '{cpu_value}'")
                    continue
                
                # Parse Memory - extract numbers from strings
                try:
                    memory_required_gb = int(float(re.findall(r'[\d.]+', memory_value)[0]))
                    if memory_required_gb <= 0:
                        raise ValueError("Memory must be positive")
                except (ValueError, IndexError, TypeError):
                    errors.append(f"Row {row_num}: Invalid memory value '{memory_value}'")
                    continue
                
                workload = {
                    "id": workload_id,
                    "cpu_required": cpu_required,
                    "memory_required_gb": memory_required_gb
                }
                
                workloads.append(workload)
                
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
                continue
        
        if not workloads and errors:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid workloads found. Errors: {'; '.join(errors[:5])}"  # Show first 5 errors
            )
        
        # Prepare response
        response = {
            "workloads": workloads,
            "count": len(workloads),
            "column_mapping": column_mapping,
            "total_rows_processed": row_num if 'row_num' in locals() else 0
        }
        
        # Include warnings if there were errors
        if errors:
            response["warnings"] = errors[:10]  # Include first 10 errors as warnings
            response["errors_count"] = len(errors)
        
        return response
        
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please save as UTF-8 CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# Add a helper endpoint to preview CSV structure
@app.post("/api/workloads/preview")
async def preview_csv_structure(file: UploadFile = File(...)):
    """Preview CSV file structure and suggested column mapping"""
    try:
        import csv
        import io
        import re
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read first few rows
        content = await file.read()
        csv_data = content.decode('utf-8')
        lines = csv_data.split('\n')[:6]  # First 5 data rows + header
        
        # Parse headers and sample data
        csv_reader = csv.DictReader(io.StringIO('\n'.join(lines)))
        fieldnames = csv_reader.fieldnames or []
        sample_rows = list(csv_reader)[:5]
        
        # Function to suggest column mapping
        def suggest_mapping(columns):
            suggestions = {}
            
            # ID column suggestions
            for col in columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in ['id', 'workload', 'task', 'job', 'number']):
                    suggestions['id_column'] = col
                    break
            
            # CPU column suggestions
            for col in columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in ['cpu', 'core', 'processor', 'vcpu', 'compute']):
                    suggestions['cpu_column'] = col
                    break
            
            # Memory column suggestions
            for col in columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in ['memory', 'mem', 'ram', 'gb', 'storage']):
                    suggestions['memory_column'] = col
                    break
            
            # Fallback to positional mapping
            if 'id_column' not in suggestions and len(columns) >= 1:
                suggestions['id_column'] = columns[0]
            if 'cpu_column' not in suggestions and len(columns) >= 2:
                suggestions['cpu_column'] = columns[1]
            if 'memory_column' not in suggestions and len(columns) >= 3:
                suggestions['memory_column'] = columns[2]
                
            return suggestions
        
        suggestions = suggest_mapping(fieldnames)
        
        return {
            "filename": file.filename,
            "columns": fieldnames,
            "sample_rows": sample_rows,
            "suggested_mapping": suggestions,
            "total_columns": len(fieldnames),
            "preview_rows": len(sample_rows),
            "mapping_confidence": len([k for k in suggestions.values() if k]) / 3 * 100
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please save as UTF-8 CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing file: {str(e)}")


# Add endpoint for manual column mapping
@app.post("/api/workloads/upload-with-mapping")
async def upload_workloads_with_mapping(
    file: UploadFile = File(...),
    id_column: str = None,
    cpu_column: str = None, 
    memory_column: str = None
):
    """Upload workloads with manual column mapping"""
    try:
        import csv
        import io
        import re
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        csv_data = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        fieldnames = csv_reader.fieldnames or []
        
        # Validate provided column names
        missing_columns = []
        if id_column and id_column not in fieldnames:
            missing_columns.append(f"ID column '{id_column}'")
        if cpu_column and cpu_column not in fieldnames:
            missing_columns.append(f"CPU column '{cpu_column}'")
        if memory_column and memory_column not in fieldnames:
            missing_columns.append(f"Memory column '{memory_column}'")
            
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found: {', '.join(missing_columns)}. Available: {', '.join(fieldnames)}"
            )
        
        workloads = []
        errors = []
        
        for row_num, row in enumerate(csv_reader, 1):
            try:
                # Extract values using provided mapping
                id_value = row.get(id_column, '').strip() if id_column else str(1000 + len(workloads))
                cpu_value = row.get(cpu_column, '').strip()
                memory_value = row.get(memory_column, '').strip()
                
                # Skip empty rows
                if not cpu_value and not memory_value:
                    continue
                
                # Parse values
                try:
                    workload_id = int(float(id_value))
                except (ValueError, TypeError):
                    workload_id = 1000 + len(workloads)
                
                try:
                    cpu_required = int(float(re.findall(r'[\d.]+', cpu_value)[0]))
                    if cpu_required <= 0:
                        raise ValueError("CPU must be positive")
                except (ValueError, IndexError, TypeError):
                    errors.append(f"Row {row_num}: Invalid CPU value '{cpu_value}'")
                    continue
                
                try:
                    memory_required_gb = int(float(re.findall(r'[\d.]+', memory_value)[0]))
                    if memory_required_gb <= 0:
                        raise ValueError("Memory must be positive")
                except (ValueError, IndexError, TypeError):
                    errors.append(f"Row {row_num}: Invalid memory value '{memory_value}'")
                    continue
                
                workload = {
                    "id": workload_id,
                    "cpu_required": cpu_required,
                    "memory_required_gb": memory_required_gb
                }
                
                workloads.append(workload)
                
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
                continue
        
        if not workloads:
            raise HTTPException(status_code=400, detail="No valid workloads found in CSV")
        
        response = {
            "workloads": workloads,
            "count": len(workloads),
            "column_mapping": {
                "id_column": id_column,
                "cpu_column": cpu_column,
                "memory_column": memory_column
            }
        }
        
        if errors:
            response["warnings"] = errors[:10]
            response["errors_count"] = len(errors)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Simulation endpoints
@app.post("/api/simulation/run")
async def run_simulation(request: SimulationRequest):
    """Run simulation with specified scheduler - Enhanced with detailed metrics"""
    try:
        # Validate scheduler type
        valid_schedulers = ["random", "lowest_cost", "round_robin", "intelligent", "hybrid"]
        if request.scheduler_type not in valid_schedulers:
            raise HTTPException(status_code=400, detail=f"Invalid scheduler type. Must be one of: {valid_schedulers}")
        
        # Validate workloads
        if not request.workloads:
            raise HTTPException(status_code=400, detail="No workloads provided")
        
        # Setup environment (get fresh VMs for each simulation)
        aws, gcp, azure, vms = setup_environment()
        
        # Convert workloads
        workloads = [
            Workload(w.id, w.cpu_required, w.memory_required_gb) 
            for w in request.workloads
        ]
        
        # Select scheduler with enhanced options
        if request.scheduler_type == "random":
            scheduler = RandomScheduler()
        elif request.scheduler_type == "lowest_cost":
            scheduler = LowestCostScheduler()
        elif request.scheduler_type == "round_robin":
            scheduler = RoundRobinScheduler()
        elif request.scheduler_type == "intelligent":
            scheduler = IntelligentScheduler()  # New scheduler
        elif request.scheduler_type == "hybrid":
            scheduler = HybridScheduler()  # New scheduler
        
        # Enhanced simulation with detailed tracking
        logs = []
        successful_assignments = 0
        failed_assignments = 0
        resource_utilization = {"cpu": 0, "memory": 0}
        assignment_times = []
        cost_breakdown = {"aws": 0, "gcp": 0, "azure": 0}
        
        import time
        simulation_start = time.time()
        
        for workload in workloads:
            assignment_start = time.time()
            
            selected_vm = scheduler.select_vm(workload, vms)
            if selected_vm and selected_vm.assign_workload(workload):
                assignment_time = time.time() - assignment_start
                assignment_times.append(assignment_time)
                
                # Calculate cost for this assignment
                workload_cost = (workload.cpu_required * selected_vm.provider.cpu_cost + 
                               workload.memory_required_gb * selected_vm.provider.memory_cost_gb)
                
                # Update cost breakdown by provider
                provider_name = selected_vm.provider.name.lower()
                if provider_name in cost_breakdown:
                    cost_breakdown[provider_name] += workload_cost
                
                logs.append({
                    "workload_id": workload.id,
                    "vm_id": selected_vm.vm_id,
                    "provider": selected_vm.provider.name,
                    "success": True,
                    "assignment_time_ms": round(assignment_time * 1000, 2),
                    "workload_cost": round(workload_cost, 4),
                    "cpu_utilized": workload.cpu_required,
                    "memory_utilized": workload.memory_required_gb,
                    "vm_cpu_remaining": selected_vm.cpu_capacity - selected_vm.cpu_used,
                    "vm_memory_remaining": selected_vm.memory_capacity_gb - selected_vm.memory_used_gb,
                    "message": f"Workload {workload.id} assigned to VM {selected_vm.vm_id} ({selected_vm.provider.name})"
                })
                successful_assignments += 1
            else:
                failed_assignments += 1
                
                # Find why assignment failed
                failure_reason = "Unknown"
                available_vms = [vm for vm in vms if 
                               vm.cpu_used + workload.cpu_required <= vm.cpu_capacity and
                               vm.memory_used_gb + workload.memory_required_gb <= vm.memory_capacity_gb]
                
                if not available_vms:
                    failure_reason = "Insufficient resources on all VMs"
                elif not selected_vm:
                    failure_reason = "Scheduler could not select a VM"
                
                logs.append({
                    "workload_id": workload.id,
                    "vm_id": None,
                    "provider": None,
                    "success": False,
                    "assignment_time_ms": 0,
                    "workload_cost": 0,
                    "cpu_required": workload.cpu_required,
                    "memory_required": workload.memory_required_gb,
                    "failure_reason": failure_reason,
                    "message": f"Could not assign workload {workload.id}: {failure_reason}"
                })
        
        simulation_end = time.time()
        total_simulation_time = simulation_end - simulation_start
        
        # Calculate enhanced metrics
        success_rate = (successful_assignments / len(workloads)) * 100 if workloads else 0
        failure_rate = (failed_assignments / len(workloads)) * 100 if workloads else 0
        total_cost = sum(vm.cost for vm in vms)
        
        # Resource utilization calculation
        total_cpu_capacity = sum(vm.cpu_capacity for vm in vms)
        total_memory_capacity = sum(vm.memory_capacity_gb for vm in vms)
        total_cpu_used = sum(vm.cpu_used for vm in vms)
        total_memory_used = sum(vm.memory_used_gb for vm in vms)
        
        cpu_utilization = (total_cpu_used / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
        memory_utilization = (total_memory_used / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0
        
        # VM-specific utilization
        vm_utilization = []
        for vm in vms:
            vm_cpu_util = (vm.cpu_used / vm.cpu_capacity) * 100 if vm.cpu_capacity > 0 else 0
            vm_memory_util = (vm.memory_used_gb / vm.memory_capacity_gb) * 100 if vm.memory_capacity_gb > 0 else 0
            
            vm_utilization.append({
                "vm_id": vm.vm_id,
                "provider": vm.provider.name,
                "cpu_utilization": round(vm_cpu_util, 2),
                "memory_utilization": round(vm_memory_util, 2),
                "cpu_used": vm.cpu_used,
                "cpu_capacity": vm.cpu_capacity,
                "memory_used": vm.memory_used_gb,
                "memory_capacity": vm.memory_capacity_gb,
                "cost": round(vm.cost, 4)
            })
        
        # Performance metrics
        avg_assignment_time = sum(assignment_times) / len(assignment_times) if assignment_times else 0
        
        summary = {
            "total_workloads": len(workloads),
            "successful_assignments": successful_assignments,
            "failed_assignments": failed_assignments,
            "success_rate": round(success_rate, 2),
            "failure_rate": round(failure_rate, 2),
            "total_cost": round(total_cost, 4),
            "cost_breakdown": {k: round(v, 4) for k, v in cost_breakdown.items()},
            "resource_utilization": {
                "overall_cpu_utilization": round(cpu_utilization, 2),
                "overall_memory_utilization": round(memory_utilization, 2),
                "total_cpu_used": total_cpu_used,
                "total_cpu_capacity": total_cpu_capacity,
                "total_memory_used": total_memory_used,
                "total_memory_capacity": total_memory_capacity
            },
            "performance_metrics": {
                "total_simulation_time_seconds": round(total_simulation_time, 4),
                "average_assignment_time_ms": round(avg_assignment_time * 1000, 2),
                "assignments_per_second": round(len(workloads) / total_simulation_time, 2) if total_simulation_time > 0 else 0
            }
        }
        
        return {
            "logs": logs,
            "summary": summary,
            "vm_utilization": vm_utilization,
            "scheduler_type": request.scheduler_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/simulation/compare")
async def compare_schedulers(request: dict):
    """Compare multiple schedulers with detailed analysis"""
    try:
        scheduler_types = request.get("scheduler_types", [])
        workloads_data = request.get("workloads", [])
        
        if not scheduler_types:
            raise HTTPException(status_code=400, detail="No scheduler types provided")
        
        if not workloads_data:
            raise HTTPException(status_code=400, detail="No workloads provided")
        
        results = {}
        comparison_metrics = {}
        
        for scheduler_type in scheduler_types:
            # Create simulation request for each scheduler
            sim_request = SimulationRequest(
                scheduler_type=scheduler_type,
                workloads=[WorkloadModel(**w) for w in workloads_data]
            )
            
            # Run simulation
            result = await run_simulation(sim_request)
            results[scheduler_type] = result
            
            # Extract comparison metrics
            summary = result["summary"]
            comparison_metrics[scheduler_type] = {
                "success_rate": summary["success_rate"],
                "total_cost": summary["total_cost"],
                "cpu_utilization": summary["resource_utilization"]["overall_cpu_utilization"],
                "memory_utilization": summary["resource_utilization"]["overall_memory_utilization"],
                "avg_assignment_time": summary["performance_metrics"]["average_assignment_time_ms"],
                "assignments_per_second": summary["performance_metrics"]["assignments_per_second"]
            }
        
        # Calculate best performer in each category
        best_performers = {
            "highest_success_rate": max(comparison_metrics.items(), key=lambda x: x[1]["success_rate"])[0],
            "lowest_cost": min(comparison_metrics.items(), key=lambda x: x[1]["total_cost"])[0],
            "best_cpu_utilization": max(comparison_metrics.items(), key=lambda x: x[1]["cpu_utilization"])[0],
            "best_memory_utilization": max(comparison_metrics.items(), key=lambda x: x[1]["memory_utilization"])[0],
            "fastest_assignment": min(comparison_metrics.items(), key=lambda x: x[1]["avg_assignment_time"])[0],
            "highest_throughput": max(comparison_metrics.items(), key=lambda x: x[1]["assignments_per_second"])[0]
        }
        
        # Generate recommendations
        recommendations = []
        
        # Cost-focused recommendation
        lowest_cost_scheduler = best_performers["lowest_cost"]
        if comparison_metrics[lowest_cost_scheduler]["success_rate"] >= 80:
            recommendations.append({
                "type": "cost_optimization",
                "scheduler": lowest_cost_scheduler,
                "reason": f"Best cost efficiency with acceptable success rate ({comparison_metrics[lowest_cost_scheduler]['success_rate']:.1f}%)"
            })
        
        # Performance-focused recommendation
        highest_success_scheduler = best_performers["highest_success_rate"]
        recommendations.append({
            "type": "performance_optimization",
            "scheduler": highest_success_scheduler,
            "reason": f"Highest success rate ({comparison_metrics[highest_success_scheduler]['success_rate']:.1f}%)"
        })
        
        # Balanced recommendation
        balanced_scores = {}
        for scheduler, metrics in comparison_metrics.items():
            # Normalize metrics and calculate balanced score
            score = (
                metrics["success_rate"] / 100 * 0.3 +  # 30% weight on success rate
                (100 - metrics["total_cost"]) / 100 * 0.2 +  # 20% weight on low cost (inverted)
                metrics["cpu_utilization"] / 100 * 0.2 +  # 20% weight on CPU utilization
                metrics["memory_utilization"] / 100 * 0.2 +  # 20% weight on memory utilization
                (1000 / (metrics["avg_assignment_time"] + 1)) / 1000 * 0.1  # 10% weight on speed
            )
            balanced_scores[scheduler] = score
        
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])[0]
        recommendations.append({
            "type": "balanced_optimization",
            "scheduler": best_balanced,
            "reason": f"Best overall balance of success rate, cost, utilization, and performance"
        })
        
        return {
            "results": results,
            "comparison_metrics": comparison_metrics,
            "best_performers": best_performers,
            "recommendations": recommendations,
            "summary": {
                "total_schedulers_compared": len(scheduler_types),
                "total_workloads": len(workloads_data),
                "comparison_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/simulation/analyze")
async def analyze_simulation_results(results_data: dict):
    """Analyze simulation results and provide insights"""
    try:
        logs = results_data.get("logs", [])
        summary = results_data.get("summary", {})
        scheduler_type = results_data.get("scheduler_type", "unknown")
        
        if not logs:
            raise HTTPException(status_code=400, detail="No simulation logs provided")
        
        analysis = {
            "scheduler_performance": {},
            "resource_patterns": {},
            "cost_analysis": {},
            "recommendations": [],
            "bottlenecks": []
        }
        
        # Analyze scheduler performance
        successful_logs = [log for log in logs if log.get("success", False)]
        failed_logs = [log for log in logs if not log.get("success", False)]
        
        if successful_logs:
            avg_assignment_time = sum(log.get("assignment_time_ms", 0) for log in successful_logs) / len(successful_logs)
            total_workload_cost = sum(log.get("workload_cost", 0) for log in successful_logs)
            
            analysis["scheduler_performance"] = {
                "efficiency_score": len(successful_logs) / len(logs) * 100,
                "average_assignment_time_ms": round(avg_assignment_time, 2),
                "total_successful_cost": round(total_workload_cost, 4),
                "cost_per_successful_assignment": round(total_workload_cost / len(successful_logs), 4) if successful_logs else 0
            }
        
        # Analyze resource patterns
        cpu_usage_pattern = [log.get("cpu_utilized", 0) for log in successful_logs]
        memory_usage_pattern = [log.get("memory_utilized", 0) for log in successful_logs]
        
        if cpu_usage_pattern:
            analysis["resource_patterns"] = {
                "cpu_usage": {
                    "min": min(cpu_usage_pattern),
                    "max": max(cpu_usage_pattern),
                    "average": round(sum(cpu_usage_pattern) / len(cpu_usage_pattern), 2),
                    "total": sum(cpu_usage_pattern)
                },
                "memory_usage": {
                    "min": min(memory_usage_pattern),
                    "max": max(memory_usage_pattern),
                    "average": round(sum(memory_usage_pattern) / len(memory_usage_pattern), 2),
                    "total": sum(memory_usage_pattern)
                }
            }
        
        # Analyze failure patterns
        if failed_logs:
            failure_reasons = {}
            for log in failed_logs:
                reason = log.get("failure_reason", "Unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            analysis["bottlenecks"] = [
                {"reason": reason, "count": count, "percentage": round(count/len(failed_logs)*100, 1)}
                for reason, count in failure_reasons.items()
            ]
        
        # Generate recommendations based on analysis
        success_rate = summary.get("success_rate", 0)
        
        if success_rate < 70:
            analysis["recommendations"].append({
                "type": "capacity_increase",
                "priority": "high",
                "message": "Consider adding more VMs or increasing VM capacity due to low success rate"
            })
        
        if success_rate > 95 and summary.get("resource_utilization", {}).get("overall_cpu_utilization", 0) < 50:
            analysis["recommendations"].append({
                "type": "resource_optimization",
                "priority": "medium",
                "message": "Excellent success rate but low resource utilization. Consider optimizing VM sizes"
            })
        
        total_cost = summary.get("total_cost", 0)
        if total_cost > 10:  # Arbitrary threshold
            analysis["recommendations"].append({
                "type": "cost_optimization",
                "priority": "medium",
                "message": "High total cost detected. Consider using cost-optimized scheduler or cheaper providers"
            })
        
        return {
            "analysis": analysis,
            "analyzed_scheduler": scheduler_type,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_quality": {
                "total_logs_analyzed": len(logs),
                "successful_assignments_analyzed": len(successful_logs),
                "failed_assignments_analyzed": len(failed_logs)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workloads/analyze")
async def analyze_workloads(workloads: List[WorkloadModel]):
    """Analyze workload patterns and provide capacity recommendations"""
    try:
        if not workloads:
            raise HTTPException(status_code=400, detail="No workloads provided")
        
        # Extract CPU and memory requirements
        cpu_requirements = [w.cpu_required for w in workloads]
        memory_requirements = [w.memory_required_gb for w in workloads]
        
        # Basic statistics
        stats = {
            "workload_count": len(workloads),
            "cpu_statistics": {
                "min": min(cpu_requirements),
                "max": max(cpu_requirements),
                "average": round(sum(cpu_requirements) / len(cpu_requirements), 2),
                "total": sum(cpu_requirements),
                "median": sorted(cpu_requirements)[len(cpu_requirements)//2]
            },
            "memory_statistics": {
                "min": min(memory_requirements),
                "max": max(memory_requirements),
                "average": round(sum(memory_requirements) / len(memory_requirements), 2),
                "total": sum(memory_requirements),
                "median": sorted(memory_requirements)[len(memory_requirements)//2]
            }
        }
        
        # Pattern analysis
        patterns = {
            "small_workloads": len([w for w in workloads if w.cpu_required <= 2 and w.memory_required_gb <= 4]),
            "medium_workloads": len([w for w in workloads if 2 < w.cpu_required <= 6 and 4 < w.memory_required_gb <= 16]),
            "large_workloads": len([w for w in workloads if w.cpu_required > 6 or w.memory_required_gb > 16])
        }
        
        # Capacity recommendations
        recommendations = []
        
        total_cpu_needed = sum(cpu_requirements)
        total_memory_needed = sum(memory_requirements)
        
        # Current VM capacity from setup
        _, _, _, vms = setup_environment()
        total_vm_cpu = sum(vm.cpu_capacity for vm in vms)
        total_vm_memory = sum(vm.memory_capacity_gb for vm in vms)
        
        cpu_utilization = (total_cpu_needed / total_vm_cpu) * 100 if total_vm_cpu > 0 else 0
        memory_utilization = (total_memory_needed / total_vm_memory) * 100 if total_vm_memory > 0 else 0
        
        if cpu_utilization > 80:
            recommendations.append({
                "type": "cpu_capacity",
                "priority": "high",
                "message": f"CPU utilization will be {cpu_utilization:.1f}%. Consider adding more CPU capacity."
            })
        
        if memory_utilization > 80:
            recommendations.append({
                "type": "memory_capacity",
                "priority": "high",
                "message": f"Memory utilization will be {memory_utilization:.1f}%. Consider adding more memory capacity."
            })
        
        # Suggest optimal VM configuration
        optimal_vm_config = {
            "recommended_vm_count": max(1, int(total_cpu_needed / 4) + 1),  # Assuming 4 CPU per VM
            "recommended_cpu_per_vm": max(4, int(max(cpu_requirements) * 1.2)),  # 20% buffer
            "recommended_memory_per_vm": max(8, int(max(memory_requirements) * 1.2))  # 20% buffer
        }
        
        return {
            "statistics": stats,
            "patterns": patterns,
            "capacity_analysis": {
                "projected_cpu_utilization": round(cpu_utilization, 2),
                "projected_memory_utilization": round(memory_utilization, 2),
                "current_vm_cpu_capacity": total_vm_cpu,
                "current_vm_memory_capacity": total_vm_memory
            },
            "recommendations": recommendations,
            "optimal_vm_config": optimal_vm_config,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
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
        "available_algorithms": ["random", "lowest_cost", "round_robin", "intelligent", "hybrid"],
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
                    "max_workloads": config.get("max_workloads_per_request"),
                    "available_algorithms": len(config.get("available_algorithms", []))
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