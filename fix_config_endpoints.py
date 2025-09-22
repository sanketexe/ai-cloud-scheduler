#!/usr/bin/env python3
"""
Add missing configuration endpoints to api.py
"""

def add_config_endpoints():
    """Add configuration endpoints to api.py"""
    
    config_code = '''
# Configuration endpoints added automatically
from typing import Dict, Any
from datetime import datetime

# Configuration storage
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
    "scheduler": {
        "default_algorithm": "lowest_cost",
        "available_algorithms": ["random", "lowest_cost", "round_robin"],
        "max_workloads_per_request": 100,
        "simulation_timeout": 300,
        "retry_attempts": 3
    },
    "providers": {
        "aws": {"enabled": True, "cpu_cost": 0.04, "memory_cost_gb": 0.01},
        "gcp": {"enabled": True, "cpu_cost": 0.035, "memory_cost_gb": 0.009},
        "azure": {"enabled": True, "cpu_cost": 0.042, "memory_cost_gb": 0.011}
    },
    "ml": {
        "model_type": "lstm",
        "training_enabled": True,
        "prediction_window": 12,
        "auto_retrain": False
    },
    "performance": {
        "cache_enabled": True,
        "concurrent_limit": 10,
        "rate_limit": 100,
        "response_compression": True
    }
}

class ConfigurationResponse(BaseModel):
    category: str
    config: Dict[str, Any] 
    last_updated: str
    editable: bool

class ConfigurationUpdate(BaseModel):
    category: str
    config: Dict[str, Any]

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
        
        for category, config in SYSTEM_CONFIG.items():
            category_info = {
                "total_settings": len(config),
                "key_settings": {},
                "status": "configured"
            }
            
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
            else:
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

'''
    
    # Read current api.py
    try:
        with open("api.py", "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading api.py: {e}")
        return
    
    # Check if config endpoints already exist
    if "show_configuration" in content:
        print("✅ Configuration endpoints already exist in api.py")
        return
    
    # Find insertion point (before the if __name__ == "__main__": line)
    insertion_point = content.find('if __name__ == "__main__":')
    if insertion_point == -1:
        # Insert at the end
        updated_content = content + "\n" + config_code
    else:
        # Insert before the main block
        updated_content = content[:insertion_point] + config_code + "\n\n" + content[insertion_point:]
    
    # Write updated content
    try:
        with open("api.py", "w", encoding="utf-8") as f:
            f.write(updated_content)
        print("✅ Added configuration endpoints to api.py")
        print("⚠️  Please restart your API server: Ctrl+C then python api.py")
    except Exception as e:
        print(f"❌ Error writing to api.py: {e}")

if __name__ == "__main__":
    print("🔧 Adding configuration endpoints to api.py...")
    add_config_endpoints()