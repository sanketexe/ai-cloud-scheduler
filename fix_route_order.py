#!/usr/bin/env python3
"""
Fix route order in api.py to resolve conflicts
"""

def fix_route_order():
    """Fix the route order in api.py"""
    
    print("🔧 Fixing route order in api.py...")
    
    # Read current api.py
    try:
        with open("api.py", "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading api.py: {e}")
        return
    
    # Find and extract the specific endpoints
    show_endpoint = '''@app.get("/api/config/show")
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
        raise HTTPException(status_code=500, detail=str(e))'''

    export_endpoint = '''@app.get("/api/config/export")
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
        raise HTTPException(status_code=500, detail=str(e))'''
    
    # Remove the conflicting endpoints from their current positions
    import re
    
    # Remove show_configuration endpoint
    show_pattern = r'@app\.get\("/api/config/show"\).*?raise HTTPException\(status_code=500, detail=str\(e\)\)'
    content = re.sub(show_pattern, '', content, flags=re.DOTALL)
    
    # Remove export_configuration endpoint  
    export_pattern = r'@app\.get\("/api/config/export"\).*?raise HTTPException\(status_code=500, detail=str\(e\)\)'
    content = re.sub(export_pattern, '', content, flags=re.DOTALL)
    
    # Find where to insert (before the generic category endpoint)
    category_endpoint_pos = content.find('@app.get("/api/config/{category}")')
    
    if category_endpoint_pos == -1:
        print("❌ Could not find category endpoint")
        return
    
    # Insert the specific endpoints before the generic one
    new_content = (
        content[:category_endpoint_pos] + 
        show_endpoint + '\n\n' +
        export_endpoint + '\n\n' +
        content[category_endpoint_pos:]
    )
    
    # Write the fixed content
    try:
        with open("api.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print("✅ Fixed route order in api.py")
        print("🔄 Please restart your API server:")
        print("   1. Stop server (Ctrl+C)")
        print("   2. Start server: python api.py")
        print("   3. Test: python demo_configuration.py")
    except Exception as e:
        print(f"❌ Error writing to api.py: {e}")

if __name__ == "__main__":
    fix_route_order()