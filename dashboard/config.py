"""
Dashboard configuration settings
"""

import os
from typing import Dict, Any

class DashboardSettings:
    """Dashboard configuration settings"""
    
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    
    # Dashboard Configuration
    PAGE_TITLE = "Cloud Intelligence Platform"
    PAGE_ICON = "☁️"
    LAYOUT = "wide"
    
    # Refresh Settings
    DEFAULT_REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "30"))
    MIN_REFRESH_INTERVAL = 10
    MAX_REFRESH_INTERVAL = 300
    
    # Time Range Options (in hours)
    TIME_RANGE_OPTIONS = {
        "Last Hour": 1,
        "Last 6 Hours": 6,
        "Last 12 Hours": 12,
        "Last 24 Hours": 24,
        "Last 3 Days": 72,
        "Last Week": 168
    }
    
    # Default Filters
    DEFAULT_PROVIDERS = ["AWS", "GCP", "Azure"]
    DEFAULT_RESOURCES = ["Compute", "Storage", "Network", "Database"]
    
    # Chart Configuration
    CHART_HEIGHT = 350
    GAUGE_HEIGHT = 250
    HEATMAP_HEIGHT = 300
    
    # Color Palette
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "warning": "#ffc107",
        "danger": "#d62728",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40"
    }
    
    # Status Colors
    STATUS_COLORS = {
        "healthy": "#28a745",
        "warning": "#ffc107",
        "critical": "#dc3545",
        "unknown": "#6c757d",
        "running": "#28a745",
        "pending": "#ffc107",
        "failed": "#dc3545",
        "completed": "#17a2b8"
    }
    
    # Severity Colors for Alerts
    SEVERITY_COLORS = {
        "critical": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        "low": "#28a745"
    }
    
    # Performance Thresholds
    PERFORMANCE_THRESHOLDS = {
        "cpu": {
            "warning": 80,
            "critical": 95
        },
        "memory": {
            "warning": 85,
            "critical": 95
        },
        "storage": {
            "warning": 80,
            "critical": 90
        },
        "network": {
            "warning": 70,
            "critical": 90
        }
    }
    
    # Dashboard Features
    FEATURES = {
        "auto_refresh": True,
        "real_time_charts": True,
        "interactive_filters": True,
        "drill_down": True,
        "export_reports": True,
        "custom_dashboards": False  # To be implemented in subtask 5.2
    }
    
    @classmethod
    def get_chart_config(cls) -> Dict[str, Any]:
        """Get standardized chart configuration"""
        return {
            "height": cls.CHART_HEIGHT,
            "margin": dict(l=20, r=20, t=40, b=20),
            "font": dict(family="Arial, sans-serif", size=12),
            "plot_bgcolor": "white",
            "paper_bgcolor": "white"
        }
    
    @classmethod
    def get_gauge_config(cls) -> Dict[str, Any]:
        """Get gauge chart configuration"""
        return {
            "height": cls.GAUGE_HEIGHT,
            "margin": dict(l=20, r=20, t=40, b=20)
        }
    
    @classmethod
    def get_color_scale(cls, chart_type: str = "default") -> list:
        """Get color scale for charts"""
        if chart_type == "performance":
            return ["#28a745", "#ffc107", "#dc3545"]  # Green to Red
        elif chart_type == "cost":
            return ["#e3f2fd", "#1976d2"]  # Light blue to Dark blue
        else:
            return list(cls.COLORS.values())[:8]  # Default color palette