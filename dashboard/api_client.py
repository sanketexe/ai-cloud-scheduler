"""
API client for dashboard data fetching
"""

import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DashboardAPIClient:
    """Client for fetching dashboard data from the backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            return None
    
    def get_system_status(self) -> Dict:
        """Get overall system status and health metrics"""
        data = self._make_request("GET", "/api/system/status")
        
        if not data:
            # Return mock data for development
            return {
                "health_score": 95,
                "health_change": 2,
                "status": "healthy",
                "last_updated": datetime.now().isoformat(),
                "components": {
                    "scheduler": "healthy",
                    "finops": "healthy", 
                    "monitoring": "warning",
                    "database": "healthy"
                }
            }
        
        return data
    
    def get_workload_metrics(self, hours: int = 24) -> Dict:
        """Get workload metrics and statistics"""
        params = {"hours": hours}
        data = self._make_request("GET", "/api/workloads/metrics", params=params)
        
        if not data:
            # Return mock data for development
            return {
                "active_count": 156,
                "change_24h": 12,
                "total_scheduled": 1247,
                "success_rate": 98.5,
                "provider_distribution": {
                    "AWS": 78,
                    "GCP": 45,
                    "Azure": 33
                },
                "workloads": [
                    {
                        "id": "wl-001",
                        "status": "running",
                        "provider": "AWS",
                        "cost": 45.67,
                        "created_at": (datetime.now() - timedelta(hours=2)).isoformat()
                    },
                    {
                        "id": "wl-002", 
                        "status": "pending",
                        "provider": "GCP",
                        "cost": 23.45,
                        "created_at": (datetime.now() - timedelta(hours=1)).isoformat()
                    },
                    {
                        "id": "wl-003",
                        "status": "completed",
                        "provider": "Azure", 
                        "cost": 67.89,
                        "created_at": (datetime.now() - timedelta(hours=3)).isoformat()
                    }
                ]
            }
        
        return data
    
    def get_cost_metrics(self, hours: int = 24) -> Dict:
        """Get cost metrics and financial data"""
        params = {"hours": hours}
        data = self._make_request("GET", "/api/costs/metrics", params=params)
        
        if not data:
            # Generate mock hourly cost data
            hourly_costs = []
            base_time = datetime.now() - timedelta(hours=hours)
            
            for i in range(hours):
                timestamp = base_time + timedelta(hours=i)
                cost = 45.0 + (i * 2.5) + (5 * (i % 3))  # Trending upward with variation
                hourly_costs.append({
                    "timestamp": timestamp.isoformat(),
                    "cost": round(cost, 2)
                })
            
            return {
                "total_cost_24h": 1247.89,
                "cost_change_percent": 5.2,
                "hourly_costs": hourly_costs,
                "by_provider": {
                    "AWS": 678.45,
                    "GCP": 345.67,
                    "Azure": 223.77
                },
                "by_service": {
                    "Compute": 567.89,
                    "Storage": 234.56,
                    "Network": 123.45,
                    "Database": 321.99
                },
                "timeline": {
                    "timestamps": [item["timestamp"] for item in hourly_costs],
                    "values": [item["cost"] for item in hourly_costs]
                }
            }
        
        return data
    
    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get performance and resource utilization metrics"""
        params = {"hours": hours}
        data = self._make_request("GET", "/api/performance/metrics", params=params)
        
        if not data:
            # Generate mock performance timeline
            timeline = []
            base_time = datetime.now() - timedelta(hours=hours)
            
            for i in range(0, hours, 1):  # Every hour
                timestamp = base_time + timedelta(hours=i)
                timeline.append({
                    "timestamp": timestamp.isoformat(),
                    "cpu": 45 + (10 * (i % 4)) + (i * 0.5),
                    "memory": 60 + (15 * (i % 3)) + (i * 0.3),
                    "network_io": 25 + (5 * (i % 5)),
                    "disk_io": 30 + (8 * (i % 6))
                })
            
            return {
                "avg_cpu": 67.5,
                "cpu_change": -2.3,
                "avg_memory": 72.1,
                "memory_change": 1.8,
                "metrics_timeline": timeline,
                "metrics": {
                    "server-001": {
                        "cpu": 78.5,
                        "memory": 65.2,
                        "network": 45.7,
                        "disk": 34.8
                    },
                    "server-002": {
                        "cpu": 56.3,
                        "memory": 78.9,
                        "network": 23.4,
                        "disk": 67.1
                    },
                    "server-003": {
                        "cpu": 89.1,
                        "memory": 45.6,
                        "network": 78.2,
                        "disk": 23.5
                    }
                },
                "timeline": {
                    "cpu": {
                        "timestamps": [item["timestamp"] for item in timeline],
                        "values": [item["cpu"] for item in timeline]
                    },
                    "memory": {
                        "timestamps": [item["timestamp"] for item in timeline],
                        "values": [item["memory"] for item in timeline]
                    }
                }
            }
        
        return data
    
    def get_alert_data(self, hours: int = 24) -> Dict:
        """Get alert and notification data"""
        params = {"hours": hours}
        data = self._make_request("GET", "/api/alerts", params=params)
        
        if not data:
            # Generate mock alert data
            alerts = [
                {
                    "id": "alert-001",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "severity": "warning",
                    "message": "High CPU utilization on server-001",
                    "resource": "server-001",
                    "status": "active"
                },
                {
                    "id": "alert-002",
                    "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
                    "severity": "critical",
                    "message": "Cost threshold exceeded for AWS",
                    "resource": "aws-account",
                    "status": "acknowledged"
                },
                {
                    "id": "alert-003",
                    "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "severity": "info",
                    "message": "Scheduled maintenance completed",
                    "resource": "system",
                    "status": "resolved"
                }
            ]
            
            return {
                "alerts": alerts,
                "active_count": 2,
                "critical_count": 1,
                "warning_count": 1
            }
        
        return data
    
    def get_provider_comparison(self) -> Dict:
        """Get cloud provider comparison data"""
        data = self._make_request("GET", "/api/providers/comparison")
        
        if not data:
            return {
                "providers": {
                    "AWS": {
                        "cost": 85.2,
                        "performance": 92.1,
                        "availability": 99.9
                    },
                    "GCP": {
                        "cost": 78.9,
                        "performance": 88.7,
                        "availability": 99.8
                    },
                    "Azure": {
                        "cost": 82.4,
                        "performance": 90.3,
                        "availability": 99.7
                    }
                }
            }
        
        return data
    
    def get_resource_utilization(self) -> Dict:
        """Get current resource utilization data"""
        data = self._make_request("GET", "/api/resources/utilization")
        
        if not data:
            return {
                "cpu": 67.5,
                "memory": 72.1,
                "storage": 45.8,
                "network": 34.2
            }
        
        return data