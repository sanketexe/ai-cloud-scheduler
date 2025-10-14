#!/usr/bin/env python3
"""
Test script for dashboard components
"""

import sys
import os
sys.path.append('dashboard')

def test_api_client():
    """Test API client functionality"""
    print("Testing API Client...")
    
    from api_client import DashboardAPIClient
    
    client = DashboardAPIClient()
    
    # Test system status
    status = client.get_system_status()
    print(f"✓ System Status: {status.get('status', 'unknown')}")
    
    # Test workload metrics
    workloads = client.get_workload_metrics()
    print(f"✓ Workload Metrics: {workloads.get('active_count', 0)} active workloads")
    
    # Test cost metrics
    costs = client.get_cost_metrics()
    print(f"✓ Cost Metrics: ${costs.get('total_cost_24h', 0):,.2f} total cost")
    
    # Test performance metrics
    performance = client.get_performance_metrics()
    print(f"✓ Performance Metrics: {performance.get('avg_cpu', 0):.1f}% avg CPU")
    
    print("API Client tests passed!\n")

def test_chart_components():
    """Test chart components"""
    print("Testing Chart Components...")
    
    from components.charts import RealTimeCharts, InteractiveCharts
    
    # Test data
    workload_data = {
        "workloads": [
            {"status": "running", "provider": "AWS"},
            {"status": "pending", "provider": "GCP"},
            {"status": "completed", "provider": "Azure"}
        ]
    }
    
    cost_data = {
        "timeline": [
            {"timestamp": "2024-01-01T00:00:00", "cost": 100},
            {"timestamp": "2024-01-01T01:00:00", "cost": 120}
        ]
    }
    
    performance_data = {
        "metrics": {
            "server-1": {"cpu": 75, "memory": 60, "network": 45, "disk": 30}
        }
    }
    
    # Test workload chart
    chart = RealTimeCharts.create_workload_status_chart(workload_data)
    print(f"✓ Workload Status Chart: {type(chart).__name__}")
    
    # Test cost timeline
    chart = RealTimeCharts.create_cost_timeline_chart(cost_data)
    print(f"✓ Cost Timeline Chart: {type(chart).__name__}")
    
    # Test performance heatmap
    chart = RealTimeCharts.create_performance_heatmap(performance_data)
    print(f"✓ Performance Heatmap: {type(chart).__name__}")
    
    print("Chart Components tests passed!\n")

def test_analytics_components():
    """Test analytics components"""
    print("Testing Analytics Components...")
    
    from components.analytics import AdvancedFilters, DrillDownAnalytics, CustomDashboards
    
    # Test drill-down analytics
    test_data = {
        "by_provider": {"AWS": 500, "GCP": 300, "Azure": 200},
        "timeline": [
            {"timestamp": "2024-01-01T00:00:00", "cost": 100},
            {"timestamp": "2024-01-01T01:00:00", "cost": 120}
        ]
    }
    
    # Test hierarchical view
    chart = DrillDownAnalytics.create_hierarchical_view(test_data, "provider")
    print(f"✓ Hierarchical View Chart: {type(chart).__name__}")
    
    # Test correlation matrix
    chart = DrillDownAnalytics.create_correlation_matrix({})
    print(f"✓ Correlation Matrix: {type(chart).__name__}")
    
    # Test custom dashboards
    dashboards = CustomDashboards()
    print(f"✓ Custom Dashboards: {type(dashboards).__name__}")
    
    print("Analytics Components tests passed!\n")

def test_layout_components():
    """Test layout components"""
    print("Testing Layout Components...")
    
    from components.layout import LayoutManager, ChartFactory, FilterManager, ThemeManager
    
    # Test layout manager
    layout = LayoutManager()
    print(f"✓ Layout Manager: {type(layout).__name__}")
    
    # Test chart factory
    factory = ChartFactory()
    print(f"✓ Chart Factory: {type(factory).__name__}")
    
    # Test filter manager
    filter_mgr = FilterManager()
    print(f"✓ Filter Manager: {type(filter_mgr).__name__}")
    
    # Test theme manager
    theme = ThemeManager()
    colors = theme.get_color_palette()
    print(f"✓ Theme Manager: {len(colors)} colors defined")
    
    print("Layout Components tests passed!\n")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Cloud Intelligence Platform Dashboard Tests")
    print("=" * 50)
    
    try:
        test_api_client()
        test_chart_components()
        test_analytics_components()
        test_layout_components()
        
        print("🎉 All tests passed successfully!")
        print("\nDashboard is ready to run:")
        print("  python dashboard/run_dashboard.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())