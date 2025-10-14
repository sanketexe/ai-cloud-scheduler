#!/usr/bin/env python3
"""
Test script for dashboard integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard'))

def test_imports():
    """Test that all dashboard components can be imported"""
    
    try:
        from dashboard.components.reporting import ReportGenerator, ReportScheduler, ReportDistributor
        print("✅ Reporting components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import reporting components: {e}")
        return False
    
    try:
        from dashboard.components.alerts import AlertManager, AlertAnalytics
        print("✅ Alert components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import alert components: {e}")
        return False
    
    try:
        from dashboard.api_client import DashboardAPIClient
        print("✅ API client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import API client: {e}")
        return False
    
    return True

def test_component_initialization():
    """Test that components can be initialized"""
    
    try:
        from dashboard.components.reporting import ReportGenerator
        from dashboard.components.alerts import AlertManager
        from dashboard.api_client import DashboardAPIClient
        
        # Test initialization
        report_gen = ReportGenerator()
        alert_mgr = AlertManager()
        api_client = DashboardAPIClient()
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of components"""
    
    try:
        from dashboard.components.reporting import ReportGenerator
        from dashboard.components.alerts import AlertManager
        from dashboard.api_client import DashboardAPIClient
        
        # Test API client
        api_client = DashboardAPIClient()
        system_status = api_client.get_system_status()
        
        if system_status and 'health_score' in system_status:
            print("✅ API client basic functionality working")
        else:
            print("❌ API client not returning expected data")
            return False
        
        # Test report generator
        report_gen = ReportGenerator()
        mock_data = {
            'time_period': 'Test Period',
            'cost_metrics': {'total_cost_24h': 100.0, 'cost_change_percent': 5.0},
            'workload_metrics': {'active_count': 10, 'success_rate': 95.0},
            'performance_metrics': {'avg_cpu': 70.0, 'avg_memory': 60.0}
        }
        
        exec_summary = report_gen.generate_executive_summary(mock_data)
        
        if exec_summary and 'title' in exec_summary:
            print("✅ Report generator basic functionality working")
        else:
            print("❌ Report generator not working as expected")
            return False
        
        # Test alert manager
        alert_mgr = AlertManager()
        
        if hasattr(alert_mgr, 'alert_types') and len(alert_mgr.alert_types) > 0:
            print("✅ Alert manager basic functionality working")
        else:
            print("❌ Alert manager not initialized properly")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing basic functionality: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧪 Testing Dashboard Integration")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing Imports...")
    if not test_imports():
        print("❌ Import tests failed")
        return False
    
    # Test initialization
    print("\n🔧 Testing Component Initialization...")
    if not test_component_initialization():
        print("❌ Initialization tests failed")
        return False
    
    # Test basic functionality
    print("\n⚙️ Testing Basic Functionality...")
    if not test_basic_functionality():
        print("❌ Functionality tests failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Dashboard integration is working correctly.")
    print("\nTo run the dashboard:")
    print("1. cd dashboard")
    print("2. streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)