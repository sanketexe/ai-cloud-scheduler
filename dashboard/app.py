"""
Cloud Intelligence Platform Dashboard
Main dashboard application with real-time data visualization
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time

# Import dashboard components
from components.layout import LayoutManager, ChartFactory, NavigationManager, FilterManager, ThemeManager
from components.charts import RealTimeCharts, InteractiveCharts
from components.analytics import InteractiveAnalytics, AdvancedFilters, DrillDownAnalytics, CustomDashboards
from components.reporting import ReportGenerator, ReportScheduler, ReportDistributor
from components.alerts import AlertManager, AlertAnalytics
from api_client import DashboardAPIClient

# Configure page
st.set_page_config(
    page_title="Cloud Intelligence Platform",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DashboardConfig:
    """Configuration for dashboard settings"""
    API_BASE_URL = "http://localhost:8000"
    REFRESH_INTERVAL = 30  # seconds
    DEFAULT_TIME_RANGE = 24  # hours
    AUTO_REFRESH_KEY = "auto_refresh_enabled"

class DashboardUI:
    """Main dashboard UI components"""
    
    def __init__(self):
        self.api_client = DashboardAPIClient()
        self.layout_manager = LayoutManager()
        self.chart_factory = ChartFactory()
        self.report_generator = ReportGenerator()
        self.report_scheduler = ReportScheduler()
        self.report_distributor = ReportDistributor()
        self.alert_manager = AlertManager()
        self.alert_analytics = AlertAnalytics()
        
        # Apply custom styling
        ThemeManager.apply_custom_css()
    
    def render_header(self):
        """Render dashboard header with title and navigation"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("☁️ Cloud Intelligence Platform")
            st.markdown("*Real-time multi-cloud intelligence and optimization*")
        
        with col2:
            # System status indicator
            system_status = self.api_client.get_system_status()
            status = system_status.get("status", "unknown")
            self.layout_manager.create_status_indicator(status, "System")
        
        with col3:
            # Last updated timestamp
            last_updated = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"**Last Updated:** {last_updated}")
        
        st.markdown("---")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", 
            "⚙️ Workloads", 
            "💰 Costs", 
            "🔧 Performance",
            "🚨 Alerts",
            "🔍 Analytics",
            "📋 Reports"
        ])
        return tab1, tab2, tab3, tab4, tab5, tab6, tab7
    
    def render_sidebar(self):
        """Render sidebar with filters and controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Time range filter
        time_range = FilterManager.create_time_filter()
        
        # Provider filter
        providers = FilterManager.create_provider_filter()
        
        # Resource filter
        resources = FilterManager.create_resource_filter()
        
        # Custom filters
        custom_filters = FilterManager.create_custom_filters()
        
        # Auto-refresh controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Refresh Settings")
        
        auto_refresh = st.sidebar.checkbox(
            "Enable Auto Refresh", 
            value=st.session_state.get(DashboardConfig.AUTO_REFRESH_KEY, True)
        )
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=DashboardConfig.REFRESH_INTERVAL,
                step=10
            )
        else:
            refresh_interval = DashboardConfig.REFRESH_INTERVAL
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            st.rerun()
        
        # Store auto-refresh state
        st.session_state[DashboardConfig.AUTO_REFRESH_KEY] = auto_refresh
        
        return {
            "time_range": time_range,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "providers": providers,
            "resources": resources,
            "custom_filters": custom_filters
        }
    
    def render_overview_tab(self, filters: Dict):
        """Render overview dashboard with key metrics"""
        st.header("📊 System Overview")
        
        # Fetch data
        system_status = self.api_client.get_system_status()
        workload_metrics = self.api_client.get_workload_metrics(filters["time_range"])
        cost_metrics = self.api_client.get_cost_metrics(filters["time_range"])
        performance_metrics = self.api_client.get_performance_metrics(filters["time_range"])
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.layout_manager.create_metric_card(
                "Active Workloads",
                workload_metrics.get("active_count", 0),
                workload_metrics.get("change_24h", 0),
                "Number of currently running workloads"
            )
        
        with col2:
            self.layout_manager.create_metric_card(
                "Total Cost (24h)",
                f"${cost_metrics.get('total_cost_24h', 0):,.2f}",
                f"{cost_metrics.get('cost_change_percent', 0):+.1f}%",
                "Total spending in the last 24 hours"
            )
        
        with col3:
            self.layout_manager.create_metric_card(
                "Avg CPU Utilization",
                f"{performance_metrics.get('avg_cpu', 0):.1f}%",
                f"{performance_metrics.get('cpu_change', 0):+.1f}%",
                "Average CPU utilization across all resources"
            )
        
        with col4:
            self.layout_manager.create_metric_card(
                "System Health",
                f"{system_status.get('health_score', 100)}%",
                system_status.get('health_change', 0),
                "Overall system health score"
            )
        
        st.markdown("---")
        
        # Real-time charts section
        st.subheader("📈 Real-time Metrics")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Workload status chart
            workload_chart = RealTimeCharts.create_workload_status_chart(workload_metrics)
            st.plotly_chart(workload_chart, use_container_width=True)
        
        with col2:
            # Cost timeline chart
            cost_chart = RealTimeCharts.create_cost_timeline_chart(cost_metrics)
            st.plotly_chart(cost_chart, use_container_width=True)
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance heatmap
            perf_chart = RealTimeCharts.create_performance_heatmap(performance_metrics)
            st.plotly_chart(perf_chart, use_container_width=True)
        
        with col2:
            # Provider comparison
            provider_data = self.api_client.get_provider_comparison()
            provider_chart = RealTimeCharts.create_provider_comparison_chart(provider_data)
            st.plotly_chart(provider_chart, use_container_width=True)
        
        # Resource utilization gauges
        st.subheader("🔧 Resource Utilization")
        
        utilization_data = self.api_client.get_resource_utilization()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_gauge = RealTimeCharts.create_resource_utilization_gauge(utilization_data, "cpu")
            st.plotly_chart(cpu_gauge, use_container_width=True)
        
        with col2:
            memory_gauge = RealTimeCharts.create_resource_utilization_gauge(utilization_data, "memory")
            st.plotly_chart(memory_gauge, use_container_width=True)
        
        with col3:
            storage_gauge = RealTimeCharts.create_resource_utilization_gauge(utilization_data, "storage")
            st.plotly_chart(storage_gauge, use_container_width=True)
        
        with col4:
            network_gauge = RealTimeCharts.create_resource_utilization_gauge(utilization_data, "network")
            st.plotly_chart(network_gauge, use_container_width=True)
    
    def render_workloads_tab(self, filters: Dict):
        """Render workloads management interface with advanced filtering"""
        st.header("⚙️ Workload Management")
        
        workload_metrics = self.api_client.get_workload_metrics(filters["time_range"])
        
        # Advanced search and filters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search Workloads",
                placeholder="Search by ID, status, provider, or tags...",
                key="workload_search"
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["running", "pending", "completed", "failed"],
                default=[],
                key="workload_status_filter"
            )
        
        # Workload summary with interactive metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.layout_manager.create_metric_card(
                "Total Workloads",
                workload_metrics.get("total_scheduled", 0),
                None,
                "Total workloads scheduled"
            )
        
        with col2:
            self.layout_manager.create_metric_card(
                "Success Rate",
                f"{workload_metrics.get('success_rate', 0):.1f}%",
                None,
                "Workload scheduling success rate"
            )
        
        with col3:
            self.layout_manager.create_metric_card(
                "Active Now",
                workload_metrics.get("active_count", 0),
                workload_metrics.get("change_24h", 0),
                "Currently active workloads"
            )
        
        with col4:
            avg_cost = sum([w.get("cost", 0) for w in workload_metrics.get("workloads", [])]) / max(len(workload_metrics.get("workloads", [])), 1)
            self.layout_manager.create_metric_card(
                "Avg Cost",
                f"${avg_cost:.2f}",
                None,
                "Average workload cost"
            )
        
        # Interactive workload distribution
        st.subheader("📊 Workload Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Provider distribution with drill-down
            provider_chart = DrillDownAnalytics.create_hierarchical_view(
                {"by_provider": workload_metrics.get("provider_distribution", {})},
                level="provider"
            )
            st.plotly_chart(provider_chart, use_container_width=True)
        
        with col2:
            # Status distribution
            if "workloads" in workload_metrics:
                status_counts = {}
                for workload in workload_metrics["workloads"]:
                    status = workload.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                status_chart = RealTimeCharts.create_workload_status_chart({"workloads": workload_metrics["workloads"]})
                st.plotly_chart(status_chart, use_container_width=True)
        
        # Enhanced workload details table with filtering
        st.subheader("📋 Workload Details")
        
        if "workloads" in workload_metrics and workload_metrics["workloads"]:
            df = pd.DataFrame(workload_metrics["workloads"])
            
            # Apply filters
            if search_query:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
                df = df[mask]
            
            if status_filter:
                df = df[df["status"].isin(status_filter)]
            
            if not df.empty:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
                
                # Add interactive features
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sort_by = st.selectbox("Sort by", options=["created_at", "cost", "status", "provider"])
                
                with col2:
                    sort_order = st.selectbox("Order", options=["Descending", "Ascending"])
                
                with col3:
                    page_size = st.selectbox("Items per page", options=[10, 25, 50, 100], index=1)
                
                # Sort data
                ascending = sort_order == "Ascending"
                df_sorted = df.sort_values(sort_by, ascending=ascending)
                
                # Pagination
                total_items = len(df_sorted)
                total_pages = (total_items - 1) // page_size + 1
                
                if total_pages > 1:
                    page = st.selectbox("Page", options=list(range(1, total_pages + 1)))
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    df_page = df_sorted.iloc[start_idx:end_idx]
                else:
                    df_page = df_sorted
                
                # Display table with styling
                st.dataframe(
                    df_page[["id", "status", "provider", "cost", "created_at"]],
                    use_container_width=True,
                    height=400
                )
                
                # Export options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Export to CSV"):
                        csv = df_sorted.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"workloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("📈 Create Report"):
                        st.info("Report generation feature - To be implemented in subtask 5.3")
                
                with col3:
                    if st.button("🔔 Set Alert"):
                        st.info("Alert configuration feature - To be implemented in subtask 5.4")
                
            else:
                st.info("No workloads match the current filters")
        else:
            st.info("No workload data available")
    
    def render_costs_tab(self, filters: Dict):
        """Render cost analytics interface with advanced drill-down"""
        st.header("💰 Cost Analytics")
        
        cost_metrics = self.api_client.get_cost_metrics(filters["time_range"])
        
        # Advanced cost filters
        with st.expander("🔍 Advanced Cost Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cost_threshold = st.slider("Cost Threshold ($)", 0, 1000, 100)
                
            with col2:
                cost_trend = st.selectbox("Cost Trend", ["All", "Increasing", "Decreasing", "Stable"])
                
            with col3:
                optimization_potential = st.selectbox("Optimization", ["All", "High", "Medium", "Low"])
        
        # Cost summary with enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.layout_manager.create_metric_card(
                "Total Cost",
                f"${cost_metrics.get('total_cost_24h', 0):,.2f}",
                f"{cost_metrics.get('cost_change_percent', 0):+.1f}%",
                f"Total cost for last {filters['time_range']} hours"
            )
        
        with col2:
            avg_hourly = cost_metrics.get('total_cost_24h', 0) / 24
            self.layout_manager.create_metric_card(
                "Avg Hourly",
                f"${avg_hourly:,.2f}",
                None,
                "Average hourly cost"
            )
        
        with col3:
            # Calculate projected monthly cost
            monthly_projection = cost_metrics.get('total_cost_24h', 0) * 30
            self.layout_manager.create_metric_card(
                "Monthly Projection",
                f"${monthly_projection:,.0f}",
                None,
                "Projected monthly cost"
            )
        
        with col4:
            # Optimization savings potential
            savings_potential = cost_metrics.get('total_cost_24h', 0) * 0.15  # Assume 15% savings potential
            self.layout_manager.create_metric_card(
                "Savings Potential",
                f"${savings_potential:,.2f}",
                None,
                "Estimated optimization savings"
            )
        
        # Interactive drill-down cost analysis
        st.subheader("📊 Interactive Cost Breakdown")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            drill_level = st.selectbox(
                "Analysis Level",
                options=["provider", "service", "resource"],
                format_func=lambda x: x.title(),
                key="cost_drill_level"
            )
            
            chart_type = st.selectbox(
                "Chart Type",
                options=["bar", "pie", "treemap"],
                format_func=lambda x: x.title(),
                key="cost_chart_type"
            )
        
        with col2:
            # Dynamic drill-down chart
            if chart_type == "treemap":
                # Create treemap for hierarchical cost view
                if "by_provider" in cost_metrics and "by_service" in cost_metrics:
                    fig = go.Figure(go.Treemap(
                        labels=list(cost_metrics["by_provider"].keys()) + list(cost_metrics["by_service"].keys()),
                        values=list(cost_metrics["by_provider"].values()) + list(cost_metrics["by_service"].values()),
                        parents=[""] * len(cost_metrics["by_provider"]) + ["Provider"] * len(cost_metrics["by_service"]),
                        textinfo="label+value+percent parent"
                    ))
                    fig.update_layout(title="Cost Hierarchy", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                cost_chart = DrillDownAnalytics.create_hierarchical_view(cost_metrics, drill_level)
                st.plotly_chart(cost_chart, use_container_width=True)
        
        # Time-based cost analysis
        st.subheader("📈 Cost Trends & Forecasting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_granularity = st.selectbox(
                "Time Granularity",
                options=["hour", "day", "week", "month"],
                index=1,
                key="cost_time_granularity"
            )
        
        with col2:
            forecast_days = st.slider("Forecast Days", 1, 30, 7)
        
        # Enhanced cost timeline with forecasting
        cost_timeline = DrillDownAnalytics.create_time_series_drilldown(
            cost_metrics, "cost", time_granularity
        )
        st.plotly_chart(cost_timeline, use_container_width=True)
        
        # Cost optimization recommendations
        st.subheader("💡 Cost Optimization Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **High Impact Optimizations:**
            - Right-size over-provisioned instances (Est. savings: $234/month)
            - Migrate to spot instances where appropriate (Est. savings: $156/month)
            - Optimize storage classes (Est. savings: $89/month)
            """)
        
        with col2:
            st.markdown("""
            **Quick Wins:**
            - Delete unused volumes (Est. savings: $45/month)
            - Schedule non-production resources (Est. savings: $123/month)
            - Review data transfer costs (Est. savings: $67/month)
            """)
        
        # Cost alerts and budgets
        st.subheader("🚨 Budget & Alerts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Create Budget"):
                st.info("Budget creation feature - To be implemented in subtask 5.4")
        
        with col2:
            if st.button("🔔 Set Cost Alert"):
                st.info("Cost alert configuration - To be implemented in subtask 5.4")
        
        with col3:
            if st.button("📈 Generate Report"):
                st.info("Cost report generation - To be implemented in subtask 5.3")
    
    def render_performance_tab(self, filters: Dict):
        """Render performance monitoring interface with advanced analytics"""
        st.header("🔧 Performance Monitoring")
        
        performance_metrics = self.api_client.get_performance_metrics(filters["time_range"])
        
        # Advanced performance filters
        with st.expander("🔍 Performance Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_threshold = st.slider("CPU Threshold (%)", 0, 100, 80)
                memory_threshold = st.slider("Memory Threshold (%)", 0, 100, 85)
            
            with col2:
                resource_filter = st.multiselect(
                    "Resource Types",
                    options=["Compute", "Storage", "Network", "Database"],
                    default=["Compute"]
                )
            
            with col3:
                anomaly_detection = st.checkbox("Show Anomalies Only", value=False)
                performance_trend = st.selectbox("Trend", ["All", "Improving", "Degrading", "Stable"])
        
        # Enhanced performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.layout_manager.create_metric_card(
                "Avg CPU",
                f"{performance_metrics.get('avg_cpu', 0):.1f}%",
                f"{performance_metrics.get('cpu_change', 0):+.1f}%",
                "Average CPU utilization"
            )
        
        with col2:
            self.layout_manager.create_metric_card(
                "Avg Memory",
                f"{performance_metrics.get('avg_memory', 0):.1f}%",
                f"{performance_metrics.get('memory_change', 0):+.1f}%",
                "Average memory utilization"
            )
        
        with col3:
            # Calculate performance score
            cpu = performance_metrics.get('avg_cpu', 0)
            memory = performance_metrics.get('avg_memory', 0)
            perf_score = 100 - ((cpu + memory) / 2)  # Simplified score
            self.layout_manager.create_metric_card(
                "Performance Score",
                f"{perf_score:.0f}",
                None,
                "Overall performance health score"
            )
        
        with col4:
            # Count resources above threshold
            high_util_count = 3  # Mock data
            self.layout_manager.create_metric_card(
                "High Utilization",
                high_util_count,
                None,
                "Resources above threshold"
            )
        
        # Interactive performance analysis
        st.subheader("📊 Interactive Performance Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metric and granularity selection
            selected_metric = st.selectbox(
                "Primary Metric",
                options=["cpu", "memory", "network_io", "disk_io"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="perf_metric"
            )
            
            time_granularity = st.selectbox(
                "Time Granularity",
                options=["hour", "day", "week"],
                index=0,
                key="perf_granularity"
            )
            
            comparison_metric = st.selectbox(
                "Compare With",
                options=["None", "cpu", "memory", "network_io", "disk_io", "cost"],
                format_func=lambda x: "None" if x == "None" else x.replace("_", " ").title(),
                key="perf_comparison"
            )
        
        with col2:
            # Multi-metric performance chart
            if comparison_metric != "None":
                # Create dual-axis chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Primary metric
                if "timeline" in performance_metrics and selected_metric in performance_metrics["timeline"]:
                    timeline_data = performance_metrics["timeline"][selected_metric]
                    fig.add_trace(
                        go.Scatter(
                            x=timeline_data["timestamps"],
                            y=timeline_data["values"],
                            name=selected_metric.title(),
                            line=dict(color="#1f77b4")
                        ),
                        secondary_y=False
                    )
                
                # Comparison metric
                if comparison_metric in performance_metrics.get("timeline", {}):
                    comp_data = performance_metrics["timeline"][comparison_metric]
                    fig.add_trace(
                        go.Scatter(
                            x=comp_data["timestamps"],
                            y=comp_data["values"],
                            name=comparison_metric.title(),
                            line=dict(color="#ff7f0e")
                        ),
                        secondary_y=True
                    )
                
                fig.update_layout(title=f"{selected_metric.title()} vs {comparison_metric.title()}", height=400)
                fig.update_yaxes(title_text=selected_metric.title(), secondary_y=False)
                fig.update_yaxes(title_text=comparison_metric.title(), secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Single metric chart with drill-down
                perf_chart = DrillDownAnalytics.create_time_series_drilldown(
                    performance_metrics, selected_metric, time_granularity
                )
                st.plotly_chart(perf_chart, use_container_width=True)
        
        # Performance correlation analysis
        st.subheader("🔗 Performance Correlations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation matrix
            correlation_chart = DrillDownAnalytics.create_correlation_matrix(performance_metrics)
            st.plotly_chart(correlation_chart, use_container_width=True)
        
        with col2:
            # Resource performance heatmap
            heatmap = RealTimeCharts.create_performance_heatmap(performance_metrics)
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Performance insights and recommendations
        st.subheader("💡 Performance Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔴 Critical Issues:**
            - Server-003: CPU at 89% (Action needed)
            - Database-01: Memory at 94% (Scale up)
            - Network bottleneck detected
            """)
        
        with col2:
            st.markdown("""
            **🟡 Optimization Opportunities:**
            - Right-size Server-001 (Over-provisioned)
            - Enable auto-scaling for Web tier
            - Optimize database queries
            """)
        
        with col3:
            st.markdown("""
            **🟢 Performing Well:**
            - Storage systems healthy
            - Network latency optimal
            - Load balancing effective
            """)
        
        # Performance actions
        st.subheader("⚡ Performance Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔔 Set Performance Alert"):
                st.info("Performance alert configuration - To be implemented in subtask 5.4")
        
        with col2:
            if st.button("📊 Generate Report"):
                st.info("Performance report generation - To be implemented in subtask 5.3")
        
        with col3:
            if st.button("🔧 Auto-Scale"):
                st.info("Auto-scaling configuration interface")
        
        with col4:
            if st.button("📈 Capacity Planning"):
                st.info("Capacity planning analysis interface")
    
    def render_alerts_tab(self, filters: Dict):
        """Render comprehensive alert management interface"""
        st.header("🚨 Alert Management")
        
        # Alert management sub-tabs
        alert_tab1, alert_tab2, alert_tab3, alert_tab4 = st.tabs([
            "🚨 Active Alerts",
            "⚙️ Alert Rules", 
            "📊 Analytics",
            "🔔 Notifications"
        ])
        
        # Get alert data
        alert_data = self.api_client.get_alert_data(filters.get('time_range', 24))
        
        with alert_tab1:
            # Active alerts management
            self.alert_manager.display_active_alerts(alert_data)
            
            # Quick actions
            st.markdown("---")
            st.subheader("⚡ Quick Actions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔕 Silence All Warning"):
                    st.info("All warning alerts silenced for 1 hour")
            
            with col2:
                if st.button("✅ Acknowledge All"):
                    st.info("All active alerts acknowledged")
            
            with col3:
                if st.button("📧 Send Summary"):
                    st.info("Alert summary sent to team")
            
            with col4:
                if st.button("🔄 Refresh Alerts"):
                    st.rerun()
        
        with alert_tab2:
            # Alert rules management
            self.alert_manager.create_alert_rules_management()
        
        with alert_tab3:
            # Alert analytics and history
            self.alert_manager.display_alert_history(alert_data)
            
            # Alert insights
            st.markdown("---")
            insights = self.alert_analytics.generate_alert_insights(alert_data)
            
            st.subheader("💡 Alert Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Metrics:**")
                st.markdown(f"• Total Alerts: {insights.get('total_alerts', 0)}")
                st.markdown(f"• Critical Alerts: {insights.get('critical_alerts', 0)}")
                st.markdown(f"• Most Common Type: {insights.get('most_common_type', 'N/A')}")
                st.markdown(f"• Busiest Resource: {insights.get('busiest_resource', 'N/A')}")
            
            with col2:
                st.markdown("**Trends & Efficiency:**")
                st.markdown(f"• Frequency Trend: {insights.get('alert_frequency_trend', 'N/A')}")
                st.markdown(f"• Resolution Efficiency: {insights.get('resolution_efficiency', 0):.1f}%")
            
            # Recommendations
            if insights.get('recommendations'):
                st.markdown("**Recommendations:**")
                for rec in insights['recommendations']:
                    st.markdown(f"• {rec}")
        
        with alert_tab4:
            # Notification settings
            self.alert_manager.create_notification_settings()
    
    def render_reports_tab(self, filters: Dict):
        """Render comprehensive reporting system interface"""
        st.header("📋 Comprehensive Reporting System")
        
        # Report generation section
        st.subheader("📊 Generate Reports")
        
        # Report configuration
        config = self.report_generator.create_report_configuration()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Generate Report", type="primary"):
                if config['name']:
                    with st.spinner("Generating report..."):
                        # Collect data for report
                        report_data = self._collect_report_data(filters, config)
                        
                        if config['type'] == "Executive Summary":
                            report = self.report_generator.generate_executive_summary(report_data)
                        else:
                            report = self.report_generator.generate_technical_report(report_data)
                        
                        # Store report in session state
                        st.session_state['generated_report'] = report
                        st.session_state['report_config'] = config
                        
                        st.success("Report generated successfully!")
                else:
                    st.error("Please provide a report name")
        
        with col2:
            if st.button("📅 Schedule Report"):
                if config['schedule_report'] and config['name']:
                    success = self.report_generator.schedule_report_generation(config)
                    if success:
                        st.success("Report scheduled successfully!")
                    else:
                        st.error("Failed to schedule report")
                else:
                    st.warning("Please enable scheduling and provide a report name")
        
        with col3:
            if st.button("📧 Send Report"):
                if 'generated_report' in st.session_state and config.get('email_recipients'):
                    report = st.session_state['generated_report']
                    recipients = [email.strip() for email in config['email_recipients'].split(',') if email.strip()]
                    
                    success = self.report_distributor.send_report_email(recipients, report)
                    if success:
                        st.success(f"Report sent to {len(recipients)} recipients!")
                    else:
                        st.error("Failed to send report")
                else:
                    st.warning("Please generate a report first and provide email recipients")
        
        # Report preview section
        if 'generated_report' in st.session_state:
            st.markdown("---")
            st.subheader("📄 Report Preview")
            
            report = st.session_state['generated_report']
            config = st.session_state.get('report_config', {})
            
            # Display report summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Report:** {report.get('title', 'Untitled Report')}")
                st.markdown(f"**Period:** {report.get('period', 'N/A')}")
                st.markdown(f"**Generated:** {report.get('generated_at', 'N/A')}")
            
            with col2:
                # Export options
                st.markdown("**Export Options:**")
                
                if 'PDF' in config.get('output_format', []):
                    if st.button("📄 Download PDF"):
                        try:
                            pdf_buffer = self.report_generator.export_to_pdf(report)
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        except ImportError as e:
                            st.error(f"PDF export not available: {e}")
                
                if 'Excel' in config.get('output_format', []):
                    if st.button("📊 Download Excel"):
                        excel_buffer = self.report_generator.export_to_excel(report)
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_buffer.getvalue(),
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            # Display report content
            if report.get('type') == 'Executive Summary' or 'key_metrics' in report:
                self._display_executive_summary(report)
            else:
                self._display_technical_report(report)
        
        # Scheduled reports section
        st.markdown("---")
        st.subheader("📅 Scheduled Reports")
        
        scheduled_reports = st.session_state.get('scheduled_reports', [])
        
        if scheduled_reports:
            # Display scheduled reports table
            df_scheduled = pd.DataFrame(scheduled_reports)
            
            # Format the dataframe for display
            display_df = df_scheduled[['report_name', 'report_type', 'frequency', 'next_run', 'active']].copy()
            display_df['next_run'] = pd.to_datetime(display_df['next_run']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_df, use_container_width=True)
            
            # Management actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Run Now"):
                    st.info("Manual execution of scheduled reports")
            
            with col2:
                if st.button("⏸️ Pause All"):
                    st.info("Pause all scheduled reports")
            
            with col3:
                if st.button("🗑️ Clear All"):
                    st.session_state['scheduled_reports'] = []
                    st.rerun()
        else:
            st.info("No scheduled reports configured")
        
        # Report templates section
        st.markdown("---")
        st.subheader("📋 Report Templates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Executive Dashboard**
            - Key performance indicators
            - Cost summary and trends
            - High-level recommendations
            - Executive-friendly visualizations
            """)
            
            if st.button("Use Executive Template", key="exec_template"):
                st.session_state['report_template'] = 'executive'
                st.info("Executive template selected")
        
        with col2:
            st.markdown("""
            **🔧 Technical Analysis**
            - Detailed performance metrics
            - Resource utilization analysis
            - Technical recommendations
            - Comprehensive data tables
            """)
            
            if st.button("Use Technical Template", key="tech_template"):
                st.session_state['report_template'] = 'technical'
                st.info("Technical template selected")
        
        with col3:
            st.markdown("""
            **💰 Financial Report**
            - Cost breakdown by service
            - Budget vs actual analysis
            - Optimization opportunities
            - ROI calculations
            """)
            
            if st.button("Use Financial Template", key="fin_template"):
                st.session_state['report_template'] = 'financial'
                st.info("Financial template selected")
    
    def _collect_report_data(self, filters: Dict, config: Dict) -> Dict[str, Any]:
        """Collect data needed for report generation"""
        
        # Determine time range based on config
        if config.get('time_period') == 'Last 24 Hours':
            hours = 24
        elif config.get('time_period') == 'Last Week':
            hours = 168
        elif config.get('time_period') == 'Last Month':
            hours = 720
        else:
            hours = filters.get('time_range', 24)
        
        # Collect all necessary data
        data = {
            'time_period': config.get('time_period', 'Last 24 Hours'),
            'workload_metrics': self.api_client.get_workload_metrics(hours),
            'cost_metrics': self.api_client.get_cost_metrics(hours),
            'performance_metrics': self.api_client.get_performance_metrics(hours),
            'alert_data': self.api_client.get_alert_data(hours),
            'system_status': self.api_client.get_system_status(),
            'include_charts': config.get('include_charts', True),
            'include_raw_data': config.get('include_raw_data', False)
        }
        
        return data
    
    def _display_executive_summary(self, report: Dict[str, Any]):
        """Display executive summary report content"""
        
        st.subheader("📊 Executive Summary")
        
        # Key metrics
        if 'key_metrics' in report:
            st.markdown("**Key Metrics**")
            
            metrics = report['key_metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cost", f"${metrics.get('total_cost', 0):,.2f}")
            
            with col2:
                st.metric("Active Workloads", metrics.get('active_workloads', 0))
            
            with col3:
                st.metric("Performance Score", f"{metrics.get('performance_score', 0):.1f}%")
            
            with col4:
                change = metrics.get('cost_change_percent', 0)
                st.metric("Cost Change", f"{change:+.1f}%")
        
        # Insights
        if 'insights' in report:
            st.markdown("**Key Insights**")
            
            insights = report['insights']
            for category, insight_list in insights.items():
                if insight_list:
                    st.markdown(f"*{category.replace('_', ' ').title()}:*")
                    for insight in insight_list:
                        st.markdown(f"• {insight}")
        
        # Recommendations
        if 'recommendations' in report:
            st.markdown("**Recommendations**")
            
            for i, rec in enumerate(report['recommendations'], 1):
                with st.expander(f"{i}. {rec.get('category', 'General')} - {rec.get('priority', 'Medium')} Priority"):
                    st.markdown(f"**Recommendation:** {rec.get('recommendation', '')}")
                    st.markdown(f"**Expected Impact:** {rec.get('impact', '')}")
    
    def _display_technical_report(self, report: Dict[str, Any]):
        """Display technical report content"""
        
        st.subheader("🔧 Technical Analysis Report")
        
        if 'sections' in report:
            sections = report['sections']
            
            # Create tabs for different sections
            section_tabs = st.tabs(list(sections.keys()))
            
            for tab, (section_name, section_data) in zip(section_tabs, sections.items()):
                with tab:
                    st.markdown(f"**{section_name.replace('_', ' ').title()}**")
                    
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, float) else str(value))
                            elif isinstance(value, dict):
                                st.json(value)
                            elif isinstance(value, list):
                                for item in value:
                                    st.markdown(f"• {item}")
                            else:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.markdown(str(section_data))ent")
        
        alert_data = self.api_client.get_alert_data(filters["time_range"])
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.layout_manager.create_metric_card(
                "Active Alerts",
                alert_data.get("active_count", 0),
                None,
                "Currently active alerts"
            )
        
        with col2:
            self.layout_manager.create_metric_card(
                "Critical Alerts",
                alert_data.get("critical_count", 0),
                None,
                "Critical severity alerts"
            )
        
        with col3:
            self.layout_manager.create_metric_card(
                "Warning Alerts",
                alert_data.get("warning_count", 0),
                None,
                "Warning severity alerts"
            )
        
        # Alert timeline
        st.subheader("📈 Alert Timeline")
        alert_timeline = RealTimeCharts.create_alert_timeline_chart(alert_data)
        st.plotly_chart(alert_timeline, use_container_width=True)
        
        # Alert details
        st.subheader("📋 Recent Alerts")
        
        if "alerts" in alert_data and alert_data["alerts"]:
            df = pd.DataFrame(alert_data["alerts"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            
            # Color code by severity
            def color_severity(val):
                colors = {
                    "critical": "background-color: #ffebee",
                    "warning": "background-color: #fff8e1", 
                    "info": "background-color: #e3f2fd",
                    "low": "background-color: #e8f5e8"
                }
                return colors.get(val, "")
            
            styled_df = df.style.applymap(color_severity, subset=["severity"])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No recent alerts")
    
    def render_analytics_tab(self, filters: Dict):
        """Render advanced analytics and custom dashboard interface"""
        st.header("🔍 Advanced Analytics")
        
        # Gather all data for analytics
        system_status = self.api_client.get_system_status()
        workload_metrics = self.api_client.get_workload_metrics(filters["time_range"])
        cost_metrics = self.api_client.get_cost_metrics(filters["time_range"])
        performance_metrics = self.api_client.get_performance_metrics(filters["time_range"])
        
        # Combine data for analytics
        analytics_data = {
            "system": system_status,
            "workloads": workload_metrics,
            "costs": cost_metrics,
            "performance": performance_metrics,
            **cost_metrics,  # Include cost breakdown data
            **performance_metrics  # Include performance data
        }
        
        # Render interactive analytics interface
        analytics_results = InteractiveAnalytics.render_analytics_interface(analytics_data)
        
        # Store analytics results in session state for persistence
        st.session_state["analytics_results"] = analytics_results
    


def main():
    """Main dashboard application"""
    dashboard = DashboardUI()
    
    # Initialize session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Render sidebar and get filters
    filters = dashboard.render_sidebar()
    
    # Render header and navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = dashboard.render_header()
    
    # Render content based on selected tab
    with tab1:
        dashboard.render_overview_tab(filters)
    
    with tab2:
        dashboard.render_workloads_tab(filters)
    
    with tab3:
        dashboard.render_costs_tab(filters)
    
    with tab4:
        dashboard.render_performance_tab(filters)
    
    with tab5:
        dashboard.render_alerts_tab(filters)
    
    with tab6:
        dashboard.render_analytics_tab(filters)
    
    with tab7:
        dashboard.render_reports_tab(filters)
    
    # Auto-refresh functionality
    if filters["auto_refresh"]:
        # Check if enough time has passed since last refresh
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= filters["refresh_interval"]:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        else:
            # Show countdown to next refresh
            remaining_time = filters["refresh_interval"] - time_since_refresh
            st.sidebar.info(f"Next refresh in {remaining_time:.0f} seconds")
            
            # Use a placeholder to trigger refresh
            time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    main()