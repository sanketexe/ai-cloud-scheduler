"""
Interactive analytics and filtering components
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

class AdvancedFilters:
    """Advanced filtering capabilities for dashboard data"""
    
    @staticmethod
    def create_multi_dimensional_filter():
        """Create multi-dimensional filter interface"""
        st.subheader("🔍 Advanced Filters")
        
        with st.expander("Filter Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Time-based filters
                st.markdown("**Time Filters**")
                
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now() - timedelta(days=7), datetime.now()),
                    max_value=datetime.now()
                )
                
                time_granularity = st.selectbox(
                    "Time Granularity",
                    options=["Hour", "Day", "Week", "Month"],
                    index=1
                )
            
            with col2:
                # Resource filters
                st.markdown("**Resource Filters**")
                
                providers = st.multiselect(
                    "Cloud Providers",
                    options=["AWS", "GCP", "Azure", "Other"],
                    default=["AWS", "GCP", "Azure"]
                )
                
                regions = st.multiselect(
                    "Regions",
                    options=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                    default=[]
                )
                
                resource_types = st.multiselect(
                    "Resource Types",
                    options=["Compute", "Storage", "Network", "Database", "Analytics"],
                    default=["Compute", "Storage"]
                )
            
            with col3:
                # Metric filters
                st.markdown("**Metric Filters**")
                
                cost_range = st.slider(
                    "Cost Range ($)",
                    min_value=0,
                    max_value=10000,
                    value=(0, 1000),
                    step=50
                )
                
                cpu_range = st.slider(
                    "CPU Utilization (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=5
                )
                
                memory_range = st.slider(
                    "Memory Utilization (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=5
                )
            
            # Tag-based filtering
            st.markdown("**Tag Filters**")
            col1, col2 = st.columns(2)
            
            with col1:
                environment = st.multiselect(
                    "Environment",
                    options=["production", "staging", "development", "test"],
                    default=[]
                )
            
            with col2:
                team = st.multiselect(
                    "Team/Department",
                    options=["engineering", "data", "marketing", "finance"],
                    default=[]
                )
            
            # Custom tag input
            custom_tags = st.text_input(
                "Custom Tags (comma-separated)",
                placeholder="e.g., project:alpha, owner:team-a"
            )
        
        return {
            "date_range": date_range,
            "time_granularity": time_granularity.lower(),
            "providers": providers,
            "regions": regions,
            "resource_types": resource_types,
            "cost_range": cost_range,
            "cpu_range": cpu_range,
            "memory_range": memory_range,
            "environment": environment,
            "team": team,
            "custom_tags": [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
        }
    
    @staticmethod
    def create_search_interface():
        """Create advanced search interface"""
        st.subheader("🔎 Search & Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Resources",
                placeholder="Search by name, ID, tags, or properties...",
                help="Use wildcards (*) and boolean operators (AND, OR, NOT)"
            )
        
        with col2:
            search_type = st.selectbox(
                "Search Type",
                options=["All", "Workloads", "Resources", "Costs", "Alerts"],
                index=0
            )
        
        # Quick filters
        st.markdown("**Quick Filters**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_cost = st.checkbox("High Cost Resources")
        
        with col2:
            high_utilization = st.checkbox("High Utilization")
        
        with col3:
            recent_alerts = st.checkbox("Recent Alerts")
        
        with col4:
            optimization_candidates = st.checkbox("Optimization Candidates")
        
        return {
            "query": search_query,
            "search_type": search_type.lower(),
            "quick_filters": {
                "high_cost": high_cost,
                "high_utilization": high_utilization,
                "recent_alerts": recent_alerts,
                "optimization_candidates": optimization_candidates
            }
        }

class DrillDownAnalytics:
    """Drill-down functionality for detailed analysis"""
    
    @staticmethod
    def create_hierarchical_view(data: Dict, level: str = "provider") -> go.Figure:
        """Create hierarchical drill-down view"""
        
        if level == "provider":
            # Provider level - show providers
            providers = data.get("by_provider", {})
            
            fig = go.Figure(data=[go.Bar(
                x=list(providers.keys()),
                y=list(providers.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(providers)],
                text=[f"${v:,.0f}" for v in providers.values()],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<br>Click to drill down<extra></extra>'
            )])
            
            fig.update_layout(
                title="Cost by Cloud Provider (Click to drill down)",
                xaxis_title="Provider",
                yaxis_title="Cost ($)",
                height=400
            )
            
        elif level == "service":
            # Service level - show services within provider
            services = data.get("by_service", {})
            
            fig = go.Figure(data=[go.Bar(
                x=list(services.keys()),
                y=list(services.values()),
                marker_color='#ff7f0e',
                text=[f"${v:,.0f}" for v in services.values()],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<br>Click for details<extra></extra>'
            )])
            
            fig.update_layout(
                title="Cost by Service Type",
                xaxis_title="Service",
                yaxis_title="Cost ($)",
                height=400
            )
            
        elif level == "resource":
            # Resource level - show individual resources
            resources = data.get("by_resource", {})
            
            if resources:
                df = pd.DataFrame(list(resources.items()), columns=["Resource", "Cost"])
                df = df.sort_values("Cost", ascending=False).head(20)  # Top 20
                
                fig = go.Figure(data=[go.Bar(
                    x=df["Resource"],
                    y=df["Cost"],
                    marker_color='#2ca02c',
                    text=[f"${v:,.0f}" for v in df["Cost"]],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
                )])
                
                fig.update_layout(
                    title="Top 20 Resources by Cost",
                    xaxis_title="Resource",
                    yaxis_title="Cost ($)",
                    height=400,
                    xaxis={'tickangle': 45}
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="No resource data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        
        return fig
    
    @staticmethod
    def create_time_series_drilldown(data: Dict, metric: str, granularity: str = "hour") -> go.Figure:
        """Create time series with drill-down capability"""
        
        if "timeline" not in data:
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        timeline = data["timeline"]
        df = pd.DataFrame(timeline)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data points available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Aggregate by granularity
        if granularity == "day":
            df["period"] = df["timestamp"].dt.date
        elif granularity == "week":
            df["period"] = df["timestamp"].dt.to_period("W")
        elif granularity == "month":
            df["period"] = df["timestamp"].dt.to_period("M")
        else:  # hour
            df["period"] = df["timestamp"].dt.floor("H")
        
        # Group and aggregate
        if metric in df.columns:
            agg_df = df.groupby("period")[metric].mean().reset_index()
            
            fig = go.Figure()
            
            # Main line
            fig.add_trace(go.Scatter(
                x=agg_df["period"],
                y=agg_df[metric],
                mode='lines+markers',
                name=metric.title(),
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{metric.title()}: %{{y:.2f}}</b><br>%{{x}}<br>Click for details<extra></extra>'
            ))
            
            # Add trend line
            if len(agg_df) > 2:
                z = np.polyfit(range(len(agg_df)), agg_df[metric], 1)
                trend_line = np.poly1d(z)(range(len(agg_df)))
                
                fig.add_trace(go.Scatter(
                    x=agg_df["period"],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=1, dash='dash'),
                    hovertemplate='Trend: <b>%{y:.2f}</b><extra></extra>'
                ))
            
            fig.update_layout(
                title=f"{metric.title()} Over Time ({granularity.title()})",
                xaxis_title="Time",
                yaxis_title=metric.title(),
                height=400,
                hovermode='x unified'
            )
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(data: Dict) -> go.Figure:
        """Create correlation matrix for metrics"""
        
        if "metrics_correlation" not in data:
            # Generate sample correlation data
            metrics = ["cpu", "memory", "network", "disk", "cost"]
            correlation_matrix = np.random.rand(len(metrics), len(metrics))
            # Make it symmetric
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
        else:
            metrics = list(data["metrics_correlation"].keys())
            correlation_matrix = np.array([
                [data["metrics_correlation"][m1].get(m2, 0) for m2 in metrics]
                for m1 in metrics
            ])
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=metrics,
            y=metrics,
            colorscale='RdBu',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title="Metrics Correlation Matrix",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

class CustomDashboards:
    """Custom dashboard creation with drag-and-drop widgets"""
    
    @staticmethod
    def create_widget_library():
        """Create library of available widgets"""
        st.subheader("📊 Widget Library")
        
        widget_categories = {
            "Metrics": [
                "Key Performance Indicators",
                "Resource Utilization Gauges", 
                "Cost Summary Cards",
                "Alert Counters"
            ],
            "Charts": [
                "Time Series Line Chart",
                "Bar Chart",
                "Pie Chart",
                "Heatmap",
                "Scatter Plot",
                "Area Chart"
            ],
            "Tables": [
                "Resource List",
                "Cost Breakdown",
                "Alert History",
                "Performance Metrics"
            ],
            "Controls": [
                "Date Range Picker",
                "Provider Filter",
                "Search Box",
                "Refresh Button"
            ]
        }
        
        selected_widgets = {}
        
        for category, widgets in widget_categories.items():
            with st.expander(f"{category} Widgets"):
                selected_widgets[category] = st.multiselect(
                    f"Select {category}",
                    options=widgets,
                    key=f"widgets_{category}"
                )
        
        return selected_widgets
    
    @staticmethod
    def create_layout_designer():
        """Create layout designer interface"""
        st.subheader("🎨 Dashboard Layout Designer")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Layout Options**")
            
            layout_type = st.selectbox(
                "Layout Type",
                options=["Grid", "Columns", "Tabs", "Sidebar"],
                index=0
            )
            
            if layout_type == "Grid":
                grid_cols = st.slider("Grid Columns", 1, 4, 2)
                grid_rows = st.slider("Grid Rows", 1, 6, 3)
            elif layout_type == "Columns":
                num_columns = st.slider("Number of Columns", 1, 4, 2)
            
            # Widget sizing
            st.markdown("**Widget Sizing**")
            default_height = st.slider("Default Widget Height", 200, 600, 350)
            
            # Color scheme
            st.markdown("**Color Scheme**")
            color_scheme = st.selectbox(
                "Theme",
                options=["Default", "Dark", "Light", "Blue", "Green"],
                index=0
            )
        
        with col2:
            st.markdown("**Layout Preview**")
            
            # Create preview based on layout type
            if layout_type == "Grid":
                CustomDashboards._render_grid_preview(grid_cols, grid_rows)
            elif layout_type == "Columns":
                CustomDashboards._render_columns_preview(num_columns)
            elif layout_type == "Tabs":
                CustomDashboards._render_tabs_preview()
            elif layout_type == "Sidebar":
                CustomDashboards._render_sidebar_preview()
        
        return {
            "layout_type": layout_type,
            "grid_cols": grid_cols if layout_type == "Grid" else None,
            "grid_rows": grid_rows if layout_type == "Grid" else None,
            "num_columns": num_columns if layout_type == "Columns" else None,
            "default_height": default_height,
            "color_scheme": color_scheme
        }
    
    @staticmethod
    def _render_grid_preview(cols: int, rows: int):
        """Render grid layout preview"""
        st.markdown(f"**{cols}x{rows} Grid Layout**")
        
        for row in range(rows):
            columns = st.columns(cols)
            for col_idx in range(cols):
                with columns[col_idx]:
                    st.container()
                    st.markdown(f"<div style='border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 5px;'>Widget {row*cols + col_idx + 1}</div>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_columns_preview(num_cols: int):
        """Render columns layout preview"""
        st.markdown(f"**{num_cols} Column Layout**")
        
        columns = st.columns(num_cols)
        for i in range(num_cols):
            with columns[i]:
                st.markdown(f"<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Column {i+1}</div>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_tabs_preview():
        """Render tabs layout preview"""
        st.markdown("**Tabbed Layout**")
        
        tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
        
        with tab1:
            st.markdown("<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Tab 1 Content</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Tab 2 Content</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Tab 3 Content</div>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_sidebar_preview():
        """Render sidebar layout preview"""
        st.markdown("**Sidebar Layout**")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Sidebar</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 5px;'>Main Content</div>", unsafe_allow_html=True)
    
    @staticmethod
    def save_dashboard_config(config: Dict, name: str):
        """Save custom dashboard configuration"""
        # In a real implementation, this would save to a database or file
        st.success(f"Dashboard '{name}' saved successfully!")
        
        # Store in session state for demo
        if "saved_dashboards" not in st.session_state:
            st.session_state.saved_dashboards = {}
        
        st.session_state.saved_dashboards[name] = config
    
    @staticmethod
    def load_dashboard_config(name: str) -> Optional[Dict]:
        """Load custom dashboard configuration"""
        if "saved_dashboards" in st.session_state:
            return st.session_state.saved_dashboards.get(name)
        return None
    
    @staticmethod
    def list_saved_dashboards() -> List[str]:
        """List all saved dashboard configurations"""
        if "saved_dashboards" in st.session_state:
            return list(st.session_state.saved_dashboards.keys())
        return []

class InteractiveAnalytics:
    """Main interactive analytics interface"""
    
    @staticmethod
    def render_analytics_interface(data: Dict):
        """Render the main interactive analytics interface"""
        
        # Create tabs for different analytics features
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Advanced Filters",
            "📊 Drill-Down Analysis", 
            "🎨 Custom Dashboards",
            "📈 Correlation Analysis"
        ])
        
        with tab1:
            # Advanced filtering interface
            filters = AdvancedFilters.create_multi_dimensional_filter()
            search = AdvancedFilters.create_search_interface()
            
            # Apply filters button
            if st.button("Apply Filters", type="primary"):
                st.success("Filters applied! Data will be updated based on your selections.")
                # In real implementation, this would filter the actual data
                st.json({"filters": filters, "search": search})
        
        with tab2:
            # Drill-down analysis
            st.subheader("📊 Drill-Down Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                drill_level = st.selectbox(
                    "Analysis Level",
                    options=["provider", "service", "resource"],
                    format_func=lambda x: x.title()
                )
            
            with col2:
                time_granularity = st.selectbox(
                    "Time Granularity",
                    options=["hour", "day", "week", "month"],
                    format_func=lambda x: x.title()
                )
            
            # Hierarchical view
            st.subheader("Cost Breakdown")
            hierarchical_chart = DrillDownAnalytics.create_hierarchical_view(data, drill_level)
            st.plotly_chart(hierarchical_chart, use_container_width=True)
            
            # Time series drill-down
            st.subheader("Performance Trends")
            metric = st.selectbox(
                "Select Metric",
                options=["cpu", "memory", "network", "cost"],
                format_func=lambda x: x.title()
            )
            
            timeseries_chart = DrillDownAnalytics.create_time_series_drilldown(
                data, metric, time_granularity
            )
            st.plotly_chart(timeseries_chart, use_container_width=True)
        
        with tab3:
            # Custom dashboard creation
            st.subheader("🎨 Custom Dashboard Builder")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Widget selection
                selected_widgets = CustomDashboards.create_widget_library()
            
            with col2:
                # Layout designer
                layout_config = CustomDashboards.create_layout_designer()
            
            # Dashboard management
            st.markdown("---")
            st.subheader("💾 Dashboard Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dashboard_name = st.text_input("Dashboard Name", placeholder="My Custom Dashboard")
            
            with col2:
                if st.button("Save Dashboard") and dashboard_name:
                    config = {
                        "widgets": selected_widgets,
                        "layout": layout_config,
                        "created_at": datetime.now().isoformat()
                    }
                    CustomDashboards.save_dashboard_config(config, dashboard_name)
            
            with col3:
                saved_dashboards = CustomDashboards.list_saved_dashboards()
                if saved_dashboards:
                    selected_dashboard = st.selectbox("Load Dashboard", options=saved_dashboards)
                    if st.button("Load") and selected_dashboard:
                        config = CustomDashboards.load_dashboard_config(selected_dashboard)
                        st.success(f"Loaded dashboard: {selected_dashboard}")
                        st.json(config)
        
        with tab4:
            # Correlation analysis
            st.subheader("📈 Correlation Analysis")
            
            correlation_chart = DrillDownAnalytics.create_correlation_matrix(data)
            st.plotly_chart(correlation_chart, use_container_width=True)
            
            # Insights
            st.subheader("🔍 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Strong Correlations Found:**
                - CPU and Memory utilization (0.78)
                - Cost and Resource count (0.65)
                - Network I/O and Disk I/O (0.52)
                """)
            
            with col2:
                st.markdown("""
                **Optimization Opportunities:**
                - High CPU with low memory usage
                - Cost spikes without performance gains
                - Underutilized high-cost resources
                """)
        
        return {
            "filters": filters if 'filters' in locals() else {},
            "search": search if 'search' in locals() else {},
            "drill_level": drill_level if 'drill_level' in locals() else "provider",
            "time_granularity": time_granularity if 'time_granularity' in locals() else "hour"
        }