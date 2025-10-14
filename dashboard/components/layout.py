"""
Dashboard layout components and utilities
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LayoutManager:
    """Manages dashboard layout and responsive design"""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Optional[Any] = None, 
                          help_text: Optional[str] = None):
        """Create a styled metric card"""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
    
    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️"):
        """Create an information card with custom styling"""
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        ">
            <h4 style="margin: 0; color: #1f77b4;">{icon} {title}</h4>
            <p style="margin: 0.5rem 0 0 0;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_status_indicator(status: str, label: str = "Status"):
        """Create a status indicator with color coding"""
        colors = {
            "healthy": "#28a745",
            "warning": "#ffc107", 
            "critical": "#dc3545",
            "unknown": "#6c757d"
        }
        
        color = colors.get(status.lower(), colors["unknown"])
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <div style="
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: {color};
                margin-right: 8px;
            "></div>
            <span style="font-weight: 500;">{label}: {status.title()}</span>
        </div>
        """, unsafe_allow_html=True)

class ChartFactory:
    """Factory for creating standardized charts"""
    
    @staticmethod
    def create_time_series_chart(data: Dict, title: str, y_label: str = "Value"):
        """Create a standardized time series chart"""
        fig = go.Figure()
        
        if "timeline" in data:
            fig.add_trace(go.Scatter(
                x=data["timeline"]["timestamps"],
                y=data["timeline"]["values"],
                mode='lines+markers',
                name=y_label,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(data: Dict, title: str, chart_type: str = "pie"):
        """Create a distribution chart (pie or bar)"""
        if chart_type == "pie":
            fig = go.Figure(data=[go.Pie(
                labels=list(data.keys()),
                values=list(data.values()),
                hole=0.3
            )])
        else:  # bar chart
            fig = go.Figure(data=[go.Bar(
                x=list(data.keys()),
                y=list(data.values())
            )])
        
        fig.update_layout(
            title=title,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_multi_metric_chart(data: Dict, title: str, metrics: List[str]):
        """Create a chart with multiple metrics"""
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            if metric in data:
                fig.add_trace(
                    go.Scatter(
                        x=data[metric].get("timestamps", []),
                        y=data[metric].get("values", []),
                        mode='lines',
                        name=metric,
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            height=150 * len(metrics),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig

class NavigationManager:
    """Manages dashboard navigation and routing"""
    
    @staticmethod
    def create_sidebar_navigation():
        """Create sidebar navigation menu"""
        st.sidebar.markdown("## Navigation")
        
        pages = {
            "🏠 Overview": "overview",
            "⚙️ Workloads": "workloads", 
            "💰 Costs": "costs",
            "📊 Performance": "performance",
            "🚨 Alerts": "alerts",
            "📋 Reports": "reports"
        }
        
        selected_page = st.sidebar.radio(
            "Select Page",
            options=list(pages.keys()),
            index=0
        )
        
        return pages[selected_page]
    
    @staticmethod
    def create_breadcrumb(path: List[str]):
        """Create breadcrumb navigation"""
        breadcrumb = " > ".join(path)
        st.markdown(f"**Navigation:** {breadcrumb}")

class FilterManager:
    """Manages dashboard filters and controls"""
    
    @staticmethod
    def create_time_filter():
        """Create time range filter"""
        time_options = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 12 Hours": 12,
            "Last 24 Hours": 24,
            "Last 3 Days": 72,
            "Last Week": 168
        }
        
        selected = st.selectbox(
            "Time Range",
            options=list(time_options.keys()),
            index=3  # Default to 24 hours
        )
        
        return time_options[selected]
    
    @staticmethod
    def create_provider_filter():
        """Create cloud provider filter"""
        providers = ["AWS", "GCP", "Azure", "Other"]
        
        selected = st.multiselect(
            "Cloud Providers",
            options=providers,
            default=providers[:3]  # Default to AWS, GCP, Azure
        )
        
        return selected
    
    @staticmethod
    def create_resource_filter():
        """Create resource type filter"""
        resources = ["Compute", "Storage", "Network", "Database", "Other"]
        
        selected = st.multiselect(
            "Resource Types",
            options=resources,
            default=resources
        )
        
        return selected
    
    @staticmethod
    def create_custom_filters():
        """Create custom filter interface"""
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                cost_range = st.slider(
                    "Cost Range ($)",
                    min_value=0,
                    max_value=10000,
                    value=(0, 1000),
                    step=50
                )
            
            with col2:
                utilization_range = st.slider(
                    "CPU Utilization (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=5
                )
            
            tags = st.text_input("Filter by Tags (comma-separated)")
            
            return {
                "cost_range": cost_range,
                "utilization_range": utilization_range,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]
            }

class ThemeManager:
    """Manages dashboard themes and styling"""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        .status-healthy {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-critical {
            color: #dc3545;
            font-weight: bold;
        }
        
        .chart-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_color_palette():
        """Get standardized color palette"""
        return {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e", 
            "success": "#2ca02c",
            "warning": "#ffc107",
            "danger": "#d62728",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40"
        }