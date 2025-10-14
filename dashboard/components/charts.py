"""
Real-time chart components for the dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

class RealTimeCharts:
    """Real-time chart components with live data updates"""
    
    @staticmethod
    def create_workload_status_chart(data: Dict) -> go.Figure:
        """Create real-time workload status chart"""
        if not data or "workloads" not in data:
            return RealTimeCharts._create_empty_chart("No workload data available")
        
        workloads = data["workloads"]
        status_counts = {}
        
        for workload in workloads:
            status = workload.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        colors = {
            "running": "#28a745",
            "pending": "#ffc107",
            "failed": "#dc3545",
            "completed": "#17a2b8",
            "unknown": "#6c757d"
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            marker_colors=[colors.get(status, "#6c757d") for status in status_counts.keys()],
            hole=0.4,
            textinfo='label+percent+value',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Workload Status Distribution",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        
        return fig
    
    @staticmethod
    def create_cost_timeline_chart(data: Dict) -> go.Figure:
        """Create real-time cost timeline chart"""
        if not data or "timeline" not in data:
            return RealTimeCharts._create_empty_chart("No cost data available")
        
        timeline = data["timeline"]
        df = pd.DataFrame(timeline)
        
        if df.empty:
            return RealTimeCharts._create_empty_chart("No cost timeline data")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        fig = go.Figure()
        
        # Add cost line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["cost"],
            mode='lines+markers',
            name='Hourly Cost',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<extra></extra>'
        ))
        
        # Add trend line if enough data points
        if len(df) > 2:
            z = np.polyfit(range(len(df)), df["cost"], 1)
            trend_line = np.poly1d(z)(range(len(df)))
            
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='#ff7f0e', width=1, dash='dash'),
                hovertemplate='Trend: <b>%{y:$,.2f}</b><extra></extra>'
            ))
        
        fig.update_layout(
            title="Cost Timeline (Real-time)",
            xaxis_title="Time",
            yaxis_title="Cost ($)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_performance_heatmap(data: Dict) -> go.Figure:
        """Create performance metrics heatmap"""
        if not data or "metrics" not in data:
            return RealTimeCharts._create_empty_chart("No performance data available")
        
        metrics = data["metrics"]
        
        # Create sample heatmap data if real data structure is different
        resources = list(metrics.keys()) if metrics else ["Resource-1", "Resource-2", "Resource-3"]
        metric_types = ["CPU", "Memory", "Network", "Disk"]
        
        # Generate heatmap matrix
        z_data = []
        for resource in resources:
            row = []
            for metric_type in metric_types:
                if resource in metrics and metric_type.lower() in metrics[resource]:
                    value = metrics[resource][metric_type.lower()]
                else:
                    value = np.random.uniform(20, 90)  # Sample data
                row.append(value)
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=metric_types,
            y=resources,
            colorscale='RdYlGn_r',
            text=[[f"{val:.1f}%" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}: <b>%{z:.1f}%</b><extra></extra>'
        ))
        
        fig.update_layout(
            title="Resource Performance Heatmap",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_provider_comparison_chart(data: Dict) -> go.Figure:
        """Create cloud provider comparison chart"""
        if not data or "providers" not in data:
            return RealTimeCharts._create_empty_chart("No provider data available")
        
        providers_data = data["providers"]
        
        metrics = ["cost", "performance", "availability"]
        providers = list(providers_data.keys())
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metrics):
            values = []
            for provider in providers:
                if provider in providers_data and metric in providers_data[provider]:
                    values.append(providers_data[provider][metric])
                else:
                    values.append(np.random.uniform(70, 95))  # Sample data
            
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=providers,
                y=values,
                marker_color=colors[i],
                text=[f"{v:.1f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Cloud Provider Comparison",
            xaxis_title="Provider",
            yaxis_title="Score",
            barmode='group',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        
        return fig
    
    @staticmethod
    def create_resource_utilization_gauge(data: Dict, resource_type: str = "cpu") -> go.Figure:
        """Create resource utilization gauge chart"""
        if not data or resource_type not in data:
            utilization = np.random.uniform(30, 85)  # Sample data
        else:
            utilization = data[resource_type]
        
        # Determine color based on utilization
        if utilization < 50:
            color = "#28a745"  # Green
        elif utilization < 80:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=utilization,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{resource_type.upper()} Utilization"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_alert_timeline_chart(data: Dict) -> go.Figure:
        """Create alert timeline chart"""
        if not data or "alerts" not in data:
            return RealTimeCharts._create_empty_chart("No alert data available")
        
        alerts = data["alerts"]
        df = pd.DataFrame(alerts)
        
        if df.empty:
            return RealTimeCharts._create_empty_chart("No alerts in timeline")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Color mapping for severity
        severity_colors = {
            "critical": "#dc3545",
            "warning": "#ffc107", 
            "info": "#17a2b8",
            "low": "#28a745"
        }
        
        fig = go.Figure()
        
        for severity in df["severity"].unique():
            severity_data = df[df["severity"] == severity]
            
            fig.add_trace(go.Scatter(
                x=severity_data["timestamp"],
                y=[severity] * len(severity_data),
                mode='markers',
                name=severity.title(),
                marker=dict(
                    color=severity_colors.get(severity, "#6c757d"),
                    size=10,
                    symbol='circle'
                ),
                text=severity_data["message"],
                hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Alert Timeline",
            xaxis_title="Time",
            yaxis_title="Severity",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def _create_empty_chart(message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

class InteractiveCharts:
    """Interactive chart components with drill-down capabilities"""
    
    @staticmethod
    def create_drilldown_cost_chart(data: Dict, level: str = "provider") -> go.Figure:
        """Create cost chart with drill-down capability"""
        if not data:
            return RealTimeCharts._create_empty_chart("No cost data available")
        
        if level == "provider":
            # Provider level view
            providers = data.get("by_provider", {})
            
            fig = go.Figure(data=[go.Bar(
                x=list(providers.keys()),
                y=list(providers.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(providers)],
                text=[f"${v:,.2f}" for v in providers.values()],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Cost by Cloud Provider (Click to drill down)",
                xaxis_title="Provider",
                yaxis_title="Cost ($)",
                height=350
            )
            
        elif level == "service":
            # Service level view
            services = data.get("by_service", {})
            
            fig = go.Figure(data=[go.Bar(
                x=list(services.keys()),
                y=list(services.values()),
                marker_color='#ff7f0e',
                text=[f"${v:,.2f}" for v in services.values()],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Cost by Service Type",
                xaxis_title="Service",
                yaxis_title="Cost ($)",
                height=350
            )
        
        return fig
    
    @staticmethod
    def create_interactive_performance_chart(data: Dict, metric: str = "cpu") -> go.Figure:
        """Create interactive performance chart with metric selection"""
        if not data or "timeline" not in data:
            return RealTimeCharts._create_empty_chart("No performance data available")
        
        timeline = data["timeline"]
        df = pd.DataFrame(timeline)
        
        if df.empty or metric not in df.columns:
            return RealTimeCharts._create_empty_chart(f"No {metric} data available")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        fig = go.Figure()
        
        # Add main metric line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[metric],
            mode='lines+markers',
            name=metric.upper(),
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{metric.upper()}: %{{y:.1f}}%</b><br>%{{x}}<extra></extra>'
        ))
        
        # Add threshold lines
        if metric == "cpu":
            fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                         annotation_text="Warning Threshold")
            fig.add_hline(y=95, line_dash="dash", line_color="red", 
                         annotation_text="Critical Threshold")
        
        fig.update_layout(
            title=f"{metric.upper()} Performance Over Time",
            xaxis_title="Time",
            yaxis_title=f"{metric.upper()} (%)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig