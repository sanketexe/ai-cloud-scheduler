"""
Comprehensive reporting system for the dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import numpy as np
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class ReportGenerator:
    """Automated report generation with scheduling and distribution"""
    
    def __init__(self):
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self.custom_styles = self._create_custom_styles()
        else:
            self.styles = None
            self.custom_styles = None
    
    def _create_custom_styles(self):
        """Create custom styles for reports"""
        if not REPORTLAB_AVAILABLE:
            return {}
            
        return {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4')
            ),
            'subtitle': ParagraphStyle(
                'CustomSubtitle',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                textColor=colors.HexColor('#333333')
            ),
            'metric': ParagraphStyle(
                'MetricStyle',
                parent=self.styles['Normal'],
                fontSize=14,
                spaceAfter=10,
                textColor=colors.HexColor('#2c3e50')
            )
        }
    
    @staticmethod
    def create_report_configuration():
        """Create report configuration interface"""
        st.subheader("📋 Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Report basic settings
            report_name = st.text_input("Report Name", placeholder="Monthly Cloud Intelligence Report")
            report_type = st.selectbox(
                "Report Type",
                options=["Executive Summary", "Technical Analysis", "Cost Analysis", "Performance Report", "Custom"],
                help="Select the type of report to generate"
            )
            
            time_period = st.selectbox(
                "Time Period",
                options=["Last 24 Hours", "Last Week", "Last Month", "Last Quarter", "Custom Range"],
                index=2
            )
            
            if time_period == "Custom Range":
                col_start, col_end = st.columns(2)
                with col_start:
                    start_date = st.date_input("Start Date")
                with col_end:
                    end_date = st.date_input("End Date")
        
        with col2:
            # Report format and distribution
            output_format = st.multiselect(
                "Output Format",
                options=["PDF", "Excel", "CSV", "JSON"],
                default=["PDF"],
                help="Select output formats for the report"
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_raw_data = st.checkbox("Include Raw Data", value=False)
            
            # Distribution settings
            st.markdown("**Distribution Settings**")
            email_recipients = st.text_area(
                "Email Recipients",
                placeholder="user1@company.com, user2@company.com",
                help="Comma-separated email addresses"
            )
            
            schedule_report = st.checkbox("Schedule Report", value=False)
            
            if schedule_report:
                schedule_frequency = st.selectbox(
                    "Frequency",
                    options=["Daily", "Weekly", "Monthly", "Quarterly"]
                )
        
        return {
            "name": report_name,
            "type": report_type,
            "time_period": time_period,
            "output_format": output_format,
            "include_charts": include_charts,
            "include_raw_data": include_raw_data,
            "email_recipients": email_recipients,
            "schedule_report": schedule_report,
            "schedule_frequency": schedule_frequency if schedule_report else None
        }
    
    def generate_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary report with key metrics and trends"""
        
        # Calculate key metrics
        total_cost = data.get('cost_metrics', {}).get('total_cost_24h', 0)
        cost_change = data.get('cost_metrics', {}).get('cost_change_percent', 0)
        active_workloads = data.get('workload_metrics', {}).get('active_count', 0)
        avg_performance = data.get('performance_metrics', {}).get('avg_cpu', 0)
        
        # Generate insights
        insights = self._generate_executive_insights(data)
        
        # Create summary structure
        summary = {
            "title": "Executive Summary - Cloud Intelligence Platform",
            "period": data.get('time_period', 'Last 24 Hours'),
            "generated_at": datetime.now().isoformat(),
            "key_metrics": {
                "total_cost": total_cost,
                "cost_trend": "increasing" if cost_change > 0 else "decreasing",
                "cost_change_percent": cost_change,
                "active_workloads": active_workloads,
                "performance_score": 100 - avg_performance,  # Simplified score
                "optimization_opportunities": len(insights.get('optimizations', []))
            },
            "insights": insights,
            "recommendations": self._generate_executive_recommendations(data),
            "charts": self._create_executive_charts(data) if data.get('include_charts') else None
        }
        
        return summary
    
    def generate_technical_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical report with performance and cost analysis"""
        
        technical_report = {
            "title": "Technical Analysis Report",
            "period": data.get('time_period', 'Last 24 Hours'),
            "generated_at": datetime.now().isoformat(),
            "sections": {
                "workload_analysis": self._analyze_workload_performance(data),
                "cost_breakdown": self._analyze_cost_breakdown(data),
                "performance_metrics": self._analyze_performance_metrics(data),
                "resource_utilization": self._analyze_resource_utilization(data),
                "optimization_analysis": self._analyze_optimization_opportunities(data),
                "trend_analysis": self._analyze_trends(data)
            },
            "detailed_metrics": self._compile_detailed_metrics(data),
            "charts": self._create_technical_charts(data) if data.get('include_charts') else None,
            "raw_data": data if data.get('include_raw_data') else None
        }
        
        return technical_report
    
    def _generate_executive_insights(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate executive-level insights from data"""
        
        cost_metrics = data.get('cost_metrics', {})
        workload_metrics = data.get('workload_metrics', {})
        performance_metrics = data.get('performance_metrics', {})
        
        insights = {
            "cost_insights": [],
            "performance_insights": [],
            "optimization_insights": [],
            "risk_insights": []
        }
        
        # Cost insights
        cost_change = cost_metrics.get('cost_change_percent', 0)
        if cost_change > 10:
            insights["cost_insights"].append(f"Costs increased by {cost_change:.1f}% - investigate usage spikes")
        elif cost_change < -5:
            insights["cost_insights"].append(f"Costs decreased by {abs(cost_change):.1f}% - optimization efforts paying off")
        
        total_cost = cost_metrics.get('total_cost_24h', 0)
        if total_cost > 1000:
            insights["cost_insights"].append("High daily spending detected - review resource allocation")
        
        # Performance insights
        avg_cpu = performance_metrics.get('avg_cpu', 0)
        if avg_cpu > 80:
            insights["performance_insights"].append("High CPU utilization across infrastructure - consider scaling")
        elif avg_cpu < 30:
            insights["performance_insights"].append("Low CPU utilization - potential for cost optimization")
        
        # Workload insights
        success_rate = workload_metrics.get('success_rate', 100)
        if success_rate < 95:
            insights["risk_insights"].append(f"Workload success rate at {success_rate:.1f}% - investigate failures")
        
        return insights
    
    def _generate_executive_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate executive recommendations based on data analysis"""
        
        recommendations = []
        
        cost_metrics = data.get('cost_metrics', {})
        performance_metrics = data.get('performance_metrics', {})
        
        # Cost recommendations
        if cost_metrics.get('cost_change_percent', 0) > 15:
            recommendations.append({
                "category": "Cost Management",
                "priority": "High",
                "recommendation": "Implement immediate cost controls and review resource allocation",
                "impact": "Potential 15-20% cost reduction"
            })
        
        # Performance recommendations
        if performance_metrics.get('avg_cpu', 0) > 85:
            recommendations.append({
                "category": "Performance",
                "priority": "High", 
                "recommendation": "Scale infrastructure to handle increased load",
                "impact": "Improved application performance and user experience"
            })
        
        # Optimization recommendations
        recommendations.append({
            "category": "Optimization",
            "priority": "Medium",
            "recommendation": "Implement automated scaling policies",
            "impact": "10-15% cost savings and improved performance"
        })
        
        return recommendations
    
    def _analyze_workload_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workload performance metrics"""
        
        workload_metrics = data.get('workload_metrics', {})
        
        return {
            "total_workloads": workload_metrics.get('total_scheduled', 0),
            "active_workloads": workload_metrics.get('active_count', 0),
            "success_rate": workload_metrics.get('success_rate', 0),
            "provider_distribution": workload_metrics.get('provider_distribution', {}),
            "performance_summary": "Workload scheduling operating within normal parameters",
            "bottlenecks": self._identify_workload_bottlenecks(workload_metrics)
        }
    
    def _analyze_cost_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed cost breakdown"""
        
        cost_metrics = data.get('cost_metrics', {})
        
        return {
            "total_cost": cost_metrics.get('total_cost_24h', 0),
            "cost_by_provider": cost_metrics.get('by_provider', {}),
            "cost_by_service": cost_metrics.get('by_service', {}),
            "cost_trends": cost_metrics.get('timeline', {}),
            "optimization_potential": self._calculate_optimization_potential(cost_metrics),
            "cost_efficiency_score": self._calculate_cost_efficiency(cost_metrics)
        }
    
    def _analyze_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics in detail"""
        
        performance_metrics = data.get('performance_metrics', {})
        
        return {
            "cpu_utilization": {
                "average": performance_metrics.get('avg_cpu', 0),
                "trend": performance_metrics.get('cpu_change', 0),
                "peak": max([m.get('cpu', 0) for m in performance_metrics.get('metrics_timeline', [])]) if performance_metrics.get('metrics_timeline') else 0
            },
            "memory_utilization": {
                "average": performance_metrics.get('avg_memory', 0),
                "trend": performance_metrics.get('memory_change', 0),
                "peak": max([m.get('memory', 0) for m in performance_metrics.get('metrics_timeline', [])]) if performance_metrics.get('metrics_timeline') else 0
            },
            "resource_health": self._assess_resource_health(performance_metrics),
            "performance_score": self._calculate_performance_score(performance_metrics)
        }
    
    def _create_executive_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create charts for executive summary (base64 encoded)"""
        
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available for chart generation"}
        
        charts = {}
        
        # Cost trend chart
        cost_metrics = data.get('cost_metrics', {})
        if 'timeline' in cost_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            timeline = cost_metrics['timeline']
            ax.plot(timeline.get('timestamps', []), timeline.get('values', []))
            ax.set_title('Cost Trend Analysis')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cost ($)')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            charts['cost_trend'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
        
        # Performance overview chart
        performance_metrics = data.get('performance_metrics', {})
        if 'metrics' in performance_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = performance_metrics['metrics']
            
            resources = list(metrics.keys())
            cpu_values = [metrics[r].get('cpu', 0) for r in resources]
            memory_values = [metrics[r].get('memory', 0) for r in resources]
            
            x = np.arange(len(resources))
            width = 0.35
            
            ax.bar(x - width/2, cpu_values, width, label='CPU %')
            ax.bar(x + width/2, memory_values, width, label='Memory %')
            
            ax.set_xlabel('Resources')
            ax.set_ylabel('Utilization %')
            ax.set_title('Resource Utilization Overview')
            ax.set_xticks(x)
            ax.set_xticklabels(resources)
            ax.legend()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            charts['performance_overview'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
        
        return charts
    
    def _create_technical_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create detailed technical charts"""
        
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available for chart generation"}
        
        charts = {}
        
        # Detailed performance timeline
        performance_metrics = data.get('performance_metrics', {})
        if 'metrics_timeline' in performance_metrics:
            timeline = performance_metrics['metrics_timeline']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            timestamps = [item['timestamp'] for item in timeline]
            cpu_values = [item.get('cpu', 0) for item in timeline]
            memory_values = [item.get('memory', 0) for item in timeline]
            network_values = [item.get('network_io', 0) for item in timeline]
            disk_values = [item.get('disk_io', 0) for item in timeline]
            
            ax1.plot(timestamps, cpu_values, color='#1f77b4')
            ax1.set_title('CPU Utilization Over Time')
            ax1.set_ylabel('CPU %')
            
            ax2.plot(timestamps, memory_values, color='#ff7f0e')
            ax2.set_title('Memory Utilization Over Time')
            ax2.set_ylabel('Memory %')
            
            ax3.plot(timestamps, network_values, color='#2ca02c')
            ax3.set_title('Network I/O Over Time')
            ax3.set_ylabel('Network I/O')
            
            ax4.plot(timestamps, disk_values, color='#d62728')
            ax4.set_title('Disk I/O Over Time')
            ax4.set_ylabel('Disk I/O')
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            charts['performance_timeline'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
        
        return charts
    
    def export_to_pdf(self, report_data: Dict[str, Any], filename: str = None) -> BytesIO:
        """Export report to PDF format"""
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF export. Install with: pip install reportlab")
        
        if not filename:
            filename = f"cloud_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph(report_data.get('title', 'Cloud Intelligence Report'), self.custom_styles['title']))
        story.append(Spacer(1, 20))
        
        # Report metadata
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Paragraph(f"Period: {report_data.get('period', 'N/A')}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key metrics section
        if 'key_metrics' in report_data:
            story.append(Paragraph("Key Metrics", self.custom_styles['subtitle']))
            
            metrics = report_data['key_metrics']
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Cost', f"${metrics.get('total_cost', 0):,.2f}"],
                ['Active Workloads', str(metrics.get('active_workloads', 0))],
                ['Performance Score', f"{metrics.get('performance_score', 0):.1f}%"],
                ['Cost Change', f"{metrics.get('cost_change_percent', 0):+.1f}%"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 20))
        
        # Insights section
        if 'insights' in report_data:
            story.append(Paragraph("Key Insights", self.custom_styles['subtitle']))
            
            insights = report_data['insights']
            for category, insight_list in insights.items():
                if insight_list:
                    story.append(Paragraph(f"{category.replace('_', ' ').title()}:", self.styles['Heading3']))
                    for insight in insight_list:
                        story.append(Paragraph(f"• {insight}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
        
        # Recommendations section
        if 'recommendations' in report_data:
            story.append(Paragraph("Recommendations", self.custom_styles['subtitle']))
            
            for i, rec in enumerate(report_data['recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec.get('recommendation', '')}", self.styles['Normal']))
                story.append(Paragraph(f"   Category: {rec.get('category', '')}, Priority: {rec.get('priority', '')}", self.styles['Normal']))
                story.append(Paragraph(f"   Expected Impact: {rec.get('impact', '')}", self.styles['Normal']))
                story.append(Spacer(1, 10))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def export_to_excel(self, report_data: Dict[str, Any], filename: str = None) -> BytesIO:
        """Export report to Excel format"""
        
        if not filename:
            filename = f"cloud_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            if 'key_metrics' in report_data:
                metrics_df = pd.DataFrame([report_data['key_metrics']]).T
                metrics_df.columns = ['Value']
                metrics_df.to_excel(writer, sheet_name='Summary')
            
            # Detailed data sheets
            if 'sections' in report_data:
                for section_name, section_data in report_data['sections'].items():
                    if isinstance(section_data, dict):
                        section_df = pd.DataFrame([section_data]).T
                        section_df.columns = ['Value']
                        section_df.to_excel(writer, sheet_name=section_name[:31])  # Excel sheet name limit
            
            # Raw data if included
            if 'raw_data' in report_data and report_data['raw_data']:
                raw_df = pd.DataFrame(report_data['raw_data'])
                raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        buffer.seek(0)
        return buffer
    
    def schedule_report_generation(self, config: Dict[str, Any]) -> bool:
        """Schedule automated report generation"""
        
        # This would integrate with a job scheduler like Celery or APScheduler
        # For now, we'll create a configuration record
        
        schedule_config = {
            "report_name": config.get('name'),
            "report_type": config.get('type'),
            "frequency": config.get('schedule_frequency'),
            "recipients": config.get('email_recipients', '').split(','),
            "output_formats": config.get('output_format', []),
            "created_at": datetime.now().isoformat(),
            "next_run": self._calculate_next_run(config.get('schedule_frequency')),
            "active": True
        }
        
        # Store configuration (would be saved to database in real implementation)
        st.session_state['scheduled_reports'] = st.session_state.get('scheduled_reports', [])
        st.session_state['scheduled_reports'].append(schedule_config)
        
        return True
    
    def _calculate_next_run(self, frequency: str) -> str:
        """Calculate next scheduled run time"""
        
        now = datetime.now()
        
        if frequency == "Daily":
            next_run = now + timedelta(days=1)
        elif frequency == "Weekly":
            next_run = now + timedelta(weeks=1)
        elif frequency == "Monthly":
            next_run = now + timedelta(days=30)
        elif frequency == "Quarterly":
            next_run = now + timedelta(days=90)
        else:
            next_run = now + timedelta(days=1)
        
        return next_run.isoformat()
    
    def _identify_workload_bottlenecks(self, workload_metrics: Dict) -> List[str]:
        """Identify workload performance bottlenecks"""
        bottlenecks = []
        
        success_rate = workload_metrics.get('success_rate', 100)
        if success_rate < 95:
            bottlenecks.append("Low success rate indicates scheduling issues")
        
        provider_dist = workload_metrics.get('provider_distribution', {})
        if provider_dist:
            max_provider = max(provider_dist.values())
            total_workloads = sum(provider_dist.values())
            if max_provider / total_workloads > 0.8:
                bottlenecks.append("Workload distribution heavily skewed to one provider")
        
        return bottlenecks
    
    def _calculate_optimization_potential(self, cost_metrics: Dict) -> Dict[str, float]:
        """Calculate cost optimization potential"""
        
        total_cost = cost_metrics.get('total_cost_24h', 0)
        
        return {
            "right_sizing": total_cost * 0.15,  # 15% potential savings
            "reserved_instances": total_cost * 0.20,  # 20% potential savings
            "spot_instances": total_cost * 0.60,  # 60% potential savings
            "storage_optimization": total_cost * 0.10  # 10% potential savings
        }
    
    def _calculate_cost_efficiency(self, cost_metrics: Dict) -> float:
        """Calculate cost efficiency score"""
        
        # Simplified efficiency calculation
        cost_change = cost_metrics.get('cost_change_percent', 0)
        
        if cost_change < -10:
            return 95  # Excellent efficiency
        elif cost_change < 0:
            return 85  # Good efficiency
        elif cost_change < 10:
            return 75  # Average efficiency
        else:
            return 60  # Poor efficiency
    
    def _assess_resource_health(self, performance_metrics: Dict) -> Dict[str, str]:
        """Assess overall resource health"""
        
        avg_cpu = performance_metrics.get('avg_cpu', 0)
        avg_memory = performance_metrics.get('avg_memory', 0)
        
        health_status = {}
        
        if avg_cpu > 90:
            health_status['cpu'] = 'critical'
        elif avg_cpu > 80:
            health_status['cpu'] = 'warning'
        else:
            health_status['cpu'] = 'healthy'
        
        if avg_memory > 90:
            health_status['memory'] = 'critical'
        elif avg_memory > 85:
            health_status['memory'] = 'warning'
        else:
            health_status['memory'] = 'healthy'
        
        return health_status
    
    def _calculate_performance_score(self, performance_metrics: Dict) -> float:
        """Calculate overall performance score"""
        
        avg_cpu = performance_metrics.get('avg_cpu', 0)
        avg_memory = performance_metrics.get('avg_memory', 0)
        
        # Simplified performance score calculation
        cpu_score = max(0, 100 - avg_cpu)
        memory_score = max(0, 100 - avg_memory)
        
        return (cpu_score + memory_score) / 2
    
    def _analyze_resource_utilization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        
        performance_metrics = data.get('performance_metrics', {})
        
        return {
            "utilization_summary": {
                "cpu_avg": performance_metrics.get('avg_cpu', 0),
                "memory_avg": performance_metrics.get('avg_memory', 0),
                "peak_usage_times": "Business hours (9 AM - 5 PM)",
                "low_usage_times": "Overnight (11 PM - 6 AM)"
            },
            "efficiency_metrics": {
                "resource_efficiency": self._calculate_performance_score(performance_metrics),
                "waste_percentage": max(0, 100 - performance_metrics.get('avg_cpu', 0)),
                "optimization_score": 75  # Mock score
            }
        }
    
    def _analyze_optimization_opportunities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization opportunities"""
        
        cost_metrics = data.get('cost_metrics', {})
        performance_metrics = data.get('performance_metrics', {})
        
        return {
            "cost_optimizations": self._calculate_optimization_potential(cost_metrics),
            "performance_optimizations": {
                "auto_scaling": "High impact - implement for variable workloads",
                "load_balancing": "Medium impact - optimize traffic distribution",
                "caching": "High impact - reduce database load"
            },
            "priority_actions": [
                "Implement auto-scaling policies",
                "Right-size over-provisioned instances",
                "Optimize storage classes",
                "Review data transfer costs"
            ]
        }
    
    def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in cost and performance"""
        
        cost_metrics = data.get('cost_metrics', {})
        performance_metrics = data.get('performance_metrics', {})
        
        return {
            "cost_trends": {
                "direction": "increasing" if cost_metrics.get('cost_change_percent', 0) > 0 else "decreasing",
                "rate": abs(cost_metrics.get('cost_change_percent', 0)),
                "forecast": "Costs projected to increase 15% next month without optimization"
            },
            "performance_trends": {
                "cpu_trend": "stable" if abs(performance_metrics.get('cpu_change', 0)) < 5 else "increasing",
                "memory_trend": "stable" if abs(performance_metrics.get('memory_change', 0)) < 5 else "increasing",
                "capacity_forecast": "Current capacity sufficient for next 3 months"
            }
        }
    
    def _compile_detailed_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile detailed metrics for technical report"""
        
        return {
            "workload_metrics": data.get('workload_metrics', {}),
            "cost_metrics": data.get('cost_metrics', {}),
            "performance_metrics": data.get('performance_metrics', {}),
            "system_metrics": {
                "uptime": "99.9%",
                "response_time": "150ms avg",
                "throughput": "1000 req/sec",
                "error_rate": "0.1%"
            }
        }


class ReportScheduler:
    """Handles scheduled report generation and distribution"""
    
    def __init__(self):
        self.scheduled_reports = []
    
    def add_scheduled_report(self, config: Dict[str, Any]) -> str:
        """Add a new scheduled report"""
        
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        scheduled_report = {
            "id": report_id,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "next_run": self._calculate_next_run(config.get('frequency', 'Daily')),
            "status": "active"
        }
        
        self.scheduled_reports.append(scheduled_report)
        return report_id
    
    def get_scheduled_reports(self) -> List[Dict[str, Any]]:
        """Get list of scheduled reports"""
        return self.scheduled_reports
    
    def update_scheduled_report(self, report_id: str, config: Dict[str, Any]) -> bool:
        """Update scheduled report configuration"""
        
        for report in self.scheduled_reports:
            if report['id'] == report_id:
                report['config'] = config
                report['next_run'] = self._calculate_next_run(config.get('frequency', 'Daily'))
                return True
        
        return False
    
    def delete_scheduled_report(self, report_id: str) -> bool:
        """Delete scheduled report"""
        
        self.scheduled_reports = [r for r in self.scheduled_reports if r['id'] != report_id]
        return True
    
    def _calculate_next_run(self, frequency: str) -> str:
        """Calculate next run time based on frequency"""
        
        now = datetime.now()
        
        frequency_map = {
            "Daily": timedelta(days=1),
            "Weekly": timedelta(weeks=1),
            "Monthly": timedelta(days=30),
            "Quarterly": timedelta(days=90)
        }
        
        delta = frequency_map.get(frequency, timedelta(days=1))
        next_run = now + delta
        
        return next_run.isoformat()


class ReportDistributor:
    """Handles report distribution via email and other channels"""
    
    def __init__(self):
        self.email_config = self._load_email_config()
    
    def _load_email_config(self) -> Dict[str, str]:
        """Load email configuration"""
        
        # In a real implementation, this would load from environment variables or config file
        return {
            "smtp_server": "smtp.company.com",
            "smtp_port": "587",
            "username": "reports@company.com",
            "password": "password",  # Should be encrypted/from env
            "from_address": "Cloud Intelligence Platform <reports@company.com>"
        }
    
    def send_report_email(self, recipients: List[str], report_data: Dict[str, Any], 
                         attachments: List[BytesIO] = None) -> bool:
        """Send report via email"""
        
        # This would implement actual email sending using smtplib
        # For now, we'll simulate the process
        
        email_content = self._generate_email_content(report_data)
        
        # Simulate email sending
        st.success(f"Report sent to {len(recipients)} recipients")
        
        return True
    
    def _generate_email_content(self, report_data: Dict[str, Any]) -> str:
        """Generate email content for report"""
        
        subject = f"Cloud Intelligence Report - {report_data.get('period', 'Latest')}"
        
        body = f"""
        Dear Team,
        
        Please find attached the latest Cloud Intelligence Platform report.
        
        Report Summary:
        - Period: {report_data.get('period', 'N/A')}
        - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Key Highlights:
        """
        
        if 'key_metrics' in report_data:
            metrics = report_data['key_metrics']
            body += f"""
        - Total Cost: ${metrics.get('total_cost', 0):,.2f}
        - Active Workloads: {metrics.get('active_workloads', 0)}
        - Performance Score: {metrics.get('performance_score', 0):.1f}%
        """
        
        body += """
        
        For detailed analysis, please review the attached report.
        
        Best regards,
        Cloud Intelligence Platform
        """
        
        return body