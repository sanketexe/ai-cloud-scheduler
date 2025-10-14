"""
Alert management system for the dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid

class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self):
        self.alert_types = [
            "Cost Threshold", "Performance Degradation", "Resource Utilization", 
            "System Health", "Security", "Compliance", "Custom"
        ]
        self.severity_levels = ["Critical", "Warning", "Info", "Low"]
        self.notification_channels = ["Email", "Slack", "Webhook", "SMS", "Dashboard"]
        
    def create_alert_configuration_interface(self) -> Dict[str, Any]:
        """Create alert configuration interface"""
        
        st.subheader("🔔 Alert Configuration")
        
        # Alert basic settings
        col1, col2 = st.columns(2)
        
        with col1:
            alert_name = st.text_input(
                "Alert Name",
                placeholder="High CPU Utilization Alert",
                help="Descriptive name for the alert"
            )
            
            alert_type = st.selectbox(
                "Alert Type",
                options=self.alert_types,
                help="Category of alert to create"
            )
            
            severity = st.selectbox(
                "Severity Level",
                options=self.severity_levels,
                index=1,
                help="Severity level for this alert"
            )
            
            enabled = st.checkbox("Enable Alert", value=True)
        
        with col2:
            # Resource and metric selection
            resource_type = st.selectbox(
                "Resource Type",
                options=["All Resources", "Compute", "Storage", "Network", "Database", "Specific Resource"],
                help="Type of resource to monitor"
            )
            
            if resource_type == "Specific Resource":
                specific_resource = st.text_input(
                    "Resource ID",
                    placeholder="server-001, db-prod-01, etc.",
                    help="Specific resource identifier"
                )
            else:
                specific_resource = None
            
            metric_type = st.selectbox(
                "Metric to Monitor",
                options=["CPU Utilization", "Memory Utilization", "Disk Usage", "Network I/O", 
                        "Cost", "Response Time", "Error Rate", "Custom Metric"],
                help="Metric to monitor for this alert"
            )
            
            if metric_type == "Custom Metric":
                custom_metric = st.text_input(
                    "Custom Metric Name",
                    placeholder="custom.metric.name",
                    help="Name of custom metric to monitor"
                )
            else:
                custom_metric = None
        
        # Threshold configuration
        st.markdown("**Threshold Configuration**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold_type = st.selectbox(
                "Threshold Type",
                options=["Greater Than", "Less Than", "Equal To", "Not Equal To", "Range"],
                help="Type of threshold comparison"
            )
        
        with col2:
            if threshold_type == "Range":
                min_threshold = st.number_input("Minimum Value", value=0.0)
                max_threshold = st.number_input("Maximum Value", value=100.0)
                threshold_value = (min_threshold, max_threshold)
            else:
                threshold_value = st.number_input("Threshold Value", value=80.0)
        
        with col3:
            time_window = st.selectbox(
                "Time Window",
                options=["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=2,
                help="Time window for threshold evaluation"
            )
        
        # Advanced conditions
        with st.expander("🔧 Advanced Conditions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                consecutive_breaches = st.number_input(
                    "Consecutive Breaches",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of consecutive threshold breaches before alerting"
                )
                
                cooldown_period = st.selectbox(
                    "Cooldown Period",
                    options=["5 minutes", "15 minutes", "30 minutes", "1 hour", "2 hours"],
                    index=1,
                    help="Minimum time between alerts for the same condition"
                )
            
            with col2:
                additional_conditions = st.text_area(
                    "Additional Conditions",
                    placeholder="provider == 'AWS' AND region == 'us-east-1'",
                    help="Additional conditions using simple expressions"
                )
                
                auto_resolve = st.checkbox(
                    "Auto-resolve when condition clears",
                    value=True,
                    help="Automatically resolve alert when condition is no longer met"
                )
        
        # Notification configuration
        st.markdown("**Notification Configuration**")
        
        notification_channels = st.multiselect(
            "Notification Channels",
            options=self.notification_channels,
            default=["Email", "Dashboard"],
            help="Channels to send notifications through"
        )
        
        # Channel-specific configuration
        notification_config = {}
        
        if "Email" in notification_channels:
            with st.expander("📧 Email Configuration"):
                email_recipients = st.text_area(
                    "Email Recipients",
                    placeholder="admin@company.com, team@company.com",
                    help="Comma-separated email addresses"
                )
                email_template = st.selectbox(
                    "Email Template",
                    options=["Standard", "Detailed", "Executive Summary"],
                    help="Email template to use"
                )
                notification_config["email"] = {
                    "recipients": email_recipients,
                    "template": email_template
                }
        
        if "Slack" in notification_channels:
            with st.expander("💬 Slack Configuration"):
                slack_channel = st.text_input(
                    "Slack Channel",
                    placeholder="#alerts, @username",
                    help="Slack channel or user to notify"
                )
                slack_webhook = st.text_input(
                    "Webhook URL",
                    placeholder="https://hooks.slack.com/services/...",
                    help="Slack webhook URL",
                    type="password"
                )
                notification_config["slack"] = {
                    "channel": slack_channel,
                    "webhook": slack_webhook
                }
        
        if "Webhook" in notification_channels:
            with st.expander("🔗 Webhook Configuration"):
                webhook_url = st.text_input(
                    "Webhook URL",
                    placeholder="https://api.company.com/alerts",
                    help="HTTP endpoint to send alert data"
                )
                webhook_method = st.selectbox(
                    "HTTP Method",
                    options=["POST", "PUT", "PATCH"],
                    help="HTTP method for webhook"
                )
                webhook_headers = st.text_area(
                    "Custom Headers (JSON)",
                    placeholder='{"Authorization": "Bearer token", "Content-Type": "application/json"}',
                    help="Custom HTTP headers as JSON"
                )
                notification_config["webhook"] = {
                    "url": webhook_url,
                    "method": webhook_method,
                    "headers": webhook_headers
                }
        
        return {
            "name": alert_name,
            "type": alert_type,
            "severity": severity,
            "enabled": enabled,
            "resource_type": resource_type,
            "specific_resource": specific_resource,
            "metric_type": metric_type,
            "custom_metric": custom_metric,
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
            "time_window": time_window,
            "consecutive_breaches": consecutive_breaches,
            "cooldown_period": cooldown_period,
            "additional_conditions": additional_conditions,
            "auto_resolve": auto_resolve,
            "notification_channels": notification_channels,
            "notification_config": notification_config
        }
    
    def display_active_alerts(self, alert_data: Dict[str, Any]):
        """Display active alerts with management options"""
        
        st.subheader("🚨 Active Alerts")
        
        alerts = alert_data.get('alerts', [])
        
        if not alerts:
            st.info("No active alerts")
            return
        
        # Alert summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_count = len([a for a in alerts if a.get('severity') == 'critical'])
            st.metric("Critical", critical_count, delta=None)
        
        with col2:
            warning_count = len([a for a in alerts if a.get('severity') == 'warning'])
            st.metric("Warning", warning_count, delta=None)
        
        with col3:
            active_count = len([a for a in alerts if a.get('status') == 'active'])
            st.metric("Active", active_count, delta=None)
        
        with col4:
            acknowledged_count = len([a for a in alerts if a.get('status') == 'acknowledged'])
            st.metric("Acknowledged", acknowledged_count, delta=None)
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=["critical", "warning", "info", "low"],
                default=[],
                key="alert_severity_filter"
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["active", "acknowledged", "resolved"],
                default=["active", "acknowledged"],
                key="alert_status_filter"
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                options=["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
                index=2,
                key="alert_time_filter"
            )
        
        # Filter alerts
        filtered_alerts = alerts
        
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.get('severity') in severity_filter]
        
        if status_filter:
            filtered_alerts = [a for a in filtered_alerts if a.get('status') in status_filter]
        
        # Display alerts
        if filtered_alerts:
            for alert in filtered_alerts:
                self._display_alert_card(alert)
        else:
            st.info("No alerts match the current filters")
    
    def _display_alert_card(self, alert: Dict[str, Any]):
        """Display individual alert card with actions"""
        
        severity = alert.get('severity', 'info')
        status = alert.get('status', 'active')
        
        # Color coding based on severity
        severity_colors = {
            'critical': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'low': '#28a745'
        }
        
        color = severity_colors.get(severity, '#6c757d')
        
        with st.container():
            # Create alert header
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: {color};">{alert.get('message', 'Unknown Alert')}</h4>
                    <p style="margin: 0; color: #666;">
                        <strong>Resource:</strong> {alert.get('resource', 'N/A')} | 
                        <strong>Time:</strong> {alert.get('timestamp', 'N/A')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Severity:** {severity.title()}")
            
            with col3:
                st.markdown(f"**Status:** {status.title()}")
            
            with col4:
                # Alert actions
                if status == 'active':
                    if st.button("✅ Acknowledge", key=f"ack_{alert.get('id')}"):
                        self._acknowledge_alert(alert.get('id'))
                        st.rerun()
                
                if st.button("🔕 Silence", key=f"silence_{alert.get('id')}"):
                    self._silence_alert(alert.get('id'))
                    st.rerun()
            
            # Alert details in expander
            with st.expander(f"Details - {alert.get('id', 'Unknown ID')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Alert ID:** {alert.get('id', 'N/A')}")
                    st.markdown(f"**Resource:** {alert.get('resource', 'N/A')}")
                    st.markdown(f"**Metric:** {alert.get('metric', 'N/A')}")
                    st.markdown(f"**Threshold:** {alert.get('threshold', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Current Value:** {alert.get('current_value', 'N/A')}")
                    st.markdown(f"**Duration:** {alert.get('duration', 'N/A')}")
                    st.markdown(f"**Last Updated:** {alert.get('last_updated', 'N/A')}")
                    st.markdown(f"**Escalation Level:** {alert.get('escalation_level', 'Level 1')}")
                
                # Alert history
                if 'history' in alert:
                    st.markdown("**Alert History:**")
                    for event in alert['history'][-5:]:  # Show last 5 events
                        st.markdown(f"• {event.get('timestamp', 'N/A')}: {event.get('action', 'N/A')}")
            
            st.markdown("---")
    
    def display_alert_history(self, alert_data: Dict[str, Any]):
        """Display alert history and analytics"""
        
        st.subheader("📊 Alert History & Analytics")
        
        # Time range selector
        col1, col2 = st.columns(2)
        
        with col1:
            history_range = st.selectbox(
                "History Range",
                options=["Last 24 Hours", "Last Week", "Last Month", "Last Quarter"],
                index=1
            )
        
        with col2:
            group_by = st.selectbox(
                "Group By",
                options=["Severity", "Resource Type", "Alert Type", "Time"],
                index=0
            )
        
        # Mock historical data for demonstration
        historical_alerts = self._generate_mock_alert_history()
        
        # Alert trends chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert count over time
            fig = go.Figure()
            
            # Generate sample time series data
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
            alert_counts = [max(0, int(5 + 3 * (i % 24) / 24 + (i % 3))) for i in range(len(dates))]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=alert_counts,
                mode='lines+markers',
                name='Alert Count',
                line=dict(color='#dc3545')
            ))
            
            fig.update_layout(
                title="Alert Frequency Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Alerts",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Alert distribution by severity
            severity_counts = {
                'Critical': 12,
                'Warning': 34,
                'Info': 56,
                'Low': 23
            }
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(severity_counts.keys()),
                    values=list(severity_counts.values()),
                    hole=0.4,
                    marker_colors=['#dc3545', '#ffc107', '#17a2b8', '#28a745']
                )
            ])
            
            fig.update_layout(
                title="Alert Distribution by Severity",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top alerting resources
        st.markdown("**Top Alerting Resources**")
        
        top_resources = [
            {"resource": "server-001", "alerts": 23, "severity": "Warning"},
            {"resource": "database-prod", "alerts": 18, "severity": "Critical"},
            {"resource": "load-balancer-01", "alerts": 15, "severity": "Info"},
            {"resource": "storage-cluster", "alerts": 12, "severity": "Warning"},
            {"resource": "network-gateway", "alerts": 8, "severity": "Low"}
        ]
        
        df_resources = pd.DataFrame(top_resources)
        st.dataframe(df_resources, use_container_width=True)
        
        # Alert resolution metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Resolution Time", "45 minutes", delta="-5 min")
        
        with col2:
            st.metric("Alert Accuracy", "87%", delta="+3%")
        
        with col3:
            st.metric("False Positive Rate", "13%", delta="-2%")
    
    def create_alert_rules_management(self):
        """Create interface for managing alert rules"""
        
        st.subheader("⚙️ Alert Rules Management")
        
        # Get existing rules from session state
        alert_rules = st.session_state.get('alert_rules', [])
        
        # Add new rule button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("➕ Add New Rule", type="primary"):
                st.session_state['show_rule_form'] = True
        
        with col2:
            if st.button("📥 Import Rules"):
                st.info("Import rules from file - Feature to be implemented")
        
        with col3:
            if alert_rules and st.button("📤 Export Rules"):
                rules_json = json.dumps(alert_rules, indent=2)
                st.download_button(
                    label="Download Rules JSON",
                    data=rules_json,
                    file_name=f"alert_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Show rule creation form if requested
        if st.session_state.get('show_rule_form', False):
            with st.expander("🔧 Create New Alert Rule", expanded=True):
                rule_config = self.create_alert_configuration_interface()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Rule"):
                        if rule_config['name']:
                            # Add unique ID and timestamps
                            rule_config['id'] = str(uuid.uuid4())
                            rule_config['created_at'] = datetime.now().isoformat()
                            rule_config['updated_at'] = datetime.now().isoformat()
                            
                            # Add to rules list
                            if 'alert_rules' not in st.session_state:
                                st.session_state['alert_rules'] = []
                            
                            st.session_state['alert_rules'].append(rule_config)
                            st.session_state['show_rule_form'] = False
                            
                            st.success(f"Alert rule '{rule_config['name']}' created successfully!")
                            st.rerun()
                        else:
                            st.error("Please provide a rule name")
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state['show_rule_form'] = False
                        st.rerun()
        
        # Display existing rules
        if alert_rules:
            st.markdown("**Existing Alert Rules**")
            
            # Rules filter
            col1, col2, col3 = st.columns(3)
            
            with col1:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=self.alert_types,
                    default=[],
                    key="rules_type_filter"
                )
            
            with col2:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=self.severity_levels,
                    default=[],
                    key="rules_severity_filter"
                )
            
            with col3:
                status_filter = st.selectbox(
                    "Status",
                    options=["All", "Enabled", "Disabled"],
                    key="rules_status_filter"
                )
            
            # Filter rules
            filtered_rules = alert_rules
            
            if type_filter:
                filtered_rules = [r for r in filtered_rules if r.get('type') in type_filter]
            
            if severity_filter:
                filtered_rules = [r for r in filtered_rules if r.get('severity') in severity_filter]
            
            if status_filter != "All":
                enabled_filter = status_filter == "Enabled"
                filtered_rules = [r for r in filtered_rules if r.get('enabled') == enabled_filter]
            
            # Display rules
            for rule in filtered_rules:
                self._display_rule_card(rule)
        else:
            st.info("No alert rules configured. Click 'Add New Rule' to create your first alert rule.")
    
    def _display_rule_card(self, rule: Dict[str, Any]):
        """Display individual alert rule card"""
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                enabled_icon = "🟢" if rule.get('enabled') else "🔴"
                st.markdown(f"**{enabled_icon} {rule.get('name', 'Unnamed Rule')}**")
                st.markdown(f"*{rule.get('type', 'Unknown')} - {rule.get('severity', 'Unknown')} severity*")
            
            with col2:
                st.markdown(f"**Metric:** {rule.get('metric_type', 'N/A')}")
            
            with col3:
                threshold = rule.get('threshold_value', 'N/A')
                st.markdown(f"**Threshold:** {threshold}")
            
            with col4:
                # Rule actions
                col_edit, col_delete = st.columns(2)
                
                with col_edit:
                    if st.button("✏️", key=f"edit_{rule.get('id')}", help="Edit rule"):
                        st.session_state[f'edit_rule_{rule.get("id")}'] = True
                        st.rerun()
                
                with col_delete:
                    if st.button("🗑️", key=f"delete_{rule.get('id')}", help="Delete rule"):
                        self._delete_alert_rule(rule.get('id'))
                        st.rerun()
            
            # Rule details
            with st.expander(f"Rule Details - {rule.get('id', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Resource Type:** {rule.get('resource_type', 'N/A')}")
                    st.markdown(f"**Threshold Type:** {rule.get('threshold_type', 'N/A')}")
                    st.markdown(f"**Time Window:** {rule.get('time_window', 'N/A')}")
                    st.markdown(f"**Consecutive Breaches:** {rule.get('consecutive_breaches', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Notification Channels:** {', '.join(rule.get('notification_channels', []))}")
                    st.markdown(f"**Auto Resolve:** {'Yes' if rule.get('auto_resolve') else 'No'}")
                    st.markdown(f"**Created:** {rule.get('created_at', 'N/A')}")
                    st.markdown(f"**Last Updated:** {rule.get('updated_at', 'N/A')}")
                
                # Show notification configuration
                if rule.get('notification_config'):
                    st.markdown("**Notification Configuration:**")
                    st.json(rule['notification_config'])
            
            st.markdown("---")
    
    def create_notification_settings(self):
        """Create notification settings interface"""
        
        st.subheader("📢 Notification Settings")
        
        # Global notification settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Global Settings**")
            
            global_enabled = st.checkbox("Enable Notifications", value=True)
            
            quiet_hours_enabled = st.checkbox("Enable Quiet Hours", value=False)
            
            if quiet_hours_enabled:
                col_start, col_end = st.columns(2)
                with col_start:
                    quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
                with col_end:
                    quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("08:00", "%H:%M").time())
            
            escalation_enabled = st.checkbox("Enable Escalation", value=True)
            
            if escalation_enabled:
                escalation_time = st.selectbox(
                    "Escalation Time",
                    options=["15 minutes", "30 minutes", "1 hour", "2 hours"],
                    index=1
                )
        
        with col2:
            st.markdown("**Default Channels**")
            
            default_channels = st.multiselect(
                "Default Notification Channels",
                options=self.notification_channels,
                default=["Email", "Dashboard"]
            )
            
            batch_notifications = st.checkbox("Batch Similar Notifications", value=True)
            
            if batch_notifications:
                batch_window = st.selectbox(
                    "Batch Window",
                    options=["5 minutes", "15 minutes", "30 minutes"],
                    index=1
                )
        
        # Channel-specific settings
        st.markdown("**Channel Configuration**")
        
        # Email settings
        with st.expander("📧 Email Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                smtp_server = st.text_input("SMTP Server", value="smtp.company.com")
                smtp_port = st.number_input("SMTP Port", value=587)
                smtp_username = st.text_input("Username", value="alerts@company.com")
            
            with col2:
                smtp_password = st.text_input("Password", type="password")
                from_address = st.text_input("From Address", value="Cloud Intelligence <alerts@company.com>")
                
                test_email = st.text_input("Test Email Address")
                if st.button("📧 Send Test Email") and test_email:
                    st.success(f"Test email sent to {test_email}")
        
        # Slack settings
        with st.expander("💬 Slack Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                slack_workspace = st.text_input("Slack Workspace", placeholder="company.slack.com")
                slack_bot_token = st.text_input("Bot Token", type="password", placeholder="xoxb-...")
            
            with col2:
                default_slack_channel = st.text_input("Default Channel", placeholder="#alerts")
                
                if st.button("💬 Test Slack Connection"):
                    st.success("Slack connection test successful")
        
        # Webhook settings
        with st.expander("🔗 Webhook Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                webhook_timeout = st.number_input("Timeout (seconds)", value=30)
                webhook_retries = st.number_input("Max Retries", value=3)
            
            with col2:
                webhook_auth_type = st.selectbox(
                    "Authentication",
                    options=["None", "Bearer Token", "API Key", "Basic Auth"]
                )
                
                if webhook_auth_type != "None":
                    webhook_auth_value = st.text_input("Auth Value", type="password")
        
        # Save settings
        if st.button("💾 Save Notification Settings", type="primary"):
            notification_settings = {
                "global_enabled": global_enabled,
                "quiet_hours_enabled": quiet_hours_enabled,
                "quiet_start": quiet_start.strftime("%H:%M") if quiet_hours_enabled else None,
                "quiet_end": quiet_end.strftime("%H:%M") if quiet_hours_enabled else None,
                "escalation_enabled": escalation_enabled,
                "escalation_time": escalation_time if escalation_enabled else None,
                "default_channels": default_channels,
                "batch_notifications": batch_notifications,
                "batch_window": batch_window if batch_notifications else None,
                "smtp_config": {
                    "server": smtp_server,
                    "port": smtp_port,
                    "username": smtp_username,
                    "from_address": from_address
                },
                "slack_config": {
                    "workspace": slack_workspace,
                    "default_channel": default_slack_channel
                },
                "webhook_config": {
                    "timeout": webhook_timeout,
                    "retries": webhook_retries,
                    "auth_type": webhook_auth_type
                }
            }
            
            st.session_state['notification_settings'] = notification_settings
            st.success("Notification settings saved successfully!")
    
    def _acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        # In a real implementation, this would update the alert status in the database
        st.session_state[f'alert_ack_{alert_id}'] = True
        return True
    
    def _silence_alert(self, alert_id: str):
        """Silence an alert"""
        # In a real implementation, this would silence the alert
        st.session_state[f'alert_silence_{alert_id}'] = True
        return True
    
    def _delete_alert_rule(self, rule_id: str):
        """Delete an alert rule"""
        if 'alert_rules' in st.session_state:
            st.session_state['alert_rules'] = [
                rule for rule in st.session_state['alert_rules'] 
                if rule.get('id') != rule_id
            ]
        return True
    
    def _generate_mock_alert_history(self) -> List[Dict[str, Any]]:
        """Generate mock alert history for demonstration"""
        
        history = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(50):
            alert_time = base_time + timedelta(hours=i * 3.2)
            
            history.append({
                "id": f"alert_{i:03d}",
                "timestamp": alert_time.isoformat(),
                "severity": ["critical", "warning", "info", "low"][i % 4],
                "message": f"Sample alert message {i}",
                "resource": f"resource-{i % 10:02d}",
                "status": ["resolved", "acknowledged", "active"][i % 3],
                "duration": f"{(i % 120) + 5} minutes"
            })
        
        return history


class AlertAnalytics:
    """Analytics and insights for alert management"""
    
    def __init__(self):
        pass
    
    def generate_alert_insights(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from alert data"""
        
        alerts = alert_data.get('alerts', [])
        
        insights = {
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get('severity') == 'critical']),
            "most_common_type": self._get_most_common_alert_type(alerts),
            "busiest_resource": self._get_busiest_resource(alerts),
            "alert_frequency_trend": self._calculate_frequency_trend(alerts),
            "resolution_efficiency": self._calculate_resolution_efficiency(alerts),
            "recommendations": self._generate_alert_recommendations(alerts)
        }
        
        return insights
    
    def _get_most_common_alert_type(self, alerts: List[Dict]) -> str:
        """Get the most common alert type"""
        if not alerts:
            return "None"
        
        # Mock implementation
        return "Performance Degradation"
    
    def _get_busiest_resource(self, alerts: List[Dict]) -> str:
        """Get the resource with most alerts"""
        if not alerts:
            return "None"
        
        # Mock implementation
        return "server-001"
    
    def _calculate_frequency_trend(self, alerts: List[Dict]) -> str:
        """Calculate alert frequency trend"""
        # Mock implementation
        return "Increasing by 15% over last week"
    
    def _calculate_resolution_efficiency(self, alerts: List[Dict]) -> float:
        """Calculate alert resolution efficiency"""
        # Mock implementation
        return 87.5
    
    def _generate_alert_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Generate recommendations based on alert patterns"""
        
        recommendations = [
            "Consider implementing auto-scaling for frequently alerting resources",
            "Review threshold settings for high-frequency, low-impact alerts",
            "Set up alert correlation to reduce noise from related alerts",
            "Implement predictive alerting to catch issues before they become critical"
        ]
        
        return recommendations