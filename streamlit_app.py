import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Cloud Scheduler",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'workloads' not in st.session_state:
    st.session_state.workloads = []
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'workload_analysis' not in st.session_state:
    st.session_state.workload_analysis = None

# Add these functions to your streamlit_app.py

def show_configuration():
    """Show and manage system configuration"""
    st.header("⚙️ Configuration")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Cloud Providers", "Virtual Machines", "Workloads", "System Settings"])
    
    with tab1:
        show_cloud_providers_config()
    
    with tab2:
        show_virtual_machines_config()
    
    with tab3:
        show_workloads_config()
    
    with tab4:
        show_system_settings_config()

def show_cloud_providers_config():
    """Configure cloud providers - UPDATED WITH BETTER UPDATE HANDLING"""
    st.subheader("☁️ Cloud Provider Configuration")
    
    try:
        # Get current provider configuration from API
        response = requests.get(f"{API_BASE_URL}/api/providers/default", timeout=5)
        
        if response.status_code == 200:
            api_response = response.json()
            
            # Handle different API response formats
            if isinstance(api_response, list):
                providers = {}
                for provider in api_response:
                    if isinstance(provider, dict) and 'name' in provider:
                        providers[provider['name'].lower()] = {
                            'cpu_cost': provider.get('cpu_cost', 0.04),
                            'memory_cost_gb': provider.get('memory_cost_gb', 0.01)
                        }
            elif isinstance(api_response, dict):
                providers = api_response
            else:
                raise ValueError("Unexpected API response format")
            
            if not providers:
                raise ValueError("No provider data received from API")
            
            # Display current providers
            st.write("**Current Provider Configuration:**")
            
            for provider_name, provider_data in providers.items():
                if not isinstance(provider_data, dict):
                    continue
                
                with st.expander(f"{provider_name.upper()} Configuration", expanded=True):
                    
                    # Current values display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        current_cpu_cost = provider_data.get('cpu_cost', 0.04)
                        st.metric(
                            "Current CPU Cost/Hour", 
                            f"${current_cpu_cost:.3f}",
                            help="Cost per CPU core per hour"
                        )
                    
                    with col2:
                        current_memory_cost = provider_data.get('memory_cost_gb', 0.01)
                        st.metric(
                            "Current Memory Cost/GB/Hour", 
                            f"${current_memory_cost:.3f}",
                            help="Cost per GB memory per hour"
                        )
                    
                    # Update form
                    st.write("**Update Configuration:**")
                    
                    with st.form(f"update_form_{provider_name}"):
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            new_cpu_cost = st.number_input(
                                "New CPU Cost",
                                min_value=0.001,
                                max_value=1.0,
                                value=float(current_cpu_cost),
                                step=0.001,
                                format="%.3f",
                                key=f"{provider_name}_cpu_form"
                            )
                        
                        with col2:
                            new_memory_cost = st.number_input(
                                "New Memory Cost",
                                min_value=0.001,
                                max_value=0.5,
                                value=float(current_memory_cost),
                                step=0.001,
                                format="%.3f",
                                key=f"{provider_name}_memory_form"
                            )
                        
                        with col3:
                            st.write("")  # Spacer
                            submitted = st.form_submit_button(
                                f"Update {provider_name.upper()}", 
                                type="primary",
                                use_container_width=True
                            )
                        
                        if submitted:
                            if new_cpu_cost != current_cpu_cost or new_memory_cost != current_memory_cost:
                                # Show what will be updated
                                st.info(f"Updating {provider_name.upper()}: CPU ${new_cpu_cost:.3f}, Memory ${new_memory_cost:.3f}")
                                
                                # Try the main update method first
                                update_provider_config(provider_name, new_cpu_cost, new_memory_cost)
                            else:
                                st.info("No changes detected.")
            
            # Cost comparison chart (same as before)
            st.subheader("💰 Cost Comparison")
            
            provider_names = list(providers.keys())
            cpu_costs = [providers[p].get('cpu_cost', 0) for p in provider_names]
            memory_costs = [providers[p].get('memory_cost_gb', 0) for p in provider_names]
            
            if provider_names and any(cpu_costs) and any(memory_costs):
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('CPU Cost ($/hour)', 'Memory Cost ($/GB/hour)'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(
                    go.Bar(
                        x=[p.upper() for p in provider_names], 
                        y=cpu_costs, 
                        name="CPU Cost", 
                        marker_color='lightblue',
                        text=[f"${c:.3f}" for c in cpu_costs],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=[p.upper() for p in provider_names], 
                        y=memory_costs, 
                        name="Memory Cost", 
                        marker_color='lightcoral',
                        text=[f"${c:.3f}" for c in memory_costs],
                        textposition='outside'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # API endpoint testing section
            with st.expander("🔧 API Testing (Debug)"):
                st.write("Test different API endpoints to see which one works:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Test Config Endpoint"):
                        test_config_endpoint()
                
                with col2:
                    if st.button("Test Provider Endpoint"):
                        test_provider_endpoint()
            
        else:
            st.error(f"Failed to load provider configuration: {response.text}")
            show_fallback_providers()
            
    except Exception as e:
        st.error(f"Error loading provider configuration: {e}")
        show_fallback_providers()

def show_fallback_providers():
    """Show fallback provider configuration when API is unavailable"""
    st.info("Showing default configuration (API unavailable)")
    default_providers = {
        "aws": {"cpu_cost": 0.04, "memory_cost_gb": 0.01},
        "gcp": {"cpu_cost": 0.035, "memory_cost_gb": 0.009},
        "azure": {"cpu_cost": 0.042, "memory_cost_gb": 0.011}
    }
    
    for provider, costs in default_providers.items():
        st.write(f"**{provider.upper()}:** CPU ${costs['cpu_cost']:.3f}/hr, Memory ${costs['memory_cost_gb']:.3f}/GB/hr")

def show_virtual_machines_config():
    """Configure virtual machines - FIXED VERSION"""
    st.subheader("🖥️ Virtual Machine Configuration")
    
    try:
        # Get current VM configuration from API
        response = requests.get(f"{API_BASE_URL}/api/vms/default", timeout=5)
        
        if response.status_code == 200:
            api_response = response.json()
            
            # Handle different API response formats
            if isinstance(api_response, list):
                vms = api_response
            elif isinstance(api_response, dict) and 'vms' in api_response:
                vms = api_response['vms']
            elif isinstance(api_response, dict):
                # Convert single VM dict to list
                vms = [api_response]
            else:
                raise ValueError("Unexpected API response format")
            
            # Ensure we have valid VM data
            if not vms:
                raise ValueError("No VM data received from API")
            
            st.write("**Current VM Fleet:**")
            
            # Display VMs in a table format
            vm_data = []
            for i, vm in enumerate(vms):
                # Handle different VM data structures
                if isinstance(vm, dict):
                    vm_id = vm.get('vm_id', vm.get('id', i+1))
                    provider = vm.get('provider', vm.get('cloud_provider', 'Unknown'))
                    cpu_capacity = vm.get('cpu_capacity', vm.get('cpu_cores', 0))
                    memory_capacity = vm.get('memory_capacity_gb', vm.get('memory_gb', 0))
                    cpu_used = vm.get('cpu_used', 0)
                    memory_used = vm.get('memory_used_gb', vm.get('memory_used', 0))
                    
                    # Calculate utilization safely
                    cpu_util = (cpu_used / cpu_capacity * 100) if cpu_capacity > 0 else 0
                    memory_util = (memory_used / memory_capacity * 100) if memory_capacity > 0 else 0
                    
                    vm_data.append({
                        "VM ID": f"VM-{vm_id}",
                        "Provider": str(provider).upper() if provider else "Unknown",
                        "CPU Cores": int(cpu_capacity) if cpu_capacity else 0,
                        "Memory (GB)": int(memory_capacity) if memory_capacity else 0,
                        "CPU Used": int(cpu_used) if cpu_used else 0,
                        "Memory Used (GB)": int(memory_used) if memory_used else 0,
                        "CPU Utilization (%)": f"{cpu_util:.1f}",
                        "Memory Utilization (%)": f"{memory_util:.1f}",
                        "Status": "🟢 Available" if cpu_used == 0 else "🟡 In Use"
                    })
                else:
                    st.warning(f"Skipping invalid VM data at index {i}: {vm}")
            
            if not vm_data:
                st.error("No valid VM data could be processed")
                show_fallback_vms()
                return
            
            df_vms = pd.DataFrame(vm_data)
            st.dataframe(df_vms, use_container_width=True)
            
            # VM utilization visualization
            st.subheader("📊 VM Utilization Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU utilization chart
                try:
                    fig_cpu = px.bar(
                        df_vms,
                        x="VM ID",
                        y="CPU Cores",
                        title="CPU Capacity by VM",
                        color="Provider",
                        text="CPU Cores"
                    )
                    fig_cpu.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig_cpu, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating CPU chart: {e}")
                    st.write("CPU Data:", df_vms[["VM ID", "CPU Cores", "Provider"]].to_dict())
            
            with col2:
                # Memory utilization chart
                try:
                    fig_memory = px.bar(
                        df_vms,
                        x="VM ID",
                        y="Memory (GB)",
                        title="Memory Capacity by VM",
                        color="Provider",
                        text="Memory (GB)"
                    )
                    fig_memory.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig_memory, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating Memory chart: {e}")
                    st.write("Memory Data:", df_vms[["VM ID", "Memory (GB)", "Provider"]].to_dict())
            
            # Fleet summary
            st.subheader("📈 Fleet Summary")
            
            # Safe calculation of totals
            try:
                total_cpu = sum(vm.get('cpu_capacity', vm.get('cpu_cores', 0)) for vm in vms if isinstance(vm, dict))
                total_memory = sum(vm.get('memory_capacity_gb', vm.get('memory_gb', 0)) for vm in vms if isinstance(vm, dict))
                total_cpu_used = sum(vm.get('cpu_used', 0) for vm in vms if isinstance(vm, dict))
                total_memory_used = sum(vm.get('memory_used_gb', vm.get('memory_used', 0)) for vm in vms if isinstance(vm, dict))
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total VMs", len(vms))
                
                with col2:
                    st.metric("Total CPU Cores", total_cpu, f"{total_cpu_used} used")
                
                with col3:
                    st.metric("Total Memory (GB)", total_memory, f"{total_memory_used} used")
                
                with col4:
                    avg_cpu_util = (total_cpu_used / total_cpu * 100) if total_cpu > 0 else 0
                    st.metric("Avg CPU Utilization", f"{avg_cpu_util:.1f}%")
            
            except Exception as e:
                st.error(f"Error calculating fleet summary: {e}")
                st.write("Raw VM data for debugging:", vms)
            
        else:
            st.error(f"Failed to load VM configuration: {response.text}")
            show_fallback_vms()
            
    except Exception as e:
        st.error(f"Error loading VM configuration: {e}")
        st.write(f"Debug info - Exception type: {type(e).__name__}")
        show_fallback_vms()

def show_fallback_vms():
    """Show fallback VM configuration when API is unavailable"""
    st.info("Showing default VM configuration (API unavailable)")
    default_vms = [
        {"vm_id": 1, "provider": "AWS", "cpu_capacity": 4, "memory_capacity_gb": 16},
        {"vm_id": 2, "provider": "GCP", "cpu_capacity": 8, "memory_capacity_gb": 32},
        {"vm_id": 3, "provider": "Azure", "cpu_capacity": 4, "memory_capacity_gb": 16},
        {"vm_id": 4, "provider": "GCP", "cpu_capacity": 2, "memory_capacity_gb": 8}
    ]
    
    for vm in default_vms:
        st.write(f"**VM-{vm['vm_id']}** ({vm['provider']}): {vm['cpu_capacity']} CPU, {vm['memory_capacity_gb']}GB RAM")

def show_workloads_config():
    """Configure and manage workloads"""
    st.subheader("📋 Workload Management")
    
    # Current workloads summary
    workload_count = len(st.session_state.workloads)
    
    if workload_count > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workloads", workload_count)
        
        with col2:
            total_cpu = sum(w['cpu_required'] for w in st.session_state.workloads)
            st.metric("Total CPU Required", total_cpu)
        
        with col3:
            total_memory = sum(w['memory_required_gb'] for w in st.session_state.workloads)
            st.metric("Total Memory Required", f"{total_memory} GB")
        
        with col4:
            avg_cpu = total_cpu / workload_count if workload_count > 0 else 0
            st.metric("Avg CPU per Workload", f"{avg_cpu:.1f}")
    else:
        st.info("No workloads loaded. Use the options below to add workloads.")
    
    # Workload management options
    st.subheader("📁 Load Workloads")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Upload CSV", "Sample Data", "Manual Entry", "Generate Random"])
    
    with tab1:
        st.write("**Upload Workloads from CSV File**")
        st.write("Your CSV can have any column names. The system will automatically detect:")
        st.write("- **ID columns:** workload_id, task_id, job_id, id, etc.")
        st.write("- **CPU columns:** cpu_required, cores, vcpu, cpu_cores, etc.")
        st.write("- **Memory columns:** memory_required_gb, memory_gb, ram, mem, etc.")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Upload a CSV file with workload specifications"
        )
        
        if uploaded_file is not None:
            # Preview CSV structure
            if st.button("🔍 Preview CSV Structure"):
                preview_csv_structure(uploaded_file)
            
            # Upload workloads
            if st.button("📤 Upload Workloads", type="primary"):
                upload_workloads(uploaded_file)
    
    with tab2:
        st.write("**Load Pre-configured Sample Data**")
        st.write("Includes 8 sample workloads with varying resource requirements for testing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Load Sample Workloads", type="primary", use_container_width=True):
                load_sample_workloads()
        
        with col2:
            if st.session_state.workloads:
                if st.button("🗑️ Clear All Workloads", use_container_width=True):
                    st.session_state.workloads = []
                    st.success("All workloads cleared!")
                    st.rerun()
    
    with tab3:
        st.write("**Add Individual Workloads Manually**")
        
        with st.form("add_workload"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                workload_id = st.number_input("Workload ID", min_value=1, value=len(st.session_state.workloads) + 1)
            
            with col2:
                cpu_required = st.number_input("CPU Required", min_value=1, max_value=16, value=2)
            
            with col3:
                memory_required = st.number_input("Memory Required (GB)", min_value=1, max_value=64, value=4)
            
            submitted = st.form_submit_button("➕ Add Workload")
            
            if submitted:
                # Check if ID already exists
                existing_ids = [w['workload_id'] for w in st.session_state.workloads]
                if workload_id in existing_ids:
                    st.error(f"Workload ID {workload_id} already exists!")
                else:
                    new_workload = {
                        "workload_id": workload_id,
                        "cpu_required": cpu_required,
                        "memory_required_gb": memory_required
                    }
                    st.session_state.workloads.append(new_workload)
                    st.success(f"Added workload {workload_id}")
                    st.rerun()
    
    with tab4:
        st.write("**Generate Random Workloads for Testing**")
        
        with st.form("generate_workloads"):
            col1, col2 = st.columns(2)
            
            with col1:
                count = st.number_input("Number of Workloads", min_value=1, max_value=100, value=10)
                cpu_min = st.number_input("Min CPU", min_value=1, max_value=8, value=1)
                cpu_max = st.number_input("Max CPU", min_value=2, max_value=16, value=8)
            
            with col2:
                st.write("")  # Spacing
                memory_min = st.number_input("Min Memory (GB)", min_value=1, max_value=16, value=2)
                memory_max = st.number_input("Max Memory (GB)", min_value=4, max_value=64, value=32)
            
            submitted = st.form_submit_button("🎲 Generate Random Workloads")
            
            if submitted:
                if cpu_min >= cpu_max or memory_min >= memory_max:
                    st.error("Min values must be less than max values!")
                else:
                    generate_random_workloads(count, cpu_min, cpu_max, memory_min, memory_max)
    
    # Display current workloads
    if st.session_state.workloads:
        st.subheader("📊 Current Workloads")
        
        # Convert to DataFrame for better display
        df_workloads = pd.DataFrame(st.session_state.workloads)
        
        # Add some calculated columns
        df_workloads['Resource Ratio'] = df_workloads['memory_required_gb'] / df_workloads['cpu_required']
        df_workloads['Size Category'] = df_workloads.apply(
            lambda row: 'Small' if row['cpu_required'] <= 2 and row['memory_required_gb'] <= 4
            else 'Large' if row['cpu_required'] > 6 or row['memory_required_gb'] > 16
            else 'Medium', axis=1
        )
        
        # Display table
        st.dataframe(df_workloads, use_container_width=True)
        
        # Workload analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Resource requirements distribution
            fig_scatter = px.scatter(
                df_workloads,
                x='cpu_required',
                y='memory_required_gb',
                color='Size Category',
                size='Resource Ratio',
                title='Workload Resource Requirements',
                labels={'cpu_required': 'CPU Required', 'memory_required_gb': 'Memory Required (GB)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Size category distribution
            size_counts = df_workloads['Size Category'].value_counts()
            fig_pie = px.pie(
                values=size_counts.values,
                names=size_counts.index,
                title='Workload Size Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Export workloads
        st.subheader("📤 Export Workloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export as CSV"):
                csv_data = df_workloads[['workload_id', 'cpu_required', 'memory_required_gb']].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="workloads.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔍 Analyze Workloads"):
                if hasattr(st.session_state, 'analyze_current_workloads'):
                    analyze_current_workloads()
                else:
                    st.info("Workload analysis feature requires API connection")

def preview_csv_structure(uploaded_file):
    """Preview CSV file structure"""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ CSV Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File Info:**")
            st.write(f"- Filename: {uploaded_file.name}")
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
        
        with col2:
            st.write("**Available Columns:**")
            for col in df.columns:
                st.write(f"- `{col}`")
        
        st.write("**Sample Data (First 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Try to suggest column mapping
        st.write("**Suggested Column Mapping:**")
        
        id_candidates = [col for col in df.columns if any(term in col.lower() for term in ['id', 'task', 'job', 'workload'])]
        cpu_candidates = [col for col in df.columns if any(term in col.lower() for term in ['cpu', 'core', 'vcpu', 'processor'])]
        memory_candidates = [col for col in df.columns if any(term in col.lower() for term in ['memory', 'ram', 'mem', 'gb'])]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**ID Column:** {id_candidates[0] if id_candidates else 'Not detected'}")
        with col2:
            st.write(f"**CPU Column:** {cpu_candidates[0] if cpu_candidates else 'Not detected'}")
        with col3:
            st.write(f"**Memory Column:** {memory_candidates[0] if memory_candidates else 'Not detected'}")
        
    except Exception as e:
        st.error(f"Error previewing CSV: {e}")

def generate_random_workloads(count, cpu_min, cpu_max, memory_min, memory_max):
    """Generate random workloads for testing"""
    import random
    
    try:
        # Get existing IDs to avoid duplicates
        existing_ids = {w['workload_id'] for w in st.session_state.workloads}
        
        new_workloads = []
        next_id = max(existing_ids) + 1 if existing_ids else 1
        
        for i in range(count):
            # Ensure unique ID
            while next_id in existing_ids:
                next_id += 1
            
            workload = {
                "workload_id": next_id,
                "cpu_required": random.randint(cpu_min, cpu_max),
                "memory_required_gb": random.randint(memory_min, memory_max)
            }
            
            new_workloads.append(workload)
            existing_ids.add(next_id)
            next_id += 1
        
        # Add to session state
        st.session_state.workloads.extend(new_workloads)
        
        st.success(f"✅ Generated {count} random workloads!")
        
        # Show summary
        total_cpu = sum(w['cpu_required'] for w in new_workloads)
        total_memory = sum(w['memory_required_gb'] for w in new_workloads)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Workloads Generated", count)
        with col2:
            st.metric("Total CPU", total_cpu)
        with col3:
            st.metric("Total Memory", f"{total_memory} GB")
        
        st.rerun()
    
    except Exception as e:
        st.error(f"Error generating random workloads: {e}")

# Also add this debug function to help troubleshoot API responses
def debug_api_response():
    """Debug function to check API response formats"""
    st.subheader("🐛 Debug API Responses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Providers API"):
            try:
                response = requests.get(f"{API_BASE_URL}/api/providers/default", timeout=5)
                st.write(f"Status Code: {response.status_code}")
                st.write("Response Type:", type(response.json()).__name__)
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("Test VMs API"):
            try:
                response = requests.get(f"{API_BASE_URL}/api/vms/default", timeout=5)
                st.write(f"Status Code: {response.status_code}")
                st.write("Response Type:", type(response.json()).__name__)
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {e}")

# Add the debug function to your configuration page (optional)
def show_system_settings_config():
    """Configure system settings - ENHANCED WITH DEBUG"""
    st.subheader("🔧 System Settings")
    
    # Add debug section
    with st.expander("🐛 Debug API Responses"):
        debug_api_response()
    
    try:
        # Get system configuration
        response = requests.get(f"{API_BASE_URL}/api/config/show", timeout=5)
        
        if response.status_code == 200:
            config = response.json()
            
            st.write("**Current System Configuration:**")
            
            # Display configuration in expandable sections
            for category, settings in config.items():
                with st.expander(f"{category.replace('_', ' ').title()} Configuration"):
                    if isinstance(settings, dict):
                        for key, value in settings.items():
                            st.write(f"**{key}:** {value}")
                    else:
                        st.write(f"**{category}:** {settings}")
            
            # Export configuration
            st.subheader("📤 Configuration Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Export Configuration"):
                    export_config()
            
            with col2:
                uploaded_config = st.file_uploader(
                    "Import Configuration",
                    type="json",
                    help="Upload a previously exported configuration file"
                )
                
                if uploaded_config and st.button("📥 Import Configuration"):
                    import_config(uploaded_config)
            
        else:
            st.error(f"Failed to load system configuration: {response.text}")
    
    except Exception as e:
        st.error(f"Error loading system configuration: {e}")
        st.info("System configuration unavailable (API offline)")

# Helper functions for configuration management
def update_provider_config(provider_name, cpu_cost, memory_cost):
    """Update provider configuration - FIXED VERSION"""
    try:
        # The API expects a ConfigurationUpdate format with category and config fields
        config_data = {
            "category": "providers",  # Required field
            "config": {               # Required field
                provider_name: {
                    "cpu_cost": cpu_cost,
                    "memory_cost_gb": memory_cost,
                    "enabled": True  # Add enabled flag
                }
            }
        }
        
        # Use the correct API endpoint for updating configuration
        response = requests.post(
            f"{API_BASE_URL}/api/config/providers",  # Updated endpoint
            json=config_data,
            timeout=10
        )
        
        if response.status_code == 200:
            st.success(f"✅ Updated {provider_name.upper()} configuration!")
            time.sleep(0.5)  # Small delay before refresh
            st.rerun()
        else:
            # More detailed error handling
            try:
                error_detail = response.json()
                st.error(f"Failed to update configuration: {error_detail}")
            except:
                st.error(f"Failed to update configuration: HTTP {response.status_code} - {response.text}")
    
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error updating provider configuration: {e}")
    except Exception as e:
        st.error(f"Error updating provider configuration: {e}")

# Also, let's create a more robust version that works with your current API
def update_provider_config_alternative(provider_name, cpu_cost, memory_cost):
    """Alternative provider update method if the main one doesn't work"""
    try:
        # Try creating a new provider with updated values
        provider_data = {
            "name": provider_name.upper(),
            "cpu_cost": cpu_cost,
            "memory_cost_gb": memory_cost
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/providers",
            json=provider_data,
            timeout=10
        )
        
        if response.status_code == 200:
            st.success(f"✅ Updated {provider_name.upper()} configuration!")
            st.rerun()
        else:
            st.error(f"Failed to update: {response.text}")
    
    except Exception as e:
        st.error(f"Error in alternative update method: {e}")

def preview_csv_structure(uploaded_file):
    """Preview CSV file structure using API"""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/api/workloads/preview", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            st.success("✅ CSV Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File Info:**")
                st.write(f"- Filename: {data['filename']}")
                st.write(f"- Columns: {data['total_columns']}")
                st.write(f"- Preview rows: {data['preview_rows']}")
                st.write(f"- Mapping confidence: {data['mapping_confidence']:.1f}%")
            
            with col2:
                st.write("**Detected Column Mapping:**")
                mapping = data['suggested_mapping']
                st.write(f"- ID Column: `{mapping.get('id_column', 'Not detected')}`")
                st.write(f"- CPU Column: `{mapping.get('cpu_column', 'Not detected')}`")
                st.write(f"- Memory Column: `{mapping.get('memory_column', 'Not detected')}`")
            
            st.write("**Available Columns:**")
            st.code(", ".join(data['columns']))
            
            if data['sample_rows']:
                st.write("**Sample Data:**")
                df_sample = pd.DataFrame(data['sample_rows'])
                st.dataframe(df_sample, use_container_width=True)
        else:
            st.error(f"Preview failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error previewing CSV: {e}")

def export_config():
    """Export system configuration"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/export")
        
        if response.status_code == 200:
            config_data = response.json()
            
            # Convert to JSON string for download
            config_json = json.dumps(config_data, indent=2)
            
            st.download_button(
                label="Download Configuration",
                data=config_json,
                file_name="cloud_scheduler_config.json",
                mime="application/json"
            )
            
            st.success("✅ Configuration exported!")
        else:
            st.error(f"Export failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error exporting configuration: {e}")

def import_config(uploaded_file):
    """Import system configuration"""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/api/config/import", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ {data['message']}")
            st.rerun()
        else:
            st.error(f"Import failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error importing configuration: {e}")

def analyze_current_workloads():
    """Analyze current workloads using the new API endpoint"""
    try:
        if not st.session_state.workloads:
            st.error("No workloads to analyze!")
            return
        
        # Convert workloads to API format
        api_workloads = []
        for w in st.session_state.workloads:
            api_workloads.append({
                "id": w["workload_id"],
                "cpu_required": w["cpu_required"],
                "memory_required_gb": w["memory_required_gb"]
            })
        
        with st.spinner("Analyzing workloads..."):
            response = requests.post(
                f"{API_BASE_URL}/api/workloads/analyze",
                json=api_workloads,
                timeout=30
            )
        
        if response.status_code == 200:
            analysis = response.json()
            st.session_state.workload_analysis = analysis
            st.success("✅ Workload analysis completed!")
            st.rerun()
        else:
            st.error(f"Analysis failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error analyzing workloads: {e}")

def run_scheduler_comparison():
    """Run comparison of multiple schedulers using enhanced API"""
    try:
        if not st.session_state.workloads:
            st.error("No workloads available for comparison!")
            return
        
        # Convert workloads to API format
        api_workloads = []
        for w in st.session_state.workloads:
            api_workloads.append({
                "id": w["workload_id"],
                "cpu_required": w["cpu_required"],
                "memory_required_gb": w["memory_required_gb"]
            })
        
        # Define schedulers to compare
        schedulers = ["random", "lowest_cost", "round_robin"]
        
        comparison_request = {
            "scheduler_types": schedulers,
            "workloads": api_workloads
        }
        
        with st.spinner("Running scheduler comparison..."):
            response = requests.post(
                f"{API_BASE_URL}/api/simulation/compare",
                json=comparison_request,
                timeout=60
            )
        
        if response.status_code == 200:
            comparison_data = response.json()
            
            # Store results in session state
            st.session_state.simulation_results = comparison_data["results"]
            st.session_state.comparison_metrics = comparison_data["comparison_metrics"]
            st.session_state.best_performers = comparison_data["best_performers"]
            st.session_state.recommendations = comparison_data["recommendations"]
            
            st.success("✅ Scheduler comparison completed!")
            st.balloons()
            
            # Show quick summary
            st.subheader("🏆 Quick Comparison Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_success = comparison_data["best_performers"]["highest_success_rate"]
                st.success(f"**Best Success Rate**\n{best_success.replace('_', ' ').title()}")
            
            with col2:
                best_cost = comparison_data["best_performers"]["lowest_cost"]
                st.success(f"**Lowest Cost**\n{best_cost.replace('_', ' ').title()}")
            
            with col3:
                best_balanced = next((r["scheduler"] for r in comparison_data["recommendations"] if r["type"] == "balanced_optimization"), "N/A")
                st.info(f"**Best Balanced**\n{best_balanced.replace('_', ' ').title()}")
            
        else:
            st.error(f"Comparison failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error running comparison: {e}")

# Update the run_simulation function to handle enhanced API response
def run_simulation(schedulers):
    """Run simulation via enhanced API"""
    try:
        if not st.session_state.workloads:
            st.error("No workloads available for simulation!")
            return
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, scheduler in enumerate(schedulers):
            status_text.text(f"Running {scheduler} simulation...")
            
            # Convert workloads to API format
            api_workloads = []
            for w in st.session_state.workloads:
                api_workloads.append({
                    "id": w["workload_id"],
                    "cpu_required": w["cpu_required"],
                    "memory_required_gb": w["memory_required_gb"]
                })
            
            simulation_request = {
                "scheduler_type": scheduler,
                "workloads": api_workloads
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/simulation/run", 
                    json=simulation_request,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Store the complete enhanced response
                    results[scheduler] = data
                    
                else:
                    st.error(f"Simulation failed for {scheduler}: HTTP {response.status_code}")
                    continue
                    
            except Exception as e:
                st.error(f"Error running {scheduler} simulation: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(schedulers))
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.session_state.simulation_results = results
            st.success(f"✅ Completed {len(results)} out of {len(schedulers)} simulations!")
            st.balloons()
        else:
            st.error("❌ All simulations failed!")
    
    except Exception as e:
        st.error(f"Error running simulation: {e}")

# Update the show_results_charts function to handle new API structure
def show_results_charts(results):
    """Debug API response first"""
    st.subheader("🐛 Debug API Response")
    
    for scheduler, data in results.items():
        with st.expander(f"Debug {scheduler} data"):
            st.json(data)  # This will show you the actual structure
    
    # Rest of your chart code...

def show_results():
    st.header("📈 Enhanced Simulation Results")
    
    if not st.session_state.simulation_results:
        st.info("No simulation results available. Please run a simulation first.")
        return
    
    results = st.session_state.simulation_results
    
    # Enhanced Summary metrics with performance indicators
    st.subheader("🎯 Performance Summary")
    
    scheduler_names = list(results.keys())
    cols = st.columns(len(scheduler_names))
    
    for i, scheduler in enumerate(scheduler_names):
        with cols[i]:
            summary = results[scheduler]['summary']
            success_rate = summary['success_rate']
            total_cost = summary.get('total_cost', 0)
            
            # Color-coded metrics based on performance
            if success_rate >= 90:
                st.success(f"**{scheduler.replace('_', ' ').title()}**")
            elif success_rate >= 70:
                st.warning(f"**{scheduler.replace('_', ' ').title()}**")
            else:
                st.error(f"**{scheduler.replace('_', ' ').title()}**")
            
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                f"{summary['successful_assignments']}/{summary['total_workloads']}"
            )
            
            st.metric(
                "Total Cost",
                f"${total_cost:.3f}",
                help="Total cost for all successful assignments"
            )
            
            # Additional performance metrics if available
            if 'performance_metrics' in summary:
                perf = summary['performance_metrics']
                st.metric(
                    "Throughput",
                    f"{perf.get('assignments_per_second', 0):.1f}/sec",
                    help="Assignments processed per second"
                )
    
    # Show comparison metrics if available (from scheduler comparison)
    if hasattr(st.session_state, 'best_performers') and st.session_state.best_performers:
        st.subheader("🏆 Best Performers")
        
        best_performers = st.session_state.best_performers
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**🎯 Highest Success Rate**\n{best_performers['highest_success_rate'].replace('_', ' ').title()}")
        
        with col2:
            st.success(f"**💰 Lowest Cost**\n{best_performers['lowest_cost'].replace('_', ' ').title()}")
        
        with col3:
            st.success(f"**⚡ Fastest**\n{best_performers['fastest_assignment'].replace('_', ' ').title()}")
    
    # Show recommendations if available
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        st.subheader("💡 AI Recommendations")
        
        for rec in st.session_state.recommendations:
            if rec["type"] == "cost_optimization":
                st.info(f"💰 **Cost Focus:** Use {rec['scheduler'].replace('_', ' ').title()} - {rec['reason']}")
            elif rec["type"] == "performance_optimization":
                st.success(f"🎯 **Performance Focus:** Use {rec['scheduler'].replace('_', ' ').title()} - {rec['reason']}")
            elif rec["type"] == "balanced_optimization":
                st.warning(f"⚖️ **Balanced Choice:** Use {rec['scheduler'].replace('_', ' ').title()} - {rec['reason']}")
    
    # Enhanced detailed charts
    show_results_charts(results)


def main():
    st.title("☁️ AI-Powered Cloud Scheduler")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Configuration", "Simulation", "Results", "Analysis", "ML Predictions"]
    )
    
    # Add API status check
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.warning("🟡 API Issues")
    except:
        st.sidebar.error("🔴 API Offline")
    
    # Page routing
    try:
        if page == "Dashboard":
            show_dashboard()
        elif page == "Configuration":
            show_configuration()
        elif page == "Simulation":
            show_simulation()
        elif page == "Results":
            show_results()
        elif page == "Analysis":
            show_analysis()
        elif page == "ML Predictions":
            show_ml_predictions()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.info("Please check if the API server is running: `python api.py`")

def show_dashboard():
    st.header("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total VMs", "4", "Default Configuration")
    
    with col2:
        st.metric("Schedulers Available", "5", "Including Enhanced Options")
    
    with col3:
        workload_count = len(st.session_state.workloads)
        st.metric("Loaded Workloads", workload_count)
        
        # Show workload analysis if available
        if st.session_state.workload_analysis:
            analysis = st.session_state.workload_analysis
            cpu_util = analysis["capacity_analysis"]["projected_cpu_utilization"]
            if cpu_util > 80:
                st.warning(f"High CPU utilization expected: {cpu_util:.1f}%")
    
    with col4:
        if st.session_state.simulation_results:
            st.metric("Last Simulation", "Completed", "✅")
        else:
            st.metric("Last Simulation", "None", "❌")
    
    st.markdown("---")
    
    # Enhanced Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Load Sample Data", use_container_width=True):
            load_sample_workloads()
    
    with col2:
        if st.button("📊 Analyze Workloads", use_container_width=True):
            if st.session_state.workloads:
                analyze_current_workloads()
            else:
                st.error("Please load workloads first!")
    
    with col3:
        if st.button("🔄 Compare Schedulers", use_container_width=True):
            if st.session_state.workloads:
                run_scheduler_comparison()
            else:
                st.error("Please load workloads first!")
    
    with col4:
        if st.button("▶️ Quick Simulation", use_container_width=True):
            if st.session_state.workloads:
                run_quick_simulation()
            else:
                st.error("Please load workloads first!")

def show_analysis():
    """New analysis page with comprehensive insights"""
    st.header("🔍 Comprehensive Analysis")
    
    if not st.session_state.simulation_results:
        st.info("No simulation results to analyze. Please run a simulation first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Result Analysis", "Workload Insights", "Performance Deep Dive"])
    
    with tab1:
        show_result_analysis()
    
    with tab2:
        show_workload_insights()
    
    with tab3:
        show_performance_deep_dive()

def show_result_analysis():
    """Analyze simulation results in detail"""
    st.subheader("📊 Simulation Result Analysis")
    
    # Select scheduler for detailed analysis
    scheduler_options = list(st.session_state.simulation_results.keys())
    selected_scheduler = st.selectbox(
        "Select scheduler for detailed analysis:",
        scheduler_options,
        key="analysis_scheduler_select"
    )
    
    if selected_scheduler:
        result_data = st.session_state.simulation_results[selected_scheduler]
        
        # Analyze using API
        if st.button("🔍 Run Deep Analysis"):
            analyze_simulation_results(selected_scheduler, result_data)

def analyze_simulation_results(scheduler, result_data):
    """Analyze simulation results using API"""
    try:
        with st.spinner(f"Analyzing {scheduler} results..."):
            response = requests.post(
                f"{API_BASE_URL}/api/simulation/analyze", 
                json=result_data,
                timeout=30
            )
        
        if response.status_code == 200:
            analysis = response.json()
            
            st.success("✅ Deep analysis completed!")
            
            # Display analysis results
            show_analysis_results(analysis, scheduler)
        else:
            st.error(f"Analysis failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error analyzing results: {e}")

def show_analysis_results(analysis, scheduler):
    """Display analysis results"""
    st.subheader(f"📈 Analysis Results for {scheduler.replace('_', ' ').title()}")
    
    # Scheduler performance
    if "scheduler_performance" in analysis["analysis"]:
        perf = analysis["analysis"]["scheduler_performance"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Efficiency Score", f"{perf.get('efficiency_score', 0):.1f}%")
        
        with col2:
            st.metric("Avg Assignment Time", f"{perf.get('average_assignment_time_ms', 0):.1f}ms")
        
        with col3:
            st.metric("Total Cost", f"${perf.get('total_successful_cost', 0):.3f}")
        
        with col4:
            st.metric("Cost per Assignment", f"${perf.get('cost_per_successful_assignment', 0):.3f}")
    
    # Resource patterns
    if "resource_patterns" in analysis["analysis"]:
        patterns = analysis["analysis"]["resource_patterns"]
        
        st.subheader("📊 Resource Usage Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**CPU Usage Pattern:**")
            cpu_pattern = patterns.get("cpu_usage", {})
            for metric, value in cpu_pattern.items():
                st.write(f"- **{metric.title()}:** {value}")
        
        with col2:
            st.write("**Memory Usage Pattern:**")
            memory_pattern = patterns.get("memory_usage", {})
            for metric, value in memory_pattern.items():
                st.write(f"- **{metric.title()}:** {value}")
    
    # Bottlenecks
    if "bottlenecks" in analysis["analysis"]:
        bottlenecks = analysis["analysis"]["bottlenecks"]
        
        if bottlenecks:
            st.subheader("⚠️ Identified Bottlenecks")
            
            for bottleneck in bottlenecks:
                st.warning(f"**{bottleneck['reason']}:** {bottleneck['count']} occurrences ({bottleneck['percentage']}%)")
    
    # Recommendations
    if "recommendations" in analysis["analysis"]:
        recommendations = analysis["analysis"]["recommendations"]
        
        if recommendations:
            st.subheader("💡 Optimization Recommendations")
            
            for rec in recommendations:
                if rec["priority"] == "high":
                    st.error(f"🔴 **HIGH PRIORITY:** {rec['message']}")
                elif rec["priority"] == "medium":
                    st.warning(f"🟡 **MEDIUM PRIORITY:** {rec['message']}")
                else:
                    st.info(f"🔵 **LOW PRIORITY:** {rec['message']}")

def show_workload_insights():
    """Show workload analysis insights"""
    st.subheader("📋 Workload Analysis Insights")
    
    if st.session_state.workload_analysis:
        analysis = st.session_state.workload_analysis
        
        # Statistics
        stats = analysis["statistics"]
        patterns = analysis["patterns"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Workload Statistics:**")
            st.json(stats)
        
        with col2:
            st.write("**Workload Patterns:**")
            
            # Pattern visualization
            pattern_data = pd.DataFrame([patterns])
            pattern_melted = pattern_data.melt(var_name='Pattern', value_name='Count')
            
            fig_patterns = px.bar(
                pattern_melted, 
                x='Pattern', 
                y='Count',
                title="Workload Size Distribution"
            )
            st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Capacity analysis
        st.subheader("🎯 Capacity Planning")
        
        capacity = analysis["capacity_analysis"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_util = capacity["projected_cpu_utilization"]
            st.metric("Projected CPU Utilization", f"{cpu_util:.1f}%")
            st.progress(min(100, cpu_util) / 100)
        
        with col2:
            mem_util = capacity["projected_memory_utilization"]
            st.metric("Projected Memory Utilization", f"{mem_util:.1f}%")
            st.progress(min(100, mem_util) / 100)
        
        # Recommendations
        if analysis["recommendations"]:
            st.subheader("💡 Capacity Recommendations")
            
            for rec in analysis["recommendations"]:
                if rec["priority"] == "high":
                    st.error(f"🔴 **{rec['type'].upper()}:** {rec['message']}")
                else:
                    st.warning(f"🟡 **{rec['type'].upper()}:** {rec['message']}")
    else:
        st.info("No workload analysis available. Please analyze your workloads first.")
        
        if st.session_state.workloads and st.button("🔍 Analyze Current Workloads"):
            analyze_current_workloads()

def show_performance_deep_dive():
    """Deep dive into performance metrics"""
    st.subheader("⚡ Performance Deep Dive")
    
    if not st.session_state.simulation_results:
        st.info("No simulation results available for performance analysis.")
        return
    
    # Performance comparison across all schedulers
    results = st.session_state.simulation_results
    
    # Extract performance metrics
    perf_data = []
    resource_data = []
    timing_data = []
    
    for scheduler, data in results.items():
        summary = data.get("summary", {})
        
        # Extract performance metrics (handle both old and new API formats)
        perf_metrics = summary.get("performance_metrics", {})
        resource_util = summary.get("resource_utilization", {})
        
        # Performance data
        perf_entry = {
            "Scheduler": scheduler.replace('_', ' ').title(),
            "Success Rate (%)": summary.get("success_rate", 0),
            "Total Cost ($)": summary.get("total_cost", 0),
            "Successful Assignments": summary.get("successful_assignments", 0),
            "Total Workloads": summary.get("total_workloads", 0),
            "Assignment Time (ms)": perf_metrics.get("average_assignment_time_ms", 0),
            "Throughput (assignments/sec)": perf_metrics.get("assignments_per_second", 0),
            "Simulation Time (s)": perf_metrics.get("total_simulation_time_seconds", 0)
        }
        perf_data.append(perf_entry)
        
        # Resource utilization data
        resource_entry = {
            "Scheduler": scheduler.replace('_', ' ').title(),
            "CPU Utilization (%)": resource_util.get("overall_cpu_utilization", 0),
            "Memory Utilization (%)": resource_util.get("overall_memory_utilization", 0),
            "Total CPU Used": resource_util.get("total_cpu_used", 0),
            "Total Memory Used (GB)": resource_util.get("total_memory_used", 0),
            "CPU Capacity": resource_util.get("total_cpu_capacity", 0),
            "Memory Capacity (GB)": resource_util.get("total_memory_capacity", 0)
        }
        resource_data.append(resource_entry)
        
        # Timing analysis from logs
        logs = data.get("logs", [])
        successful_logs = [log for log in logs if log.get("success", False)]
        
        if successful_logs:
            assignment_times = [log.get("assignment_time_ms", 0) for log in successful_logs]
            avg_time = sum(assignment_times) / len(assignment_times) if assignment_times else 0
            min_time = min(assignment_times) if assignment_times else 0
            max_time = max(assignment_times) if assignment_times else 0
            
            timing_entry = {
                "Scheduler": scheduler.replace('_', ' ').title(),
                "Avg Assignment Time (ms)": avg_time,
                "Min Assignment Time (ms)": min_time,
                "Max Assignment Time (ms)": max_time,
                "Total Assignments": len(successful_logs),
                "Time Std Dev": 0  # Calculate standard deviation if needed
            }
            
            # Calculate standard deviation
            if len(assignment_times) > 1:
                mean = avg_time
                variance = sum((x - mean) ** 2 for x in assignment_times) / len(assignment_times)
                timing_entry["Time Std Dev"] = variance ** 0.5
            
            timing_data.append(timing_entry)
    
    if not perf_data:
        st.error("No performance data available for analysis.")
        return
    
    # Convert to DataFrames
    df_perf = pd.DataFrame(perf_data)
    df_resource = pd.DataFrame(resource_data)
    df_timing = pd.DataFrame(timing_data) if timing_data else pd.DataFrame()
    
    # Performance Overview
    st.subheader("📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_success = df_perf.loc[df_perf["Success Rate (%)"].idxmax(), "Scheduler"]
        max_success = df_perf["Success Rate (%)"].max()
        st.metric("Best Success Rate", f"{max_success:.1f}%", f"{best_success}")
    
    with col2:
        best_cost = df_perf.loc[df_perf["Total Cost ($)"].idxmin(), "Scheduler"]
        min_cost = df_perf["Total Cost ($)"].min()
        st.metric("Lowest Cost", f"${min_cost:.3f}", f"{best_cost}")
    
    with col3:
        if not df_timing.empty:
            best_speed = df_timing.loc[df_timing["Avg Assignment Time (ms)"].idxmin(), "Scheduler"]
            min_time = df_timing["Avg Assignment Time (ms)"].min()
            st.metric("Fastest Assignment", f"{min_time:.1f}ms", f"{best_speed}")
        else:
            st.metric("Fastest Assignment", "N/A", "No timing data")
    
    with col4:
        best_throughput = df_perf.loc[df_perf["Throughput (assignments/sec)"].idxmax(), "Scheduler"]
        max_throughput = df_perf["Throughput (assignments/sec)"].max()
        st.metric("Highest Throughput", f"{max_throughput:.1f}/s", f"{best_throughput}")
    
    # Detailed Performance Charts
    st.subheader("📈 Detailed Performance Analysis")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["Success & Cost", "Resource Utilization", "Timing Analysis", "Efficiency Metrics"])
    
    with tab1:
        # Success Rate vs Cost Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_success = px.bar(
                df_perf, 
                x="Scheduler", 
                y="Success Rate (%)",
                title="Success Rate by Scheduler",
                color="Success Rate (%)",
                color_continuous_scale="RdYlGn"
            )
            fig_success.update_layout(height=400)
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            fig_cost = px.bar(
                df_perf, 
                x="Scheduler", 
                y="Total Cost ($)",
                title="Total Cost by Scheduler",
                color="Total Cost ($)",
                color_continuous_scale="RdYlBu_r"
            )
            fig_cost.update_layout(height=400)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Success Rate vs Cost Scatter Plot
        fig_scatter = px.scatter(
            df_perf,
            x="Total Cost ($)",
            y="Success Rate (%)",
            size="Successful Assignments",
            color="Scheduler",
            title="Success Rate vs Cost Trade-off",
            hover_data=["Throughput (assignments/sec)"]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        # Resource Utilization Analysis
        if not df_resource.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cpu = px.bar(
                    df_resource,
                    x="Scheduler",
                    y="CPU Utilization (%)",
                    title="CPU Utilization by Scheduler",
                    color="CPU Utilization (%)",
                    color_continuous_scale="Blues"
                )
                fig_cpu.update_layout(height=400)
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                fig_memory = px.bar(
                    df_resource,
                    x="Scheduler",
                    y="Memory Utilization (%)",
                    title="Memory Utilization by Scheduler",
                    color="Memory Utilization (%)",
                    color_continuous_scale="Reds"
                )
                fig_memory.update_layout(height=400)
                st.plotly_chart(fig_memory, use_container_width=True)
            
            # Combined resource utilization heatmap
            resource_matrix = df_resource[["Scheduler", "CPU Utilization (%)", "Memory Utilization (%)"]].set_index("Scheduler").T
            
            fig_heatmap = px.imshow(
                resource_matrix,
                title="Resource Utilization Heatmap",
                color_continuous_scale="RdYlBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Resource efficiency table
            st.subheader("📋 Resource Efficiency Summary")
            st.dataframe(df_resource.round(2), use_container_width=True)
        else:
            st.info("No resource utilization data available.")
    
    with tab3:
        # Timing Analysis
        if not df_timing.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_avg_time = px.bar(
                    df_timing,
                    x="Scheduler",
                    y="Avg Assignment Time (ms)",
                    title="Average Assignment Time",
                    color="Avg Assignment Time (ms)",
                    color_continuous_scale="Viridis"
                )
                fig_avg_time.update_layout(height=400)
                st.plotly_chart(fig_avg_time, use_container_width=True)
            
            with col2:
                # Min/Max timing comparison
                fig_minmax = go.Figure()
                
                fig_minmax.add_trace(go.Bar(
                    x=df_timing["Scheduler"],
                    y=df_timing["Min Assignment Time (ms)"],
                    name="Minimum Time",
                    marker_color='lightblue'
                ))
                
                fig_minmax.add_trace(go.Bar(
                    x=df_timing["Scheduler"],
                    y=df_timing["Max Assignment Time (ms)"],
                    name="Maximum Time",
                    marker_color='lightcoral'
                ))
                
                fig_minmax.update_layout(
                    title="Min/Max Assignment Times",
                    xaxis_title="Scheduler",
                    yaxis_title="Time (ms)",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig_minmax, use_container_width=True)
            
            # Timing variability analysis
            fig_variability = px.bar(
                df_timing,
                x="Scheduler",
                y="Time Std Dev",
                title="Assignment Time Variability (Standard Deviation)",
                color="Time Std Dev",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig_variability, use_container_width=True)
            
            # Detailed timing table
            st.subheader("⏱️ Timing Analysis Summary")
            st.dataframe(df_timing.round(2), use_container_width=True)
        else:
            st.info("No timing data available for detailed analysis.")
    
    with tab4:
        # Efficiency Metrics
        st.subheader("⚡ Efficiency Metrics")
        
        # Calculate efficiency scores
        efficiency_data = []
        
        for _, row in df_perf.iterrows():
            scheduler = row["Scheduler"]
            success_rate = row["Success Rate (%)"]
            cost = row["Total Cost ($)"]
            throughput = row["Throughput (assignments/sec)"]
            
            # Calculate various efficiency metrics
            cost_efficiency = success_rate / (cost + 0.001)  # Success per dollar (avoid division by zero)
            time_efficiency = throughput  # Assignments per second
            overall_efficiency = (success_rate * 0.4) + (cost_efficiency * 0.3) + (time_efficiency * 0.3)
            
            efficiency_data.append({
                "Scheduler": scheduler,
                "Cost Efficiency (Success/$ * 100)": cost_efficiency * 100,
                "Time Efficiency (assignments/sec)": time_efficiency,
                "Overall Efficiency Score": overall_efficiency
            })
        
        df_efficiency = pd.DataFrame(efficiency_data)
        
        # Efficiency visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cost_eff = px.bar(
                df_efficiency,
                x="Scheduler",
                y="Cost Efficiency (Success/$ * 100)",
                title="Cost Efficiency (Success Rate per Dollar)",
                color="Cost Efficiency (Success/$ * 100)",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_cost_eff, use_container_width=True)
        
        with col2:
            fig_overall_eff = px.bar(
                df_efficiency,
                x="Scheduler",
                y="Overall Efficiency Score",
                title="Overall Efficiency Score",
                color="Overall Efficiency Score",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_overall_eff, use_container_width=True)
        
        # Efficiency radar chart
        if len(df_efficiency) > 0:
            fig_radar = go.Figure()
            
            for _, row in df_efficiency.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        row["Cost Efficiency (Success/$ * 100)"] / df_efficiency["Cost Efficiency (Success/$ * 100)"].max() * 100,
                        row["Time Efficiency (assignments/sec)"] / df_efficiency["Time Efficiency (assignments/sec)"].max() * 100,
                        row["Overall Efficiency Score"] / df_efficiency["Overall Efficiency Score"].max() * 100
                    ],
                    theta=["Cost Efficiency", "Time Efficiency", "Overall Efficiency"],
                    fill='toself',
                    name=row["Scheduler"]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Efficiency Comparison Radar Chart"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Efficiency recommendations
        st.subheader("💡 Efficiency Recommendations")
        
        best_cost_eff = df_efficiency.loc[df_efficiency["Cost Efficiency (Success/$ * 100)"].idxmax(), "Scheduler"]
        best_time_eff = df_efficiency.loc[df_efficiency["Time Efficiency (assignments/sec)"].idxmax(), "Scheduler"]
        best_overall = df_efficiency.loc[df_efficiency["Overall Efficiency Score"].idxmax(), "Scheduler"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**💰 Best Cost Efficiency**\n{best_cost_eff}")
        
        with col2:
            st.info(f"**⚡ Best Time Efficiency**\n{best_time_eff}")
        
        with col3:
            st.warning(f"**🏆 Best Overall Efficiency**\n{best_overall}")
        
        # Detailed efficiency table
        st.subheader("📊 Efficiency Metrics Table")
        st.dataframe(df_efficiency.round(3), use_container_width=True)
    
    # Performance Insights
    st.subheader("🔍 Performance Insights")
    
    insights = []
    
    # Generate insights based on the data
    max_success_scheduler = df_perf.loc[df_perf["Success Rate (%)"].idxmax(), "Scheduler"]
    min_cost_scheduler = df_perf.loc[df_perf["Total Cost ($)"].idxmin(), "Scheduler"]
    
    if max_success_scheduler == min_cost_scheduler:
        insights.append(f"🎯 **{max_success_scheduler}** excels in both success rate and cost efficiency!")
    else:
        insights.append(f"⚖️ Trade-off detected: **{max_success_scheduler}** has the highest success rate, while **{min_cost_scheduler}** has the lowest cost.")
    
    # Check for resource utilization insights
    if not df_resource.empty:
        high_cpu_schedulers = df_resource[df_resource["CPU Utilization (%)"] > 70]["Scheduler"].tolist()
        if high_cpu_schedulers:
            insights.append(f"🔥 High CPU utilization (>70%) detected in: {', '.join(high_cpu_schedulers)}")
        
        low_util_schedulers = df_resource[
            (df_resource["CPU Utilization (%)"] < 30) & 
            (df_resource["Memory Utilization (%)"] < 30)
        ]["Scheduler"].tolist()
        
        if low_util_schedulers:
            insights.append(f"💤 Low resource utilization (<30%) in: {', '.join(low_util_schedulers)}. Consider optimization.")
    
    # Timing insights
    if not df_timing.empty:
        fast_schedulers = df_timing[df_timing["Avg Assignment Time (ms)"] < 5]["Scheduler"].tolist()
        if fast_schedulers:
            insights.append(f"⚡ Fast assignment times (<5ms) in: {', '.join(fast_schedulers)}")
    
    # Display insights
    for insight in insights:
        st.info(insight)
    
    # Export option
    st.subheader("📤 Export Performance Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Performance Data"):
            csv_data = df_perf.to_csv(index=False)
            st.download_button(
                label="Download Performance CSV",
                data=csv_data,
                file_name="scheduler_performance_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if not df_resource.empty and st.button("🖥️ Export Resource Data"):
            csv_data = df_resource.to_csv(index=False)
            st.download_button(
                label="Download Resource CSV",
                data=csv_data,
                file_name="resource_utilization_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        if not df_timing.empty and st.button("⏱️ Export Timing Data"):
            csv_data = df_timing.to_csv(index=False)
            st.download_button(
                label="Download Timing CSV",
                data=csv_data,
                file_name="timing_analysis.csv",
                mime="text/csv"
            )

def show_simulation():
    st.header("▶️ Run Simulation")
    if not st.session_state.workloads:
        st.warning("Please load workloads first!")
        return
    
    # Scheduler selection
    st.subheader("Select Schedulers")
    schedulers = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.checkbox("Random Scheduler", value=True):
            schedulers.append("random")
    
    with col2:
        if st.checkbox("Lowest Cost Scheduler", value=True):
            schedulers.append("lowest_cost")
    
    with col3:
        if st.checkbox("Round Robin Scheduler", value=True):
            schedulers.append("round_robin")
    
    if not schedulers:
        st.warning("Please select at least one scheduler!")
        return
    
    # Simulation options
    st.subheader("Simulation Options")
    
    # Start simulation
    if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
        run_simulation(schedulers)

def show_results():
    st.header("📈 Simulation Results")
    
    if not st.session_state.simulation_results:
        st.info("No simulation results available. Please run a simulation first.")
        return
    
    results = st.session_state.simulation_results
    
    # Summary metrics
    st.subheader("Summary")
    
    scheduler_names = list(results.keys())
    cols = st.columns(len(scheduler_names))
    
    for i, scheduler in enumerate(scheduler_names):
        with cols[i]:
            summary = results[scheduler]['summary']
            st.metric(
                f"{scheduler.replace('_', ' ').title()}",
                f"{summary['success_rate']:.1f}%",
                f"{summary['successful_workloads']}/{summary['total_workloads']} success"
            )
    
    # Detailed charts
    show_results_charts(results)

def show_results_charts(results):
    """Debug API response first"""
    st.subheader("🐛 Debug API Response")
    
    for scheduler, data in results.items():
        with st.expander(f"Debug {scheduler} data"):
            st.json(data)  # This will show you the actual structure
    
    # Rest of your chart code...

def show_ml_predictions():
    st.header("🤖 ML Predictions")
    
    # Check model status
    try:
        response = requests.get(f"{API_BASE_URL}/api/ml/model-status")
        if response.status_code == 200:
            status = response.json()
            model_trained = status["model_trained"]
        else:
            model_trained = False
    except:
        model_trained = False
    
    # Model status indicator
    if model_trained:
        st.success("✅ ML Model is trained and ready!")
    else:
        st.warning("⚠️ ML Model not trained yet")
    
    tab1, tab2, tab3 = st.tabs(["Train Model", "Make Predictions", "Model Info"])
    
    with tab1:
        st.subheader("📚 Train LSTM Model")
        
        # Upload training data
        uploaded_file = st.file_uploader(
            "Upload Historical CPU Usage Data",
            type="csv",
            help="CSV should contain 'timestamp' and 'cpu_usage' columns"
        )
        
        if uploaded_file:
            if st.button("Upload Training Data"):
                upload_training_data(uploaded_file)
        
        st.markdown("---")
        
        # Train model
        if st.button("🚀 Train Model", type="primary"):
            train_model()
    
    with tab2:
        st.subheader("🔮 CPU Usage Predictions")
        
        if not model_trained:
            st.info("Please train the model first!")
            return
        
        # Single prediction
        st.write("**Single Step Prediction**")
        
        # Input for sequence data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Enter last 12 CPU usage values (%):")
            sequence_input = st.text_area(
                "Comma-separated values",
                value="45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7",
                height=100
            )
        
        with col2:
            if st.button("Predict Next Value"):
                make_single_prediction(sequence_input)
        
        st.markdown("---")
        
        # Multi-step prediction
        st.write("**Multi-Step Prediction**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            steps = st.number_input("Number of future steps", min_value=1, max_value=20, value=5)
        
        with col2:
            if st.button("Predict Multiple Steps"):
                make_multiple_predictions(sequence_input, steps)
    
    with tab3:
        st.subheader("ℹ️ Model Information")
        
        if model_trained:
            st.write("**Model Architecture:**")
            st.code("""
LSTM Model:
- Input: 12 time steps (1 hour of 5-minute intervals)
- LSTM Layer 1: 50 units (return sequences)
- LSTM Layer 2: 50 units
- Dense Layer 1: 25 units
- Output Layer: 1 unit (CPU usage prediction)

Training Details:
- Optimizer: Adam
- Loss: Mean Squared Error
- Batch Size: 32
- Epochs: 50
            """)
            
            st.write("**Use Cases:**")
            st.write("- Proactive resource scaling")
            st.write("- Load balancing decisions")
            st.write("- Capacity planning")
            st.write("- Anomaly detection")
        else:
            st.info("Train the model to see detailed information")

# Helper functions for ML
def upload_training_data(uploaded_file):
    """Upload training data for ML model"""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/api/ml/upload-training-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ {data['message']}")
            st.write(f"Rows: {data['rows']}")
            st.write(f"Columns: {data['columns']}")
        else:
            st.error(f"Upload failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error uploading training data: {e}")

def train_model():
    """Train the ML model"""
    try:
        with st.spinner("Training model... This may take a few minutes."):
            response = requests.post(f"{API_BASE_URL}/api/ml/train")
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ {data['message']}")
            st.balloons()
            st.rerun()
        else:
            st.error(f"Training failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error training model: {e}")

def make_single_prediction(sequence_input):
    """Make a single CPU usage prediction"""
    try:
        # Parse input
        sequence_data = [float(x.strip()) for x in sequence_input.split(',')]
        
        if len(sequence_data) != 12:
            st.error("Please provide exactly 12 values")
            return
        
        response = requests.post(f"{API_BASE_URL}/api/ml/predict", json=sequence_data)
        
        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            
            st.success(f"🎯 Next CPU Usage Prediction: **{prediction:.2f}%**")
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=sequence_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Prediction
            fig.add_trace(go.Scatter(
                x=[13],
                y=[prediction],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title="CPU Usage Prediction",
                xaxis_title="Time Step",
                yaxis_title="CPU Usage (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")

def make_multiple_predictions(sequence_input, steps):
    """Make multiple CPU usage predictions"""
    try:
        # Parse input
        sequence_data = [float(x.strip()) for x in sequence_input.split(',')]
        
        if len(sequence_data) != 12:
            st.error("Please provide exactly 12 values")
            return
        
        response = requests.post(
            f"{API_BASE_URL}/api/ml/predict-multiple",
            json=sequence_data,
            params={"steps": steps}
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]
            
            st.success(f"🎯 Next {steps} CPU Usage Predictions:")
            
            # Display predictions
            for i, pred in enumerate(predictions, 1):
                st.write(f"Step {i}: **{pred:.2f}%**")
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=sequence_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=list(range(13, 13 + steps)),
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"CPU Usage - Next {steps} Steps Prediction",
                xaxis_title="Time Step",
                yaxis_title="CPU Usage (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error making predictions: {e}")

# Helper functions
def load_sample_workloads():
    """Load sample workloads for testing"""
    try:
        # Try to get sample workloads from API first
        response = requests.get(f"{API_BASE_URL}/api/workloads/sample", timeout=5)
    
    fig.add_trace(
        go.Bar(x=scheduler_names, y=success_rates, name="Success Rate", marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=scheduler_names, y=total_costs, name="Total Cost", marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=scheduler_names, y=successful_workloads, name="Successful Workloads", marker_color='coral'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results for each scheduler
    st.subheader("Detailed Scheduler Results")
    
    for scheduler, data in results.items():
        logs = data['logs']
        
        with st.expander(f"{scheduler.replace('_', ' ').title()} Details"):
            # Show summary metrics
            summary = data['summary']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
            with col2:
                st.metric("Successful", f"{summary['successful_workloads']}")
            with col3:
                st.metric("Total Workloads", f"{summary['total_workloads']}")
            with col4:
                st.metric("Total Cost", f"${summary.get('total_cost', 0):.2f}")
            
            # Show assignment details
            st.subheader("Workload Assignments")
            
            # Convert logs to DataFrame for better display
            if logs:
                df_logs = pd.DataFrame(logs)
                
                # Add status indicator
                df_logs['Status'] = df_logs['success'].apply(lambda x: '✅ Success' if x else '❌ Failed')
                
                # Display the table - FIXED syntax error here
                display_columns = ['workload_id', 'vm_id', 'Status', 'message']
                available_columns = [col for col in display_columns if col in df_logs.columns]  # FIXED: removed 'are'
                
                st.dataframe(df_logs[available_columns], use_container_width=True)
                
                # Simple success/failure chart
                success_count = df_logs['success'].sum()
                failure_count = len(df_logs) - success_count
                
                if success_count > 0 or failure_count > 0:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Success', 'Failed'],
                        values=[success_count, failure_count],
                        hole=.3,
                        marker_colors=['lightgreen', 'lightcoral']
                    )])
                    
                    fig_pie.update_layout(
                        title=f"{scheduler.replace('_', ' ').title()} - Assignment Results",
                        height=300
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No assignment logs available for this scheduler.")

def show_ml_predictions():
    st.header("🤖 ML Predictions")
    
    # Check model status
    try:
        response = requests.get(f"{API_BASE_URL}/api/ml/model-status")
        if response.status_code == 200:
            status = response.json()
            model_trained = status["model_trained"]
        else:
            model_trained = False
    except:
        model_trained = False
    
    # Model status indicator
    if model_trained:
        st.success("✅ ML Model is trained and ready!")
    else:
        st.warning("⚠️ ML Model not trained yet")
    
    tab1, tab2, tab3 = st.tabs(["Train Model", "Make Predictions", "Model Info"])
    
    with tab1:
        st.subheader("📚 Train LSTM Model")
        
        # Upload training data
        uploaded_file = st.file_uploader(
            "Upload Historical CPU Usage Data",
            type="csv",
            help="CSV should contain 'timestamp' and 'cpu_usage' columns"
        )
        
        if uploaded_file:
            if st.button("Upload Training Data"):
                upload_training_data(uploaded_file)
        
        st.markdown("---")
        
        # Train model
        if st.button("🚀 Train Model", type="primary"):
            train_model()
    
    with tab2:
        st.subheader("🔮 CPU Usage Predictions")
        
        if not model_trained:
            st.info("Please train the model first!")
            return
        
        # Single prediction
        st.write("**Single Step Prediction**")
        
        # Input for sequence data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Enter last 12 CPU usage values (%):")
            sequence_input = st.text_area(
                "Comma-separated values",
                value="45.2, 52.3, 48.1, 55.7, 42.8, 38.9, 51.2, 47.6, 49.3, 44.1, 53.8, 46.7",
                height=100
            )
        
        with col2:
            if st.button("Predict Next Value"):
                make_single_prediction(sequence_input)
        
        st.markdown("---")
        
        # Multi-step prediction
        st.write("**Multi-Step Prediction**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            steps = st.number_input("Number of future steps", min_value=1, max_value=20, value=5)
        
        with col2:
            if st.button("Predict Multiple Steps"):
                make_multiple_predictions(sequence_input, steps)
    
    with tab3:
        st.subheader("ℹ️ Model Information")
        
        if model_trained:
            st.write("**Model Architecture:**")
            st.code("""
LSTM Model:
- Input: 12 time steps (1 hour of 5-minute intervals)
- LSTM Layer 1: 50 units (return sequences)
- LSTM Layer 2: 50 units
- Dense Layer 1: 25 units
- Output Layer: 1 unit (CPU usage prediction)

Training Details:
- Optimizer: Adam
- Loss: Mean Squared Error
- Batch Size: 32
- Epochs: 50
            """)
            
            st.write("**Use Cases:**")
            st.write("- Proactive resource scaling")
            st.write("- Load balancing decisions")
            st.write("- Capacity planning")
            st.write("- Anomaly detection")
        else:
            st.info("Train the model to see detailed information")

# Helper functions for ML
def upload_training_data(uploaded_file):
    """Upload training data for ML model"""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/api/ml/upload-training-data", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ {data['message']}")
            st.write(f"Rows: {data['rows']}")
            st.write(f"Columns: {data['columns']}")
        else:
            st.error(f"Upload failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error uploading training data: {e}")

def train_model():
    """Train the ML model"""
    try:
        with st.spinner("Training model... This may take a few minutes."):
            response = requests.post(f"{API_BASE_URL}/api/ml/train")
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ {data['message']}")
            st.balloons()
            st.rerun()
        else:
            st.error(f"Training failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error training model: {e}")

def make_single_prediction(sequence_input):
    """Make a single CPU usage prediction"""
    try:
        # Parse input
        sequence_data = [float(x.strip()) for x in sequence_input.split(',')]
        
        if len(sequence_data) != 12:
            st.error("Please provide exactly 12 values")
            return
        
        response = requests.post(f"{API_BASE_URL}/api/ml/predict", json=sequence_data)
        
        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            
            st.success(f"🎯 Next CPU Usage Prediction: **{prediction:.2f}%**")
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=sequence_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Prediction
            fig.add_trace(go.Scatter(
                x=[13],
                y=[prediction],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title="CPU Usage Prediction",
                xaxis_title="Time Step",
                yaxis_title="CPU Usage (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")

def make_multiple_predictions(sequence_input, steps):
    """Make multiple CPU usage predictions"""
    try:
        # Parse input
        sequence_data = [float(x.strip()) for x in sequence_input.split(',')]
        
        if len(sequence_data) != 12:
            st.error("Please provide exactly 12 values")
            return
        
        response = requests.post(
            f"{API_BASE_URL}/api/ml/predict-multiple",
            json=sequence_data,
            params={"steps": steps}
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]
            
            st.success(f"🎯 Next {steps} CPU Usage Predictions:")
            
            # Display predictions
            for i, pred in enumerate(predictions, 1):
                st.write(f"Step {i}: **{pred:.2f}%**")
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=sequence_data,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=list(range(13, 13 + steps)),
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"CPU Usage - Next {steps} Steps Prediction",
                xaxis_title="Time Step",
                yaxis_title="CPU Usage (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error making predictions: {e}")

# Helper functions
def load_sample_workloads():
    """Load sample workloads for testing"""
    try:
        # Try to get sample workloads from API first
        response = requests.get(f"{API_BASE_URL}/api/workloads/sample", timeout=5)
        if response.status_code == 200:
            api_workloads = response.json()
            # Convert API format to frontend format
            workloads = []
            for w in api_workloads:
                workloads.append({
                    "workload_id": w["id"],
                    "cpu_required": w["cpu_required"], 
                    "memory_required_gb": w["memory_required_gb"]
                })
            st.session_state.workloads = workloads
            st.success(f"✅ Loaded {len(workloads)} sample workloads from API!")
        else:
            # Fallback to hardcoded sample data
            raise Exception("API not available")
            
    except Exception:
        # Fallback to hardcoded sample workloads
        sample_workloads = [
            {"workload_id": 201, "cpu_required": 2, "memory_required_gb": 4},
            {"workload_id": 202, "cpu_required": 1, "memory_required_gb": 2},
            {"workload_id": 203, "cpu_required": 4, "memory_required_gb": 8},
            {"workload_id": 204, "cpu_required": 2, "memory_required_gb": 2},
            {"workload_id": 205, "cpu_required": 3, "memory_required_gb": 6},
            {"workload_id": 206, "cpu_required": 1, "memory_required_gb": 4},
            {"workload_id": 207, "cpu_required": 3, "memory_required_gb": 6},
            {"workload_id": 208, "cpu_required": 2, "memory_required_gb": 8},
        ]
        st.session_state.workloads = sample_workloads
        st.success(f"✅ Loaded {len(sample_workloads)} sample workloads (offline mode)!")
    
    st.rerun()

def upload_workloads(uploaded_file):
    """Upload workloads via API"""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/api/workloads/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            # Convert API response to match session state format
            workloads = []
            for w in data["workloads"]:
                workloads.append({
                    "workload_id": w["id"],  # API uses 'id', frontend uses 'workload_id'
                    "cpu_required": w["cpu_required"],
                    "memory_required_gb": w["memory_required_gb"]
                })
            st.session_state.workloads = workloads
            st.success(f"Uploaded {data['count']} workloads successfully!")
            st.rerun()
        else:
            st.error(f"Upload failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error uploading workloads: {e}")

def generate_random_workloads(count, cpu_min, cpu_max, memory_min, memory_max):
    """Generate random workloads via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/workloads/generate",
            params={"count": count}  # Note: API doesn't support min/max params yet
        )
        
        if response.status_code == 200:
            data = response.json()
            # Convert API response to frontend format
            workloads = []
            for w in data:
                workloads.append({
                    "workload_id": w["id"],  # Convert from API format
                    "cpu_required": w["cpu_required"],
                    "memory_required_gb": w["memory_required_gb"]
                })
            st.session_state.workloads = workloads
            st.success(f"Generated {len(workloads)} random workloads!")
            st.rerun()
        else:
            st.error(f"Generation failed: {response.text}")
    
    except Exception as e:
        st.error(f"Error generating workloads: {e}")

def run_simulation(schedulers):
    """Run simulation via API"""
    try:
        # Use individual simulation calls since compare endpoint structure changed
        results = {}
        
        for scheduler in schedulers:
            # Convert workloads to API format (using 'id' instead of 'workload_id')
            api_workloads = []
            for w in st.session_state.workloads:
                api_workloads.append({
                    "id": w["workload_id"],  # Convert to API format
                    "cpu_required": w["cpu_required"],
                    "memory_required_gb": w["memory_required_gb"]
                })
            
            simulation_request = {
                "scheduler_type": scheduler,
                "workloads": api_workloads
            }
            
            with st.spinner(f"Running {scheduler} simulation..."):
                response = requests.post(f"{API_BASE_URL}/api/simulation/run", json=simulation_request)
            
            if response.status_code == 200:
                data = response.json()
                results[scheduler] = {
                    "summary": {
                        "success_rate": data["summary"]["success_rate"],
                        "successful_workloads": data["summary"]["successful_assignments"],
                        "total_workloads": data["summary"]["total_workloads"],
                        "total_cost": data["summary"]["total_cost"],
                        # Add mock values for charts (since API doesn't provide these)
                        "final_cpu_usage": min(100, data["summary"]["success_rate"] * 1.2),
                        "final_memory_usage": min(100, data["summary"]["success_rate"] * 0.8)
                    },
                    "logs": data["logs"]
                }
            else:
                st.error(f"Simulation failed for {scheduler}: {response.text}")
                return
        
        st.session_state.simulation_results = results
        st.success("All simulations completed successfully!")
        st.balloons()
    
    except Exception as e:
        st.error(f"Error running simulation: {e}")

def run_quick_simulation():
    """Run simulation with default settings"""
    run_simulation(["random", "lowest_cost", "round_robin"])

if __name__ == "__main__":
    main()