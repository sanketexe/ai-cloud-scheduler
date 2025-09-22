import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

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

def main():
    st.title("☁️ AI-Powered Cloud Scheduler")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Configuration", "Simulation", "Results", "ML Predictions"]
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
        st.metric("Schedulers Available", "3", "Random, Cost, Round Robin")
    
    with col3:
        workload_count = len(st.session_state.workloads)
        st.metric("Loaded Workloads", workload_count)
    
    with col4:
        if st.session_state.simulation_results:
            st.metric("Last Simulation", "Completed", "✅")
        else:
            st.metric("Last Simulation", "None", "❌")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Load Sample Data", use_container_width=True):
            load_sample_workloads()
    
    with col2:
        if st.button("⚙️ Configure Simulation", use_container_width=True):
            st.info("💡 Use the sidebar navigation to go to Configuration page")
    
    with col3:
        if st.button("▶️ Quick Simulation", use_container_width=True):
            if st.session_state.workloads:
                run_quick_simulation()
            else:
                st.error("Please load workloads first!")

def show_configuration():
    st.header("⚙️ Configuration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Cloud Providers", "Virtual Machines", "Workloads", "System Config"])
    
    with tab1:
        show_provider_config()
    
    with tab2:
        show_vm_config()
    
    with tab3:
        show_workload_config()
    
    with tab4:
        show_system_config()

def show_provider_config():
    st.subheader("☁️ Cloud Provider Configuration")
    
    # Get default providers - Updated API response structure
    try:
        response = requests.get(f"{API_BASE_URL}/api/providers/default")
        if response.status_code == 200:
            providers = response.json()  # Remove ["providers"] - now returns list directly
            
            st.session_state.providers = []
            
            for i, provider in enumerate(providers):
                with st.expander(f"{provider['name']} Configuration"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        name = st.text_input(
                            "Provider Name", 
                            value=provider['name'], 
                            key=f"provider_name_{i}"
                        )
                    
                    with col2:
                        cpu_cost = st.number_input(
                            "CPU Cost/hour ($)", 
                            value=provider['cpu_cost'], 
                            step=0.001,
                            key=f"cpu_cost_{i}"
                        )
                    
                    with col3:
                        memory_cost = st.number_input(
                            "Memory Cost/GB/hour ($)", 
                            value=provider['memory_cost_gb'], 
                            step=0.001,
                            key=f"memory_cost_{i}"
                        )
                    
                    st.session_state.providers.append({
                        "name": name,
                        "cpu_cost": cpu_cost,
                        "memory_cost_gb": memory_cost
                    })
        
    except Exception as e:
        st.error(f"Error loading providers: {e}")

def show_vm_config():
    st.subheader("🖥️ Virtual Machine Configuration")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/vms/default")
        if response.status_code == 200:
            vms = response.json()  # Remove ["vms"] - now returns list directly
            
            st.session_state.vms = []
            
            for i, vm in enumerate(vms):
                with st.expander(f"VM {vm['vm_id']} Configuration"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        vm_id = st.number_input(
                            "VM ID", 
                            value=vm['vm_id'], 
                            key=f"vm_id_{i}"
                        )
                    
                    with col2:
                        cpu_capacity = st.number_input(
                            "CPU Cores", 
                            value=vm['cpu_capacity'], 
                            min_value=1,
                            key=f"cpu_capacity_{i}"
                        )
                    
                    with col3:
                        memory_capacity = st.number_input(
                            "Memory (GB)", 
                            value=vm['memory_capacity_gb'], 
                            min_value=1,
                            key=f"memory_capacity_{i}"
                        )
                    
                    with col4:
                        # Extract provider name from nested structure
                        provider_name = vm['provider']['name']
                        provider_name = st.selectbox(
                            "Provider", 
                            ["AWS", "GCP", "Azure"],
                            index=["AWS", "GCP", "Azure"].index(provider_name),
                            key=f"provider_{i}"
                        )
                    
                    st.session_state.vms.append({
                        "vm_id": int(vm_id),
                        "cpu_capacity": int(cpu_capacity),
                        "memory_capacity_gb": int(memory_capacity),
                        "provider_name": provider_name
                    })
    
    except Exception as e:
        st.error(f"Error loading VMs: {e}")

def show_workload_config():
    st.subheader("📋 Workload Configuration")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV", "Manual Entry", "Generate Random", "Use Sample Data"]
    )
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should contain: workload_id, cpu_required, memory_required_gb"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Workloads"):
                upload_workloads(uploaded_file)
    
    elif input_method == "Manual Entry":
        show_manual_workload_entry()
    
    elif input_method == "Generate Random":
        show_random_workload_generator()
    
    elif input_method == "Use Sample Data":
        if st.button("Load Sample Workloads"):
            load_sample_workloads()
    
    # Display current workloads
    if st.session_state.workloads:
        st.subheader("Current Workloads")
        df = pd.DataFrame(st.session_state.workloads)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Clear All Workloads"):
            st.session_state.workloads = []
            st.rerun()

def show_manual_workload_entry():
    with st.form("manual_workload"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            workload_id = st.number_input("Workload ID", min_value=1, value=len(st.session_state.workloads) + 1)
        
        with col2:
            cpu_required = st.number_input("CPU Required", min_value=1, max_value=32, value=2)
        
        with col3:
            memory_required = st.number_input("Memory Required (GB)", min_value=1, max_value=128, value=4)
        
        if st.form_submit_button("Add Workload"):
            st.session_state.workloads.append({
                "workload_id": int(workload_id),
                "cpu_required": int(cpu_required),
                "memory_required_gb": int(memory_required)
            })
            st.success("Workload added!")
            st.rerun()

def show_random_workload_generator():
    with st.form("random_workloads"):
        col1, col2 = st.columns(2)
        
        with col1:
            count = st.number_input("Number of Workloads", min_value=1, max_value=100, value=10)
            cpu_min = st.number_input("Min CPU", min_value=1, max_value=16, value=1)
            memory_min = st.number_input("Min Memory (GB)", min_value=1, max_value=64, value=2)
        
        with col2:
            st.write("")  # Spacing
            cpu_max = st.number_input("Max CPU", min_value=cpu_min, max_value=32, value=8)
            memory_max = st.number_input("Max Memory (GB)", min_value=memory_min, max_value=128, value=32)
        
        if st.form_submit_button("Generate Random Workloads"):
            generate_random_workloads(count, cpu_min, cpu_max, memory_min, memory_max)

def show_system_config():
    """Show system configuration management"""
    st.subheader("🔧 System Configuration")
    
    # Get configuration overview
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/show")
        if response.status_code == 200:
            config_data = response.json()
            
            # Display overview
            overview = config_data["system_overview"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Categories", overview["total_categories"])
            with col2:
                st.metric("Status", overview["status"])
            with col3:
                st.metric("Last Updated", "Just now")
            
            # Display categories
            st.subheader("Configuration Categories")
            
            for category, info in config_data["categories"].items():
                with st.expander(f"📁 {category.upper()} ({info['total_settings']} settings)"):
                    st.write(f"**Status:** {info['status']}")
                    st.write("**Key Settings:**")
                    for key, value in info["key_settings"].items():
                        st.write(f"- {key}: `{value}`")
                    
                    if st.button(f"Edit {category}", key=f"edit_{category}"):
                        st.info(f"Edit functionality for {category} coming soon!")
        
        else:
            st.error("Failed to load system configuration")
    
    except Exception as e:
        st.error(f"Error loading system configuration: {e}")

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
    # Performance comparison chart
    st.subheader("Performance Comparison")
    
    scheduler_names = []
    success_rates = []
    total_costs = []
    successful_workloads = []
    
    for scheduler, data in results.items():
        summary = data['summary']
        scheduler_names.append(scheduler.replace('_', ' ').title())
        success_rates.append(summary['success_rate'])
        total_costs.append(summary.get('total_cost', 0))
        successful_workloads.append(summary['successful_workloads'])
    
    # Create comparison charts
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Success Rate (%)', 'Total Cost ($)', 'Successful Workloads'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
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