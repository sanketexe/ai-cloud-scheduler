# Cloud Intelligence Platform Dashboard

A comprehensive web-based dashboard for real-time multi-cloud intelligence, cost optimization, and performance monitoring.

## Features

### Core Dashboard Infrastructure (Subtask 5.1) ✅

- **Responsive Web Interface**: Modern, mobile-friendly dashboard built with Streamlit
- **Real-time Data Visualization**: Interactive charts and graphs with live data updates
- **Unified Navigation**: Seamless navigation between workload, cost, and performance views
- **Multi-tab Interface**: Organized views for Overview, Workloads, Costs, Performance, and Alerts
- **Auto-refresh Capability**: Configurable automatic data refresh with manual override

### Real-time Visualizations

- **System Overview**: Key metrics dashboard with health indicators
- **Workload Status**: Real-time workload distribution and status tracking
- **Cost Analytics**: Interactive cost trends and provider comparisons
- **Performance Monitoring**: Resource utilization gauges and performance heatmaps
- **Alert Management**: Alert timeline and severity tracking

### Interactive Components

- **Time Range Filters**: Flexible time range selection (1 hour to 1 week)
- **Provider Filters**: Multi-select cloud provider filtering
- **Resource Filters**: Filter by resource types (Compute, Storage, Network, etc.)
- **Advanced Filters**: Custom cost and utilization range filters
- **Drill-down Charts**: Interactive charts with detailed views

## Architecture

```
dashboard/
├── app.py                 # Main dashboard application
├── config.py             # Configuration settings
├── api_client.py         # API client for backend communication
├── run_dashboard.py      # Startup script
├── components/           # Dashboard components
│   ├── layout.py        # Layout managers and UI components
│   └── charts.py        # Chart components and visualizations
└── README.md            # This file
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install streamlit plotly pandas requests
   ```

2. **Set Environment Variables** (optional):
   ```bash
   export API_BASE_URL="http://localhost:8000"
   export REFRESH_INTERVAL="30"
   ```

## Usage

### Quick Start

1. **Start the Dashboard**:
   ```bash
   python dashboard/run_dashboard.py
   ```

2. **Access the Dashboard**:
   Open your browser to `http://localhost:8501`

### Advanced Usage

```bash
# Custom port and host
python dashboard/run_dashboard.py --port 8502 --host 0.0.0.0

# Custom API URL
python dashboard/run_dashboard.py --api-url http://api.example.com:8000
```

### Using Streamlit Directly

```bash
cd dashboard
streamlit run app.py --server.port 8501
```

## Configuration

### Environment Variables

- `API_BASE_URL`: Backend API URL (default: http://localhost:8000)
- `API_TIMEOUT`: API request timeout in seconds (default: 30)
- `REFRESH_INTERVAL`: Default auto-refresh interval in seconds (default: 30)

### Dashboard Settings

The dashboard can be configured through `config.py`:

- **Time Range Options**: Customize available time ranges
- **Color Themes**: Modify chart colors and themes
- **Performance Thresholds**: Set warning and critical thresholds
- **Feature Flags**: Enable/disable dashboard features

## API Integration

The dashboard connects to the Cloud Intelligence Platform backend API:

### Required Endpoints

- `GET /api/system/status` - System health and status
- `GET /api/workloads/metrics` - Workload metrics and statistics
- `GET /api/costs/metrics` - Cost data and financial metrics
- `GET /api/performance/metrics` - Performance and resource metrics
- `GET /api/alerts` - Alert and notification data
- `GET /api/providers/comparison` - Cloud provider comparison
- `GET /api/resources/utilization` - Resource utilization data

### Mock Data

The dashboard includes mock data generation for development and testing when the backend API is not available.

## Dashboard Views

### 📊 Overview Tab

- **Key Metrics**: Active workloads, total costs, CPU utilization, system health
- **Real-time Charts**: Workload status, cost trends, performance heatmaps
- **Resource Gauges**: CPU, Memory, Storage, Network utilization
- **Provider Comparison**: Multi-cloud performance comparison

### ⚙️ Workloads Tab

- **Workload Summary**: Total scheduled, success rate, active count
- **Recent Workloads**: Detailed workload table with status and costs
- **Provider Distribution**: Workload distribution across cloud providers

### 💰 Costs Tab

- **Cost Summary**: Total costs with trend indicators
- **Cost Breakdown**: Interactive charts by provider and service
- **Cost Timeline**: Historical cost trends with forecasting

### 🔧 Performance Tab

- **Performance Summary**: Average CPU, memory utilization
- **Interactive Metrics**: Selectable performance metric charts
- **Performance Heatmap**: Resource performance across infrastructure

### 🚨 Alerts Tab

- **Alert Summary**: Active, critical, and warning alert counts
- **Alert Timeline**: Visual timeline of alert occurrences
- **Alert Details**: Detailed alert table with severity color coding

## Customization

### Adding New Charts

1. Create chart function in `components/charts.py`
2. Add chart to appropriate tab in `app.py`
3. Update API client if new data endpoints needed

### Custom Themes

Modify the `ThemeManager.apply_custom_css()` function in `components/layout.py` to customize styling.

### New Filters

Add filter functions to `FilterManager` class in `components/layout.py`.

## Development

### Mock Data

The dashboard includes comprehensive mock data for development:

- Realistic workload patterns
- Time-series cost data
- Performance metrics with trends
- Alert data with various severities

### Testing

```bash
# Test dashboard components
python -m pytest dashboard/tests/

# Test API client
python -c "from dashboard.api_client import DashboardAPIClient; client = DashboardAPIClient(); print(client.get_system_status())"
```

## Troubleshooting

### Common Issues

1. **Dashboard won't start**:
   - Check Python version (3.8+ required)
   - Verify Streamlit installation: `pip install streamlit`
   - Check port availability

2. **No data displayed**:
   - Verify backend API is running
   - Check API_BASE_URL configuration
   - Review browser console for errors

3. **Charts not loading**:
   - Verify Plotly installation: `pip install plotly`
   - Check browser JavaScript console
   - Clear browser cache

### Performance Optimization

- Adjust refresh intervals for better performance
- Use time range filters to limit data volume
- Enable browser caching for static assets

## Next Steps

This completes **Subtask 5.1: Develop core dashboard infrastructure**. The next subtasks will add:

- **5.2**: Interactive analytics and filtering capabilities
- **5.3**: Comprehensive reporting system
- **5.4**: Alert management interface
- **5.5**: Dashboard and reporting tests

## Requirements Satisfied

✅ **Requirement 4.1**: Real-time workload, cost, and performance visibility  
✅ **Requirement 4.2**: Interactive analytics with filtering and custom views  
✅ Modern responsive web dashboard using Streamlit framework  
✅ Real-time data visualization with Plotly charts and graphs  
✅ Unified navigation and layout for all views  
✅ Auto-refresh functionality with configurable intervals  
✅ Comprehensive mock data for development and testing