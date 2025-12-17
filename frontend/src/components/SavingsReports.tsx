import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Alert,
  Tabs,
  Tab,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Assessment as ReportIcon,
  Timeline as TimelineIcon,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { LocalizationProvider, DatePicker as MuiDatePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs, { Dayjs } from 'dayjs';
import { useQuery } from 'react-query';
import toast from 'react-hot-toast';

// Types
interface SavingsData {
  date: string;
  total_savings: number;
  ec2_savings: number;
  storage_savings: number;
  network_savings: number;
  actions_count: number;
}

interface ActionTypeSavings {
  action_type: string;
  total_savings: number;
  actions_count: number;
  avg_savings_per_action: number;
}

interface MonthlySummary {
  month: string;
  total_savings: number;
  total_actions: number;
  top_action_type: string;
  cost_reduction_percentage: number;
}

interface SavingsReport {
  summary: {
    total_lifetime_savings: number;
    monthly_savings: number;
    total_actions: number;
    avg_savings_per_action: number;
    cost_reduction_percentage: number;
  };
  daily_savings: SavingsData[];
  action_type_breakdown: ActionTypeSavings[];
  monthly_summaries: MonthlySummary[];
  trend_analysis: {
    savings_trend: 'increasing' | 'decreasing' | 'stable';
    trend_percentage: number;
    projection_next_month: number;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`savings-tabpanel-${index}`}
      aria-labelledby={`savings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const COLORS = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4'];

const SavingsReports: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [dateRange, setDateRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');
  const [startDate, setStartDate] = useState<Dayjs | null>(dayjs().subtract(30, 'day'));
  const [endDate, setEndDate] = useState<Dayjs | null>(dayjs());
  const [chartType, setChartType] = useState<'line' | 'area' | 'bar'>('area');

  // Fetch savings report data
  const { data: report, isLoading, refetch } = useQuery<SavingsReport>(
    ['savings-report', dateRange, startDate?.format('YYYY-MM-DD'), endDate?.format('YYYY-MM-DD')],
    async () => {
      const params = new URLSearchParams({
        range: dateRange,
        start_date: startDate?.format('YYYY-MM-DD') || '',
        end_date: endDate?.format('YYYY-MM-DD') || '',
      });
      const response = await fetch(`/api/automation/reports/savings?${params}`);
      if (!response.ok) throw new Error('Failed to fetch savings report');
      return response.json();
    },
    { refetchInterval: 300000 } // Refetch every 5 minutes
  );

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleDateRangeChange = (range: '7d' | '30d' | '90d' | '1y') => {
    setDateRange(range);
    const now = dayjs();
    switch (range) {
      case '7d':
        setStartDate(now.subtract(7, 'day'));
        break;
      case '30d':
        setStartDate(now.subtract(30, 'day'));
        break;
      case '90d':
        setStartDate(now.subtract(90, 'day'));
        break;
      case '1y':
        setStartDate(now.subtract(1, 'year'));
        break;
    }
    setEndDate(now);
  };

  const handleExportReport = async () => {
    try {
      const params = new URLSearchParams({
        range: dateRange,
        start_date: startDate?.format('YYYY-MM-DD') || '',
        end_date: endDate?.format('YYYY-MM-DD') || '',
        format: 'csv',
      });
      const response = await fetch(`/api/automation/reports/savings/export?${params}`);
      if (!response.ok) throw new Error('Failed to export report');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `savings-report-${dayjs().format('YYYY-MM-DD')}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Report exported successfully');
    } catch (error) {
      toast.error('Failed to export report');
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const renderSavingsChart = () => {
    if (!report?.daily_savings) return null;

    const data = report.daily_savings.map(item => ({
      ...item,
      date: dayjs(item.date).format('MMM DD'),
    }));

    const ChartComponent = chartType === 'line' ? LineChart : chartType === 'area' ? AreaChart : BarChart;

    return (
      <ResponsiveContainer width="100%" height={400}>
        <ChartComponent data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis tickFormatter={(value) => `$${value}`} />
          <RechartsTooltip formatter={(value) => formatCurrency(value as number)} />
          <Legend />
          
          {chartType === 'line' && (
            <>
              <Line type="monotone" dataKey="total_savings" stroke="#2196f3" strokeWidth={2} name="Total Savings" />
              <Line type="monotone" dataKey="ec2_savings" stroke="#4caf50" strokeWidth={2} name="EC2 Savings" />
              <Line type="monotone" dataKey="storage_savings" stroke="#ff9800" strokeWidth={2} name="Storage Savings" />
              <Line type="monotone" dataKey="network_savings" stroke="#f44336" strokeWidth={2} name="Network Savings" />
            </>
          )}
          
          {chartType === 'area' && (
            <>
              <Area type="monotone" dataKey="total_savings" stackId="1" stroke="#2196f3" fill="#2196f3" fillOpacity={0.6} name="Total Savings" />
              <Area type="monotone" dataKey="ec2_savings" stackId="2" stroke="#4caf50" fill="#4caf50" fillOpacity={0.6} name="EC2 Savings" />
              <Area type="monotone" dataKey="storage_savings" stackId="2" stroke="#ff9800" fill="#ff9800" fillOpacity={0.6} name="Storage Savings" />
              <Area type="monotone" dataKey="network_savings" stackId="2" stroke="#f44336" fill="#f44336" fillOpacity={0.6} name="Network Savings" />
            </>
          )}
          
          {chartType === 'bar' && (
            <>
              <Bar dataKey="ec2_savings" fill="#4caf50" name="EC2 Savings" />
              <Bar dataKey="storage_savings" fill="#ff9800" name="Storage Savings" />
              <Bar dataKey="network_savings" fill="#f44336" name="Network Savings" />
            </>
          )}
        </ChartComponent>
      </ResponsiveContainer>
    );
  };

  const renderActionTypeChart = () => {
    if (!report?.action_type_breakdown) return null;

    const data = report.action_type_breakdown.map(item => ({
      name: item.action_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: item.total_savings,
      count: item.actions_count,
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
            outerRadius={120}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <RechartsTooltip formatter={(value) => formatCurrency(value as number)} />
        </PieChart>
      </ResponsiveContainer>
    );
  };

  if (isLoading) {
    return <Typography>Loading savings report...</Typography>;
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box>
        {/* Header */}
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
              Savings Reports & Analytics
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Track and analyze cost optimization savings over time
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={() => refetch()}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              startIcon={<DownloadIcon />}
              onClick={handleExportReport}
            >
              Export Report
            </Button>
          </Box>
        </Box>

        {/* Summary Cards */}
        {report?.summary && (
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <ReportIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Savings</Typography>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatCurrency(report.summary.total_lifetime_savings)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Lifetime optimization
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <TimelineIcon color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Monthly Savings</Typography>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatCurrency(report.summary.monthly_savings)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    {report.trend_analysis.savings_trend === 'increasing' ? (
                      <TrendingUpIcon color="success" sx={{ mr: 0.5 }} />
                    ) : (
                      <TrendingDownIcon color="error" sx={{ mr: 0.5 }} />
                    )}
                    <Typography variant="body2" color="text.secondary">
                      {formatPercentage(report.trend_analysis.trend_percentage)} vs last month
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <BarChartIcon color="info" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Actions</Typography>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>
                    {report.summary.total_actions.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Optimization actions
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <PieChartIcon color="warning" sx={{ mr: 1 }} />
                    <Typography variant="h6">Avg per Action</Typography>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>
                    {formatCurrency(report.summary.avg_savings_per_action)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Average savings
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <TrendingDownIcon color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Cost Reduction</Typography>
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatPercentage(report.summary.cost_reduction_percentage)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Overall reduction
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Trend Analysis Alert */}
        {report?.trend_analysis && (
          <Alert 
            severity={report.trend_analysis.savings_trend === 'increasing' ? 'success' : 'info'}
            sx={{ mb: 3 }}
          >
            <Typography variant="body2">
              <strong>Trend Analysis:</strong> Savings are {report.trend_analysis.savings_trend} by{' '}
              {formatPercentage(Math.abs(report.trend_analysis.trend_percentage))} compared to last month.
              Projected savings for next month: {formatCurrency(report.trend_analysis.projection_next_month)}
            </Typography>
          </Alert>
        )}

        {/* Controls */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Date Range</InputLabel>
                  <Select
                    value={dateRange}
                    onChange={(e) => handleDateRangeChange(e.target.value as any)}
                    label="Date Range"
                  >
                    <MenuItem value="7d">Last 7 Days</MenuItem>
                    <MenuItem value="30d">Last 30 Days</MenuItem>
                    <MenuItem value="90d">Last 90 Days</MenuItem>
                    <MenuItem value="1y">Last Year</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <MuiDatePicker
                  label="Start Date"
                  value={startDate}
                  onChange={setStartDate}
                  slotProps={{ textField: { size: 'small', fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <MuiDatePicker
                  label="End Date"
                  value={endDate}
                  onChange={setEndDate}
                  slotProps={{ textField: { size: 'small', fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Chart Type</InputLabel>
                  <Select
                    value={chartType}
                    onChange={(e) => setChartType(e.target.value as any)}
                    label="Chart Type"
                  >
                    <MenuItem value="area">Area Chart</MenuItem>
                    <MenuItem value="line">Line Chart</MenuItem>
                    <MenuItem value="bar">Bar Chart</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Charts and Tables */}
        <Card>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Savings Trend" />
              <Tab label="Action Breakdown" />
              <Tab label="Monthly Summary" />
              <Tab label="Detailed Data" />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <Box>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Savings Trend Over Time
              </Typography>
              {renderSavingsChart()}
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Savings by Action Type
                </Typography>
                {renderActionTypeChart()}
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Action Type Details
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Action Type</TableCell>
                        <TableCell align="right">Total Savings</TableCell>
                        <TableCell align="right">Actions Count</TableCell>
                        <TableCell align="right">Avg per Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {report?.action_type_breakdown?.map((item) => (
                        <TableRow key={item.action_type}>
                          <TableCell>
                            {item.action_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(item.total_savings)}
                          </TableCell>
                          <TableCell align="right">
                            {item.actions_count.toLocaleString()}
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(item.avg_savings_per_action)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" sx={{ mb: 3 }}>
              Monthly Summary
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Month</TableCell>
                    <TableCell align="right">Total Savings</TableCell>
                    <TableCell align="right">Total Actions</TableCell>
                    <TableCell>Top Action Type</TableCell>
                    <TableCell align="right">Cost Reduction %</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {report?.monthly_summaries?.map((summary) => (
                    <TableRow key={summary.month}>
                      <TableCell>{summary.month}</TableCell>
                      <TableCell align="right">
                        {formatCurrency(summary.total_savings)}
                      </TableCell>
                      <TableCell align="right">
                        {summary.total_actions.toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={summary.top_action_type.replace(/_/g, ' ')}
                          size="small"
                          color="primary"
                        />
                      </TableCell>
                      <TableCell align="right">
                        {formatPercentage(summary.cost_reduction_percentage)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <Typography variant="h6" sx={{ mb: 3 }}>
              Daily Savings Data
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Date</TableCell>
                    <TableCell align="right">Total Savings</TableCell>
                    <TableCell align="right">EC2 Savings</TableCell>
                    <TableCell align="right">Storage Savings</TableCell>
                    <TableCell align="right">Network Savings</TableCell>
                    <TableCell align="right">Actions Count</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {report?.daily_savings?.map((data) => (
                    <TableRow key={data.date}>
                      <TableCell>{dayjs(data.date).format('MMM DD, YYYY')}</TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.total_savings)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.ec2_savings)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.storage_savings)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.network_savings)}
                      </TableCell>
                      <TableCell align="right">
                        {data.actions_count}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>
        </Card>
      </Box>
    </LocalizationProvider>
  );
};

export default SavingsReports;