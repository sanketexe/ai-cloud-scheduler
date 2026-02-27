import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Legend,
  ReferenceLine,
} from 'recharts';
import { format, addDays } from 'date-fns';
import { anomalyApiService, ForecastRequest, ForecastResponse } from '../services/anomalyApi';

interface ForecastVisualizationProps {
  accountId: string;
  onForecastGenerated?: (forecast: ForecastResponse) => void;
}

const ForecastVisualization: React.FC<ForecastVisualizationProps> = ({
  accountId,
  onForecastGenerated
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Forecast parameters
  const [horizonDays, setHorizonDays] = useState(30);
  const [confidenceLevel, setConfidenceLevel] = useState(0.8);
  const [granularity, setGranularity] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [includeSeasonality, setIncludeSeasonality] = useState(true);
  const [selectedServices, setSelectedServices] = useState<string[]>([]);

  const availableServices = ['EC2', 'S3', 'RDS', 'Lambda', 'CloudWatch', 'EBS', 'ELB'];
  const horizonOptions = [7, 14, 30, 60, 90];
  const confidenceOptions = [0.8, 0.85, 0.9, 0.95];

  useEffect(() => {
    generateForecast();
  }, [accountId]);

  const generateForecast = async () => {
    setLoading(true);
    setError(null);

    try {
      const request: ForecastRequest = {
        account_id: accountId,
        forecast_horizon_days: horizonDays,
        confidence_level: confidenceLevel,
        include_seasonality: includeSeasonality,
        services: selectedServices.length > 0 ? selectedServices : undefined,
        granularity
      };

      const response = await anomalyApiService.generateForecast(request);
      setForecast(response);
      onForecastGenerated?.(response);
    } catch (err: any) {
      setError('Failed to generate forecast: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const prepareForecastChartData = () => {
    if (!forecast) return [];

    return forecast.forecasts.map((item, index) => {
      const date = addDays(new Date(), index + 1);
      
      return {
        date: format(date, granularity === 'daily' ? 'MMM dd' : granularity === 'weekly' ? 'MMM dd' : 'MMM yyyy'),
        forecast: item.predicted_cost,
        upper: forecast.confidence_intervals.upper_bound[index],
        lower: forecast.confidence_intervals.lower_bound[index],
        trend: item.factors.trend,
        seasonality: item.factors.seasonality,
        baseline: item.factors.baseline,
      };
    });
  };

  const calculateForecastSummary = () => {
    if (!forecast) return null;

    const totalForecast = forecast.forecasts.reduce((sum, f) => sum + f.predicted_cost, 0);
    const avgDaily = totalForecast / forecast.forecasts.length;
    const firstWeek = forecast.forecasts.slice(0, 7).reduce((sum, f) => sum + f.predicted_cost, 0);
    const lastWeek = forecast.forecasts.slice(-7).reduce((sum, f) => sum + f.predicted_cost, 0);
    const trendDirection = lastWeek > firstWeek ? 'increasing' : 'decreasing';
    const trendPercentage = ((lastWeek - firstWeek) / firstWeek) * 100;

    return {
      totalForecast,
      avgDaily,
      trendDirection,
      trendPercentage,
      accuracy: forecast.metadata.forecast_accuracy_estimate,
      modelVersion: forecast.metadata.model_version
    };
  };

  const summary = calculateForecastSummary();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getTrendColor = (direction: string) => {
    return direction === 'increasing' ? 'error' : 'success';
  };

  const getTrendIcon = (direction: string) => {
    return direction === 'increasing' ? <TrendingUpIcon /> : <TrendingDownIcon />;
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6">
              Cost Forecast Analysis
            </Typography>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
              onClick={generateForecast}
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Generate Forecast'}
            </Button>
          </Box>

          {/* Forecast Parameters */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Horizon</InputLabel>
                <Select
                  value={horizonDays}
                  label="Horizon"
                  onChange={(e) => setHorizonDays(e.target.value as number)}
                >
                  {horizonOptions.map((days) => (
                    <MenuItem key={days} value={days}>
                      {days} days
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Confidence</InputLabel>
                <Select
                  value={confidenceLevel}
                  label="Confidence"
                  onChange={(e) => setConfidenceLevel(e.target.value as number)}
                >
                  {confidenceOptions.map((level) => (
                    <MenuItem key={level} value={level}>
                      {(level * 100).toFixed(0)}%
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Granularity</InputLabel>
                <Select
                  value={granularity}
                  label="Granularity"
                  onChange={(e) => setGranularity(e.target.value as 'daily' | 'weekly' | 'monthly')}
                >
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="monthly">Monthly</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Services</InputLabel>
                <Select
                  multiple
                  value={selectedServices}
                  label="Services"
                  onChange={(e) => setSelectedServices(e.target.value as string[])}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {availableServices.map((service) => (
                    <MenuItem key={service} value={service}>
                      {service}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          {/* Summary Cards */}
          {summary && (
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      ${summary.totalForecast.toFixed(0)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Total Forecast ({horizonDays} days)
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="secondary">
                      ${summary.avgDaily.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Daily Average
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                      {getTrendIcon(summary.trendDirection)}
                      <Typography variant="h4" color={`${getTrendColor(summary.trendDirection)}.main`} sx={{ ml: 1 }}>
                        {Math.abs(summary.trendPercentage).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary">
                      Trend ({summary.trendDirection})
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="success.main">
                      {(summary.accuracy * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Model Accuracy
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Tabs */}
          <Tabs value={activeTab} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab label="Forecast Chart" />
            <Tab label="Confidence Intervals" />
            <Tab label="Trend Analysis" />
            <Tab label="Model Details" />
          </Tabs>

          <Box sx={{ mt: 3 }}>
            {/* Forecast Chart Tab */}
            {activeTab === 0 && forecast && (
              <Box>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={prepareForecastChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={(value) => `$${value.toFixed(0)}`} />
                    <RechartsTooltip 
                      formatter={(value: number, name: string) => [`$${value.toFixed(2)}`, name]}
                      labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="forecast" 
                      stroke="#1976d2" 
                      strokeWidth={3}
                      name="Forecast"
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            )}

            {/* Confidence Intervals Tab */}
            {activeTab === 1 && forecast && (
              <Box>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={prepareForecastChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={(value) => `$${value.toFixed(0)}`} />
                    <RechartsTooltip 
                      formatter={(value: number, name: string) => [`$${value.toFixed(2)}`, name]}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="upper" 
                      stackId="1" 
                      stroke="none" 
                      fill="#e3f2fd" 
                      name="Upper Bound"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="lower" 
                      stackId="1" 
                      stroke="none" 
                      fill="#ffffff" 
                      name="Lower Bound"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="forecast" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      name="Forecast"
                    />
                  </AreaChart>
                </ResponsiveContainer>
                
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    The shaded area represents the {(confidenceLevel * 100).toFixed(0)}% confidence interval. 
                    Actual costs are expected to fall within this range with {(confidenceLevel * 100).toFixed(0)}% probability.
                  </Typography>
                </Alert>
              </Box>
            )}

            {/* Trend Analysis Tab */}
            {activeTab === 2 && forecast && (
              <Box>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={prepareForecastChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip 
                      formatter={(value: number, name: string) => [value.toFixed(2), name]}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="trend" 
                      stroke="#ff9800" 
                      strokeWidth={2}
                      name="Trend Component"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="seasonality" 
                      stroke="#4caf50" 
                      strokeWidth={2}
                      name="Seasonal Component"
                    />
                    <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                  </LineChart>
                </ResponsiveContainer>

                <Grid container spacing={2} sx={{ mt: 2 }}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Trend Analysis
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          {getTrendIcon(summary?.trendDirection || 'stable')}
                        </ListItemIcon>
                        <ListItemText
                          primary={`${summary?.trendDirection.toUpperCase()} trend detected`}
                          secondary={`${Math.abs(summary?.trendPercentage || 0).toFixed(1)}% change over forecast period`}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Seasonal Patterns
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <InfoIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary={includeSeasonality ? "Seasonality included" : "Seasonality excluded"}
                          secondary={includeSeasonality ? "Historical seasonal patterns are factored into the forecast" : "Forecast based on trend only"}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Model Details Tab */}
            {activeTab === 3 && forecast && (
              <Box>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Model Information
                    </Typography>
                    <List>
                      <ListItem>
                        <ListItemText
                          primary="Model Version"
                          secondary={forecast.metadata.model_version}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Forecast Accuracy"
                          secondary={`${(forecast.metadata.forecast_accuracy_estimate * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Generated At"
                          secondary={new Date(forecast.metadata.forecast_generated_at).toLocaleString()}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Services Included"
                          secondary={forecast.metadata.services_included.join(', ')}
                        />
                      </ListItem>
                    </List>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Forecast Parameters
                    </Typography>
                    <List>
                      <ListItem>
                        <ListItemText
                          primary="Horizon"
                          secondary={`${forecast.metadata.horizon_days} days`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Granularity"
                          secondary={forecast.metadata.granularity}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Seasonality"
                          secondary={forecast.metadata.seasonality_included ? 'Included' : 'Excluded'}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Confidence Level"
                          secondary={`${(forecast.confidence_intervals.confidence_level * 100).toFixed(0)}%`}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 2 }} />

                <Alert severity="info">
                  <Typography variant="body2">
                    This forecast is generated using advanced machine learning models that analyze historical cost patterns, 
                    seasonal trends, and usage patterns. The accuracy estimate is based on backtesting against historical data.
                  </Typography>
                </Alert>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ForecastVisualization;