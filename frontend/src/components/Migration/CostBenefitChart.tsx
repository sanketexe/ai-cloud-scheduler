/**
 * Cost-Benefit Chart Component
 * 
 * Visualizes ROI, break-even analysis, and cost projections
 * for migration planning with interactive charts.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Alert,
  Tooltip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Schedule as TimelineIcon,
  AccountBalance as AccountBalanceIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
  ReferenceLine
} from 'recharts';

import { MigrationAnalysis, multiCloudApi } from '../../services/multiCloudApi';

interface CostBenefitChartProps {
  analysis: MigrationAnalysis;
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
      id={`cost-benefit-tabpanel-${index}`}
      aria-labelledby={`cost-benefit-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const CostBenefitChart: React.FC<CostBenefitChartProps> = ({ analysis }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  // Generate cost projection data
  const generateCostProjection = () => {
    const months = 36; // 3 years
    const data = [];
    
    const migrationCost = analysis.migration_cost;
    const monthlySavings = analysis.monthly_savings || 0;
    const breakEvenMonth = analysis.break_even_months || 12;
    
    let cumulativeCost = migrationCost;
    let cumulativeSavings = 0;
    
    for (let month = 0; month <= months; month++) {
      if (month === 0) {
        // Initial migration cost
        data.push({
          month: 0,
          cumulativeCost: migrationCost,
          cumulativeSavings: 0,
          netBenefit: -migrationCost,
          monthlySavings: 0
        });
      } else {
        cumulativeSavings += monthlySavings;
        const netBenefit = cumulativeSavings - migrationCost;
        
        data.push({
          month,
          cumulativeCost: migrationCost,
          cumulativeSavings,
          netBenefit,
          monthlySavings
        });
      }
    }
    
    return data;
  };

  // Generate cost breakdown data
  const generateCostBreakdown = () => {
    if (!analysis.cost_breakdown) return [];
    
    return Object.entries(analysis.cost_breakdown).map(([category, cost]) => ({
      category: category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      cost: Number(cost),
      percentage: (Number(cost) / analysis.migration_cost) * 100
    }));
  };

  // Generate ROI timeline data
  const generateROITimeline = () => {
    const data = [];
    const monthlySavings = analysis.monthly_savings || 0;
    const migrationCost = analysis.migration_cost;
    
    for (let year = 1; year <= 5; year++) {
      const totalSavings = monthlySavings * 12 * year;
      const roi = ((totalSavings - migrationCost) / migrationCost) * 100;
      
      data.push({
        year: `Year ${year}`,
        roi: Math.round(roi),
        totalSavings,
        netBenefit: totalSavings - migrationCost
      });
    }
    
    return data;
  };

  const costProjectionData = generateCostProjection();
  const costBreakdownData = generateCostBreakdown();
  const roiTimelineData = generateROITimeline();

  const formatCurrency = (value: number) => {
    return multiCloudApi.formatCurrency(value);
  };

  const getBreakEvenPoint = () => {
    return costProjectionData.find(point => point.netBenefit >= 0);
  };

  const breakEvenPoint = getBreakEvenPoint();

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0'];

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AccountBalanceIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Migration Cost</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                {formatCurrency(analysis.migration_cost)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                One-time investment
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Monthly Savings</Typography>
              </Box>
              <Typography variant="h4" color="success.main">
                {formatCurrency(analysis.monthly_savings || 0)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Recurring benefit
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TimelineIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Break-even</Typography>
              </Box>
              <Typography variant="h4" color="info.main">
                {analysis.break_even_months || 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Months to break-even
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">3-Year ROI</Typography>
              </Box>
              <Typography variant="h4" color="warning.main">
                {analysis.roi_percentage ? `${analysis.roi_percentage.toFixed(1)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Return on investment
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabbed Charts */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="cost-benefit analysis tabs">
            <Tab label="Break-even Analysis" />
            <Tab label="Cost Breakdown" />
            <Tab label="ROI Timeline" />
            <Tab label="Financial Summary" />
          </Tabs>
        </Box>

        {/* Break-even Analysis Tab */}
        <TabPanel value={activeTab} index={0}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Break-even Analysis
            </Typography>
            
            {breakEvenPoint && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Break-even point reached at month {breakEvenPoint.month} with net benefit of {formatCurrency(breakEvenPoint.netBenefit)}
              </Alert>
            )}
            
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={costProjectionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                <RechartsTooltip 
                  formatter={(value, name) => [formatCurrency(Number(value)), name]}
                  labelFormatter={(label) => `Month ${label}`}
                />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                <Area
                  type="monotone"
                  dataKey="netBenefit"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.3}
                  name="Net Benefit"
                />
                <Line
                  type="monotone"
                  dataKey="cumulativeSavings"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  name="Cumulative Savings"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        {/* Cost Breakdown Tab */}
        <TabPanel value={activeTab} index={1}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Migration Cost Breakdown
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={costBreakdownData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="cost"
                      label={({ category, percentage }) => `${category}: ${percentage.toFixed(1)}%`}
                    >
                      {costBreakdownData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip formatter={(value) => formatCurrency(Number(value))} />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Category</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">%</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {costBreakdownData.map((row, index) => (
                        <TableRow key={row.category}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box
                                sx={{
                                  width: 12,
                                  height: 12,
                                  backgroundColor: COLORS[index % COLORS.length],
                                  mr: 1,
                                  borderRadius: 1
                                }}
                              />
                              {row.category}
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(row.cost)}
                          </TableCell>
                          <TableCell align="right">
                            {row.percentage.toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </Box>
        </TabPanel>

        {/* ROI Timeline Tab */}
        <TabPanel value={activeTab} index={2}>
          <Box>
            <Typography variant="h6" gutterBottom>
              ROI Timeline (5-Year Projection)
            </Typography>
            
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={roiTimelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis tickFormatter={(value) => `${value}%`} />
                <RechartsTooltip 
                  formatter={(value, name) => {
                    if (name === 'roi') return [`${value}%`, 'ROI'];
                    return [formatCurrency(Number(value)), name];
                  }}
                />
                <Bar dataKey="roi" fill="#8884d8" name="ROI %" />
              </BarChart>
            </ResponsiveContainer>
            
            <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Year</TableCell>
                    <TableCell align="right">Total Savings</TableCell>
                    <TableCell align="right">Net Benefit</TableCell>
                    <TableCell align="right">ROI %</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {roiTimelineData.map((row) => (
                    <TableRow key={row.year}>
                      <TableCell>{row.year}</TableCell>
                      <TableCell align="right">
                        {formatCurrency(row.totalSavings)}
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                          {row.netBenefit >= 0 ? (
                            <TrendingUpIcon color="success" sx={{ mr: 0.5, fontSize: 16 }} />
                          ) : (
                            <TrendingDownIcon color="error" sx={{ mr: 0.5, fontSize: 16 }} />
                          )}
                          {formatCurrency(row.netBenefit)}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${row.roi}%`}
                          color={row.roi >= 0 ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </TabPanel>

        {/* Financial Summary Tab */}
        <TabPanel value={activeTab} index={3}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Financial Summary & Recommendations
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom color="primary">
                      Investment Analysis
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Initial Investment
                      </Typography>
                      <Typography variant="h5">
                        {formatCurrency(analysis.migration_cost)}
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Annual Savings
                      </Typography>
                      <Typography variant="h5" color="success.main">
                        {formatCurrency((analysis.monthly_savings || 0) * 12)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Payback Period
                      </Typography>
                      <Typography variant="h5" color="info.main">
                        {analysis.break_even_months || 'N/A'} months
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom color="success.main">
                      Business Impact
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        3-Year Net Benefit
                      </Typography>
                      <Typography variant="h5">
                        {formatCurrency(((analysis.monthly_savings || 0) * 36) - analysis.migration_cost)}
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        5-Year Net Benefit
                      </Typography>
                      <Typography variant="h5" color="success.main">
                        {formatCurrency(((analysis.monthly_savings || 0) * 60) - analysis.migration_cost)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Risk-Adjusted ROI
                      </Typography>
                      <Typography variant="h5" color="warning.main">
                        {analysis.roi_percentage ? `${(analysis.roi_percentage * 0.8).toFixed(1)}%` : 'N/A'}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            {/* Recommendations */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Financial Recommendations
              </Typography>
              {analysis.recommendations && analysis.recommendations.length > 0 ? (
                analysis.recommendations.map((recommendation, index) => (
                  <Alert key={index} severity="info" sx={{ mb: 1 }}>
                    {recommendation}
                  </Alert>
                ))
              ) : (
                <Alert severity="success">
                  Based on the financial analysis, this migration shows positive ROI and is recommended to proceed.
                </Alert>
              )}
            </Box>
          </Box>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default CostBenefitChart;