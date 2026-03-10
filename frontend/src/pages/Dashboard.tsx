import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Box, LinearProgress, Alert, Chip, Button, IconButton, Tooltip } from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  AttachMoney,
  Savings,
  Warning,
  AccountBalance,
  TrendingDown,
  CloudOff,
  Refresh,
  Info,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, Legend } from 'recharts';
import toast from 'react-hot-toast';
import apiService from '../services/api';
import numeral from 'numeral';
import { useNavigate } from 'react-router-dom';
import { SkeletonLoader } from '../components/Loading';
import { executeParallel } from '../utils/parallelApiExecutor';
import { getPrefetchedData } from '../utils/dataPrefetcher';
import { usePrefetch } from '../hooks/usePrefetch';
import { performanceMonitor } from '../utils/performanceMonitor';
import {
  TOOLTIP_STYLE,
  AXIS_STYLE,
  GRID_STYLE,
  formatCurrency,
  formatCurrencyCompact,
  CurrencyTooltip,
  PieChartCurrencyTooltip,
  LEGEND_CONFIG,
  CustomPieLabel,
  HOVER_CONFIG,
} from '../utils/chartConfig';

const StatCard: React.FC<{
  title: string;
  value: string;
  change: string;
  icon: React.ReactNode;
  color: string;
}> = React.memo(({ title, value, change, icon, color }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
  >
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
              {title}
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
              {value}
            </Typography>
            <Typography variant="body2" sx={{ color: color }}>
              {change}
            </Typography>
          </Box>
          <Box sx={{ color: color, opacity: 0.8 }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  </motion.div>
));

const Dashboard: React.FC = () => {
  const [finopsData, setFinopsData] = useState<any>(null);
  const [costTrendData, setCostTrendData] = useState<any[]>([]);
  const [serviceBreakdown, setServiceBreakdown] = useState<any[]>([]);
  const [budgetData, setBudgetData] = useState<any[]>([]);
  const [accountId, setAccountId] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [noAws, setNoAws] = useState(false);
  const navigate = useNavigate();
  
  // Enable automatic prefetching for likely next pages
  usePrefetch();

  useEffect(() => {
    performanceMonitor.start('Dashboard-Load');
    loadFinopsData();
  }, []);

  const loadFinopsData = async () => {
    try {
      const isRefresh = !loading;
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      
      // Check for prefetched data first (only on initial load)
      if (!isRefresh) {
        const prefetchedDashboard = getPrefetchedData('dashboard-data');
        const prefetchedBudgets = getPrefetchedData('budgets');
        
        if (prefetchedDashboard && prefetchedBudgets) {
          console.log('[Dashboard] Using prefetched data');
          
          if (prefetchedDashboard.error === 'no_aws_account') {
            setNoAws(true);
            setLoading(false);
            return;
          }
          
          setAccountId(prefetchedDashboard.account_id || '');
          setFinopsData(prefetchedDashboard.finops_summary);
          setCostTrendData(prefetchedDashboard.cost_trend || []);
          setServiceBreakdown(prefetchedDashboard.service_breakdown || []);
          
          if (prefetchedBudgets.budgets) {
            setBudgetData(prefetchedBudgets.budgets.map((b: any) => ({
              name: b.name,
              budget: b.amount,
              spent: b.spent,
              utilization: b.utilization,
            })));
          }
          
          setLoading(false);
          setLastRefresh(new Date());
          return;
        }
      }
      
      // Execute API calls in parallel for faster loading
      const apiCalls = [
        {
          name: 'dashboard',
          fn: async () => {
            const response = await fetch('http://localhost:8000/api/dashboard');
            return response.json();
          },
          priority: 10,
        },
        {
          name: 'budgets',
          fn: async () => {
            const response = await fetch('http://localhost:8000/api/budgets');
            return response.json();
          },
          priority: 8,
        },
      ];
      
      const results = await executeParallel(apiCalls);
      
      // Process dashboard data
      const dashboardResult = results.find(r => r.name === 'dashboard');
      if (dashboardResult?.success && dashboardResult.data) {
        const data = dashboardResult.data;
        
        if (data.error === 'no_aws_account') {
          setNoAws(true);
          setLoading(false);
          setRefreshing(false);
          return;
        }

        setAccountId(data.account_id || '');
        setFinopsData(data.finops_summary);
        setCostTrendData(data.cost_trend || []);
        setServiceBreakdown(data.service_breakdown || []);
      }
      
      // Process budget data
      const budgetResult = results.find(r => r.name === 'budgets');
      if (budgetResult?.success && budgetResult.data?.budgets) {
        setBudgetData(budgetResult.data.budgets.map((b: any) => ({
          name: b.name,
          budget: b.amount,
          spent: b.spent,
          utilization: b.utilization,
        })));
      }

      setLoading(false);
      setRefreshing(false);
      setLastRefresh(new Date());
      performanceMonitor.end('Dashboard-Load');
      
      if (isRefresh) {
        toast.success('Dashboard data refreshed');
      }
    } catch (error) {
      console.error('Error loading FinOps data:', error);
      toast.error('Failed to load FinOps data');
      setLoading(false);
      setRefreshing(false);
      performanceMonitor.end('Dashboard-Load');
    }
  };

  const handleRefresh = () => {
    loadFinopsData();
  };

  if (loading) {
    return (
      <Box>
        <SkeletonLoader variant="dashboard" />
      </Box>
    );
  }

  if (noAws) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <CloudOff sx={{ fontSize: 80, color: 'text.secondary', mb: 3 }} />
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>
          No AWS Account Connected
        </Typography>
        <Typography variant="body1" sx={{ mb: 4, color: 'text.secondary' }}>
          Connect your AWS account to see real cost data, optimization recommendations, and more.
        </Typography>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/onboarding')}
          sx={{ px: 4, py: 1.5 }}
        >
          Connect AWS Account
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 4 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
            CloudPilot Dashboard
          </Typography>
          {accountId && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={<AccountBalance />}
                label={`AWS Account: ${accountId}`}
                size="small"
                sx={{ 
                  backgroundColor: 'rgba(33, 150, 243, 0.1)',
                  color: '#2196f3',
                  fontWeight: 600,
                }}
              />
              {lastRefresh && (
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Last updated: {lastRefresh.toLocaleTimeString()}
                </Typography>
              )}
            </Box>
          )}
        </Box>
        <Tooltip title="Refresh dashboard data">
          <IconButton 
            onClick={handleRefresh} 
            disabled={refreshing}
            sx={{ 
              backgroundColor: 'rgba(33, 150, 243, 0.1)',
              '&:hover': { backgroundColor: 'rgba(33, 150, 243, 0.2)' }
            }}
          >
            <Refresh sx={{ 
              animation: refreshing ? 'spin 1s linear infinite' : 'none',
              '@keyframes spin': {
                '0%': { transform: 'rotate(0deg)' },
                '100%': { transform: 'rotate(360deg)' },
              }
            }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* FinOps Status Alert */}
      <Alert
        severity={finopsData?.budgetUtilization > 95 ? 'error' : finopsData?.budgetUtilization > 85 ? 'warning' : 'success'}
        sx={{ mb: 4 }}
      >
        Budget Status: {finopsData?.budgetUtilization > 95 ? 'Critical - Budget exceeded' :
          finopsData?.budgetUtilization > 85 ? 'Warning - Approaching budget limit' :
            'On track - Within budget'}
        {` (${finopsData?.budgetUtilization}% utilized)`}
      </Alert>

      {/* FinOps KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Monthly Cost"
            value={numeral(finopsData?.totalMonthlyCost).format('$0,0')}
            change={`${finopsData?.budgetUtilization}% of budget`}
            icon={<AttachMoney sx={{ fontSize: 40 }} />}
            color={finopsData?.budgetUtilization > 90 ? "#f44336" : "#4caf50"}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Potential Savings"
            value={numeral(finopsData?.monthlySavings).format('$0,0')}
            change="identified opportunities"
            icon={<Savings sx={{ fontSize: 40 }} />}
            color="#4caf50"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Waste Detected"
            value={`${finopsData?.wastePercentage}%`}
            change={`${finopsData?.optimizationOpportunities} opportunities`}
            icon={<Warning sx={{ fontSize: 40 }} />}
            color={finopsData?.wastePercentage > 20 ? "#f44336" : finopsData?.wastePercentage > 10 ? "#ff9800" : "#4caf50"}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Cost Anomalies"
            value={finopsData?.anomaliesCount?.toString() || '0'}
            change="require attention"
            icon={<TrendingUp sx={{ fontSize: 40 }} />}
            color={finopsData?.anomaliesCount > 5 ? "#f44336" : finopsData?.anomaliesCount > 2 ? "#ff9800" : "#4caf50"}
          />
        </Grid>
      </Grid>

      {/* FinOps Charts */}
      <Grid container spacing={3}>
        {/* Cost Trend Analysis */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Cost Trend (Last 30 Days)
                </Typography>
                {costTrendData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={costTrendData}>
                      <CartesianGrid {...GRID_STYLE} />
                      <XAxis 
                        dataKey="date" 
                        {...AXIS_STYLE}
                        label={{ value: 'Date', position: 'insideBottom', offset: -5, style: { fill: '#b0bec5' } }}
                      />
                      <YAxis 
                        {...AXIS_STYLE}
                        tickFormatter={formatCurrencyCompact}
                        label={{ value: 'Cost', angle: -90, position: 'insideLeft', style: { fill: '#b0bec5' } }}
                      />
                      <ChartTooltip content={<CurrencyTooltip />} />
                      <Legend {...LEGEND_CONFIG} />
                      <Area 
                        type="monotone" 
                        dataKey="cost" 
                        name="Daily Cost"
                        stroke="#2196f3" 
                        fill="rgba(33,150,243,0.2)" 
                        strokeWidth={3}
                        {...HOVER_CONFIG}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Info sx={{ fontSize: 48, color: 'text.secondary', mb: 2, opacity: 0.5 }} />
                    <Typography variant="body1" sx={{ color: 'text.secondary', mb: 1, fontWeight: 600 }}>
                      No cost data available yet
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 400, mx: 'auto' }}>
                      AWS Cost Explorer data typically appears 24 hours after your first AWS usage. 
                      Check back tomorrow to see your cost trends.
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Service Cost Breakdown */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Cost by Service
                </Typography>
                {serviceBreakdown.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={serviceBreakdown}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                          label={CustomPieLabel}
                        >
                          {serviceBreakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <ChartTooltip content={<PieChartCurrencyTooltip />} />
                        <Legend {...LEGEND_CONFIG} />
                      </PieChart>
                    </ResponsiveContainer>
                    <Box sx={{ mt: 2 }}>
                      {serviceBreakdown.slice(0, 6).map((service) => (
                        <Box key={service.name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              backgroundColor: service.color,
                              mr: 1,
                            }}
                          />
                          <Typography variant="body2" sx={{ flexGrow: 1 }}>
                            {service.name}
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {numeral(service.cost).format('$0,0')}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Info sx={{ fontSize: 48, color: 'text.secondary', mb: 2, opacity: 0.5 }} />
                    <Typography variant="body1" sx={{ color: 'text.secondary', mb: 1, fontWeight: 600 }}>
                      No service data available yet
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 350, mx: 'auto' }}>
                      Service breakdown will appear once AWS Cost Explorer processes your usage data (typically within 24 hours).
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Budget Status by Team */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Budget Utilization
                </Typography>
                {budgetData.length > 0 ? (
                  <Grid container spacing={3}>
                    {budgetData.map((team) => (
                      <Grid item xs={12} md={3} key={team.name}>
                        <Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {team.name}
                            </Typography>
                            <Chip
                              label={`${team.utilization}%`}
                              size="small"
                              sx={{
                                backgroundColor: team.utilization > 90 ? 'rgba(244, 67, 54, 0.2)' :
                                  team.utilization > 75 ? 'rgba(255, 152, 0, 0.2)' :
                                    'rgba(76, 175, 80, 0.2)',
                                color: team.utilization > 90 ? '#f44336' :
                                  team.utilization > 75 ? '#ff9800' :
                                    '#4caf50',
                              }}
                            />
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={team.utilization}
                            sx={{
                              height: 8,
                              borderRadius: 4,
                              backgroundColor: 'rgba(255,255,255,0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: team.utilization > 90 ? '#f44336' :
                                  team.utilization > 75 ? '#ff9800' :
                                    '#4caf50',
                              },
                            }}
                          />
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            {numeral(team.spent).format('$0,0')} / {numeral(team.budget).format('$0,0')}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                ) : (
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                    No AWS Budgets configured. Configure budgets in AWS to see utilization data.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;