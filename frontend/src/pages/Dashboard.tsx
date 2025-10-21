import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Box, LinearProgress, Alert, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  AttachMoney,
  Savings,
  Warning,
  AccountBalance,
  TrendingDown,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import toast from 'react-hot-toast';
import apiService from '../services/api';
import numeral from 'numeral';

// FinOps Mock Data
const costTrendData = [
  { date: '2024-01-01', cost: 45230, budget: 50000, forecast: 48500 },
  { date: '2024-01-02', cost: 46100, budget: 50000, forecast: 48200 },
  { date: '2024-01-03', cost: 44800, budget: 50000, forecast: 47900 },
  { date: '2024-01-04', cost: 47200, budget: 50000, forecast: 48100 },
  { date: '2024-01-05', cost: 45900, budget: 50000, forecast: 47800 },
  { date: '2024-01-06', cost: 46500, budget: 50000, forecast: 48000 },
  { date: '2024-01-07', cost: 45100, budget: 50000, forecast: 47600 },
];

const serviceBreakdown = [
  { name: 'Compute (EC2)', value: 35, cost: 15830, color: '#ff9800' },
  { name: 'Storage (S3)', value: 25, cost: 11308, color: '#4caf50' },
  { name: 'Database (RDS)', value: 20, cost: 9046, color: '#2196f3' },
  { name: 'Network', value: 12, cost: 5428, color: '#9c27b0' },
  { name: 'Other', value: 8, cost: 3618, color: '#607d8b' },
];

const budgetData = [
  { name: 'Engineering', budget: 25000, spent: 21500, utilization: 86 },
  { name: 'Data Science', budget: 15000, spent: 11200, utilization: 75 },
  { name: 'DevOps', budget: 10000, spent: 9800, utilization: 98 },
  { name: 'QA', budget: 5000, spent: 3200, utilization: 64 },
];

const StatCard: React.FC<{
  title: string;
  value: string;
  change: string;
  icon: React.ReactNode;
  color: string;
}> = ({ title, value, change, icon, color }) => (
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
);

const Dashboard: React.FC = () => {
  const [finopsData, setFinopsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadFinopsData();
  }, []);

  const loadFinopsData = async () => {
    try {
      setLoading(true);
      
      // In a real implementation, these would be actual API calls
      // For now, we'll simulate the data
      setTimeout(() => {
        setFinopsData({
          totalMonthlyCost: 45230,
          monthlyBudget: 50000,
          monthlySavings: 8750,
          wastePercentage: 15.2,
          budgetUtilization: 90.5,
          costTrend: 'decreasing',
          anomaliesCount: 3,
          optimizationOpportunities: 12
        });
        setLoading(false);
      }, 1000);
      
    } catch (error) {
      console.error('Error loading FinOps data:', error);
      toast.error('Failed to load FinOps data');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>
          FinOps Dashboard
        </Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>
          Loading FinOps data...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>
        FinOps Platform Dashboard
      </Typography>

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
            title="Monthly Savings"
            value={numeral(finopsData?.monthlySavings).format('$0,0')}
            change="vs previous month"
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
            value={finopsData?.anomaliesCount.toString()}
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
                  Cost Trend vs Budget
                </Typography>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={costTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="date" stroke="#b0bec5" />
                    <YAxis stroke="#b0bec5" tickFormatter={(value) => numeral(value).format('$0a')} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any) => [numeral(value).format('$0,0'), '']}
                    />
                    <Area type="monotone" dataKey="budget" stroke="#9e9e9e" fill="rgba(158,158,158,0.1)" strokeDasharray="5 5" />
                    <Area type="monotone" dataKey="forecast" stroke="#ff9800" fill="rgba(255,152,0,0.1)" />
                    <Area type="monotone" dataKey="cost" stroke="#2196f3" fill="rgba(33,150,243,0.2)" strokeWidth={3} />
                  </AreaChart>
                </ResponsiveContainer>
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
                    >
                      {serviceBreakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any, name: any, props: any) => [
                        `${value}% (${numeral(props.payload.cost).format('$0,0')})`,
                        props.payload.name
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <Box sx={{ mt: 2 }}>
                  {serviceBreakdown.map((service) => (
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
                  Budget Utilization by Team
                </Typography>
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
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;