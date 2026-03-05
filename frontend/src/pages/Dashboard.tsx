import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Box, LinearProgress, Alert, Chip, Button } from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  AttachMoney,
  Savings,
  Warning,
  AccountBalance,
  TrendingDown,
  CloudOff,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import toast from 'react-hot-toast';
import apiService from '../services/api';
import numeral from 'numeral';
import { useNavigate } from 'react-router-dom';

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
  const [costTrendData, setCostTrendData] = useState<any[]>([]);
  const [serviceBreakdown, setServiceBreakdown] = useState<any[]>([]);
  const [budgetData, setBudgetData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [noAws, setNoAws] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadFinopsData();
  }, []);

  const loadFinopsData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/dashboard');
      const data = await response.json();

      if (data.error === 'no_aws_account') {
        setNoAws(true);
        setLoading(false);
        return;
      }

      setFinopsData(data.finops_summary);
      setCostTrendData(data.cost_trend || []);
      setServiceBreakdown(data.service_breakdown || []);

      // Build budget data from budgets endpoint
      try {
        const budgetResponse = await fetch('http://localhost:8000/api/budgets');
        const budgetResult = await budgetResponse.json();
        if (budgetResult.budgets) {
          setBudgetData(budgetResult.budgets.map((b: any) => ({
            name: b.name,
            budget: b.amount,
            spent: b.spent,
            utilization: b.utilization,
          })));
        }
      } catch (e) {
        console.warn('Could not load budgets:', e);
      }

      setLoading(false);
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
          Loading FinOps data from AWS...
        </Typography>
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
                      <Area type="monotone" dataKey="cost" stroke="#2196f3" fill="rgba(33,150,243,0.2)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                    No cost data available yet. Cost data will appear after AWS processes your usage.
                  </Typography>
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
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                    No service data available yet.
                  </Typography>
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