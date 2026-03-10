import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Refresh,
  FileDownload,
  CloudOff,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import numeral from 'numeral';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';
import { SkeletonLoader } from '../components/Loading';

const CostAnalysis: React.FC = () => {
  const [timeRange, setTimeRange] = useState('30');
  const [groupBy, setGroupBy] = useState('service');
  const [loading, setLoading] = useState(true);
  const [noAws, setNoAws] = useState(false);
  const [costAnalysisData, setCostAnalysisData] = useState<any[]>([]);
  const [serviceBreakdown, setServiceBreakdown] = useState<any[]>([]);
  const [accounts, setAccounts] = useState<any[]>([]);
  const [totalCost, setTotalCost] = useState(0);
  const [potentialSavings, setPotentialSavings] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    loadCostData();
  }, [timeRange]);

  const loadCostData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/cost-analysis');
      const data = await response.json();

      if (data.error === 'no_aws_account') {
        setNoAws(true);
        setLoading(false);
        return;
      }

      setCostAnalysisData(data.cost_trend || []);
      setAccounts(data.accounts || []);
      setTotalCost(data.total_cost || 0);
      setPotentialSavings(data.potential_savings || 0);

      // Load service breakdown from dashboard
      const dashResponse = await fetch('http://localhost:8000/api/dashboard');
      const dashData = await dashResponse.json();
      if (dashData.service_breakdown) {
        setServiceBreakdown(dashData.service_breakdown);
      }

      setLoading(false);
    } catch (error) {
      console.error('Error loading cost data:', error);
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    loadCostData();
  };

  const handleExport = () => {
    // Export cost data
    const csv = costAnalysisData.map(d => `${d.date},${d.cost},${d.savings}`).join('\n');
    const blob = new Blob([`Date,Cost,Savings\n${csv}`], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cost_analysis.csv';
    a.click();
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp sx={{ fontSize: 16, color: '#f44336' }} />;
      case 'down':
        return <TrendingDown sx={{ fontSize: 16, color: '#4caf50' }} />;
      default:
        return <TrendingFlat sx={{ fontSize: 16, color: '#ff9800' }} />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return '#f44336';
      case 'down':
        return '#4caf50';
      default:
        return '#ff9800';
    }
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>Cost Analysis</Typography>
        <SkeletonLoader variant="dashboard" />
      </Box>
    );
  }

  if (noAws) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <CloudOff sx={{ fontSize: 80, color: 'text.secondary', mb: 3 }} />
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>No AWS Account Connected</Typography>
        <Typography variant="body1" sx={{ mb: 4, color: 'text.secondary' }}>
          Connect your AWS account to see real cost analysis data.
        </Typography>
        <Button variant="contained" size="large" onClick={() => navigate('/onboarding')}>
          Connect AWS Account
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Cost Analysis
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select value={timeRange} label="Time Range" onChange={(e) => setTimeRange(e.target.value)}>
              <MenuItem value="7">7 Days</MenuItem>
              <MenuItem value="30">30 Days</MenuItem>
              <MenuItem value="90">90 Days</MenuItem>
            </Select>
          </FormControl>
          <Button variant="outlined" startIcon={<Refresh />} onClick={handleRefresh}>Refresh</Button>
          <Button variant="outlined" startIcon={<FileDownload />} onClick={handleExport}>Export</Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Total Cost ({timeRange} days)</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{numeral(totalCost).format('$0,0')}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Potential Savings</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#4caf50' }}>{numeral(potentialSavings).format('$0,0')}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Active Accounts</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{accounts.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Cost Trend Chart */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>Daily Cost Trend</Typography>
                {costAnalysisData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart data={costAnalysisData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="date" stroke="#b0bec5" />
                      <YAxis stroke="#b0bec5" tickFormatter={(value) => numeral(value).format('$0a')} />
                      <RechartsTooltip
                        contentStyle={{ backgroundColor: '#1a1d3a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                        formatter={(value: any) => [numeral(value).format('$0,0'), '']}
                      />
                      <Line type="monotone" dataKey="cost" stroke="#2196f3" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                    No cost trend data available.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} lg={4}>
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>Service Breakdown</Typography>
                {serviceBreakdown.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie data={serviceBreakdown} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={5} dataKey="value">
                          {serviceBreakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <RechartsTooltip
                          contentStyle={{ backgroundColor: '#1a1d3a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
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
                          <Box sx={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: service.color, mr: 1 }} />
                          <Typography variant="body2" sx={{ flexGrow: 1, fontSize: '0.8rem' }}>{service.name}</Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>{numeral(service.cost).format('$0,0')}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </>
                ) : (
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 8, color: 'text.secondary' }}>
                    No service data available.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Account Cost Table */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>Cost by Account</Typography>
                {accounts.length > 0 ? (
                  <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Account</TableCell>
                          <TableCell>Account ID</TableCell>
                          <TableCell align="right">Monthly Cost</TableCell>
                          <TableCell align="right">Potential Savings</TableCell>
                          <TableCell align="center">Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {accounts.map((account: any) => (
                          <TableRow key={account.id}>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>{account.name}</Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>{account.id}</Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>{numeral(account.monthly_cost).format('$0,0')}</Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" sx={{ color: '#4caf50' }}>{numeral(account.potential_savings).format('$0,0')}</Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Chip label={account.status} size="small" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }} />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                    No account data available.
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

export default CostAnalysis;