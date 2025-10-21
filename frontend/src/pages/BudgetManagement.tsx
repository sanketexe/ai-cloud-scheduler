import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  LinearProgress,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
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
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Add,
  Edit,
  Delete,
  Warning,
  CheckCircle,
  Error,
  Notifications,
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
} from 'recharts';
import numeral from 'numeral';

// Mock data
const budgets = [
  {
    id: 1,
    name: 'Engineering Team Q1 2024',
    amount: 50000,
    spent: 42500,
    utilization: 85,
    period: 'quarterly',
    status: 'warning',
    team: 'Engineering',
    alerts: [
      { threshold: 75, triggered: true },
      { threshold: 90, triggered: false },
    ],
  },
  {
    id: 2,
    name: 'Data Science Monthly',
    amount: 15000,
    spent: 11200,
    utilization: 75,
    period: 'monthly',
    status: 'good',
    team: 'Data Science',
    alerts: [
      { threshold: 80, triggered: false },
      { threshold: 95, triggered: false },
    ],
  },
  {
    id: 3,
    name: 'DevOps Infrastructure',
    amount: 25000,
    spent: 24100,
    utilization: 96,
    period: 'monthly',
    status: 'critical',
    team: 'DevOps',
    alerts: [
      { threshold: 85, triggered: true },
      { threshold: 95, triggered: true },
    ],
  },
  {
    id: 4,
    name: 'QA Environment',
    amount: 8000,
    spent: 5100,
    utilization: 64,
    period: 'monthly',
    status: 'good',
    team: 'QA',
    alerts: [
      { threshold: 75, triggered: false },
      { threshold: 90, triggered: false },
    ],
  },
];

const budgetTrendData = [
  { month: 'Oct', planned: 45000, actual: 42000, forecast: 44000 },
  { month: 'Nov', planned: 47000, actual: 45500, forecast: 46000 },
  { month: 'Dec', planned: 50000, actual: 48200, forecast: 49000 },
  { month: 'Jan', planned: 52000, actual: 50100, forecast: 51000 },
  { month: 'Feb', planned: 48000, actual: 46800, forecast: 47500 },
  { month: 'Mar', planned: 51000, actual: 0, forecast: 49500 },
];

const BudgetManagement: React.FC = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<any>(null);
  const [newBudget, setNewBudget] = useState({
    name: '',
    amount: '',
    period: 'monthly',
    team: '',
    alerts: [75, 90],
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'critical':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good':
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
      case 'warning':
        return <Warning sx={{ color: '#ff9800', fontSize: 20 }} />;
      case 'critical':
        return <Error sx={{ color: '#f44336', fontSize: 20 }} />;
      default:
        return <CheckCircle sx={{ color: '#9e9e9e', fontSize: 20 }} />;
    }
  };

  const handleCreateBudget = () => {
    // Implement budget creation
    console.log('Creating budget:', newBudget);
    setCreateDialogOpen(false);
    setNewBudget({
      name: '',
      amount: '',
      period: 'monthly',
      team: '',
      alerts: [75, 90],
    });
  };

  const handleEditBudget = (budget: any) => {
    setSelectedBudget(budget);
    setEditDialogOpen(true);
  };

  const handleDeleteBudget = (budgetId: number) => {
    // Implement budget deletion
    console.log('Deleting budget:', budgetId);
  };

  const totalBudget = budgets.reduce((sum, budget) => sum + budget.amount, 0);
  const totalSpent = budgets.reduce((sum, budget) => sum + budget.spent, 0);
  const overallUtilization = (totalSpent / totalBudget) * 100;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Budget Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Budget
        </Button>
      </Box>

      {/* Budget Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Total Budget
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {numeral(totalBudget).format('$0,0')}
                </Typography>
                <Chip
                  label={`${budgets.length} active budgets`}
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(33, 150, 243, 0.2)',
                    color: '#2196f3',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Total Spent
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {numeral(totalSpent).format('$0,0')}
                </Typography>
                <Chip
                  label={`${overallUtilization.toFixed(1)}% utilized`}
                  size="small"
                  sx={{
                    backgroundColor: overallUtilization > 90 ? 'rgba(244, 67, 54, 0.2)' : 'rgba(76, 175, 80, 0.2)',
                    color: overallUtilization > 90 ? '#f44336' : '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Remaining Budget
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {numeral(totalBudget - totalSpent).format('$0,0')}
                </Typography>
                <Chip
                  label={`${((totalBudget - totalSpent) / totalBudget * 100).toFixed(1)}% remaining`}
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    color: '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Active Alerts
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: '#ff9800' }}>
                  {budgets.filter(b => b.alerts.some(a => a.triggered)).length}
                </Typography>
                <Chip
                  label="Need attention"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    color: '#ff9800',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Budget Trend Chart */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Budget vs Actual Spending Trend
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={budgetTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="month" stroke="#b0bec5" />
                    <YAxis stroke="#b0bec5" tickFormatter={(value) => numeral(value).format('$0a')} />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any) => [numeral(value).format('$0,0'), '']}
                    />
                    <Line type="monotone" dataKey="planned" stroke="#9e9e9e" strokeDasharray="5 5" name="Planned" />
                    <Line type="monotone" dataKey="actual" stroke="#2196f3" strokeWidth={3} name="Actual" />
                    <Line type="monotone" dataKey="forecast" stroke="#ff9800" strokeWidth={2} name="Forecast" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Budget List */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Budget Details
                </Typography>
                <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Budget Name</TableCell>
                        <TableCell>Team</TableCell>
                        <TableCell align="right">Amount</TableCell>
                        <TableCell align="right">Spent</TableCell>
                        <TableCell align="center">Utilization</TableCell>
                        <TableCell align="center">Status</TableCell>
                        <TableCell align="center">Alerts</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {budgets.map((budget) => (
                        <TableRow key={budget.id}>
                          <TableCell component="th" scope="row">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {budget.name}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {budget.period}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={budget.team}
                              size="small"
                              sx={{
                                backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                color: '#2196f3',
                              }}
                            />
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {numeral(budget.amount).format('$0,0')}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {numeral(budget.spent).format('$0,0')}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={budget.utilization}
                                sx={{
                                  height: 8,
                                  borderRadius: 4,
                                  backgroundColor: 'rgba(255,255,255,0.1)',
                                  '& .MuiLinearProgress-bar': {
                                    backgroundColor: getStatusColor(budget.status),
                                  },
                                }}
                              />
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                {budget.utilization}%
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            {getStatusIcon(budget.status)}
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
                              {budget.alerts.map((alert, index) => (
                                <Chip
                                  key={index}
                                  label={`${alert.threshold}%`}
                                  size="small"
                                  sx={{
                                    backgroundColor: alert.triggered ? 'rgba(244, 67, 54, 0.2)' : 'rgba(158, 158, 158, 0.2)',
                                    color: alert.triggered ? '#f44336' : '#9e9e9e',
                                    fontSize: '0.7rem',
                                  }}
                                />
                              ))}
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                              <Tooltip title="Edit Budget">
                                <IconButton size="small" onClick={() => handleEditBudget(budget)}>
                                  <Edit sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Configure Alerts">
                                <IconButton size="small">
                                  <Notifications sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete Budget">
                                <IconButton size="small" onClick={() => handleDeleteBudget(budget.id)}>
                                  <Delete sx={{ fontSize: 16, color: '#f44336' }} />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Create Budget Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Budget</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Budget Name"
                value={newBudget.name}
                onChange={(e) => setNewBudget({ ...newBudget, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Amount"
                type="number"
                value={newBudget.amount}
                onChange={(e) => setNewBudget({ ...newBudget, amount: e.target.value })}
                InputProps={{
                  startAdornment: '$',
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Period</InputLabel>
                <Select
                  value={newBudget.period}
                  label="Period"
                  onChange={(e) => setNewBudget({ ...newBudget, period: e.target.value })}
                >
                  <MenuItem value="monthly">Monthly</MenuItem>
                  <MenuItem value="quarterly">Quarterly</MenuItem>
                  <MenuItem value="annual">Annual</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Team"
                value={newBudget.team}
                onChange={(e) => setNewBudget({ ...newBudget, team: e.target.value })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateBudget} variant="contained">
            Create Budget
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BudgetManagement;