import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
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
  Switch,
  FormControlLabel,
  Alert,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Add,
  Edit,
  Delete,
  NotificationsActive,
  NotificationsOff,
  Warning,
  Error,
  Info,
  CheckCircle,
} from '@mui/icons-material';
import numeral from 'numeral';

// Mock data
const alerts = [
  {
    id: 1,
    name: 'Engineering Budget Alert',
    type: 'budget_threshold',
    condition: 'Budget utilization > 80%',
    threshold: 80,
    currentValue: 85,
    status: 'triggered',
    severity: 'warning',
    lastTriggered: '2024-01-15T14:30:00Z',
    channels: ['email', 'slack'],
    enabled: true,
    team: 'Engineering',
  },
  {
    id: 2,
    name: 'Cost Anomaly Detection',
    type: 'cost_anomaly',
    condition: 'Daily cost increase > 25%',
    threshold: 25,
    currentValue: 32,
    status: 'triggered',
    severity: 'critical',
    lastTriggered: '2024-01-15T09:15:00Z',
    channels: ['email', 'slack', 'pagerduty'],
    enabled: true,
    team: 'FinOps',
  },
  {
    id: 3,
    name: 'Unused Resources Alert',
    type: 'waste_detection',
    condition: 'Unused resources > $500/month',
    threshold: 500,
    currentValue: 750,
    status: 'triggered',
    severity: 'info',
    lastTriggered: '2024-01-14T16:45:00Z',
    channels: ['email'],
    enabled: true,
    team: 'DevOps',
  },
  {
    id: 4,
    name: 'Tagging Compliance',
    type: 'compliance',
    condition: 'Tagging compliance < 90%',
    threshold: 90,
    currentValue: 87,
    status: 'triggered',
    severity: 'warning',
    lastTriggered: '2024-01-13T11:20:00Z',
    channels: ['email'],
    enabled: false,
    team: 'Platform',
  },
  {
    id: 5,
    name: 'Monthly Forecast Alert',
    type: 'forecast',
    condition: 'Forecasted monthly cost > budget',
    threshold: 100,
    currentValue: 95,
    status: 'normal',
    severity: 'info',
    lastTriggered: null,
    channels: ['email'],
    enabled: true,
    team: 'Finance',
  },
];

const alertTypes = [
  { value: 'budget_threshold', label: 'Budget Threshold' },
  { value: 'cost_anomaly', label: 'Cost Anomaly' },
  { value: 'waste_detection', label: 'Waste Detection' },
  { value: 'compliance', label: 'Compliance' },
  { value: 'forecast', label: 'Forecast Alert' },
];

const Alerts: React.FC = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newAlert, setNewAlert] = useState({
    name: '',
    type: '',
    condition: '',
    threshold: '',
    channels: [],
    team: '',
  });

  const handleCreateAlert = () => {
    console.log('Creating alert:', newAlert);
    setCreateDialogOpen(false);
    setNewAlert({
      name: '',
      type: '',
      condition: '',
      threshold: '',
      channels: [],
      team: '',
    });
  };

  const handleToggleAlert = (alertId: number) => {
    console.log('Toggling alert:', alertId);
  };

  const handleDeleteAlert = (alertId: number) => {
    console.log('Deleting alert:', alertId);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <Error sx={{ color: '#f44336', fontSize: 20 }} />;
      case 'warning':
        return <Warning sx={{ color: '#ff9800', fontSize: 20 }} />;
      case 'info':
        return <Info sx={{ color: '#2196f3', fontSize: 20 }} />;
      default:
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#f44336';
      case 'warning':
        return '#ff9800';
      case 'info':
        return '#2196f3';
      default:
        return '#4caf50';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'triggered':
        return '#f44336';
      case 'normal':
        return '#4caf50';
      default:
        return '#9e9e9e';
    }
  };

  const triggeredAlerts = alerts.filter(alert => alert.status === 'triggered').length;
  const enabledAlerts = alerts.filter(alert => alert.enabled).length;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Alerts & Notifications
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Alert
        </Button>
      </Box>

      {/* Active Alerts Banner */}
      {triggeredAlerts > 0 && (
        <Alert severity="warning" sx={{ mb: 4 }}>
          <Typography variant="body2">
            You have {triggeredAlerts} active alert{triggeredAlerts > 1 ? 's' : ''} that require attention.
          </Typography>
        </Alert>
      )}

      {/* Alert Statistics */}
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
                  Active Alerts
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: '#f44336' }}>
                  {triggeredAlerts}
                </Typography>
                <Chip
                  label="Triggered"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(244, 67, 54, 0.2)',
                    color: '#f44336',
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
                  Total Alerts
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {alerts.length}
                </Typography>
                <Chip
                  label="Configured"
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
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Enabled Alerts
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {enabledAlerts}
                </Typography>
                <Chip
                  label={`${((enabledAlerts / alerts.length) * 100).toFixed(0)}% active`}
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
                  Notifications Sent
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  47
                </Typography>
                <Chip
                  label="This week"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(156, 39, 176, 0.2)',
                    color: '#9c27b0',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Alerts Table */}
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
                  Alert Configuration
                </Typography>
                <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Alert Name</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Condition</TableCell>
                        <TableCell align="center">Current Value</TableCell>
                        <TableCell align="center">Severity</TableCell>
                        <TableCell align="center">Status</TableCell>
                        <TableCell align="center">Channels</TableCell>
                        <TableCell align="center">Enabled</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {alerts.map((alert) => (
                        <TableRow key={alert.id}>
                          <TableCell component="th" scope="row">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {alert.name}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {alert.team}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={alert.type.replace('_', ' ')}
                              size="small"
                              sx={{
                                backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                color: '#2196f3',
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {alert.condition}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontWeight: 600,
                                color: alert.status === 'triggered' ? '#f44336' : 'text.primary'
                              }}
                            >
                              {alert.currentValue}
                              {alert.type.includes('threshold') || alert.type.includes('compliance') ? '%' : 
                               alert.type.includes('cost') || alert.type.includes('waste') ? '' : ''}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              Threshold: {alert.threshold}
                              {alert.type.includes('threshold') || alert.type.includes('compliance') ? '%' : ''}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            {getSeverityIcon(alert.severity)}
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={alert.status}
                              size="small"
                              sx={{
                                backgroundColor: `${getStatusColor(alert.status)}20`,
                                color: getStatusColor(alert.status),
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                              {alert.channels.map((channel) => (
                                <Chip
                                  key={channel}
                                  label={channel}
                                  size="small"
                                  sx={{
                                    backgroundColor: 'rgba(156, 39, 176, 0.2)',
                                    color: '#9c27b0',
                                    fontSize: '0.7rem',
                                  }}
                                />
                              ))}
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Switch
                              checked={alert.enabled}
                              onChange={() => handleToggleAlert(alert.id)}
                              color="primary"
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                              <Tooltip title="Edit Alert">
                                <IconButton size="small">
                                  <Edit sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Test Alert">
                                <IconButton size="small">
                                  <NotificationsActive sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete Alert">
                                <IconButton 
                                  size="small" 
                                  onClick={() => handleDeleteAlert(alert.id)}
                                >
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

      {/* Create Alert Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Alert</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Alert Name"
                value={newAlert.name}
                onChange={(e) => setNewAlert({ ...newAlert, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Alert Type</InputLabel>
                <Select
                  value={newAlert.type}
                  label="Alert Type"
                  onChange={(e) => setNewAlert({ ...newAlert, type: e.target.value })}
                >
                  {alertTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Threshold"
                type="number"
                value={newAlert.threshold}
                onChange={(e) => setNewAlert({ ...newAlert, threshold: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Condition Description"
                value={newAlert.condition}
                onChange={(e) => setNewAlert({ ...newAlert, condition: e.target.value })}
                placeholder="e.g., Budget utilization > 80%"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Team"
                value={newAlert.team}
                onChange={(e) => setNewAlert({ ...newAlert, team: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Notification Channels</InputLabel>
                <Select
                  multiple
                  value={newAlert.channels}
                  label="Notification Channels"
                  onChange={(e) => setNewAlert({ ...newAlert, channels: e.target.value as string[] })}
                >
                  <MenuItem value="email">Email</MenuItem>
                  <MenuItem value="slack">Slack</MenuItem>
                  <MenuItem value="teams">Microsoft Teams</MenuItem>
                  <MenuItem value="pagerduty">PagerDuty</MenuItem>
                  <MenuItem value="webhook">Webhook</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateAlert} variant="contained">
            Create Alert
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Alerts;