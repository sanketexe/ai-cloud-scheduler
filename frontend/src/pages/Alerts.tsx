import React, { useState, useEffect } from 'react';
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
  Alert,
  LinearProgress,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Add,
  Edit,
  Delete,
  NotificationsActive,
  Warning,
  Error as ErrorIcon,
  Info,
  CheckCircle,
  CloudOff,
} from '@mui/icons-material';
import numeral from 'numeral';
import { useNavigate } from 'react-router-dom';

const alertTypes = [
  { value: 'budget_threshold', label: 'Budget Threshold' },
  { value: 'cost_anomaly', label: 'Cost Anomaly' },
  { value: 'waste_detection', label: 'Waste Detection' },
  { value: 'compliance', label: 'Compliance' },
  { value: 'forecast', label: 'Forecast Alert' },
];

const Alerts: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [noAws, setNoAws] = useState(false);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newAlert, setNewAlert] = useState({
    name: '',
    type: '',
    condition: '',
    threshold: '',
    channels: [] as string[],
    team: '',
  });
  const navigate = useNavigate();

  useEffect(() => {
    loadAlerts();
  }, []);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/alerts');
      const data = await response.json();

      if (data.error === 'no_aws_account') {
        setNoAws(true);
        setLoading(false);
        return;
      }

      setAlerts(data.alerts || []);
      setLoading(false);
    } catch (error) {
      console.error('Error loading alerts:', error);
      setLoading(false);
    }
  };

  const handleCreateAlert = () => {
    console.log('Creating alert:', newAlert);
    setCreateDialogOpen(false);
    setNewAlert({ name: '', type: '', condition: '', threshold: '', channels: [], team: '' });
  };

  const handleToggleAlert = (alertId: number) => {
    console.log('Toggling alert:', alertId);
  };

  const handleDeleteAlert = (alertId: number) => {
    console.log('Deleting alert:', alertId);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': case 'high': return <ErrorIcon sx={{ color: '#f44336', fontSize: 20 }} />;
      case 'warning': return <Warning sx={{ color: '#ff9800', fontSize: 20 }} />;
      case 'info': return <Info sx={{ color: '#2196f3', fontSize: 20 }} />;
      default: return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'triggered': return '#f44336';
      case 'normal': return '#4caf50';
      default: return '#9e9e9e';
    }
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>Alerts & Notifications</Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2, textAlign: 'center' }}>Loading alerts from AWS...</Typography>
      </Box>
    );
  }

  if (noAws) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <CloudOff sx={{ fontSize: 80, color: 'text.secondary', mb: 3 }} />
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>No AWS Account Connected</Typography>
        <Typography variant="body1" sx={{ mb: 4, color: 'text.secondary' }}>
          Connect your AWS account to see cost anomaly alerts and budget notifications.
        </Typography>
        <Button variant="contained" size="large" onClick={() => navigate('/onboarding')}>
          Connect AWS Account
        </Button>
      </Box>
    );
  }

  const triggeredAlerts = alerts.filter(alert => alert.status === 'triggered').length;
  const enabledAlerts = alerts.filter(alert => alert.enabled).length;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>Alerts & Notifications</Typography>
        <Button variant="contained" startIcon={<Add />} onClick={() => setCreateDialogOpen(true)}>
          Create Alert
        </Button>
      </Box>

      {triggeredAlerts > 0 && (
        <Alert severity="warning" sx={{ mb: 4 }}>
          You have {triggeredAlerts} active alert{triggeredAlerts > 1 ? 's' : ''} that require attention.
        </Alert>
      )}

      {/* Alert Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <Card><CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Active Alerts</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: triggeredAlerts > 0 ? '#f44336' : '#4caf50' }}>
                {triggeredAlerts}
              </Typography>
              <Chip label="Triggered" size="small" sx={{ backgroundColor: 'rgba(244, 67, 54, 0.2)', color: '#f44336' }} />
            </CardContent></Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <Card><CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Total Alerts</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>{alerts.length}</Typography>
              <Chip label="Configured" size="small" sx={{ backgroundColor: 'rgba(33, 150, 243, 0.2)', color: '#2196f3' }} />
            </CardContent></Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
            <Card><CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Enabled Alerts</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>{enabledAlerts}</Typography>
              <Chip label={alerts.length > 0 ? `${((enabledAlerts / alerts.length) * 100).toFixed(0)}% active` : '0%'} size="small"
                sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }} />
            </CardContent></Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <Card><CardContent>
              <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>Alert Sources</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                {new Set(alerts.map(a => a.type)).size}
              </Typography>
              <Chip label="Types" size="small" sx={{ backgroundColor: 'rgba(156, 39, 176, 0.2)', color: '#9c27b0' }} />
            </CardContent></Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Alerts Table */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
            <Card><CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>Alert Configuration</Typography>
              {alerts.length > 0 ? (
                <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Alert Name</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Condition</TableCell>
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
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{alert.name}</Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>{alert.team}</Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={alert.type.replace('_', ' ')} size="small"
                              sx={{ backgroundColor: 'rgba(33, 150, 243, 0.2)', color: '#2196f3', textTransform: 'capitalize' }} />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">{alert.condition}</Typography>
                          </TableCell>
                          <TableCell align="center">{getSeverityIcon(alert.severity)}</TableCell>
                          <TableCell align="center">
                            <Chip label={alert.status} size="small"
                              sx={{ backgroundColor: `${getStatusColor(alert.status)}20`, color: getStatusColor(alert.status), textTransform: 'capitalize' }} />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                              {(alert.channels || []).map((channel: string) => (
                                <Chip key={channel} label={channel} size="small"
                                  sx={{ backgroundColor: 'rgba(156, 39, 176, 0.2)', color: '#9c27b0', fontSize: '0.7rem' }} />
                              ))}
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Switch checked={alert.enabled} onChange={() => handleToggleAlert(alert.id)} color="primary" />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                              <Tooltip title="Edit Alert"><IconButton size="small"><Edit sx={{ fontSize: 16 }} /></IconButton></Tooltip>
                              <Tooltip title="Test Alert"><IconButton size="small"><NotificationsActive sx={{ fontSize: 16 }} /></IconButton></Tooltip>
                              <Tooltip title="Delete Alert">
                                <IconButton size="small" onClick={() => handleDeleteAlert(alert.id)}>
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
              ) : (
                <Alert severity="success">
                  No active alerts — your AWS costs are within expected parameters!
                </Alert>
              )}
            </CardContent></Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Create Alert Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Alert</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField fullWidth label="Alert Name" value={newAlert.name}
                onChange={(e) => setNewAlert({ ...newAlert, name: e.target.value })} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Alert Type</InputLabel>
                <Select value={newAlert.type} label="Alert Type"
                  onChange={(e) => setNewAlert({ ...newAlert, type: e.target.value })}>
                  {alertTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>{type.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField fullWidth label="Threshold" type="number" value={newAlert.threshold}
                onChange={(e) => setNewAlert({ ...newAlert, threshold: e.target.value })} />
            </Grid>
            <Grid item xs={12}>
              <TextField fullWidth label="Condition Description" value={newAlert.condition}
                onChange={(e) => setNewAlert({ ...newAlert, condition: e.target.value })}
                placeholder="e.g., Budget utilization > 80%" />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField fullWidth label="Team" value={newAlert.team}
                onChange={(e) => setNewAlert({ ...newAlert, team: e.target.value })} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Notification Channels</InputLabel>
                <Select multiple value={newAlert.channels} label="Notification Channels"
                  onChange={(e) => setNewAlert({ ...newAlert, channels: e.target.value as string[] })}>
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
          <Button onClick={handleCreateAlert} variant="contained">Create Alert</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Alerts;