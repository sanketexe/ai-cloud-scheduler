import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Alert,
  Chip,
  LinearProgress,
  Badge,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Fab,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Notifications as NotificationsIcon,
  NotificationsActive as NotificationsActiveIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import toast from 'react-hot-toast';

// Types
interface CostAlert {
  alert_id: string;
  alert_type: string;
  severity: string;
  title: string;
  description: string;
  current_cost: number;
  threshold_cost?: number;
  percentage_change?: number;
  service_affected?: string;
  recommended_actions: string[];
  created_at: string;
  resolved: boolean;
}

interface DailySummary {
  date: string;
  total_cost: number;
  cost_change: number;
  cost_change_percentage: number;
  top_services: Array<{
    service: string;
    cost: number;
    percentage: number;
  }>;
  alerts_count: number;
  optimization_opportunities: number;
}

interface AlertSummary {
  total_active: number;
  by_severity: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
  by_type: {
    [key: string]: number;
  };
  most_recent?: CostAlert;
}

interface BudgetThreshold {
  name: string;
  monthly_budget: number;
  warning_threshold: number;
  critical_threshold: number;
  services: string[];
  enabled: boolean;
}

const AWSCostAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<CostAlert[]>([]);
  const [alertSummary, setAlertSummary] = useState<AlertSummary | null>(null);
  const [dailySummary, setDailySummary] = useState<DailySummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [setupDialogOpen, setSetupDialogOpen] = useState(false);
  const [budgetThresholds, setBudgetThresholds] = useState<BudgetThreshold[]>([
    {
      name: 'Monthly AWS Budget',
      monthly_budget: 1000,
      warning_threshold: 80,
      critical_threshold: 95,
      services: [],
      enabled: true
    }
  ]);
  const [notificationConfig, setNotificationConfig] = useState({
    email_enabled: false,
    email_config: {
      to_email: '',
      from_email: 'alerts@finops.com',
      smtp_server: 'smtp.gmail.com',
      smtp_port: 587,
      use_tls: true,
      username: '',
      password: ''
    },
    slack_enabled: false,
    slack_config: {
      webhook_url: ''
    }
  });

  const [credentials, setCredentials] = useState({
    aws_access_key_id: '',
    aws_secret_access_key: '',
    region: 'us-east-1'
  });

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'high':
        return <WarningIcon color="warning" />;
      case 'medium':
        return <InfoIcon color="info" />;
      case 'low':
        return <CheckCircleIcon color="success" />;
      default:
        return <InfoIcon />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              AWS Cost Alerts & Monitoring
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Proactive cost monitoring with real-time alerts and anomaly detection
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={<SettingsIcon />}
              onClick={() => setSetupDialogOpen(true)}
            >
              Setup Monitoring
            </Button>
            <Button
              variant="contained"
              startIcon={<RefreshIcon />}
              onClick={() => {/* Run monitoring */}}
              disabled={loading}
            >
              Run Check
            </Button>
          </Box>
        </Box>

        {/* Alert Summary Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Active Alerts
                    </Typography>
                    <Typography variant="h4">
                      {alertSummary?.total_active || 0}
                    </Typography>
                  </Box>
                  <Badge badgeContent={alertSummary?.by_severity.critical || 0} color="error">
                    <NotificationsActiveIcon fontSize="large" />
                  </Badge>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Today's Cost
                </Typography>
                <Typography variant="h4">
                  ${dailySummary?.total_cost.toFixed(2) || '0.00'}
                </Typography>
                <Typography 
                  variant="body2" 
                  color={dailySummary && dailySummary.cost_change >= 0 ? 'error.main' : 'success.main'}
                >
                  {dailySummary?.cost_change_percentage.toFixed(1) || '0.0'}% vs yesterday
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Critical Alerts
                </Typography>
                <Typography variant="h4" color="error.main">
                  {alertSummary?.by_severity.critical || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Require immediate attention
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Budget Status
                </Typography>
                <Typography variant="h4" color="warning.main">
                  85%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Of monthly budget used
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Active Alerts */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Active Alerts
          </Typography>
          
          {alerts.length === 0 ? (
            <Alert severity="success" sx={{ mt: 2 }}>
              🎉 No active alerts! Your AWS costs are within normal parameters.
            </Alert>
          ) : (
            alerts.map((alert) => (
              <Accordion key={alert.alert_id} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    <Box sx={{ mr: 2 }}>
                      {getSeverityIcon(alert.severity)}
                    </Box>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="subtitle1">
                        {alert.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {alert.description}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Chip
                        label={alert.severity.toUpperCase()}
                        color={getSeverityColor(alert.severity) as any}
                        size="small"
                      />
                      <Chip
                        label={alert.alert_type}
                        variant="outlined"
                        size="small"
                      />
                    </Box>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" gutterBottom>
                        <strong>Service:</strong> {alert.service_affected || 'Multiple'}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Current Cost:</strong> ${alert.current_cost.toFixed(2)}
                      </Typography>
                      {alert.threshold_cost && (
                        <Typography variant="body2" gutterBottom>
                          <strong>Threshold:</strong> ${alert.threshold_cost.toFixed(2)}
                        </Typography>
                      )}
                      <Typography variant="body2" gutterBottom>
                        <strong>Created:</strong> {new Date(alert.created_at).toLocaleString()}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" gutterBottom>
                        <strong>Recommended Actions:</strong>
                      </Typography>
                      {alert.recommended_actions.map((action, index) => (
                        <Typography key={index} variant="body2" sx={{ ml: 2 }}>
                          • {action}
                        </Typography>
                      ))}
                      <Box sx={{ mt: 2 }}>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => {/* Resolve alert */}}
                        >
                          Mark as Resolved
                        </Button>
                      </Box>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            ))
          )}
        </Paper>

        {/* Daily Summary */}
        {dailySummary && (
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Today's Cost Summary
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Typography variant="body1" gutterBottom>
                  <strong>Total Cost:</strong> ${dailySummary.total_cost.toFixed(2)}
                </Typography>
                <Typography variant="body1" gutterBottom>
                  <strong>Change from Yesterday:</strong> 
                  <span style={{ color: dailySummary.cost_change >= 0 ? '#f44336' : '#4caf50' }}>
                    {dailySummary.cost_change >= 0 ? '+' : ''}${dailySummary.cost_change.toFixed(2)} 
                    ({dailySummary.cost_change_percentage.toFixed(1)}%)
                  </span>
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" gutterBottom>
                  Top Services Today
                </Typography>
                {dailySummary.top_services.slice(0, 3).map((service, index) => (
                  <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">{service.service}</Typography>
                    <Typography variant="body2">${service.cost.toFixed(2)}</Typography>
                  </Box>
                ))}
              </Grid>
            </Grid>
          </Paper>
        )}

        {/* Setup Dialog */}
        <Dialog open={setupDialogOpen} onClose={() => setSetupDialogOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle>Setup Cost Monitoring & Alerts</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              {/* AWS Credentials */}
              <Typography variant="h6" gutterBottom>
                AWS Credentials
              </Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="AWS Access Key ID"
                    value={credentials.aws_access_key_id}
                    onChange={(e) => setCredentials({ ...credentials, aws_access_key_id: e.target.value })}
                    type="password"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="AWS Secret Access Key"
                    value={credentials.aws_secret_access_key}
                    onChange={(e) => setCredentials({ ...credentials, aws_secret_access_key: e.target.value })}
                    type="password"
                  />
                </Grid>
              </Grid>

              {/* Budget Thresholds */}
              <Typography variant="h6" gutterBottom>
                Budget Thresholds
              </Typography>
              {budgetThresholds.map((threshold, index) => (
                <Paper key={index} sx={{ p: 2, mb: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="Budget Name"
                        value={threshold.name}
                        onChange={(e) => {
                          const newThresholds = [...budgetThresholds];
                          newThresholds[index].name = e.target.value;
                          setBudgetThresholds(newThresholds);
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="Monthly Budget ($)"
                        type="number"
                        value={threshold.monthly_budget}
                        onChange={(e) => {
                          const newThresholds = [...budgetThresholds];
                          newThresholds[index].monthly_budget = parseFloat(e.target.value);
                          setBudgetThresholds(newThresholds);
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <TextField
                        fullWidth
                        label="Warning %"
                        type="number"
                        value={threshold.warning_threshold}
                        onChange={(e) => {
                          const newThresholds = [...budgetThresholds];
                          newThresholds[index].warning_threshold = parseFloat(e.target.value);
                          setBudgetThresholds(newThresholds);
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <TextField
                        fullWidth
                        label="Critical %"
                        type="number"
                        value={threshold.critical_threshold}
                        onChange={(e) => {
                          const newThresholds = [...budgetThresholds];
                          newThresholds[index].critical_threshold = parseFloat(e.target.value);
                          setBudgetThresholds(newThresholds);
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={threshold.enabled}
                            onChange={(e) => {
                              const newThresholds = [...budgetThresholds];
                              newThresholds[index].enabled = e.target.checked;
                              setBudgetThresholds(newThresholds);
                            }}
                          />
                        }
                        label="Enabled"
                      />
                    </Grid>
                  </Grid>
                </Paper>
              ))}

              {/* Notifications */}
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                Notification Settings
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationConfig.email_enabled}
                    onChange={(e) => setNotificationConfig({ 
                      ...notificationConfig, 
                      email_enabled: e.target.checked 
                    })}
                  />
                }
                label="Email Notifications"
              />
              {notificationConfig.email_enabled && (
                <Box sx={{ ml: 4, mt: 2 }}>
                  <TextField
                    fullWidth
                    label="Email Address"
                    value={notificationConfig.email_config.to_email}
                    onChange={(e) => setNotificationConfig({
                      ...notificationConfig,
                      email_config: { ...notificationConfig.email_config, to_email: e.target.value }
                    })}
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              <FormControlLabel
                control={
                  <Switch
                    checked={notificationConfig.slack_enabled}
                    onChange={(e) => setNotificationConfig({ 
                      ...notificationConfig, 
                      slack_enabled: e.target.checked 
                    })}
                  />
                }
                label="Slack Notifications"
              />
              {notificationConfig.slack_enabled && (
                <Box sx={{ ml: 4, mt: 2 }}>
                  <TextField
                    fullWidth
                    label="Slack Webhook URL"
                    value={notificationConfig.slack_config.webhook_url}
                    onChange={(e) => setNotificationConfig({
                      ...notificationConfig,
                      slack_config: { ...notificationConfig.slack_config, webhook_url: e.target.value }
                    })}
                  />
                </Box>
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSetupDialogOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={() => {/* Setup monitoring */}}>
              Setup Monitoring
            </Button>
          </DialogActions>
        </Dialog>

        {/* Floating Action Button */}
        <Fab
          color="primary"
          aria-label="add"
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => setSetupDialogOpen(true)}
        >
          <AddIcon />
        </Fab>
      </Box>
    </Container>
  );
};

export default AWSCostAlerts;