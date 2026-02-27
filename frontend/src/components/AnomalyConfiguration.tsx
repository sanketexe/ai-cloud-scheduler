import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { anomalyApiService, AnomalyConfiguration } from '../services/anomalyApi';

interface AnomalyConfigurationProps {
  accountId: string;
  onConfigurationChange?: (config: AnomalyConfiguration) => void;
}

const AnomalyConfigurationComponent: React.FC<AnomalyConfigurationProps> = ({
  accountId,
  onConfigurationChange
}) => {
  const [configuration, setConfiguration] = useState<AnomalyConfiguration>({
    sensitivity_level: 'balanced',
    threshold_percentage: 20.0,
    baseline_period_days: 30,
    min_cost_threshold: 1.0,
    excluded_services: [],
    maintenance_windows: [],
    notification_channels: ['email'],
    escalation_rules: {}
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [maintenanceDialogOpen, setMaintenanceDialogOpen] = useState(false);
  const [newMaintenanceWindow, setNewMaintenanceWindow] = useState({
    start: new Date(),
    end: new Date()
  });

  const availableServices = [
    'EC2', 'S3', 'RDS', 'Lambda', 'CloudWatch', 'CloudTrail', 'EBS', 'ELB',
    'CloudFront', 'Route53', 'VPC', 'NAT Gateway', 'ElastiCache', 'Redshift'
  ];

  const availableChannels = [
    'email', 'slack', 'webhook', 'sms', 'teams', 'pagerduty'
  ];

  useEffect(() => {
    loadConfiguration();
  }, [accountId]);

  const loadConfiguration = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await anomalyApiService.getConfiguration(accountId);
      setConfiguration(response.configuration);
    } catch (err: any) {
      setError('Failed to load configuration: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const saveConfiguration = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await anomalyApiService.updateConfiguration(accountId, configuration);
      setSuccess('Configuration saved successfully');
      onConfigurationChange?.(response.configuration);
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError('Failed to save configuration: ' + (err.message || 'Unknown error'));
    } finally {
      setSaving(false);
    }
  };

  const handleConfigChange = (field: keyof AnomalyConfiguration, value: any) => {
    setConfiguration(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const addExcludedService = (service: string) => {
    if (!configuration.excluded_services.includes(service)) {
      handleConfigChange('excluded_services', [...configuration.excluded_services, service]);
    }
  };

  const removeExcludedService = (service: string) => {
    handleConfigChange('excluded_services', 
      configuration.excluded_services.filter(s => s !== service)
    );
  };

  const addNotificationChannel = (channel: string) => {
    if (!configuration.notification_channels.includes(channel)) {
      handleConfigChange('notification_channels', [...configuration.notification_channels, channel]);
    }
  };

  const removeNotificationChannel = (channel: string) => {
    handleConfigChange('notification_channels', 
      configuration.notification_channels.filter(c => c !== channel)
    );
  };

  const addMaintenanceWindow = () => {
    const newWindow = {
      start: newMaintenanceWindow.start.toISOString(),
      end: newMaintenanceWindow.end.toISOString()
    };
    
    handleConfigChange('maintenance_windows', [...configuration.maintenance_windows, newWindow]);
    setMaintenanceDialogOpen(false);
    setNewMaintenanceWindow({
      start: new Date(),
      end: new Date()
    });
  };

  const removeMaintenanceWindow = (index: number) => {
    const windows = [...configuration.maintenance_windows];
    windows.splice(index, 1);
    handleConfigChange('maintenance_windows', windows);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading configuration...</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6">
              Anomaly Detection Configuration
            </Typography>
            <Box>
              <IconButton onClick={loadConfiguration} disabled={loading}>
                <RefreshIcon />
              </IconButton>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={saveConfiguration}
                disabled={saving}
                sx={{ ml: 1 }}
              >
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>
            </Box>
          </Box>

          <Grid container spacing={3}>
            {/* Basic Settings */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Detection Settings
              </Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Sensitivity Level</InputLabel>
                <Select
                  value={configuration.sensitivity_level}
                  label="Sensitivity Level"
                  onChange={(e) => handleConfigChange('sensitivity_level', e.target.value)}
                >
                  <MenuItem value="conservative">Conservative (Fewer alerts)</MenuItem>
                  <MenuItem value="balanced">Balanced (Recommended)</MenuItem>
                  <MenuItem value="aggressive">Aggressive (More alerts)</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Threshold Percentage"
                type="number"
                value={configuration.threshold_percentage}
                onChange={(e) => handleConfigChange('threshold_percentage', parseFloat(e.target.value))}
                inputProps={{ min: 1, max: 100, step: 0.1 }}
                helperText="Minimum percentage deviation to trigger anomaly"
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="Baseline Period (Days)"
                type="number"
                value={configuration.baseline_period_days}
                onChange={(e) => handleConfigChange('baseline_period_days', parseInt(e.target.value))}
                inputProps={{ min: 7, max: 365 }}
                helperText="Historical period for baseline calculation"
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="Minimum Cost Threshold ($)"
                type="number"
                value={configuration.min_cost_threshold}
                onChange={(e) => handleConfigChange('min_cost_threshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, step: 0.01 }}
                helperText="Ignore anomalies below this cost threshold"
              />
            </Grid>

            {/* Exclusions and Channels */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Exclusions & Notifications
              </Typography>

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Excluded Services
                </Typography>
                <Box sx={{ mb: 1 }}>
                  {configuration.excluded_services.map((service) => (
                    <Chip
                      key={service}
                      label={service}
                      onDelete={() => removeExcludedService(service)}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Add Service</InputLabel>
                  <Select
                    label="Add Service"
                    value=""
                    onChange={(e) => addExcludedService(e.target.value)}
                  >
                    {availableServices
                      .filter(service => !configuration.excluded_services.includes(service))
                      .map((service) => (
                        <MenuItem key={service} value={service}>
                          {service}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Box>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Notification Channels
                </Typography>
                <Box sx={{ mb: 1 }}>
                  {configuration.notification_channels.map((channel) => (
                    <Chip
                      key={channel}
                      label={channel}
                      onDelete={() => removeNotificationChannel(channel)}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Add Channel</InputLabel>
                  <Select
                    label="Add Channel"
                    value=""
                    onChange={(e) => addNotificationChannel(e.target.value)}
                  >
                    {availableChannels
                      .filter(channel => !configuration.notification_channels.includes(channel))
                      .map((channel) => (
                        <MenuItem key={channel} value={channel}>
                          {channel}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Box>
            </Grid>

            {/* Escalation Rules */}
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle1" gutterBottom>
                Escalation Rules
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="High Severity Threshold ($)"
                    type="number"
                    value={configuration.escalation_rules.high_severity_threshold || ''}
                    onChange={(e) => handleConfigChange('escalation_rules', {
                      ...configuration.escalation_rules,
                      high_severity_threshold: parseFloat(e.target.value) || undefined
                    })}
                    inputProps={{ min: 0, step: 0.01 }}
                    helperText="Cost threshold for high severity escalation"
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Escalation Delay (Minutes)"
                    type="number"
                    value={configuration.escalation_rules.escalation_delay_minutes || ''}
                    onChange={(e) => handleConfigChange('escalation_rules', {
                      ...configuration.escalation_rules,
                      escalation_delay_minutes: parseInt(e.target.value) || undefined
                    })}
                    inputProps={{ min: 1, max: 1440 }}
                    helperText="Delay before escalating unacknowledged alerts"
                  />
                </Grid>
              </Grid>
            </Grid>

            {/* Maintenance Windows */}
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle1">
                  Maintenance Windows
                </Typography>
                <Button
                  startIcon={<AddIcon />}
                  onClick={() => setMaintenanceDialogOpen(true)}
                  size="small"
                >
                  Add Window
                </Button>
              </Box>

              {configuration.maintenance_windows.length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No maintenance windows configured
                </Typography>
              ) : (
                <List>
                  {configuration.maintenance_windows.map((window, index) => (
                    <ListItem key={index}>
                      <ListItemText
                        primary={`${new Date(window.start).toLocaleString()} - ${new Date(window.end).toLocaleString()}`}
                        secondary="Anomaly detection will be suppressed during this period"
                      />
                      <ListItemSecondaryAction>
                        <IconButton onClick={() => removeMaintenanceWindow(index)}>
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Maintenance Window Dialog */}
      <Dialog open={maintenanceDialogOpen} onClose={() => setMaintenanceDialogOpen(false)}>
        <DialogTitle>Add Maintenance Window</DialogTitle>
        <DialogContent>
          <LocalizationProvider dateAdapter={AdapterDateFns}>
            <Box sx={{ pt: 1 }}>
              <DateTimePicker
                label="Start Time"
                value={newMaintenanceWindow.start}
                onChange={(date) => setNewMaintenanceWindow(prev => ({ ...prev, start: date || new Date() }))}
                sx={{ mb: 2, width: '100%' }}
              />
              <DateTimePicker
                label="End Time"
                value={newMaintenanceWindow.end}
                onChange={(date) => setNewMaintenanceWindow(prev => ({ ...prev, end: date || new Date() }))}
                sx={{ width: '100%' }}
              />
            </Box>
          </LocalizationProvider>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMaintenanceDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={addMaintenanceWindow} variant="contained">
            Add Window
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AnomalyConfigurationComponent;