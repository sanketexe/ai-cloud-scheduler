import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  Chip,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Tab,
  Tabs,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Cloud as CloudIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Speed as PerformanceIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Settings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState({
    // Cloud Provider Settings
    awsEnabled: true,
    gcpEnabled: true,
    azureEnabled: false,
    awsRegion: 'us-west-2',
    gcpRegion: 'us-central1',
    azureRegion: 'eastus',
    
    // Performance Settings
    metricsInterval: 5,
    anomalyThreshold: 0.8,
    autoScaling: true,
    maxInstances: 10,
    
    // Notification Settings
    emailNotifications: true,
    slackNotifications: false,
    webhookUrl: '',
    alertThresholds: {
      cpu: 85,
      memory: 90,
      storage: 95,
    },
    
    // Security Settings
    apiRateLimit: 1000,
    sessionTimeout: 30,
    auditLogging: true,
    encryptionEnabled: true,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleNestedSettingChange = (parent: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [parent]: {
        ...prev[parent as keyof typeof prev],
        [key]: value
      }
    }));
  };

  const handleSave = () => {
    console.log('Saving settings:', settings);
    // In real app, this would call the API
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>
        Settings
      </Typography>

      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="settings tabs">
            <Tab icon={<CloudIcon />} label="Cloud Providers" />
            <Tab icon={<PerformanceIcon />} label="Performance" />
            <Tab icon={<NotificationsIcon />} label="Notifications" />
            <Tab icon={<SecurityIcon />} label="Security" />
          </Tabs>
        </Box>

        {/* Cloud Providers Tab */}
        <TabPanel value={tabValue} index={0}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h6" sx={{ mb: 3 }}>
              Cloud Provider Configuration
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6">AWS</Typography>
                    <Chip
                      label={settings.awsEnabled ? 'Connected' : 'Disabled'}
                      color={settings.awsEnabled ? 'success' : 'default'}
                      size="small"
                    />
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.awsEnabled}
                        onChange={(e) => handleSettingChange('awsEnabled', e.target.checked)}
                      />
                    }
                    label="Enable AWS"
                  />
                  <FormControl fullWidth sx={{ mt: 2 }}>
                    <InputLabel>Region</InputLabel>
                    <Select
                      value={settings.awsRegion}
                      label="Region"
                      onChange={(e) => handleSettingChange('awsRegion', e.target.value)}
                      disabled={!settings.awsEnabled}
                    >
                      <MenuItem value="us-west-2">US West (Oregon)</MenuItem>
                      <MenuItem value="us-east-1">US East (N. Virginia)</MenuItem>
                      <MenuItem value="eu-west-1">Europe (Ireland)</MenuItem>
                    </Select>
                  </FormControl>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6">Google Cloud</Typography>
                    <Chip
                      label={settings.gcpEnabled ? 'Connected' : 'Disabled'}
                      color={settings.gcpEnabled ? 'success' : 'default'}
                      size="small"
                    />
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.gcpEnabled}
                        onChange={(e) => handleSettingChange('gcpEnabled', e.target.checked)}
                      />
                    }
                    label="Enable GCP"
                  />
                  <FormControl fullWidth sx={{ mt: 2 }}>
                    <InputLabel>Region</InputLabel>
                    <Select
                      value={settings.gcpRegion}
                      label="Region"
                      onChange={(e) => handleSettingChange('gcpRegion', e.target.value)}
                      disabled={!settings.gcpEnabled}
                    >
                      <MenuItem value="us-central1">US Central</MenuItem>
                      <MenuItem value="us-west1">US West</MenuItem>
                      <MenuItem value="europe-west1">Europe West</MenuItem>
                    </Select>
                  </FormControl>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6">Azure</Typography>
                    <Chip
                      label={settings.azureEnabled ? 'Connected' : 'Disabled'}
                      color={settings.azureEnabled ? 'success' : 'default'}
                      size="small"
                    />
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.azureEnabled}
                        onChange={(e) => handleSettingChange('azureEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Azure"
                  />
                  <FormControl fullWidth sx={{ mt: 2 }}>
                    <InputLabel>Region</InputLabel>
                    <Select
                      value={settings.azureRegion}
                      label="Region"
                      onChange={(e) => handleSettingChange('azureRegion', e.target.value)}
                      disabled={!settings.azureEnabled}
                    >
                      <MenuItem value="eastus">East US</MenuItem>
                      <MenuItem value="westus2">West US 2</MenuItem>
                      <MenuItem value="westeurope">West Europe</MenuItem>
                    </Select>
                  </FormControl>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        </TabPanel>

        {/* Performance Tab */}
        <TabPanel value={tabValue} index={1}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h6" sx={{ mb: 3 }}>
              Performance & Monitoring Settings
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ mb: 2 }}>
                  Metrics Collection Interval (minutes)
                </Typography>
                <Slider
                  value={settings.metricsInterval}
                  onChange={(e, value) => handleSettingChange('metricsInterval', value)}
                  min={1}
                  max={60}
                  marks={[
                    { value: 1, label: '1m' },
                    { value: 5, label: '5m' },
                    { value: 15, label: '15m' },
                    { value: 60, label: '1h' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ mb: 2 }}>
                  Anomaly Detection Threshold
                </Typography>
                <Slider
                  value={settings.anomalyThreshold}
                  onChange={(e, value) => handleSettingChange('anomalyThreshold', value)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: 'Sensitive' },
                    { value: 0.5, label: 'Normal' },
                    { value: 1.0, label: 'Conservative' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.autoScaling}
                      onChange={(e) => handleSettingChange('autoScaling', e.target.checked)}
                    />
                  }
                  label="Enable Auto-scaling"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Maximum Instances"
                  type="number"
                  value={settings.maxInstances}
                  onChange={(e) => handleSettingChange('maxInstances', parseInt(e.target.value))}
                  disabled={!settings.autoScaling}
                />
              </Grid>
            </Grid>
          </motion.div>
        </TabPanel>

        {/* Notifications Tab */}
        <TabPanel value={tabValue} index={2}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h6" sx={{ mb: 3 }}>
              Notification Settings
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.emailNotifications}
                      onChange={(e) => handleSettingChange('emailNotifications', e.target.checked)}
                    />
                  }
                  label="Email Notifications"
                />
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.slackNotifications}
                      onChange={(e) => handleSettingChange('slackNotifications', e.target.checked)}
                    />
                  }
                  label="Slack Notifications"
                />
              </Grid>

              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Webhook URL"
                  value={settings.webhookUrl}
                  onChange={(e) => handleSettingChange('webhookUrl', e.target.value)}
                  placeholder="https://hooks.slack.com/services/..."
                />
              </Grid>

              <Grid item xs={12}>
                <Typography variant="subtitle1" sx={{ mb: 2 }}>
                  Alert Thresholds
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      label="CPU (%)"
                      type="number"
                      value={settings.alertThresholds.cpu}
                      onChange={(e) => handleNestedSettingChange('alertThresholds', 'cpu', parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      label="Memory (%)"
                      type="number"
                      value={settings.alertThresholds.memory}
                      onChange={(e) => handleNestedSettingChange('alertThresholds', 'memory', parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      label="Storage (%)"
                      type="number"
                      value={settings.alertThresholds.storage}
                      onChange={(e) => handleNestedSettingChange('alertThresholds', 'storage', parseInt(e.target.value))}
                    />
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </motion.div>
        </TabPanel>

        {/* Security Tab */}
        <TabPanel value={tabValue} index={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h6" sx={{ mb: 3 }}>
              Security Settings
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="API Rate Limit (requests/hour)"
                  type="number"
                  value={settings.apiRateLimit}
                  onChange={(e) => handleSettingChange('apiRateLimit', parseInt(e.target.value))}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Session Timeout (minutes)"
                  type="number"
                  value={settings.sessionTimeout}
                  onChange={(e) => handleSettingChange('sessionTimeout', parseInt(e.target.value))}
                />
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.auditLogging}
                      onChange={(e) => handleSettingChange('auditLogging', e.target.checked)}
                    />
                  }
                  label="Enable Audit Logging"
                />
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.encryptionEnabled}
                      onChange={(e) => handleSettingChange('encryptionEnabled', e.target.checked)}
                    />
                  }
                  label="Enable Data Encryption"
                />
              </Grid>

              <Grid item xs={12}>
                <Alert severity="info">
                  Security settings require administrator privileges to modify.
                </Alert>
              </Grid>
            </Grid>
          </motion.div>
        </TabPanel>

        <Divider />
        <Box sx={{ p: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSave}
            size="large"
          >
            Save Settings
          </Button>
        </Box>
      </Card>
    </Box>
  );
};

export default Settings;