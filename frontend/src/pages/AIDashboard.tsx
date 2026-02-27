import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  Tab,
  Tabs,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Psychology as AIIcon,
  TrendingUp as ScalingIcon,
  CloudQueue as WorkloadIcon,
  Chat as ChatIcon,
  Science as MLIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import PredictiveScalingInterface from '../components/AI/PredictiveScalingInterface';
import WorkloadPlacementWizard from '../components/AI/WorkloadPlacementWizard';
import NaturalLanguageChat from '../components/AI/NaturalLanguageChat';
import MLModelManagement from '../components/AI/MLModelManagement';

interface AISystemStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'offline';
  uptime: string;
  lastUpdate: string;
  metrics: {
    accuracy?: number;
    latency?: number;
    throughput?: number;
    errorRate?: number;
  };
}

interface AISystemMetrics {
  predictiveScaling: AISystemStatus;
  workloadIntelligence: AISystemStatus;
  reinforcementLearning: AISystemStatus;
  naturalLanguage: AISystemStatus;
  graphNeuralNetwork: AISystemStatus;
  predictiveMaintenance: AISystemStatus;
  smartContractOptimizer: AISystemStatus;
  mlModelManagement: AISystemStatus;
}

const AIDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch AI system metrics
  const { data: aiMetrics, isLoading, refetch } = useQuery<AISystemMetrics>(
    'ai-system-metrics',
    async () => {
      const response = await fetch('/api/ai/system-metrics');
      if (!response.ok) {
        throw new Error('Failed to fetch AI system metrics');
      }
      return response.json();
    },
    {
      refetchInterval: autoRefresh ? 30000 : false, // Refresh every 30 seconds if auto-refresh is enabled
      refetchOnWindowFocus: false,
    }
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      case 'offline':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'offline':
        return <ErrorIcon color="disabled" />;
      default:
        return <ErrorIcon color="disabled" />;
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    refetch();
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          AI Dashboard
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <AIIcon sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 600 }}>
              AI Dashboard
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Real-time AI system monitoring and management
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Refresh"
          />
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} color="primary">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="AI Settings">
            <IconButton color="primary">
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* System Status Overview */}
      {aiMetrics && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {Object.entries(aiMetrics).map(([key, system]) => (
            <Grid item xs={12} sm={6} md={3} key={key}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{ fontSize: '0.9rem', fontWeight: 600 }}>
                      {system.name}
                    </Typography>
                    {getStatusIcon(system.status)}
                  </Box>
                  <Chip
                    label={system.status.toUpperCase()}
                    color={getStatusColor(system.status) as any}
                    size="small"
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Uptime: {system.uptime}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Last Update: {system.lastUpdate}
                  </Typography>
                  {system.metrics.accuracy && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        Accuracy: {(system.metrics.accuracy * 100).toFixed(1)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={system.metrics.accuracy * 100}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  )}
                  {system.metrics.latency && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                      Latency: {system.metrics.latency}ms
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* System Alerts */}
      <Box sx={{ mb: 3 }}>
        <Alert severity="info" sx={{ mb: 2 }}>
          All AI systems are operating within normal parameters. Predictive scaling has prevented 3 potential issues in the last 24 hours.
        </Alert>
      </Box>

      {/* AI Interface Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="AI interface tabs">
            <Tab
              icon={<ScalingIcon />}
              label="Predictive Scaling"
              iconPosition="start"
              sx={{ textTransform: 'none' }}
            />
            <Tab
              icon={<WorkloadIcon />}
              label="Workload Placement"
              iconPosition="start"
              sx={{ textTransform: 'none' }}
            />
            <Tab
              icon={<ChatIcon />}
              label="Natural Language"
              iconPosition="start"
              sx={{ textTransform: 'none' }}
            />
            <Tab
              icon={<MLIcon />}
              label="ML Models"
              iconPosition="start"
              sx={{ textTransform: 'none' }}
            />
          </Tabs>
        </Box>

        <CardContent sx={{ p: 0 }}>
          {activeTab === 0 && <PredictiveScalingInterface />}
          {activeTab === 1 && <WorkloadPlacementWizard />}
          {activeTab === 2 && <NaturalLanguageChat />}
          {activeTab === 3 && <MLModelManagement />}
        </CardContent>
      </Card>
    </Box>
  );
};

export default AIDashboard;