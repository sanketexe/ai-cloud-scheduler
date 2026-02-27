import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Science as MLIcon,
  PlayArrow as DeployIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  TrendingUp as MetricsIcon,
  Compare as CompareIcon,
  ExpandMore as ExpandMoreIcon,
  CloudUpload as UploadIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useQuery, useMutation, useQueryClient } from 'react-query';

interface MLModel {
  id: string;
  name: string;
  type: 'predictive_scaling' | 'workload_intelligence' | 'anomaly_detection' | 'cost_optimization';
  version: string;
  status: 'training' | 'deployed' | 'testing' | 'failed' | 'archived';
  accuracy: number;
  lastTrained: string;
  deployedAt?: string;
  trainingMetrics: {
    loss: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
  performanceMetrics: {
    latency: number;
    throughput: number;
    errorRate: number;
  };
  description: string;
  framework: string;
  datasetSize: number;
  hyperparameters: Record<string, any>;
}

interface Experiment {
  id: string;
  name: string;
  modelType: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  startTime: string;
  endTime?: string;
  metrics: Record<string, number>;
  hyperparameters: Record<string, any>;
  notes: string;
}

interface ABTest {
  id: string;
  name: string;
  modelA: string;
  modelB: string;
  status: 'running' | 'completed' | 'paused';
  startDate: string;
  endDate?: string;
  trafficSplit: number;
  metrics: {
    modelA: Record<string, number>;
    modelB: Record<string, number>;
  };
  winner?: 'A' | 'B' | 'inconclusive';
}

const MLModelManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [experimentDialogOpen, setExperimentDialogOpen] = useState(false);
  const [abTestDialogOpen, setABTestDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  // Fetch ML models
  const { data: models, isLoading: modelsLoading } = useQuery<MLModel[]>(
    'ml-models',
    async () => {
      const response = await fetch('/api/ai/ml-models');
      if (!response.ok) throw new Error('Failed to fetch ML models');
      return response.json();
    }
  );

  // Fetch experiments
  const { data: experiments, isLoading: experimentsLoading } = useQuery<Experiment[]>(
    'ml-experiments',
    async () => {
      const response = await fetch('/api/ai/ml-experiments');
      if (!response.ok) throw new Error('Failed to fetch experiments');
      return response.json();
    }
  );

  // Fetch A/B tests
  const { data: abTests, isLoading: abTestsLoading } = useQuery<ABTest[]>(
    'ab-tests',
    async () => {
      const response = await fetch('/api/ai/ab-tests');
      if (!response.ok) throw new Error('Failed to fetch A/B tests');
      return response.json();
    }
  );

  // Model deployment mutation
  const deployModelMutation = useMutation(
    async (modelId: string) => {
      const response = await fetch(`/api/ai/ml-models/${modelId}/deploy`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to deploy model');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('ml-models');
      },
    }
  );

  // Model retraining mutation
  const retrainModelMutation = useMutation(
    async (modelId: string) => {
      const response = await fetch(`/api/ai/ml-models/${modelId}/retrain`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to retrain model');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('ml-models');
      },
    }
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'success';
      case 'training': return 'info';
      case 'testing': return 'warning';
      case 'failed': return 'error';
      case 'archived': return 'default';
      default: return 'default';
    }
  };

  const getModelTypeLabel = (type: string) => {
    switch (type) {
      case 'predictive_scaling': return 'Predictive Scaling';
      case 'workload_intelligence': return 'Workload Intelligence';
      case 'anomaly_detection': return 'Anomaly Detection';
      case 'cost_optimization': return 'Cost Optimization';
      default: return type;
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleDeployModel = (modelId: string) => {
    deployModelMutation.mutate(modelId);
  };

  const handleRetrainModel = (modelId: string) => {
    retrainModelMutation.mutate(modelId);
  };

  const renderModelsTab = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">ML Models</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setModelDialogOpen(true)}
        >
          Train New Model
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Version</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>Last Trained</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models?.map((model) => (
              <TableRow key={model.id}>
                <TableCell>
                  <Typography variant="subtitle2">{model.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {model.framework}
                  </Typography>
                </TableCell>
                <TableCell>{getModelTypeLabel(model.type)}</TableCell>
                <TableCell>{model.version}</TableCell>
                <TableCell>
                  <Chip
                    label={model.status}
                    color={getStatusColor(model.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={model.accuracy * 100}
                      sx={{ width: 60, height: 6 }}
                    />
                    <Typography variant="body2">
                      {(model.accuracy * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>{model.lastTrained}</TableCell>
                <TableCell>
                  <Tooltip title="View Details">
                    <IconButton
                      size="small"
                      onClick={() => setSelectedModel(model)}
                    >
                      <ViewIcon />
                    </IconButton>
                  </Tooltip>
                  {model.status === 'testing' && (
                    <Tooltip title="Deploy">
                      <IconButton
                        size="small"
                        onClick={() => handleDeployModel(model.id)}
                        color="primary"
                      >
                        <DeployIcon />
                      </IconButton>
                    </Tooltip>
                  )}
                  <Tooltip title="Retrain">
                    <IconButton
                      size="small"
                      onClick={() => handleRetrainModel(model.id)}
                      color="secondary"
                    >
                      <RefreshIcon />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Model Details Dialog */}
      <Dialog
        open={!!selectedModel}
        onClose={() => setSelectedModel(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Model Details: {selectedModel?.name}
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Training Metrics
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Accuracy: {(selectedModel.trainingMetrics.accuracy * 100).toFixed(2)}%
                  </Typography>
                  <Typography variant="body2">
                    Precision: {(selectedModel.trainingMetrics.precision * 100).toFixed(2)}%
                  </Typography>
                  <Typography variant="body2">
                    Recall: {(selectedModel.trainingMetrics.recall * 100).toFixed(2)}%
                  </Typography>
                  <Typography variant="body2">
                    F1 Score: {selectedModel.trainingMetrics.f1Score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2">
                    Loss: {selectedModel.trainingMetrics.loss.toFixed(4)}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Performance Metrics
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Latency: {selectedModel.performanceMetrics.latency}ms
                  </Typography>
                  <Typography variant="body2">
                    Throughput: {selectedModel.performanceMetrics.throughput} req/sec
                  </Typography>
                  <Typography variant="body2">
                    Error Rate: {(selectedModel.performanceMetrics.errorRate * 100).toFixed(2)}%
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Description
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedModel.description}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Hyperparameters
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(selectedModel.hyperparameters).map(([key, value]) => (
                    <Chip
                      key={key}
                      label={`${key}: ${value}`}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedModel(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );

  const renderExperimentsTab = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Experiments</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setExperimentDialogOpen(true)}
        >
          New Experiment
        </Button>
      </Box>

      <Grid container spacing={3}>
        {experiments?.map((experiment) => (
          <Grid item xs={12} md={6} key={experiment.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box>
                    <Typography variant="h6">{experiment.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {experiment.modelType}
                    </Typography>
                  </Box>
                  <Chip
                    label={experiment.status}
                    color={getStatusColor(experiment.status) as any}
                    size="small"
                  />
                </Box>

                <Typography variant="body2" sx={{ mb: 2 }}>
                  Started: {experiment.startTime}
                </Typography>

                {experiment.status === 'completed' && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Results
                    </Typography>
                    {Object.entries(experiment.metrics).map(([metric, value]) => (
                      <Typography key={metric} variant="body2">
                        {metric}: {typeof value === 'number' ? value.toFixed(3) : value}
                      </Typography>
                    ))}
                  </Box>
                )}

                {experiment.notes && (
                  <Typography variant="body2" color="text.secondary">
                    {experiment.notes}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const renderABTestsTab = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">A/B Tests</Typography>
        <Button
          variant="contained"
          startIcon={<CompareIcon />}
          onClick={() => setABTestDialogOpen(true)}
        >
          New A/B Test
        </Button>
      </Box>

      {abTests?.map((test) => (
        <Card key={test.id} sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
              <Box>
                <Typography variant="h6">{test.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {test.modelA} vs {test.modelB}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Chip
                  label={test.status}
                  color={getStatusColor(test.status) as any}
                  size="small"
                />
                {test.winner && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Winner: Model {test.winner}
                  </Typography>
                )}
              </Box>
            </Box>

            <Typography variant="body2" sx={{ mb: 2 }}>
              Traffic Split: {test.trafficSplit}% / {100 - test.trafficSplit}%
            </Typography>

            {test.status === 'completed' && (
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Model A Metrics
                  </Typography>
                  {Object.entries(test.metrics.modelA).map(([metric, value]) => (
                    <Typography key={metric} variant="body2">
                      {metric}: {value.toFixed(3)}
                    </Typography>
                  ))}
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Model B Metrics
                  </Typography>
                  {Object.entries(test.metrics.modelB).map(([metric, value]) => (
                    <Typography key={metric} variant="body2">
                      {metric}: {value.toFixed(3)}
                    </Typography>
                  ))}
                </Grid>
              </Grid>
            )}
          </CardContent>
        </Card>
      ))}
    </Box>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <MLIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">ML Model Management</Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Manage your machine learning models, track experiments, and run A/B tests to continuously improve AI performance.
      </Alert>

      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Models" />
            <Tab label="Experiments" />
            <Tab label="A/B Tests" />
          </Tabs>
        </Box>
        <CardContent>
          {activeTab === 0 && renderModelsTab()}
          {activeTab === 1 && renderExperimentsTab()}
          {activeTab === 2 && renderABTestsTab()}
        </CardContent>
      </Card>
    </Box>
  );
};

export default MLModelManagement;