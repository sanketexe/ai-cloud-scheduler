import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useQuery, useMutation, useQueryClient } from 'react-query';

interface ScalingRule {
  id: string;
  name: string;
  resourceType: string;
  provider: string;
  enabled: boolean;
  minInstances: number;
  maxInstances: number;
  targetUtilization: number;
  scaleUpCooldown: number;
  scaleDownCooldown: number;
  predictiveHorizon: number;
  confidence: number;
  lastTriggered?: string;
  status: 'active' | 'inactive' | 'error';
}

interface PredictionData {
  timestamp: string;
  predicted: number;
  actual?: number;
  confidence: number;
  action?: 'scale_up' | 'scale_down' | 'no_action';
}

const PredictiveScalingInterface: React.FC = () => {
  const [selectedRule, setSelectedRule] = useState<string>('');
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingRule, setEditingRule] = useState<ScalingRule | null>(null);
  const queryClient = useQueryClient();

  // Fetch scaling rules
  const { data: scalingRules, isLoading: rulesLoading } = useQuery<ScalingRule[]>(
    'predictive-scaling-rules',
    async () => {
      const response = await fetch('/api/ai/predictive-scaling/rules');
      if (!response.ok) throw new Error('Failed to fetch scaling rules');
      return response.json();
    }
  );

  // Fetch prediction data for selected rule
  const { data: predictionData, isLoading: predictionLoading } = useQuery<PredictionData[]>(
    ['predictive-scaling-data', selectedRule],
    async () => {
      if (!selectedRule) return [];
      const response = await fetch(`/api/ai/predictive-scaling/predictions/${selectedRule}`);
      if (!response.ok) throw new Error('Failed to fetch prediction data');
      return response.json();
    },
    { enabled: !!selectedRule }
  );

  // Mutation for updating scaling rules
  const updateRuleMutation = useMutation(
    async (rule: ScalingRule) => {
      const response = await fetch(`/api/ai/predictive-scaling/rules/${rule.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rule),
      });
      if (!response.ok) throw new Error('Failed to update scaling rule');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('predictive-scaling-rules');
        setConfigDialogOpen(false);
        setEditingRule(null);
      },
    }
  );

  // Mutation for toggling rule status
  const toggleRuleMutation = useMutation(
    async ({ ruleId, enabled }: { ruleId: string; enabled: boolean }) => {
      const response = await fetch(`/api/ai/predictive-scaling/rules/${ruleId}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error('Failed to toggle scaling rule');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('predictive-scaling-rules');
      },
    }
  );

  const handleRuleToggle = (ruleId: string, enabled: boolean) => {
    toggleRuleMutation.mutate({ ruleId, enabled });
  };

  const handleEditRule = (rule: ScalingRule) => {
    setEditingRule(rule);
    setConfigDialogOpen(true);
  };

  const handleSaveRule = () => {
    if (editingRule) {
      updateRuleMutation.mutate(editingRule);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'inactive': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <TrendingUpIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6">Predictive Scaling Configuration</Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setEditingRule({
              id: '',
              name: '',
              resourceType: 'ec2',
              provider: 'aws',
              enabled: true,
              minInstances: 1,
              maxInstances: 10,
              targetUtilization: 70,
              scaleUpCooldown: 300,
              scaleDownCooldown: 600,
              predictiveHorizon: 60,
              confidence: 0.8,
              status: 'inactive',
            });
            setConfigDialogOpen(true);
          }}
        >
          Add Rule
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Scaling Rules List */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scaling Rules
              </Typography>
              {rulesLoading ? (
                <Typography>Loading rules...</Typography>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Resource</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Enabled</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {scalingRules?.map((rule) => (
                        <TableRow key={rule.id}>
                          <TableCell>{rule.name}</TableCell>
                          <TableCell>{rule.resourceType}</TableCell>
                          <TableCell>
                            <Chip
                              label={rule.status}
                              color={getStatusColor(rule.status) as any}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Switch
                              checked={rule.enabled}
                              onChange={(e) => handleRuleToggle(rule.id, e.target.checked)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Tooltip title="View Details">
                              <IconButton
                                size="small"
                                onClick={() => setSelectedRule(rule.id)}
                              >
                                <ViewIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Edit Rule">
                              <IconButton
                                size="small"
                                onClick={() => handleEditRule(rule)}
                              >
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Visualization */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Demand Predictions
              </Typography>
              {selectedRule ? (
                predictionLoading ? (
                  <Typography>Loading predictions...</Typography>
                ) : (
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={predictionData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <RechartsTooltip />
                        <Area
                          type="monotone"
                          dataKey="predicted"
                          stroke="#2196f3"
                          fill="#2196f3"
                          fillOpacity={0.3}
                          name="Predicted Demand"
                        />
                        <Line
                          type="monotone"
                          dataKey="actual"
                          stroke="#4caf50"
                          strokeWidth={2}
                          name="Actual Demand"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Box>
                )
              ) : (
                <Alert severity="info">
                  Select a scaling rule to view demand predictions
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Scaling Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Scaling Actions
              </Typography>
              <Alert severity="success" sx={{ mb: 2 }}>
                Predictive scaling prevented 3 potential performance issues in the last 24 hours
              </Alert>
              <Typography variant="body2" color="text.secondary">
                • Pre-scaled web servers 15 minutes before traffic spike at 2:30 PM
              </Typography>
              <Typography variant="body2" color="text.secondary">
                • Reduced database instances during low usage period at 11:45 PM
              </Typography>
              <Typography variant="body2" color="text.secondary">
                • Optimized compute resources based on seasonal patterns at 6:00 AM
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Rule Configuration Dialog */}
      <Dialog open={configDialogOpen} onClose={() => setConfigDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingRule?.id ? 'Edit Scaling Rule' : 'Create Scaling Rule'}
        </DialogTitle>
        <DialogContent>
          {editingRule && (
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Rule Name"
                  value={editingRule.name}
                  onChange={(e) => setEditingRule({ ...editingRule, name: e.target.value })}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Resource Type</InputLabel>
                  <Select
                    value={editingRule.resourceType}
                    onChange={(e) => setEditingRule({ ...editingRule, resourceType: e.target.value })}
                  >
                    <MenuItem value="ec2">EC2 Instances</MenuItem>
                    <MenuItem value="ecs">ECS Services</MenuItem>
                    <MenuItem value="lambda">Lambda Functions</MenuItem>
                    <MenuItem value="rds">RDS Instances</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Min Instances"
                  value={editingRule.minInstances}
                  onChange={(e) => setEditingRule({ ...editingRule, minInstances: parseInt(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Max Instances"
                  value={editingRule.maxInstances}
                  onChange={(e) => setEditingRule({ ...editingRule, maxInstances: parseInt(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>Target Utilization: {editingRule.targetUtilization}%</Typography>
                <Slider
                  value={editingRule.targetUtilization}
                  onChange={(_, value) => setEditingRule({ ...editingRule, targetUtilization: value as number })}
                  min={10}
                  max={90}
                  step={5}
                  marks
                  valueLabelDisplay="auto"
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>Prediction Confidence: {(editingRule.confidence * 100).toFixed(0)}%</Typography>
                <Slider
                  value={editingRule.confidence}
                  onChange={(_, value) => setEditingRule({ ...editingRule, confidence: value as number })}
                  min={0.5}
                  max={0.95}
                  step={0.05}
                  marks
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveRule} variant="contained">
            Save Rule
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PredictiveScalingInterface;