/**
 * Migration Planner Page
 * 
 * Main interface for on-premises to cloud migration planning.
 * Helps startups plan migration from physical servers to AWS/cloud,
 * with TCO comparison, timeline, and risk assessment.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Schedule as TimelineIcon,
  Assessment as AssessmentIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  TrendingUp as TrendingUpIcon,
  Storage as StorageIcon,
  Cloud as CloudIcon
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import { Helmet } from 'react-helmet-async';
import toast from 'react-hot-toast';

import {
  multiCloudApi,
  MigrationAnalysis,
  MigrationRequest,
  WorkloadListResponse
} from '../services/multiCloudApi';
import MigrationTimeline from '../components/Migration/MigrationTimeline';
import CostBenefitChart from '../components/Migration/CostBenefitChart';
import RiskAssessment from '../components/Migration/RiskAssessment';

interface MigrationStep {
  label: string;
  description: string;
  completed: boolean;
  optional?: boolean;
}

const migrationSteps: MigrationStep[] = [
  {
    label: 'Infrastructure Audit',
    description: 'Inventory your on-premises servers, storage, and network',
    completed: false
  },
  {
    label: 'TCO Comparison',
    description: 'Compare on-premises costs vs. cloud operational costs',
    completed: false
  },
  {
    label: 'Risk Assessment',
    description: 'Identify migration risks and data transfer challenges',
    completed: false
  },
  {
    label: 'Migration Timeline',
    description: 'Plan phased migration from on-prem to cloud',
    completed: false
  },
  {
    label: 'Team Training',
    description: 'Prepare team for cloud operations and DevOps practices',
    completed: false,
    optional: true
  },
  {
    label: 'Go-Live Planning',
    description: 'Finalize cutover plan and rollback strategy',
    completed: false
  }
];

const MigrationPlanner: React.FC = () => {
  const [selectedWorkloadId, setSelectedWorkloadId] = useState<string>('');
  const [targetProvider, setTargetProvider] = useState<'aws' | 'gcp' | 'azure'>('aws');
  const [serverCount, setServerCount] = useState<number>(5);
  const [migrationAnalysis, setMigrationAnalysis] = useState<MigrationAnalysis | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [showExportDialog, setShowExportDialog] = useState(false);

  // Fetch user workloads
  const {
    data: workloads,
    isLoading: workloadsLoading,
    error: workloadsError,
    refetch: refetchWorkloads
  } = useQuery<WorkloadListResponse>(
    'userWorkloads',
    () => multiCloudApi.getWorkloadSpecifications(1, 50),
    {
      staleTime: 2 * 60 * 1000,
      onError: (error) => {
        console.error('Failed to fetch workloads:', error);
        toast.error('Failed to load workloads');
      }
    }
  );

  // Migration analysis mutation
  const migrationMutation = useMutation(
    (request: MigrationRequest) => multiCloudApi.analyzeMigration(request),
    {
      onSuccess: (data) => {
        setMigrationAnalysis(data);
        setCompletedSteps(new Set([0, 1, 2, 3])); // Mark first 4 steps as completed
        setActiveStep(4);
        toast.success('Migration analysis completed successfully');
      },
      onError: (error: any) => {
        toast.error(`Migration analysis failed: ${error.message}`);
      }
    }
  );

  const handleStartAnalysis = () => {
    if (!selectedWorkloadId || !targetProvider) {
      toast.error('Please select a workload and target cloud provider');
      return;
    }

    const request: MigrationRequest = {
      workload_id: selectedWorkloadId,
      source_provider: 'on_premises' as any,
      target_provider: targetProvider,
      team_size: 5,
      include_training_costs: true
    };

    migrationMutation.mutate(request);
  };

  const handleStepClick = (stepIndex: number) => {
    if (completedSteps.has(stepIndex) || stepIndex <= activeStep) {
      setActiveStep(stepIndex);
    }
  };

  const handleExportPlan = () => {
    if (!migrationAnalysis) {
      toast.error('No migration analysis to export');
      return;
    }

    const exportData = {
      analysis: migrationAnalysis,
      workload_id: selectedWorkloadId,
      source: 'on_premises',
      target_provider: targetProvider,
      server_count: serverCount,
      export_date: new Date().toISOString()
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `migration-plan-${migrationAnalysis.id}.json`;
    link.click();
    URL.revokeObjectURL(url);

    toast.success('Migration plan exported successfully');
    setShowExportDialog(false);
  };

  const getProviderDisplayName = (provider: string) => {
    return multiCloudApi.getProviderDisplayName(provider);
  };

  const getStepIcon = (stepIndex: number) => {
    if (completedSteps.has(stepIndex)) {
      return <CheckCircleIcon color="success" />;
    }
    if (stepIndex === activeStep) {
      return <ScheduleIcon color="primary" />;
    }
    return <ScheduleIcon color="disabled" />;
  };

  if (workloadsError) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load workloads. Please check your connection and try again.
        </Alert>
        <Button variant="contained" onClick={() => refetchWorkloads()} startIcon={<RefreshIcon />}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <>
      <Helmet>
        <title>Migration Planner - FinOps Platform</title>
        <meta name="description" content="Plan and analyze cloud migrations with cost-benefit analysis" />
      </Helmet>

      <Box sx={{ width: '100%' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              On-Premises → Cloud Migration Planner
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Plan your migration from local servers to AWS cloud with TCO analysis
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Refresh data">
              <IconButton onClick={() => refetchWorkloads()} disabled={workloadsLoading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            {migrationAnalysis && (
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => setShowExportDialog(true)}
              >
                Export Plan
              </Button>
            )}
          </Box>
        </Box>

        <Grid container spacing={3}>
          {/* Configuration Panel */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  On-Premises Details
                </Typography>

                <Box sx={{ mb: 3 }}>
                  {/* Source: On-Premises (fixed) */}
                  <Box sx={{ mb: 2, p: 2, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 2, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <StorageIcon color="warning" />
                    <Box>
                      <Typography variant="caption" color="text.secondary">SOURCE</Typography>
                      <Typography variant="body1" fontWeight={600}>On-Premises Infrastructure</Typography>
                    </Box>
                  </Box>

                  <Box sx={{ textAlign: 'center', my: 1 }}>
                    <Typography variant="body2" color="text.secondary">↓ migrating to ↓</Typography>
                  </Box>

                  {/* Target: Cloud Provider */}
                  <Box sx={{ mb: 2, p: 2, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 2, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <CloudIcon color="primary" />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="caption" color="text.secondary">TARGET</Typography>
                      <FormControl fullWidth size="small" sx={{ mt: 0.5 }}>
                        <Select
                          value={targetProvider}
                          onChange={(e) => setTargetProvider(e.target.value as 'aws' | 'gcp' | 'azure')}
                        >
                          <MenuItem value="aws">
                            Amazon Web Services (AWS)
                          </MenuItem>
                          <MenuItem value="gcp">
                            Google Cloud Platform
                          </MenuItem>
                          <MenuItem value="azure">
                            Microsoft Azure
                          </MenuItem>
                        </Select>
                      </FormControl>
                    </Box>
                  </Box>

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Select On-Prem Workload</InputLabel>
                    <Select
                      value={selectedWorkloadId}
                      onChange={(e) => setSelectedWorkloadId(e.target.value)}
                      disabled={workloadsLoading}
                    >
                      {workloads?.workloads?.map((workload) => (
                        <MenuItem key={workload.id} value={workload.id}>
                          {workload.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <TextField
                    fullWidth
                    type="number"
                    label="Number of Physical Servers"
                    value={serverCount}
                    onChange={(e) => setServerCount(parseInt(e.target.value) || 1)}
                    inputProps={{ min: 1, max: 500 }}
                    sx={{ mb: 2 }}
                  />

                  <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    onClick={handleStartAnalysis}
                    disabled={!selectedWorkloadId || !targetProvider || migrationMutation.isLoading}
                    startIcon={migrationMutation.isLoading ? <CircularProgress size={20} /> : <StartIcon />}
                  >
                    {migrationMutation.isLoading ? 'Analyzing...' : 'Analyze Migration to Cloud'}
                  </Button>
                </Box>

                {/* Migration Steps */}
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>
                  On-Prem → Cloud Process
                </Typography>

                <Stepper activeStep={activeStep} orientation="vertical">
                  {migrationSteps.map((step, index) => (
                    <Step key={step.label} completed={completedSteps.has(index)}>
                      <StepLabel
                        icon={getStepIcon(index)}
                        onClick={() => handleStepClick(index)}
                        sx={{ cursor: completedSteps.has(index) ? 'pointer' : 'default' }}
                        optional={step.optional && (
                          <Typography variant="caption">Optional</Typography>
                        )}
                      >
                        {step.label}
                      </StepLabel>
                      <StepContent>
                        <Typography variant="body2" color="text.secondary">
                          {step.description}
                        </Typography>
                      </StepContent>
                    </Step>
                  ))}
                </Stepper>
              </CardContent>
            </Card>
          </Grid>

          {/* Analysis Results */}
          <Grid item xs={12} lg={8}>
            {!migrationAnalysis ? (
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 8 }}>
                  <TimelineIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No Migration Analysis Available
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Enter your on-premises infrastructure details and select a cloud target to get a full migration plan
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={handleStartAnalysis}
                    disabled={!selectedWorkloadId || !targetProvider}
                  >
                    Start Analysis
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <Box>
                {/* Migration Summary */}
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">Migration Summary</Typography>
                      <Chip
                        label={`On-Premises → ${getProviderDisplayName(migrationAnalysis.target_provider)}`}
                        color="primary"
                        variant="outlined"
                      />
                    </Box>

                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6} md={3}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="primary">
                            {multiCloudApi.formatCurrency(migrationAnalysis.migration_cost)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Migration Cost
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="info.main">
                            {migrationAnalysis.migration_timeline_days}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Days to Complete
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="success.main">
                            {migrationAnalysis.break_even_months || 'N/A'}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Break-even (Months)
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h4" color="warning.main">
                            {migrationAnalysis.risk_assessment.overall_risk_level.toUpperCase()}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Risk Level
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>

                {/* Migration Timeline */}
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Migration Timeline
                    </Typography>
                    <MigrationTimeline analysis={migrationAnalysis} />
                  </CardContent>
                </Card>

                {/* Cost-Benefit Analysis */}
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Cost-Benefit Analysis
                    </Typography>
                    <CostBenefitChart analysis={migrationAnalysis} />
                  </CardContent>
                </Card>

                {/* Risk Assessment */}
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Risk Assessment
                    </Typography>
                    <RiskAssessment risks={migrationAnalysis.risk_assessment} />
                  </CardContent>
                </Card>
              </Box>
            )}
          </Grid>
        </Grid>

        {/* Export Dialog */}
        <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)}>
          <DialogTitle>Export Migration Plan</DialogTitle>
          <DialogContent>
            <Typography variant="body1" gutterBottom>
              Export your migration plan including analysis results, timeline, and recommendations.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              The exported file will contain all migration data in JSON format.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
            <Button onClick={handleExportPlan} variant="contained">
              Export
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </>
  );
};

export default MigrationPlanner;