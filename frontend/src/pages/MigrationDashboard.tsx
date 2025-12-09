import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Divider,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import {
  CheckCircle,
  RadioButtonUnchecked,
  Schedule,
  Warning,
  TrendingUp,
  CloudUpload,
  Assessment,
  PlayArrow,
  Pause,
  Edit,
  Refresh,
} from '@mui/icons-material';
import { migrationApi, MigrationPlan, MigrationPhase, MigrationTask } from '../services/migrationApi';
import toast from 'react-hot-toast';

const MigrationDashboard: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [plan, setPlan] = useState<MigrationPlan | null>(null);
  const [activePhase, setActivePhase] = useState<number>(0);

  useEffect(() => {
    if (projectId) {
      loadMigrationPlan();
    }
  }, [projectId]);

  const loadMigrationPlan = async () => {
    try {
      setLoading(true);
      const data = await migrationApi.getMigrationPlan(projectId!);
      setPlan(data);
      
      // Find the first in-progress or not-started phase
      const currentPhaseIndex = data.phases.findIndex(
        (p: MigrationPhase) => p.status === 'in_progress' || p.status === 'not_started'
      );
      setActivePhase(currentPhaseIndex >= 0 ? currentPhaseIndex : 0);
    } catch (error) {
      toast.error('Failed to load migration plan');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handlePhaseStatusUpdate = async (phaseId: string, newStatus: string) => {
    try {
      await migrationApi.updatePhaseStatus(projectId!, phaseId, newStatus);
      toast.success('Phase status updated');
      loadMigrationPlan();
    } catch (error) {
      toast.error('Failed to update phase status');
      console.error(error);
    }
  };

  const getStatusColor = (status: string): 'inherit' | 'grey' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'in_progress':
        return 'primary';
      case 'blocked':
        return 'error';
      default:
        return 'grey';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle />;
      case 'in_progress':
        return <Schedule />;
      case 'blocked':
        return <Warning />;
      default:
        return <RadioButtonUnchecked />;
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>Loading migration dashboard...</Typography>
        </Box>
      </Container>
    );
  }

  if (!plan) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4 }}>
          <Alert severity="warning">
            No migration plan found. Please complete the assessment first.
          </Alert>
          <Button
            variant="contained"
            sx={{ mt: 2 }}
            onClick={() => navigate(`/migration-wizard/${projectId}`)}
          >
            Go to Assessment
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              Migration Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Project ID: {projectId}
            </Typography>
          </Box>
          <Box>
            <Tooltip title="Refresh">
              <IconButton onClick={loadMigrationPlan} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
            <Button
              variant="outlined"
              startIcon={<Edit />}
              sx={{ ml: 2 }}
              onClick={() => navigate(`/migration-wizard/${projectId}`)}
            >
              Edit Assessment
            </Button>
          </Box>
        </Box>

        {/* Overview Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <TrendingUp color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Overall Progress</Typography>
                </Box>
                <Typography variant="h3" sx={{ mb: 1 }}>
                  {plan.overall_progress}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={plan.overall_progress}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Assessment color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Phases</Typography>
                </Box>
                <Typography variant="h3" sx={{ mb: 1 }}>
                  {plan.completed_phases}/{plan.total_phases}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Completed
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Schedule color="info" sx={{ mr: 1 }} />
                  <Typography variant="h6">Est. Completion</Typography>
                </Box>
                <Typography variant="h6" sx={{ mb: 1 }}>
                  {new Date(plan.estimated_completion_date).toLocaleDateString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {Math.ceil(
                    (new Date(plan.estimated_completion_date).getTime() - Date.now()) /
                      (1000 * 60 * 60 * 24)
                  )}{' '}
                  days remaining
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CloudUpload color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">Total Cost</Typography>
                </Box>
                <Typography variant="h3" sx={{ mb: 1 }}>
                  ${(plan.total_cost_estimate / 1000).toFixed(0)}K
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Estimated
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Risks Alert */}
        {plan.risks && plan.risks.length > 0 && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Migration Risks Identified:
            </Typography>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {plan.risks.map((risk, index) => (
                <li key={index}>
                  <Typography variant="body2">{risk}</Typography>
                </li>
              ))}
            </ul>
          </Alert>
        )}

        {/* Main Content */}
        <Grid container spacing={3}>
          {/* Timeline View */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Migration Timeline
              </Typography>
              <Divider sx={{ mb: 3 }} />

              <Timeline position="right">
                {plan.phases.map((phase, index) => (
                  <TimelineItem key={phase.phase_id}>
                    <TimelineOppositeContent color="text.secondary">
                      <Typography variant="body2">
                        {phase.estimated_duration_days} days
                      </Typography>
                      {phase.start_date && (
                        <Typography variant="caption">
                          {new Date(phase.start_date).toLocaleDateString()}
                        </Typography>
                      )}
                    </TimelineOppositeContent>
                    <TimelineSeparator>
                      <TimelineDot color={getStatusColor(phase.status)}>
                        {getStatusIcon(phase.status)}
                      </TimelineDot>
                      {index < plan.phases.length - 1 && <TimelineConnector />}
                    </TimelineSeparator>
                    <TimelineContent>
                      <Paper
                        elevation={3}
                        sx={{
                          p: 2,
                          mb: 2,
                          cursor: 'pointer',
                          '&:hover': { boxShadow: 6 },
                        }}
                        onClick={() => setActivePhase(index)}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="h6">{phase.phase_name}</Typography>
                          <Chip
                            label={phase.status.replace('_', ' ')}
                            color={getStatusColor(phase.status) as any}
                            size="small"
                          />
                        </Box>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          {phase.description}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={phase.progress_percentage}
                          sx={{ height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                          {phase.progress_percentage}% complete
                        </Typography>
                      </Paper>
                    </TimelineContent>
                  </TimelineItem>
                ))}
              </Timeline>
            </Paper>
          </Grid>

          {/* Phase Details */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, position: 'sticky', top: 20 }}>
              <Typography variant="h6" gutterBottom>
                Phase Details
              </Typography>
              <Divider sx={{ mb: 2 }} />

              {plan.phases[activePhase] && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    {plan.phases[activePhase].phase_name}
                  </Typography>
                  <Chip
                    label={plan.phases[activePhase].status.replace('_', ' ')}
                    color={getStatusColor(plan.phases[activePhase].status) as any}
                    size="small"
                    sx={{ mb: 2 }}
                  />

                  <Typography variant="body2" color="text.secondary" paragraph>
                    {plan.phases[activePhase].description}
                  </Typography>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Progress
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={plan.phases[activePhase].progress_percentage}
                      sx={{ height: 8, borderRadius: 4, mt: 1 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {plan.phases[activePhase].progress_percentage}%
                    </Typography>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" gutterBottom>
                    Tasks ({plan.phases[activePhase].tasks?.length || 0})
                  </Typography>
                  {plan.phases[activePhase].tasks?.map((task) => (
                    <Box
                      key={task.task_id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 1,
                        mb: 1,
                        borderRadius: 1,
                        bgcolor: 'background.default',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {getStatusIcon(task.status)}
                        <Typography variant="body2" sx={{ ml: 1 }}>
                          {task.task_name}
                        </Typography>
                      </Box>
                      <Chip
                        label={task.status}
                        size="small"
                        color={getStatusColor(task.status) as any}
                      />
                    </Box>
                  ))}

                  <Divider sx={{ my: 2 }} />

                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    {plan.phases[activePhase].status === 'not_started' && (
                      <Button
                        fullWidth
                        variant="contained"
                        startIcon={<PlayArrow />}
                        onClick={() =>
                          handlePhaseStatusUpdate(
                            plan.phases[activePhase].phase_id,
                            'in_progress'
                          )
                        }
                      >
                        Start Phase
                      </Button>
                    )}
                    {plan.phases[activePhase].status === 'in_progress' && (
                      <>
                        <Button
                          fullWidth
                          variant="outlined"
                          startIcon={<Pause />}
                          onClick={() =>
                            handlePhaseStatusUpdate(
                              plan.phases[activePhase].phase_id,
                              'not_started'
                            )
                          }
                        >
                          Pause
                        </Button>
                        <Button
                          fullWidth
                          variant="contained"
                          color="success"
                          startIcon={<CheckCircle />}
                          onClick={() =>
                            handlePhaseStatusUpdate(
                              plan.phases[activePhase].phase_id,
                              'completed'
                            )
                          }
                        >
                          Complete
                        </Button>
                      </>
                    )}
                  </Box>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default MigrationDashboard;
