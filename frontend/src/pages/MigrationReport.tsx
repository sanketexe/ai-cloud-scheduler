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
  Button,
  Divider,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
} from '@mui/material';
import {
  ExpandMore,
  Download,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Schedule,
  AttachMoney,
  Assessment,
  Lightbulb,
  Print,
} from '@mui/icons-material';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { migrationApi } from '../services/migrationApi';
import toast from 'react-hot-toast';

interface MigrationReport {
  report_id: string;
  project_id: string;
  generated_date: string;
  migration_summary: MigrationSummary;
  cost_analysis: CostAnalysis;
  timeline_analysis: TimelineAnalysis;
  optimization_opportunities: OptimizationOpportunity[];
  lessons_learned: string[];
  recommendations: string[];
}

interface MigrationSummary {
  total_resources_migrated: number;
  migration_duration_days: number;
  success_rate: number;
  phases_completed: number;
  total_phases: number;
}

interface CostAnalysis {
  estimated_cost: number;
  actual_cost: number;
  variance: number;
  variance_percentage: number;
  cost_breakdown: CostBreakdown[];
  monthly_savings: number;
}

interface CostBreakdown {
  category: string;
  estimated: number;
  actual: number;
}

interface TimelineAnalysis {
  estimated_duration: number;
  actual_duration: number;
  variance_days: number;
  phases: PhaseTimeline[];
}

interface PhaseTimeline {
  phase_name: string;
  estimated_days: number;
  actual_days: number;
  status: string;
}

interface OptimizationOpportunity {
  opportunity_id: string;
  title: string;
  description: string;
  potential_savings: number;
  priority: 'high' | 'medium' | 'low';
  implementation_effort: string;
}

const COLORS = ['#2196f3', '#f50057', '#4caf50', '#ff9800', '#9c27b0'];

const MigrationReport: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [report, setReport] = useState<MigrationReport | null>(null);

  useEffect(() => {
    if (projectId) {
      loadReport();
    }
  }, [projectId]);

  const loadReport = async () => {
    try {
      setLoading(true);
      const data = await migrationApi.getMigrationReport(projectId!);
      setReport(data as any);
    } catch (error) {
      toast.error('Failed to load migration report');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `migration-report-${projectId}.json`;
    a.click();
    toast.success('Report downloaded');
  };

  const handlePrint = () => {
    window.print();
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>Loading migration report...</Typography>
        </Box>
      </Container>
    );
  }

  if (!report) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4 }}>
          <Alert severity="info">
            No migration report available yet. Complete the migration to generate a report.
          </Alert>
        </Box>
      </Container>
    );
  }

  const costChartData = report.cost_analysis.cost_breakdown.map((item) => ({
    name: item.category,
    Estimated: item.estimated,
    Actual: item.actual,
  }));

  const timelineChartData = report.timeline_analysis.phases.map((phase) => ({
    name: phase.phase_name,
    Estimated: phase.estimated_days,
    Actual: phase.actual_days,
  }));

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              Migration Final Report
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Generated: {new Date(report.generated_date).toLocaleDateString()}
            </Typography>
          </Box>
          <Box>
            <Button
              variant="outlined"
              startIcon={<Print />}
              onClick={handlePrint}
              sx={{ mr: 2 }}
            >
              Print
            </Button>
            <Button
              variant="contained"
              startIcon={<Download />}
              onClick={handleDownloadReport}
            >
              Download Report
            </Button>
          </Box>
        </Box>

        {/* Executive Summary */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Executive Summary
          </Typography>
          <Divider sx={{ mb: 3 }} />
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <CheckCircle color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Success Rate</Typography>
                  </Box>
                  <Typography variant="h3">
                    {report.migration_summary.success_rate}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={report.migration_summary.success_rate}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Assessment color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Resources</Typography>
                  </Box>
                  <Typography variant="h3">
                    {report.migration_summary.total_resources_migrated}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Migrated
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Schedule color="info" sx={{ mr: 1 }} />
                    <Typography variant="h6">Duration</Typography>
                  </Box>
                  <Typography variant="h3">
                    {report.migration_summary.migration_duration_days}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Days
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <AttachMoney color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Monthly Savings</Typography>
                  </Box>
                  <Typography variant="h3">
                    ${(report.cost_analysis.monthly_savings / 1000).toFixed(1)}K
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Projected
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>

        {/* Cost Analysis */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Cost Analysis
          </Typography>
          <Divider sx={{ mb: 3 }} />

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Cost Comparison
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                      <Typography variant="body2" color="text.secondary">
                        Estimated Cost
                      </Typography>
                      <Typography variant="h5">
                        ${report.cost_analysis.estimated_cost.toLocaleString()}
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                      <Typography variant="body2" color="text.secondary">
                        Actual Cost
                      </Typography>
                      <Typography variant="h5">
                        ${report.cost_analysis.actual_cost.toLocaleString()}
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
                  {report.cost_analysis.variance_percentage > 0 ? (
                    <TrendingUp color="error" />
                  ) : (
                    <TrendingDown color="success" />
                  )}
                  <Typography
                    variant="body1"
                    sx={{ ml: 1 }}
                    color={report.cost_analysis.variance_percentage > 0 ? 'error' : 'success'}
                  >
                    {Math.abs(report.cost_analysis.variance_percentage).toFixed(1)}% variance
                    (${Math.abs(report.cost_analysis.variance).toLocaleString()})
                  </Typography>
                </Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Cost Breakdown
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={costChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Estimated" fill="#2196f3" />
                  <Bar dataKey="Actual" fill="#f50057" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </Paper>

        {/* Timeline Analysis */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Timeline Analysis
          </Typography>
          <Divider sx={{ mb: 3 }} />

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Duration Comparison
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                      <Typography variant="body2" color="text.secondary">
                        Estimated Duration
                      </Typography>
                      <Typography variant="h5">
                        {report.timeline_analysis.estimated_duration} days
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                      <Typography variant="body2" color="text.secondary">
                        Actual Duration
                      </Typography>
                      <Typography variant="h5">
                        {report.timeline_analysis.actual_duration} days
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
                  {report.timeline_analysis.variance_days > 0 ? (
                    <TrendingUp color="warning" />
                  ) : (
                    <TrendingDown color="success" />
                  )}
                  <Typography variant="body1" sx={{ ml: 1 }}>
                    {Math.abs(report.timeline_analysis.variance_days)} days{' '}
                    {report.timeline_analysis.variance_days > 0 ? 'over' : 'under'} estimate
                  </Typography>
                </Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Phase Timeline
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={timelineChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Estimated" fill="#2196f3" />
                  <Bar dataKey="Actual" fill="#4caf50" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </Paper>

        {/* Optimization Opportunities */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Lightbulb color="warning" sx={{ mr: 1 }} />
            <Typography variant="h5">Optimization Opportunities</Typography>
          </Box>
          <Divider sx={{ mb: 3 }} />

          <Grid container spacing={2}>
            {report.optimization_opportunities.map((opp) => (
              <Grid item xs={12} md={6} key={opp.opportunity_id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="h6">{opp.title}</Typography>
                      <Chip
                        label={opp.priority}
                        color={getPriorityColor(opp.priority) as any}
                        size="small"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {opp.description}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                      <Typography variant="body2">
                        Potential Savings: <strong>${opp.potential_savings.toLocaleString()}/mo</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Effort: {opp.implementation_effort}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Lessons Learned */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6">Lessons Learned</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <ul>
              {report.lessons_learned.map((lesson, index) => (
                <li key={index}>
                  <Typography variant="body1" paragraph>
                    {lesson}
                  </Typography>
                </li>
              ))}
            </ul>
          </AccordionDetails>
        </Accordion>

        {/* Recommendations */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6">Recommendations for Future Migrations</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <ul>
              {report.recommendations.map((rec, index) => (
                <li key={index}>
                  <Typography variant="body1" paragraph>
                    {rec}
                  </Typography>
                </li>
              ))}
            </ul>
          </AccordionDetails>
        </Accordion>

        {/* Actions */}
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/dashboard')}
          >
            Go to Dashboard
          </Button>
          <Button
            variant="contained"
            onClick={() => navigate(`/migration/${projectId}/dashboard`)}
          >
            View Migration Dashboard
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default MigrationReport;
