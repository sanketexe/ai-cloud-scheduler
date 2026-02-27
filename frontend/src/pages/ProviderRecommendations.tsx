import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Divider,
  Alert,
  CircularProgress,
  Stack,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Checkbox,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import HistoryIcon from '@mui/icons-material/History';
import EditIcon from '@mui/icons-material/Edit';
import { migrationApi, ProviderRecommendation } from '../services/migrationApi';

const ProviderRecommendations: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [recommendations, setRecommendations] = useState<ProviderRecommendation[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [showWeightAdjustment, setShowWeightAdjustment] = useState(false);
  const [showExportOptions, setShowExportOptions] = useState(false);
  const [assessmentData, setAssessmentData] = useState<any>(null);
  const [recommendationMetadata, setRecommendationMetadata] = useState<any>(null);
  const [showScenarioManager, setShowScenarioManager] = useState(false);
  const [scenarioHistory, setScenarioHistory] = useState<any[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [scenarioComparison, setScenarioComparison] = useState<any>(null);
  const [showAssessmentModifier, setShowAssessmentModifier] = useState(false);
  const [weights, setWeights] = useState({
    service_weight: 0.25,
    cost_weight: 0.25,
    compliance_weight: 0.20,
    performance_weight: 0.20,
    migration_complexity_weight: 0.10,
  });

  useEffect(() => {
    loadRecommendations();
    loadAssessmentData();
  }, [projectId]);

  const loadAssessmentData = async () => {
    try {
      // Load assessment data to show context and enable scenario comparison
      const [orgData, workloadData, perfData, complianceData, budgetData, techData] = await Promise.allSettled([
        migrationApi.getProject(projectId!).then(p => ({ organization: p })).catch(() => null),
        // Note: These endpoints might not exist yet, but we'll handle gracefully
        fetch(`/api/v1/api/migrations/${projectId}/assessment/organization`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/api/v1/api/migrations/${projectId}/workloads`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/api/v1/api/migrations/${projectId}/performance-requirements`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/api/v1/api/migrations/${projectId}/compliance-requirements`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/api/v1/api/migrations/${projectId}/budget-constraints`).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      
      const assessment = {
        organization: orgData.status === 'fulfilled' ? orgData.value : null,
        workload: workloadData.status === 'fulfilled' ? workloadData.value : null,
        performance: perfData.status === 'fulfilled' ? perfData.value : null,
        compliance: complianceData.status === 'fulfilled' ? complianceData.value : null,
        budget: budgetData.status === 'fulfilled' ? budgetData.value : null,
        technical: techData.status === 'fulfilled' ? techData.value : null,
      };
      
      setAssessmentData(assessment);
    } catch (error) {
      console.error('Failed to load assessment data:', error);
    }
  };

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const data = await migrationApi.getRecommendations(projectId!);
      if (data && data.length > 0) {
        setRecommendations(data);
      } else {
        // No recommendations yet, generate them
        await generateRecommendations();
      }
    } catch (error) {
      console.error('Failed to load recommendations', error);
      // Try to generate if not found
      await generateRecommendations();
    } finally {
      setLoading(false);
    }
  };

  const generateRecommendations = async () => {
    setGenerating(true);
    try {
      const response = await migrationApi.generateRecommendations(projectId!);
      
      // Handle both old and new response formats
      if (Array.isArray(response)) {
        setRecommendations(response);
      } else if (response && typeof response === 'object' && 'recommendations' in response) {
        setRecommendations((response as any).recommendations);
        setRecommendationMetadata((response as any).metadata);
      } else {
        setRecommendations(response as ProviderRecommendation[]);
      }
      
      toast.success('Recommendations generated successfully');
    } catch (error: any) {
      console.error('Failed to generate recommendations', error);
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.error?.message || 
                          error.message || 
                          'Failed to generate recommendations';
      toast.error(errorMessage);
    } finally {
      setGenerating(false);
    }
  };

  const handleExportPDF = async () => {
    try {
      // Generate PDF export (this would typically call a backend service)
      const exportData = {
        projectId,
        recommendations,
        assessmentData,
        metadata: recommendationMetadata,
        exportDate: new Date().toISOString(),
      };
      
      // For now, we'll create a downloadable JSON file
      // In a real implementation, this would generate a PDF
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `migration-recommendations-${projectId}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
      
      toast.success('Recommendations exported successfully');
    } catch (error) {
      console.error('Failed to export recommendations:', error);
      toast.error('Failed to export recommendations');
    }
  };

  const handleShareLink = async () => {
    try {
      const shareUrl = `${window.location.origin}/migration/${projectId}/recommendations`;
      await navigator.clipboard.writeText(shareUrl);
      toast.success('Share link copied to clipboard');
    } catch (error) {
      console.error('Failed to copy share link:', error);
      toast.error('Failed to copy share link');
    }
  };

  const handleRegenerateWithNewWeights = async (newWeights: typeof weights) => {
    setWeights(newWeights);
    setLoading(true);
    try {
      const data = await migrationApi.updateWeights(projectId!, newWeights);
      setRecommendations(data);
      toast.success('Recommendations updated with new weights');
    } catch (error) {
      console.error('Failed to update weights', error);
      // Fallback: regenerate recommendations
      await generateRecommendations();
    } finally {
      setLoading(false);
    }
  };

  const handleProviderSelection = (provider: string) => {
    setSelectedProvider(provider);
  };

  const handleProceedToPlanning = async () => {
    if (!selectedProvider) {
      toast.error('Please select a provider first');
      return;
    }
    
    try {
      await migrationApi.generateMigrationPlan(projectId!, selectedProvider);
      toast.success('Migration plan generated');
      navigate(`/migrations/${projectId}/plan`);
    } catch (error) {
      console.error('Failed to generate migration plan', error);
      toast.error('Failed to generate migration plan');
    }
  };

  // Scenario Management Functions
  const loadScenarioHistory = async () => {
    try {
      const data = await migrationApi.getScenarioHistory(projectId!);
      setScenarioHistory(data.scenarios);
    } catch (error) {
      console.error('Failed to load scenario history:', error);
    }
  };

  const handleActivateScenario = async (scenarioId: string) => {
    try {
      await migrationApi.activateScenario(projectId!, scenarioId);
      toast.success('Scenario activated successfully');
      await loadRecommendations(); // Reload current recommendations
      await loadScenarioHistory(); // Refresh scenario list
    } catch (error) {
      console.error('Failed to activate scenario:', error);
      toast.error('Failed to activate scenario');
    }
  };

  const handleCompareScenarios = async () => {
    if (selectedScenarios.length < 2) {
      toast.error('Please select at least 2 scenarios to compare');
      return;
    }

    try {
      const comparison = await migrationApi.compareScenarios(projectId!, selectedScenarios);
      setScenarioComparison(comparison);
    } catch (error) {
      console.error('Failed to compare scenarios:', error);
      toast.error('Failed to compare scenarios');
    }
  };

  const handleModifyAssessment = async (modifications: any) => {
    try {
      const result = await migrationApi.modifyAssessmentAndRegenerate(projectId!, modifications);
      setRecommendations(result.recommendations);
      toast.success(`Assessment modified: ${result.modifications_applied.join(', ')}`);
      await loadScenarioHistory(); // Refresh scenario list
      setShowAssessmentModifier(false);
    } catch (error) {
      console.error('Failed to modify assessment:', error);
      toast.error('Failed to modify assessment and regenerate recommendations');
    }
  };

  const openScenarioManager = async () => {
    await loadScenarioHistory();
    setShowScenarioManager(true);
  };

  const getProviderLogo = (provider: string) => {
    const logos: Record<string, string> = {
      AWS: '☁️',
      GCP: '🌐',
      Azure: '🔷',
    };
    return logos[provider] || '☁️';
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  if (loading && recommendations.length === 0) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          Cloud Provider Recommendations
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Based on your requirements, we've analyzed and ranked cloud providers for your migration.
        </Typography>
      </Box>

      {generating && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Box display="flex" alignItems="center" gap={2}>
            <CircularProgress size={20} />
            <Typography>Generating recommendations based on your requirements...</Typography>
          </Box>
        </Alert>
      )}

      {recommendations.length === 0 && !generating && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          No recommendations available. Please complete the assessment first.
        </Alert>
      )}

      {recommendations.length > 0 && (
        <>
          <Box mb={3} display="flex" gap={2} justifyContent="space-between" alignItems="center">
            <Box>
              {recommendationMetadata && (
                <Typography variant="body2" color="text.secondary">
                  Generated {new Date(recommendationMetadata.generated_at).toLocaleString()} • 
                  {recommendationMetadata.total_recommendations} providers analyzed
                </Typography>
              )}
            </Box>
            <Box display="flex" gap={1}>
              <Button
                variant="outlined"
                startIcon={<TuneIcon />}
                onClick={() => setShowWeightAdjustment(!showWeightAdjustment)}
              >
                Adjust Weights
              </Button>
              <Button
                variant="outlined"
                startIcon={<CompareArrowsIcon />}
                onClick={() => setShowComparison(true)}
              >
                Compare Providers
              </Button>
              <Button
                variant="outlined"
                onClick={openScenarioManager}
              >
                Manage Scenarios
              </Button>
              <Button
                variant="outlined"
                onClick={() => setShowAssessmentModifier(true)}
              >
                Modify Assessment
              </Button>
              <Button
                variant="outlined"
                onClick={() => setShowExportOptions(!showExportOptions)}
              >
                Export & Share
              </Button>
            </Box>
          </Box>

          {/* Constraint Violations Warning */}
          {recommendations.some((rec: any) => rec.is_viable === false || (rec.constraint_violations && rec.constraint_violations > 0)) && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              <Typography variant="body2">
                <strong>Requirement Constraints Detected:</strong> Some providers may not fully meet your specified requirements. 
                Providers marked as "Not Viable" have critical constraint violations, while others may have warnings that should be considered.
                {recommendations.some((rec: any) => rec.is_viable === false) && (
                  <span> Consider adjusting your requirements if no viable options meet your needs.</span>
                )}
              </Typography>
            </Alert>
          )}

          {showExportOptions && (
            <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.paper' }}>
              <Box display="flex" gap={2} alignItems="center">
                <Typography variant="subtitle2">Export Options:</Typography>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleExportPDF}
                >
                  Download Report
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleShareLink}
                >
                  Copy Share Link
                </Button>
                <Button
                  size="small"
                  variant="text"
                  onClick={() => setShowExportOptions(false)}
                >
                  Close
                </Button>
              </Box>
            </Paper>
          )}

          {showWeightAdjustment && (
            <WeightAdjustmentPanel
              weights={weights}
              onWeightsChange={handleRegenerateWithNewWeights}
              onClose={() => setShowWeightAdjustment(false)}
            />
          )}

          {showScenarioManager && (
            <ScenarioManagerPanel
              projectId={projectId!}
              scenarioHistory={scenarioHistory}
              onClose={() => setShowScenarioManager(false)}
              onActivateScenario={handleActivateScenario}
              onCompareScenarios={handleCompareScenarios}
              selectedScenarios={selectedScenarios}
              setSelectedScenarios={setSelectedScenarios}
            />
          )}

          {showAssessmentModifier && (
            <AssessmentModifierPanel
              projectId={projectId!}
              assessmentData={assessmentData}
              onClose={() => setShowAssessmentModifier(false)}
              onModifyAssessment={handleModifyAssessment}
            />
          )}

          {scenarioComparison && (
            <ScenarioComparisonModal
              comparison={scenarioComparison}
              onClose={() => setScenarioComparison(null)}
            />
          )}

          <Grid container spacing={3}>
            {recommendations.map((rec, index) => (
              <Grid item xs={12} key={rec.provider}>
                <ProviderRecommendationCard
                  recommendation={rec}
                  rank={index + 1}
                  isSelected={selectedProvider === rec.provider}
                  onSelect={() => handleProviderSelection(rec.provider)}
                  getProviderLogo={getProviderLogo}
                  getScoreColor={getScoreColor}
                  getConfidenceLabel={getConfidenceLabel}
                />
              </Grid>
            ))}
          </Grid>

          {selectedProvider && (
            <Box mt={4} display="flex" justifyContent="center">
              <Button
                variant="contained"
                size="large"
                endIcon={<ArrowForwardIcon />}
                onClick={handleProceedToPlanning}
              >
                Proceed to Migration Planning
              </Button>
            </Box>
          )}

          {showComparison && (
            <ProviderComparisonModal
              projectId={projectId!}
              recommendations={recommendations}
              onClose={() => setShowComparison(false)}
            />
          )}
        </>
      )}
    </Container>
  );
};

interface ProviderRecommendationCardProps {
  recommendation: ProviderRecommendation & {
    is_viable?: boolean;
    constraint_violations?: number;
    critical_violations?: number;
    constraint_summary?: any;
  };
  rank: number;
  isSelected: boolean;
  onSelect: () => void;
  getProviderLogo: (provider: string) => string;
  getScoreColor: (score: number) => 'success' | 'warning' | 'error';
  getConfidenceLabel: (confidence: number) => string;
}

const ProviderRecommendationCard: React.FC<ProviderRecommendationCardProps> = ({
  recommendation,
  rank,
  isSelected,
  onSelect,
  getProviderLogo,
  getScoreColor,
  getConfidenceLabel,
}) => {
  return (
    <Card
      sx={{
        border: isSelected ? 3 : 1,
        borderColor: isSelected ? 'primary.main' : 'divider',
        cursor: 'pointer',
        transition: 'all 0.3s',
        '&:hover': {
          boxShadow: 6,
          transform: 'translateY(-2px)',
        },
      }}
      onClick={onSelect}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h3">{getProviderLogo(recommendation.provider)}</Typography>
            <Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Typography variant="h5">{recommendation.provider}</Typography>
                {rank === 1 && recommendation.is_viable !== false && (
                  <Chip
                    label="Recommended"
                    color="primary"
                    size="small"
                    icon={<CheckCircleIcon />}
                  />
                )}
                {recommendation.is_viable === false && (
                  <Chip
                    label="Has Constraints"
                    color="error"
                    size="small"
                    icon={<WarningIcon />}
                  />
                )}
                {recommendation.constraint_violations && recommendation.constraint_violations > 0 && recommendation.is_viable !== false && (
                  <Chip
                    label={`${recommendation.constraint_violations} Warnings`}
                    color="warning"
                    size="small"
                    icon={<InfoIcon />}
                  />
                )}
              </Box>
              <Typography variant="body2" color="text.secondary">
                Rank #{rank}
                {recommendation.critical_violations && recommendation.critical_violations > 0 && (
                  <span style={{ color: 'red', marginLeft: 8 }}>
                    • {recommendation.critical_violations} critical constraint{recommendation.critical_violations > 1 ? 's' : ''}
                  </span>
                )}
              </Typography>
            </Box>
          </Box>
          <Box textAlign="right">
            <Typography 
              variant="h4" 
              color={recommendation.is_viable === false ? 'error.main' : `${getScoreColor(recommendation.overall_score)}.main`}
            >
              {recommendation.overall_score.toFixed(1)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Overall Score
            </Typography>
            {recommendation.is_viable === false && (
              <Typography variant="caption" color="error" display="block">
                Not Viable
              </Typography>
            )}
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={2} mb={2}>
          <Grid item xs={12} sm={6} md={2.4}>
            <ScoreItem
              label="Service Match"
              score={recommendation.service_score}
              getScoreColor={getScoreColor}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2.4}>
            <ScoreItem
              label="Cost"
              score={recommendation.cost_score}
              getScoreColor={getScoreColor}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2.4}>
            <ScoreItem
              label="Compliance"
              score={recommendation.compliance_score}
              getScoreColor={getScoreColor}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2.4}>
            <ScoreItem
              label="Performance"
              score={recommendation.performance_score}
              getScoreColor={getScoreColor}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2.4}>
            <ScoreItem
              label="Migration"
              score={recommendation.migration_complexity_score}
              getScoreColor={getScoreColor}
            />
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom color="success.main">
              Strengths
            </Typography>
            <Stack spacing={0.5}>
              {recommendation.strengths.map((strength, idx) => (
                <Box key={idx} display="flex" alignItems="flex-start" gap={1}>
                  <CheckCircleIcon fontSize="small" color="success" />
                  <Typography variant="body2">{strength}</Typography>
                </Box>
              ))}
            </Stack>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom color="warning.main">
              Considerations
            </Typography>
            <Stack spacing={0.5}>
              {recommendation.weaknesses.map((weakness, idx) => (
                <Box key={idx} display="flex" alignItems="flex-start" gap={1}>
                  {weakness.startsWith('❌') ? (
                    <WarningIcon fontSize="small" color="error" />
                  ) : weakness.startsWith('⚠️') ? (
                    <WarningIcon fontSize="small" color="warning" />
                  ) : (
                    <WarningIcon fontSize="small" color="warning" />
                  )}
                  <Typography 
                    variant="body2" 
                    color={weakness.startsWith('❌') ? 'error.main' : 'text.primary'}
                  >
                    {weakness}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Grid>
        </Grid>

        {/* Constraint Violations Section */}
        {recommendation.constraint_summary && recommendation.constraint_summary.total_violations > 0 && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box>
              <Typography variant="subtitle2" gutterBottom color="error.main">
                Requirement Constraints
              </Typography>
              <Alert 
                severity={recommendation.is_viable === false ? "error" : "warning"} 
                sx={{ mb: 1 }}
              >
                <Typography variant="body2">
                  {recommendation.is_viable === false 
                    ? `This provider cannot meet ${recommendation.critical_violations || 0} critical requirement${(recommendation.critical_violations || 0) > 1 ? 's' : ''}.`
                    : `This provider has ${recommendation.constraint_summary.warning_violations} warning${recommendation.constraint_summary.warning_violations > 1 ? 's' : ''} but can still be considered.`
                  }
                </Typography>
              </Alert>
              
              {Object.entries(recommendation.constraint_summary.violations_by_type).map(([type, violations]: [string, any]) => (
                <Box key={type} sx={{ mb: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                    {type.replace('_', ' ')} Issues:
                  </Typography>
                  {violations.slice(0, 2).map((violation: any, idx: number) => (
                    <Typography 
                      key={idx} 
                      variant="body2" 
                      color={violation.severity === 'critical' ? 'error.main' : 'warning.main'}
                      sx={{ ml: 1, fontSize: '0.875rem' }}
                    >
                      • {violation.message}
                    </Typography>
                  ))}
                </Box>
              ))}
            </Box>
          </>
        )}

        <Divider sx={{ my: 2 }} />

        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h6" color="primary">
              ${recommendation.estimated_monthly_cost.toLocaleString()}/month
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Estimated Cost
            </Typography>
          </Box>
          <Box display="flex" gap={1} alignItems="center">
            <Chip
              label={getConfidenceLabel(recommendation.confidence_score)}
              color={recommendation.confidence_score >= 0.8 ? 'success' : 'default'}
              size="small"
            />
            <Chip
              label="3-6 months"
              size="small"
              variant="outlined"
            />
            <Tooltip title="Migration complexity based on your requirements">
              <Chip
                label="Medium"
                color="warning"
                size="small"
                variant="outlined"
              />
            </Tooltip>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

interface ScoreItemProps {
  label: string;
  score: number;
  getScoreColor: (score: number) => 'success' | 'warning' | 'error';
}

const ScoreItem: React.FC<ScoreItemProps> = ({ label, score, getScoreColor }) => {
  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="caption" fontWeight="bold">
          {score.toFixed(0)}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={score}
        color={getScoreColor(score)}
        sx={{ height: 6, borderRadius: 3 }}
      />
    </Box>
  );
};

// Weight Adjustment Panel Component
interface WeightAdjustmentPanelProps {
  weights: {
    service_weight: number;
    cost_weight: number;
    compliance_weight: number;
    performance_weight: number;
    migration_complexity_weight: number;
  };
  onWeightsChange: (weights: any) => void;
  onClose: () => void;
}

const WeightAdjustmentPanel: React.FC<WeightAdjustmentPanelProps> = ({
  weights,
  onWeightsChange,
  onClose,
}) => {
  const [localWeights, setLocalWeights] = useState(weights);

  const handleSliderChange = (key: string, value: number) => {
    setLocalWeights((prev) => ({
      ...prev,
      [key]: value / 100,
    }));
  };

  const handleApply = () => {
    // Normalize weights to sum to 1
    const total = Object.values(localWeights).reduce((sum, val) => sum + val, 0);
    const normalized = Object.entries(localWeights).reduce(
      (acc, [key, val]) => ({
        ...acc,
        [key]: val / total,
      }),
      {} as typeof localWeights
    );
    onWeightsChange(normalized);
    onClose();
  };

  const weightLabels = {
    service_weight: 'Service Availability',
    cost_weight: 'Cost Optimization',
    compliance_weight: 'Compliance Fit',
    performance_weight: 'Performance',
    migration_complexity_weight: 'Migration Ease',
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Adjust Recommendation Weights</Typography>
        <Tooltip title="Adjust the importance of each factor in the recommendation algorithm">
          <IconButton size="small">
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(localWeights).map(([key, value]) => (
          <Grid item xs={12} key={key}>
            <Typography variant="body2" gutterBottom>
              {weightLabels[key as keyof typeof weightLabels]}
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <Box flex={1}>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={value * 100}
                  onChange={(e) => handleSliderChange(key, Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </Box>
              <Typography variant="body2" sx={{ minWidth: 40, textAlign: 'right' }}>
                {(value * 100).toFixed(0)}%
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>

      <Box mt={3} display="flex" gap={2} justifyContent="flex-end">
        <Button variant="outlined" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="contained" onClick={handleApply}>
          Apply Weights
        </Button>
      </Box>
    </Paper>
  );
};

// Provider Comparison Modal Component
interface ProviderComparisonModalProps {
  projectId: string;
  recommendations: ProviderRecommendation[];
  onClose: () => void;
}

const ProviderComparisonModal: React.FC<ProviderComparisonModalProps> = ({
  projectId,
  recommendations,
  onClose,
}) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        bgcolor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1300,
        p: 2,
      }}
      onClick={onClose}
    >
      <Paper
        sx={{
          maxWidth: 1200,
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          p: 3,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h5">Provider Comparison</Typography>
          <Button onClick={onClose}>Close</Button>
        </Box>

        <Box sx={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '12px', borderBottom: '2px solid #ddd' }}>
                  Criteria
                </th>
                {recommendations.map((rec) => (
                  <th
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '2px solid #ddd' }}
                  >
                    {rec.provider}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd', fontWeight: 'bold' }}>
                  Overall Score
                </td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    <Typography variant="h6" color="primary">
                      {rec.overall_score.toFixed(1)}
                    </Typography>
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Service Match</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {rec.service_score.toFixed(1)}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Cost Score</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {rec.cost_score.toFixed(1)}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Compliance</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {rec.compliance_score.toFixed(1)}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Performance</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {rec.performance_score.toFixed(1)}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Migration Ease</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {rec.migration_complexity_score.toFixed(1)}
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd', fontWeight: 'bold' }}>
                  Estimated Monthly Cost
                </td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    <Typography variant="body1" color="primary" fontWeight="bold">
                      ${rec.estimated_monthly_cost.toLocaleString()}
                    </Typography>
                  </td>
                ))}
              </tr>
              <tr>
                <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>Confidence</td>
                {recommendations.map((rec) => (
                  <td
                    key={rec.provider}
                    style={{ textAlign: 'center', padding: '12px', borderBottom: '1px solid #ddd' }}
                  >
                    {(rec.confidence_score * 100).toFixed(0)}%
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </Box>
      </Paper>
    </Box>
  );
};

// Scenario Manager Panel Component
interface ScenarioManagerPanelProps {
  projectId: string;
  scenarioHistory: any[];
  onClose: () => void;
  onActivateScenario: (scenarioId: string) => void;
  onCompareScenarios: () => void;
  selectedScenarios: string[];
  setSelectedScenarios: (scenarios: string[]) => void;
}

const ScenarioManagerPanel: React.FC<ScenarioManagerPanelProps> = ({
  projectId,
  scenarioHistory,
  onClose,
  onActivateScenario,
  onCompareScenarios,
  selectedScenarios,
  setSelectedScenarios,
}) => {
  const handleScenarioToggle = (scenarioId: string) => {
    setSelectedScenarios(
      selectedScenarios.includes(scenarioId)
        ? selectedScenarios.filter(id => id !== scenarioId)
        : [...selectedScenarios, scenarioId]
    );
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Scenario Management</Typography>
        <Button onClick={onClose}>Close</Button>
      </Box>

      {scenarioHistory.length === 0 ? (
        <Alert severity="info">
          No scenarios available. Generate recommendations with different weights or modify your assessment to create scenarios.
        </Alert>
      ) : (
        <>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Select scenarios to compare or activate a previous scenario:
          </Typography>
          
          <List>
            {scenarioHistory.map((scenario) => (
              <ListItem key={scenario.scenario_id} divider>
                <Checkbox
                  checked={selectedScenarios.includes(scenario.scenario_id)}
                  onChange={() => handleScenarioToggle(scenario.scenario_id)}
                />
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="subtitle2">
                        {scenario.description}
                      </Typography>
                      <Chip
                        label={scenario.type.replace('_', ' ')}
                        size="small"
                        color={scenario.type === 'weight_adjustment' ? 'primary' : 'secondary'}
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(scenario.timestamp).toLocaleString()} • 
                        Top: {scenario.top_provider} ({scenario.top_score?.toFixed(1)})
                      </Typography>
                      {scenario.weights && (
                        <Box mt={0.5}>
                          <Typography variant="caption" color="text.secondary">
                            Weights: Service {(scenario.weights.service_weight * 100).toFixed(0)}%, 
                            Cost {(scenario.weights.cost_weight * 100).toFixed(0)}%, 
                            Compliance {(scenario.weights.compliance_weight * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  }
                />
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => onActivateScenario(scenario.scenario_id)}
                >
                  Activate
                </Button>
              </ListItem>
            ))}
          </List>

          <Box mt={2} display="flex" gap={2}>
            <Button
              variant="contained"
              disabled={selectedScenarios.length < 2}
              onClick={onCompareScenarios}
            >
              Compare Selected ({selectedScenarios.length})
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
              Select at least 2 scenarios to compare
            </Typography>
          </Box>
        </>
      )}
    </Paper>
  );
};

// Assessment Modifier Panel Component
interface AssessmentModifierPanelProps {
  projectId: string;
  assessmentData: any;
  onClose: () => void;
  onModifyAssessment: (modifications: any) => void;
}

const AssessmentModifierPanel: React.FC<AssessmentModifierPanelProps> = ({
  projectId,
  assessmentData,
  onClose,
  onModifyAssessment,
}) => {
  const [modifications, setModifications] = useState<any>({});
  const [modificationSection, setModificationSection] = useState<string>('organization');

  const handleFieldChange = (section: string, field: string, value: any) => {
    setModifications((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  const handleSubmit = () => {
    if (Object.keys(modifications).length === 0) {
      toast.error('No modifications made');
      return;
    }
    onModifyAssessment(modifications);
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Modify Assessment Data</Typography>
        <Button onClick={onClose}>Close</Button>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Modify your assessment data to explore different scenarios. Changes will generate new recommendations and create a new scenario for comparison.
      </Alert>

      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel>Section to Modify</InputLabel>
        <Select
          value={modificationSection}
          onChange={(e) => setModificationSection(e.target.value)}
        >
          <MenuItem value="organization">Organization Profile</MenuItem>
          <MenuItem value="workload">Workload Profile</MenuItem>
          <MenuItem value="requirements">Requirements</MenuItem>
        </Select>
      </FormControl>

      {modificationSection === 'organization' && assessmentData?.organization && (
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Company Size</InputLabel>
              <Select
                value={modifications.organization?.size || assessmentData.organization.size || ''}
                onChange={(e) => handleFieldChange('organization', 'size', e.target.value)}
              >
                <MenuItem value="SMALL">Small (1-50 employees)</MenuItem>
                <MenuItem value="MEDIUM">Medium (51-500 employees)</MenuItem>
                <MenuItem value="LARGE">Large (501-5000 employees)</MenuItem>
                <MenuItem value="ENTERPRISE">Enterprise (5000+ employees)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Cloud Experience</InputLabel>
              <Select
                value={modifications.organization?.experience || assessmentData.organization.experience || ''}
                onChange={(e) => handleFieldChange('organization', 'experience', e.target.value)}
              >
                <MenuItem value="BEGINNER">Beginner</MenuItem>
                <MenuItem value="INTERMEDIATE">Intermediate</MenuItem>
                <MenuItem value="ADVANCED">Advanced</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      )}

      {modificationSection === 'workload' && assessmentData?.workload && (
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Total Compute Cores"
              type="number"
              value={modifications.workload?.total_compute_cores || assessmentData.workload.total_compute_cores || ''}
              onChange={(e) => handleFieldChange('workload', 'total_compute_cores', parseInt(e.target.value))}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Total Memory (GB)"
              type="number"
              value={modifications.workload?.total_memory_gb || assessmentData.workload.total_memory_gb || ''}
              onChange={(e) => handleFieldChange('workload', 'total_memory_gb', parseInt(e.target.value))}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Total Storage (TB)"
              type="number"
              value={modifications.workload?.total_storage_tb || assessmentData.workload.total_storage_tb || ''}
              onChange={(e) => handleFieldChange('workload', 'total_storage_tb', parseFloat(e.target.value))}
            />
          </Grid>
        </Grid>
      )}

      {modificationSection === 'requirements' && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>Budget Constraints</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Target Monthly Cost ($)"
              type="number"
              value={modifications.requirements?.budget?.target_monthly_cost || assessmentData?.budget?.target_monthly_cost || ''}
              onChange={(e) => handleFieldChange('requirements', 'budget', { 
                ...modifications.requirements?.budget, 
                target_monthly_cost: parseFloat(e.target.value) 
              })}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Cost Priority</InputLabel>
              <Select
                value={modifications.requirements?.budget?.cost_priority || assessmentData?.budget?.cost_priority || ''}
                onChange={(e) => handleFieldChange('requirements', 'budget', { 
                  ...modifications.requirements?.budget, 
                  cost_priority: e.target.value 
                })}
              >
                <MenuItem value="LOW">Low</MenuItem>
                <MenuItem value="MEDIUM">Medium</MenuItem>
                <MenuItem value="HIGH">High</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      )}

      <Box mt={3} display="flex" gap={2} justifyContent="flex-end">
        <Button variant="outlined" onClick={onClose}>
          Cancel
        </Button>
        <Button 
          variant="contained" 
          onClick={handleSubmit}
          disabled={Object.keys(modifications).length === 0}
        >
          Apply Changes & Regenerate
        </Button>
      </Box>
    </Paper>
  );
};

// Scenario Comparison Modal Component
interface ScenarioComparisonModalProps {
  comparison: any;
  onClose: () => void;
}

const ScenarioComparisonModal: React.FC<ScenarioComparisonModalProps> = ({
  comparison,
  onClose,
}) => {
  return (
    <Dialog open={true} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Scenario Comparison</Typography>
          <Button onClick={onClose}>Close</Button>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Comparing {comparison.scenarios.length} scenarios
        </Typography>

        {/* Scenario Overview */}
        <Box mb={3}>
          <Typography variant="subtitle1" gutterBottom>Scenarios</Typography>
          <Grid container spacing={2}>
            {comparison.scenarios.map((scenario: any, index: number) => (
              <Grid item xs={12} md={6} key={scenario.scenario_id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">
                      Scenario {index + 1}: {scenario.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(scenario.timestamp).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Top Provider: {scenario.recommendations[0]?.provider} 
                      ({scenario.recommendations[0]?.overall_score.toFixed(1)})
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Provider Rankings Comparison */}
        <Box mb={3}>
          <Typography variant="subtitle1" gutterBottom>Provider Rankings</Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Provider</TableCell>
                  {comparison.scenarios.map((scenario: any, index: number) => (
                    <TableCell key={scenario.scenario_id} align="center">
                      Scenario {index + 1}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(comparison.provider_rankings).map(([provider, rankings]: [string, any]) => (
                  <TableRow key={provider}>
                    <TableCell component="th" scope="row">
                      {provider}
                    </TableCell>
                    {rankings.map((ranking: any) => (
                      <TableCell key={ranking.scenario_id} align="center">
                        #{ranking.rank} ({ranking.score.toFixed(1)})
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>

        {/* Score Changes (for 2-scenario comparison) */}
        {comparison.score_changes && Object.keys(comparison.score_changes).length > 0 && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>Score Changes</Typography>
            <Grid container spacing={2}>
              {Object.entries(comparison.score_changes).map(([provider, change]: [string, any]) => (
                <Grid item xs={12} sm={4} key={provider}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">{provider}</Typography>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography 
                          variant="h6" 
                          color={change.direction === 'increase' ? 'success.main' : 
                                change.direction === 'decrease' ? 'error.main' : 'text.secondary'}
                        >
                          {change.direction === 'increase' ? '+' : change.direction === 'decrease' ? '-' : ''}
                          {change.magnitude.toFixed(1)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {change.direction === 'no_change' ? 'No change' : 
                           change.direction === 'increase' ? 'Improved' : 'Decreased'}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default ProviderRecommendations;
