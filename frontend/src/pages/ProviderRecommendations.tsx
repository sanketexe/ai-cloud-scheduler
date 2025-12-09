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
} from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
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
  const [weights, setWeights] = useState({
    service_weight: 0.25,
    cost_weight: 0.25,
    compliance_weight: 0.20,
    performance_weight: 0.20,
    migration_complexity_weight: 0.10,
  });

  useEffect(() => {
    loadRecommendations();
  }, [projectId]);

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
      const data = await migrationApi.generateRecommendations(projectId!);
      setRecommendations(data);
      toast.success('Recommendations generated successfully');
    } catch (error) {
      console.error('Failed to generate recommendations', error);
      toast.error('Failed to generate recommendations');
    } finally {
      setGenerating(false);
    }
  };

  const handleWeightChange = async (newWeights: typeof weights) => {
    setWeights(newWeights);
    setLoading(true);
    try {
      const data = await migrationApi.updateWeights(projectId!, newWeights);
      setRecommendations(data);
      toast.success('Recommendations updated with new weights');
    } catch (error) {
      console.error('Failed to update weights', error);
      toast.error('Failed to update weights');
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
          <Box mb={3} display="flex" gap={2} justifyContent="flex-end">
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
          </Box>

          {showWeightAdjustment && (
            <WeightAdjustmentPanel
              weights={weights}
              onWeightsChange={handleWeightChange}
              onClose={() => setShowWeightAdjustment(false)}
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
  recommendation: ProviderRecommendation;
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
                {rank === 1 && (
                  <Chip
                    label="Recommended"
                    color="primary"
                    size="small"
                    icon={<CheckCircleIcon />}
                  />
                )}
              </Box>
              <Typography variant="body2" color="text.secondary">
                Rank #{rank}
              </Typography>
            </Box>
          </Box>
          <Box textAlign="right">
            <Typography variant="h4" color={`${getScoreColor(recommendation.overall_score)}.main`}>
              {recommendation.overall_score.toFixed(1)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Overall Score
            </Typography>
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
                  <WarningIcon fontSize="small" color="warning" />
                  <Typography variant="body2">{weakness}</Typography>
                </Box>
              ))}
            </Stack>
          </Grid>
        </Grid>

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
          <Chip
            label={getConfidenceLabel(recommendation.confidence_score)}
            color={recommendation.confidence_score >= 0.8 ? 'success' : 'default'}
            size="small"
          />
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

export default ProviderRecommendations;
