import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Paper,
  Chip,
  Alert,
  Skeleton
} from '@mui/material';
import { TrendingUp, CheckCircle } from '@mui/icons-material';
import { migrationApi } from '../../services/migrationApi';
import { getProviderIcon, getProviderColor } from '../../utils/providerUtils';

interface ScorePreviewProps {
  projectId: string;
  visible: boolean;
  onScoresUpdate?: (scores: Record<string, number>) => void;
}

interface ScoreData {
  scores: Record<string, number>;
  top_provider: string | null;
  sections_completed: number;
  total_sections: number;
  completion_percentage: number;
  eligible_providers: string[];
}

const ScorePreview: React.FC<ScorePreviewProps> = ({ projectId, visible, onScoresUpdate }) => {
  const [scoreData, setScoreData] = useState<ScoreData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (visible && projectId) {
      loadScores();
    }
  }, [visible, projectId]);

  const loadScores = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/migrations/projects/${projectId}/score-preview`);
      if (!response.ok) {
        throw new Error('Failed to load scores');
      }
      const data = await response.json();
      setScoreData(data);
      
      if (onScoresUpdate && data.scores) {
        onScoresUpdate(data.scores);
      }
    } catch (err: any) {
      console.error('Failed to load score preview:', err);
      const errorMessage = err?.response?.data?.detail || 
                          err?.response?.data?.error?.message || 
                          err?.message || 
                          'Unable to calculate scores. Please complete more sections.';
      setError(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  if (!visible) return null;

  return (
    <Paper 
      sx={{ 
        p: 3, 
        mt: 3, 
        bgcolor: 'info.light',
        border: '2px solid',
        borderColor: 'info.main',
        borderRadius: 2
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <TrendingUp sx={{ mr: 1, color: 'info.dark' }} />
        <Typography variant="h6" sx={{ color: 'info.dark', fontWeight: 'bold' }}>
          📊 Current Recommendation Preview
        </Typography>
      </Box>

      <Typography variant="body2" color="info.dark" sx={{ mb: 3 }}>
        Based on your answers so far. Complete remaining sections to refine your recommendation.
      </Typography>

      {loading ? (
        <Box>
          <Skeleton variant="rectangular" height={40} sx={{ mb: 2 }} />
          <Skeleton variant="rectangular" height={40} sx={{ mb: 2 }} />
          <Skeleton variant="rectangular" height={40} />
        </Box>
      ) : error ? (
        <Alert severity="info" sx={{ mb: 2 }}>
          {error}
        </Alert>
      ) : scoreData && Object.keys(scoreData.scores).length > 0 ? (
        <Box>
          {/* Progress indicator */}
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                Assessment Progress
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {scoreData.sections_completed} of {scoreData.total_sections} sections complete
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={scoreData.completion_percentage}
              sx={{
                height: 8,
                borderRadius: 4,
                bgcolor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  bgcolor: 'success.main'
                }
              }}
            />
          </Box>

          {/* Top provider highlight */}
          {scoreData.top_provider && (
            <Box 
              sx={{ 
                p: 2, 
                mb: 2, 
                bgcolor: 'success.light', 
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'success.main'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <CheckCircle sx={{ mr: 1, color: 'success.dark', fontSize: 20 }} />
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'success.dark' }}>
                  Current Top Recommendation
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="h5" sx={{ mr: 1 }}>
                  {getProviderIcon(scoreData.top_provider)}
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'success.dark' }}>
                  {scoreData.top_provider}
                </Typography>
                <Chip 
                  label={`${scoreData.scores[scoreData.top_provider]}%`}
                  size="small"
                  sx={{ ml: 2, bgcolor: 'success.main', color: 'white', fontWeight: 'bold' }}
                />
              </Box>
            </Box>
          )}

          {/* Score bars */}
          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 'bold' }}>
            Provider Scores
          </Typography>
          {Object.entries(scoreData.scores)
            .sort(([, a], [, b]) => b - a)
            .map(([provider, score]) => (
              <Box key={provider} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="body2" sx={{ mr: 1 }}>
                      {getProviderIcon(provider)}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {provider}
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ fontWeight: 'bold', color: getProviderColor(provider) }}>
                    {Math.round(score)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={score}
                  sx={{
                    height: 10,
                    borderRadius: 5,
                    bgcolor: 'grey.200',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: getProviderColor(provider),
                      borderRadius: 5
                    }
                  }}
                />
              </Box>
            ))}

          {/* Eligible providers count */}
          <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {scoreData.eligible_providers.length} provider{scoreData.eligible_providers.length !== 1 ? 's' : ''} eligible based on your requirements
            </Typography>
          </Box>
        </Box>
      ) : (
        <Alert severity="info">
          Complete more sections to see provider recommendations
        </Alert>
      )}
    </Paper>
  );
};

export default ScorePreview;
