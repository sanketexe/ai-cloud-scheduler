import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Alert,
} from '@mui/material';
import { CheckCircle, Warning, Info } from '@mui/icons-material';
import { getProviderIcon, getProviderColor, getProviderDetails } from '../../utils/providerUtils';

interface ProviderComparisonProps {
  scores: Record<string, number>;
  topProvider: string;
}

const ProviderComparison: React.FC<ProviderComparisonProps> = ({ scores, topProvider }) => {
  const sortedProviders = Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .map(([provider]) => provider);

  const capabilities = [
    { name: 'Scalability', key: 'scalability' },
    { name: 'Cost Efficiency', key: 'cost' },
    { name: 'Developer Ecosystem', key: 'ecosystem' },
    { name: 'AI/ML Services', key: 'ai_ml' },
    { name: 'Enterprise Integration', key: 'enterprise' },
    { name: 'Migration Tooling', key: 'migration' },
  ];

  const getCapabilityScore = (provider: string, capability: string): number => {
    // Simplified capability scoring based on provider strengths
    const capabilityScores: Record<string, Record<string, number>> = {
      'AWS': { scalability: 9, cost: 7, ecosystem: 10, ai_ml: 8, enterprise: 7, migration: 9 },
      'Azure': { scalability: 8, cost: 7, ecosystem: 8, ai_ml: 8, enterprise: 10, migration: 9 },
      'GCP': { scalability: 8, cost: 9, ecosystem: 8, ai_ml: 10, enterprise: 7, migration: 7 },
      'IBM': { scalability: 7, cost: 6, ecosystem: 6, ai_ml: 8, enterprise: 9, migration: 7 },
      'Oracle': { scalability: 7, cost: 8, ecosystem: 5, ai_ml: 6, enterprise: 8, migration: 7 },
    };
    return capabilityScores[provider]?.[capability] || 5;
  };

  const renderStars = (score: number) => {
    const fullStars = Math.floor(score / 2);
    const halfStar = score % 2 >= 1;
    return '★'.repeat(fullStars) + (halfStar ? '☆' : '') + '☆'.repeat(5 - fullStars - (halfStar ? 1 : 0));
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ mt: 4, mb: 2 }}>
        Provider Comparison Matrix
      </Typography>

      <TableContainer component={Paper} sx={{ mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow sx={{ bgcolor: 'grey.100' }}>
              <TableCell sx={{ fontWeight: 'bold' }}>Capability</TableCell>
              {sortedProviders.slice(0, 3).map((provider) => (
                <TableCell key={provider} align="center" sx={{ fontWeight: 'bold' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" sx={{ mr: 0.5 }}>
                      {getProviderIcon(provider)}
                    </Typography>
                    <Typography variant="body2">{provider}</Typography>
                  </Box>
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {capabilities.map((capability) => (
              <TableRow key={capability.key}>
                <TableCell>{capability.name}</TableCell>
                {sortedProviders.slice(0, 3).map((provider) => {
                  const score = getCapabilityScore(provider, capability.key);
                  return (
                    <TableCell key={provider} align="center">
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: getProviderColor(provider),
                          fontWeight: provider === topProvider ? 'bold' : 'normal'
                        }}
                      >
                        {renderStars(score)}
                      </Typography>
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
            <TableRow sx={{ bgcolor: 'grey.50' }}>
              <TableCell sx={{ fontWeight: 'bold' }}>Overall Score</TableCell>
              {sortedProviders.slice(0, 3).map((provider) => (
                <TableCell key={provider} align="center">
                  <Chip
                    label={`${scores[provider]}/10`}
                    size="small"
                    sx={{
                      bgcolor: provider === topProvider ? 'success.main' : 'grey.300',
                      color: provider === topProvider ? 'white' : 'text.primary',
                      fontWeight: 'bold'
                    }}
                  />
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>

      {/* Provider Details Cards */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4, mb: 2 }}>
        Detailed Provider Information
      </Typography>

      <Grid container spacing={2}>
        {sortedProviders.slice(0, 3).map((provider) => {
          const details = getProviderDetails(provider);
          if (!details) return null;

          return (
            <Grid item xs={12} md={4} key={provider}>
              <Card 
                sx={{ 
                  height: '100%',
                  border: provider === topProvider ? '2px solid' : '1px solid',
                  borderColor: provider === topProvider ? 'success.main' : 'grey.300'
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h4" sx={{ mr: 1 }}>
                      {details.icon}
                    </Typography>
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                        {provider}
                      </Typography>
                      {provider === topProvider && (
                        <Chip 
                          label="Recommended" 
                          size="small" 
                          color="success"
                          icon={<CheckCircle />}
                          sx={{ mt: 0.5 }}
                        />
                      )}
                    </Box>
                  </Box>

                  <Alert severity="info" icon={<Info />} sx={{ mb: 2, fontSize: '0.875rem' }}>
                    {details.keyStrength}
                  </Alert>

                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    Top Strengths:
                  </Typography>
                  {details.strengths.slice(0, 3).map((strength, idx) => (
                    <Typography key={idx} variant="body2" sx={{ mb: 0.5, fontSize: '0.875rem' }}>
                      • {strength}
                    </Typography>
                  ))}

                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mt: 2, mb: 1 }}>
                    Best For:
                  </Typography>
                  {details.bestFor.slice(0, 2).map((use, idx) => (
                    <Typography key={idx} variant="body2" sx={{ mb: 0.5, fontSize: '0.875rem' }}>
                      • {use}
                    </Typography>
                  ))}

                  {details.watchFor.length > 0 && (
                    <>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mt: 2, mb: 1, color: 'warning.main' }}>
                        <Warning sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                        Watch For:
                      </Typography>
                      {details.watchFor.slice(0, 2).map((warning, idx) => (
                        <Typography key={idx} variant="body2" sx={{ mb: 0.5, fontSize: '0.875rem', color: 'text.secondary' }}>
                          • {warning}
                        </Typography>
                      ))}
                    </>
                  )}
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default ProviderComparison;
