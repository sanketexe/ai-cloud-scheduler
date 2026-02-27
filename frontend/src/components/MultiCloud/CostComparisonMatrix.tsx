/**
 * Cost Comparison Matrix Component
 * 
 * Displays side-by-side cost comparison across cloud providers
 * with detailed breakdown by service category.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Tooltip,
  IconButton,
  Collapse
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

import { multiCloudApi, CostComparison, WorkloadListResponse } from '../../services/multiCloudApi';

interface CostComparisonMatrixProps {
  workloadId?: string | null;
}

const CostComparisonMatrix: React.FC<CostComparisonMatrixProps> = ({ workloadId }) => {
  const [selectedComparison, setSelectedComparison] = useState<CostComparison | null>(null);
  const [expandedBreakdown, setExpandedBreakdown] = useState<string | null>(null);

  // Fetch workloads if no workloadId provided
  const { data: workloads } = useQuery<WorkloadListResponse>(
    'userWorkloads',
    () => multiCloudApi.getWorkloadSpecifications(1, 20),
    {
      enabled: !workloadId,
      staleTime: 2 * 60 * 1000
    }
  );

  // Fetch cost comparisons for the selected workload
  const {
    data: comparisons,
    isLoading,
    error,
    refetch
  } = useQuery(
    ['workloadComparisons', workloadId || workloads?.workloads?.[0]?.id],
    () => {
      const id = workloadId || workloads?.workloads?.[0]?.id;
      if (!id) return Promise.resolve({ comparisons: [], total_count: 0, page: 1, page_size: 10 });
      return multiCloudApi.getWorkloadComparisons(id, 1, 10);
    },
    {
      enabled: !!(workloadId || workloads?.workloads?.[0]?.id),
      staleTime: 1 * 60 * 1000,
      onSuccess: (data) => {
        if (data.comparisons.length > 0 && !selectedComparison) {
          setSelectedComparison(data.comparisons[0]);
        }
      }
    }
  );

  const getProviderColor = (provider: string) => {
    const colors = {
      aws: '#FF9900',
      gcp: '#4285F4',
      azure: '#0078D4'
    };
    return colors[provider.toLowerCase() as keyof typeof colors] || '#666';
  };

  const formatCurrency = (amount: number | undefined) => {
    if (amount === undefined || amount === null) return 'N/A';
    return multiCloudApi.formatCurrency(amount);
  };

  const calculateSavings = (baseCost: number, compareCost: number) => {
    if (!baseCost || !compareCost) return 0;
    return ((baseCost - compareCost) / baseCost) * 100;
  };

  const getChartData = () => {
    if (!selectedComparison) return [];
    
    return [
      {
        provider: 'AWS',
        monthly: selectedComparison.aws_monthly_cost || 0,
        annual: selectedComparison.aws_annual_cost || 0,
        color: getProviderColor('aws')
      },
      {
        provider: 'GCP',
        monthly: selectedComparison.gcp_monthly_cost || 0,
        annual: selectedComparison.gcp_annual_cost || 0,
        color: getProviderColor('gcp')
      },
      {
        provider: 'Azure',
        monthly: selectedComparison.azure_monthly_cost || 0,
        annual: selectedComparison.azure_annual_cost || 0,
        color: getProviderColor('azure')
      }
    ];
  };

  const getPieChartData = (provider: string) => {
    if (!selectedComparison?.cost_breakdown?.[provider]) return [];
    
    const breakdown = selectedComparison.cost_breakdown[provider];
    return [
      { name: 'Compute', value: breakdown.compute, color: '#8884d8' },
      { name: 'Storage', value: breakdown.storage, color: '#82ca9d' },
      { name: 'Network', value: breakdown.network, color: '#ffc658' },
      { name: 'Database', value: breakdown.database || 0, color: '#ff7c7c' },
      { name: 'Support', value: breakdown.support, color: '#8dd1e1' }
    ].filter(item => item.value > 0);
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load cost comparison data. Please try again.
      </Alert>
    );
  }

  if (!comparisons?.comparisons?.length) {
    return (
      <Box sx={{ textAlign: 'center', p: 4 }}>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No cost comparisons available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Create a workload specification and run a cost comparison to see results here.
        </Typography>
      </Box>
    );
  }

  const chartData = getChartData();
  const lowestCostProvider = selectedComparison?.lowest_cost_provider;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Cost Comparison Results</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton onClick={() => refetch()} size="small">
            <RefreshIcon />
          </IconButton>
          <Button
            variant="outlined"
            size="small"
            startIcon={<DownloadIcon />}
            onClick={() => {
              // TODO: Implement export functionality
              console.log('Export comparison data');
            }}
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Comparison Selection */}
      {comparisons.comparisons.length > 1 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Select Comparison:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {comparisons.comparisons.map((comparison) => (
              <Chip
                key={comparison.id}
                label={new Date(comparison.comparison_date).toLocaleDateString()}
                onClick={() => setSelectedComparison(comparison)}
                color={selectedComparison?.id === comparison.id ? 'primary' : 'default'}
                variant={selectedComparison?.id === comparison.id ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Box>
      )}

      {selectedComparison && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    AWS
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(selectedComparison.aws_monthly_cost)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    per month
                  </Typography>
                  {lowestCostProvider === 'aws' && (
                    <Chip
                      label="Lowest Cost"
                      color="success"
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ color: getProviderColor('gcp') }} gutterBottom>
                    GCP
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(selectedComparison.gcp_monthly_cost)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    per month
                  </Typography>
                  {lowestCostProvider === 'gcp' && (
                    <Chip
                      label="Lowest Cost"
                      color="success"
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ color: getProviderColor('azure') }} gutterBottom>
                    Azure
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(selectedComparison.azure_monthly_cost)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    per month
                  </Typography>
                  {lowestCostProvider === 'azure' && (
                    <Chip
                      label="Lowest Cost"
                      color="success"
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Monthly Cost Comparison
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="provider" />
                      <YAxis />
                      <RechartsTooltip formatter={(value) => formatCurrency(Number(value))} />
                      <Bar dataKey="monthly" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Cost Difference
                  </Typography>
                  {selectedComparison.cost_difference_percentage && (
                    <Box>
                      {Object.entries(selectedComparison.cost_difference_percentage).map(([provider, percentage]) => (
                        <Box key={provider} sx={{ mb: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ textTransform: 'uppercase' }}>
                              {provider}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {percentage > 0 ? (
                                <TrendingUpIcon color="error" sx={{ mr: 0.5, fontSize: 16 }} />
                              ) : percentage < 0 ? (
                                <TrendingDownIcon color="success" sx={{ mr: 0.5, fontSize: 16 }} />
                              ) : null}
                              <Typography
                                variant="body2"
                                color={percentage > 0 ? 'error' : percentage < 0 ? 'success.main' : 'text.primary'}
                              >
                                {percentage > 0 ? '+' : ''}{percentage.toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Detailed Breakdown */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Cost Breakdown by Service
              </Typography>
              
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Service Category</TableCell>
                      <TableCell align="right">AWS</TableCell>
                      <TableCell align="right">GCP</TableCell>
                      <TableCell align="right">Azure</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {['compute', 'storage', 'network', 'database', 'support'].map((category) => (
                      <TableRow key={category}>
                        <TableCell sx={{ textTransform: 'capitalize', fontWeight: 'medium' }}>
                          {category}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(selectedComparison.cost_breakdown?.aws?.[category as keyof typeof selectedComparison.cost_breakdown.aws])}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(selectedComparison.cost_breakdown?.gcp?.[category as keyof typeof selectedComparison.cost_breakdown.gcp])}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(selectedComparison.cost_breakdown?.azure?.[category as keyof typeof selectedComparison.cost_breakdown.azure])}
                        </TableCell>
                      </TableRow>
                    ))}
                    <TableRow sx={{ backgroundColor: 'action.hover' }}>
                      <TableCell sx={{ fontWeight: 'bold' }}>Total</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(selectedComparison.cost_breakdown?.aws?.total)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(selectedComparison.cost_breakdown?.gcp?.total)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(selectedComparison.cost_breakdown?.azure?.total)}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {/* Recommendations */}
          {selectedComparison.recommendations && selectedComparison.recommendations.length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Optimization Recommendations
                </Typography>
                <Box>
                  {selectedComparison.recommendations.map((recommendation, index) => (
                    <Alert key={index} severity="info" sx={{ mb: 1 }}>
                      {recommendation}
                    </Alert>
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
};

export default CostComparisonMatrix;