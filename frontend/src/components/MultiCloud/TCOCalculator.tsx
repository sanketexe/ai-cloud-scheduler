/**
 * TCO Calculator Component
 * 
 * Interactive Total Cost of Ownership calculator with
 * multi-year projections and hidden cost analysis.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Divider
} from '@mui/material';
import {
  Schedule as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useMutation, useQuery } from 'react-query';
import toast from 'react-hot-toast';

import { multiCloudApi, TCOAnalysis, TCORequest, WorkloadListResponse } from '../../services/multiCloudApi';

interface TCOCalculatorProps {
  workloadId?: string | null;
  open?: boolean;
  onClose?: () => void;
}

const TCOCalculator: React.FC<TCOCalculatorProps> = ({ 
  workloadId, 
  open = false, 
  onClose 
}) => {
  const [timeHorizon, setTimeHorizon] = useState(3);
  const [includeHiddenCosts, setIncludeHiddenCosts] = useState(true);
  const [discountRate, setDiscountRate] = useState(0.05);
  const [selectedWorkloadId, setSelectedWorkloadId] = useState<string | null>(workloadId || null);
  const [tcoResults, setTcoResults] = useState<TCOAnalysis | null>(null);

  // Fetch workloads if no workloadId provided
  const { data: workloads } = useQuery<WorkloadListResponse>(
    'userWorkloads',
    () => multiCloudApi.getWorkloadSpecifications(1, 20),
    {
      enabled: !workloadId,
      staleTime: 2 * 60 * 1000
    }
  );

  // TCO calculation mutation
  const tcoMutation = useMutation(
    (request: TCORequest) => multiCloudApi.calculateTCO(request),
    {
      onSuccess: (data) => {
        setTcoResults(data);
        toast.success('TCO analysis completed');
      },
      onError: (error: any) => {
        toast.error(`TCO calculation failed: ${error.message}`);
      }
    }
  );

  useEffect(() => {
    if (workloadId) {
      setSelectedWorkloadId(workloadId);
    } else if (workloads?.workloads?.[0]?.id) {
      setSelectedWorkloadId(workloads.workloads[0].id);
    }
  }, [workloadId, workloads]);

  const handleCalculateTCO = () => {
    if (!selectedWorkloadId) {
      toast.error('Please select a workload');
      return;
    }

    const request: TCORequest = {
      workload_id: selectedWorkloadId,
      time_horizon_years: timeHorizon,
      include_hidden_costs: includeHiddenCosts,
      discount_rate: discountRate
    };

    tcoMutation.mutate(request);
  };

  const formatCurrency = (amount: number) => {
    return multiCloudApi.formatCurrency(amount);
  };

  const getProviderColor = (provider: string) => {
    const colors = {
      aws: '#FF9900',
      gcp: '#4285F4',
      azure: '#0078D4'
    };
    return colors[provider.toLowerCase() as keyof typeof colors] || '#666';
  };

  const getProjectionChartData = () => {
    if (!tcoResults?.cost_projections) return [];
    
    const years = Object.keys(tcoResults.cost_projections).sort();
    return years.map(year => ({
      year: year.replace('year_', 'Year '),
      aws: tcoResults.cost_projections[year].aws,
      gcp: tcoResults.cost_projections[year].gcp,
      azure: tcoResults.cost_projections[year].azure
    }));
  };

  const getHiddenCostsPieData = () => {
    if (!tcoResults?.hidden_costs) return [];
    
    return Object.entries(tcoResults.hidden_costs).map(([key, value], index) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: Number(value),
      color: ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'][index % 5]
    }));
  };

  const content = (
    <Box sx={{ p: open ? 0 : 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <AssessmentIcon sx={{ mr: 2, color: 'primary.main' }} />
        <Typography variant="h5" component="h2">
          Total Cost of Ownership Calculator
        </Typography>
      </Box>

      {/* Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            TCO Analysis Configuration
          </Typography>
          
          <Grid container spacing={3}>
            {!workloadId && (
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Select Workload</InputLabel>
                  <Select
                    value={selectedWorkloadId || ''}
                    onChange={(e) => setSelectedWorkloadId(e.target.value)}
                  >
                    {workloads?.workloads?.map((workload) => (
                      <MenuItem key={workload.id} value={workload.id}>
                        {workload.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            )}
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Time Horizon: {timeHorizon} years</Typography>
              <Slider
                value={timeHorizon}
                onChange={(_, value) => setTimeHorizon(value as number)}
                min={1}
                max={10}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Discount Rate: {(discountRate * 100).toFixed(1)}%</Typography>
              <Slider
                value={discountRate}
                onChange={(_, value) => setDiscountRate(value as number)}
                min={0}
                max={0.2}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${(value * 100).toFixed(1)}%`}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={handleCalculateTCO}
                disabled={!selectedWorkloadId || tcoMutation.isLoading}
                startIcon={tcoMutation.isLoading ? <CircularProgress size={20} /> : <TimelineIcon />}
                size="large"
              >
                {tcoMutation.isLoading ? 'Calculating...' : 'Calculate TCO'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results */}
      {tcoResults && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ color: getProviderColor('aws') }} gutterBottom>
                    AWS TCO
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(tcoResults.total_tco_comparison.aws)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {timeHorizon}-year total cost
                  </Typography>
                  {tcoResults.recommended_provider === 'aws' && (
                    <Chip
                      label="Recommended"
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
                    GCP TCO
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(tcoResults.total_tco_comparison.gcp)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {timeHorizon}-year total cost
                  </Typography>
                  {tcoResults.recommended_provider === 'gcp' && (
                    <Chip
                      label="Recommended"
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
                    Azure TCO
                  </Typography>
                  <Typography variant="h4">
                    {formatCurrency(tcoResults.total_tco_comparison.azure)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {timeHorizon}-year total cost
                  </Typography>
                  {tcoResults.recommended_provider === 'azure' && (
                    <Chip
                      label="Recommended"
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
                    Cost Projections Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={getProjectionChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis />
                      <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                      <Line 
                        type="monotone" 
                        dataKey="aws" 
                        stroke={getProviderColor('aws')} 
                        strokeWidth={2}
                        name="AWS"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="gcp" 
                        stroke={getProviderColor('gcp')} 
                        strokeWidth={2}
                        name="GCP"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="azure" 
                        stroke={getProviderColor('azure')} 
                        strokeWidth={2}
                        name="Azure"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Hidden Costs Breakdown
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={getHiddenCostsPieData()}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {getHiddenCostsPieData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Detailed Breakdown */}
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Detailed TCO Breakdown
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={() => {
                    // TODO: Implement export functionality
                    console.log('Export TCO analysis');
                  }}
                >
                  Export Report
                </Button>
              </Box>
              
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Cost Component</TableCell>
                      <TableCell align="right">AWS</TableCell>
                      <TableCell align="right">GCP</TableCell>
                      <TableCell align="right">Azure</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'medium' }}>Base Infrastructure</TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.aws_tco.base_infrastructure)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.gcp_tco.base_infrastructure)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.azure_tco.base_infrastructure)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'medium' }}>Support Costs</TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.aws_tco.support_costs)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.gcp_tco.support_costs)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.azure_tco.support_costs)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'medium' }}>Operational Overhead</TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.aws_tco.operational_overhead)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.gcp_tco.operational_overhead)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(tcoResults.azure_tco.operational_overhead)}
                      </TableCell>
                    </TableRow>
                    <TableRow sx={{ backgroundColor: 'action.hover' }}>
                      <TableCell sx={{ fontWeight: 'bold' }}>Total TCO</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(tcoResults.aws_tco.total)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(tcoResults.gcp_tco.total)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {formatCurrency(tcoResults.azure_tco.total)}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {/* Recommendation */}
          {tcoResults.recommended_provider && (
            <Alert severity="success" sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Recommendation: {multiCloudApi.getProviderDisplayName(tcoResults.recommended_provider)}
              </Typography>
              <Typography variant="body2">
                Based on the {timeHorizon}-year TCO analysis, {multiCloudApi.getProviderDisplayName(tcoResults.recommended_provider)} 
                offers the lowest total cost of ownership for your workload requirements.
              </Typography>
            </Alert>
          )}
        </>
      )}
    </Box>
  );

  if (open) {
    return (
      <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
        <DialogTitle>TCO Calculator</DialogTitle>
        <DialogContent>
          {content}
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Close</Button>
        </DialogActions>
      </Dialog>
    );
  }

  return content;
};

export default TCOCalculator;