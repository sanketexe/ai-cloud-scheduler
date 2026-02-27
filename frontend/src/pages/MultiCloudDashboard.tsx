/**
 * Multi-Cloud Cost Comparison Dashboard
 * 
 * Main dashboard for comparing costs across AWS, GCP, and Azure.
 * Provides overview of workloads, cost comparisons, and quick actions.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  CloudQueue as CloudIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  CompareArrows as CompareIcon,
  Schedule as TimelineIcon
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { Helmet } from 'react-helmet-async';
import toast from 'react-hot-toast';

import { multiCloudApi, CloudProvider, WorkloadListResponse } from '../services/multiCloudApi';
import CostComparisonMatrix from '../components/MultiCloud/CostComparisonMatrix';
import WorkloadSpecWizard from '../components/MultiCloud/WorkloadSpecWizard';
import TCOCalculator from '../components/MultiCloud/TCOCalculator';
import ProviderOverview from '../components/MultiCloud/ProviderOverview';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`multi-cloud-tabpanel-${index}`}
      aria-labelledby={`multi-cloud-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const MultiCloudDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [showWorkloadWizard, setShowWorkloadWizard] = useState(false);
  const [showTCOCalculator, setShowTCOCalculator] = useState(false);
  const [selectedWorkloadId, setSelectedWorkloadId] = useState<string | null>(null);

  // Fetch supported providers
  const {
    data: providers,
    isLoading: providersLoading,
    error: providersError,
    refetch: refetchProviders
  } = useQuery<CloudProvider[]>(
    'supportedProviders',
    multiCloudApi.getSupportedProviders,
    {
      staleTime: 5 * 60 * 1000, // 5 minutes
      onError: (error) => {
        console.error('Failed to fetch providers:', error);
        toast.error('Failed to load cloud providers');
      }
    }
  );

  // Fetch user workloads
  const {
    data: workloads,
    isLoading: workloadsLoading,
    error: workloadsError,
    refetch: refetchWorkloads
  } = useQuery<WorkloadListResponse>(
    'userWorkloads',
    () => multiCloudApi.getWorkloadSpecifications(1, 20),
    {
      staleTime: 2 * 60 * 1000, // 2 minutes
      onError: (error) => {
        console.error('Failed to fetch workloads:', error);
        toast.error('Failed to load workloads');
      }
    }
  );

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    refetchProviders();
    refetchWorkloads();
    toast.success('Data refreshed');
  };

  const handleWorkloadCreated = () => {
    setShowWorkloadWizard(false);
    refetchWorkloads();
    toast.success('Workload specification created successfully');
  };

  const handleCompareWorkload = (workloadId: string) => {
    setSelectedWorkloadId(workloadId);
    setActiveTab(1); // Switch to comparison tab
  };

  const handleTCOAnalysis = (workloadId: string) => {
    setSelectedWorkloadId(workloadId);
    setShowTCOCalculator(true);
  };

  if (providersError || workloadsError) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load dashboard data. Please check your connection and try again.
        </Alert>
        <Button variant="contained" onClick={handleRefresh} startIcon={<RefreshIcon />}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <>
      <Helmet>
        <title>Multi-Cloud Cost Comparison - FinOps Platform</title>
        <meta name="description" content="Compare costs across AWS, GCP, and Azure cloud providers" />
      </Helmet>

      <Box sx={{ width: '100%' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Multi-Cloud Cost Comparison
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Compare costs and analyze workloads across AWS, GCP, and Azure
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Refresh data">
              <IconButton onClick={handleRefresh} disabled={providersLoading || workloadsLoading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setShowWorkloadWizard(true)}
            >
              New Workload
            </Button>
          </Box>
        </Box>

        {/* Provider Overview Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {providersLoading ? (
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            </Grid>
          ) : (
            providers?.map((provider) => (
              <Grid item xs={12} md={4} key={provider.provider_type}>
                <ProviderOverview provider={provider} />
              </Grid>
            ))
          )}
        </Grid>

        {/* Quick Stats */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CloudIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Total Workloads</Typography>
                </Box>
                <Typography variant="h4" color="primary">
                  {workloadsLoading ? '-' : workloads?.total_count || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Configured workloads
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CompareIcon color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Active Comparisons</Typography>
                </Box>
                <Typography variant="h4" color="secondary">
                  {workloadsLoading ? '-' : workloads?.workloads?.length || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Recent comparisons
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">Avg. Savings</Typography>
                </Box>
                <Typography variant="h4" color="success.main">
                  15%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Potential savings
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <AssessmentIcon color="info" sx={{ mr: 1 }} />
                  <Typography variant="h6">TCO Analyses</Typography>
                </Box>
                <Typography variant="h4" color="info.main">
                  {workloadsLoading ? '-' : Math.floor((workloads?.total_count || 0) * 0.7)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Completed analyses
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Main Content Tabs */}
        <Card>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="multi-cloud dashboard tabs">
              <Tab 
                label="Workloads" 
                icon={<CloudIcon />} 
                iconPosition="start"
                id="multi-cloud-tab-0"
                aria-controls="multi-cloud-tabpanel-0"
              />
              <Tab 
                label="Cost Comparison" 
                icon={<CompareIcon />} 
                iconPosition="start"
                id="multi-cloud-tab-1"
                aria-controls="multi-cloud-tabpanel-1"
              />
              <Tab 
                label="TCO Analysis" 
                icon={<TimelineIcon />} 
                iconPosition="start"
                id="multi-cloud-tab-2"
                aria-controls="multi-cloud-tabpanel-2"
              />
            </Tabs>
          </Box>

          {/* Workloads Tab */}
          <TabPanel value={activeTab} index={0}>
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Your Workloads</Typography>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => setShowWorkloadWizard(true)}
                >
                  Add Workload
                </Button>
              </Box>
              
              {workloadsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : workloads?.workloads?.length === 0 ? (
                <Box sx={{ textAlign: 'center', p: 4 }}>
                  <CloudIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No workloads configured
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Create your first workload specification to start comparing costs
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setShowWorkloadWizard(true)}
                  >
                    Create Workload
                  </Button>
                </Box>
              ) : (
                <Grid container spacing={2}>
                  {workloads?.workloads?.map((workload) => (
                    <Grid item xs={12} md={6} lg={4} key={workload.id}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {workload.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {workload.description || 'No description'}
                          </Typography>
                          
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                              Regions:
                            </Typography>
                            <Box sx={{ mt: 0.5 }}>
                              {workload.regions.slice(0, 3).map((region) => (
                                <Chip
                                  key={region}
                                  label={region}
                                  size="small"
                                  sx={{ mr: 0.5, mb: 0.5 }}
                                />
                              ))}
                              {workload.regions.length > 3 && (
                                <Chip
                                  label={`+${workload.regions.length - 3} more`}
                                  size="small"
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          </Box>

                          <Divider sx={{ my: 1 }} />
                          
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => handleCompareWorkload(workload.id)}
                            >
                              Compare Costs
                            </Button>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => handleTCOAnalysis(workload.id)}
                            >
                              TCO Analysis
                            </Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Box>
          </TabPanel>

          {/* Cost Comparison Tab */}
          <TabPanel value={activeTab} index={1}>
            <CostComparisonMatrix workloadId={selectedWorkloadId} />
          </TabPanel>

          {/* TCO Analysis Tab */}
          <TabPanel value={activeTab} index={2}>
            <TCOCalculator workloadId={selectedWorkloadId} />
          </TabPanel>
        </Card>

        {/* Workload Specification Wizard Modal */}
        <WorkloadSpecWizard
          open={showWorkloadWizard}
          onClose={() => setShowWorkloadWizard(false)}
          onComplete={handleWorkloadCreated}
        />

        {/* TCO Calculator Modal */}
        <TCOCalculator
          open={showTCOCalculator}
          onClose={() => setShowTCOCalculator(false)}
          workloadId={selectedWorkloadId}
        />
      </Box>
    </>
  );
};

export default MultiCloudDashboard;