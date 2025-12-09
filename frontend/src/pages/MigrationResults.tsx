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
  Chip,
  Divider,
  LinearProgress,
  Tab,
  Tabs,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Warning,
  Info,
  ArrowBack,
  Download,
  Share,
  CloudQueue,
  Storage,
  Memory,
  Speed,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { alpha } from '@mui/material/styles';
import axios from 'axios';
import toast from 'react-hot-toast';

interface CloudProvider {
  name: string;
  logo: string;
  color: string;
  totalCost: number;
  breakdown: {
    compute: number;
    storage: number;
    database: number;
    network: number;
    other: number;
  };
  pros: string[];
  cons: string[];
  recommendation: string;
  migrationTime: string;
  complexity: 'Low' | 'Medium' | 'High';
}

interface MigrationResults {
  project_id: string;
  organization_name: string;
  providers: {
    aws: CloudProvider;
    gcp: CloudProvider;
    azure: CloudProvider;
  };
  recommended_provider: string;
  estimated_savings: number;
  analysis_date: string;
}

const PROVIDER_COLORS = {
  aws: '#FF9900',
  gcp: '#4285F4',
  azure: '#0078D4',
};

const MigrationResults: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<MigrationResults | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    loadResults();
  }, [projectId]);

  const loadResults = async () => {
    try {
      setLoading(true);
      // TODO: Replace with actual API call
      // const response = await axios.get(`/api/v1/migration/${projectId}/results`);
      
      // Mock data for demonstration
      const mockResults: MigrationResults = {
        project_id: projectId || '',
        organization_name: 'Your Company',
        providers: {
          aws: {
            name: 'Amazon Web Services',
            logo: '☁️',
            color: PROVIDER_COLORS.aws,
            totalCost: 5420,
            breakdown: {
              compute: 2100,
              storage: 800,
              database: 1500,
              network: 720,
              other: 300,
            },
            pros: [
              'Largest market share and ecosystem',
              'Most comprehensive service catalog',
              'Excellent documentation and community',
              'Strong enterprise support',
            ],
            cons: [
              'Complex pricing model',
              'Steeper learning curve',
              'Can be more expensive for certain workloads',
            ],
            recommendation: 'Best for: Enterprise applications, complex architectures, need for specialized services',
            migrationTime: '4-6 months',
            complexity: 'Medium',
          },
          gcp: {
            name: 'Google Cloud Platform',
            logo: '🔵',
            color: PROVIDER_COLORS.gcp,
            totalCost: 4890,
            breakdown: {
              compute: 1850,
              storage: 750,
              database: 1400,
              network: 650,
              other: 240,
            },
            pros: [
              'Best-in-class data analytics and ML',
              'Competitive pricing',
              'Excellent Kubernetes support',
              'Strong commitment to sustainability',
            ],
            cons: [
              'Smaller service catalog than AWS',
              'Less enterprise adoption',
              'Fewer regional data centers',
            ],
            recommendation: 'Best for: Data-intensive applications, machine learning, containerized workloads',
            migrationTime: '3-5 months',
            complexity: 'Low',
          },
          azure: {
            name: 'Microsoft Azure',
            logo: '⚡',
            color: PROVIDER_COLORS.azure,
            totalCost: 5150,
            breakdown: {
              compute: 2000,
              storage: 780,
              database: 1450,
              network: 700,
              other: 220,
            },
            pros: [
              'Seamless Microsoft integration',
              'Strong hybrid cloud capabilities',
              'Excellent for .NET applications',
              'Good enterprise support',
            ],
            cons: [
              'Can be complex for non-Microsoft stacks',
              'Pricing can be confusing',
              'Some services lag behind AWS',
            ],
            recommendation: 'Best for: Microsoft-centric organizations, hybrid cloud, .NET applications',
            migrationTime: '4-6 months',
            complexity: 'Medium',
          },
        },
        recommended_provider: 'gcp',
        estimated_savings: 530,
        analysis_date: new Date().toISOString(),
      };

      setResults(mockResults);
    } catch (error) {
      console.error('Failed to load results:', error);
      toast.error('Failed to load migration results');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>Analyzing your migration options...</Typography>
        </Box>
      </Container>
    );
  }

  if (!results) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Alert severity="error">Failed to load migration results</Alert>
        </Box>
      </Container>
    );
  }

  // Prepare data for charts
  const costComparisonData = [
    { provider: 'AWS', cost: results.providers.aws.totalCost, color: PROVIDER_COLORS.aws },
    { provider: 'GCP', cost: results.providers.gcp.totalCost, color: PROVIDER_COLORS.gcp },
    { provider: 'Azure', cost: results.providers.azure.totalCost, color: PROVIDER_COLORS.azure },
  ];

  const breakdownData = (provider: CloudProvider) => [
    { name: 'Compute', value: provider.breakdown.compute },
    { name: 'Storage', value: provider.breakdown.storage },
    { name: 'Database', value: provider.breakdown.database },
    { name: 'Network', value: provider.breakdown.network },
    { name: 'Other', value: provider.breakdown.other },
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  const getProviderByTab = (tab: number): CloudProvider => {
    switch (tab) {
      case 0:
        return results.providers.aws;
      case 1:
        return results.providers.gcp;
      case 2:
        return results.providers.azure;
      default:
        return results.providers.aws;
    }
  };

  const currentProvider = getProviderByTab(activeTab);

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <IconButton onClick={() => navigate(-1)}>
              <ArrowBack />
            </IconButton>
            <Box>
              <Typography variant="h4" fontWeight={600}>
                Migration Cost Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {results.organization_name} • Generated {new Date(results.analysis_date).toLocaleDateString()}
              </Typography>
            </Box>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button startIcon={<Download />} variant="outlined">
              Export PDF
            </Button>
            <Button startIcon={<Share />} variant="outlined">
              Share
            </Button>
          </Box>
        </Box>

        {/* Recommended Provider Alert */}
        <Alert
          severity="success"
          icon={<CheckCircle />}
          sx={{ mb: 3 }}
        >
          <Typography variant="subtitle1" fontWeight={600}>
            Recommended: {results.providers[results.recommended_provider as keyof typeof results.providers].name}
          </Typography>
          <Typography variant="body2">
            Based on your requirements, we recommend {results.providers[results.recommended_provider as keyof typeof results.providers].name} with an estimated monthly cost of ${results.providers[results.recommended_provider as keyof typeof results.providers].totalCost.toLocaleString()} and potential savings of ${results.estimated_savings.toLocaleString()}/month compared to the highest option.
          </Typography>
        </Alert>

        {/* Cost Comparison Chart */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Monthly Cost Comparison
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={costComparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis />
              <RechartsTooltip
                formatter={(value: number) => `$${value.toLocaleString()}`}
                contentStyle={{ backgroundColor: '#1a1d3a', border: '1px solid rgba(255,255,255,0.1)' }}
              />
              <Bar dataKey="cost" radius={[8, 8, 0, 0]}>
                {costComparisonData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Paper>

        {/* Provider Details Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="AWS" />
            <Tab label="GCP" />
            <Tab label="Azure" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              {/* Cost Breakdown */}
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Cost Breakdown
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={breakdownData(currentProvider)}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {breakdownData(currentProvider).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip
                      formatter={(value: number) => `$${value.toLocaleString()}`}
                      contentStyle={{ backgroundColor: '#1a1d3a', border: '1px solid rgba(255,255,255,0.1)' }}
                    />
                  </PieChart>
                </ResponsiveContainer>

                <List dense>
                  {Object.entries(currentProvider.breakdown).map(([key, value]) => (
                    <ListItem key={key}>
                      <ListItemText
                        primary={key.charAt(0).toUpperCase() + key.slice(1)}
                        secondary={`$${value.toLocaleString()}/month`}
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>

              {/* Provider Details */}
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Total Monthly Cost
                  </Typography>
                  <Typography variant="h3" color="primary" fontWeight={700}>
                    ${currentProvider.totalCost.toLocaleString()}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                    <Chip
                      label={`Migration Time: ${currentProvider.migrationTime}`}
                      icon={<Speed />}
                      size="small"
                    />
                    <Chip
                      label={`Complexity: ${currentProvider.complexity}`}
                      color={currentProvider.complexity === 'Low' ? 'success' : currentProvider.complexity === 'Medium' ? 'warning' : 'error'}
                      size="small"
                    />
                  </Box>
                </Box>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                  Pros
                </Typography>
                <List dense>
                  {currentProvider.pros.map((pro, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <CheckCircle color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={pro} />
                    </ListItem>
                  ))}
                </List>

                <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ mt: 2 }}>
                  Cons
                </Typography>
                <List dense>
                  {currentProvider.cons.map((con, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Warning color="warning" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary={con} />
                    </ListItem>
                  ))}
                </List>

                <Alert severity="info" sx={{ mt: 2 }}>
                  {currentProvider.recommendation}
                </Alert>
              </Grid>
            </Grid>
          </Box>
        </Paper>

        {/* Next Steps */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Next Steps
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    1. Review & Refine
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Review the cost estimates and adjust your requirements if needed.
                  </Typography>
                  <Button variant="outlined" fullWidth sx={{ mt: 2 }}>
                    Adjust Requirements
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    2. Create Migration Plan
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Generate a detailed migration plan with timelines and tasks.
                  </Typography>
                  <Button variant="contained" fullWidth sx={{ mt: 2 }}>
                    Create Plan
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    3. Talk to Expert
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Schedule a consultation with our cloud migration experts.
                  </Typography>
                  <Button variant="outlined" fullWidth sx={{ mt: 2 }}>
                    Schedule Call
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
};

export default MigrationResults;
