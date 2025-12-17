import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Alert,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AttachMoney as MoneyIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Storage as StorageIcon,
  Computer as ComputerIcon,
} from '@mui/icons-material';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';

// Types
interface AWSCredentials {
  aws_access_key_id: string;
  aws_secret_access_key: string;
  region: string;
}

interface OptimizationOpportunity {
  service: string;
  opportunity_type: string;
  current_monthly_cost: number;
  potential_monthly_savings: number;
  confidence_level: string;
  description: string;
  action_required: string;
  implementation_effort: string;
  risk_level: string;
}

interface ServiceCost {
  service_name: string;
  current_month_cost: number;
  percentage_of_total: number;
  cost_trend: string;
}

interface CostAnalysis {
  total_monthly_cost: number;
  cost_trend: string;
  top_cost_drivers: ServiceCost[];
  optimization_opportunities: OptimizationOpportunity[];
  potential_monthly_savings: number;
  roi_analysis: {
    monthly_savings: number;
    annual_savings: number;
    savings_percentage: number;
    implementation_effort: string;
  };
  recommendations_summary: string[];
  analysis_date: string;
}

const AWSCostAnalysis: React.FC = () => {
  const [credentials, setCredentials] = useState<AWSCredentials>({
    aws_access_key_id: '',
    aws_secret_access_key: '',
    region: 'us-east-1'
  });
  const [analysis, setAnalysis] = useState<CostAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [credentialsDialogOpen, setCredentialsDialogOpen] = useState(false);
  const [connectionTested, setConnectionTested] = useState(false);
  const [selectedOpportunityType, setSelectedOpportunityType] = useState<string>('all');

  const regions = [
    'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
    'eu-west-1', 'eu-west-2', 'eu-central-1',
    'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1'
  ];

  const testConnection = async () => {
    if (!credentials.aws_access_key_id || !credentials.aws_secret_access_key) {
      toast.error('Please enter AWS credentials');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/aws-cost/test-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const result = await response.json();

      if (response.ok && result.status === 'success') {
        toast.success('AWS connection successful!');
        setConnectionTested(true);
        setCredentialsDialogOpen(false);
      } else {
        toast.error(result.message || 'Connection failed');
      }
    } catch (error) {
      toast.error('Failed to test connection');
      console.error('Connection test error:', error);
    } finally {
      setLoading(false);
    }
  };

  const runCostAnalysis = async () => {
    if (!connectionTested) {
      toast.error('Please test AWS connection first');
      setCredentialsDialogOpen(true);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/aws-cost/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...credentials,
          days_back: 30,
          include_recommendations: true,
        }),
      });

      if (response.ok) {
        const analysisResult = await response.json();
        setAnalysis(analysisResult);
        toast.success(`Analysis complete! Found $${analysisResult.potential_monthly_savings.toFixed(2)}/month in potential savings`);
      } else {
        const error = await response.json();
        toast.error(error.detail || 'Analysis failed');
      }
    } catch (error) {
      toast.error('Failed to run cost analysis');
      console.error('Analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getOpportunityIcon = (type: string) => {
    switch (type) {
      case 'rightsizing':
        return <ComputerIcon />;
      case 'unused_resources':
        return <SpeedIcon />;
      case 'reserved_instances':
        return <MoneyIcon />;
      case 'storage_optimization':
        return <StorageIcon />;
      default:
        return <SecurityIcon />;
    }
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'success';
      case 'medium':
        return 'warning';
      case 'low':
        return 'error';
      default:
        return 'default';
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'success';
      case 'medium':
        return 'warning';
      case 'high':
        return 'error';
      default:
        return 'default';
    }
  };

  const filteredOpportunities = analysis?.optimization_opportunities.filter(opp => 
    selectedOpportunityType === 'all' || opp.opportunity_type === selectedOpportunityType
  ) || [];

  const pieChartData = analysis?.top_cost_drivers.slice(0, 8).map(service => ({
    name: service.service_name,
    value: service.current_month_cost,
    percentage: service.percentage_of_total
  })) || [];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          AWS Cost Analysis & Optimization
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Analyze your AWS spending and discover opportunities to reduce costs through intelligent recommendations.
        </Typography>

        {/* Action Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setCredentialsDialogOpen(true)}
            disabled={loading}
          >
            Configure AWS Credentials
          </Button>
          <Button
            variant="contained"
            onClick={runCostAnalysis}
            disabled={loading || !connectionTested}
            startIcon={loading ? <LinearProgress /> : <TrendingUpIcon />}
          >
            {loading ? 'Analyzing...' : 'Run Cost Analysis'}
          </Button>
        </Box>

        {/* Connection Status */}
        {connectionTested && (
          <Alert severity="success" sx={{ mb: 3 }}>
            AWS connection verified. Ready to analyze costs.
          </Alert>
        )}

        {/* Analysis Results */}
        {analysis && (
          <Grid container spacing={3}>
            {/* Summary Cards */}
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Monthly Cost
                  </Typography>
                  <Typography variant="h4">
                    ${analysis.total_monthly_cost.toFixed(2)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    {analysis.cost_trend === 'increasing' ? (
                      <TrendingUpIcon color="error" />
                    ) : analysis.cost_trend === 'decreasing' ? (
                      <TrendingDownIcon color="success" />
                    ) : null}
                    <Typography variant="body2" sx={{ ml: 1 }}>
                      {analysis.cost_trend}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Potential Monthly Savings
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    ${analysis.potential_monthly_savings.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {analysis.roi_analysis.savings_percentage.toFixed(1)}% of total cost
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Annual Savings Potential
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    ${analysis.roi_analysis.annual_savings.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Implementation: {analysis.roi_analysis.implementation_effort}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Opportunities Found
                  </Typography>
                  <Typography variant="h4">
                    {analysis.optimization_opportunities.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Optimization recommendations
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Cost Breakdown Chart */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Cost Breakdown by Service
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Cost']} />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            {/* Top Services Table */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Top Cost Drivers
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Service</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">% of Total</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {analysis.top_cost_drivers.slice(0, 8).map((service) => (
                        <TableRow key={service.service_name}>
                          <TableCell>{service.service_name}</TableCell>
                          <TableCell align="right">${service.current_month_cost.toFixed(2)}</TableCell>
                          <TableCell align="right">{service.percentage_of_total.toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>

            {/* Recommendations Summary */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Key Recommendations
                </Typography>
                <Grid container spacing={2}>
                  {analysis.recommendations_summary.map((recommendation, index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <Alert severity="info" sx={{ height: '100%' }}>
                        {recommendation}
                      </Alert>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>

            {/* Optimization Opportunities */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Optimization Opportunities
                  </Typography>
                  <FormControl size="small" sx={{ minWidth: 200 }}>
                    <InputLabel>Filter by Type</InputLabel>
                    <Select
                      value={selectedOpportunityType}
                      label="Filter by Type"
                      onChange={(e) => setSelectedOpportunityType(e.target.value)}
                    >
                      <MenuItem value="all">All Types</MenuItem>
                      <MenuItem value="rightsizing">Rightsizing</MenuItem>
                      <MenuItem value="unused_resources">Unused Resources</MenuItem>
                      <MenuItem value="reserved_instances">Reserved Instances</MenuItem>
                      <MenuItem value="storage_optimization">Storage Optimization</MenuItem>
                    </Select>
                  </FormControl>
                </Box>

                {filteredOpportunities.map((opportunity, index) => (
                  <Accordion key={index}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <Box sx={{ mr: 2 }}>
                          {getOpportunityIcon(opportunity.opportunity_type)}
                        </Box>
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="subtitle1">
                            {opportunity.description}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Potential savings: ${opportunity.potential_monthly_savings.toFixed(2)}/month
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip
                            label={opportunity.confidence_level}
                            color={getConfidenceColor(opportunity.confidence_level) as any}
                            size="small"
                          />
                          <Chip
                            label={`${opportunity.risk_level} risk`}
                            color={getRiskColor(opportunity.risk_level) as any}
                            size="small"
                          />
                        </Box>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Service:</strong> {opportunity.service}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Current Monthly Cost:</strong> ${opportunity.current_monthly_cost.toFixed(2)}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Implementation Effort:</strong> {opportunity.implementation_effort}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2" gutterBottom>
                            <strong>Action Required:</strong>
                          </Typography>
                          <Typography variant="body2">
                            {opportunity.action_required}
                          </Typography>
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </Paper>
            </Grid>
          </Grid>
        )}

        {/* AWS Credentials Dialog */}
        <Dialog open={credentialsDialogOpen} onClose={() => setCredentialsDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Configure AWS Credentials</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              <TextField
                fullWidth
                label="AWS Access Key ID"
                value={credentials.aws_access_key_id}
                onChange={(e) => setCredentials({ ...credentials, aws_access_key_id: e.target.value })}
                margin="normal"
                type="password"
              />
              <TextField
                fullWidth
                label="AWS Secret Access Key"
                value={credentials.aws_secret_access_key}
                onChange={(e) => setCredentials({ ...credentials, aws_secret_access_key: e.target.value })}
                margin="normal"
                type="password"
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Region</InputLabel>
                <Select
                  value={credentials.region}
                  label="Region"
                  onChange={(e) => setCredentials({ ...credentials, region: e.target.value })}
                >
                  {regions.map((region) => (
                    <MenuItem key={region} value={region}>
                      {region}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Alert severity="info" sx={{ mt: 2 }}>
                Your credentials are used only for this session and are not stored permanently.
                Ensure your AWS user has Cost Explorer and EC2 read permissions.
              </Alert>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCredentialsDialogOpen(false)}>Cancel</Button>
            <Button onClick={testConnection} variant="contained" disabled={loading}>
              {loading ? 'Testing...' : 'Test Connection'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default AWSCostAnalysis;