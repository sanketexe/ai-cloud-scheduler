import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  LinearProgress,
  Tabs,
  Tab,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  PlayArrow,
  Pause,
  CheckCircle,
  Warning,
  Info,
  Schedule,
  Savings,
  Memory,
  Storage,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import numeral from 'numeral';

// Mock data
const optimizationOpportunities = [
  {
    id: 1,
    type: 'rightsizing',
    resource: 'i-0123456789abcdef0',
    resourceType: 'EC2 Instance',
    currentConfig: 'm5.2xlarge (8 vCPU, 32 GB)',
    recommendedConfig: 'm5.xlarge (4 vCPU, 16 GB)',
    monthlySavings: 140.16,
    annualSavings: 1681.92,
    confidence: 95,
    riskLevel: 'low',
    status: 'new',
    team: 'Engineering',
    utilization: {
      cpu: 15.2,
      memory: 45.8,
    },
  },
  {
    id: 2,
    type: 'termination',
    resource: 'vol-0987654321fedcba0',
    resourceType: 'EBS Volume',
    currentConfig: 'gp3 500GB (Unattached)',
    recommendedConfig: 'Delete unused volume',
    monthlySavings: 50.00,
    annualSavings: 600.00,
    confidence: 100,
    riskLevel: 'none',
    status: 'new',
    team: 'DevOps',
    utilization: {
      cpu: 0,
      memory: 0,
    },
  },
  {
    id: 3,
    type: 'ri_purchase',
    resource: 'Multiple m5.large instances',
    resourceType: 'Reserved Instance',
    currentConfig: 'On-Demand pricing',
    recommendedConfig: '1-year Standard RI',
    monthlySavings: 245.50,
    annualSavings: 2946.00,
    confidence: 88,
    riskLevel: 'low',
    status: 'in_progress',
    team: 'Platform',
    utilization: {
      cpu: 85.0,
      memory: 78.5,
    },
  },
  {
    id: 4,
    type: 'storage_class',
    resource: 'finops-backup-bucket',
    resourceType: 'S3 Bucket',
    currentConfig: 'Standard storage class',
    recommendedConfig: 'Intelligent Tiering',
    monthlySavings: 89.25,
    annualSavings: 1071.00,
    confidence: 92,
    riskLevel: 'none',
    status: 'implemented',
    team: 'Data Science',
    utilization: {
      cpu: 0,
      memory: 0,
    },
  },
];

const savingsData = [
  { month: 'Oct', potential: 2500, realized: 1800 },
  { month: 'Nov', potential: 2800, realized: 2100 },
  { month: 'Dec', potential: 3200, realized: 2400 },
  { month: 'Jan', potential: 3500, realized: 2800 },
  { month: 'Feb', potential: 3100, realized: 2600 },
  { month: 'Mar', potential: 3400, realized: 0 },
];

const optimizationTypes = [
  { name: 'Right-sizing', count: 12, savings: 1680, color: '#ff9800' },
  { name: 'Termination', count: 8, savings: 950, color: '#f44336' },
  { name: 'Reserved Instances', count: 5, savings: 2946, color: '#4caf50' },
  { name: 'Storage Optimization', count: 15, savings: 1200, color: '#2196f3' },
];

const Optimization: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [implementDialogOpen, setImplementDialogOpen] = useState(false);
  const [selectedRecommendation, setSelectedRecommendation] = useState<any>(null);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'rightsizing':
        return <Memory sx={{ fontSize: 16 }} />;
      case 'termination':
        return <TrendingDown sx={{ fontSize: 16 }} />;
      case 'ri_purchase':
        return <Savings sx={{ fontSize: 16 }} />;
      case 'storage_class':
        return <Storage sx={{ fontSize: 16 }} />;
      default:
        return <TrendingUp sx={{ fontSize: 16 }} />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'rightsizing':
        return '#ff9800';
      case 'termination':
        return '#f44336';
      case 'ri_purchase':
        return '#4caf50';
      case 'storage_class':
        return '#2196f3';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new':
        return '#2196f3';
      case 'in_progress':
        return '#ff9800';
      case 'implemented':
        return '#4caf50';
      case 'dismissed':
        return '#9e9e9e';
      default:
        return '#9e9e9e';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'none':
        return '#4caf50';
      case 'low':
        return '#8bc34a';
      case 'medium':
        return '#ff9800';
      case 'high':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const handleImplement = (recommendation: any) => {
    setSelectedRecommendation(recommendation);
    setImplementDialogOpen(true);
  };

  const confirmImplementation = () => {
    // Implement the recommendation
    console.log('Implementing recommendation:', selectedRecommendation);
    setImplementDialogOpen(false);
    setSelectedRecommendation(null);
  };

  const totalPotentialSavings = optimizationOpportunities.reduce((sum, opp) => sum + opp.annualSavings, 0);
  const totalRealizedSavings = savingsData.reduce((sum, data) => sum + data.realized, 0);

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>
        Cost Optimization
      </Typography>

      {/* Optimization Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Potential Savings
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {numeral(totalPotentialSavings).format('$0,0')}
                </Typography>
                <Chip
                  label="Annual"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    color: '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Realized Savings
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {numeral(totalRealizedSavings).format('$0,0')}
                </Typography>
                <Chip
                  label="YTD"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(33, 150, 243, 0.2)',
                    color: '#2196f3',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Active Recommendations
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {optimizationOpportunities.filter(o => o.status === 'new').length}
                </Typography>
                <Chip
                  label="Need review"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    color: '#ff9800',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Implementation Rate
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  75%
                </Typography>
                <Chip
                  label="Last 30 days"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    color: '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Savings Trend and Optimization Types */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Savings Trend: Potential vs Realized
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={savingsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="month" stroke="#b0bec5" />
                    <YAxis stroke="#b0bec5" tickFormatter={(value) => numeral(value).format('$0a')} />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any) => [numeral(value).format('$0,0'), '']}
                    />
                    <Bar dataKey="potential" fill="rgba(255, 152, 0, 0.5)" name="Potential Savings" />
                    <Bar dataKey="realized" fill="#4caf50" name="Realized Savings" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Optimization Types
                </Typography>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={optimizationTypes}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="savings"
                    >
                      {optimizationTypes.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any, name: any, props: any) => [
                        `${numeral(value).format('$0,0')} (${props.payload.count} items)`,
                        props.payload.name
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <Box sx={{ mt: 2 }}>
                  {optimizationTypes.map((type) => (
                    <Box key={type.name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          backgroundColor: type.color,
                          mr: 1,
                        }}
                      />
                      <Typography variant="body2" sx={{ flexGrow: 1, fontSize: '0.8rem' }}>
                        {type.name}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>
                        {type.count}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Recommendations Table */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6">
                    Optimization Recommendations
                  </Typography>
                  <Tabs value={selectedTab} onChange={(e, newValue) => setSelectedTab(newValue)}>
                    <Tab label="All" />
                    <Tab label="New" />
                    <Tab label="In Progress" />
                    <Tab label="Implemented" />
                  </Tabs>
                </Box>
                <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Resource</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Current Config</TableCell>
                        <TableCell>Recommended</TableCell>
                        <TableCell align="right">Monthly Savings</TableCell>
                        <TableCell align="center">Confidence</TableCell>
                        <TableCell align="center">Risk</TableCell>
                        <TableCell align="center">Status</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {optimizationOpportunities.map((opportunity) => (
                        <TableRow key={opportunity.id}>
                          <TableCell component="th" scope="row">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {opportunity.resource}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {opportunity.resourceType} • {opportunity.team}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box sx={{ color: getTypeColor(opportunity.type), mr: 1 }}>
                                {getTypeIcon(opportunity.type)}
                              </Box>
                              <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                                {opportunity.type.replace('_', ' ')}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {opportunity.currentConfig}
                            </Typography>
                            {opportunity.utilization.cpu > 0 && (
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                CPU: {opportunity.utilization.cpu}% • Mem: {opportunity.utilization.memory}%
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {opportunity.recommendedConfig}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontWeight: 600, color: '#4caf50' }}>
                              {numeral(opportunity.monthlySavings).format('$0,0')}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {numeral(opportunity.annualSavings).format('$0,0')}/year
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <LinearProgress
                              variant="determinate"
                              value={opportunity.confidence}
                              sx={{
                                width: 60,
                                height: 6,
                                borderRadius: 3,
                                backgroundColor: 'rgba(255,255,255,0.1)',
                                '& .MuiLinearProgress-bar': {
                                  backgroundColor: opportunity.confidence > 90 ? '#4caf50' : 
                                                 opportunity.confidence > 70 ? '#ff9800' : '#f44336',
                                },
                              }}
                            />
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {opportunity.confidence}%
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={opportunity.riskLevel}
                              size="small"
                              sx={{
                                backgroundColor: `${getRiskColor(opportunity.riskLevel)}20`,
                                color: getRiskColor(opportunity.riskLevel),
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={opportunity.status}
                              size="small"
                              sx={{
                                backgroundColor: `${getStatusColor(opportunity.status)}20`,
                                color: getStatusColor(opportunity.status),
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                              {opportunity.status === 'new' && (
                                <Tooltip title="Implement Recommendation">
                                  <IconButton 
                                    size="small" 
                                    onClick={() => handleImplement(opportunity)}
                                    sx={{ color: '#4caf50' }}
                                  >
                                    <PlayArrow sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {opportunity.status === 'in_progress' && (
                                <Tooltip title="Pause Implementation">
                                  <IconButton size="small" sx={{ color: '#ff9800' }}>
                                    <Pause sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {opportunity.status === 'implemented' && (
                                <Tooltip title="Implemented">
                                  <IconButton size="small" sx={{ color: '#4caf50' }}>
                                    <CheckCircle sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title="View Details">
                                <IconButton size="small">
                                  <Info sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Implementation Dialog */}
      <Dialog open={implementDialogOpen} onClose={() => setImplementDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Implement Optimization Recommendation</DialogTitle>
        <DialogContent>
          {selectedRecommendation && (
            <Box>
              <Alert severity="info" sx={{ mb: 3 }}>
                You are about to implement an optimization that will save approximately{' '}
                <strong>{numeral(selectedRecommendation.monthlySavings).format('$0,0')}</strong> per month.
              </Alert>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Current Configuration:</Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {selectedRecommendation.currentConfig}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Recommended Configuration:</Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {selectedRecommendation.recommendedConfig}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Impact Analysis:</Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Monthly Savings: {numeral(selectedRecommendation.monthlySavings).format('$0,0')}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Annual Savings: {numeral(selectedRecommendation.annualSavings).format('$0,0')}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Confidence Level: {selectedRecommendation.confidence}%
                  </Typography>
                  <Typography variant="body2">
                    • Risk Level: {selectedRecommendation.riskLevel}
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImplementDialogOpen(false)}>Cancel</Button>
          <Button onClick={confirmImplementation} variant="contained" color="primary">
            Implement Now
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Optimization;