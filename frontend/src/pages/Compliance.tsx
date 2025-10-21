import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Chip,
  LinearProgress,
  Tabs,
  Tab,
  Alert,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Security,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Download,
  Visibility,
  Label,
  Policy,
  Assessment,
} from '@mui/icons-material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
} from 'recharts';
import numeral from 'numeral';

// Mock data
const complianceOverview = {
  overallScore: 87,
  taggingCompliance: 82,
  policyCompliance: 91,
  securityCompliance: 89,
  totalResources: 1247,
  compliantResources: 1084,
  nonCompliantResources: 163,
};

const taggingCompliance = [
  { team: 'Engineering', total: 450, compliant: 380, compliance: 84 },
  { team: 'Data Science', total: 280, compliant: 245, compliance: 88 },
  { team: 'DevOps', total: 320, compliant: 250, compliance: 78 },
  { team: 'QA', total: 197, compliant: 185, compliance: 94 },
];

const policyViolations = [
  {
    id: 1,
    resource: 'i-0123456789abcdef0',
    resourceType: 'EC2 Instance',
    policy: 'Required Tags Policy',
    violation: 'Missing CostCenter tag',
    severity: 'medium',
    team: 'Engineering',
    detected: '2024-01-15T10:30:00Z',
    status: 'open',
  },
  {
    id: 2,
    resource: 'vol-0987654321fedcba0',
    resourceType: 'EBS Volume',
    policy: 'Encryption Policy',
    violation: 'Volume not encrypted',
    severity: 'high',
    team: 'DevOps',
    detected: '2024-01-14T14:20:00Z',
    status: 'open',
  },
  {
    id: 3,
    resource: 'finops-bucket-logs',
    resourceType: 'S3 Bucket',
    policy: 'Public Access Policy',
    violation: 'Public read access enabled',
    severity: 'critical',
    team: 'Data Science',
    detected: '2024-01-13T09:15:00Z',
    status: 'remediated',
  },
  {
    id: 4,
    resource: 'rds-prod-database',
    resourceType: 'RDS Instance',
    policy: 'Backup Policy',
    violation: 'Backup retention < 7 days',
    severity: 'medium',
    team: 'Engineering',
    detected: '2024-01-12T16:45:00Z',
    status: 'open',
  },
];

const complianceByService = [
  { service: 'EC2', compliant: 85, nonCompliant: 15, color: '#ff9800' },
  { service: 'S3', compliant: 92, nonCompliant: 8, color: '#4caf50' },
  { service: 'RDS', compliant: 78, nonCompliant: 22, color: '#2196f3' },
  { service: 'Lambda', compliant: 95, nonCompliant: 5, color: '#9c27b0' },
];

const Compliance: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#f44336';
      case 'high':
        return '#ff5722';
      case 'medium':
        return '#ff9800';
      case 'low':
        return '#ffc107';
      default:
        return '#9e9e9e';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <Error sx={{ color: '#f44336', fontSize: 16 }} />;
      case 'high':
        return <Warning sx={{ color: '#ff5722', fontSize: 16 }} />;
      case 'medium':
        return <Warning sx={{ color: '#ff9800', fontSize: 16 }} />;
      case 'low':
        return <Warning sx={{ color: '#ffc107', fontSize: 16 }} />;
      default:
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 16 }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open':
        return '#f44336';
      case 'remediated':
        return '#4caf50';
      case 'in_progress':
        return '#ff9800';
      default:
        return '#9e9e9e';
    }
  };

  const openViolations = policyViolations.filter(v => v.status === 'open').length;
  const criticalViolations = policyViolations.filter(v => v.severity === 'critical' && v.status === 'open').length;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Compliance & Governance
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<Refresh />}>
            Refresh Scan
          </Button>
          <Button variant="outlined" startIcon={<Download />}>
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Critical Violations Alert */}
      {criticalViolations > 0 && (
        <Alert severity="error" sx={{ mb: 4 }}>
          <Typography variant="body2">
            You have {criticalViolations} critical compliance violation{criticalViolations > 1 ? 's' : ''} that require immediate attention.
          </Typography>
        </Alert>
      )}

      {/* Compliance Overview Cards */}
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
                  Overall Compliance
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {complianceOverview.overallScore}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={complianceOverview.overallScore}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: complianceOverview.overallScore > 90 ? '#4caf50' : 
                                     complianceOverview.overallScore > 75 ? '#ff9800' : '#f44336',
                    },
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
                  Tagging Compliance
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {complianceOverview.taggingCompliance}%
                </Typography>
                <Chip
                  label={`${complianceOverview.compliantResources}/${complianceOverview.totalResources} resources`}
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
                  Policy Violations
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: '#f44336' }}>
                  {openViolations}
                </Typography>
                <Chip
                  label="Open violations"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(244, 67, 54, 0.2)',
                    color: '#f44336',
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
                  Security Score
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {complianceOverview.securityCompliance}%
                </Typography>
                <Chip
                  label="Good standing"
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

      {/* Compliance Charts */}
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
                  Tagging Compliance by Team
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={taggingCompliance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="team" stroke="#b0bec5" />
                    <YAxis stroke="#b0bec5" />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#1a1d3a',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: any, name: any) => [
                        name === 'compliant' ? `${value} compliant` : `${value} total`,
                        ''
                      ]}
                    />
                    <Bar dataKey="total" fill="rgba(158, 158, 158, 0.5)" name="Total Resources" />
                    <Bar dataKey="compliant" fill="#4caf50" name="Compliant Resources" />
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
                  Compliance by Service
                </Typography>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={complianceByService}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="compliant"
                    >
                      {complianceByService.map((entry, index) => (
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
                        `${value}% compliant`,
                        props.payload.service
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <Box sx={{ mt: 2 }}>
                  {complianceByService.map((service) => (
                    <Box key={service.service} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          backgroundColor: service.color,
                          mr: 1,
                        }}
                      />
                      <Typography variant="body2" sx={{ flexGrow: 1, fontSize: '0.8rem' }}>
                        {service.service}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>
                        {service.compliant}%
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Compliance Details */}
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
                    Compliance Details
                  </Typography>
                  <Tabs value={selectedTab} onChange={(e, newValue) => setSelectedTab(newValue)}>
                    <Tab label="Policy Violations" />
                    <Tab label="Tagging Issues" />
                    <Tab label="Security Findings" />
                  </Tabs>
                </Box>

                {selectedTab === 0 && (
                  <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Resource</TableCell>
                          <TableCell>Policy</TableCell>
                          <TableCell>Violation</TableCell>
                          <TableCell align="center">Severity</TableCell>
                          <TableCell>Team</TableCell>
                          <TableCell>Detected</TableCell>
                          <TableCell align="center">Status</TableCell>
                          <TableCell align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {policyViolations.map((violation) => (
                          <TableRow key={violation.id}>
                            <TableCell component="th" scope="row">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {violation.resource}
                              </Typography>
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                {violation.resourceType}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Policy sx={{ fontSize: 16, mr: 1, color: 'text.secondary' }} />
                                <Typography variant="body2">
                                  {violation.policy}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {violation.violation}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                {getSeverityIcon(violation.severity)}
                                <Typography 
                                  variant="body2" 
                                  sx={{ 
                                    ml: 1, 
                                    color: getSeverityColor(violation.severity),
                                    textTransform: 'capitalize'
                                  }}
                                >
                                  {violation.severity}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={violation.team}
                                size="small"
                                sx={{
                                  backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                  color: '#2196f3',
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {new Date(violation.detected).toLocaleDateString()}
                              </Typography>
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                {new Date(violation.detected).toLocaleTimeString()}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Chip
                                label={violation.status}
                                size="small"
                                sx={{
                                  backgroundColor: `${getStatusColor(violation.status)}20`,
                                  color: getStatusColor(violation.status),
                                  textTransform: 'capitalize',
                                }}
                              />
                            </TableCell>
                            <TableCell align="center">
                              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                                <Tooltip title="View Details">
                                  <IconButton size="small">
                                    <Visibility sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                                <Tooltip title="Remediate">
                                  <IconButton size="small" sx={{ color: '#4caf50' }}>
                                    <CheckCircle sx={{ fontSize: 16 }} />
                                  </IconButton>
                                </Tooltip>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}

                {selectedTab === 1 && (
                  <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Team</TableCell>
                          <TableCell align="right">Total Resources</TableCell>
                          <TableCell align="right">Compliant</TableCell>
                          <TableCell align="right">Non-Compliant</TableCell>
                          <TableCell align="center">Compliance Rate</TableCell>
                          <TableCell align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {taggingCompliance.map((team) => (
                          <TableRow key={team.team}>
                            <TableCell component="th" scope="row">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {team.team}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                {team.total}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" sx={{ color: '#4caf50', fontWeight: 600 }}>
                                {team.compliant}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" sx={{ color: '#f44336', fontWeight: 600 }}>
                                {team.total - team.compliant}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Box sx={{ width: '100%', mr: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={team.compliance}
                                  sx={{
                                    height: 8,
                                    borderRadius: 4,
                                    backgroundColor: 'rgba(255,255,255,0.1)',
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: team.compliance > 90 ? '#4caf50' : 
                                                     team.compliance > 75 ? '#ff9800' : '#f44336',
                                    },
                                  }}
                                />
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                  {team.compliance}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell align="center">
                              <Tooltip title="View Details">
                                <IconButton size="small">
                                  <Assessment sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}

                {selectedTab === 2 && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Security sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Security Findings
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Security compliance monitoring is coming soon. This will include vulnerability assessments, 
                      security group analysis, and encryption compliance.
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Compliance;