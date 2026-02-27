/**
 * Risk Assessment Component
 * 
 * Displays migration risks, mitigation strategies,
 * and success probability with visual indicators.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
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
  Paper,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Security as SecurityIcon,
  Business as BusinessIcon,
  Build as BuildIcon,
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  Shield as ShieldIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

import { RiskAssessment as RiskAssessmentType } from '../../services/multiCloudApi';

interface RiskAssessmentProps {
  risks: RiskAssessmentType;
}

interface RiskCategory {
  name: string;
  risks: string[];
  severity: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  probability: number;
  icon: React.ReactNode;
  color: string;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ risks }) => {
  const [expandedAccordion, setExpandedAccordion] = useState<string | false>('technical');

  const handleAccordionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedAccordion(isExpanded ? panel : false);
  };

  // Process and categorize risks
  const processRiskCategories = (): RiskCategory[] => {
    return [
      {
        name: 'Technical Risks',
        risks: risks.technical_risks || [],
        severity: risks.technical_risks?.length > 3 ? 'high' : risks.technical_risks?.length > 1 ? 'medium' : 'low',
        impact: 'high',
        probability: 0.3,
        icon: <BuildIcon />,
        color: '#ff7c7c'
      },
      {
        name: 'Business Risks',
        risks: risks.business_risks || [],
        severity: risks.business_risks?.length > 2 ? 'high' : risks.business_risks?.length > 0 ? 'medium' : 'low',
        impact: 'medium',
        probability: 0.2,
        icon: <BusinessIcon />,
        color: '#ffc658'
      },
      {
        name: 'Security Risks',
        risks: ['Data exposure during migration', 'Temporary security gaps', 'Access control misconfigurations'],
        severity: 'medium',
        impact: 'high',
        probability: 0.15,
        icon: <SecurityIcon />,
        color: '#8884d8'
      }
    ];
  };

  const riskCategories = processRiskCategories();

  // Generate risk matrix data
  const generateRiskMatrix = () => {
    return riskCategories.map(category => ({
      category: category.name,
      probability: category.probability * 100,
      impact: category.impact === 'high' ? 3 : category.impact === 'medium' ? 2 : 1,
      severity: category.severity,
      riskScore: (category.probability * 100) * (category.impact === 'high' ? 3 : category.impact === 'medium' ? 2 : 1)
    }));
  };

  const riskMatrixData = generateRiskMatrix();

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high':
        return <ErrorIcon color="error" />;
      case 'medium':
        return <WarningIcon color="warning" />;
      case 'low':
        return <CheckCircleIcon color="success" />;
      default:
        return <InfoIcon />;
    }
  };

  const getOverallRiskLevel = () => {
    return risks.overall_risk_level || 'medium';
  };

  const getSuccessProbabilityColor = (probability: number) => {
    if (probability >= 0.8) return 'success';
    if (probability >= 0.6) return 'warning';
    return 'error';
  };

  const COLORS = ['#ff7c7c', '#ffc658', '#8884d8', '#82ca9d', '#8dd1e1'];

  return (
    <Box>
      {/* Risk Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AssessmentIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Overall Risk</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                {getSeverityIcon(getOverallRiskLevel())}
                <Typography variant="h4" sx={{ ml: 1, textTransform: 'uppercase' }}>
                  {getOverallRiskLevel()}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Success Rate</Typography>
              </Box>
              <Typography variant="h4" color={`${getSuccessProbabilityColor(risks.success_probability)}.main`}>
                {(risks.success_probability * 100).toFixed(0)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={risks.success_probability * 100}
                color={getSuccessProbabilityColor(risks.success_probability) as any}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <WarningIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Risk Areas</Typography>
              </Box>
              <Typography variant="h4" color="warning.main">
                {riskCategories.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Categories identified
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ShieldIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Mitigations</Typography>
              </Box>
              <Typography variant="h4" color="info.main">
                {risks.mitigation_strategies?.length || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Strategies available
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Risk Matrix Visualization */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Assessment Matrix
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskMatrixData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <RechartsTooltip 
                    formatter={(value, name) => {
                      if (name === 'riskScore') return [value, 'Risk Score'];
                      if (name === 'probability') return [`${value}%`, 'Probability'];
                      return [value, name];
                    }}
                  />
                  <Bar dataKey="riskScore" fill="#8884d8" name="Risk Score" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskCategories}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="probability"
                    label={({ name, probability }) => `${name}: ${(probability * 100).toFixed(0)}%`}
                  >
                    {riskCategories.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Risk Analysis */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Detailed Risk Analysis
          </Typography>
          
          {riskCategories.map((category, index) => (
            <Accordion
              key={category.name}
              expanded={expandedAccordion === category.name.toLowerCase().replace(' ', '')}
              onChange={handleAccordionChange(category.name.toLowerCase().replace(' ', ''))}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                    {category.icon}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      {category.name}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip
                      label={category.severity}
                      color={getSeverityColor(category.severity) as any}
                      size="small"
                    />
                    <Chip
                      label={`${(category.probability * 100).toFixed(0)}% probability`}
                      variant="outlined"
                      size="small"
                    />
                  </Box>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Identified Risks:
                    </Typography>
                    <List dense>
                      {category.risks.map((risk, riskIndex) => (
                        <ListItem key={riskIndex}>
                          <ListItemIcon>
                            {getSeverityIcon(category.severity)}
                          </ListItemIcon>
                          <ListItemText primary={risk} />
                        </ListItem>
                      ))}
                    </List>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Risk Metrics:
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Impact Level: {category.impact}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={category.impact === 'high' ? 100 : category.impact === 'medium' ? 66 : 33}
                        color={getSeverityColor(category.impact) as any}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Probability: {(category.probability * 100).toFixed(0)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={category.probability * 100}
                        color="info"
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          ))}
        </CardContent>
      </Card>

      {/* Mitigation Strategies */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Risk Mitigation Strategies
          </Typography>
          
          {risks.mitigation_strategies && risks.mitigation_strategies.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Strategy</TableCell>
                    <TableCell>Priority</TableCell>
                    <TableCell>Effort</TableCell>
                    <TableCell>Impact</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {risks.mitigation_strategies.map((strategy, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <ShieldIcon color="primary" sx={{ mr: 1, fontSize: 20 }} />
                          {strategy}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={index < 2 ? 'High' : index < 4 ? 'Medium' : 'Low'}
                          color={index < 2 ? 'error' : index < 4 ? 'warning' : 'success'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={index % 3 === 0 ? 'Low' : index % 3 === 1 ? 'Medium' : 'High'}
                          variant="outlined"
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <LinearProgress
                            variant="determinate"
                            value={80 - (index * 10)}
                            color="success"
                            sx={{ width: 60, mr: 1 }}
                          />
                          <Typography variant="caption">
                            {80 - (index * 10)}%
                          </Typography>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              No specific mitigation strategies provided. Consider developing risk mitigation plans based on the identified risks above.
            </Alert>
          )}
          
          {/* Risk Recommendations */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Recommendations:
            </Typography>
            <Alert severity="success" sx={{ mb: 1 }}>
              <strong>Proceed with Caution:</strong> The migration shows a {(risks.success_probability * 100).toFixed(0)}% success probability. 
              Implement the recommended mitigation strategies to improve outcomes.
            </Alert>
            <Alert severity="info" sx={{ mb: 1 }}>
              <strong>Focus Areas:</strong> Pay special attention to technical risks and ensure adequate testing phases.
            </Alert>
            <Alert severity="warning">
              <strong>Contingency Planning:</strong> Develop rollback procedures and maintain parallel systems during critical phases.
            </Alert>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default RiskAssessment;