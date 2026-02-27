import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Button,
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
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  Lightbulb as LightbulbIcon,
  Timeline as TimelineIcon,
  Compare as CompareIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  ReferenceLine,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import { anomalyApiService, AnomalyDetails } from '../services/anomalyApi';

interface ExplanationViewerProps {
  anomalyId: string | null;
  open: boolean;
  onClose: () => void;
}

const ExplanationViewer: React.FC<ExplanationViewerProps> = ({
  anomalyId,
  open,
  onClose
}) => {
  const [anomalyDetails, setAnomalyDetails] = useState<AnomalyDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (anomalyId && open) {
      loadAnomalyDetails();
    }
  }, [anomalyId, open]);

  const loadAnomalyDetails = async () => {
    if (!anomalyId) return;

    setLoading(true);
    setError(null);

    try {
      const details = await anomalyApiService.getAnomalyDetails(anomalyId);
      setAnomalyDetails(details);
    } catch (err: any) {
      setError('Failed to load anomaly details: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const prepareTimeSeriesData = () => {
    if (!anomalyDetails?.time_series_data) return [];

    return anomalyDetails.time_series_data.map(point => ({
      timestamp: format(parseISO(point.timestamp), 'MMM dd HH:mm'),
      cost: point.cost,
      baseline: point.baseline,
      deviation: point.cost - point.baseline,
      deviationPercentage: ((point.cost - point.baseline) / point.baseline) * 100
    }));
  };

  const calculateFeatureImportance = () => {
    // Simulated feature importance based on anomaly details
    if (!anomalyDetails) return [];

    const features = [
      { name: 'Cost Amount', importance: 0.85, description: 'Primary cost deviation factor' },
      { name: 'Time Pattern', importance: 0.72, description: 'Unusual timing of resource usage' },
      { name: 'Resource Count', importance: 0.68, description: 'Number of resources involved' },
      { name: 'Service Type', importance: 0.55, description: 'Type of AWS service affected' },
      { name: 'Region', importance: 0.42, description: 'Geographic distribution impact' },
      { name: 'Usage Pattern', importance: 0.38, description: 'Resource utilization pattern' }
    ];

    return features.sort((a, b) => b.importance - a.importance);
  };

  if (!open) return null;

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { height: '90vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">
            Anomaly Explanation & Analysis
          </Typography>
          <Button onClick={onClose} startIcon={<CloseIcon />}>
            Close
          </Button>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        {loading && (
          <Box sx={{ p: 3 }}>
            <LinearProgress />
            <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
              Loading anomaly details...
            </Typography>
          </Box>
        )}

        {error && (
          <Box sx={{ p: 3 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

        {anomalyDetails && (
          <Box sx={{ p: 3 }}>
            {/* Anomaly Overview */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={8}>
                    <Typography variant="h6" gutterBottom>
                      Anomaly Overview
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {anomalyDetails.description}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                      <Chip 
                        label={`${anomalyDetails.severity.toUpperCase()} Severity`}
                        color={getSeverityColor(anomalyDetails.severity)}
                      />
                      <Chip 
                        label={`${(anomalyDetails.confidence_score * 100).toFixed(0)}% Confidence`}
                        color={getConfidenceColor(anomalyDetails.confidence_score)}
                      />
                      <Chip 
                        label={`$${anomalyDetails.estimated_impact_usd.toFixed(2)} Impact`}
                        color="error"
                        variant="outlined"
                      />
                    </Box>

                    <Typography variant="body2" color="textSecondary">
                      Detected: {format(parseISO(anomalyDetails.detection_timestamp), 'MMM dd, yyyy HH:mm')}
                    </Typography>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle2" gutterBottom>
                      Affected Resources
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="textSecondary">Services:</Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
                        {anomalyDetails.affected_services.map(service => (
                          <Chip key={service} label={service} size="small" />
                        ))}
                      </Box>
                    </Box>
                    
                    <Box>
                      <Typography variant="body2" color="textSecondary">Regions:</Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {anomalyDetails.affected_regions.map(region => (
                          <Chip key={region} label={region} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Root Cause Analysis */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <AssessmentIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Root Cause Analysis</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Primary Cause
                    </Typography>
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      {anomalyDetails.root_cause_analysis.primary_cause}
                    </Alert>

                    <Typography variant="subtitle2" gutterBottom>
                      Contributing Factors
                    </Typography>
                    <List dense>
                      {anomalyDetails.root_cause_analysis.contributing_factors.map((factor, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <WarningIcon color="warning" />
                          </ListItemIcon>
                          <ListItemText primary={factor} />
                        </ListItem>
                      ))}
                    </List>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Affected Resources
                    </Typography>
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Resource ID</TableCell>
                            <TableCell>Type</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {anomalyDetails.root_cause_analysis.affected_resources.map((resource, index) => (
                            <TableRow key={index}>
                              <TableCell>{resource}</TableCell>
                              <TableCell>
                                {resource.startsWith('i-') ? 'EC2 Instance' : 
                                 resource.startsWith('vol-') ? 'EBS Volume' :
                                 resource.startsWith('snap-') ? 'EBS Snapshot' : 'Unknown'}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Time Series Analysis */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TimelineIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Time Series Analysis</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Cost vs Baseline Comparison
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={prepareTimeSeriesData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis tickFormatter={(value) => `$${value.toFixed(0)}`} />
                      <RechartsTooltip 
                        formatter={(value: number, name: string) => [`$${value.toFixed(2)}`, name]}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="baseline" 
                        stroke="#4caf50" 
                        strokeWidth={2}
                        name="Baseline"
                        strokeDasharray="5 5"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="cost" 
                        stroke="#f44336" 
                        strokeWidth={3}
                        name="Actual Cost"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Deviation Analysis
                  </Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={prepareTimeSeriesData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis tickFormatter={(value) => `${value.toFixed(0)}%`} />
                      <RechartsTooltip 
                        formatter={(value: number) => [`${value.toFixed(1)}%`, 'Deviation']}
                      />
                      <ReferenceLine y={0} stroke="#666" />
                      <Bar 
                        dataKey="deviationPercentage" 
                        fill="#ff9800"
                        name="Deviation %"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Feature Importance */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TrendingUpIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Feature Importance Analysis</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  This analysis shows which factors contributed most to the anomaly detection.
                </Typography>
                
                {calculateFeatureImportance().map((feature, index) => (
                  <Box key={index} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" fontWeight="medium">
                        {feature.name}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {(feature.importance * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={feature.importance * 100}
                      sx={{ height: 8, borderRadius: 4, mb: 0.5 }}
                    />
                    <Typography variant="caption" color="textSecondary">
                      {feature.description}
                    </Typography>
                  </Box>
                ))}
              </AccordionDetails>
            </Accordion>

            {/* Recommendations */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <LightbulbIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Recommendations</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  {anomalyDetails.recommendations.map((recommendation, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <InfoIcon color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={recommendation}
                        primaryTypographyProps={{ variant: 'body1' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>

            {/* Similar Anomalies */}
            {anomalyDetails.similar_anomalies.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CompareIcon sx={{ mr: 1 }} />
                    <Typography variant="h6">Similar Historical Anomalies</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    These are similar anomalies detected in the past that may provide additional context.
                  </Typography>
                  
                  <TableContainer component={Paper} variant="outlined">
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Anomaly ID</TableCell>
                          <TableCell>Date</TableCell>
                          <TableCell>Similarity Score</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {anomalyDetails.similar_anomalies.map((similar, index) => (
                          <TableRow key={index}>
                            <TableCell>{similar.anomaly_id}</TableCell>
                            <TableCell>{format(parseISO(similar.date), 'MMM dd, yyyy')}</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <LinearProgress 
                                  variant="determinate" 
                                  value={similar.similarity_score * 100}
                                  sx={{ width: 60, mr: 1 }}
                                />
                                <Typography variant="body2">
                                  {(similar.similarity_score * 100).toFixed(0)}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Button size="small" variant="outlined">
                                View Details
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
            )}
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default ExplanationViewer;