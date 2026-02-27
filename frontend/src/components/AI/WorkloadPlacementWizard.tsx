import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Rating,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Slider,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  CloudQueue as WorkloadIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  TrendingUp as PerformanceIcon,
  AttachMoney as CostIcon,
  Security as ComplianceIcon,
  Speed as LatencyIcon,
} from '@mui/icons-material';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from 'recharts';
import { useQuery, useMutation } from 'react-query';

interface WorkloadSpec {
  name: string;
  description: string;
  resourceRequirements: {
    cpu: number;
    memory: number;
    storage: number;
    networkBandwidth: number;
  };
  performanceRequirements: {
    latencyThreshold: number;
    throughputRequirement: number;
    availabilityTarget: number;
  };
  complianceRequirements: string[];
  costSensitivity: 'low' | 'medium' | 'high';
  workloadType: 'web' | 'api' | 'batch' | 'ml' | 'database' | 'storage';
}

interface PlacementRecommendation {
  provider: string;
  region: string;
  instanceType: string;
  estimatedCost: number;
  performanceScore: number;
  complianceScore: number;
  latencyScore: number;
  overallScore: number;
  reasoning: string;
  pros: string[];
  cons: string[];
  migrationComplexity: 'low' | 'medium' | 'high';
  estimatedMigrationTime: string;
}

const steps = ['Workload Definition', 'Requirements', 'Compliance', 'Recommendations'];

const complianceOptions = [
  'SOC 2',
  'HIPAA',
  'PCI DSS',
  'GDPR',
  'ISO 27001',
  'FedRAMP',
  'FISMA',
];

const WorkloadPlacementWizard: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [workloadSpec, setWorkloadSpec] = useState<WorkloadSpec>({
    name: '',
    description: '',
    resourceRequirements: {
      cpu: 2,
      memory: 8,
      storage: 100,
      networkBandwidth: 1000,
    },
    performanceRequirements: {
      latencyThreshold: 100,
      throughputRequirement: 1000,
      availabilityTarget: 99.9,
    },
    complianceRequirements: [],
    costSensitivity: 'medium',
    workloadType: 'web',
  });
  const [recommendations, setRecommendations] = useState<PlacementRecommendation[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Mutation for getting placement recommendations
  const getRecommendationsMutation = useMutation(
    async (spec: WorkloadSpec) => {
      const response = await fetch('/api/ai/workload-placement/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(spec),
      });
      if (!response.ok) throw new Error('Failed to get placement recommendations');
      return response.json();
    },
    {
      onSuccess: (data) => {
        setRecommendations(data);
        setIsAnalyzing(false);
        setActiveStep(3);
      },
      onError: () => {
        setIsAnalyzing(false);
      },
    }
  );

  const handleNext = () => {
    if (activeStep === 2) {
      // Start analysis
      setIsAnalyzing(true);
      getRecommendationsMutation.mutate(workloadSpec);
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setRecommendations([]);
    setWorkloadSpec({
      name: '',
      description: '',
      resourceRequirements: {
        cpu: 2,
        memory: 8,
        storage: 100,
        networkBandwidth: 1000,
      },
      performanceRequirements: {
        latencyThreshold: 100,
        throughputRequirement: 1000,
        availabilityTarget: 99.9,
      },
      complianceRequirements: [],
      costSensitivity: 'medium',
      workloadType: 'web',
    });
  };

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'success';
    if (score >= 6) return 'warning';
    return 'error';
  };

  const renderWorkloadDefinition = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Workload Name"
          value={workloadSpec.name}
          onChange={(e) => setWorkloadSpec({ ...workloadSpec, name: e.target.value })}
          required
        />
      </Grid>
      <Grid item xs={12}>
        <TextField
          fullWidth
          multiline
          rows={3}
          label="Description"
          value={workloadSpec.description}
          onChange={(e) => setWorkloadSpec({ ...workloadSpec, description: e.target.value })}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Workload Type</InputLabel>
          <Select
            value={workloadSpec.workloadType}
            onChange={(e) => setWorkloadSpec({ ...workloadSpec, workloadType: e.target.value as any })}
          >
            <MenuItem value="web">Web Application</MenuItem>
            <MenuItem value="api">API Service</MenuItem>
            <MenuItem value="batch">Batch Processing</MenuItem>
            <MenuItem value="ml">Machine Learning</MenuItem>
            <MenuItem value="database">Database</MenuItem>
            <MenuItem value="storage">Storage Service</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Cost Sensitivity</InputLabel>
          <Select
            value={workloadSpec.costSensitivity}
            onChange={(e) => setWorkloadSpec({ ...workloadSpec, costSensitivity: e.target.value as any })}
          >
            <MenuItem value="low">Low (Performance Priority)</MenuItem>
            <MenuItem value="medium">Medium (Balanced)</MenuItem>
            <MenuItem value="high">High (Cost Priority)</MenuItem>
          </Select>
        </FormControl>
      </Grid>
    </Grid>
  );

  const renderRequirements = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Resource Requirements
        </Typography>
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>CPU Cores: {workloadSpec.resourceRequirements.cpu}</Typography>
        <Slider
          value={workloadSpec.resourceRequirements.cpu}
          onChange={(_, value) => setWorkloadSpec({
            ...workloadSpec,
            resourceRequirements: { ...workloadSpec.resourceRequirements, cpu: value as number }
          })}
          min={1}
          max={64}
          step={1}
          marks
          valueLabelDisplay="auto"
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>Memory (GB): {workloadSpec.resourceRequirements.memory}</Typography>
        <Slider
          value={workloadSpec.resourceRequirements.memory}
          onChange={(_, value) => setWorkloadSpec({
            ...workloadSpec,
            resourceRequirements: { ...workloadSpec.resourceRequirements, memory: value as number }
          })}
          min={1}
          max={256}
          step={1}
          marks
          valueLabelDisplay="auto"
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>Storage (GB): {workloadSpec.resourceRequirements.storage}</Typography>
        <Slider
          value={workloadSpec.resourceRequirements.storage}
          onChange={(_, value) => setWorkloadSpec({
            ...workloadSpec,
            resourceRequirements: { ...workloadSpec.resourceRequirements, storage: value as number }
          })}
          min={10}
          max={10000}
          step={10}
          marks
          valueLabelDisplay="auto"
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>Network Bandwidth (Mbps): {workloadSpec.resourceRequirements.networkBandwidth}</Typography>
        <Slider
          value={workloadSpec.resourceRequirements.networkBandwidth}
          onChange={(_, value) => setWorkloadSpec({
            ...workloadSpec,
            resourceRequirements: { ...workloadSpec.resourceRequirements, networkBandwidth: value as number }
          })}
          min={100}
          max={10000}
          step={100}
          marks
          valueLabelDisplay="auto"
        />
      </Grid>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          Performance Requirements
        </Typography>
      </Grid>
      <Grid item xs={12} md={4}>
        <TextField
          fullWidth
          type="number"
          label="Max Latency (ms)"
          value={workloadSpec.performanceRequirements.latencyThreshold}
          onChange={(e) => setWorkloadSpec({
            ...workloadSpec,
            performanceRequirements: {
              ...workloadSpec.performanceRequirements,
              latencyThreshold: parseInt(e.target.value)
            }
          })}
        />
      </Grid>
      <Grid item xs={12} md={4}>
        <TextField
          fullWidth
          type="number"
          label="Throughput (req/sec)"
          value={workloadSpec.performanceRequirements.throughputRequirement}
          onChange={(e) => setWorkloadSpec({
            ...workloadSpec,
            performanceRequirements: {
              ...workloadSpec.performanceRequirements,
              throughputRequirement: parseInt(e.target.value)
            }
          })}
        />
      </Grid>
      <Grid item xs={12} md={4}>
        <TextField
          fullWidth
          type="number"
          label="Availability Target (%)"
          value={workloadSpec.performanceRequirements.availabilityTarget}
          onChange={(e) => setWorkloadSpec({
            ...workloadSpec,
            performanceRequirements: {
              ...workloadSpec.performanceRequirements,
              availabilityTarget: parseFloat(e.target.value)
            }
          })}
          inputProps={{ step: 0.1, min: 90, max: 99.99 }}
        />
      </Grid>
    </Grid>
  );

  const renderCompliance = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Compliance Requirements
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Select all compliance standards that your workload must meet
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <FormGroup>
          {complianceOptions.map((option) => (
            <FormControlLabel
              key={option}
              control={
                <Checkbox
                  checked={workloadSpec.complianceRequirements.includes(option)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setWorkloadSpec({
                        ...workloadSpec,
                        complianceRequirements: [...workloadSpec.complianceRequirements, option]
                      });
                    } else {
                      setWorkloadSpec({
                        ...workloadSpec,
                        complianceRequirements: workloadSpec.complianceRequirements.filter(req => req !== option)
                      });
                    }
                  }}
                />
              }
              label={option}
            />
          ))}
        </FormGroup>
      </Grid>
      {workloadSpec.complianceRequirements.length > 0 && (
        <Grid item xs={12}>
          <Alert severity="info">
            Selected compliance requirements will filter available cloud providers and regions
          </Alert>
        </Grid>
      )}
    </Grid>
  );

  const renderRecommendations = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Placement Recommendations
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          AI-powered analysis of optimal cloud placement options for your workload
        </Typography>
      </Grid>
      {recommendations.map((rec, index) => (
        <Grid item xs={12} key={index}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box>
                  <Typography variant="h6">
                    {rec.provider} - {rec.region}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {rec.instanceType}
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="h6" color="primary">
                    ${rec.estimatedCost.toFixed(2)}/month
                  </Typography>
                  <Rating value={rec.overallScore / 2} readOnly precision={0.1} />
                </Box>
              </Box>

              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <PerformanceIcon color="primary" />
                    <Typography variant="caption" display="block">
                      Performance
                    </Typography>
                    <Chip
                      label={rec.performanceScore.toFixed(1)}
                      color={getScoreColor(rec.performanceScore) as any}
                      size="small"
                    />
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <CostIcon color="primary" />
                    <Typography variant="caption" display="block">
                      Cost
                    </Typography>
                    <Chip
                      label="8.5"
                      color="success"
                      size="small"
                    />
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <ComplianceIcon color="primary" />
                    <Typography variant="caption" display="block">
                      Compliance
                    </Typography>
                    <Chip
                      label={rec.complianceScore.toFixed(1)}
                      color={getScoreColor(rec.complianceScore) as any}
                      size="small"
                    />
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <LatencyIcon color="primary" />
                    <Typography variant="caption" display="block">
                      Latency
                    </Typography>
                    <Chip
                      label={rec.latencyScore.toFixed(1)}
                      color={getScoreColor(rec.latencyScore) as any}
                      size="small"
                    />
                  </Box>
                </Grid>
              </Grid>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Detailed Analysis</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" color="success.main" gutterBottom>
                        Advantages
                      </Typography>
                      {rec.pros.map((pro, i) => (
                        <Typography key={i} variant="body2" sx={{ mb: 0.5 }}>
                          • {pro}
                        </Typography>
                      ))}
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" color="warning.main" gutterBottom>
                        Considerations
                      </Typography>
                      {rec.cons.map((con, i) => (
                        <Typography key={i} variant="body2" sx={{ mb: 0.5 }}>
                          • {con}
                        </Typography>
                      ))}
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                        AI Reasoning
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {rec.reasoning}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2">
                        <strong>Migration Complexity:</strong> {rec.migrationComplexity}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2">
                        <strong>Estimated Migration Time:</strong> {rec.estimatedMigrationTime}
                      </Typography>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <WorkloadIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">Intelligent Workload Placement</Typography>
      </Box>

      <Card>
        <CardContent>
          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          {isAnalyzing ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" gutterBottom>
                Analyzing Workload Requirements
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                AI is evaluating optimal placement options across cloud providers...
              </Typography>
              <LinearProgress sx={{ mt: 2 }} />
            </Box>
          ) : (
            <>
              {activeStep === 0 && renderWorkloadDefinition()}
              {activeStep === 1 && renderRequirements()}
              {activeStep === 2 && renderCompliance()}
              {activeStep === 3 && renderRecommendations()}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                <Button
                  disabled={activeStep === 0}
                  onClick={handleBack}
                >
                  Back
                </Button>
                <Box>
                  {activeStep === 3 ? (
                    <Button onClick={handleReset} variant="contained">
                      Start New Analysis
                    </Button>
                  ) : (
                    <Button
                      onClick={handleNext}
                      variant="contained"
                      disabled={activeStep === 0 && !workloadSpec.name}
                    >
                      {activeStep === 2 ? 'Analyze Placement' : 'Next'}
                    </Button>
                  )}
                </Box>
              </Box>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default WorkloadPlacementWizard;