/**
 * Workload Specification Wizard Component
 * 
 * Step-by-step wizard for creating workload specifications
 * for multi-cloud cost comparison.
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Chip,
  Alert,
  CircularProgress,
  Autocomplete
} from '@mui/material';
import { useMutation } from 'react-query';
import toast from 'react-hot-toast';

import { 
  multiCloudApi, 
  WorkloadSpec, 
  ComputeSpec, 
  StorageSpec, 
  NetworkSpec, 
  DatabaseSpec, 
  UsagePatterns 
} from '../../services/multiCloudApi';

interface WorkloadSpecWizardProps {
  open: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const steps = [
  'Basic Information',
  'Compute Requirements',
  'Storage & Database',
  'Network & Usage',
  'Review & Create'
];

const regions = [
  'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
  'eu-west-1', 'eu-west-2', 'eu-central-1',
  'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1'
];

const complianceOptions = [
  'SOC2', 'GDPR', 'HIPAA', 'PCI-DSS', 'ISO-27001', 'FedRAMP'
];

const WorkloadSpecWizard: React.FC<WorkloadSpecWizardProps> = ({
  open,
  onClose,
  onComplete
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [workloadData, setWorkloadData] = useState<Partial<WorkloadSpec>>({
    name: '',
    description: '',
    additional_services: [],
    compliance_requirements: [],
    regions: [],
    compute_spec: {
      cpu_cores: 2,
      memory_gb: 8,
      operating_system: 'linux',
      architecture: 'x86_64',
      gpu_required: false
    },
    storage_spec: {
      primary_storage_gb: 100,
      storage_type: 'ssd'
    },
    network_spec: {
      bandwidth_mbps: 1000,
      data_transfer_gb_monthly: 100,
      load_balancer_required: false,
      cdn_required: false,
      vpn_required: false
    },
    usage_patterns: {
      hours_per_day: 24,
      days_per_week: 7,
      peak_usage_multiplier: 1.0,
      seasonal_variation: false,
      auto_scaling: false
    }
  });

  const [validation, setValidation] = useState<{
    isValid: boolean;
    errors: string[];
  }>({ isValid: true, errors: [] });

  // Mutation for creating workload
  const createWorkloadMutation = useMutation(
    (workload: WorkloadSpec) => multiCloudApi.compareWorkloadCosts(workload),
    {
      onSuccess: () => {
        toast.success('Workload created and cost comparison initiated');
        onComplete();
        handleReset();
      },
      onError: (error: any) => {
        toast.error(`Failed to create workload: ${error.message}`);
      }
    }
  );

  // Mutation for validation
  const validateMutation = useMutation(
    (workload: WorkloadSpec) => multiCloudApi.validateWorkloadSpecification(workload),
    {
      onSuccess: (result) => {
        setValidation({
          isValid: result.is_valid,
          errors: result.errors.map(e => e.message)
        });
      },
      onError: () => {
        setValidation({
          isValid: false,
          errors: ['Validation failed. Please check your inputs.']
        });
      }
    }
  );

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      handleSubmit();
    } else {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setWorkloadData({
      name: '',
      description: '',
      additional_services: [],
      compliance_requirements: [],
      regions: [],
      compute_spec: {
        cpu_cores: 2,
        memory_gb: 8,
        operating_system: 'linux',
        architecture: 'x86_64',
        gpu_required: false
      },
      storage_spec: {
        primary_storage_gb: 100,
        storage_type: 'ssd'
      },
      network_spec: {
        bandwidth_mbps: 1000,
        data_transfer_gb_monthly: 100,
        load_balancer_required: false,
        cdn_required: false,
        vpn_required: false
      },
      usage_patterns: {
        hours_per_day: 24,
        days_per_week: 7,
        peak_usage_multiplier: 1.0,
        seasonal_variation: false,
        auto_scaling: false
      }
    });
    setValidation({ isValid: true, errors: [] });
  };

  const handleSubmit = () => {
    if (workloadData.name && workloadData.compute_spec && workloadData.storage_spec && 
        workloadData.network_spec && workloadData.usage_patterns && workloadData.regions?.length) {
      
      // Validate before submission
      validateMutation.mutate(workloadData as WorkloadSpec);
      
      if (validation.isValid) {
        createWorkloadMutation.mutate(workloadData as WorkloadSpec);
      }
    }
  };

  const updateWorkloadData = (field: string, value: any) => {
    setWorkloadData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const updateNestedData = (section: string, field: string, value: any) => {
    setWorkloadData(prev => ({
      ...prev,
      [section]: {
        ...(prev[section as keyof typeof prev] as Record<string, any> || {}),
        [field]: value
      }
    }));
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Workload Name"
                value={workloadData.name}
                onChange={(e) => updateWorkloadData('name', e.target.value)}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={3}
                value={workloadData.description}
                onChange={(e) => updateWorkloadData('description', e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <Autocomplete
                multiple
                options={regions}
                value={workloadData.regions || []}
                onChange={(_, newValue) => updateWorkloadData('regions', newValue)}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip variant="outlined" label={option} {...getTagProps({ index })} />
                  ))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Preferred Regions"
                    placeholder="Select regions"
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Autocomplete
                multiple
                options={complianceOptions}
                value={workloadData.compliance_requirements || []}
                onChange={(_, newValue) => updateWorkloadData('compliance_requirements', newValue)}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip variant="outlined" label={option} {...getTagProps({ index })} />
                  ))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Compliance Requirements"
                    placeholder="Select compliance standards"
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="CPU Cores"
                value={workloadData.compute_spec?.cpu_cores}
                onChange={(e) => updateNestedData('compute_spec', 'cpu_cores', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 1000 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Memory (GB)"
                value={workloadData.compute_spec?.memory_gb}
                onChange={(e) => updateNestedData('compute_spec', 'memory_gb', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 10000 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Operating System</InputLabel>
                <Select
                  value={workloadData.compute_spec?.operating_system}
                  onChange={(e) => updateNestedData('compute_spec', 'operating_system', e.target.value)}
                >
                  <MenuItem value="linux">Linux</MenuItem>
                  <MenuItem value="windows">Windows</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Architecture</InputLabel>
                <Select
                  value={workloadData.compute_spec?.architecture}
                  onChange={(e) => updateNestedData('compute_spec', 'architecture', e.target.value)}
                >
                  <MenuItem value="x86_64">x86_64</MenuItem>
                  <MenuItem value="arm64">ARM64</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={workloadData.compute_spec?.gpu_required}
                    onChange={(e) => updateNestedData('compute_spec', 'gpu_required', e.target.checked)}
                  />
                }
                label="GPU Required"
              />
            </Grid>
            {workloadData.compute_spec?.gpu_required && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="GPU Type"
                  value={workloadData.compute_spec?.gpu_type || ''}
                  onChange={(e) => updateNestedData('compute_spec', 'gpu_type', e.target.value)}
                  placeholder="e.g., NVIDIA V100, T4"
                />
              </Grid>
            )}
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Storage Requirements</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Primary Storage (GB)"
                value={workloadData.storage_spec?.primary_storage_gb}
                onChange={(e) => updateNestedData('storage_spec', 'primary_storage_gb', parseInt(e.target.value))}
                inputProps={{ min: 1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Storage Type</InputLabel>
                <Select
                  value={workloadData.storage_spec?.storage_type}
                  onChange={(e) => updateNestedData('storage_spec', 'storage_type', e.target.value)}
                >
                  <MenuItem value="ssd">SSD</MenuItem>
                  <MenuItem value="hdd">HDD</MenuItem>
                  <MenuItem value="nvme">NVMe</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Backup Storage (GB)"
                value={workloadData.storage_spec?.backup_storage_gb || ''}
                onChange={(e) => updateNestedData('storage_spec', 'backup_storage_gb', parseInt(e.target.value) || undefined)}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="IOPS Requirement"
                value={workloadData.storage_spec?.iops_requirement || ''}
                onChange={(e) => updateNestedData('storage_spec', 'iops_requirement', parseInt(e.target.value) || undefined)}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Database (Optional)</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Database Type"
                value={workloadData.database_spec?.database_type || ''}
                onChange={(e) => updateNestedData('database_spec', 'database_type', e.target.value)}
                placeholder="e.g., mysql, postgresql"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Database Storage (GB)"
                value={workloadData.database_spec?.storage_gb || ''}
                onChange={(e) => updateNestedData('database_spec', 'storage_gb', parseInt(e.target.value) || undefined)}
              />
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Network Requirements</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Bandwidth (Mbps)"
                value={workloadData.network_spec?.bandwidth_mbps}
                onChange={(e) => updateNestedData('network_spec', 'bandwidth_mbps', parseInt(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Data Transfer (GB/month)"
                value={workloadData.network_spec?.data_transfer_gb_monthly}
                onChange={(e) => updateNestedData('network_spec', 'data_transfer_gb_monthly', parseInt(e.target.value))}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={workloadData.network_spec?.load_balancer_required}
                    onChange={(e) => updateNestedData('network_spec', 'load_balancer_required', e.target.checked)}
                  />
                }
                label="Load Balancer Required"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={workloadData.network_spec?.cdn_required}
                    onChange={(e) => updateNestedData('network_spec', 'cdn_required', e.target.checked)}
                  />
                }
                label="CDN Required"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Usage Patterns</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Hours per Day"
                value={workloadData.usage_patterns?.hours_per_day}
                onChange={(e) => updateNestedData('usage_patterns', 'hours_per_day', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 24 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Days per Week"
                value={workloadData.usage_patterns?.days_per_week}
                onChange={(e) => updateNestedData('usage_patterns', 'days_per_week', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 7 }}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={workloadData.usage_patterns?.auto_scaling}
                    onChange={(e) => updateNestedData('usage_patterns', 'auto_scaling', e.target.checked)}
                  />
                }
                label="Auto Scaling Enabled"
              />
            </Grid>
          </Grid>
        );

      case 4:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>Review Your Workload Specification</Typography>
            
            {!validation.isValid && (
              <Alert severity="error" sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Validation Errors:</Typography>
                <ul>
                  {validation.errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </Alert>
            )}
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Name:</Typography>
                <Typography>{workloadData.name}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Regions:</Typography>
                <Typography>{workloadData.regions?.join(', ')}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">CPU Cores:</Typography>
                <Typography>{workloadData.compute_spec?.cpu_cores}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Memory:</Typography>
                <Typography>{workloadData.compute_spec?.memory_gb} GB</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Storage:</Typography>
                <Typography>{workloadData.storage_spec?.primary_storage_gb} GB {workloadData.storage_spec?.storage_type}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Data Transfer:</Typography>
                <Typography>{workloadData.network_spec?.data_transfer_gb_monthly} GB/month</Typography>
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Create Workload Specification</DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%', mt: 2 }}>
          <Stepper activeStep={activeStep}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
          
          <Box sx={{ mt: 3, mb: 2 }}>
            {renderStepContent(activeStep)}
          </Box>
        </Box>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={handleNext}
          disabled={createWorkloadMutation.isLoading || validateMutation.isLoading}
        >
          {createWorkloadMutation.isLoading || validateMutation.isLoading ? (
            <CircularProgress size={20} />
          ) : activeStep === steps.length - 1 ? (
            'Create & Compare'
          ) : (
            'Next'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default WorkloadSpecWizard;