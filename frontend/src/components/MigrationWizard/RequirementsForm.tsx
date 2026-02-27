import React, { useEffect } from 'react';
import {
  Box,
  TextField,
  Typography,
  Grid,
  Chip,
  FormControl,
  InputLabel,
  Select,
  OutlinedInput,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useForm, Controller } from 'react-hook-form';

interface RequirementsFormProps {
  data: any;
  onChange: (data: any) => void;
}

const regulatoryFrameworks = ['GDPR', 'HIPAA', 'SOC2', 'PCI-DSS', 'ISO 27001', 'FedRAMP'];
const securityStandards = ['SOC 2', 'ISO 27001', 'PCI DSS', 'NIST', 'CIS'];
const cloudServices = [
  'Compute',
  'Storage',
  'Database',
  'Networking',
  'Machine Learning',
  'Analytics',
  'Containers',
  'Serverless',
  'IoT',
  'CDN',
];
const costPriorities = ['LOW', 'MEDIUM', 'HIGH'];

const RequirementsForm: React.FC<RequirementsFormProps> = ({
  data,
  onChange,
}) => {
  const defaultValues = {
    performance: {
      latency_target_ms: 100,
      availability_target: 99.9,
      disaster_recovery_rto_minutes: 60,
      disaster_recovery_rpo_minutes: 15,
      geographic_distribution: [],
    },
    compliance: {
      regulatory_frameworks: [],
      data_residency_requirements: [],
      industry_certifications: [],
      security_standards: [],
    },
    budget: {
      current_monthly_cost: 5000,
      migration_budget: 50000,
      target_monthly_cost: 4000,
      cost_optimization_priority: 'MEDIUM',
    },
    technical: {
      required_services: ['Compute', 'Storage', 'Database'],
      ml_ai_required: false,
      analytics_required: false,
      container_orchestration: false,
      serverless_required: false,
    },
  };

  const { control, watch, reset } = useForm({
    defaultValues: data || defaultValues,
  });

  const formData = watch();

  useEffect(() => {
    onChange(formData);
  }, [formData, onChange]);

  // Reset form when data prop changes
  useEffect(() => {
    if (data) {
      reset(data);
    }
  }, [data, reset]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Requirements
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Define your performance, compliance, budget, and technical requirements.
      </Typography>

      <Box sx={{ mt: 2 }}>
        {/* Performance Requirements */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">Performance Requirements</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Controller
                  name="performance.latency_target_ms"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="Latency Target (ms)"
                      helperText="Maximum acceptable latency"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Controller
                  name="performance.availability_target"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="Availability Target (%)"
                      helperText="e.g., 99.9, 99.99"
                      inputProps={{ min: 90, max: 100, step: 0.1 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Controller
                  name="performance.disaster_recovery_rto_minutes"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="RTO (minutes)"
                      helperText="Recovery Time Objective"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Controller
                  name="performance.disaster_recovery_rpo_minutes"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="RPO (minutes)"
                      helperText="Recovery Point Objective"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Compliance Requirements */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">Compliance Requirements</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Controller
                  name="compliance.regulatory_frameworks"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth>
                      <InputLabel>Regulatory Frameworks</InputLabel>
                      <Select
                        {...field}
                        multiple
                        input={<OutlinedInput label="Regulatory Frameworks" />}
                        renderValue={(selected) => (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {(selected as string[]).map((value) => (
                              <Chip key={value} label={value} size="small" />
                            ))}
                          </Box>
                        )}
                      >
                        {regulatoryFrameworks.map((framework) => (
                          <MenuItem key={framework} value={framework}>
                            {framework}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="compliance.security_standards"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth>
                      <InputLabel>Security Standards</InputLabel>
                      <Select
                        {...field}
                        multiple
                        input={<OutlinedInput label="Security Standards" />}
                        renderValue={(selected) => (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {(selected as string[]).map((value) => (
                              <Chip key={value} label={value} size="small" />
                            ))}
                          </Box>
                        )}
                      >
                        {securityStandards.map((standard) => (
                          <MenuItem key={standard} value={standard}>
                            {standard}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Budget Constraints */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">Budget Constraints</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Controller
                  name="budget.current_monthly_cost"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="Current Monthly Cost ($)"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12} md={4}>
                <Controller
                  name="budget.migration_budget"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="Migration Budget ($)"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12} md={4}>
                <Controller
                  name="budget.target_monthly_cost"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      fullWidth
                      label="Target Monthly Cost ($)"
                      inputProps={{ min: 0 }}
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="budget.cost_optimization_priority"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      select
                      fullWidth
                      label="Cost Optimization Priority"
                    >
                      {costPriorities.map((priority) => (
                        <MenuItem key={priority} value={priority}>
                          {priority}
                        </MenuItem>
                      ))}
                    </TextField>
                  )}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Technical Requirements */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">Technical Requirements</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Controller
                  name="technical.required_services"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth>
                      <InputLabel>Required Services</InputLabel>
                      <Select
                        {...field}
                        multiple
                        input={<OutlinedInput label="Required Services" />}
                        renderValue={(selected) => (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {(selected as string[]).map((value) => (
                              <Chip key={value} label={value} size="small" />
                            ))}
                          </Box>
                        )}
                      >
                        {cloudServices.map((service) => (
                          <MenuItem key={service} value={service}>
                            {service}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="technical.ml_ai_required"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Machine Learning / AI Required"
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="technical.analytics_required"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Analytics Required"
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="technical.container_orchestration"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Container Orchestration (Kubernetes)"
                    />
                  )}
                />
              </Grid>

              <Grid item xs={12}>
                <Controller
                  name="technical.serverless_required"
                  control={control}
                  render={({ field }) => (
                    <FormControlLabel
                      control={<Checkbox {...field} checked={field.value} />}
                      label="Serverless Computing"
                    />
                  )}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </Box>
    </Box>
  );
};

export default RequirementsForm;
