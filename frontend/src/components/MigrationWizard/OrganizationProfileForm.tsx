import React, { useEffect } from 'react';
import {
  Box,
  TextField,
  MenuItem,
  Typography,
  Grid,
  Chip,
  FormControl,
  InputLabel,
  Select,
  OutlinedInput,
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';

interface OrganizationProfileFormProps {
  data: any;
  onChange: (data: any) => void;
}

const companySizes = ['SMALL', 'MEDIUM', 'LARGE', 'ENTERPRISE'];
// Focused on Physical to Cloud migration
const infrastructureTypes = ['ON_PREMISES'];
const experienceLevels = ['BEGINNER', 'INTERMEDIATE', 'ADVANCED'];
const industries = [
  'Technology',
  'Finance',
  'Healthcare',
  'Retail',
  'Manufacturing',
  'Education',
  'Government',
  'Other',
];
const regions = [
  'North America',
  'Europe',
  'Asia Pacific',
  'Latin America',
  'Middle East',
  'Africa',
];

const OrganizationProfileForm: React.FC<OrganizationProfileFormProps> = ({
  data,
  onChange,
}) => {
  const defaultValues = {
    company_size: 'MEDIUM',
    industry: 'Technology',
    current_infrastructure: 'ON_PREMISES',
    geographic_presence: ['North America'],
    it_team_size: 10,
    cloud_experience_level: 'BEGINNER',
  };

  const { control, watch, setValue, reset } = useForm({
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
        Tell Us About Your Organization
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        This information helps us understand your company's size, industry, and cloud readiness.
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Controller
            name="company_size"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                select
                fullWidth
                label="Company Size"
                helperText="How many employees work at your organization?"
              >
                {companySizes.map((size) => (
                  <MenuItem key={size} value={size}>
                    {size === 'SMALL' && '1-50 employees'}
                    {size === 'MEDIUM' && '51-500 employees'}
                    {size === 'LARGE' && '501-5,000 employees'}
                    {size === 'ENTERPRISE' && '5,000+ employees'}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="industry"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                select
                fullWidth
                label="Industry"
                helperText="What industry does your organization operate in?"
              >
                {industries.map((industry) => (
                  <MenuItem key={industry} value={industry}>
                    {industry}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="current_infrastructure"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                select
                fullWidth
                label="Current Infrastructure"
                helperText="Where is your infrastructure currently hosted?"
              >
                {infrastructureTypes.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type === 'ON_PREMISES' && 'On-Premises (Physical Servers)'}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="it_team_size"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="IT Team Size"
                helperText="How many IT/DevOps staff do you have?"
                inputProps={{ min: 1 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="cloud_experience_level"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                select
                fullWidth
                label="Cloud Experience Level"
                helperText="What's your team's experience with cloud platforms?"
              >
                {experienceLevels.map((level) => (
                  <MenuItem key={level} value={level}>
                    {level === 'BEGINNER' && 'Beginner - New to cloud'}
                    {level === 'INTERMEDIATE' && 'Intermediate - Some cloud experience'}
                    {level === 'ADVANCED' && 'Advanced - Extensive cloud experience'}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="geographic_presence"
            control={control}
            render={({ field }) => (
              <FormControl fullWidth>
                <InputLabel>Geographic Presence</InputLabel>
                <Select
                  {...field}
                  multiple
                  input={<OutlinedInput label="Geographic Presence" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {(selected as string[]).map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {regions.map((region) => (
                    <MenuItem key={region} value={region}>
                      {region}
                    </MenuItem>
                  ))}
                </Select>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, ml: 1.5 }}>
                  Where do your users and operations exist?
                </Typography>
              </FormControl>
            )}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default OrganizationProfileForm;
