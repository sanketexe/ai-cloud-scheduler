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
const infrastructureTypes = ['ON_PREMISES', 'CLOUD', 'HYBRID'];
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
  const { control, watch, setValue } = useForm({
    defaultValues: data || {
      company_size: 'MEDIUM',
      industry: 'Technology',
      current_infrastructure: 'ON_PREMISES',
      geographic_presence: [],
      it_team_size: 10,
      cloud_experience_level: 'BEGINNER',
    },
  });

  const formData = watch();

  useEffect(() => {
    onChange(formData);
  }, [formData, onChange]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Organization Profile
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Tell us about your organization to help us understand your migration needs.
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
                helperText="Select your organization size"
              >
                {companySizes.map((size) => (
                  <MenuItem key={size} value={size}>
                    {size.replace('_', ' ')}
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
                helperText="Select your industry"
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
                    {type.replace('_', ' ')}
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
                helperText="Number of IT staff members"
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
                helperText="Your team's cloud expertise"
              >
                {experienceLevels.map((level) => (
                  <MenuItem key={level} value={level}>
                    {level}
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
              </FormControl>
            )}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default OrganizationProfileForm;
