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
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';

interface WorkloadProfileFormProps {
  data: any;
  onChange: (data: any) => void;
}

const databaseTypes = [
  'PostgreSQL',
  'MySQL',
  'MongoDB',
  'Redis',
  'Oracle',
  'SQL Server',
  'Cassandra',
  'DynamoDB',
];

const WorkloadProfileForm: React.FC<WorkloadProfileFormProps> = ({
  data,
  onChange,
}) => {
  const defaultValues = {
    total_compute_cores: 4,
    total_memory_gb: 16,
    total_storage_tb: 1,
    database_types: [],
    data_volume_tb: 0.5,
    peak_transaction_rate: 100,
    // Physical Infrastructure Fields
    physical_servers: 2,
    power_consumption_watts: 500,
    hardware_age_years: 3,
    storage_type: 'HDD', // HDD, SSD, Hybrid
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
        Workload Profile
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Provide details about your current workload and resource requirements.
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Controller
            name="total_compute_cores"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Total Compute Cores"
                helperText="Total CPU cores across all systems"
                inputProps={{ min: 0 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="total_memory_gb"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Total Memory (GB)"
                helperText="Total RAM across all systems"
                inputProps={{ min: 0 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="total_storage_tb"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Total Storage (TB)"
                helperText="Total storage capacity"
                inputProps={{ min: 0, step: 0.1 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="data_volume_tb"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Data Volume (TB)"
                helperText="Total data to be migrated"
                inputProps={{ min: 0, step: 0.1 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="peak_transaction_rate"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Peak Transaction Rate"
                helperText="Transactions per second at peak"
                inputProps={{ min: 0 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12}>
          <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
            Physical Infrastructure Details (For TCO Calculation)
          </Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="physical_servers"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Number of Physical Servers"
                helperText="Total count of physical machines"
                inputProps={{ min: 1 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="power_consumption_watts"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Avg. Power per Server (Watts)"
                helperText="Average power consumption per server"
                inputProps={{ min: 0 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="hardware_age_years"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                type="number"
                fullWidth
                label="Hardware Age (Years)"
                helperText="Average age of current hardware"
                inputProps={{ min: 0 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="storage_type"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                select
                fullWidth
                label="Current Storage Type"
                helperText="Primary storage technology used"
              >
                <MenuItem value="HDD">HDD (Hard Disk Drive)</MenuItem>
                <MenuItem value="SSD">SSD (Solid State Drive)</MenuItem>
                <MenuItem value="SAS">SAS (Serial Attached SCSI)</MenuItem>
                <MenuItem value="Hybrid">Hybrid / Tiered</MenuItem>
              </TextField>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="database_types"
            control={control}
            render={({ field }) => (
              <FormControl fullWidth>
                <InputLabel>Database Types</InputLabel>
                <Select
                  {...field}
                  multiple
                  input={<OutlinedInput label="Database Types" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {(selected as string[]).map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {databaseTypes.map((db) => (
                    <MenuItem key={db} value={db}>
                      {db}
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

export default WorkloadProfileForm;
