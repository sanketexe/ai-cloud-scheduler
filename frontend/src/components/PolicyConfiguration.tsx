import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Button,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormGroup,
  Checkbox,
  Slider,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  Schedule as ScheduleIcon,
  FilterList as FilterIcon,
  Policy as PolicyIcon,
} from '@mui/icons-material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs, { Dayjs } from 'dayjs';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';

// Types
interface AutomationPolicy {
  policy_id: string;
  name: string;
  automation_level: 'conservative' | 'balanced' | 'aggressive';
  enabled_actions: string[];
  approval_required_actions: string[];
  blocked_actions: string[];
  resource_filters: {
    exclude_tags: string[];
    include_services: string[];
    min_cost_threshold: number;
  };
  time_restrictions: {
    business_hours: {
      enabled: boolean;
      timezone: string;
      start_time: string;
      end_time: string;
      days: string[];
    };
    maintenance_windows: Array<{
      name: string;
      start_time: string;
      end_time: string;
      days: string[];
    }>;
    blackout_periods: Array<{
      name: string;
      start_date: string;
      end_date: string;
    }>;
  };
  safety_overrides: {
    production_tag_protection: boolean;
    auto_scaling_protection: boolean;
    load_balancer_protection: boolean;
  };
  created_at: string;
  updated_at: string;
}

interface PolicyConfigurationProps {
  onPolicyChange?: (policy: AutomationPolicy) => void;
}

const AVAILABLE_ACTIONS = [
  'stop_unused_instances',
  'terminate_zombie_instances',
  'resize_underutilized_instances',
  'delete_unattached_volumes',
  'upgrade_gp2_to_gp3',
  'release_unused_elastic_ips',
  'delete_unused_load_balancers',
  'cleanup_unused_security_groups',
];

const AWS_SERVICES = [
  'EC2',
  'EBS',
  'EIP',
  'ELB',
  'RDS',
  'S3',
  'Lambda',
  'CloudWatch',
];

const DAYS_OF_WEEK = [
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
];

const PolicyConfiguration: React.FC<PolicyConfigurationProps> = ({ onPolicyChange }) => {
  const [selectedPolicy, setSelectedPolicy] = useState<AutomationPolicy | null>(null);
  const [editingPolicy, setEditingPolicy] = useState<AutomationPolicy | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [newPolicyDialog, setNewPolicyDialog] = useState(false);
  const queryClient = useQueryClient();

  // Fetch policies
  const { data: policies, isLoading } = useQuery<AutomationPolicy[]>(
    'automation-policies',
    async () => {
      const response = await fetch('/api/automation/policies');
      if (!response.ok) throw new Error('Failed to fetch policies');
      return response.json();
    }
  );

  // Save policy mutation
  const savePolicyMutation = useMutation(
    async (policy: AutomationPolicy) => {
      const url = policy.policy_id 
        ? `/api/automation/policies/${policy.policy_id}`
        : '/api/automation/policies';
      const method = policy.policy_id ? 'PUT' : 'POST';
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(policy),
      });
      if (!response.ok) throw new Error('Failed to save policy');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('automation-policies');
        toast.success('Policy saved successfully');
        setIsEditing(false);
        setNewPolicyDialog(false);
        setEditingPolicy(null);
      },
      onError: () => {
        toast.error('Failed to save policy');
      },
    }
  );

  // Delete policy mutation
  const deletePolicyMutation = useMutation(
    async (policyId: string) => {
      const response = await fetch(`/api/automation/policies/${policyId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete policy');
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('automation-policies');
        toast.success('Policy deleted successfully');
        setSelectedPolicy(null);
      },
      onError: () => {
        toast.error('Failed to delete policy');
      },
    }
  );

  const createNewPolicy = (): AutomationPolicy => ({
    policy_id: '',
    name: 'New Policy',
    automation_level: 'balanced',
    enabled_actions: [],
    approval_required_actions: [],
    blocked_actions: [],
    resource_filters: {
      exclude_tags: [],
      include_services: AWS_SERVICES,
      min_cost_threshold: 10.0,
    },
    time_restrictions: {
      business_hours: {
        enabled: true,
        timezone: 'UTC',
        start_time: '09:00',
        end_time: '17:00',
        days: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
      },
      maintenance_windows: [],
      blackout_periods: [],
    },
    safety_overrides: {
      production_tag_protection: true,
      auto_scaling_protection: true,
      load_balancer_protection: true,
    },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  const handleEditPolicy = (policy: AutomationPolicy) => {
    setEditingPolicy({ ...policy });
    setIsEditing(true);
  };

  const handleSavePolicy = () => {
    if (editingPolicy) {
      savePolicyMutation.mutate(editingPolicy);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditingPolicy(null);
  };

  const handleNewPolicy = () => {
    setEditingPolicy(createNewPolicy());
    setNewPolicyDialog(true);
  };

  const updateEditingPolicy = (updates: Partial<AutomationPolicy>) => {
    if (editingPolicy) {
      setEditingPolicy({ ...editingPolicy, ...updates });
    }
  };

  const handleActionChange = (action: string, category: 'enabled' | 'approval' | 'blocked') => {
    if (!editingPolicy) return;

    const newPolicy = { ...editingPolicy };
    
    // Remove from all categories first
    newPolicy.enabled_actions = newPolicy.enabled_actions.filter(a => a !== action);
    newPolicy.approval_required_actions = newPolicy.approval_required_actions.filter(a => a !== action);
    newPolicy.blocked_actions = newPolicy.blocked_actions.filter(a => a !== action);

    // Add to selected category
    switch (category) {
      case 'enabled':
        newPolicy.enabled_actions.push(action);
        break;
      case 'approval':
        newPolicy.approval_required_actions.push(action);
        break;
      case 'blocked':
        newPolicy.blocked_actions.push(action);
        break;
    }

    setEditingPolicy(newPolicy);
  };

  const getActionCategory = (action: string): 'enabled' | 'approval' | 'blocked' | 'none' => {
    if (!editingPolicy) return 'none';
    
    if (editingPolicy.enabled_actions.includes(action)) return 'enabled';
    if (editingPolicy.approval_required_actions.includes(action)) return 'approval';
    if (editingPolicy.blocked_actions.includes(action)) return 'blocked';
    return 'none';
  };

  if (isLoading) {
    return <Typography>Loading policies...</Typography>;
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box>
        {/* Header */}
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
              Policy Configuration
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Configure automation policies and safety rules
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleNewPolicy}
          >
            New Policy
          </Button>
        </Box>

        <Grid container spacing={3}>
          {/* Policy List */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardHeader title="Automation Policies" />
              <CardContent>
                <List>
                  {policies?.map((policy) => (
                    <React.Fragment key={policy.policy_id}>
                      <ListItem
                        button
                        selected={selectedPolicy?.policy_id === policy.policy_id}
                        onClick={() => setSelectedPolicy(policy)}
                      >
                        <ListItemText
                          primary={policy.name}
                          secondary={`Level: ${policy.automation_level}`}
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            size="small"
                            onClick={() => handleEditPolicy(policy)}
                          >
                            <EditIcon />
                          </IconButton>
                          <IconButton
                            size="small"
                            onClick={() => deletePolicyMutation.mutate(policy.policy_id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Policy Details/Editor */}
          <Grid item xs={12} md={8}>
            {(selectedPolicy || isEditing) && (
              <Card>
                <CardHeader
                  title={isEditing ? 'Edit Policy' : 'Policy Details'}
                  action={
                    isEditing ? (
                      <Box>
                        <IconButton onClick={handleSavePolicy}>
                          <SaveIcon />
                        </IconButton>
                        <IconButton onClick={handleCancelEdit}>
                          <CancelIcon />
                        </IconButton>
                      </Box>
                    ) : (
                      <IconButton onClick={() => selectedPolicy && handleEditPolicy(selectedPolicy)}>
                        <EditIcon />
                      </IconButton>
                    )
                  }
                />
                <CardContent>
                  {isEditing && editingPolicy ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {/* Basic Settings */}
                      <Accordion defaultExpanded>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Basic Settings</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            <Grid item xs={12}>
                              <TextField
                                fullWidth
                                label="Policy Name"
                                value={editingPolicy.name}
                                onChange={(e) => updateEditingPolicy({ name: e.target.value })}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <FormControl fullWidth>
                                <InputLabel>Automation Level</InputLabel>
                                <Select
                                  value={editingPolicy.automation_level}
                                  onChange={(e) => updateEditingPolicy({ 
                                    automation_level: e.target.value as any 
                                  })}
                                >
                                  <MenuItem value="conservative">Conservative</MenuItem>
                                  <MenuItem value="balanced">Balanced</MenuItem>
                                  <MenuItem value="aggressive">Aggressive</MenuItem>
                                </Select>
                              </FormControl>
                            </Grid>
                          </Grid>
                        </AccordionDetails>
                      </Accordion>

                      {/* Action Configuration */}
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Action Configuration</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Configure how each optimization action should be handled
                          </Typography>
                          {AVAILABLE_ACTIONS.map((action) => (
                            <Box key={action} sx={{ mb: 2 }}>
                              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                {action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </Typography>
                              <FormControl component="fieldset">
                                <Grid container spacing={1}>
                                  <Grid item>
                                    <FormControlLabel
                                      control={
                                        <Checkbox
                                          checked={getActionCategory(action) === 'enabled'}
                                          onChange={() => handleActionChange(action, 'enabled')}
                                        />
                                      }
                                      label="Auto Execute"
                                    />
                                  </Grid>
                                  <Grid item>
                                    <FormControlLabel
                                      control={
                                        <Checkbox
                                          checked={getActionCategory(action) === 'approval'}
                                          onChange={() => handleActionChange(action, 'approval')}
                                        />
                                      }
                                      label="Require Approval"
                                    />
                                  </Grid>
                                  <Grid item>
                                    <FormControlLabel
                                      control={
                                        <Checkbox
                                          checked={getActionCategory(action) === 'blocked'}
                                          onChange={() => handleActionChange(action, 'blocked')}
                                        />
                                      }
                                      label="Block"
                                    />
                                  </Grid>
                                </Grid>
                              </FormControl>
                            </Box>
                          ))}
                        </AccordionDetails>
                      </Accordion>

                      {/* Resource Filters */}
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Resource Filters</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            <Grid item xs={12}>
                              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                Minimum Cost Threshold ($)
                              </Typography>
                              <Slider
                                value={editingPolicy.resource_filters.min_cost_threshold}
                                onChange={(_, value) => updateEditingPolicy({
                                  resource_filters: {
                                    ...editingPolicy.resource_filters,
                                    min_cost_threshold: value as number,
                                  }
                                })}
                                min={0}
                                max={1000}
                                step={10}
                                marks={[
                                  { value: 0, label: '$0' },
                                  { value: 100, label: '$100' },
                                  { value: 500, label: '$500' },
                                  { value: 1000, label: '$1000' },
                                ]}
                                valueLabelDisplay="on"
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                Include Services
                              </Typography>
                              <FormGroup row>
                                {AWS_SERVICES.map((service) => (
                                  <FormControlLabel
                                    key={service}
                                    control={
                                      <Checkbox
                                        checked={editingPolicy.resource_filters.include_services.includes(service)}
                                        onChange={(e) => {
                                          const services = e.target.checked
                                            ? [...editingPolicy.resource_filters.include_services, service]
                                            : editingPolicy.resource_filters.include_services.filter(s => s !== service);
                                          updateEditingPolicy({
                                            resource_filters: {
                                              ...editingPolicy.resource_filters,
                                              include_services: services,
                                            }
                                          });
                                        }}
                                      />
                                    }
                                    label={service}
                                  />
                                ))}
                              </FormGroup>
                            </Grid>
                          </Grid>
                        </AccordionDetails>
                      </Accordion>

                      {/* Time Restrictions */}
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Time Restrictions</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            <Grid item xs={12}>
                              <FormControlLabel
                                control={
                                  <Switch
                                    checked={editingPolicy.time_restrictions.business_hours.enabled}
                                    onChange={(e) => updateEditingPolicy({
                                      time_restrictions: {
                                        ...editingPolicy.time_restrictions,
                                        business_hours: {
                                          ...editingPolicy.time_restrictions.business_hours,
                                          enabled: e.target.checked,
                                        }
                                      }
                                    })}
                                  />
                                }
                                label="Respect Business Hours"
                              />
                            </Grid>
                            {editingPolicy.time_restrictions.business_hours.enabled && (
                              <>
                                <Grid item xs={6}>
                                  <TextField
                                    fullWidth
                                    label="Start Time"
                                    type="time"
                                    value={editingPolicy.time_restrictions.business_hours.start_time}
                                    onChange={(e) => updateEditingPolicy({
                                      time_restrictions: {
                                        ...editingPolicy.time_restrictions,
                                        business_hours: {
                                          ...editingPolicy.time_restrictions.business_hours,
                                          start_time: e.target.value,
                                        }
                                      }
                                    })}
                                    InputLabelProps={{ shrink: true }}
                                  />
                                </Grid>
                                <Grid item xs={6}>
                                  <TextField
                                    fullWidth
                                    label="End Time"
                                    type="time"
                                    value={editingPolicy.time_restrictions.business_hours.end_time}
                                    onChange={(e) => updateEditingPolicy({
                                      time_restrictions: {
                                        ...editingPolicy.time_restrictions,
                                        business_hours: {
                                          ...editingPolicy.time_restrictions.business_hours,
                                          end_time: e.target.value,
                                        }
                                      }
                                    })}
                                    InputLabelProps={{ shrink: true }}
                                  />
                                </Grid>
                                <Grid item xs={12}>
                                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                    Business Days
                                  </Typography>
                                  <FormGroup row>
                                    {DAYS_OF_WEEK.map((day) => (
                                      <FormControlLabel
                                        key={day}
                                        control={
                                          <Checkbox
                                            checked={editingPolicy.time_restrictions.business_hours.days.includes(day)}
                                            onChange={(e) => {
                                              const days = e.target.checked
                                                ? [...editingPolicy.time_restrictions.business_hours.days, day]
                                                : editingPolicy.time_restrictions.business_hours.days.filter(d => d !== day);
                                              updateEditingPolicy({
                                                time_restrictions: {
                                                  ...editingPolicy.time_restrictions,
                                                  business_hours: {
                                                    ...editingPolicy.time_restrictions.business_hours,
                                                    days,
                                                  }
                                                }
                                              });
                                            }}
                                          />
                                        }
                                        label={day.charAt(0).toUpperCase() + day.slice(1)}
                                      />
                                    ))}
                                  </FormGroup>
                                </Grid>
                              </>
                            )}
                          </Grid>
                        </AccordionDetails>
                      </Accordion>

                      {/* Safety Overrides */}
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Safety Overrides</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <FormGroup>
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={editingPolicy.safety_overrides.production_tag_protection}
                                  onChange={(e) => updateEditingPolicy({
                                    safety_overrides: {
                                      ...editingPolicy.safety_overrides,
                                      production_tag_protection: e.target.checked,
                                    }
                                  })}
                                />
                              }
                              label="Production Tag Protection"
                            />
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={editingPolicy.safety_overrides.auto_scaling_protection}
                                  onChange={(e) => updateEditingPolicy({
                                    safety_overrides: {
                                      ...editingPolicy.safety_overrides,
                                      auto_scaling_protection: e.target.checked,
                                    }
                                  })}
                                />
                              }
                              label="Auto Scaling Group Protection"
                            />
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={editingPolicy.safety_overrides.load_balancer_protection}
                                  onChange={(e) => updateEditingPolicy({
                                    safety_overrides: {
                                      ...editingPolicy.safety_overrides,
                                      load_balancer_protection: e.target.checked,
                                    }
                                  })}
                                />
                              }
                              label="Load Balancer Protection"
                            />
                          </FormGroup>
                        </AccordionDetails>
                      </Accordion>
                    </Box>
                  ) : selectedPolicy ? (
                    <Box>
                      <Grid container spacing={2}>
                        <Grid item xs={12}>
                          <Typography variant="h6">{selectedPolicy.name}</Typography>
                          <Chip
                            label={selectedPolicy.automation_level}
                            color="primary"
                            sx={{ mt: 1 }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Enabled Actions ({selectedPolicy.enabled_actions.length})
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                            {selectedPolicy.enabled_actions.map((action) => (
                              <Chip
                                key={action}
                                label={action.replace(/_/g, ' ')}
                                color="success"
                                size="small"
                              />
                            ))}
                          </Box>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Approval Required ({selectedPolicy.approval_required_actions.length})
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                            {selectedPolicy.approval_required_actions.map((action) => (
                              <Chip
                                key={action}
                                label={action.replace(/_/g, ' ')}
                                color="warning"
                                size="small"
                              />
                            ))}
                          </Box>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Blocked Actions ({selectedPolicy.blocked_actions.length})
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                            {selectedPolicy.blocked_actions.map((action) => (
                              <Chip
                                key={action}
                                label={action.replace(/_/g, ' ')}
                                color="error"
                                size="small"
                              />
                            ))}
                          </Box>
                        </Grid>
                      </Grid>
                    </Box>
                  ) : null}
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>

        {/* New Policy Dialog */}
        <Dialog
          open={newPolicyDialog}
          onClose={() => setNewPolicyDialog(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Policy</DialogTitle>
          <DialogContent>
            {editingPolicy && (
              <TextField
                fullWidth
                label="Policy Name"
                value={editingPolicy.name}
                onChange={(e) => updateEditingPolicy({ name: e.target.value })}
                sx={{ mt: 2 }}
              />
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setNewPolicyDialog(false)}>Cancel</Button>
            <Button
              variant="contained"
              onClick={() => editingPolicy && savePolicyMutation.mutate(editingPolicy)}
            >
              Create
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </LocalizationProvider>
  );
};

export default PolicyConfiguration;