import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  ExpandMore,
  FilterList,
  Add,
  Delete,
  Download,
  ViewModule,
  ViewList,
  Business,
  Group,
  LocationOn,
  Category,
  CloudQueue,
} from '@mui/icons-material';
import { migrationApi, DimensionalView } from '../services/migrationApi';
import toast from 'react-hot-toast';

interface FilterCondition {
  id: string;
  field: string;
  operator: string;
  value: string;
  logicalOperator?: 'AND' | 'OR';
}

interface ReportConfig {
  name: string;
  dimensions: string[];
  filters: FilterCondition[];
  groupBy: string[];
  sortBy: string;
}

const DimensionalFiltering: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [loading, setLoading] = useState(false);
  const [selectedDimension, setSelectedDimension] = useState<string>('team');
  const [dimensionalView, setDimensionalView] = useState<DimensionalView | null>(null);
  const [filters, setFilters] = useState<FilterCondition[]>([]);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [reportConfig, setReportConfig] = useState<ReportConfig>({
    name: '',
    dimensions: [],
    filters: [],
    groupBy: [],
    sortBy: 'cost',
  });

  const dimensions = [
    { value: 'team', label: 'Team', icon: <Group /> },
    { value: 'project', label: 'Project', icon: <Business /> },
    { value: 'region', label: 'Region', icon: <LocationOn /> },
    { value: 'category', label: 'Category', icon: <Category /> },
    { value: 'provider', label: 'Provider', icon: <CloudQueue /> },
  ];

  const filterFields = [
    'resource_type',
    'provider',
    'region',
    'team',
    'project',
    'category',
    'cost_monthly',
  ];

  const operators = ['equals', 'not_equals', 'contains', 'greater_than', 'less_than'];

  useEffect(() => {
    if (projectId) {
      loadDimensionalView();
    }
  }, [projectId, selectedDimension, filters]);

  const loadDimensionalView = async () => {
    try {
      setLoading(true);
      const data = await migrationApi.getDimensionalView(
        selectedDimension,
        filters
      );
      setDimensionalView(data);
    } catch (error) {
      toast.error('Failed to load dimensional view');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const addFilter = () => {
    const newFilter: FilterCondition = {
      id: Date.now().toString(),
      field: 'resource_type',
      operator: 'equals',
      value: '',
      logicalOperator: filters.length > 0 ? 'AND' : undefined,
    };
    setFilters([...filters, newFilter]);
  };

  const updateFilter = (id: string, updates: Partial<FilterCondition>) => {
    setFilters(filters.map((f) => (f.id === id ? { ...f, ...updates } : f)));
  };

  const removeFilter = (id: string) => {
    setFilters(filters.filter((f) => f.id !== id));
  };

  const applyFilters = () => {
    loadDimensionalView();
  };

  const clearFilters = () => {
    setFilters([]);
  };

  const generateReport = async () => {
    try {
      const report = await migrationApi.generateInventoryReport(projectId!, reportConfig);
      // Download report
      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${reportConfig.name || 'report'}.json`;
      a.click();
      toast.success('Report generated successfully');
      setReportDialogOpen(false);
    } catch (error) {
      toast.error('Failed to generate report');
      console.error(error);
    }
  };

  const getDimensionIcon = (dimension: string) => {
    const dim = dimensions.find((d) => d.value === dimension);
    return dim?.icon || <Category />;
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4">Dimensional Filtering & Reports</Typography>
          <Box>
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, v) => v && setViewMode(v)}
              size="small"
              sx={{ mr: 2 }}
            >
              <ToggleButton value="grid">
                <ViewModule />
              </ToggleButton>
              <ToggleButton value="list">
                <ViewList />
              </ToggleButton>
            </ToggleButtonGroup>
            <Button
              variant="contained"
              startIcon={<Download />}
              onClick={() => setReportDialogOpen(true)}
            >
              Generate Report
            </Button>
          </Box>
        </Box>

        {/* Dimension Selector */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Select Dimension
          </Typography>
          <Grid container spacing={2}>
            {dimensions.map((dim) => (
              <Grid item xs={12} sm={6} md={2.4} key={dim.value}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: selectedDimension === dim.value ? 2 : 1,
                    borderColor:
                      selectedDimension === dim.value ? 'primary.main' : 'divider',
                    '&:hover': { borderColor: 'primary.main' },
                  }}
                  onClick={() => setSelectedDimension(dim.value)}
                >
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Box sx={{ color: 'primary.main', mb: 1 }}>{dim.icon}</Box>
                    <Typography variant="body2">{dim.label}</Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Filter Builder */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Advanced Filters</Typography>
            <Box>
              <Button startIcon={<Add />} onClick={addFilter} sx={{ mr: 1 }}>
                Add Filter
              </Button>
              {filters.length > 0 && (
                <>
                  <Button onClick={clearFilters} sx={{ mr: 1 }}>
                    Clear All
                  </Button>
                  <Button variant="contained" onClick={applyFilters}>
                    Apply Filters
                  </Button>
                </>
              )}
            </Box>
          </Box>

          {filters.length > 0 && (
            <Box>
              {filters.map((filter, index) => (
                <Box key={filter.id} sx={{ mb: 2 }}>
                  {index > 0 && (
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <ToggleButtonGroup
                        value={filter.logicalOperator}
                        exclusive
                        onChange={(_, v) => v && updateFilter(filter.id, { logicalOperator: v })}
                        size="small"
                      >
                        <ToggleButton value="AND">AND</ToggleButton>
                        <ToggleButton value="OR">OR</ToggleButton>
                      </ToggleButtonGroup>
                    </Box>
                  )}
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={3}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Field</InputLabel>
                        <Select
                          value={filter.field}
                          label="Field"
                          onChange={(e) => updateFilter(filter.id, { field: e.target.value })}
                        >
                          {filterFields.map((field) => (
                            <MenuItem key={field} value={field}>
                              {field.replace('_', ' ')}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={3}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Operator</InputLabel>
                        <Select
                          value={filter.operator}
                          label="Operator"
                          onChange={(e) => updateFilter(filter.id, { operator: e.target.value })}
                        >
                          {operators.map((op) => (
                            <MenuItem key={op} value={op}>
                              {op.replace('_', ' ')}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={5}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Value"
                        value={filter.value}
                        onChange={(e) => updateFilter(filter.id, { value: e.target.value })}
                      />
                    </Grid>
                    <Grid item xs={1}>
                      <IconButton onClick={() => removeFilter(filter.id)} color="error">
                        <Delete />
                      </IconButton>
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </Box>
          )}
        </Paper>

        {/* Results */}
        {loading ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <LinearProgress />
            <Typography sx={{ mt: 2 }}>Loading dimensional view...</Typography>
          </Box>
        ) : dimensionalView ? (
          <>
            {/* Summary */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" color="text.secondary">
                      Total Resources
                    </Typography>
                    <Typography variant="h3">{dimensionalView.total_resources}</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" color="text.secondary">
                      Total Monthly Cost
                    </Typography>
                    <Typography variant="h3">
                      ${dimensionalView.total_cost.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Dimensional Groups */}
            {viewMode === 'grid' ? (
              <Grid container spacing={3}>
                {dimensionalView.groups.map((group) => (
                  <Grid item xs={12} md={4} key={group.group_name}>
                    <Card>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          {getDimensionIcon(selectedDimension)}
                          <Typography variant="h6" sx={{ ml: 1 }}>
                            {group.group_name}
                          </Typography>
                        </Box>
                        <Divider sx={{ mb: 2 }} />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            Resources
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {group.resource_count}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            Monthly Cost
                          </Typography>
                          <Typography variant="body2" fontWeight="bold" color="primary">
                            ${group.total_cost.toFixed(2)}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Paper>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>{selectedDimension}</TableCell>
                        <TableCell align="right">Resources</TableCell>
                        <TableCell align="right">Monthly Cost</TableCell>
                        <TableCell align="right">% of Total</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {dimensionalView.groups.map((group) => (
                        <TableRow key={group.group_name}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {getDimensionIcon(selectedDimension)}
                              <Typography sx={{ ml: 1 }}>{group.group_name}</Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">{group.resource_count}</TableCell>
                          <TableCell align="right">${group.total_cost.toFixed(2)}</TableCell>
                          <TableCell align="right">
                            {((group.total_cost / dimensionalView.total_cost) * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            )}
          </>
        ) : (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Typography color="text.secondary">
              Select a dimension and apply filters to view results
            </Typography>
          </Paper>
        )}

        {/* Report Generation Dialog */}
        <Dialog
          open={reportDialogOpen}
          onClose={() => setReportDialogOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Generate Custom Report</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              label="Report Name"
              value={reportConfig.name}
              onChange={(e) => setReportConfig({ ...reportConfig, name: e.target.value })}
              sx={{ mt: 2, mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Group By</InputLabel>
              <Select
                multiple
                value={reportConfig.groupBy}
                label="Group By"
                onChange={(e) =>
                  setReportConfig({ ...reportConfig, groupBy: e.target.value as string[] })
                }
              >
                {dimensions.map((dim) => (
                  <MenuItem key={dim.value} value={dim.value}>
                    {dim.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={reportConfig.sortBy}
                label="Sort By"
                onChange={(e) => setReportConfig({ ...reportConfig, sortBy: e.target.value })}
              >
                <MenuItem value="cost">Cost</MenuItem>
                <MenuItem value="resource_count">Resource Count</MenuItem>
                <MenuItem value="name">Name</MenuItem>
              </Select>
            </FormControl>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setReportDialogOpen(false)}>Cancel</Button>
            <Button onClick={generateReport} variant="contained">
              Generate
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default DimensionalFiltering;
